import os, warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch
from PIL import Image

from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

from dataloaders.datasets import RandAug

# Try to import tifffile for Sentinel-2 multi-band images
try:
    import tifffile
    HAS_TIFFFILE = True
except ImportError:
    HAS_TIFFFILE = False


class So2Sat_POP_Unlabeled(Dataset):
    """
    Unlabeled dataset for So2Sat POP - handles both:
    - DEM (Digital Elevation Model): float32 single-channel elevation data
    - Sentinel-2: uint16 13-band satellite imagery
    
    The data type is auto-detected from the file path.
    """
    
    # Sentinel-2 band indices for RGB (B4=Red, B3=Green, B2=Blue)
    SEN2_RGB_BANDS = [3, 2, 1]  # R, G, B order
    
    def __init__(self, df, data_dir, img_size, split='train', label_mean=None, label_std=None,
                 dem_min=None, dem_max=None, sen2_min=None, sen2_max=None):
        self.df = df
        self.data_dir = data_dir
        self.img_size = img_size
        self.split = split
        self.label_mean = label_mean
        self.label_std = label_std
        
        # DEM normalization statistics (use provided values or fallback to defaults)
        self.dem_min = dem_min if dem_min is not None else -2.0
        self.dem_max = dem_max if dem_max is not None else 2.0

        # Sentinel-2 global min/max per band after clip(/4000) normalization
        self.sen2_min = sen2_min if sen2_min is not None else [0.0, 0.0, 0.0]
        self.sen2_max = sen2_max if sen2_max is not None else [1.0, 1.0, 1.0]
        
        # Auto-detect data type from first path
        first_path = df.iloc[0]['path']
        self.is_sentinel2 = 'sen2' in first_path
        
        if self.is_sentinel2 and not HAS_TIFFFILE:
            raise ImportError("tifffile is required for Sentinel-2 data. Install with: pip install tifffile")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        index = index % len(self.df)
        row = self.df.iloc[index]
        
        # So2Sat_POP images are in So2Sat_POP subdirectory
        img_path = os.path.join(self.data_dir, 'So2Sat_POP', row['path'])
        
        if self.is_sentinel2:
            img_rgb = self._load_sentinel2(img_path)
        else:
            img_rgb = self._load_dem(img_path)
        
        # Weak augmentation
        weak_aug = self.apply_weak_transforms(img_rgb)
        
        # Strong augmentation (RandAugment)
        # Use grayscale=True for DEM (skip color augmentations)
        # Use grayscale=False for Sentinel-2 RGB (use color augmentations)
        strong_aug = self.apply_strong_transforms(img_rgb, use_grayscale_aug=not self.is_sentinel2)
        
        label = np.asarray([row['age']]).astype('float32')
        
        # Normalize label if normalization stats are provided
        if self.label_mean is not None and self.label_std is not None:
            label = (label - self.label_mean) / self.label_std

        return {'weak': weak_aug.float(),
                'strong': strong_aug.float(),
                'label': label}

    def _load_sentinel2(self, img_path):
        """Load Sentinel-2 multi-band image and extract RGB."""
        # Read multi-band TIFF (shape: H, W, 13)
        img_array = tifffile.imread(img_path)
        
        # Extract RGB bands (B4, B3, B2 = indices 3, 2, 1)
        image_bands = img_array[:, :, self.SEN2_RGB_BANDS].astype(np.float32)
        
        # Clip to valid range, scale to [0, 1]
        image_bands = np.clip(image_bands, 0, 4000)
        image_bands = image_bands / 4000.0

        # Global min-max normalization per band
        for i in range(3):
            denom = self.sen2_max[i] - self.sen2_min[i]
            if denom < 1e-6:
                denom = 1.0
            image_bands[:, :, i] = (image_bands[:, :, i] - self.sen2_min[i]) / denom

        image_bands = np.clip(image_bands, 0.0, 1.0)
        rgb = (image_bands * 255.0).round().astype(np.uint8)
        return Image.fromarray(rgb, mode='RGB')

    def _load_dem(self, img_path):
        """Load DEM (elevation) image."""
        # Load DEM image as float32
        img_pil = Image.open(img_path)
        img_array = np.array(img_pil, dtype=np.float32)
        
        # Normalize DEM values to [0, 1] range
        img_array = np.clip(img_array, self.dem_min, self.dem_max)
        img_array = (img_array - self.dem_min) / (self.dem_max - self.dem_min)
        
        # Convert to PIL Image (8-bit grayscale, then RGB)
        img_uint8 = (img_array * 255).astype(np.uint8)
        img_pil = Image.fromarray(img_uint8, mode='L')
        return img_pil.convert('RGB')

    def apply_weak_transforms(self, img):
        """Apply weak augmentation (resize, random crop, random flip)."""
        # Resize
        img = TF.resize(img, (self.img_size, self.img_size))
        
        if self.split == 'train':
            # Random crop with padding
            img = TF.pad(img, 16)
            i, j, h, w = transforms.RandomCrop.get_params(img, (self.img_size, self.img_size))
            img = TF.crop(img, i, j, h, w)
            
            # Random horizontal flip
            if torch.rand(1) > 0.5:
                img = TF.hflip(img)
        
        # Convert to tensor and normalize
        img = TF.to_tensor(img)
        img = TF.normalize(img, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        
        return img

    def apply_strong_transforms(self, img, use_grayscale_aug=False):
        """Apply strong augmentation (RandAugment + resize + normalize)."""
        # For small images (e.g. 100x100 Sentinel-2 patches), resize FIRST to target size
        # before applying RandAugment. This ensures augmentation parameters work consistently
        # regardless of original image size.
        img = TF.resize(img, (self.img_size, self.img_size))
        
        # Apply RandAugment on resized image (n=3 so color/geometric mix is more visible)
        rand_aug = RandAug.RandAugmentPC(n=3, m=10, img_size=self.img_size, grayscale=use_grayscale_aug)
        img = rand_aug(img)
        
        # Convert to tensor and normalize
        img = TF.to_tensor(img)
        img = TF.normalize(img, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        
        return img
