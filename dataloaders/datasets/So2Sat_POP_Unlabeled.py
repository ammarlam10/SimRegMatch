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
    
    # Global normalization values for Sentinel-2 RGB bands (computed from training data)
    SEN2_NORM = {
        3: (318, 2130),   # Red (B4)
        2: (560, 1776),   # Green (B3)
        1: (713, 1733),   # Blue (B2)
    }
    
    def __init__(self, df, data_dir, img_size, split='train', label_mean=None, label_std=None,
                 dem_min=None, dem_max=None):
        self.df = df
        self.data_dir = data_dir
        self.img_size = img_size
        self.split = split
        self.label_mean = label_mean
        self.label_std = label_std
        
        # DEM normalization statistics (use provided values or fallback to defaults)
        self.dem_min = dem_min if dem_min is not None else -2.0
        self.dem_max = dem_max if dem_max is not None else 2.0
        
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
        rgb = np.zeros((img_array.shape[0], img_array.shape[1], 3), dtype=np.float32)
        
        # Apply global normalization (fixed percentiles from training data)
        for i, band_idx in enumerate(self.SEN2_RGB_BANDS):
            band = img_array[:, :, band_idx].astype(np.float32)
            p2, p98 = self.SEN2_NORM[band_idx]
            
            # Clip to global percentile range and scale to [0, 255]
            band = np.clip(band, p2, p98)
            band = (band - p2) / (p98 - p2) * 255
            rgb[:, :, i] = band
        
        rgb = rgb.astype(np.uint8)
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
        # Apply RandAugment
        rand_aug = RandAug.RandAugmentPC(n=2, m=10, img_size=self.img_size, grayscale=use_grayscale_aug)
        img = rand_aug(img)
        
        # Resize and convert to tensor
        img = TF.resize(img, (self.img_size, self.img_size))
        img = TF.to_tensor(img)
        img = TF.normalize(img, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        
        return img
