import os, warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch
from PIL import Image

from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

from dataloaders.datasets import RandAug


class So2Sat_POP_Unlabeled(Dataset):
    """
    Unlabeled dataset for So2Sat POP - handles float32 DEM (Digital Elevation Model) images.
    
    DEM images are single-channel float32 with values typically in range [-2, +2].
    We normalize them properly and expand to 3 channels for CNN compatibility.
    """
    
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

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        index = index % len(self.df)
        row = self.df.iloc[index]
        
        # So2Sat_POP images are in So2Sat_POP subdirectory
        img_path = os.path.join(self.data_dir, 'So2Sat_POP', row['path'])
        
        # Load DEM image as float32 (DO NOT use .convert('RGB') - it destroys float data!)
        img_pil = Image.open(img_path)
        img_array = np.array(img_pil, dtype=np.float32)
        
        # Normalize DEM values to [0, 1] range using min-max normalization
        img_array = np.clip(img_array, self.dem_min, self.dem_max)
        img_array = (img_array - self.dem_min) / (self.dem_max - self.dem_min)
        
        # Convert to PIL Image for transforms (as 'L' mode - 8-bit grayscale)
        img_uint8 = (img_array * 255).astype(np.uint8)
        img_pil = Image.fromarray(img_uint8, mode='L')
        
        # Convert to RGB (3 channels)
        img_rgb = img_pil.convert('RGB')
        
        # Weak augmentation
        weak_aug = self.apply_weak_transforms(img_rgb)
        
        # Strong augmentation (RandAugment)
        strong_aug = self.apply_strong_transforms(img_rgb)
        
        label = np.asarray([row['age']]).astype('float32')
        
        # Normalize label if normalization stats are provided
        if self.label_mean is not None and self.label_std is not None:
            label = (label - self.label_mean) / self.label_std

        return {'weak': weak_aug.float(),
                'strong': strong_aug.float(),
                'label': label}

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

    def apply_strong_transforms(self, img):
        """Apply strong augmentation (RandAugment + resize + normalize)."""
        # Apply RandAugment (grayscale=True to skip color-based augmentations for DEM data)
        rand_aug = RandAug.RandAugmentPC(n=2, m=10, img_size=self.img_size, grayscale=True)
        img = rand_aug(img)
        
        # Resize and convert to tensor
        img = TF.resize(img, (self.img_size, self.img_size))
        img = TF.to_tensor(img)
        img = TF.normalize(img, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        
        return img
