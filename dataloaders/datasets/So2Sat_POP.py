import os, warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch
from PIL import Image

from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF


class So2Sat_POP(Dataset):
    """
    Dataset for So2Sat POP - handles float32 DEM (Digital Elevation Model) images.
    
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
        # Path structure: data_dir/So2Sat_POP/So2Sat_POP_Part2/train/.../file.tif
        img_path = os.path.join(self.data_dir, 'So2Sat_POP', row['path'])
        
        # Load DEM image as float32 (DO NOT use .convert('RGB') - it destroys float data!)
        img_pil = Image.open(img_path)
        img_array = np.array(img_pil, dtype=np.float32)
        
        # Normalize DEM values to [0, 1] range using min-max normalization
        # Clip to expected range first to handle outliers
        img_array = np.clip(img_array, self.dem_min, self.dem_max)
        img_array = (img_array - self.dem_min) / (self.dem_max - self.dem_min)
        
        # Convert to PIL Image for transforms (as 'L' mode - 8-bit grayscale)
        # Scale to [0, 255] for PIL compatibility
        img_uint8 = (img_array * 255).astype(np.uint8)
        img_pil = Image.fromarray(img_uint8, mode='L')
        
        # Convert to RGB (3 channels) by repeating the grayscale channel
        img_pil = img_pil.convert('RGB')
        
        # Apply transforms
        img = self.apply_transforms(img_pil)
        
        label = np.asarray([row['age']]).astype('float32')
        
        # Normalize label if normalization stats are provided
        if self.label_mean is not None and self.label_std is not None:
            label = (label - self.label_mean) / self.label_std

        return {'input': img,
                'label': label}

    def apply_transforms(self, img):
        """Apply appropriate transforms based on split."""
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
        
        # Convert to tensor [0, 1]
        img = TF.to_tensor(img)
        
        # Normalize to [-1, 1]
        img = TF.normalize(img, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        
        return img
