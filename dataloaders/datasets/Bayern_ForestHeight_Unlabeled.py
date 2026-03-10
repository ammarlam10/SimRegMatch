import os, warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch
import h5py
from PIL import Image

from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

from dataloaders.datasets import RandAug


class Bayern_ForestHeight_Unlabeled(Dataset):
    """
    Unlabeled dataset for Bayern Forest Height - handles HDF5 files with RGB and nDSM data.
    
    This dataset applies weak and strong augmentations for semi-supervised learning.
    - Weak augmentation: standard geometric transforms
    - Strong augmentation: RandAugment (geometric only, no color for nDSM)
    """
    
    def __init__(self, df, data_dir, img_size, split='train', label_mean=None, label_std=None,
                 rgb_min=None, rgb_max=None):
        self.df = df
        self.data_dir = data_dir
        self.img_size = img_size
        self.split = split
        self.label_mean = label_mean
        self.label_std = label_std
        
        # Global RGB normalization statistics (use provided or fallback to uint8 range)
        self.rgb_min = rgb_min if rgb_min is not None else 0.0
        self.rgb_max = rgb_max if rgb_max is not None else 255.0
        
        # Pre-load all HDF5 files into memory (to avoid multiprocessing issues with h5py)
        print(f"Bayern_ForestHeight_Unlabeled: {split} set with {len(df)} patches")
        print(f"  Label normalization: mean={label_mean}, std={label_std}")
        print(f"  Pre-loading HDF5 files into memory...")
        
        self.h5_cache = {}
        unique_files = df['path'].str.split(',').str[0].unique()
        
        for filename in unique_files:
            h5_path = os.path.join(self.data_dir, 'Bayern_forest_height_reduced', filename)
            try:
                with h5py.File(h5_path, 'r') as f:
                    # Load entire file into memory
                    self.h5_cache[filename] = {
                        'rgb': np.array(f['rgb'][:]),  # Convert to numpy array
                        'ndsm': np.array(f['ndsm'][:])  # Convert to numpy array
                    }
            except Exception as e:
                print(f"Warning: Could not load {filename}: {e}")
                raise
        
        print(f"  Loaded {len(self.h5_cache)} HDF5 files into memory")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        index = index % len(self.df)
        row = self.df.iloc[index]
        
        # Parse path: "filename,patch_idx"
        path_parts = row['path'].split(',')
        filename = path_parts[0]
        patch_idx = int(path_parts[1])
        
        # Get patch data from pre-loaded cache
        rgb = self.h5_cache[filename]['rgb'][patch_idx]  # (256, 256, 3)
        ndsm = self.h5_cache[filename]['ndsm'][patch_idx]  # (256, 256, 1)
        
        # Convert to PIL Image
        # RGB: float32 in [0, 255] range, cast directly to uint8
        if rgb.dtype == np.float32:
            rgb = rgb.astype(np.uint8)
        rgb_pil = Image.fromarray(rgb, mode='RGB')
        
        # nDSM: float32
        ndsm = ndsm.squeeze(-1)  # (256, 256)
        
        # Weak augmentation
        weak_rgb, weak_ndsm = self.apply_weak_transforms(rgb_pil, ndsm)
        
        # Strong augmentation (RandAugment)
        strong_rgb, strong_ndsm = self.apply_strong_transforms(rgb_pil, ndsm)
        
        # Compute mean height for this patch (for compatibility)
        label = np.asarray([np.mean(ndsm)]).astype('float32')
        
        # Normalize label if normalization stats are provided
        if self.label_mean is not None and self.label_std is not None:
            label = (label - self.label_mean) / self.label_std
        
        return {
            'weak': weak_rgb.float(),  # (3, 256, 256)
            'strong': strong_rgb.float(),  # (3, 256, 256)
            'label': weak_ndsm.float()  # (1, 256, 256) - use weak aug for label
        }

    def apply_weak_transforms(self, rgb_pil, ndsm_array):
        """Apply weak augmentation (resize, random crop, random flip)."""
        # Resize (both)
        rgb_pil = TF.resize(rgb_pil, (self.img_size, self.img_size))
        ndsm_pil = Image.fromarray(ndsm_array.astype(np.float32), mode='F')
        ndsm_pil = TF.resize(ndsm_pil, (self.img_size, self.img_size), interpolation=Image.BILINEAR)
        
        if self.split == 'train':
            # Random crop with padding (synchronized)
            rgb_pil = TF.pad(rgb_pil, 16)
            ndsm_pil = TF.pad(ndsm_pil, 16, fill=0)
            i, j, h, w = transforms.RandomCrop.get_params(rgb_pil, (self.img_size, self.img_size))
            rgb_pil = TF.crop(rgb_pil, i, j, h, w)
            ndsm_pil = TF.crop(ndsm_pil, i, j, h, w)
            
            # Random horizontal flip (synchronized)
            if torch.rand(1) > 0.5:
                rgb_pil = TF.hflip(rgb_pil)
                ndsm_pil = TF.hflip(ndsm_pil)
        
        # Convert RGB to tensor
        rgb_tensor = TF.to_tensor(rgb_pil)  # (3, H, W) in [0, 1]
        
        # Global min-max normalization for RGB (using dataset-wide statistics)
        rgb_tensor = rgb_tensor * 255.0  # Back to [0, 255] range
        rgb_tensor = (rgb_tensor - self.rgb_min) / (self.rgb_max - self.rgb_min)
        rgb_tensor = torch.clamp(rgb_tensor, 0.0, 1.0)  # Clip to [0, 1]
        
        # Normalize to [-1, 1]
        rgb_tensor = TF.normalize(rgb_tensor, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        
        # Convert nDSM to tensor
        ndsm_array = np.array(ndsm_pil, dtype=np.float32)
        ndsm_tensor = torch.from_numpy(ndsm_array).unsqueeze(0)  # (1, H, W)
        
        # Normalize nDSM using label mean and std (consistent with SimRegMatch framework)
        if self.label_mean is not None and self.label_std is not None:
            ndsm_tensor = (ndsm_tensor - self.label_mean) / self.label_std
        
        return rgb_tensor, ndsm_tensor

    def apply_strong_transforms(self, rgb_pil, ndsm_array):
        """
        Apply strong augmentation (RandAugment + resize + normalize).
        Color augmentations only applied to RGB, not nDSM.
        """
        # Apply RandAugment to RGB only (with color augmentations)
        rand_aug = RandAug.RandAugmentPC(n=2, m=10, img_size=self.img_size, grayscale=False)
        rgb_pil_aug = rand_aug(rgb_pil)
        
        # For nDSM, apply only geometric transforms (no color)
        ndsm_pil = Image.fromarray(ndsm_array.astype(np.float32), mode='F')
        
        # Resize both
        rgb_pil_aug = TF.resize(rgb_pil_aug, (self.img_size, self.img_size))
        ndsm_pil = TF.resize(ndsm_pil, (self.img_size, self.img_size), interpolation=Image.BILINEAR)
        
        # Apply same geometric transforms to nDSM (simplified version)
        if self.split == 'train':
            # Random horizontal flip
            if torch.rand(1) > 0.5:
                ndsm_pil = TF.hflip(ndsm_pil)
            
            # Random vertical flip
            if torch.rand(1) > 0.5:
                ndsm_pil = TF.vflip(ndsm_pil)
        
        # Convert RGB to tensor
        rgb_tensor = TF.to_tensor(rgb_pil_aug)  # (3, H, W) in [0, 1]
        
        # Global min-max normalization for RGB (using dataset-wide statistics)
        rgb_tensor = rgb_tensor * 255.0  # Back to [0, 255] range
        rgb_tensor = (rgb_tensor - self.rgb_min) / (self.rgb_max - self.rgb_min)
        rgb_tensor = torch.clamp(rgb_tensor, 0.0, 1.0)  # Clip to [0, 1]
        
        # Normalize to [-1, 1]
        rgb_tensor = TF.normalize(rgb_tensor, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        
        # Convert nDSM to tensor
        ndsm_array = np.array(ndsm_pil, dtype=np.float32)
        ndsm_tensor = torch.from_numpy(ndsm_array).unsqueeze(0)  # (1, H, W)
        
        # Normalize nDSM using label mean and std (consistent with SimRegMatch framework)
        if self.label_mean is not None and self.label_std is not None:
            ndsm_tensor = (ndsm_tensor - self.label_mean) / self.label_std
        
        return rgb_tensor, ndsm_tensor
