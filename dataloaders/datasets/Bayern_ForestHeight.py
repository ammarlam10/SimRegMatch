import os, warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch
import h5py
from PIL import Image

from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF


class Bayern_ForestHeight(Dataset):
    """
    Dataset for Bayern Forest Height - handles HDF5 files with RGB and nDSM data.
    
    Each HDF5 file contains:
    - 'rgb': (361, 256, 256, 3) - RGB aerial imagery
    - 'ndsm': (361, 256, 256, 1) - Normalized Digital Surface Model (forest height)
    
    This dataset loads patches for labeled training/validation/testing.
    """
    
    def __init__(self, df, data_dir, img_size, split='train', label_mean=None, label_std=None):
        self.df = df
        self.data_dir = data_dir
        self.img_size = img_size
        self.split = split
        self.label_mean = label_mean
        self.label_std = label_std
        
        # Pre-load all HDF5 files into memory (to avoid multiprocessing issues with h5py)
        print(f"Bayern_ForestHeight: {split} set with {len(df)} patches")
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
        
        # Convert to PIL Image for transforms
        # RGB: uint8 [0, 255] or float32 [0, 1]
        if rgb.dtype == np.float32:
            rgb = (rgb * 255).astype(np.uint8)
        rgb_pil = Image.fromarray(rgb, mode='RGB')
        
        # nDSM: float32, keep as numpy for now
        ndsm = ndsm.squeeze(-1)  # (256, 256)
        
        # Apply transforms (synchronized for RGB and nDSM)
        rgb_tensor, ndsm_tensor = self.apply_transforms(rgb_pil, ndsm)
        
        # Compute mean height for this patch (for compatibility with SimRegMatch)
        label = np.asarray([np.mean(ndsm)]).astype('float32')
        
        # Normalize label if normalization stats are provided
        if self.label_mean is not None and self.label_std is not None:
            label = (label - self.label_mean) / self.label_std
        
        # Note: We return the full nDSM map as 'label' for pixel-wise regression
        # The scalar 'label' is kept for compatibility but not used in training
        return {
            'input': rgb_tensor,  # (3, 256, 256)
            'label': ndsm_tensor  # (1, 256, 256) - pixel-wise labels
        }

    def apply_transforms(self, rgb_pil, ndsm_array):
        """
        Apply transforms to RGB and nDSM.
        - Geometric transforms: synchronized (both RGB and nDSM)
        - Color transforms: only RGB
        - Normalization: per-image min-max for RGB, mean-std for nDSM
        """
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
            
            # Random vertical flip (synchronized)
            if torch.rand(1) > 0.5:
                rgb_pil = TF.vflip(rgb_pil)
                ndsm_pil = TF.vflip(ndsm_pil)
            
            # Color jitter (only RGB)
            if torch.rand(1) > 0.5:
                rgb_pil = transforms.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.1
                )(rgb_pil)
        
        # Convert RGB to tensor
        rgb_tensor = TF.to_tensor(rgb_pil)  # (3, H, W) in [0, 1]
        
        # Per-image min-max normalization for RGB
        rgb_min = rgb_tensor.min()
        rgb_max = rgb_tensor.max()
        if rgb_max - rgb_min > 1e-6:
            rgb_tensor = (rgb_tensor - rgb_min) / (rgb_max - rgb_min)
        
        # Normalize to [-1, 1] for model input
        rgb_tensor = TF.normalize(rgb_tensor, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        
        # Convert nDSM to tensor
        ndsm_array = np.array(ndsm_pil, dtype=np.float32)
        ndsm_tensor = torch.from_numpy(ndsm_array).unsqueeze(0)  # (1, H, W)
        
        # Normalize nDSM using label mean and std (consistent with SimRegMatch framework)
        if self.label_mean is not None and self.label_std is not None:
            ndsm_tensor = (ndsm_tensor - self.label_mean) / self.label_std
        
        return rgb_tensor, ndsm_tensor
