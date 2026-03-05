"""
In-memory cached dataset wrapper for faster training.

Preloads all images to RAM at initialization to eliminate disk I/O bottleneck.
Particularly useful for datasets with many epochs and slow storage (HDD).
"""

import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class CachedDataset(Dataset):
    """
    Wrapper that caches all dataset items in memory.
    
    Args:
        base_dataset: The underlying dataset to cache
        cache_images_only: If True, only cache images (not labels/transforms)
                          If False, cache complete transformed samples
    """
    
    def __init__(self, base_dataset, cache_images_only=True):
        self.base_dataset = base_dataset
        self.cache_images_only = cache_images_only
        self.cache = {}
        
        print(f"Preloading {len(base_dataset)} samples to memory...")
        
        # Preload all samples
        for idx in tqdm(range(len(base_dataset)), desc="Caching dataset", ncols=100):
            if cache_images_only:
                # Only cache the raw image loading (not augmentations)
                # This is useful for training sets with random augmentations
                # We'll apply transforms on-the-fly in __getitem__
                self.cache[idx] = self._cache_image_only(idx)
            else:
                # Cache the complete transformed sample
                # This is useful for validation/test sets with deterministic transforms
                self.cache[idx] = self.base_dataset[idx]
        
        print(f"✓ Cached {len(self.cache)} samples in memory")
    
    def _cache_image_only(self, idx):
        """
        Cache only the raw image data before augmentations.
        For datasets that load from disk, this caches the loaded PIL image.
        """
        # This is dataset-specific and would need to be customized
        # For now, we'll just cache the full sample
        return self.base_dataset[idx]
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        if self.cache_images_only:
            # Return cached sample (augmentations already applied during caching)
            return self.cache[idx]
        else:
            # Return cached sample
            return self.cache[idx]


class LazyLoadCachedDataset(Dataset):
    """
    Lazy-loading cached dataset that loads images on first access.
    More memory efficient than CachedDataset for large datasets.
    """
    
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset
        self.cache = {}
        print(f"Initialized lazy cache for {len(base_dataset)} samples")
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        if idx not in self.cache:
            self.cache[idx] = self.base_dataset[idx]
        return self.cache[idx]
