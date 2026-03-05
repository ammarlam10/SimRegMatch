import os, warnings
warnings.filterwarnings('ignore')

import numpy as np
from PIL import Image

from torch.utils.data import Dataset
import torchvision.transforms as transforms


class UTKFace(Dataset):
    def __init__(self, df, data_dir, img_size, split='train', label_mean=None, label_std=None):
        self.df = df
        self.data_dir = data_dir
        self.img_size = img_size
        self.split = split
        self.label_mean = label_mean
        self.label_std = label_std
        
        # Pre-load all images into memory to avoid network I/O bottleneck
        print(f"UTKFace ({split}): Pre-loading {len(df)} images into memory...")
        self.image_cache = {}
        
        for idx, row in df.iterrows():
            img_path = os.path.join(self.data_dir, 'UTKFace_all', 'utkface_aligned_cropped', 'UTKFace', row['path'])
            try:
                img = Image.open(img_path).convert('RGB')
                # Store as numpy array to save memory (PIL images have overhead)
                self.image_cache[row['path']] = np.array(img)
            except Exception as e:
                print(f"Warning: Could not load {row['path']}: {e}")
                raise
        
        print(f"  Loaded {len(self.image_cache)} images into memory (~{len(self.image_cache) * 7.6 / 1024:.1f} MB)")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        index = index % len(self.df)
        row = self.df.iloc[index]

        # Load image from memory cache
        img_array = self.image_cache[row['path']]
        img = Image.fromarray(img_array)
        
        transform = self.get_transform()
        img = transform(img)
        
        label = np.asarray([row['age']]).astype('float32')
        
        # Normalize label if normalization stats are provided
        if self.label_mean is not None and self.label_std is not None:
            label = (label - self.label_mean) / self.label_std

        return {'input': img,
                'label': label}

    def get_transform(self):
        # ImageNet normalization for pretrained models
        if self.split == 'train':
            transform = transforms.Compose([
                transforms.Resize((self.img_size, self.img_size)),
                transforms.RandomCrop(self.img_size, padding=16),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
        return transform
