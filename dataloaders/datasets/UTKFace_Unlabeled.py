import os, warnings
warnings.filterwarnings('ignore')

import numpy as np
from PIL import Image

from torch.utils.data import Dataset
import torchvision.transforms as transforms

from dataloaders.datasets import RandAug


class UTKFace_Unlabeled(Dataset):
    def __init__(self, df, data_dir, img_size, split='train', label_mean=None, label_std=None):
        self.df = df
        self.data_dir = data_dir
        self.img_size = img_size
        self.split = split
        self.label_mean = label_mean
        self.label_std = label_std

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        index = index % len(self.df)
        row = self.df.iloc[index]
        
        # UTKFace images are in a subdirectory
        # Path structure: data_dir/UTKFace_all/utkface_aligned_cropped/UTKFace/filename.jpg
        img_path = os.path.join(self.data_dir, 'UTKFace_all', 'utkface_aligned_cropped', 'UTKFace', row['path'])
        img = Image.open(img_path).convert('RGB')
        transform = self.get_transform()
        weak_aug = transform(img)
        
        strong_aug = self.get_strong(img)
        strong_aug = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize([.5, .5, .5], [.5, .5, .5])
        ])(strong_aug)
        label = np.asarray([row['age']]).astype('float32')
        
        # Normalize label if normalization stats are provided
        if self.label_mean is not None and self.label_std is not None:
            label = (label - self.label_mean) / self.label_std

        return {'weak': weak_aug.float(),
                'strong': strong_aug.float(),
                'label': label}

    def get_transform(self):
        if self.split == 'train':
            transform = transforms.Compose([
                transforms.Resize((self.img_size, self.img_size)),
                transforms.RandomCrop(self.img_size, padding=16),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([.5, .5, .5], [.5, .5, .5]),
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor(),
                transforms.Normalize([.5, .5, .5], [.5, .5, .5]),
            ])
        return transform

    def get_strong(self, img):
        rand_aug = RandAug.RandAugmentPC(n=2, m=10, img_size=self.img_size, grayscale=False)
        img = rand_aug(img)
        return img
