from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os
import torch
import pandas as pd
from tqdm import tqdm


class LiveSpoofDataset(Dataset):
    def __init__(self, live_dir, spoof_dir, transform=None):
        self.samples = []
        self.transform = transform

        for path in sorted(os.listdir(live_dir)):
            self.samples.append((os.path.join(live_dir, path), 0))  # live → label 0

        for path in sorted(os.listdir(spoof_dir)):
            self.samples.append((os.path.join(spoof_dir, path), 1))  # spoof → label 1

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)

        depth_label = torch.ones((32, 32)) if label == 0 else torch.zeros((32, 32))
        return img, depth_label, label
    
    
class LiveSpoofCelebDataset(Dataset):
    def __init__(self, root_dir, df_csv_path, transform=None, name='train'):
        self.samples = []
        self.transform = transform
        
        df = pd.read_csv(df_csv_path, sep=",", usecols=['path', 'label'])
        
        # live → label 0, spoof → label 1
        for _, (path, label) in tqdm(enumerate(df.values), desc=f"Load Dataset {name} : "):
            self.samples.append((os.path.join(root_dir, path), int(label)))  

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)

        depth_label = torch.ones((32, 32)) if label == 0 else torch.zeros((32, 32))
        return img, depth_label, label
