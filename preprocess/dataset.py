from torch.utils.data import Dataset
from PIL import Image
import os
import torch
import pandas as pd
from tqdm import tqdm
import cv2, random
from preprocess.moire import Moire
from torchvision import transforms
import numpy as np


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


class FAS_BCE_Dataset(Dataset):
    def __init__(self, dataframe, base_dir, transform=None, is_train=True, random_frame=False, tf_ratio=0.5):
        self.dataframe = dataframe
        self.base_dir = base_dir
        self.transform = transform
        self.is_train = is_train
        self.moire = Moire()
        self.random_frame = random_frame
        self.tf_ratio = tf_ratio

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = os.path.join(self.base_dir, self.dataframe.iloc[idx, 0])
        image = None
        _, file_extension = os.path.splitext(img_path)
        file_extension = file_extension.lower()

        if file_extension.lower() in ['.jpg', '.jpeg', '.png']:
            
            # Load image
            frame = cv2.imread(img_path)
            if frame is not None:
                image = frame.copy()
            else:
                print(f"Error: Could not read image file {img_path}")

        elif file_extension.lower() in ['.mp4', '.mov', '.avi', '.mkv']:
            
            # Load video and extract a random frame
            cap = cv2.VideoCapture(img_path)
            if not cap.isOpened():
                print(f"Error: Could not open video file {img_path}")
            else:
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if frame_count > 0:
                    random_frame_index = int(frame_count * self.tf_ratio)
                    if self.random_frame:
                        random_frame_index = random.randint(5, frame_count - 5) # chọn ngẫu nhiên 1 frame
                    cap.set(cv2.CAP_PROP_POS_FRAMES, random_frame_index) # chọn ngẫu nhiên 1 frame
                    ret, frame = cap.read()
                    if ret:
                        image = frame.copy() # Convert to RGB
                    else:
                        print(f"Warning: Could not read frame {random_frame_index} from {img_path}")
                else:
                    print(f"Warning: Video file {img_path} has no frames.")
                cap.release()

        else:
            print(f"Warning: Unsupported file format: {file_extension} for file {img_path}")

        # process label
        label = self.dataframe.iloc[idx, 1] # 1 -> fake, 0 -> real
        
        if label == 0 and self.is_train:
            prob_value = random.random()
            
            if prob_value < 0.3:
                label = 1
                image = self.moire(image)
                
            elif prob_value >= 0.3 and prob_value < 0.6:
                label = 1
                color_jitter = transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.4)
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)      
                image_pil = Image.fromarray(image_rgb)
                transformed_image_pil = color_jitter(image_pil)
                transformed_image_np = np.array(transformed_image_pil)
                image = cv2.cvtColor(transformed_image_np, cv2.COLOR_RGB2BGR) # Convert to BRG

        # image = Image.fromarray(image)
        if self.transform:
            sample = self.transform(image)

        return sample, label
    
    
class FAS_MSEMask_Dataset(Dataset):
    def __init__(self, dataframe, base_dir, transform=None, is_train=True, random_frame=False, tf_ratio=0.5, size_mask=32):
        self.dataframe = dataframe
        self.base_dir = base_dir
        self.transform = transform
        self.is_train = is_train
        self.moire = Moire()
        self.random_frame = random_frame
        self.tf_ratio = tf_ratio
        self.size_mask = size_mask

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = os.path.join(self.base_dir, self.dataframe.iloc[idx, 0])
        image = None
        _, file_extension = os.path.splitext(img_path)
        file_extension = file_extension.lower()

        if file_extension.lower() in ['.jpg', '.jpeg', '.png']:
            
            # Load image
            frame = cv2.imread(img_path)
            if frame is not None:
                image = frame.copy()
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Convert to RGB
            else:
                print(f"Error: Could not read image file {img_path}")

        elif file_extension.lower() in ['.mp4', '.mov', '.avi', '.mkv']:
            
            # Load video and extract a random frame
            cap = cv2.VideoCapture(img_path)
            if not cap.isOpened():
                print(f"Error: Could not open video file {img_path}")
            else:
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if frame_count > 0:
                    random_frame_index = int(frame_count * self.tf_ratio)
                    if self.random_frame:
                        random_frame_index = random.randint(5, frame_count - 5) # chọn ngẫu nhiên 1 frame
                    cap.set(cv2.CAP_PROP_POS_FRAMES, random_frame_index) # chọn ngẫu nhiên 1 frame
                    ret, frame = cap.read()
                    if ret:
                        image = frame.copy()
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Convert to RGB
                    else:
                        print(f"Warning: Could not read frame {random_frame_index} from {img_path}")
                else:
                    print(f"Warning: Video file {img_path} has no frames.")
                cap.release()

        else:
            print(f"Warning: Unsupported file format: {file_extension} for file {img_path}")

        # process label
        label_name = self.dataframe.iloc[idx, 1]
        label, binary_mask = self.get_binary_mask(label_name=label_name, size_mask=self.size_mask)
        
        
        # sample = {
        #     'input_image': image, 'label_name': label_name, 'binary_mask': binary_mask, 'label': label
        # }        
        
        if self.transform:
            image = self.transform(image)

        return image, label
    
    def get_binary_mask(self, label_name, size_mask=32):
        sizes = (size_mask, size_mask)
        
        if label_name == 'live' and self.is_train:
            prob_value = random.random()
            
            if prob_value < 0.3:
                label_name = 'spoof'
                image = self.moire(image)
                
            elif prob_value >= 0.3 and prob_value < 0.6:
                label_name = 'spoof'
                color_jitter = transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.4)
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # RGB  
                image_pil = Image.fromarray(image_rgb)
                transformed_image_pil = color_jitter(image_pil)
                image = np.array(transformed_image_pil) # RGB
        
        return 0, np.zeros(sizes) if label_name == 'spoof' else 1, np.ones(sizes)