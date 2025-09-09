from __future__ import print_function, division
import os
import torch
import pandas as pd
import cv2
import numpy as np
import random
import math
import imgaug.augmenters as iaa
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms


# data augment from 'imgaug' --> Add (value=(-40,40), per_channel=True), GammaContrast (gamma=(0.5,1.5))
seq = iaa.Sequential([
    iaa.Add(value=(-40, 40), per_channel=True), # Add color 
    iaa.GammaContrast(gamma=(0.5, 1.5)) # GammaContrast with a gamma of 0.5 to 1.5
])


# array
class RandomErasing(object):
    '''
    Class that performs Random Erasing in Random Erasing Data Augmentation by Zhong et al. 
    -------------------------------------------------------------------------------------
    probability: The probability that the operation will be performed.
    sl: min erasing area
    sh: max erasing area
    r1: min aspect ratio
    mean: erasing value
    -------------------------------------------------------------------------------------
    '''
    def __init__(self, probability = 0.5, sl = 0.01, sh = 0.05, r1 = 0.5, mean=[0.4914, 0.4822, 0.4465]):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1
       
    def __call__(self, sample):
        img, binary_mask, spoofing_label = sample['image_x'], sample['binary_mask'],sample['spoofing_label']
        
        if random.uniform(0, 1) < self.probability:
            attempts = np.random.randint(1, 3)
            for attempt in range(attempts):
                area = img.shape[0] * img.shape[1]
           
                target_area = random.uniform(self.sl, self.sh) * area
                aspect_ratio = random.uniform(self.r1, 1/self.r1)
    
                h = int(round(math.sqrt(target_area * aspect_ratio)))
                w = int(round(math.sqrt(target_area / aspect_ratio)))
    
                if w < img.shape[1] and h < img.shape[0]:
                    x1 = random.randint(0, img.shape[0] - h)
                    y1 = random.randint(0, img.shape[1] - w)

                    img[x1:x1+h, y1:y1+w, 0] = self.mean[0]
                    img[x1:x1+h, y1:y1+w, 1] = self.mean[1]
                    img[x1:x1+h, y1:y1+w, 2] = self.mean[2]
                    
        return {'image_x': img, 'binary_mask': binary_mask, 'spoofing_label': spoofing_label}


# Tensor
class Cutout(object):
    def __init__(self, length=50):
        self.length = length

    def __call__(self, sample):
        img, binary_mask, spoofing_label = sample['image_x'], sample['binary_mask'],sample['spoofing_label']
        h, w = img.shape[1], img.shape[2]    # Tensor [1][2],  nparray [0][1]
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)
        length_new = np.random.randint(1, self.length)
        
        y1 = np.clip(y - length_new // 2, 0, h)
        y2 = np.clip(y + length_new // 2, 0, h)
        x1 = np.clip(x - length_new // 2, 0, w)
        x2 = np.clip(x + length_new // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return {'image_x': img, 'binary_mask': binary_mask, 'spoofing_label': spoofing_label}


class Normaliztion(object):
    """
        same as mxnet, normalize into [-1, 1]
        image = (image - 127.5)/128
    """
    def __call__(self, sample):
        image_x, binary_mask, spoofing_label = sample['image_x'], sample['binary_mask'],sample['spoofing_label']
        
        new_image_x = (image_x - 127.5)/128     # [-1,1]

        return {'image_x': new_image_x, 'binary_mask': binary_mask, 'spoofing_label': spoofing_label}


class RandomHorizontalFlip(object):
    """Horizontally flip the given Image randomly with a probability of 0.5."""
    def __call__(self, sample):
        image_x, binary_mask, spoofing_label = sample['image_x'], sample['binary_mask'], sample['spoofing_label']
        
        new_image_x = np.zeros((256, 256, 3))
        new_binary_mask = np.zeros((32, 32))

        p = random.random()
        if p < 0.5:
            #print('Flip')

            new_image_x = cv2.flip(image_x, 1)
            new_binary_mask = cv2.flip(binary_mask, 1)
            
            return {'image_x': new_image_x, 'binary_mask': new_binary_mask, 'spoofing_label': spoofing_label}
        else:
            #print('no Flip')
            return {'image_x': image_x, 'binary_mask': binary_mask, 'spoofing_label': spoofing_label}


class ToTensor(object):
    """
        Convert ndarrays in sample to Tensors.
        process only one batch every time
    """

    def __call__(self, sample):
        image_x, binary_mask, spoofing_label = sample['image_x'], sample['binary_mask'],sample['spoofing_label']
        
        # swap color axis because
        # numpy image: (batch_size) x H x W x C
        # torch image: (batch_size) x C X H X W
        image_x = image_x[:,:,::-1].transpose((2, 0, 1))
        image_x = np.array(image_x)
        
        binary_mask = np.array(binary_mask)

                        
        spoofing_label_np = np.array([0],dtype=np.longlong)
        spoofing_label_np[0] = spoofing_label
        
        
        return {
            'image_x': torch.from_numpy(image_x.astype(np.float32)).float(), 
            'binary_mask': torch.from_numpy(binary_mask.astype(np.float32)).float(), 
            'spoofing_label': torch.from_numpy(spoofing_label_np.astype(np.float32)).float()
        }


class Spoofing_train(Dataset):

    def __init__(self, info_list, root_dir, transform=None):

        self.landmarks_frame = pd.read_csv(info_list, delimiter=' ', header=None)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_frame)

    
    def __getitem__(self, idx):
        
        #print(self.landmarks_frame.iloc[idx, 0])
        videoname = str(self.landmarks_frame.iloc[idx, 0])
        image_path = os.path.join(self.root_dir, videoname)
    
        image_x, binary_mask = self.get_single_image_x(image_path)
        
        spoofing_label = self.landmarks_frame.iloc[idx, 1]
        if spoofing_label == 1:
            spoofing_label = 1            # real
        else:
            spoofing_label = 0            # fake
            binary_mask = np.zeros((32, 32))    
        
        #frequency_label = self.landmarks_frame.iloc[idx, 2:2+50].values  

        sample = {'image_x': image_x, 'binary_mask': binary_mask, 'spoofing_label': spoofing_label}

        if self.transform:
            sample = self.transform(sample)
        return sample

    def get_single_image_x(self, image_path):
        
        image_x = np.zeros((256, 256, 3))
        binary_mask = np.zeros((32, 32))
 
 
        image_x_temp = cv2.imread(image_path)
        image_x_temp_gray = cv2.imread(image_path, 0)

        image_x = cv2.resize(image_x_temp, (256, 256))
        image_x_temp_gray = cv2.resize(image_x_temp_gray, (32, 32))
        image_x_aug = seq.augment_image(image_x) 
        
        for i in range(32):
            for j in range(32):
                if image_x_temp_gray[i,j]>0:
                    binary_mask[i,j]=1
                else:
                    binary_mask[i,j]=0
        
        return image_x_aug, binary_mask
    
    
class Spoofing_Train_Images_Custom(Dataset):

    def __init__(self, csv_path, root_dir, transform=None):

        self.data_frame = pd.read_csv(csv_path, sep=" ", header=None)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        file_name = str(self.data_frame.iloc[idx, 0])
        image_path = os.path.join(self.root_dir, file_name)
    
        image_x, binary_mask = self.get_single_image_x(image_path)
        
        spoofing_label = self.data_frame.iloc[idx, 1]
        
        if spoofing_label == 1:
            spoofing_label = 1            # real
        else:
            spoofing_label = 0            # fake
            binary_mask = np.zeros((32, 32))    
        
        #frequency_label = self.landmarks_frame.iloc[idx, 2:2+50].values  

        sample = {'image_x': image_x, 'binary_mask': binary_mask, 'spoofing_label': spoofing_label}

        if self.transform:
            sample = self.transform(sample)
        return sample

    def get_single_image_x(self, image_path):
        
        image_x = np.zeros((256, 256, 3))
        binary_mask = np.zeros((32, 32))
 
 
        image_x_temp = cv2.imread(image_path)
        image_x_temp_gray = cv2.imread(image_path, 0)

        image_x = cv2.resize(image_x_temp, (256, 256))
        image_x_temp_gray = cv2.resize(image_x_temp_gray, (32, 32))
        image_x_aug = seq.augment_image(image_x) 
        
        for i in range(32):
            for j in range(32):
                if image_x_temp_gray[i,j]>0:
                    binary_mask[i,j]=1
                else:
                    binary_mask[i,j]=0
        
        return image_x_aug, binary_mask


class FaceAntiSpoofing_TrainDataset(Dataset):
    def __init__(self, dataframe, base_dir, resize=(256, 256), size_mask=(32, 32), transform=None):
        self.dataframe = dataframe
        self.base_dir = base_dir
        self.transform = transform
        self.resize = resize
        self.size_mask = size_mask

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = os.path.join(self.base_dir, self.dataframe.iloc[idx, 0])
        
        _, file_extension = os.path.splitext(img_path)
        file_extension = file_extension.lower()

        if file_extension.lower() in ['.jpg', '.jpeg', '.png']:
            
            # Load image
            frame = cv2.imread(img_path)
            if frame is not None:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Convert to RGB
                image_x, binary_mask = self.get_single_image_x(frame)
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
                    random_frame_index = random.randint(0, frame_count - 1) # chọn ngẫu nhiên 1 frame
                    cap.set(cv2.CAP_PROP_POS_FRAMES, random_frame_index) # chọn ngẫu nhiên 1 frame
                    ret, frame = cap.read()
                    if ret:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Convert to RGB
                        image_x, binary_mask = self.get_single_image_x(frame)
                    else:
                        print(f"Warning: Could not read frame {random_frame_index} from {img_path}")
                else:
                    print(f"Warning: Video file {img_path} has no frames.")
                cap.release()

        else:
            print(f"Warning: Unsupported file format: {file_extension} for file {img_path}")

        # process label
        spoofing_label = self.dataframe.iloc[idx, 1]
        
        if spoofing_label == 1:
            spoofing_label = 1            # real
        else:
            spoofing_label = 0            # fake
            binary_mask = np.zeros(self.size_mask) 
        
        sample = {'image_x': image_x, 'binary_mask': binary_mask, 'spoofing_label': spoofing_label}
        
        if self.transform:
            sample = self.transform(sample)

        return sample
    
    def get_single_image_x(self, frame):
        RGB_C = (3,)
        shapes = self.resize + RGB_C
        image_x = np.zeros(shapes)
        binary_mask = np.zeros(self.size_mask)
 
 
        image_x_temp = frame.copy()
        image_x_temp_gray = cv2.cvtColor(image_x_temp, cv2.COLOR_BGR2GRAY)

        image_x = cv2.resize(image_x_temp, self.resize)
        image_x_temp_gray = cv2.resize(image_x_temp_gray, self.size_mask)
        image_x_aug = seq.augment_image(image_x) 
        
        for i in range(self.size_mask[0]):
            for j in range(self.size_mask[1]):
                if image_x_temp_gray[i, j] > 0:
                    binary_mask[i, j] = 1
                else:
                    binary_mask[i, j] = 0
        
        return image_x_aug, binary_mask