import os
import torch
import pandas as pd
import cv2
import numpy as np
from torch.utils.data import Dataset
import random
import imgaug.augmenters as iaa


frames_total = 8    # each video 8 uniform samples

# data augment from 'imgaug' --> Add (value=(-40,40), per_channel=True), GammaContrast (gamma=(0.5,1.5))
seq = iaa.Sequential([
    iaa.Add(value=(-40, 40), per_channel=True), # Add color 
    iaa.GammaContrast(gamma=(0.5, 1.5)) # GammaContrast with a gamma of 0.5 to 1.5
])


class Spoofing_valtest(Dataset):

    def __init__(self, info_list, root_dir,  transform=None):

        self.landmarks_frame = pd.read_csv(info_list, delimiter=' ', header=None)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):

        videoname = str(self.landmarks_frame.iloc[idx, 0])
        image_path = os.path.join(self.root_dir, videoname)
        image_path = os.path.join(image_path, 'profile')
             
        image_x, binary_mask = self.get_single_image_x(image_path, videoname)
		    
            
        sample = {'image_x': image_x, 'binary_mask': binary_mask, 'string_name': videoname}

        if self.transform:
            sample = self.transform(sample)
        return sample

    def get_single_image_x(self, image_path, videoname):

        files_total = len([name for name in os.listdir(image_path) if os.path.isfile(os.path.join(image_path, name))])
        interval = files_total//frames_total
        
        image_x = np.zeros((frames_total, 256, 256, 3))
        
        binary_mask = np.zeros((frames_total, 32, 32))
        
        # random choose 1 frame
        for ii in range(frames_total):
            image_id = ii*interval + 1 
            
            s = "%04d.jpg" % image_id            
            
            # RGB
            image_path2 = os.path.join(image_path, s)
            image_x_temp = cv2.imread(image_path2)
            
            image_x_temp_gray = cv2.imread(image_path2, 0)
            image_x_temp_gray = cv2.resize(image_x_temp_gray, (32, 32))

            image_x[ii,:,:,:] = cv2.resize(image_x_temp, (256, 256))
            
            #print(image_path2)
            
            for i in range(32):
                for j in range(32):
                    if image_x_temp_gray[i,j]>0:
                        binary_mask[ii, i, j]=1.0
                    else:
                        binary_mask[ii, i, j]=0.0
        return image_x, binary_mask
    

class Spoofing_Val_Images_Custom(Dataset):

    def __init__(self, csv_path, root_dir,  transform=None):

        self.data_frame = pd.read_csv(csv_path, sep=' ', header=None)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):

        file_name = str(self.data_frame.iloc[idx, 0])
        image_path = os.path.join(self.root_dir, file_name)
           
        image_x, binary_mask = self.get_single_image_x(image_path, file_name)
		    
            
        sample = {'image_x': image_x, 'binary_mask': binary_mask, 'string_name': file_name}

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
        
        for i in range(32):
            for j in range(32):
                if image_x_temp_gray[i, j]>0:
                    binary_mask[i, j] = 1.0
                else:
                    binary_mask[i, j] = 0.0
        
        return image_x, binary_mask
    

class FaceAntiSpoofing_ValDataset(Dataset):
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
                    random_frame_index = random.randint(5, frame_count - 5) # chọn ngẫu nhiên 1 frame
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
        
        sample = {
            'image_x': image_x, 
            'binary_mask': binary_mask, 
            'spoofing_label': spoofing_label, 
            # 'string_name': img_path
        }
        
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