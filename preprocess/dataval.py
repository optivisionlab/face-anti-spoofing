import os
import torch
import pandas as pd
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import os 


frames_total = 8    # each video 8 uniform samples


class Normaliztion_valtest(object):
    """
        same as mxnet, normalize into [-1, 1]
        image = (image - 127.5)/128
    """
    def __call__(self, sample):
        image_x, binary_mask, string_name = sample['image_x'],sample['binary_mask'],sample['string_name']
        new_image_x = (image_x - 127.5)/128     # [-1,1]
        
        return {'image_x': new_image_x, 'binary_mask': binary_mask, 'string_name': string_name}


class ToTensor_valtest(object):
    """
        Convert ndarrays in sample to Tensors.
        process only one batch every time
    """

    def __call__(self, sample):
        image_x, binary_mask, string_name = sample['image_x'], sample['binary_mask'], sample['string_name']
        
        # swap color axis because    BGR2RGB
        # numpy image: (batch_size) x T x H x W x C
        # torch image: (batch_size) x T x C X H X W
        image_x = image_x[:,:,:,::-1].transpose((0, 3, 1, 2))
        image_x = np.array(image_x)
                        
        binary_mask = np.array(binary_mask)
        
        return {'image_x': torch.from_numpy(image_x.astype(np.float)).float(), 'binary_mask': torch.from_numpy(binary_mask.astype(np.float)).float(), 'string_name': string_name} 


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