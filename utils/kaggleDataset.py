import os
import sys
import pandas as pd
import cv2
import torch
import torch.nn as nn

import albumentations as A
from albumentations.pytorch import ToTensorV2

class KaggleDataset(nn.Module):
    def __init__(self, mode = 'train'):
        super(KaggleDataset, self).__init__()
        self.mode = mode

        if mode == 'train':
            self.dataset = train_data
        else:
            self.dataset = val_data
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img_name = self.dataset[idx][0]
        img = cv2.imread(os.path.join('train_shuffle', img_name))
        #for now resize the image to 64 x 64 using CUBIC interpolation
        img = cv2.resize(img, (64, 64), interpolation = cv2.INTER_CUBIC)   
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)    #convert image from BGR to RGB format

        super_class = torch.tensor(self.dataset[idx][1], dtype = torch.float32)
        sub_class = torch.tensor(self.dataset[idx][2], dtype = torch.float32)

        apply_transform = self.transform_data()
        image = apply_transform(image = img)['image']

        return image, super_class, sub_class

    def transform_data(self):

        if self.mode == 'train':
            transform_func = A.Compose(
              [
                  #always resize the image to 329x224
                  #A.Resize(height = 329, width= 224, interpolation = cv2.INTER_AREA, p=1),
                  A.HorizontalFlip(p=0.4),  
                  A.ShiftScaleRotate(shift_limit=0.025, scale_limit=0, rotate_limit=15, p=0.5),
                  #A.RandomCrop(height = 224, width = 224, p=1),
                  #randomly change brightness, contrast, and saturation of the image 50% of the time
                  A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue = 0, p=0.5), 
                  A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p=1), 
                  ToTensorV2(p=1),
              ])
        else:     #augmentations during validation and testing
          transform_func = A.Compose(
          [
              #always resize the image to 329x224
              #A.Resize(height = 329, width = 224, p=1),   
              A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p=1),
              ToTensorV2(p=1),
          ])
    
        return transform_func