import os
import pandas as pd
import numpy as np
import cv2
from PIL import Image

import torch
import torchvision.transforms as tf

from torch.utils import data


# csv, image
class HISDataset(data.Dataset):
    def __init__(self, args, transform=None, train=True):
        self.train = train
        self.transform = transform
        self.path = args.path

        csv = pd.read_csv(args.csv_name,
            names = ['dummy_num', 'image_name', 'gender', 'time_stamp', 
                'attention', 'emotion', 'traffic', 'head', 'hand', 'upper'])
        
        self.img_name_list = csv['image_name']
        self.label_list = csv[args.class_category]
        self.label_list = list(self.label_list)
        self.emotion = csv['emotion']
    
    def __getitem__(self, x):        
        img = Image.open(os.path.join(self.path, self.img_name_list[x]))
        label = self.label_list[x]
        emotion = self.emotion[x]
        if label == '1.0' or label == 'S': label = torch.LongTensor([0]).squeeze()
        else: label = torch.LongTensor([1]).squeeze()
        if emotion == '1.0':
            emotion = torch.FloatTensor([1, 0, 0])
        elif emotion == '2.0':
            emotion = torch.FloatTensor([0, 1, 0])
        elif emotion == '3.0':
            emotion = torch.FloatTensor([0, 0, 1])
        else:
            emotion = torch.FloatTensor([0, 0, 0])
        if self.transform:
            img = self.transform(img)
            #label = self.transform(label)
        return img, label, emotion

    def __len__(self):
        return len(self.img_name_list)

    
