from __future__ import division
import torch
from torch.utils import data

# general libs
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import math
import time
import tqdm
import os
import random
import argparse
import glob
import json

from scipy import ndimage, signal
import pdb

def temporal_transform(frame_indices, sample_range):
    tmp = np.random.randint(0,len(frame_indices)-sample_range)
    return frame_indices[tmp:tmp+sample_range]

DAVIS_2016 = ['bear'
,'bmx-bumps','boat','breakdance-flare','bus','car-turn','dance-jump','dog-agility','drift-turn','elephant','flamingo','hike','hockey','horsejump-low','kite-walk','lucia','mallard-fly','mallard-water','motocross-bumps','motorbike','paragliding','rhino','rollerblade','scooter-gray','soccerball','stroller','surf','swing','tennis','train','blackswan','bmx-trees','breakdance','camel','car-roundabout','car-shadow','cows','dance-twirl','dog','drift-chicane','drift-straight','goat','horsejump-high','kite-surf','libby','motocross-jump','paragliding-launch','parkour','scooter-black','soapbox']

class DAVIS(data.Dataset):
    def __init__(self, root, mask_dilation, resolution='480p', size=(256, 256), sample_duration=0):
        self.mask_dilation = mask_dilation
        self.sample_duration = sample_duration
        self.root = root
        self.mask_dir = root + '_mask'
        self.image_dir = root + '_frame'

        self.size = size

        if '.' in root.split('/')[-1]:
            self.videos = (root.split('/')[-1].split('.')[0])
        else:
            self.videos = (root.split('/')[-1])

        self.num_frames = len(glob.glob(os.path.join(self.image_dir, '*.jpg')))
        _mask = np.array(Image.open(os.path.join(self.mask_dir, '00000.png')).convert("P"))
        self.num_objects = np.max(_mask)
        self.shape = np.shape(_mask)

    def __len__(self):
        return 1


    def __getitem__(self, index):
        video = self.videos
        info = {}
        info['name'] = video
        info['num_frames'] = self.num_frames
        num_objects = 1
        info['num_objects'] = num_objects        
        
        images = []
        masks = []
        struct = ndimage.generate_binary_structure(2, 2)

        f_list = list(range(self.num_frames))
        if self.sample_duration > 0:
            f_list = temporal_transform(f_list, self.sample_duration)

        for f in f_list:
                            
            img_file = os.path.join(self.image_dir, '{:05d}.jpg'.format(f))
            image_ = cv2.resize(cv2.imread(img_file), self.size, cv2.INTER_CUBIC)
            image_ = np.float32(image_)/255.0
            images.append(torch.from_numpy(image_))

            try:
                mask_file = os.path.join(self.mask_dir, '{:05d}.png'.format(f))
                mask_ = np.array(Image.open(mask_file).convert('P'), np.uint8)
                mask_ = cv2.resize(mask_,self.size, cv2.INTER_NEAREST)
            except:
                mask_file = os.path.join(self.mask_dir, '00000.png')
                mask_ = np.array(Image.open(mask_file).convert('P'), np.uint8)
                mask_ = cv2.resize(mask_, self.size, cv2.INTER_NEAREST)

            if video in DAVIS_2016:
                mask_ = (mask_ != 0)
            else:
                select_mask = max(1, mask_.max())
                mask_ = (mask_ == select_mask).astype(np.float)
            
            w_k = np.ones((10, 6))
            mask2 = signal.convolve2d(mask_.astype(np.float), w_k, 'same')
            mask2 = 1 - (mask2 == 0)
            mask_ = np.float32(mask2)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (self.mask_dilation, self.mask_dilation))
            mask_ = cv2.dilate(mask_, kernel)
            masks.append(torch.from_numpy(mask_))

        masks = torch.stack(masks)
        masks = (masks == 1).type(torch.FloatTensor).unsqueeze(0)
        images = torch.stack(images).permute(3,0,1,2)

        return images, masks, info
