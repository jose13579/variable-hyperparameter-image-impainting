import os
import cv2
import io
import glob
import scipy
import json
import zipfile
import random
import collections
import torch
import math
import numpy as np

from PIL import Image, ImageFilter
from skimage.color import rgb2gray, gray2rgb

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from core.utils import continuous_mask
from core.utils import Stack, ToTorchFormatTensor, GroupRandomHorizontalFlip

def get_images_names(train_data_path, dataset_name):
    if dataset_name == "places":
       fnames = glob.glob(train_data_path + '/*/*/*.jpg')
       fnames = fnames + glob.glob(train_data_path + '/*/*/*/*.jpg')
    elif dataset_name == "celeba":
       fnames = glob.glob(train_data_path + '/train/*.jpg')
    elif dataset_name == "psv":
       fnames = glob.glob(train_data_path + '*.JPG')
    else:
       fnames = []

    return fnames

class Dataset(torch.utils.data.Dataset):
    def __init__(self, args: dict, split='train', dataset_name='places'):
        self.args = args
        self.split = split
        self.size = self.w, self.h = (args['w'], args['h'])

        path = os.path.join(args['data_root'])
        self.image_names = get_images_names(path, dataset_name)

        self._to_tensors = transforms.Compose([
            Stack(),
            ToTorchFormatTensor(), ])

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, index):
        try:
            item = self.load_item(index)
        except:
            print('Loading error in image {}'.format(self.image_names[index]))
            item = self.load_item(0)
        return item

    def load_item(self, index):
        image_name = self.image_names[index]
        mask = continuous_mask(self.h,self.w,60,360,50,50)
        
        # read images
        img = Image.open('{}'.format(image_name)).convert('RGB')
        img = img.resize(self.size)
        
        images = img
        masks = mask[0]

        if self.split == 'train':
            images = GroupRandomHorizontalFlip()(images)
            
        # To tensors
        image_tensors = self._to_tensors(images)*2.0 - 1.0
        mask_tensors = self._to_tensors(masks)
        return image_tensors, mask_tensors
