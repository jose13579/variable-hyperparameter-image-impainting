import os
import sys
import io
import cv2
import time
import argparse
import shutil
import random
import zipfile
from glob import glob
import math
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image, ImageOps, ImageDraw, ImageFilter

import torch
import torchvision
import torch.nn as nn
import torch.distributed as dist

import matplotlib
import matplotlib.patches as patches
from matplotlib.path import Path
from matplotlib import pyplot as plt
matplotlib.use('agg')


# ###########################################################################
# ###########################################################################


class GroupRandomHorizontalFlip(object):
    """Randomly horizontally flips the given PIL.Image with a probability of 0.5
    """

    def __init__(self, is_flow=False):
        self.is_flow = is_flow

    def __call__(self, img, is_flow=False):
        v = random.random()
        if v < 0.5:
            ret = img.transpose(Image.FLIP_LEFT_RIGHT)
            if self.is_flow:
              ret = ImageOps.invert(ret)
            return ret
        else:
            return img


class Stack(object):
    def __init__(self, roll=False):
        self.roll = roll

    def __call__(self, img):
        mode = img.mode
        if mode == '1':
            img = img.convert('L')
            mode = 'L'
        if mode == 'L':
            return np.expand_dims(img, 2)
        elif mode == 'RGB':
            if self.roll:
                return np.array(img)[:, :, ::-1]
            else:
                return img
        else:
            raise NotImplementedError(f"Image mode {mode}")


class ToTorchFormatTensor(object):
    """ Converts a PIL.Image (RGB) or numpy.ndarray (H x W x C) in the range [0, 255]
    to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0] """

    def __init__(self, div=True):
        self.div = div

    def __call__(self, pic):
        if isinstance(pic, np.ndarray):
            # numpy img: [C, H, W]
            img = torch.from_numpy(pic).permute(2, 0, 1).contiguous()
        else:
            # handle PIL Image
            img = torch.ByteTensor(
                torch.ByteStorage.from_buffer(pic.tobytes()))
            img = img.view(pic.size[1], pic.size[0], len(pic.mode))
            # put it from HWC to CHW format
            # yikes, this transpose takes 80% of the loading time/CPU
            img = img.transpose(0, 1).transpose(0, 2).contiguous()
        img = img.float().div(255) if self.div else img.float()
        return img


# ##########################################
# ##########################################

def continuous_mask(height, width,num,maxAngle,maxLength,maxBrushWidth,channels=3):
    """Generates a continuous mask with lines, circles and elipses"""

    img = np.zeros((height, width, channels), np.uint8)

    for j in range(1):
        startX = random.randint(0, width)
        startY = random.randint(0, height)
        for i in range(0,random.randint(1,num)):
            angle = random.randint(0,maxAngle)
            if i%2==0:
                angle = 360 - angle
            length = random.randint(maxLength//2,maxLength)
            brushWidth = random.randint(1, maxBrushWidth)
            endX   = startX + int(length * np.sin(angle))
            endY   = startY + int(length * np.cos(angle))
            if endX>255:
                endX = 255
            if endX<0:
                endX = 0
            if endY>255:
                endY = 255
            if endY<0:
                endY = 0        
            cv2.line(img, (startX,startY),(endX,endY),(255,255,255),brushWidth)
            cv2.circle(img, (endX,endY),brushWidth//2,(255,255,255),-1)
            startY = endY
            startX = endX


    img2 = np.zeros((height, width,1))
    img2[:, :,0] = img[:, :, 0]
    img2[img2>1] = 1

    img2 = np.squeeze(img2, axis=2)

    m = Image.fromarray((img2 * 255).astype(np.uint8))
    masks = [m.convert('L')]

    return masks

# ##############################################
# ##############################################

if __name__ == '__main__':

    trials = 10
    for _ in range(trials):
        mask_length = 5

        i = 0
        for i in range(mask_length):
          # The returned masks are either stationary (50%) or moving (50%)
          mask = continuous_mask(256,256,60,360,50,50)
          mask = np.array(mask[0])
          cv2.imwrite("mask_{}.png".format(i),(1-mask)*255)
          i += 1
