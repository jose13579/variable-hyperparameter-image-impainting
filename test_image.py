# -*- coding: utf-8 -*-
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import math
import time
import importlib
import os
import argparse
import copy
import datetime
import random
import sys
import json
import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.utils.model_zoo as model_zoo
import torch.multiprocessing as mp
from torchvision import models
from torchvision import transforms
from core.utils import Stack, ToTorchFormatTensor
from torch.autograd import Variable


parser = argparse.ArgumentParser(description="VHII")
parser.add_argument("--image", type=str, required=True)
parser.add_argument("--mask",   type=str, required=True)
parser.add_argument("--ckpt",   type=str, required=True)
parser.add_argument("--width",   type=int, default=256)
parser.add_argument("--height",   type=int, default=256)
parser.add_argument("--output_name",   type=str, default="test_0")
parser.add_argument("--output_path",   type=str, default="results")
parser.add_argument("--model",   type=str, default='VHII')

args = parser.parse_args()

output_path = args.output_path

w, h = args.width, args.height

_to_tensors = transforms.Compose([
    Stack(),
    ToTorchFormatTensor()])

# read frame-wise masks 
def read_mask(mpath):
    m = Image.open(mpath)
    m = m.resize((w, h), Image.NEAREST)
    m = np.array(m.convert('L'))
    m = np.array(m > 0).astype(np.uint8)
    return Image.fromarray(m*255)   


def main_worker():
    # set up models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = importlib.import_module('model.' + args.model)
    model = net.InpaintGenerator().to(device)
    model_path = args.ckpt
    data = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(data['netG'])
    
    print('loading from: {}'.format(args.ckpt))
    model.eval()
    
    base=os.path.basename(args.mask)
    base=os.path.splitext(base)[0]

    # prepare dataset, encode all frames into deep space 
    frames = Image.open(args.image).convert('RGB')
    frames = frames.resize((w, h))

    feats = _to_tensors(frames).unsqueeze(0)*2-1
    frames = np.array(frames).astype(np.uint8)

    masks = read_mask(args.mask)
    binary_masks = np.expand_dims((np.array(masks) != 0).astype(np.uint8), 2)
    
    cv2.imwrite(f"{output_path}/{base}_mask_{args.output_name}.png",binary_masks*255)
    masks = _to_tensors(masks).unsqueeze(0)
    
    feats, masks = feats.to(device), masks.to(device)
    
    # begin inference 
    with torch.no_grad():
         masked_imgs = feats*(1-masks)
         current_img = model(masked_imgs,masks)
         pred_img = (current_img+1)/2
         pred_img = pred_img.cpu().permute(0,2,3,1).numpy()*255    

    print('loading image and mask from: {} and {}'.format(args.image,args.mask))
        
    inpainted_img = np.array(pred_img[0]).astype(np.uint8)
    inpainted_img = inpainted_img*(binary_masks)+frames*(1-binary_masks)
    
    incompleted_img = frames*(1-binary_masks)+255*binary_masks
    
    cv2.imwrite(f"{output_path}/{base}_groundtruth_{args.output_name}.png",frames[...,::-1])
    cv2.imwrite(f"{output_path}/{base}_incompleted_{args.output_name}.png",incompleted_img[...,::-1])
    cv2.imwrite(f"{output_path}/{base}_inpainted_{args.output_name}.png",inpainted_img[...,::-1])
    
    print('Finish')

if __name__ == '__main__':
    main_worker()
