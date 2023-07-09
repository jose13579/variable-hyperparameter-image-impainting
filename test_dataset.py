# -*- coding: utf-8 -*-
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from scipy import stats
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
import glob
import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.utils.model_zoo as model_zoo
import torch.multiprocessing as mp

from torch.autograd import Variable
from torchvision import models
from torchvision import transforms
from core.utils import Stack, ToTorchFormatTensor
from thop import profile, clever_format

parser = argparse.ArgumentParser(description="VHII efficient")
parser.add_argument("--input_path_images", type=str, required=True)
parser.add_argument("--input_path_masks",   type=str, required=True)
parser.add_argument("--ckpt",   type=str, required=True)
parser.add_argument("--width",   type=int, default=256)
parser.add_argument("--height",   type=int, default=256)
parser.add_argument("--output_path_masks",   type=str, default="results_masks")
parser.add_argument("--output_path_inpainted",   type=str, default="results_inpainted")
parser.add_argument("--output_path_incompleted",   type=str, default="results_incompleted")
parser.add_argument("--output_path_groundtruth",   type=str, default="results_groundtruth")
parser.add_argument("--output_path_full_inpainted",   type=str, default="results_full_inpainted")
parser.add_argument("--print_samples", action='store_true')
parser.add_argument("--set_seed",  type=int, default=0)
parser.add_argument("--model",   type=str, default='VHII_efficient')

args = parser.parse_args()

output_path_masks = args.output_path_masks
output_path_inpainted = args.output_path_inpainted
output_path_incompleted = args.output_path_incompleted
output_path_groundtruth = args.output_path_groundtruth
output_path_full_inpainted = args.output_path_full_inpainted
print_samples = args.print_samples
set_seed = args.set_seed
number_samples = 5

if os.path.isdir(output_path_masks) == False:
   if print_samples:
      os.mkdir(output_path_masks)
		
if os.path.isdir(output_path_inpainted) == False:
   if print_samples:
      os.mkdir(output_path_inpainted)
      
if os.path.isdir(output_path_incompleted) == False:
   if print_samples:
      os.mkdir(output_path_incompleted)
      
if os.path.isdir(output_path_groundtruth) == False:
   if print_samples:
      os.mkdir(output_path_groundtruth)
      
if os.path.isdir(output_path_full_inpainted) == False:
   os.mkdir(output_path_full_inpainted)


w, h = args.width, args.height

_to_tensors = transforms.Compose([
    Stack(),
    ToTorchFormatTensor()])


# read images-wise masks 
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

   imgfile = sorted(glob.glob(os.path.join(args.input_path_images, '*.jpg'))) + sorted(glob.glob(os.path.join(args.input_path_images, '*.png')))
   maskfile = sorted(glob.glob(os.path.join(args.input_path_masks, '*.png')))

   total_number = len(imgfile)
   mask_number = len(maskfile)
     
   print("number of images: ", total_number)
   print("number of masks: ", mask_number)

   totalTime = 0

   random.seed(set_seed)
   for i in range(total_number):
      startTime = time.time() 
      images = Image.open(imgfile[i]).convert('RGB')
      images = images.resize((w, h))

      base=os.path.basename(imgfile[i])
      base=os.path.splitext(base)[0]

      feats = _to_tensors(images).unsqueeze(0)*2-1

      images = np.array(images).astype(np.uint8)

      rand_mask = random.randint(0, mask_number - 1)
      masks = read_mask(maskfile[rand_mask])

      binary_masks = np.expand_dims((np.array(masks) != 0).astype(np.uint8), 2)

      masks = _to_tensors(masks).unsqueeze(0)

      feats, masks = feats.to(device), masks.to(device)
 
      # begin inference 
      with torch.no_grad():
         masked_imgs = feats*(1-masks)
         current_img = model(masked_imgs,masks)
         pred_img = (current_img+1)/2
         pred_img = pred_img.cpu().permute(0,2,3,1).numpy()*255

      inpainted_img = np.array(pred_img[0]).astype(np.uint8)
      inpainted_img = inpainted_img*(binary_masks)+images*(1-binary_masks)
 
      incompleted_img = images*(1-binary_masks)+255*binary_masks
     
      endTime = (time.time() - startTime)
      totalTime += endTime

      if print_samples:
         if (i % (total_number // number_samples)) == 0:
            cv2.imwrite(f"{output_path_masks}/{base}.png",binary_masks*255)
            cv2.imwrite(f"{output_path_groundtruth}/{base}.png",images[...,::-1])
            cv2.imwrite(f"{output_path_incompleted}/{base}.png",incompleted_img[...,::-1])
            cv2.imwrite(f"{output_path_inpainted}/{base}.png",inpainted_img[...,::-1])

      cv2.imwrite(f"{output_path_full_inpainted}/{base}.png",inpainted_img[...,::-1])

   totalTime = totalTime / total_number
   print("--- %s seconds ---" % totalTime)
   macs, params = profile(model,inputs=(masked_imgs,masks))
   flops = 2*macs
   macs, params, flops = clever_format([macs, params, flops], "%.3f")
   print(" --- macs: ",macs, " params: ",params, " flops: ", flops)
   print('Finish')


if __name__ == '__main__':
   main_worker()
