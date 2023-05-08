from .inception_score.inception_score import inception_score
from .fid.fid import calculate_fid_given_paths
from .ssim.ssim import ssim
from .psnr.psnr import psnr
import os
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import cv2
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity

import evaluation.lpips


"""
def inception_score(imgs, cuda=True, batch_size=32, resize=False, splits=1):
def calculate_fid_given_paths(paths, batch_size, cuda, dims):
def ssim(img1, img2, window_size = 11, size_average = True):
"""
SIZE = (256,256)
_transforms_fun=transforms.Compose([transforms.Resize((299,299)), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])

def _inception_score(path, cuda=False, batch_size=1, resize=True, splits=1):
    imgs = []
    for file in os.listdir(path):
        if file.endswith("png") or file.endswith("jpg"):
            img = Image.open(os.path.join(path, file)).convert("RGB")
            imgs.append(_transforms_fun(img))
    imgs = torch.stack(imgs)
    #print(imgs.size())
    return inception_score(imgs, cuda, batch_size, resize, splits)

def _fid(paths, batch_size=10, cuda=False, dims=2048):
    return calculate_fid_given_paths(paths, batch_size, cuda, dims)
    
def _psnr(paths):
    path1, path2 = paths
    imgs1, imgs2 = [], []
    psnr_total = 0
    num = 0
    for file in os.listdir(path1):
        if file.endswith("png") or file.endswith("jpg"):
            base=os.path.basename(file)
            base=os.path.splitext(base)[0]

            if os.path.exists(os.path.join(path1, base+".png")):
               img1 = Image.open(os.path.join(path1, base+".png")).convert("RGB")
            else:
               img1 = Image.open(os.path.join(path1, base+".jpg")).convert("RGB")

            if os.path.exists(os.path.join(path2, base+".png")):
               img2 = Image.open(os.path.join(path2, base+".png")).convert("RGB")
            else:
               img2 = Image.open(os.path.join(path2, base+".jpg")).convert("RGB")

            psnr_value = peak_signal_noise_ratio(cv2.resize(np.array(img1),SIZE), cv2.resize(np.array(img2), SIZE))
            print(base+" PSNR",psnr_value)
            psnr_total = psnr_total + psnr_value
            num = num + 1

    return psnr_total / num

def _ssim(paths):
    path1, path2 = paths
    imgs1, imgs2 = [], []
    ssim_total = 0
    num = 0
    for file in os.listdir(path1):
        if file.endswith("png") or file.endswith("jpg"):
            base=os.path.basename(file)
            base=os.path.splitext(base)[0]
            
            if os.path.exists(os.path.join(path1, base+".png")):
                img1 = Image.open(os.path.join(path1, base+".png")).convert("RGB")
            else:
                img1 = Image.open(os.path.join(path1, base+".jpg")).convert("RGB")

            if os.path.exists(os.path.join(path2, base+".png")):
                img2 = Image.open(os.path.join(path2, base+".png")).convert("RGB")
            else:
                img2 = Image.open(os.path.join(path2, base+".jpg")).convert("RGB")

            ssim_value = structural_similarity(cv2.resize(np.array(img1),SIZE), cv2.resize(np.array(img2), SIZE),multichannel=True)
            print(base+" SSIM",ssim_value)
            ssim_total = ssim_total + ssim_value
            num = num + 1

    return ssim_total / num

def _meanl1(paths):
    path1, path2 = paths
    imgs1, imgs2 = [], []
    total_error = 0
    num = 1
    for file in os.listdir(path1):
        if file.endswith("png") or file.endswith("jpg"):
            base=os.path.basename(file)
            base=os.path.splitext(base)[0]
            
            if os.path.exists(os.path.join(path1, base+".png")):
               img1 = Image.open(os.path.join(path1, base+".png")).convert("RGB")
            else:
               img1 = Image.open(os.path.join(path1, base+".jpg")).convert("RGB")

            if os.path.exists(os.path.join(path2, base+".png")):
               img2 = Image.open(os.path.join(path2, base+".png")).convert("RGB")
            else:
               img2 = Image.open(os.path.join(path2, base+".jpg")).convert("RGB")

            l1_error = np.mean(np.abs(cv2.resize(np.array(img1),SIZE)-cv2.resize(np.array(img2), SIZE)))
            print(base+" MeanL1",l1_error)
            total_error = total_error + l1_error
            num = num + 1

    return total_error / num
    
## Initializing the model
loss_fn = lpips.LPIPS(net='alex',version='0.1')
loss_fn.cuda()
	
def _lpips(paths):
    path1, path2 = paths
    imgs1, imgs2 = [], []
    lips_value = 0
    num = 0
    for file in os.listdir(path1):
        if file.endswith("png") or file.endswith("jpg"):
            base=os.path.basename(file)
            base=os.path.splitext(base)[0]
            
            if os.path.exists(os.path.join(path1, base+".png")):
               img1 = lpips.im2tensor(lpips.load_image(os.path.join(path1,base+".png"))) # RGB image from [-1,1]
            else:
               img1 = lpips.im2tensor(lpips.load_image(os.path.join(path1,base+".jpg")))

            if os.path.exists(os.path.join(path2, base+".png")):
               img2 = lpips.im2tensor(lpips.load_image(os.path.join(path2,base+".png")))
            else:
               img2 = lpips.im2tensor(lpips.load_image(os.path.join(path2,base+".jpg")))
            
            img1 = img1.cuda()
            img2 = img2.cuda()
	    
	    # Compute distance
            dist = loss_fn.forward(img1,img2).item()
            lips_value = lips_value + dist
            print(base+" LPIPS",dist)

            num = num + 1

    return lips_value / num

metrics = {"is":_inception_score, "fid":_fid, "lpips":_lpips ,"ssim":_ssim, "psnr":_psnr, "meanl1":_meanl1}
