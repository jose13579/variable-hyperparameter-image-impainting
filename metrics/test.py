import torch, os
from evaluation import metrics
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('--path', type=str, nargs=2,
                    help=('Path to the generated images'))

args = parser.parse_args()
path1, path2 = args.path

psnr_score = metrics['psnr']([path1, path2])
print("Generated PSNR:{}".format(psnr_score))
ssim_score = metrics['ssim']([path1, path2])
print("Generated SSIM:{}".format(ssim_score))
meanl1_error = metrics['meanl1']([path1, path2])
print("Generated Mean L1:{}".format(meanl1_error))
lpips_score = metrics['lpips']([path1, path2])
print("Generated LPIPS:{}".format(lpips_score))
fid_score = metrics['fid']([path1, path2],cuda=True)
print("Generated FID:{}".format(fid_score))
