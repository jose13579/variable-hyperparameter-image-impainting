export CUDA_VISIBLE_DEVICES=0

model_name=$1 #VHII_efficient
load_model_dir=$2 #trained_models/celeba/celeba_VHII_efficient_seed_0/gen_00050.pth

img_path=$3 #examples/img/100_000100_gt.png
mask_path=$4 #examples/mask/100_000100_mask.png
output_path=$5 #examples/output
output_name=$6 #"100_000100_output"

mkdir $output_path

python test_image.py --model_name $model_name --ckpt $load_model_dir  --img_path $img_path --mask_path $mask_path --output_path $output_path --output_name $output_name
