export CUDA_VISIBLE_DEVICES=0

model_name=$1 #VHII_efficient
load_model_dir=$2 #trained_models/celeba/celeba_VHII_efficient_seed_0/gen_00050.pth

img_path=$3 #examples/img/100_000100_gt.png
mask_path=$4 #examples/mask/100_000100_mask.png
output_path=$5 #examples/output
output_name=$6 #"100_000100_output"

mkdir $output_path

python test.py --model $model --ckpt $load_model_dir  --image $img_path --mask $mask_path --output_path $output_path --output_name $output_name
