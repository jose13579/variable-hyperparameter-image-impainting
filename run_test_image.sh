export CUDA_VISIBLE_DEVICES=0
input_path=$1 #examples/imgs/rgb/img_6.png
mask_path=$2 #examples/imgs/mask/img_6.png
output_path=$3 #examples/output
mkdir $output_path

model_path=$4 #trained_models/celeba/celeba_proposal_inpainting_proposal_seed_0/gen_00050.pth

model="VHII"

output_name="000000010764_sttn_new_path_sizes"

python test.py --image $input_path --mask $mask_path --ckpt $model_path --output_name $output_name --output_path $output_path --model $model
