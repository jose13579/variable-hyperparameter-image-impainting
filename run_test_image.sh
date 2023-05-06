export CUDA_VISIBLE_DEVICES=0
input_path=examples/imgs/rgb/img_6.png
mask_path=examples/imgs/mask/img_6.png
output_path==examples/output
mkdir $output_path

model_path=trained_models/trained_model_lsgan_perceptual_style_continuous_proposed/sttn_new_path_sizes_places_lsgan_continuous_proposed/gen_00050.pth

model="VHII"

output_name="000000010764_sttn_new_path_sizes"

python test.py --image $input_path --mask $mask_path --ckpt $model_path --output_name $output_name --output_path $output_path --model $model
