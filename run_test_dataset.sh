export CUDA_VISIBLE_DEVICES=0

model_name=$1 # VHII_efficient

load_model_dir=$2 #trained_models/celeba/celeba_VHII_efficient_seed_0/gen_00050.pth

set_seed=$3 #0, 42 and 123

images_dataset=$4 #/data/celeba/celeba_dataset/test/

mask_dataset=$5 #/data/pconv/test_mask/20-30/

output_path_Samples=$6 #test_trained_models/trained_celeba_VHII_efficient_seed_0

mkdir $output_path_Samples

output_path_masks_samples=$output_path_Samples/output_path_masks
output_path_inpainted_samples=$output_path_Samples/output_path_inpainted
output_path_incompleted_samples=$output_path_Samples/output_path_incompleted
output_path_groundtruth_samples=$output_path_Samples/output_path_groundtruth
output_path_full_inpainted_samples=$output_path_Samples/output_images

python test_dataset.py --model $model_name --ckpt $load_model_dir --set_seed $set_seed --input_path_images $images_dataset --input_path_masks $mask_dataset --output_path_masks $output_path_masks_samples --output_path_inpainted $output_path_inpainted_samples --output_path_incompleted $output_path_incompleted_samples --output_path_groundtruth $output_path_groundtruth_samples --output_path_full_inpainted $output_path_full_inpainted_samples --print_samples
