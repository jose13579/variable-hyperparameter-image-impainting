export CUDA_VISIBLE_DEVICES=0

model="inpainting_proposal"
set_seed=0 # 42 and 123

model_path=trained_models/psv/psv_proposal_inpainting_proposal_seed_$set_seed/gen_00050.pth
output_path_model=test_trained_models/trained_psv_inpainting_proposal_seed_$set_seed

print_samples=1

mkdir $output_path_model



input_path_images=/data/psv/paris_eval/paris_eval_gt/
input_path_masks=/data/pconv/test_mask/20-30/

output_path=$output_path_model/20-30
mkdir $output_path

output_path_masks=$output_path/output_path_masks
output_path_inpainted=$output_path/output_path_inpainted
output_path_incompleted=$output_path/output_path_incompleted
output_path_groundtruth=$output_path/output_path_groundtruth
output_path_full_inpainted=$output_path/output_images

python test_dataset.py --input_path_images $input_path_images --input_path_masks $input_path_masks --ckpt $model_path --output_path_masks $output_path_masks --output_path_inpainted $output_path_inpainted --output_path_incompleted $output_path_incompleted --output_path_groundtruth $output_path_groundtruth --output_path_full_inpainted $output_path_full_inpainted --print_samples $print_samples --set_seed $set_seed --model $model






input_path_images=/data/psv/paris_eval/paris_eval_gt/
input_path_masks=/data/pconv/test_mask/30-40/

output_path=$output_path_model/30-40
mkdir $output_path

output_path_masks=$output_path/output_path_masks
output_path_inpainted=$output_path/output_path_inpainted
output_path_incompleted=$output_path/output_path_incompleted
output_path_groundtruth=$output_path/output_path_groundtruth
output_path_full_inpainted=$output_path/output_images

python test_dataset.py --input_path_images $input_path_images --input_path_masks $input_path_masks --ckpt $model_path --output_path_masks $output_path_masks --output_path_inpainted $output_path_inpainted --output_path_incompleted $output_path_incompleted --output_path_groundtruth $output_path_groundtruth --output_path_full_inpainted $output_path_full_inpainted --print_samples $print_samples --set_seed $set_seed --model $model







input_path_images=/data/psv/paris_eval/paris_eval_gt/
input_path_masks=/data/pconv/test_mask/40-50/

output_path=$output_path_model/40-50
mkdir $output_path

output_path_masks=$output_path/output_path_masks
output_path_inpainted=$output_path/output_path_inpainted
output_path_incompleted=$output_path/output_path_incompleted
output_path_groundtruth=$output_path/output_path_groundtruth
output_path_full_inpainted=$output_path/output_images

python test_dataset.py --input_path_images $input_path_images --input_path_masks $input_path_masks --ckpt $model_path --output_path_masks $output_path_masks --output_path_inpainted $output_path_inpainted --output_path_incompleted $output_path_incompleted --output_path_groundtruth $output_path_groundtruth --output_path_full_inpainted $output_path_full_inpainted --print_samples $print_samples --set_seed $set_seed --model $model
