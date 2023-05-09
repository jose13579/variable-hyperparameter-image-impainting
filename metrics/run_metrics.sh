export CUDA_VISIBLE_DEVICES=0

groundtruth_path=$1 #/data/celeba/celeba_dataset/test/
predicted_path=$2 #/config/variable-hyperparameter-image-impainting/test_trained_models/trained_celeba_VHII_efficient_seed_0

python test.py --path $groundtruth_path $predicted_path
