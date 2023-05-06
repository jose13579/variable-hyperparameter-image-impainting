export CUDA_VISIBLE_DEVICES=0

config=$1 #configs/celeba_proposal.json
model_name=$2 #VHII_efficient
python3 train_celeba.py --config $config --model $model_name
