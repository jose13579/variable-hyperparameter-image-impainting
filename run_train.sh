export CUDA_VISIBLE_DEVICES=0

config=$1 #configs/celeba_proposal.json
python3 train.py --config $config
