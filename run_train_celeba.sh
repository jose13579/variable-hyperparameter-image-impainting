export CUDA_VISIBLE_DEVICES=2,3
python3 train_celeba.py --config configs/celeba_proposal.json --model inpainting_proposal
