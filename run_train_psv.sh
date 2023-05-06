export CUDA_VISIBLE_DEVICES=2,3
python3 train_psv.py --config configs/psv_proposal.json --model inpainting_proposal
