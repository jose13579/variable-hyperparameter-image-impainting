export CUDA_VISIBLE_DEVICES=2,3
python3 train_places.py --config configs/places_proposal.json --model inpainting_proposal
