python3 text2image.py \
  --pretrained_model_name_or_path /apdcephfs_cq2/share_1290939/shaoshuyang/stable-diffusion-2-1-base \
  --validation_prompt "A corgi sits on a beach chair on a beautiful beach, with palm trees behind, high details" \
  --seed 23 \
  --config ./configs/sd2.1_2048x2048_disperse.yaml \
  --logging_dir /apdcephfs/share_1290939/shaoshuyang/t2i/addm/disperse_test
