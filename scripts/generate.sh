# ------------------------------------------------------------------------------------
# Copyright 2023–2024 Nota Inc. All Rights Reserved.
# ------------------------------------------------------------------------------------

: <<'END'
Assuming the below directory structure from training (scripts/kd_train.sh):
    ./results/kd_bk_small
    |-- checkpoint-40000
    |    |-- unet
    |    |-- unet_ema
    |-- checkpoint-45000
    |    |-- unet
    |    |-- unet_ema
    |-- ...
    |-- text_encoder
    |-- unet
    |-- vae
END

# (C) Examples for SD-v2:
IMG_DIR=./outputs/C-ddp_v2-base/checkpoint-4000
python3 src/generate.py \
    --unet_path ./results/C-ddp_v2-base_kd_bk_base/checkpoint-4000 --model_id stabilityai/stable-diffusion-2-1 \
    --save_dir $IMG_DIR --img_sz 768

