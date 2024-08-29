IMG_DIR=./outputs/bksdm-v2-base-212k/checkpoint-30k
python src/ddp_gen_launcher.py \
    --unet_path ./results/bksdm-v2-base-212k/checkpoint-30000 \
    --img_save_dir $IMG_DIR --img_sz 768 \
    