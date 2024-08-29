import os
import csv
import argparse
import numpy as np
from ipdb import set_trace as st

def get_file_list_from_csv(csv_file_path):
    file_list = []
    with open(csv_file_path, newline='') as csvfile:
        csv_reader = csv.reader(csvfile)        
        next(csv_reader, None) # Skip the header row
        for row in csv_reader: # (row[0], row[1]) = (img name, txt prompt) 
            file_list.append(row)
    return file_list

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--seed', type=int, default=10515,
                        help='random seed to select samples')
    parser.add_argument("--data_list", type=str, 
                        default="/home/wanghesong/Datasets/mscoco_val2014_30k/metadata.csv") 
    parser.add_argument("--save_dir", type=str, 
                        default="/home/wanghesong/Datasets/mscoco_val2014_30k")
    
    parser.add_argument('--num_samples', type=int, default=4,
                        help='how many samples are selected')
    parser.add_argument('--num_splits', type=int, default=1,
                        help='how many samples are selected')
    
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    # region-devide captions
    args = parse_args()
    file_list = get_file_list_from_csv(args.data_list)
    
    np.random.seed(args.seed)
    picked_index = np.random.permutation(len(file_list))[:args.num_samples]
    picked_index = picked_index.tolist()
    
    temp = []
    total = []
    for cnt, idx in enumerate(picked_index):
        file = file_list[idx]
        temp.append(file)
        total.append(file)
        
        assert args.num_samples % args.num_splits == 0, f"Error: {args.num_samples} is not divisible by {args.num_splits}."
        num_per_split = args.num_samples // args.num_splits
        
        if (cnt+1) % num_per_split == 0:
            csv_file = f"subdata_{((cnt+1)//num_per_split):02}.csv"
            csv_path = os.path.join(args.save_dir, csv_file)
            
            with open(csv_path, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["file_name", "text"])
                for row in temp:
                    writer.writerow(row)
                    
            temp.clear()
    
    #* 整个subdata存一份
    csv_file = "subdata_all.csv"
    csv_path = os.path.join(args.save_dir, csv_file)
    
    with open(csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["file_name", "text"])
        for row in total:
            writer.writerow(row)
    # endregion-devide captions
    
    # option = sys.argv[1]
    # print(option)
    for s in range(args.num_splits): 
        if False: 
            log_dir = './outputs/ddp-gen/logs'
            os.makedirs(log_dir, exist_ok=True)
            os.system(f"nohup bash -c 'IMG_DIR=./outputs/ddp-gen && "
                        f"python3 src/generate.py "
                        f"--unet_path ./results/ddp_v2-base/checkpoint-5000 "
                        f"--model_id stabilityai/stable-diffusion-2-1 "
                        f"--save_dir $IMG_DIR --img_sz 768 "
                        f"--data_list {args.save_dir}/subdata_{s+1:02}.csv "
                        f"--device \"cuda:{s}\"' > ./outputs/ddp-gen/logs/nohup_{s+1:02}.out 2>&1 &")
        elif False: 
            log_dir = './outputs/bksdm-v2-base-212k-ckpt_30k/logs'
            os.makedirs(log_dir, exist_ok=True)
            os.system(f"nohup bash -c 'IMG_DIR=./outputs/bksdm-v2-base-212k-ckpt_30k && "
                        f"python3 src/generate.py "
                        f"--unet_path ./results/bksdm-v2-base-212k/checkpoint-30000 "
                        f"--model_id stabilityai/stable-diffusion-2-1 "
                        f"--save_dir $IMG_DIR --img_sz 768 "
                        f"--data_list {args.save_dir}/subdata_{s+1:02}.csv "
                        f"--device \"cuda:{s}\"' > ./outputs/bksdm-v2-base-212k-ckpt_30k/logs/nohup_{s+1:02}.out 2>&1 &")
        elif False: 
            log_dir = './outputs/bksdm-v2-base-212k-ckpt_50k-3w/logs'
            os.makedirs(log_dir, exist_ok=True)
            os.system(f"nohup bash -c 'IMG_DIR=./outputs/bksdm-v2-base-212k-ckpt_50k-3w && "
                        f"python3 src/generate.py "
                        f"--unet_path nota-ai/bk-sdm-v2-base "
                        f"--model_id nota-ai/bk-sdm-v2-base "
                        f"--save_dir $IMG_DIR --img_sz 768 "
                        f"--data_list {args.save_dir}/subdata_{s+1:02}.csv "
                        f"--device \"cuda:{s}\"' > ./outputs/bksdm-v2-base-212k-ckpt_50k-3w/logs/nohup_{s+1:02}.out 2>&1 &")
        else: 
            log_dir = './outputs/4-imgs-test/logs'
            os.makedirs(log_dir, exist_ok=True)
            os.system(f"nohup bash -c 'IMG_DIR=./outputs/4-imgs-test && "
                        f"python3 src/generate.py "
                        f"--unet_path nota-ai/bk-sdm-v2-base "
                        f"--model_id nota-ai/bk-sdm-v2-base "
                        f"--save_dir $IMG_DIR --img_sz 768 "
                        f"--data_list {args.save_dir}/subdata_{s+1:02}.csv "
                        f"--device \"cuda:{s}\"' > ./outputs/4-imgs-test/logs/nohup_{s+1:02}.out 2>&1 &")         

    print("=======DONE=======")
    
    
    
    
    
    
    
    
    
    