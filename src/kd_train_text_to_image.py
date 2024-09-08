# region-comments
# ------------------------------------------------------------------------------------
# Copyright 2023. Nota Inc. All Rights Reserved.
# Code modified from https://github.com/huggingface/diffusers/blob/v0.15.0/examples/text_to_image/train_text_to_image.py
# ------------------------------------------------------------------------------------
#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# endregion-comments

# region-imports
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import argparse
import logging
import math
import os
import random
from pathlib import Path
from typing import Optional

import accelerate
import datasets
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset
from packaging import version
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import check_min_version, deprecate
from diffusers.utils.import_utils import is_xformers_available

import csv
import time
import copy

from ipdb import set_trace as st

from smilelogging import Logger  # ==> Add this line

# try to import wandb
try:
    import wandb
    has_wandb = True
except:
    has_wandb = False
# endregion-imports

# region-out_of_main
# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.15.0")

logger = get_logger(__name__, log_level="INFO")

def get_activation(mem, name):
    def get_output_hook(module, input, output):
        mem[name] = output

    return get_output_hook

def add_hook(net, mem, mapping_layers):
    #* named_modules(): 返回模型中所有子模块，按层遍历
    #* n 是子模块的名称（如 'up_blocks.0'、'down_blocks.1'），m是实际的对象
    for n, m in net.named_modules():
        if n in mapping_layers:
            #* register_forward_hook()： pytorch中的方法
            m.register_forward_hook(get_activation(mem, n))

#todo 返回继承了参数但未经过剪枝的学生模型
def copy_weight_from_teacher(unet_stu, unet_tea, student_type):

    #* 首先定义一个字典，用于存储stu和tea之间的映射关系
    connect_info = {} # connect_info['TO-student'] = 'FROM-teacher'
    if student_type in ["bk_base", "bk_small"]:
        connect_info['up_blocks.0.resnets.1.'] = 'up_blocks.0.resnets.2.'
        connect_info['up_blocks.1.resnets.1.'] = 'up_blocks.1.resnets.2.'
        connect_info['up_blocks.1.attentions.1.'] = 'up_blocks.1.attentions.2.'
        connect_info['up_blocks.2.resnets.1.'] = 'up_blocks.2.resnets.2.'
        connect_info['up_blocks.2.attentions.1.'] = 'up_blocks.2.attentions.2.'
        connect_info['up_blocks.3.resnets.1.'] = 'up_blocks.3.resnets.2.'
        connect_info['up_blocks.3.attentions.1.'] = 'up_blocks.3.attentions.2.'
    elif student_type in ["bk_tiny"]:
        connect_info['up_blocks.0.resnets.0.'] = 'up_blocks.1.resnets.0.'
        connect_info['up_blocks.0.attentions.0.'] = 'up_blocks.1.attentions.0.'
        connect_info['up_blocks.0.resnets.1.'] = 'up_blocks.1.resnets.2.'
        connect_info['up_blocks.0.attentions.1.'] = 'up_blocks.1.attentions.2.'
        connect_info['up_blocks.0.upsamplers.'] = 'up_blocks.1.upsamplers.'
        connect_info['up_blocks.1.resnets.0.'] = 'up_blocks.2.resnets.0.'
        connect_info['up_blocks.1.attentions.0.'] = 'up_blocks.2.attentions.0.'
        connect_info['up_blocks.1.resnets.1.'] = 'up_blocks.2.resnets.2.'
        connect_info['up_blocks.1.attentions.1.'] = 'up_blocks.2.attentions.2.'
        connect_info['up_blocks.1.upsamplers.'] = 'up_blocks.2.upsamplers.'
        connect_info['up_blocks.2.resnets.0.'] = 'up_blocks.3.resnets.0.'
        connect_info['up_blocks.2.attentions.0.'] = 'up_blocks.3.attentions.0.'
        connect_info['up_blocks.2.resnets.1.'] = 'up_blocks.3.resnets.2.'
        connect_info['up_blocks.2.attentions.1.'] = 'up_blocks.3.attentions.2.'       
    else:
        raise NotImplementedError

    for k in unet_stu.state_dict().keys(): #* 变量模型所有参数（的key）, .state_dict(): 返回模型的状态字典，包含了模型所有参数的名称和值
        flag = 0 #* 标志变量，用于指示当前键是否需要强制复制
        k_orig = k #* 初始的键名，用于在需要时替换前缀
        for prefix_key in connect_info.keys(): #* 遍历需要被复制的参数
            if k.startswith(prefix_key): #* 判断当前参数是否需要继承teacher model
                flag = 1
                k_orig = k_orig.replace(prefix_key, connect_info[prefix_key]) #* 替换前缀, str.replace(old, new)         
                break

        if flag == 1:
            print(f"** forced COPY {k_orig} -> {k}")
        else:
            print(f"normal COPY {k_orig} -> {k}")
        unet_stu.state_dict()[k].copy_(unet_tea.state_dict()[k_orig]) #* copy_(): 将一个张量的值复制到另一个张量中

    return unet_stu

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    # from smilelogging import argparser as parser  #* ==> Replace above with this line
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="/home/wanghesong/.cache/huggingface/hub/models--stabilityai--stable-diffusion-2-1",
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sd-model-finetuned",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument("--use_ema", action="store_true", help="Whether to use EMA model.")
    parser.add_argument(
        "--non_ema_revision",
        type=str,
        default=None,
        required=False,
        help=(
            "Revision of pretrained non-ema model identifier. Must be a branch, tag or git identifier of the local or"
            " remote repository specified with --pretrained_model_name_or_path."
        ),
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=(
            "Max number of checkpoints to store. Passed as `total_limit` to the `Accelerator` `ProjectConfiguration`."
            " See Accelerator::save_state https://huggingface.co/docs/accelerate/package_reference/accelerator#accelerate.Accelerator.save_state"
            " for more docs"
        ),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )

    parser.add_argument("--unet_config_path", type=str, default="./src/unet_config")     
    parser.add_argument("--unet_config_name", type=str, default="bk_small", choices=["bk_base", "bk_small", "bk_tiny"])   
    parser.add_argument("--lambda_sd", type=float, default=1.0, help="weighting for the denoising task loss")  
    parser.add_argument("--lambda_kd_output", type=float, default=1.0, help="weighting for output KD loss")  
    parser.add_argument("--lambda_kd_feat", type=float, default=1.0, help="weighting for feature KD loss")  
    parser.add_argument("--valid_prompt", type=str, default="a golden vase with different flowers")
    parser.add_argument("--valid_steps", type=int, default=500)
    parser.add_argument("--num_valid_images", type=int, default=2)
    parser.add_argument("--use_copy_weight_from_teacher", action="store_true", help="Whether to initialize unet student with teacher's weights",)

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    # default to using the same revision for the non-ema model if not specified
    if args.non_ema_revision is None:
        args.non_ema_revision = args.revision

    return args
# endregion-out_of_main

def main():
    args = parse_args()
    # logger = Logger(args, overwrite_print=True) 

    #* 处理过时的配置, 根本不执行
    if args.non_ema_revision is not None:
        deprecate(
            "non_ema_revision!=None", #* condition: 表达弃用条件的字符串
            "0.15.0", #* version: 弃用生效的版本
            message=( #* message: 弃用警告的信息
                "Downloading 'non_ema' weights from revision branches of the Hub is deprecated. Please make sure to"
                " use `--variant=non_ema` instead."
            ),
        )
    logging_dir = os.path.join(args.output_dir, args.logging_dir)  #* ./results/xx; logs, 好像并不存在

    #* 创建一个ProjCfg类，设置最大ckpt数量，多了会删除最前的
    #* 与Accelerator配合使用：在后续代码中，这个配置对象将传递给 Accelerator，用于管理训练过程中的检查点保存和删除
    accelerator_project_config = ProjectConfiguration(total_limit=args.checkpoints_total_limit)

    accelerator = Accelerator( #* Acc类：提供了简化和加速分布式训练和混合精度训练的功能
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,  #* fp16: 计算权重更新用fp32，其他用fp16
        log_with=args.report_to,  #* 'all': 日志输出到哪个平台，e.g. wandb, tensorboard, etc.
        logging_dir=logging_dir,
        project_config=accelerator_project_config,
    )

    # Add custom csv logger and validation image folder
    val_img_dir = os.path.join(args.output_dir, 'val_img')  #* './results/xx/val_img'
    os.makedirs(val_img_dir, exist_ok=True)

    csv_log_path = os.path.join(args.output_dir, 'log_loss.csv')  #* './results/xx/log_loss.csv'
    if not os.path.exists(csv_log_path): #* 如果是第一次创建，写入表头
        with open(csv_log_path, 'w') as logfile:
            logwriter = csv.writer(logfile, delimiter=',')
            logwriter.writerow(['epoch', 'step', 'global_step',
                                'loss_total', 'loss_sd', 'loss_kd_output', 'loss_kd_feat',
                                'lr', 'lamb_sd', 'lamb_kd_output', 'lamb_kd_feat'])

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(  #* 配置日志输出的format（日志系统的基本设置
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,  #* 只记录INFO级别及以上
    )
    logger.info(accelerator.state, main_process_only=False) #* 记录有关分布式训练的信息
    #* 根据当前进程是否为本地主进程（local main process），设置不同库的日志详细程度
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else: #* 如果不是本地主进程，只有error及以上才会被汇报
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)  #* 一键设置 Python、NumPy 和 PyTorch 的随机数种子

    # Handle the repository creation
    #* 在分布式训练环境中，仅在主进程中创建输出目录; 避免多个进程同时创建同一目录，导致冲突或冗余操作
    if accelerator.is_main_process and (args.output_dir is not None): #* 主进程 & 指定了输出目录
        os.makedirs(args.output_dir, exist_ok=True)

    # Load scheduler, tokenizer and models.
    #* 加载模型预设好的部分，revision是指定模型的版本或修订版，这里revision = None
    #* revision可以是版本，分支，commit id
    noise_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision
    )
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision
    )

    # Define teacher and student
    unet_teacher = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.non_ema_revision  #* None
    )

    #* load_config(path-文件夹，subfolder-子文件夹-可以没有): 把配置文件保存到path里面
    config_student = UNet2DConditionModel.load_config(args.unet_config_path, subfolder=args.unet_config_name)
    #* 根据配置创建一个 UNet2DConditionModel 模型实例
    unet = UNet2DConditionModel.from_config(config_student, revision=args.non_ema_revision)

    # Copy weights from teacher to student
    if args.use_copy_weight_from_teacher: #* 使unet继承了unet_teacher的权重，nothing more
        copy_weight_from_teacher(unet, unet_teacher, args.unet_config_name)
   

    # Freeze student's vae and text_encoder and teacher's unet
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet_teacher.requires_grad_(False)

    # Create EMA for the unet.
    if args.use_ema:
        #* 和前面的unet相同，除了revision（revision也是None，先不用管了
        ema_unet = UNet2DConditionModel.from_config(config_student, revision=args.revision) 
        #* 将unet模型包装到EMAModel中，并指定模型的类和配置
        #** 作用：将普通的unet模型变为一个带有ema机制的模型
        ema_unet = EMAModel(ema_unet.parameters(), model_cls=UNet2DConditionModel, model_config=ema_unet.config)

    #* 如果可以的话，使用xFormer加速运算
    #** xFormers 是一个库，专注于优化Transformer中的注意力计算，更加内存和计算高效（这个库挺大的
    if args.enable_xformers_memory_efficient_attention:  #* 这里默认是False, 没有使用
        if is_xformers_available():
            import xformers

            #* version.parser(): 用于将版本号字符串解析成一个可以进行比较的对象
            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        #* 在保存模型状态之前执行，确保 EMA 模型和普通模型的权重都被正确保存
        def save_model_hook(models, weights, output_dir):
            if args.use_ema: #* 保存ema模型
                #* save_pretrained(path): 生成两个文件，config.json-模型配置，pytorch_model.bin-模型权重
                ema_unet.save_pretrained(os.path.join(output_dir, "unet_ema"))

            for i, model in enumerate(models): #! 保存普通模型, 目前不太清楚是怎么运作的
                model.save_pretrained(os.path.join(output_dir, "unet"))

                # make sure to pop weight so that corresponding model is not saved again
                weights.pop()

        #* 在加载模型状态之前执行，确保 EMA 模型和普通模型的权重都被正确加载
        def load_model_hook(models, input_dir):
            if args.use_ema:
                load_model = EMAModel.from_pretrained(os.path.join(input_dir, "unet_ema"), UNet2DConditionModel)
                ema_unet.load_state_dict(load_model.state_dict())
                ema_unet.to(accelerator.device)
                del load_model

            for i in range(len(models)):  #! 暂时也不太清楚是怎么运作的
                # pop models so that they are not loaded again
                model = models.pop()

                # load diffusers style into model
                #* register_to_config(): 用于将特定的模型属性注册到模型的配置中, 确保模型的配置与当前实例的属性保持一致
                #* load_state_dict(): 用于将加载的权重应用到当前模型上
                load_model = UNet2DConditionModel.from_pretrained(input_dir, subfolder="unet")
                model.register_to_config(**load_model.config)
                model.load_state_dict(load_model.state_dict())
                del load_model

        #* 注册到accelerator对象上
        accelerator.register_save_state_pre_hook(save_model_hook)  #* 调用accelerator.save_state()时，首先执行这个hook
        accelerator.register_load_state_pre_hook(load_model_hook)  #* 调用accelerator.load_state()时，首先执行这个hook

    if args.gradient_checkpointing: #* 一种用于减少训练模型使用显存的技术, 默认为True
        #** 减少显存占用，增加计算时间 & 开销
        unet.enable_gradient_checkpointing()

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    #* 根据配置参数启用 Tensor-Float 32 计算模式以加速训练，并根据训练配置动态调整学习率
    #** 比FP32快，但精度更低
    if args.allow_tf32:  #* 默认为False
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr: #* 是否scale学习率以适应多卡训练，默认为Flase
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Initialize the optimizer
    if args.use_8bit_adam: #* 根据选择是否使用8bit优化器, 默认为False
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW  #* 默认优化器

    optimizer = optimizer_cls(
        unet.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),  #* Adam优化器的两个超参数，用来控制一阶和二阶动量的计算
        weight_decay=args.adam_weight_decay, #* 权重衰减，用于防止过拟合: 更新权重时对权重施加衰减(L2正则化), 减少权重的大小
        eps=args.adam_epsilon,  #* 避免分母为0
    )

    # Get the datasets. As the amount of data grows, the time taken by load_dataset also increases.
    print("*** load dataset: start")
    t0 = time.time()
    dataset = load_dataset("imagefolder", data_dir=args.train_data_dir, split="train")
    #*返回值:    #* Dataset({ features: ['image', 'text'],
                #*           num_rows: 10274 })
    print(f"*** load dataset: end --- {time.time()-t0} sec")
    
    # Preprocessing the datasets.
    column_names = dataset.column_names #* ['image', 'text']
    image_column = column_names[0] #* 'iamge'
    caption_column = column_names[1] #* 'text'

    # Preprocessing the datasets.
    # We need to tokenize input captions and transform the images.
    #* 输入example，输出inputs的id
    #** 把一个batch的caption提取出来，然后用tokenizer处理，返回
    def tokenize_captions(examples, is_train=True):  
        captions = []
        for caption in examples[caption_column]: #* 遍历所有的caption
            if isinstance(caption, str): #* 单个caption
                captions.append(caption)
            elif isinstance(caption, (list, np.ndarray)): #* 多个caption随机选取一个
                # take a random caption if there are multiple
                #* 如果在训练随机选一个，否则选第一个
                captions.append(random.choice(caption) if is_train else caption[0]) 
            else:
                raise ValueError(
                    f"Caption column `{caption_column}` should contain either strings or lists of strings."
                )
            #* captions: [bsz, ]
        inputs = tokenizer(
            captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        #* inputs是个字典, dict_keys(['input_ids', 'attention_mask'])
        return inputs.input_ids #* 返回输入的id，是训练/推理时的输入

    # Preprocessing the datasets.
    train_transforms = transforms.Compose(
        [
            transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(args.resolution) if args.center_crop else transforms.RandomCrop(args.resolution),
            transforms.RandomHorizontalFlip() if args.random_flip else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    def preprocess_train(examples):  
        #* 会在数据加载的时候被调用，to be specific, enumerate(train_dataloader)的时候
        #* 每次处理一个batch的数据
        images = [image.convert("RGB") for image in examples[image_column]] #* 将图像转为RGB格式
        examples["pixel_values"] = [train_transforms(image) for image in images] #* for image
        examples["input_ids"] = tokenize_captions(examples) #* for text
        return examples  #* 字典: dict_keys(['image', 'text', 'pixel_values', 'input_ids'])

    with accelerator.main_process_first(): #* 确保数据预处理操作在主进程中首先完成，避免分布式训练中的重复操作
        if args.max_train_samples is not None:  #* 默认是None
            #!* 选择前max_train_samples个样本用于调试
            dataset = dataset.shuffle(seed=args.seed).select(range(args.max_train_samples))
        # Set the training transforms #* 在数据加载时，数据集中的每个样本都会被传递到preprocess_train函数中进行预处理
        train_dataset = dataset.with_transform(preprocess_train)  #* with_trans会返回一个新的对象

    def collate_fn(examples):
        #* 将多个样本合成一个批次并返回, 在这里去掉了没有用的两个keys
        #* 由于是dataloader用的，所以也是按批次处理
        #* examples：[2, ], 内含两个4-key字典
        pixel_values = torch.stack([example["pixel_values"] for example in examples])  #* [bsz, 3, 768, 768]
        #* 将张量的内存格式设置为连续格式，以提高后续操作的效率; 同时转为float
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float() 
        input_ids = torch.stack([example["input_ids"] for example in examples])  #* [bsz, 77(tokenizer.model_max_length)]
        return {"pixel_values": pixel_values, "input_ids": input_ids}

    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, #* 包含了 .with_transform(preprocess_train)
        shuffle=True,
        collate_fn=collate_fn, #* 自定义的 collate 函数，用于将一批样本组合成一个 batch
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers, #* 用于数据加载的子进程数量
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False #* 指示是否覆盖了最大训练步骤数，表明最大训练步骤数是用户手动设置的，还是根据其他参数自动计算得出的
    #? len(dataloader) = len(dataset) / args.batch_size, 也就是批次数量
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps) #* 每个epoch中的模型更新的次数
    if args.max_train_steps is None: #* 如果没有指定max training steps，那么在这里指定
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(  #* HF的内置函数
        args.lr_scheduler, #* 学习率调度器的类型，例如 "linear"、"cosine" 等
        optimizer=optimizer,
        #* 定义了在训练的初期有多少步会逐步增加学习率，直到达到设定的最大学习率
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        #* 总训练步骤数
        #** 初步来看，nts对应前向传播的次数，mts对应反向传播的次数（i.e. 模型更新次数
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )
    
    # Prepare everything with our `accelerator`.
    #* accelerator.prepare(): 自动处理模型和数据在多个GPU上的分布和同步
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )

    if args.use_ema:
        ema_unet.to(accelerator.device)  #* 只追踪模型的变化，创建一个平滑版的模型，不直接参与forward和backward

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16": #* 同样占用16位，但具有更大的动态范围，适用于更广泛的场景
        weight_dtype = torch.bfloat16

    # Move student's text_encode and vae and teacher's unet to gpu and cast to weight_dtype
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    unet_teacher.to(accelerator.device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)  #* 反向传播的次数
    #* 一个是用epoch推training step，一个是用step推epoch，最终训练要用epoch来实现
    if overrode_max_train_steps:  #* epoch -> step
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs  #* step -> epoch
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)  

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:  #* tracker通常会与外部工具集成，如：wandb, etc.
        accelerator.init_trackers("text2image-fine-tune", config=vars(args))  #* vars: 将args转换成字典形式

    # Train!
    #* 总批次大小: 每个设备上的batch size * 设备数 * 梯度累积
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:  #! 没有细看过
        #* get checkpoint
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        #* 输出提示信息
        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (num_update_steps_per_epoch * args.gradient_accumulation_steps)

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    # Add hook for feature KD
    acts_tea = {} #* 用于存储教师模型中间特征的字典
    acts_stu = {} #* 用于存储学生模型中间特征的字典
    if args.unet_config_name in ["bk_base", "bk_small"]: 
        #* 会被用来提取特征的层
        mapping_layers = ['up_blocks.0', 'up_blocks.1', 'up_blocks.2', 'up_blocks.3',
                        'down_blocks.0', 'down_blocks.1', 'down_blocks.2', 'down_blocks.3']    
        mapping_layers_tea = copy.deepcopy(mapping_layers)
        mapping_layers_stu = copy.deepcopy(mapping_layers)

    elif args.unet_config_name in ["bk_tiny"]:
        mapping_layers_tea = ['down_blocks.0', 'down_blocks.1', 'down_blocks.2.attentions.1.proj_out',
                                'up_blocks.1', 'up_blocks.2', 'up_blocks.3']    
        mapping_layers_stu = ['down_blocks.0', 'down_blocks.1', 'down_blocks.2.attentions.0.proj_out',
                                'up_blocks.0', 'up_blocks.1', 'up_blocks.2']  

    #* 处理多gpu环境的mapping名称
    if torch.cuda.device_count() > 1:
        print(f"use multi-gpu: # gpus {torch.cuda.device_count()}")
        # revise the hooked feature names for student (to consider ddp wrapper)
        #* 多gpu训练的情况下，子模块会被包裹在一个叫module的外层模块中
        for i, m_stu in enumerate(mapping_layers_stu):
            mapping_layers_stu[i] = 'module.'+m_stu

    add_hook(unet_teacher, acts_tea, mapping_layers_tea) #! 这两行没有特别看懂
    add_hook(unet, acts_stu, mapping_layers_stu)

    # get wandb_tracker (if it exists)
    wandb_tracker = accelerator.get_tracker("wandb")

    for epoch in range(first_epoch, args.num_train_epochs):

        unet.train()

        train_loss = 0.0
        train_loss_sd = 0.0
        train_loss_kd_output = 0.0
        train_loss_kd_feat = 0.0

        for step, batch in enumerate(train_dataloader):
            #* batch.keys() = dict_keys(['pixel_values', 'input_ids'])
            #* batch['pixel_values'].shape = [bsz, 3, 768, 768]
            #* batch['input_ids'].shape = [bsz, 77], 77是tokenizer.model_max_length，超参
            # Skip steps until we reach the resumed step
            if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % args.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue

            with accelerator.accumulate(unet): #* grad_accu在acc初始化的时候定义
                # Convert images to latent space
                #* .latent_dist.sample()：这一步从潜在分布中采样得到潜在表示
                latents = vae.encode(batch["pixel_values"].to(weight_dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0] #* batch size
                # Sample a random timestep for each image
                #* 生成bsz个0到x的整数
                timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long() #* 类型转换为long

                # Add noise to the latents according to the noise magnitude at eachdc timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get the text embedding for conditioning
                encoder_hidden_states = text_encoder(batch["input_ids"])[0]

                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                # Predict the noise residual and compute loss
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
                loss_sd = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                # Predict output-KD loss
                model_pred_teacher = unet_teacher(noisy_latents, timesteps, encoder_hidden_states).sample
                loss_kd_output = F.mse_loss(model_pred.float(), model_pred_teacher.float(), reduction="mean")

                # Predict feature-KD loss
                losses_kd_feat = []
                for (m_tea, m_stu) in zip(mapping_layers_tea, mapping_layers_stu): 
                    a_tea = acts_tea[m_tea] #* activation-激活值，这里是特征
                    a_stu = acts_stu[m_stu]

                    if type(a_tea) is tuple: a_tea = a_tea[0]                        
                    if type(a_stu) is tuple: a_stu = a_stu[0]

                    tmp = F.mse_loss(a_stu.float(), a_tea.detach().float(), reduction="mean")
                    losses_kd_feat.append(tmp)
                loss_kd_feat = sum(losses_kd_feat)

                # Compute the final loss
                loss = args.lambda_sd * loss_sd + args.lambda_kd_output * loss_kd_output + args.lambda_kd_feat * loss_kd_feat

                # Gather the losses across all processes for logging (if we use distributed training).
                #* 这段代码的目的是在分布式训练环境中聚合和计算平均损失
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                avg_loss_sd = accelerator.gather(loss_sd.repeat(args.train_batch_size)).mean()
                train_loss_sd += avg_loss_sd.item() / args.gradient_accumulation_steps

                avg_loss_kd_output = accelerator.gather(loss_kd_output.repeat(args.train_batch_size)).mean()
                train_loss_kd_output += avg_loss_kd_output.item() / args.gradient_accumulation_steps

                avg_loss_kd_feat = accelerator.gather(loss_kd_feat.repeat(args.train_batch_size)).mean()
                train_loss_kd_feat += avg_loss_kd_feat.item() / args.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients: #* 梯度裁剪
                    accelerator.clip_grad_norm_(unet.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                if args.use_ema:
                    ema_unet.step(unet.parameters())
                progress_bar.update(1)
                global_step += 1
                accelerator.log(
                    {
                        "train_loss": train_loss, 
                        "train_loss_sd": train_loss_sd,
                        "train_loss_kd_output": train_loss_kd_output,
                        "train_loss_kd_feat": train_loss_kd_feat,
                        "lr": lr_scheduler.get_last_lr()[0]
                    }, 
                    step=global_step
                )

                if accelerator.is_main_process:
                    with open(csv_log_path, 'a') as logfile:
                        logwriter = csv.writer(logfile, delimiter=',')
                        logwriter.writerow([epoch, step, global_step,
                                            train_loss, train_loss_sd, train_loss_kd_output, train_loss_kd_feat,
                                            lr_scheduler.get_last_lr()[0],
                                            args.lambda_sd, args.lambda_kd_output, args.lambda_kd_feat])

                train_loss = 0.0
                train_loss_sd = 0.0
                train_loss_kd_output = 0.0
                train_loss_kd_feat = 0.0

                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

            logs = {"step_loss": loss.detach().item(),
                    "sd_loss": loss_sd.detach().item(),
                    "kd_output_loss": loss_kd_output.detach().item(),
                    "kd_feat_loss": loss_kd_feat.detach().item(),
                    "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs) #* 通过 set_postfix 方法将 logs 字典中的信息添加到训练进度条的后缀中

            # save validation images
            if (args.valid_prompt is not None) and (step % args.valid_steps == 0) and accelerator.is_main_process:
                logger.info(
                    f"Running validation... \n Generating {args.num_valid_images} images with prompt:"
                    f" {args.valid_prompt}."
                )
                # create pipeline
                pipeline = StableDiffusionPipeline.from_pretrained(
                    args.pretrained_model_name_or_path,
                    safety_checker=None,
                    revision=args.revision,
                )
                pipeline = pipeline.to(accelerator.device)
                pipeline.set_progress_bar_config(disable=True)
                #* 创建一个随机种子生成器，用于确保生成图像的随机性
                generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)

                #* 生成teacher image，只会执行一次
                if not os.path.exists(os.path.join(val_img_dir, "teacher_0.png")):
                    for kk in range(args.num_valid_images):
                        image = pipeline(args.valid_prompt, num_inference_steps=25, generator=generator).images[0]
                        tmp_name = os.path.join(val_img_dir, f"teacher_{kk}.png")
                        image.save(tmp_name)

                # set `keep_fp32_wrapper` to True because we do not want to remove
                # mixed precision hooks while we are still training
                pipeline.unet = accelerator.unwrap_model(unet, keep_fp32_wrapper=True).to(accelerator.device)
              
                #* 生成student image，会按周期执行
                for kk in range(args.num_valid_images):
                    image = pipeline(args.valid_prompt, num_inference_steps=25, generator=generator).images[0]
                    tmp_name = os.path.join(val_img_dir, f"gstep{global_step}_epoch{epoch}_step{step}_{kk}.png")
                    print(tmp_name)
                    image.save(tmp_name)

                del pipeline #* 删除 pipeline 对象，释放内存。
                torch.cuda.empty_cache() #* 清理未使用的 GPU 内存，防止内存溢出。

            if global_step >= args.max_train_steps:
                break

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = accelerator.unwrap_model(unet)
        if args.use_ema:
            ema_unet.copy_to(unet.parameters())

        pipeline = StableDiffusionPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            text_encoder=text_encoder,
            vae=vae,
            unet=unet,
            revision=args.revision,
        )
        pipeline.save_pretrained(args.output_dir)

    accelerator.end_training()

if __name__ == "__main__":
    main()
