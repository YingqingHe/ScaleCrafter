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
# limitations under the License.
"""Fine-tuning script for Stable Diffusion for text2image with support for LoRA."""

import argparse
import logging
import scipy
import math
import glob
import os
import random
from pathlib import Path
from typing import Optional

import datasets
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.utils.checkpoint

import transformers
from PIL import Image
from omegaconf import OmegaConf
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset
from huggingface_hub import HfFolder, Repository, create_repo, whoami
from packaging import version
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer, CLIPTextModelWithProjection

import diffusers
from diffusers import \
    AutoencoderKL, DDPMScheduler, EulerDiscreteScheduler, DiffusionPipeline, UNet2DConditionModel, DDIMScheduler, \
    StableDiffusionXLPipeline
from diffusers.loaders import AttnProcsLayers
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
# check_min_version("0.15.0.dev0")

logger = get_logger(__name__, log_level="INFO")


def save_model_card(repo_name, images=None, base_model=str, dataset_name=str, repo_folder=None):
    img_str = ""
    for i, image in enumerate(images):
        image.save(os.path.join(repo_folder, f"image_{i}.png"))
        img_str += f"![img_{i}](./image_{i}.png)\n"

    yaml = f"""
---
license: creativeml-openrail-m
base_model: {base_model}
tags:
- stable-diffusion
- stable-diffusion-diffusers
- text-to-image
- diffusers
- lora
inference: true
---
    """
    model_card = f"""
# LoRA text2image fine-tuning - {repo_name}
These are LoRA adaption weights for {base_model}. The weights were fine-tuned on the {dataset_name} dataset. You can find some example images in the following. \n
{img_str}
"""
    with open(os.path.join(repo_folder, "README.md"), "w") as f:
        f.write(yaml + model_card)


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="stabilityai/stable-diffusion-xl-base-1.0",
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
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
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
        "--image_column", type=str, default="image", help="The column of the dataset containing an image."
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default="text",
        help="The column of the dataset containing a caption or a list of captions.",
    )
    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument("--gpu_num", type=int, default=None)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument(
        "--validation_prompt", type=str,
        # default="A professional photograph of a tiger walking in a forest",
        # default="a professional photograph of an astronaut riding a horse",
        # default="a professional photograph of optimus prime",
        # default="An emtpy white image",
        # default="Close up of a pair of vibrant koi fish swim upstream, surmounting a waterfall, oil painting style.",
        # default="Astronaut on Mars During sunset",
        # default="/apdcephfs/share_1290939/shaoshuyang/benchmarks/addm/res/teaser_prompts.txt",
        # default="Portrait photo of an anthropomorphic farmer cat holding a shovel in a garden",
        # default="Pink llama with a fuzzy hairdo, positive energy, happy, octane, substance, art history museum 8k",
        # default="Portrait of robot Terminator, cybord, evil, in dynamics, highly detailed, "
        #         "packed with hidden details, style, high dynamic range, hyper realistic,
        #         realistic attention to detail, highly detailed",
        # default="A pair of glowing jellyfish floating through a foggy glowing mushroom forest at twilight.",
        # default="Pink llama with a fuzzy hairdo standing in front of a cafe shop,
        # positive energy, happy, octane, substance, art history museum 8k",
        # default="Miniature house with plants in the potted area, hyper realism, dramatic ambient lighting, high detail",
        # default="A cherry blossom tree in full bloom amidst an arctic tundra showering petals on a polar bear.jpg",
        # default="/apdcephfs_cq3/share_1290939/yingqinghe/datasets/laion2B-3w-morethan512/captions",
        # default="/apdcephfs/share_1290939/shaoshuyang/t2i/addm/captions/test",
        default="/apdcephfs/share_1290939/shaoshuyang/t2i/addm/captions/apdcephfs_cq3"
                "/share_1290939/yingqinghe/datasets/laion2B-1w-morethan512/captions",
        # default="",
        help="A prompt that is sampled during training for inference."
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=1,
        help="Number of images that should be generated during validation with `validation_prompt`.",
    )
    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=1,
        help=(
            "Run fine-tuning validation every X epochs. The validation process consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`."
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
        default="./outputs/text2image",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument("--seed", type=int, default=23, help="A seed for reproducible training.")
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
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="/apdcephfs/share_1290939/shaoshuyang/t2i/addm/",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default='fp16',
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
    parser.add_argument("--noise_offset", type=float, default=0, help="The scale of noise offset.")

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    # Sanity checks
    # if args.dataset_name is None and args.train_data_dir is None:
    #     raise ValueError("Need either a dataset name or a training folder.")

    return args


from torch import Tensor


def disable_downsample_processor(self):
    def forward(hidden_states):
        assert hidden_states.shape[1] == self.channels
        if self.use_conv and self.padding == 0:
            pad = (0, 1, 0, 1)
            hidden_states = F.pad(hidden_states, pad, mode="constant", value=0)

        assert hidden_states.shape[1] == self.channels
        hidden_states = F.interpolate(hidden_states, scale_factor=2, mode='bilinear')
        hidden_states = self.conv(hidden_states)

        return hidden_states

    return forward


def disable_upsample_processor(self):
    def forward(hidden_states, output_size=None):
        assert hidden_states.shape[1] == self.channels

        if self.use_conv_transpose:
            return self.conv(hidden_states)

        # Cast to float32 to as 'upsample_nearest2d_out_frame' op does not support bfloat16
        # TODO(Suraj): Remove this cast once the issue is fixed in PyTorch
        # https://github.com/pytorch/pytorch/issues/86679
        dtype = hidden_states.dtype
        if dtype == torch.bfloat16:
            hidden_states = hidden_states.to(torch.float32)

        # upsample_nearest_nhwc fails with large batch sizes. see https://github.com/huggingface/diffusers/issues/984
        if hidden_states.shape[0] >= 64:
            hidden_states = hidden_states.contiguous()

        # if `output_size` is passed we force the interpolation output
        # size and do not make use of `scale_factor=2`
        # if output_size is None:
        #     hidden_states = F.interpolate(hidden_states, scale_factor=2.0, mode="nearest")
        # else:
        #     hidden_states = F.interpolate(hidden_states, size=output_size, mode="nearest")

        # If the input is bfloat16, we cast back to bfloat16
        if dtype == torch.bfloat16:
            hidden_states = hidden_states.to(dtype)

        # TODO(Suraj, Patrick) - clean up after weight dicts are correctly renamed
        if self.use_conv:
            if self.name == "conv":
                hidden_states = self.conv(hidden_states)
            else:
                hidden_states = self.Conv2d_0(hidden_states)

        return hidden_states

    return forward


def dilate_conv_processor(self, pf_factor=1.0, mode='bilinear'):
    dilation = math.ceil(pf_factor)
    factor = float(dilation / pf_factor)

    def forward(input: Tensor) -> Tensor:
        if counter < 10:
            ori_dilation, ori_padding = self.dilation, self.padding
            self.dilation, self.padding = dilation, dilation
            ori_size, new_size = (
                (int(input.shape[-2] / self.stride[0]), int(input.shape[-1] / self.stride[1])),
                (round(input.shape[-2] * factor), round(input.shape[-1] * factor))
            )
            input = F.interpolate(input, size=new_size, mode=mode)
            input = self._conv_forward(input, self.weight, self.bias)
            self.dilation, self.padding = ori_dilation, ori_padding
            result = F.interpolate(input, size=ori_size, mode=mode)
            return result
        else:
            return self._conv_forward(input, self.weight, self.bias)

    return forward


class DilateConvProcessor:
    def __init__(self, module, pf_factor=1.0, mode='bilinear', activate=True):
        self.dilation = math.ceil(pf_factor)
        self.factor = float(self.dilation / pf_factor)
        self.module = module
        self.mode = mode
        self.activate = activate

    def __call__(self, input: Tensor, **kwargs) -> Tensor:
        if self.activate:
            # def tensor_erode(bin_img, ksize=5):
            #     # padding for keeping size
            #     B, C, H, W = bin_img.shape
            #     pad = (ksize - 1) // 2
            #     bin_img = F.pad(bin_img, [pad, pad, pad, pad], mode='constant', value=0)
            #
            #     patches = bin_img.unfold(dimension=2, size=ksize, step=1)
            #     patches = patches.unfold(dimension=3, size=ksize, step=1)
            #     # B x C x H x W x k x k
            #
            #     eroded, _ = patches.reshape(B, C, H, W, -1).min(dim=-1)
            #     return eroded
            #
            # kernel_size = 7
            # kernel1 = create_low_pass_filter(
            #     kernel_size=kernel_size,
            #     sigma=1,
            #     device=input.device,
            #     dtype=input.dtype,
            # )
            # kernel2 = create_low_pass_filter(
            #     kernel_size=kernel_size,
            #     sigma=5,
            #     device=input.device,
            #     dtype=input.dtype,
            # )
            # kernel1 = kernel1.repeat(input.shape[1], 1, 1, 1)
            # kernel2 = kernel2.repeat(input.shape[1], 1, 1, 1)
            #
            # k1_output = F.conv2d(
            #     input, kernel1, padding=kernel_size // 2, groups=input.shape[1])
            # k2_output = F.conv2d(
            #     input, kernel2, padding=kernel_size // 2, groups=input.shape[1]
            # )
            # edge = torch.abs(k1_output - k2_output).mean(dim=1, keepdim=True)
            # std, mean = torch.std_mean(edge.view((edge.shape[0], edge.shape[1], -1)), dim=-1, keepdim=True)
            # threshold = (mean + 1.5 * std).unsqueeze(-1)
            # threshold, _ = torch.mean(
            #     edge.view((edge.shape[0], edge.shape[1], -1)), dim=-1, keepdim=True)
            # threshold = threshold.unsqueeze(-1) / 10

            # vanilla_result = self.module._conv_forward(input, self.module.weight, self.module.bias)

            ori_dilation, ori_padding = self.module.dilation, self.module.padding
            inflation_kernel_size = (self.module.weight.shape[-1] - 3) // 2
            self.module.dilation, self.module.padding = self.dilation, (
                self.dilation * (1 + inflation_kernel_size), self.dilation * (1 + inflation_kernel_size)
            )
            ori_size, new_size = (
                (int(input.shape[-2] / self.module.stride[0]), int(input.shape[-1] / self.module.stride[1])),
                (round(input.shape[-2] * self.factor), round(input.shape[-1] * self.factor))
            )
            input = F.interpolate(input, size=new_size, mode=self.mode)
            input = self.module._conv_forward(input, self.module.weight, self.module.bias)
            self.module.dilation, self.module.padding = ori_dilation, ori_padding
            result = F.interpolate(input, size=ori_size, mode=self.mode)
            # result = tensor_erode(result)
            # print(edge.shape)
            # print(threshold.shape)
            # print(vanilla_result.shape)
            # print(result.shape)
            # print(torch.gt(edge, threshold).shape)
            # result = torch.where(torch.repeat_interleave(
            #     torch.gt(edge, threshold), result.shape[1], dim=1), vanilla_result, result)
            return result
        else:
            return self.module._conv_forward(input, self.module.weight, self.module.bias)


def create_gaussian_kernel(kernel_size, sigma):
    ax = np.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2. * sigma ** 2))
    return kernel / kernel.sum()


def create_low_pass_filter(kernel_size=7, sigma=1.0, device=None, dtype=None):
    gaussian_kernel = create_gaussian_kernel(kernel_size, sigma)
    kernel_tensor = torch.tensor(gaussian_kernel, device=device, dtype=dtype)
    kernel_tensor = kernel_tensor.view(1, 1, kernel_size, kernel_size)
    return kernel_tensor


class SmoothedDilateConvProcessor:
    def __init__(self, module, pf_factor=1.0, mode='bilinear', activate=True, log_snr=1.0):
        self.dilation = math.ceil(pf_factor)
        self.factor = float(self.dilation / pf_factor)
        self.module = module
        self.mode = mode
        self.activate = activate
        self.log_snr = np.clip(log_snr, -0.8, 1.0)

    def __call__(self, input: Tensor, **kwargs) -> Tensor:
        if self.activate:
            sigma_scale = (self.log_snr + 0.8) / 1.8
            kernel_size = 5
            kernel = create_low_pass_filter(
                kernel_size=kernel_size,
                sigma=0.01 * sigma_scale + 3 * (1 - sigma_scale),
                device=input.device,
                dtype=input.dtype,
            )
            kernel = kernel.repeat(input.shape[1], 1, 1, 1)
            lf_input = F.conv2d(input, kernel, padding=kernel_size // 2, groups=input.shape[1])
            hf_input = input - lf_input
            hf_output = self.module._conv_forward(hf_input, self.module.weight, self.module.bias)

            ori_dilation, ori_padding = self.module.dilation, self.module.padding
            self.module.dilation, self.module.padding = self.dilation, (self.dilation, self.dilation)
            ori_size, new_size = (
                (int(input.shape[-2] / self.module.stride[0]), int(input.shape[-1] / self.module.stride[1])),
                (round(input.shape[-2] * self.factor), round(input.shape[-1] * self.factor))
            )
            lf_input = F.interpolate(lf_input, size=new_size, mode=self.mode)
            lf_output = self.module._conv_forward(lf_input, self.module.weight, self.module.bias)
            self.module.dilation, self.module.padding = ori_dilation, ori_padding
            lf_output = F.interpolate(lf_output, size=ori_size, mode=self.mode)
            return hf_output + lf_output
        else:
            return self.module._conv_forward(input, self.module.weight, self.module.bias)


def dilate_conv_processor_for_visualization(self, pf_factor=1.0, mode='bilinear'):
    dilation = math.ceil(pf_factor)
    factor = float(dilation / pf_factor)

    def forward(input: Tensor) -> Tensor:
        ori_dilation, ori_padding = self.dilation, self.padding
        self.dilation, self.padding = dilation, dilation
        ori_size, new_size = (
            (int(input.shape[-2] / self.stride[0]), int(input.shape[-1] / self.stride[1])),
            (round(input.shape[-2] * factor), round(input.shape[-1] * factor))
        )
        input_dilate = F.interpolate(input, size=new_size, mode=mode)
        input_dilate = self._conv_forward(input_dilate, self.weight, self.bias)
        self.dilation, self.padding = ori_dilation, ori_padding
        input_dilate = F.interpolate(input_dilate, size=ori_size, mode=mode)

        input = self._conv_forward(input, self.weight, self.bias)
        loss_map = ((input_dilate - input) ** 2).mean(dim=1)[1]
        loss_map = (loss_map / loss_map.max()).squeeze(0)
        loss_map = Image.fromarray(np.array(loss_map.detach().cpu() * 255, dtype=np.uint8))
        global index
        loss_map.save(f'/apdcephfs/share_1290939/shaoshuyang/t2i/addm/loss_maps/{index}.png')
        index = index + 1
        return input

    return forward


def scale_conv_processor(self, pf_factor=1.0, mode=None):
    scale = pf_factor

    def forward(input: Tensor) -> Tensor:
        input = input / scale
        input = self._conv_forward(input, self.weight, self.bias)
        input = input * scale
        return input

    return forward


class ScaledAttnProcessor:
    r"""
    Default processor for performing attention-related computations.
    """

    def __init__(self, processor, test_res, train_res):
        self.processor = processor
        self.test_res = test_res
        self.train_res = train_res

    def __call__(
            self,
            attn,
            hidden_states,
            encoder_hidden_states=None,
            attention_mask=None,
            temb=None,
    ):
        input_ndim = hidden_states.ndim
        # print(f"cross attention: {not encoder_hidden_states is None}")
        if encoder_hidden_states is None:
            if input_ndim == 4:
                batch_size, channel, height, width = hidden_states.shape
                sequence_length = height * width
            else:
                batch_size, sequence_length, _ = hidden_states.shape

            # test_train_ratio = (self.test_res ** 2.0) / (self.train_res ** 2.0)
            test_train_ratio = float(self.test_res / self.train_res)
            train_sequence_length = sequence_length / test_train_ratio
            scale_factor = math.log(sequence_length, train_sequence_length) ** 0.5
        else:
            scale_factor = 1
        # print(f"scale factor: {scale_factor}")

        original_scale = attn.scale
        attn.scale = attn.scale * scale_factor
        hidden_states = self.processor(attn, hidden_states, encoder_hidden_states, attention_mask, temb)
        # hidden_states = super(ScaledAttnProcessor, self).__call__(
        #     attn, hidden_states, encoder_hidden_states, attention_mask, temb)
        attn.scale = original_scale
        return hidden_states


from diffusers.pipelines.stable_diffusion_xl import StableDiffusionXLPipelineOutput
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import rescale_noise_cfg
from diffusers.utils.torch_utils import randn_tensor
from diffusers.models.lora import LoRACompatibleConv
from einops import rearrange


def inflate_kernels(unet, inflate_conv_list, inflation_transform):
    for name, module in unet.named_modules():
        if name in inflate_conv_list:
            weight, bias = module.weight.detach(), module.bias.detach()
            (i, o, *_), kernel_size = (
                weight.shape, int(math.sqrt(inflation_transform.shape[0]))
            )
            transformed_weight = torch.einsum(
                "mn, ion -> iom", inflation_transform.to(dtype=weight.dtype), weight.view(i, o, -1))
            module = LoRACompatibleConv(
                o, i, (kernel_size, kernel_size),
                stride=module.stride, padding=module.padding, device=weight.device, dtype=weight.dtype
            )
            module.weight.detach().copy_(transformed_weight.view(i, o, kernel_size, kernel_size))
            module.bias.detach().copy_(bias)


def pipeline_processor(
        self,
        dilate_conv_list,
        dilate_conv_list2,
        any_res_cfg_tau=0,
        any_res_cfg_dilate=1,
        sdedit_tau=20,
        dilate_tau=0,
        dilate=1,
        dilate2=1,
        progressive=False,
):
    @torch.no_grad()
    def forward(
            prompt=None,
            prompt_2=None,
            height: Optional[int] = None,
            width: Optional[int] = None,
            num_inference_steps: int = 50,
            denoising_end: Optional[float] = None,
            guidance_scale: float = 5.0,
            negative_prompt=None,
            negative_prompt_2=None,
            num_images_per_prompt: Optional[int] = 1,
            eta: float = 1.0,
            generator=None,
            latents: Optional[torch.FloatTensor] = None,
            prompt_embeds: Optional[torch.FloatTensor] = None,
            negative_prompt_embeds: Optional[torch.FloatTensor] = None,
            pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
            negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
            output_type: Optional[str] = "pil",
            return_dict: bool = True,
            callback=None,
            callback_steps: int = 1,
            cross_attention_kwargs=None,
            guidance_rescale: float = 0.0,
            original_size=None,
            crops_coords_top_left=(0, 0),
            target_size=None,
    ):
        # 0. Default height and width to unet
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        original_size = original_size or (height, width)
        target_size = target_size or (height, width)

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            prompt_2,
            height,
            width,
            callback_steps,
            negative_prompt,
            negative_prompt_2,
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        )

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]
        device = self._execution_device

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        text_encoder_lora_scale = (
            cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None
        )
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            lora_scale=text_encoder_lora_scale,
        )

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. Prepare added time ids & embeddings
        add_text_embeds = pooled_prompt_embeds
        add_time_ids = self._get_add_time_ids(
            original_size, crops_coords_top_left, target_size, dtype=prompt_embeds.dtype
        )

        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            add_text_embeds = torch.cat([negative_pooled_prompt_embeds, add_text_embeds], dim=0)
            add_time_ids = torch.cat([add_time_ids, add_time_ids], dim=0)

        prompt_embeds = prompt_embeds.to(device)
        add_text_embeds = add_text_embeds.to(device)
        add_time_ids = add_time_ids.to(device).repeat(batch_size * num_images_per_prompt, 1)

        # 8. Denoising loop
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)

        # 7.1 Apply denoising_end
        if denoising_end is not None and type(denoising_end) == float and denoising_end > 0 and denoising_end < 1:
            discrete_timestep_cutoff = int(
                round(
                    self.scheduler.config.num_train_timesteps
                    - (denoising_end * self.scheduler.config.num_train_timesteps)
                )
            )
            num_inference_steps = len(list(filter(lambda ts: ts >= discrete_timestep_cutoff, timesteps)))
            timesteps = timesteps[:num_inference_steps]

        if sdedit_tau is not None:
            timesteps = timesteps[sdedit_tau:]
        with self.progress_bar(total=len(timesteps)) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                backup_forwards = dict()
                for name, module in self.unet.named_modules():
                    if dilate_tau > i:
                        if name in dilate_conv_list:
                            # print(name)
                            backup_forwards[name] = module.forward
                            print(f"list dilate: {max(math.ceil(dilate * ((dilate_tau - i) / dilate_tau)), 2)}")
                            module.forward = DilateConvProcessor(
                                module,
                                max(math.ceil(dilate * ((dilate_tau - i) / (dilate_tau - sdedit_tau))), 2)
                                if progressive else dilate,
                                mode='bilinear', activate=True
                            )

                    if dilate_tau > i:
                        if name in dilate_conv_list2:
                            # print(name)
                            backup_forwards[name] = module.forward
                            print(f"list2 dilate: {max(math.ceil(dilate2 * ((dilate_tau - i) / dilate_tau)), 2)}")
                            module.forward = DilateConvProcessor(
                                module,
                                max(math.ceil(dilate2 * ((dilate_tau - i) / (dilate_tau - sdedit_tau))), 2)
                                if progressive else dilate2,
                                mode='bilinear', activate=True
                            )
                            # log_snr = torch.sqrt(alphas_cumprod[t]) - torch.sqrt(one_minus_alphas_cumprod[t])
                            # print(name)
                            # backup_forwards[name] = module.forward
                            # module.forward = SmoothedDilateConvProcessor(
                            #     module, 4, mode='bilinear', activate=i < 80, log_snr=log_snr
                            # )

                # predict the noise residual
                added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs=cross_attention_kwargs,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                )[0]

                for name, module in self.unet.named_modules():
                    if name in backup_forwards.keys():
                        module.forward = backup_forwards[name]

                if any_res_cfg_tau > i:
                    backup_forwards = dict()
                    for name, module in self.unet.named_modules():
                        if name in dilate_conv_list:
                            # print(name)
                            backup_forwards[name] = module.forward
                            print(
                                f"ndcfg dilate: {max(math.ceil(any_res_cfg_dilate * ((any_res_cfg_tau - i) / any_res_cfg_tau)), 2)}")
                            module.forward = DilateConvProcessor(
                                module,
                                max(math.ceil(any_res_cfg_dilate * (
                                        (any_res_cfg_tau - i) / (any_res_cfg_tau - sdedit_tau))), 2
                                    )
                                if progressive else any_res_cfg_dilate,
                                mode='bilinear', activate=i < 80
                            )

                    # if i == 0:
                    #     latent_model_input = torch.cat([-latents] * 2) if do_classifier_free_guidance else -latents
                    #     latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                    # else:
                    #     latent_model_input = torch.cat(
                    #         [latents - 2 * beta_prod_t_prev * epsilon] * 2
                    #     ) if do_classifier_free_guidance else latents - 2 * beta_prod_t_prev * epsilon
                    #     latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                    # print(f"latent_model_input: {latent_model_input[..., :10]}")
                    noise_pred_vanilla = self.unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=prompt_embeds,
                        cross_attention_kwargs=cross_attention_kwargs,
                        added_cond_kwargs=added_cond_kwargs,
                        return_dict=False,
                    )[0]

                    for name, module in self.unet.named_modules():
                        if name in backup_forwards.keys():
                            module.forward = backup_forwards[name]

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    if any_res_cfg_tau > i:
                        noise_pred_vanilla, _ = noise_pred_vanilla.chunk(2)

                        # noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                        noise_pred = noise_pred_vanilla + guidance_scale * (noise_pred_text - noise_pred_uncond)
                    else:
                        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                if do_classifier_free_guidance and guidance_rescale > 0.0:
                    # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                    noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=guidance_rescale)

                # compute the previous noisy sample x_t -> x_t-1
                variance_noise = None
                results = self.scheduler.step(
                    noise_pred, t, latents, **extra_step_kwargs, variance_noise=variance_noise, return_dict=True)
                latents, ori_latents = results.prev_sample, results.pred_original_sample

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        # make sure the VAE is in float32 mode, as it overflows in float16
        if self.vae.dtype == torch.float16 and self.vae.config.force_upcast:
            self.upcast_vae()
            latents = latents.to(next(iter(self.vae.post_quant_conv.parameters())).dtype)

        if not output_type == "latent":
            image = self.vae.tiled_decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
        else:
            image = latents
            return StableDiffusionXLPipelineOutput(images=image)

        # apply watermark if available
        if self.watermark is not None:
            image = self.watermark.apply_watermark(image)

        image = self.image_processor.postprocess(image, output_type=output_type)

        # Offload last model to CPU
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.final_offload_hook.offload()

        if not return_dict:
            return (image,)

        return StableDiffusionXLPipelineOutput(images=image)

    return forward


def main(rank, gpu_num):
    args = parse_args()
    logging_dir = os.path.join(args.logging_dir)
    rank = rank if args.rank is None else args.rank
    gpu_num = gpu_num if args.gpu_num is None else args.gpu_num
    config = OmegaConf.load(args.config)
    print(f"Rank {rank}, saving results to {logging_dir}.")

    accelerator_project_config = ProjectConfiguration(
        total_limit=args.checkpoints_total_limit, logging_dir=logging_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Final inference
    # Load previous pipeline
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision, torch_dtype=weight_dtype
    )
    tokenizer_2 = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer_2", revision=args.revision, torch_dtype=weight_dtype
    )
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, torch_dtype=weight_dtype
    )
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder_2", revision=args.revision, torch_dtype=weight_dtype
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, torch_dtype=weight_dtype
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision, torch_dtype=weight_dtype
    )
    noise_scheduler = DDIMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    # noise_scheduler = EulerDiscreteScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")

    if config.use_attn_scale_factor:
        attn_procs = {}
        for name in unet.attn_processors.keys():
            attn_procs[name] = ScaledAttnProcessor(
                processor=unet.attn_processors[name],
                test_res=config.test_pixels,
                train_res=config.train_pixels
            )
        unet.set_attn_processor(attn_procs)

    pipeline = StableDiffusionXLPipeline(
        vae=vae,
        text_encoder=text_encoder,
        text_encoder_2=text_encoder_2,
        tokenizer=tokenizer,
        tokenizer_2=tokenizer_2,
        unet=unet,
        scheduler=noise_scheduler,
    )
    pipeline = pipeline.to(accelerator.device)

    # load attention processors
    # pipeline.unet.load_attn_procs(args.output_dir)
    # for name, module in unet.named_modules():
    #     if isinstance(module, nn.Conv2d):
    #         print(name)

    # run inference
    if config.dilate_conv_list is not None:
        with open(config.dilate_conv_list, 'r') as f:
            dilate_conv_list = f.readlines()
            dilate_conv_list = [name.strip() for name in dilate_conv_list]
    else:
        dilate_conv_list = list()
    if config.dilate_conv_list2 is not None:
        with open(config.dilate_conv_list2, 'r') as f:
            dilate_conv_list2 = f.readlines()
            dilate_conv_list2 = [name.strip() for name in dilate_conv_list2]
    else:
        dilate_conv_list2 = list()
    # dilate_conv_list2 = list()
    # transform = scipy.io.loadmat('/apdcephfs/share_1290939/shaoshuyang/benchmarks/addm/res/49R1000to1.mat')['R']
    # transform = torch.tensor(transform, device=accelerator.device)
    # inflate_kernels(unet, dilate_conv_list2, transform)
    unet.eval()

    total_num = 0
    print(f"Using prompt {args.validation_prompt}")
    if os.path.isfile(args.validation_prompt):
        with open(args.validation_prompt, 'r') as f:
            validation_prompt = f.readlines()
            validation_prompt = [line.strip() for line in validation_prompt]
        validation_prompt = validation_prompt[rank::gpu_num]
    elif os.path.isdir(args.validation_prompt):
        caption_files = sorted(glob.glob(os.path.join(args.validation_prompt, '*.txt')))
        # caption_files = caption_files[:10000]
        validation_prompt = caption_files[rank::gpu_num]

        finished_list = sorted(
            glob.glob(os.path.join(logging_dir, f'*_{rank}.txt')),
            key=lambda x: int(x.split('/')[-1].split('_')[0])
        )
        if len(finished_list) >= 2:
            with open(finished_list[-2], 'r') as f:
                last_caption = f.readline().strip()
            with open(validation_prompt[len(finished_list[:-1]) - 1], 'r') as f:
                assert last_caption == f.readline().strip()
            total_num = len(finished_list[:-1])
            validation_prompt = validation_prompt[len(finished_list[:-1]):]
        print(f"Evaluating on {len(validation_prompt)} prompts.")
    else:
        validation_prompt = [args.validation_prompt, ]

    inference_batch_size = config.inference_batch_size
    num_batches = math.ceil(len(validation_prompt) / inference_batch_size)
    # for i in range(10000):
    for i in range(num_batches):
        # for prompt in validation_prompt:
        prompts = validation_prompt[i * inference_batch_size:min(
            (i + 1) * inference_batch_size, len(validation_prompt))]
        output_prompts = list()
        for prompt in prompts:
            if os.path.isfile(prompt):
                with open(prompt, 'r') as f:
                    prompt = f.readline().strip()
                    output_prompts.append(prompt)
            else:
                output_prompts.append(prompt)

        for n in range(1):
            set_seed(61236123 + total_num)
            # generator = torch.Generator(device=accelerator.device).manual_seed(898716 + n)
            generator = None

            sdedit_scale = config.sdedit_scale if hasattr(config, 'sdedit_scale') else 1
            latents = torch.randn(
                (len(output_prompts), 4,
                 config.latent_height // sdedit_scale,
                 config.latent_width // sdedit_scale
                 ),
                device=accelerator.device,
                dtype=weight_dtype
            )
            results = pipeline(
                output_prompts,
                num_inference_steps=config.num_inference_steps,
                eta=0.0,
                generator=generator,
                latents=latents,
                output_type="pil",
                return_dict=True,
            )
            noise_scheduler.set_timesteps(config.num_inference_steps, device=accelerator.device)
            timesteps = noise_scheduler.timesteps
            latents, sdedit_timestep = results.images, timesteps[config.sdedit_tau]
            # *_, h, w = latents[0].shape
            latents = [latent.resize((config.pixel_height, config.pixel_width)) for latent in latents]
            for i, latent in enumerate(latents):
                latent.save(os.path.join(logging_dir, f"low_res_{i}.jpg"))
            latents = torch.cat(
                [torch.tensor(np.array(latent), device=accelerator.device)
                     .float().unsqueeze(0).permute((0, 3, 1, 2)) for latent in latents], dim=0
            )
            # latents = torch.tensor(np.array(latents), device=accelerator.device, dtype=weight_dtype)
            print(latents.shape)
            latents = latents / 127.5 - 1.0
            with torch.no_grad():
                latents = vae.tiled_encode(latents).latent_dist.sample()
                latents = latents * vae.config.scaling_factor
            latents = latents.to(dtype=weight_dtype)
            print(latents.dtype)

            noise = randn_tensor(
                latents.shape, generator=generator, device=latents.device, dtype=latents.dtype)
            latents = noise_scheduler.add_noise(latents, noise, sdedit_timestep)
            pipeline.forward = pipeline_processor(
                pipeline, dilate_conv_list, dilate_conv_list2,
                any_res_cfg_tau=config.any_res_cfg_tau,
                any_res_cfg_dilate=config.any_res_cfg_dilate,
                sdedit_tau=config.sdedit_tau,
                dilate_tau=config.dilate_tau,
                dilate=config.dilate,
                dilate2=config.dilate2
            )
            images = pipeline.forward(
                output_prompts,
                num_inference_steps=config.num_inference_steps,
                generator=generator,
                latents=latents,
                height=config.pixel_height,
                width=config.pixel_width
            ).images

            os.makedirs(os.path.join(logging_dir), exist_ok=True)
            for image, prompt in zip(images, output_prompts):
                total_num = total_num + 1
                # save_dir = os.path.join(logging_dir, f"{total_num}_{rank}.jpg")
                image.save(fp=os.path.join(logging_dir, f"{total_num}_{rank}.jpg"))
                with open(os.path.join(logging_dir, f"{total_num}_{rank}.txt"), 'w') as f:
                    f.writelines([prompt, ])


def setup_dist():
    if dist.is_initialized():
        return
    torch.distributed.init_process_group('nccl', init_method='env://')


def get_dist_info():
    if dist.is_available():
        initialized = dist.is_initialized()
    else:
        initialized = False
    if initialized:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    return rank, world_size


if __name__ == "__main__":
    if "RANK" in os.environ:
        setup_dist()
        rank, gpu_num = get_dist_info()
    else:
        rank, gpu_num = 0, 1
    main(rank, gpu_num)
