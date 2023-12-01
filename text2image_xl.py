import argparse
import copy
import math
import os
from typing import Optional

import torch
import scipy
import glob
import numpy as np
import torch.utils.checkpoint
from omegaconf import OmegaConf
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from transformers import CLIPTextModel, CLIPTokenizer, CLIPTextModelWithProjection

from diffusers import (
    AutoencoderKL, UNet2DConditionModel, DDIMScheduler, StableDiffusionXLPipeline,
)
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import rescale_noise_cfg
from diffusers.pipelines.stable_diffusion_xl import StableDiffusionXLPipelineOutput
from diffusers.utils.torch_utils import randn_tensor
from model import ReDilateConvProcessor, inflate_kernels
from free_lunch_utils import register_free_upblock2d, register_free_crossattn_upblock2d
from sync_tiled_decode import apply_sync_tiled_decode, apply_tiled_processors


logger = get_logger(__name__, log_level="INFO")


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
        "--validation_prompt", type=str,
        default="a professional photograph of an astronaut riding a horse",
        help="A prompt that is sampled during training for inference."
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument("--seed", type=int, default=23, help="A seed for reproducible training.")
    parser.add_argument("--config", type=str, default="./configs/sdxl_2048x2048.yaml")
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="",
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
    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument("--gpu_num", type=int, default=None)
    parser.add_argument("--disable_freeu", action="store_true", help="disable freeU", default=False)
    parser.add_argument("--vae_tiling", action="store_true", help="enable vae tiling")

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    # Sanity checks
    # if args.dataset_name is None and args.train_data_dir is None:
    #     raise ValueError("Need either a dataset name or a training folder.")

    return args


def pipeline_processor(
        self,
        ndcfg_tau=0,
        dilate_tau=0,
        inflate_tau=0,
        sdedit_tau=0,
        dilate_settings=None,
        inflate_settings=None,
        ndcfg_dilate_settings=None,
        transform=None,
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
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

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

        # 8. Denoising loop
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        unet_inflate, unet_inflate_vanilla = None, None
        if transform is not None:
            unet_inflate = copy.deepcopy(self.unet)
            if inflate_settings is not None:
                inflate_kernels(unet_inflate, inflate_settings, transform)

        if transform is not None and ndcfg_tau > 0:
            unet_inflate_vanilla = copy.deepcopy(self.unet)
            if inflate_settings is not None:
                inflate_kernels(unet_inflate_vanilla, inflate_settings, transform)

        if sdedit_tau is not None:
            timesteps = timesteps[sdedit_tau:]
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                unet = unet_inflate if i < inflate_tau and transform is not None else self.unet
                backup_forwards = dict()
                for name, module in unet.named_modules():
                    if name in dilate_settings.keys():
                        backup_forwards[name] = module.forward
                        dilate = dilate_settings[name]
                        if progressive:
                            dilate = max(math.ceil(dilate * ((dilate_tau - i) / dilate_tau)), 2)
                        if i < inflate_tau and name in inflate_settings:
                            dilate = dilate / 2
                        module.forward = ReDilateConvProcessor(
                            module, dilate, mode='bilinear', activate=i < dilate_tau
                        )

                # predict the noise residual
                added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
                noise_pred = unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs=cross_attention_kwargs,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                )[0]

                for name, module in unet.named_modules():
                    if name in backup_forwards.keys():
                        module.forward = backup_forwards[name]

                if i < ndcfg_tau:
                    unet = unet_inflate_vanilla if i < inflate_tau and transform is not None else self.unet
                    backup_forwards = dict()
                    for name, module in unet.named_modules():
                        if name in ndcfg_dilate_settings.keys():
                            backup_forwards[name] = module.forward
                            dilate = ndcfg_dilate_settings[name]
                            if progressive:
                                dilate = max(math.ceil(dilate * ((ndcfg_tau - i) / ndcfg_tau)), 2)
                            if i < inflate_tau and name in inflate_settings:
                                dilate = dilate / 2
                            module.forward = ReDilateConvProcessor(
                                module, dilate, mode='bilinear', activate=i < ndcfg_tau
                            )

                    noise_pred_vanilla = unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=prompt_embeds,
                        cross_attention_kwargs=cross_attention_kwargs,
                        added_cond_kwargs=added_cond_kwargs,
                        return_dict=False,
                    )[0]

                    for name, module in unet.named_modules():
                        if name in backup_forwards.keys():
                            module.forward = backup_forwards[name]

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    if i < ndcfg_tau:
                        noise_pred_vanilla, _ = noise_pred_vanilla.chunk(2)
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
            image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
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


def read_module_list(path):
    with open(path, 'r') as f:
        module_list = f.readlines()
        module_list = [name.strip() for name in module_list]
    return module_list


def read_dilate_settings(path):
    print(f"Reading dilation settings")
    dilate_settings = dict()
    with open(path, 'r') as f:
        raw_lines = f.readlines()
        for raw_line in raw_lines:
            name, dilate = raw_line.split(':')
            dilate_settings[name] = float(dilate)
            print(f"{name} : {dilate_settings[name]}")
    return dilate_settings


def main():
    args = parse_args()
    rank, gpu_num = 0, 1
    rank = rank if args.rank is None else args.rank
    gpu_num = gpu_num if args.gpu_num is None else args.gpu_num

    logging_dir = os.path.join(args.logging_dir)
    config = OmegaConf.load(args.config)

    accelerator_project_config = ProjectConfiguration(logging_dir=logging_dir)
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
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
    if not args.disable_freeu:
        register_free_upblock2d(pipeline, b1=1.1, b2=1.2, s1=0.6, s2=0.4)
        register_free_crossattn_upblock2d(pipeline, b1=1.1, b2=1.2, s1=0.6, s2=0.4)
    if args.vae_tiling:
        pipeline.enable_vae_tiling()
        apply_sync_tiled_decode(pipeline.vae)
        apply_tiled_processors(pipeline.vae.decoder)

    dilate_settings = read_dilate_settings(config.dilate_settings) \
        if config.dilate_settings is not None else dict()
    ndcfg_dilate_settings = read_dilate_settings(config.ndcfg_dilate_settings) \
        if config.ndcfg_dilate_settings is not None else dict()
    inflate_settings = read_module_list(config.disperse_settings) \
        if config.disperse_settings is not None else list()
    if config.disperse_transform is not None:
        print(f"Using inflated conv {config.disperse_transform}")
        transform = scipy.io.loadmat(config.disperse_transform)['R']
        transform = torch.tensor(transform, device=accelerator.device)
    else:
        transform = None

    unet.eval()
    
    os.makedirs(os.path.join(logging_dir), exist_ok=True)
    # total_num = len(glob.glob(os.path.join(logging_dir, '*.jpg'))) - 1
    #
    # print(f"Using prompt {args.validation_prompt}")
    # if os.path.isfile(args.validation_prompt):
    #     with open(args.validation_prompt, 'r') as f:
    #         validation_prompt = f.readlines()
    #         validation_prompt = [line.strip() for line in validation_prompt]
    # else:
    #     validation_prompt = [args.validation_prompt, ]

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
    for i in range(num_batches):
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

        for n in range(config.num_iters_per_prompt):
            seed = args.seed + n
            set_seed(seed)

            sdedit_scale = config.sdedit_scale if hasattr(config, 'sdedit_scale') else 1
            if hasattr(config, 'sdedit_tau') and config.sdedit_tau > 0:
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
                    generator=None,
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
                latents = latents / 127.5 - 1.0
                with torch.no_grad():
                    latents = vae.tiled_encode(latents).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor
                latents = latents.to(dtype=weight_dtype)
                noise = randn_tensor(
                    latents.shape, generator=None, device=latents.device, dtype=latents.dtype)
                latents = noise_scheduler.add_noise(latents, noise, sdedit_timestep)
            else:
                latents = torch.randn(
                    (len(output_prompts), 4, config.latent_height, config.latent_width),
                    device=accelerator.device, dtype=weight_dtype
                )
            pipeline.enable_vae_tiling()
            pipeline.forward = pipeline_processor(
                pipeline,
                ndcfg_tau=config.ndcfg_tau,
                dilate_tau=config.dilate_tau,
                inflate_tau=config.inflate_tau,
                dilate_settings=dilate_settings,
                inflate_settings=inflate_settings,
                ndcfg_dilate_settings=ndcfg_dilate_settings,
                transform=transform,
                progressive=config.progressive,
            )
            images = pipeline.forward(
                output_prompts,
                num_inference_steps=config.num_inference_steps,
                generator=None,
                latents=latents,
                height=config.pixel_height,
                width=config.pixel_width
            ).images

            for image, prompt in zip(images, output_prompts):
                total_num = total_num + 1
                img_path = os.path.join(logging_dir, f"{total_num}_seed{seed}_{rank}.jpg")
                image.save(img_path)
                with open(os.path.join(logging_dir, f"{total_num}_{rank}.txt"), 'w') as f:
                    f.writelines([prompt, ])


if __name__ == "__main__":
    main()
