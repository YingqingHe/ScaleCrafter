import argparse
import copy
import math
import cv2
import os
from typing import Optional

import torch
import scipy
import glob
import numpy as np
import torch.utils.checkpoint
from PIL import Image
from omegaconf import OmegaConf
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from transformers import (
    CLIPTextModel,
    CLIPTokenizer,
    CLIPTextModelWithProjection,
    DPTFeatureExtractor,
    DPTForDepthEstimation
)

from diffusers import (
    AutoencoderKL,
    UNet2DConditionModel,
    ControlNetModel,
    DDIMScheduler,
    StableDiffusionXLControlNetPipeline,
)
from diffusers.pipelines.stable_diffusion_xl import StableDiffusionXLPipelineOutput
from diffusers.pipelines.controlnet import MultiControlNetModel
from diffusers.utils.torch_utils import randn_tensor, is_compiled_module
from diffusers.utils import load_image
from model import ReDilateConvProcessor, inflate_kernels


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
        "--controlnet_model_name_or_path",
        type=str,
        default="lllyasviel/sd-controlnet-canny",
        help="Path to pretrained controlnet or controlnet identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--midas_model_name_or_path",
        type=str,
        default="Intel/dpt-hybrid-midas",
        help="Path to pretrained midas or midas identifier from huggingface.co/models.",
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
        "--image_path", type=str,
        default="",
        help="A image path used for inference."
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
        image=None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
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
        controlnet_conditioning_scale=1.0,
        guess_mode: bool = False,
        control_guidance_start=0.0,
        control_guidance_end=1.0,
        original_size=None,
        crops_coords_top_left=(0, 0),
        target_size=None,
    ):
        controlnet = self.controlnet._orig_mod if is_compiled_module(self.controlnet) else self.controlnet

        # align format for control guidance
        if not isinstance(control_guidance_start, list) and isinstance(control_guidance_end, list):
            control_guidance_start = len(control_guidance_end) * [control_guidance_start]
        elif not isinstance(control_guidance_end, list) and isinstance(control_guidance_start, list):
            control_guidance_end = len(control_guidance_start) * [control_guidance_end]
        elif not isinstance(control_guidance_start, list) and not isinstance(control_guidance_end, list):
            mult = len(controlnet.nets) if isinstance(controlnet, MultiControlNetModel) else 1
            control_guidance_start, control_guidance_end = mult * [control_guidance_start], mult * [
                control_guidance_end
            ]

        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        original_size = original_size or (height, width)
        target_size = target_size or (height, width)

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            prompt_2,
            image,
            callback_steps,
            negative_prompt,
            negative_prompt_2,
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
            controlnet_conditioning_scale,
            control_guidance_start,
            control_guidance_end,
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

        if isinstance(controlnet, MultiControlNetModel) and isinstance(controlnet_conditioning_scale, float):
            controlnet_conditioning_scale = [controlnet_conditioning_scale] * len(controlnet.nets)

        global_pool_conditions = (
            controlnet.config.global_pool_conditions
            if isinstance(controlnet, ControlNetModel)
            else controlnet.nets[0].config.global_pool_conditions
        )
        guess_mode = guess_mode or global_pool_conditions

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

        # 4. Prepare image
        if isinstance(controlnet, ControlNetModel):
            image = self.prepare_image(
                image=image,
                width=width,
                height=height,
                batch_size=batch_size * num_images_per_prompt,
                num_images_per_prompt=num_images_per_prompt,
                device=device,
                dtype=controlnet.dtype,
                do_classifier_free_guidance=do_classifier_free_guidance,
                guess_mode=guess_mode,
            )
            height, width = image.shape[-2:]
        elif isinstance(controlnet, MultiControlNetModel):
            images = []

            for image_ in image:
                image_ = self.prepare_image(
                    image=image_,
                    width=width,
                    height=height,
                    batch_size=batch_size * num_images_per_prompt,
                    num_images_per_prompt=num_images_per_prompt,
                    device=device,
                    dtype=controlnet.dtype,
                    do_classifier_free_guidance=do_classifier_free_guidance,
                    guess_mode=guess_mode,
                )

                images.append(image_)

            image = images
            height, width = image[0].shape[-2:]
        else:
            assert False

        # 5. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 6. Prepare latent variables
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

        # 7. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7.1 Create tensor stating which controlnets to keep
        controlnet_keep = []
        for i in range(len(timesteps)):
            keeps = [
                1.0 - float(i / len(timesteps) < s or (i + 1) / len(timesteps) > e)
                for s, e in zip(control_guidance_start, control_guidance_end)
            ]
            controlnet_keep.append(keeps[0] if isinstance(controlnet, ControlNetModel) else keeps)

        # 7.2 Prepare added time ids & embeddings
        if isinstance(image, list):
            original_size = original_size or image[0].shape[-2:]
        else:
            original_size = original_size or image.shape[-2:]
        target_size = target_size or (height, width)

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

                # controlnet(s) inference
                if guess_mode and do_classifier_free_guidance:
                    # Infer ControlNet only for the conditional batch.
                    control_model_input = latents
                    control_model_input = self.scheduler.scale_model_input(control_model_input, t)
                    controlnet_prompt_embeds = prompt_embeds.chunk(2)[1]
                else:
                    control_model_input = latent_model_input
                    controlnet_prompt_embeds = prompt_embeds

                if isinstance(controlnet_keep[i], list):
                    cond_scale = [c * s for c, s in zip(controlnet_conditioning_scale, controlnet_keep[i])]
                else:
                    controlnet_cond_scale = controlnet_conditioning_scale
                    if isinstance(controlnet_cond_scale, list):
                        controlnet_cond_scale = controlnet_cond_scale[0]
                    cond_scale = controlnet_cond_scale * controlnet_keep[i]

                added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
                down_block_res_samples, mid_block_res_sample = self.controlnet(
                    control_model_input,
                    t,
                    encoder_hidden_states=controlnet_prompt_embeds,
                    controlnet_cond=image,
                    conditioning_scale=cond_scale,
                    guess_mode=guess_mode,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                )

                if guess_mode and do_classifier_free_guidance:
                    # Infered ControlNet only for the conditional batch.
                    # To apply the output of ControlNet to both the unconditional and conditional batches,
                    # add 0 to the unconditional batch to keep it unchanged.
                    down_block_res_samples = [torch.cat([torch.zeros_like(d), d]) for d in down_block_res_samples]
                    mid_block_res_sample = torch.cat([torch.zeros_like(mid_block_res_sample), mid_block_res_sample])

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
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample,
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
                        down_block_additional_residuals=down_block_res_samples,
                        mid_block_additional_residual=mid_block_res_sample,
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

        # If we do sequential model offloading, let's offload unet and controlnet
        # manually for max memory savings
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.unet.to("cpu")
            self.controlnet.to("cpu")
            torch.cuda.empty_cache()

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


def get_depth_map(image, feature_extractor, depth_estimator):
    image = feature_extractor(images=image, return_tensors="pt").pixel_values.to("cuda")
    with torch.no_grad(), torch.autocast("cuda"):
        depth_map = depth_estimator(image).predicted_depth

    depth_map = torch.nn.functional.interpolate(
        depth_map.unsqueeze(1),
        size=(1024, 1024),
        mode="bicubic",
        align_corners=False,
    )
    depth_min = torch.amin(depth_map, dim=[1, 2, 3], keepdim=True)
    depth_max = torch.amax(depth_map, dim=[1, 2, 3], keepdim=True)
    depth_map = (depth_map - depth_min) / (depth_max - depth_min)
    image = torch.cat([depth_map] * 3, dim=1)

    image = image.permute(0, 2, 3, 1).cpu().numpy()[0]
    image = Image.fromarray((image * 255.0).clip(0, 255).astype(np.uint8))
    return image


def main():
    args = parse_args()
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
    controlnet = ControlNetModel.from_pretrained(args.controlnet_model_name_or_path, torch_dtype=weight_dtype)
    noise_scheduler = DDIMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    pipeline = StableDiffusionXLControlNetPipeline(
        vae=vae,
        text_encoder=text_encoder,
        text_encoder_2=text_encoder_2,
        tokenizer=tokenizer,
        tokenizer_2=tokenizer_2,
        unet=unet,
        controlnet=controlnet,
        scheduler=noise_scheduler,
    )
    pipeline = pipeline.to(accelerator.device)

    depth_estimator = DPTForDepthEstimation.from_pretrained(args.midas_model_name_or_path).to("cuda")
    feature_extractor = DPTFeatureExtractor.from_pretrained(args.midas_model_name_or_path)

    dilate_settings = read_dilate_settings(config.dilate_settings) \
        if config.dilate_settings is not None else dict()
    ndcfg_dilate_settings = read_dilate_settings(config.ndcfg_dilate_settings) \
        if config.ndcfg_dilate_settings is not None else dict()
    inflate_settings = read_module_list(config.inflate_settings) \
        if config.inflate_settings is not None else list()
    if config.inflate_transform is not None:
        print(f"Using inflated conv {config.inflate_transform}")
        transform = scipy.io.loadmat(config.inflate_transform)['R']
        transform = torch.tensor(transform, device=accelerator.device)
    else:
        transform = None

    unet.eval()
    controlnet.eval()
    
    os.makedirs(os.path.join(logging_dir), exist_ok=True)
    total_num = len(glob.glob(os.path.join(logging_dir, '*.jpg'))) - 1

    print(f"Using prompt {args.validation_prompt}")
    if os.path.isfile(args.validation_prompt):
        with open(args.validation_prompt, 'r') as f:
            validation_prompt = f.readlines()
            validation_prompt = [line.strip() for line in validation_prompt]
    else:
        validation_prompt = [args.validation_prompt, ]

    print(f"Using image {args.image_path}")
    if args.image_path.endswith('.txt'):
        with open(args.image_path, 'r') as f:
            image_path = f.readlines()
            image_path = [line.strip() for line in image_path]
    else:
        image_path = [args.image_path, ]
    assert len(image_path) == len(validation_prompt)

    inference_batch_size = config.inference_batch_size
    num_batches = math.ceil(len(validation_prompt) / inference_batch_size)
    for i in range(num_batches):
        output_prompts, paths = (
            validation_prompt[i * inference_batch_size:min((i + 1) * inference_batch_size, len(validation_prompt))],
            image_path[i * inference_batch_size:min((i + 1) * inference_batch_size, len(image_path))]
        )

        # Read controlnet input image
        output_images, pixel_height, pixel_width, latent_height, latent_width = (
            list(), None, None, None, None
        )
        for path in paths:
            image = load_image(path)
            if pixel_height is not None:
                assert pixel_height == (image.height // 64) * 64 \
                       and pixel_width == (image.width // 64) * 64
            else:
                latent_height, latent_width = (image.height // 64) * 8, (image.width // 64) * 8
                pixel_height, pixel_width = latent_height * 8, latent_width * 8

            image = np.array(load_image(path).resize((pixel_height, pixel_width)))
            depth_image = get_depth_map(image, feature_extractor, depth_estimator)
            output_images.append(depth_image)
        for n in range(config.num_iters_per_prompt):
            seed = args.seed + n
            set_seed(seed)

            latents = torch.randn(
                (len(output_prompts), 4, latent_height, latent_width),
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
                image=output_images,
                num_inference_steps=config.num_inference_steps,
                generator=None,
                latents=latents,
                height=pixel_height,
                width=pixel_width
            ).images

            for image, prompt in zip(images, output_prompts):
                total_num = total_num + 1
                img_path = os.path.join(logging_dir, f"{total_num}_{prompt[:200]}_seed{seed}.jpg")
                image.save(img_path)
                with open(os.path.join(logging_dir, f"{total_num}.txt"), 'w') as f:
                    f.writelines([prompt, ])

if __name__ == "__main__":
    main()
