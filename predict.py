# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

import argparse
import copy
import math
import os
from typing import Optional

import torch
import scipy
import numpy as np
import torch.utils.checkpoint
from omegaconf import OmegaConf
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed
from transformers import CLIPTextModel, CLIPTokenizer, CLIPTextModelWithProjection

from diffusers import (
    AutoencoderKL,
    UNet2DConditionModel,
    DDIMScheduler,
    StableDiffusionXLPipeline,
)
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import (
    rescale_noise_cfg,
)
from diffusers.pipelines.stable_diffusion_xl import StableDiffusionXLPipelineOutput
from diffusers.utils.torch_utils import randn_tensor
from model import ReDilateConvProcessor, inflate_kernels
from free_lunch_utils import register_free_upblock2d, register_free_crossattn_upblock2d
from sync_tiled_decode import apply_sync_tiled_decode, apply_tiled_processors

from cog import BasePredictor, Input, Path


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        # Default config setting.
        args = argparse.Namespace(
            pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0",
            revision=None,
            mixed_precision="fp16",
            disable_freeu=False,
            vae_tiling=False,
        )

        self.accelerator = Accelerator(mixed_precision=args.mixed_precision)

        # Load previous pipeline, use local_files_only=True is weights are downloaded to cache_dir beforehand
        tokenizer = CLIPTokenizer.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="tokenizer",
            revision=args.revision,
            torch_dtype=torch.float16,
            cache_dir="model_cache",
            local_files_only=True,
        )
        tokenizer_2 = CLIPTokenizer.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="tokenizer_2",
            revision=args.revision,
            torch_dtype=torch.float16,
            cache_dir="model_cache",
            local_files_only=True,
        )
        text_encoder = CLIPTextModel.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="text_encoder",
            revision=args.revision,
            torch_dtype=torch.float16,
            cache_dir="model_cache",
            local_files_only=True,
        )
        text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="text_encoder_2",
            revision=args.revision,
            torch_dtype=torch.float16,
            cache_dir="model_cache",
            local_files_only=True,
        )
        self.vae = AutoencoderKL.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="vae",
            revision=args.revision,
            torch_dtype=torch.float16,
            cache_dir="model_cache",
            local_files_only=True,
        )
        self.unet = UNet2DConditionModel.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="unet",
            revision=args.revision,
            torch_dtype=torch.float16,
            cache_dir="model_cache",
            local_files_only=True,
        )
        self.noise_scheduler = DDIMScheduler.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="scheduler"
        )
        self.pipeline = StableDiffusionXLPipeline(
            vae=self.vae,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            unet=self.unet,
            scheduler=self.noise_scheduler,
        )
        self.pipeline = self.pipeline.to("cuda")

        if not args.disable_freeu:
            register_free_upblock2d(self.pipeline, b1=1.1, b2=1.2, s1=0.6, s2=0.4)
            register_free_crossattn_upblock2d(
                self.pipeline, b1=1.1, b2=1.2, s1=0.6, s2=0.4
            )
        if args.vae_tiling:
            self.pipeline.enable_vae_tiling()
            apply_sync_tiled_decode(self.pipeline.vae)
            apply_tiled_processors(self.pipeline.vae.decoder)

        self.unet.eval()

    def predict(
        self,
        prompt: str = Input(
            description="Input prompt",
            default="a professional photograph of an astronaut riding a horse",
        ),
        negative_prompt: str = Input(
            description="Specify things to not see in the output",
            default=None,
        ),
        width: int = Input(
            description="Width of output image. Lower the setting if out of memory.",
            choices=[
                1024,
                1280,
                1536,
                1792,
                2048,
                2304,
                2560,
                2944,
                3328,
                3712,
                4096,
            ],
            default=2048,
        ),
        height: int = Input(
            description="Height of output image. Lower the setting if out of memory.",
            choices=[
                1024,
                1280,
                1536,
                1792,
                2048,
                2304,
                2560,
                2944,
                3328,
                3712,
                4096,
            ],
            default=2048,
        ),
        num_inference_steps: int = Input(
            description="Number of denoising steps", ge=1, le=500, default=50
        ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed", default=None
        ),
        dilate_scale: str = Input(
            description="You can provide customised setting to specify the layer to use our method and its dilation scale, for example see assets/dilate_settings/sdxl_4096x4096.txt in the github repo.",
            default=None,
        ),
    ) -> Path:
        """Run a single prediction on the model"""
        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")

        if height <= 2560 or width <= 2560:
            config = OmegaConf.load("./configs/sdxl_2048x2048.yaml")
        else:
            config = OmegaConf.load("./configs/sdxl_4096x4096.yaml")
        config.pixel_height = height
        config.pixel_width = width
        config.latent_height = height // 8
        config.latent_width = width // 8
        set_seed(seed)

        if dilate_scale is not None:
            dilate_settings = dict()

            for raw_line in dilate_scale.strip().splitlines():
                name, dilate = raw_line.split(":")
                dilate_settings[name] = float(dilate)
                print(f"{name} : {dilate_settings[name]}")

        else:
            dilate_settings = (
                read_dilate_settings(config.dilate_settings)
                if config.dilate_settings is not None
                else dict()
            )
        ndcfg_dilate_settings = (
            read_dilate_settings(config.ndcfg_dilate_settings)
            if config.ndcfg_dilate_settings is not None
            else dict()
        )
        inflate_settings = (
            read_module_list(config.inflate_settings)
            if config.inflate_settings is not None
            else list()
        )
        if config.inflate_transform is not None:
            print(f"Using inflated conv {config.inflate_transform}")
            transform = scipy.io.loadmat(config.inflate_transform)["R"]
            transform = torch.tensor(transform, device="cuda")
        else:
            transform = None

        # Final inference
        sdedit_scale = config.sdedit_scale if hasattr(config, "sdedit_scale") else 1
        if hasattr(config, "sdedit_tau") and config.sdedit_tau > 0:
            latents = torch.randn(
                (
                    1,
                    4,
                    config.latent_height // sdedit_scale,
                    config.latent_width // sdedit_scale,
                ),
                device="cuda",
                dtype=torch.float16,
            )
            results = self.pipeline(
                prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                eta=0.0,
                generator=None,
                latents=latents,
                output_type="pil",
                return_dict=True,
            )
            self.noise_scheduler.set_timesteps(
                config.num_inference_steps, device="cuda"
            )
            timesteps = self.noise_scheduler.timesteps
            latents, sdedit_timestep = results.images, timesteps[config.sdedit_tau]
            latents = [
                latent.resize((config.pixel_height, config.pixel_width))
                for latent in latents
            ]
            latents = torch.cat(
                [
                    torch.tensor(np.array(latent), device="cuda")
                    .float()
                    .unsqueeze(0)
                    .permute((0, 3, 1, 2))
                    for latent in latents
                ],
                dim=0,
            )
            latents = latents / 127.5 - 1.0
            with torch.no_grad():
                latents = self.vae.tiled_encode(latents).latent_dist.sample()
                latents = latents * self.vae.config.scaling_factor
            latents = latents.to(dtype=torch.float16)
            noise = randn_tensor(
                latents.shape,
                generator=None,
                device=latents.device,
                dtype=latents.dtype,
            )
            latents = self.noise_scheduler.add_noise(latents, noise, sdedit_timestep)
        else:
            latents = torch.randn(
                (1, 4, config.latent_height, config.latent_width),
                device="cuda",
                dtype=torch.float16,
            )
        self.pipeline.enable_vae_tiling()
        self.pipeline.forward = pipeline_processor(
            self.pipeline,
            ndcfg_tau=config.ndcfg_tau,
            dilate_tau=config.dilate_tau,
            inflate_tau=config.inflate_tau,
            dilate_settings=dilate_settings,
            inflate_settings=inflate_settings,
            ndcfg_dilate_settings=ndcfg_dilate_settings,
            transform=transform,
            progressive=config.progressive,
        )
        images = self.pipeline.forward(
            prompt,
            num_inference_steps=num_inference_steps,
            generator=None,
            latents=latents,
            height=config.pixel_height,
            width=config.pixel_width,
        ).images

        out_path = "/tmp/out.png"
        images[0].save(out_path)
        return Path(out_path)


def read_dilate_settings(path):
    print(f"Reading dilation settings")
    dilate_settings = dict()
    with open(path, "r") as f:
        raw_lines = f.readlines()
        for raw_line in raw_lines:
            name, dilate = raw_line.split(":")
            dilate_settings[name] = float(dilate)
            print(f"{name} : {dilate_settings[name]}")
    return dilate_settings


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
            cross_attention_kwargs.get("scale", None)
            if cross_attention_kwargs is not None
            else None
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
            add_text_embeds = torch.cat(
                [negative_pooled_prompt_embeds, add_text_embeds], dim=0
            )
            add_time_ids = torch.cat([add_time_ids, add_time_ids], dim=0)

        prompt_embeds = prompt_embeds.to(device)
        add_text_embeds = add_text_embeds.to(device)
        add_time_ids = add_time_ids.to(device).repeat(
            batch_size * num_images_per_prompt, 1
        )

        # 7.1 Apply denoising_end
        if (
            denoising_end is not None
            and type(denoising_end) == float
            and denoising_end > 0
            and denoising_end < 1
        ):
            discrete_timestep_cutoff = int(
                round(
                    self.scheduler.config.num_train_timesteps
                    - (denoising_end * self.scheduler.config.num_train_timesteps)
                )
            )
            num_inference_steps = len(
                list(filter(lambda ts: ts >= discrete_timestep_cutoff, timesteps))
            )
            timesteps = timesteps[:num_inference_steps]

        # 8. Denoising loop
        num_warmup_steps = max(
            len(timesteps) - num_inference_steps * self.scheduler.order, 0
        )
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
                latent_model_input = (
                    torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                )
                latent_model_input = self.scheduler.scale_model_input(
                    latent_model_input, t
                )

                unet = (
                    unet_inflate
                    if i < inflate_tau and transform is not None
                    else self.unet
                )
                backup_forwards = dict()
                for name, module in unet.named_modules():
                    if name in dilate_settings.keys():
                        backup_forwards[name] = module.forward
                        dilate = dilate_settings[name]
                        if progressive:
                            dilate = max(
                                math.ceil(dilate * ((dilate_tau - i) / dilate_tau)), 2
                            )
                        if i < inflate_tau and name in inflate_settings:
                            dilate = dilate / 2
                        module.forward = ReDilateConvProcessor(
                            module, dilate, mode="bilinear", activate=i < dilate_tau
                        )

                # predict the noise residual
                added_cond_kwargs = {
                    "text_embeds": add_text_embeds,
                    "time_ids": add_time_ids,
                }
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
                    unet = (
                        unet_inflate_vanilla
                        if i < inflate_tau and transform is not None
                        else self.unet
                    )
                    backup_forwards = dict()
                    for name, module in unet.named_modules():
                        if name in ndcfg_dilate_settings.keys():
                            backup_forwards[name] = module.forward
                            dilate = ndcfg_dilate_settings[name]
                            if progressive:
                                dilate = max(
                                    math.ceil(dilate * ((ndcfg_tau - i) / ndcfg_tau)), 2
                                )
                            if i < inflate_tau and name in inflate_settings:
                                dilate = dilate / 2
                            module.forward = ReDilateConvProcessor(
                                module, dilate, mode="bilinear", activate=i < ndcfg_tau
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
                        noise_pred = noise_pred_vanilla + guidance_scale * (
                            noise_pred_text - noise_pred_uncond
                        )
                    else:
                        noise_pred = noise_pred_uncond + guidance_scale * (
                            noise_pred_text - noise_pred_uncond
                        )

                if do_classifier_free_guidance and guidance_rescale > 0.0:
                    # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                    noise_pred = rescale_noise_cfg(
                        noise_pred, noise_pred_text, guidance_rescale=guidance_rescale
                    )

                # compute the previous noisy sample x_t -> x_t-1
                variance_noise = None
                results = self.scheduler.step(
                    noise_pred,
                    t,
                    latents,
                    **extra_step_kwargs,
                    variance_noise=variance_noise,
                    return_dict=True,
                )
                latents, ori_latents = results.prev_sample, results.pred_original_sample

                # call the callback, if provided
                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        # make sure the VAE is in float32 mode, as it overflows in float16
        if self.vae.dtype == torch.float16 and self.vae.config.force_upcast:
            self.upcast_vae()
            latents = latents.to(
                next(iter(self.vae.post_quant_conv.parameters())).dtype
            )

        if not output_type == "latent":
            image = self.vae.decode(
                latents / self.vae.config.scaling_factor, return_dict=False
            )[0]
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
