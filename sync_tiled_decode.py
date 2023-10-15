from typing import Union
from einops import rearrange
from diffusers.utils import check_min_version
from diffusers.models.vae import DecoderOutput
from torch.nn import (
    Conv2d, Dropout, Mish, SiLU, GELU, ReLU, GroupNorm
)
from diffusers.models.resnet import ResnetBlock2D, Upsample2D
from diffusers.models.attention import Attention
from diffusers.models.vae import Decoder

import torch.nn.functional as F
import torch


# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.20.2")


def tiled_processor(self):  # Conv2d, Dropout, Mish, SiLU, GELU, ReLU
    backup_forward = self.forward

    def forward(tiles):
        assert isinstance(tiles, list)
        return [backup_forward(tile) for tile in tiles]
    return forward


def tiled_gn_processor(self):
    backup_forward = self.forward

    def forward(tiles):
        assert isinstance(tiles, list)
        return [backup_forward(tile) for tile in tiles]
    return forward


def sync_tiled_gn_processor(self):  # GroupNorm
    def forward(tiles):
        used_dtype = torch.float32
        b, dtype, device = tiles[0].shape[0], tiles[0].dtype, tiles[0].device
        tiles = [tile.to(used_dtype) for tile in tiles]
        shapes, tmp_tiles, num_elements = list(), list(), 0
        for tile in tiles:
            *_, h, w = tile.shape
            shapes.append((h, w))
            tmp_tile = rearrange(tile, 'b (g c) h w -> b g (c h w)', g=self.num_groups)
            tmp_tiles.append(tmp_tile)
            num_elements = num_elements + tmp_tile.shape[-1]
        mean, var = (
            torch.zeros((b, self.num_groups, 1), dtype=used_dtype, device=device),
            torch.zeros((b, self.num_groups, 1), dtype=used_dtype, device=device)
        )

        for tile in tmp_tiles:
            mean = mean + tile.mean(-1, keepdim=True) * float(tile.shape[-1] / num_elements)

        for tile in tmp_tiles:
            # Unbiased variance estimation
            var = var + (
                    ((tile - mean) ** 2) * (tile.shape[-1] / (tile.shape[-1] - 1))
            ).mean(-1, keepdim=True) * float(tile.shape[-1] / num_elements)

        tiles = list()
        for shape, tile in zip(shapes, tmp_tiles):
            h, w = shape
            tile = rearrange((tile - mean) / (var + self.eps).sqrt(), 'b g (c h w) -> b (g c) h w', h=h, w=w)
            tiles.append(tile * self.weight.unsqueeze(-1).unsqueeze(-1) + self.bias.unsqueeze(-1).unsqueeze(-1))
        tiles = [tile.to(dtype) for tile in tiles]
        return tiles
    return forward


def tiled_resnet_processor(self):
    def forward(tiles, temb):
        hidden_states = tiles
        if self.time_embedding_norm == "ada_group" or self.time_embedding_norm == "spatial":
            hidden_states = self.norm1(hidden_states, temb)
        else:
            hidden_states = self.norm1(hidden_states)

        hidden_states = self.nonlinearity(hidden_states)

        if self.upsample is not None:
            if hidden_states[0].shape[0] >= 64:
                tiles = [tile.contiguous() for tile in tiles]
                hidden_states = [hidden_state.contiguous() for hidden_state in hidden_states]
            tiles = self.upsample(tiles)
            hidden_states = self.upsample(hidden_states)
        elif self.downsample is not None:
            tiles = self.downsample(tiles)
            hidden_states = self.downsample(hidden_states)

        hidden_states = self.conv1(hidden_states)

        if self.time_emb_proj is not None:
            if not self.skip_time_act:
                temb = self.nonlinearity(temb)
            temb = self.time_emb_proj(temb)[:, :, None, None]

        if temb is not None and self.time_embedding_norm == "default":
            hidden_states = [hidden_state + temb for hidden_state in hidden_states]

        if self.time_embedding_norm == "ada_group" or self.time_embedding_norm == "spatial":
            hidden_states = self.norm2(hidden_states, temb)
        else:
            hidden_states = self.norm2(hidden_states)

        if temb is not None and self.time_embedding_norm == "scale_shift":
            scale, shift = torch.chunk(temb, 2, dim=1)
            hidden_states = [hidden_state * (1 + scale) + shift for hidden_state in hidden_states]

        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv2(hidden_states)

        if self.conv_shortcut is not None:
            tiles = self.conv_shortcut(tiles)
        output_tensor = [
            (tile + hidden_state) / self.output_scale_factor for (tile, hidden_state) in zip(tiles, hidden_states)
        ]
        return output_tensor
    return forward


def tiled_attention_processor(self):
    def forward(hidden_states, encoder_hidden_states=None, attention_mask=None, **cross_attention_kwargs):
        # The `Attention` class can call different attention processors / attention functions
        # here we simply pass along all tensors to the selected processor class
        # For standard processors that are defined here, `**cross_attention_kwargs` is empty
        return [
            self.processor(
                self,
                hidden_state,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                **cross_attention_kwargs,
            ) for hidden_state in hidden_states
        ]
    return forward


def tiled_upsample2d_processor(self):
    def forward(hidden_states, output_size=None):
        assert hidden_states[0].shape[1] == self.channels

        if self.use_conv_transpose:
            return self.conv(hidden_states)

        # Cast to float32 to as 'upsample_nearest2d_out_frame' op does not support bfloat16
        # https://github.com/pytorch/pytorch/issues/86679
        dtype = hidden_states[0].dtype
        if dtype == torch.bfloat16:
            hidden_states = [hidden_state.to(torch.float32) for hidden_state in hidden_states]

        # upsample_nearest_nhwc fails with large batch sizes. see https://github.com/huggingface/diffusers/issues/984
        if hidden_states[0].shape[0] >= 64:
            hidden_states = [hidden_state.contiguous() for hidden_state in hidden_states]

        # if `output_size` is passed we force the interpolation output
        # size and do not make use of `scale_factor=2`
        if output_size is None:
            hidden_states = [
                F.interpolate(hidden_state, scale_factor=2.0, mode="nearest") for hidden_state in hidden_states
            ]
        else:
            hidden_states = [
                F.interpolate(hidden_state, size=output_size, mode="nearest") for hidden_state in hidden_states
            ]

        # If the input is bfloat16, we cast back to bfloat16
        if dtype == torch.bfloat16:
            hidden_states = [hidden_state.to(dtype) for hidden_state in hidden_states]

        if self.use_conv:
            if self.name == "conv":
                hidden_states = self.conv(hidden_states)
            else:
                hidden_states = self.Conv2d_0(hidden_states)

        return hidden_states
    return forward


def tiled_decoder_processor(self):
    def forward(tiles, latent_embeds=None):
        tiles = self.conv_in(tiles)

        upscale_dtype = next(iter(self.up_blocks.parameters())).dtype
        # middle
        tiles = self.mid_block(tiles, latent_embeds)
        tiles = [tile.to(upscale_dtype) for tile in tiles]

        # up
        for up_block in self.up_blocks:
            tiles = up_block(tiles, latent_embeds)

        # post-process
        if latent_embeds is None:
            tiles = self.conv_norm_out(tiles)
        else:
            tiles = self.conv_norm_out(tiles, latent_embeds)
        tiles = self.conv_act(tiles)
        tiles = self.conv_out(tiles)

        return tiles
    return forward


def apply_tiled_processors(model):
    for name, module in model.named_modules():
        if isinstance(module, (Conv2d, Dropout, Mish, SiLU, GELU, ReLU)):
            if 'attentions' in name:
                continue
            module.forward = tiled_processor(module)
        elif isinstance(module, GroupNorm):
            if 'attentions' in name:
                continue
            module.forward = sync_tiled_gn_processor(module)
        elif isinstance(module, ResnetBlock2D):
            module.forward = tiled_resnet_processor(module)
        elif isinstance(module, Upsample2D):
            module.forward = tiled_upsample2d_processor(module)
        elif isinstance(module, Attention):
            module.forward = tiled_attention_processor(module)
        elif isinstance(module, Decoder):
            module.forward = tiled_decoder_processor(module)
    return model


def apply_sync_tiled_decode(vae):
    vae.tiled_decode = sync_tiled_decode_processor(vae)
    return vae


def sync_tiled_decode_processor(self):
    def sync_tiled_decode(z: torch.FloatTensor, return_dict: bool = True) -> Union[DecoderOutput, torch.FloatTensor]:
        r"""
        Decode a batch of images using a synchronized tiled decoder.

        Args:
            z (`torch.FloatTensor`): Input batch of latent vectors.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.vae.DecoderOutput`] instead of a plain tuple.

        Returns:
            [`~models.vae.DecoderOutput`] or `tuple`:
                If return_dict is True, a [`~models.vae.DecoderOutput`] is returned, otherwise a plain `tuple` is
                returned.
        """
        overlap_size = int(self.tile_latent_min_size * (1 - self.tile_overlap_factor))
        blend_extent = int(self.tile_sample_min_size * self.tile_overlap_factor)
        row_limit = self.tile_sample_min_size - blend_extent

        # Split z into overlapping 64x64 tiles and decode them separately.
        # The tiles have an overlap to avoid seams between tiles.
        tiles = list()
        for i in range(0, z.shape[2], overlap_size):
            for j in range(0, z.shape[3], overlap_size):
                tile = z[:, :, i:i + self.tile_latent_min_size, j:j + self.tile_latent_min_size]
                tile = self.post_quant_conv(tile)
                tiles.append(tile)
        decoded = self.decoder(tiles)
        rows = list()
        for i in range(0, z.shape[2], overlap_size):
            row = list()
            for j in range(0, z.shape[3], overlap_size):
                row.append(decoded.pop(0))
            rows.append(row)

        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                # blend the above tile and the left tile
                # to the current tile and add the current tile to the result row
                if i > 0:
                    tile = self.blend_v(rows[i - 1][j], tile, blend_extent)
                if j > 0:
                    tile = self.blend_h(row[j - 1], tile, blend_extent)
                result_row.append(tile[:, :, :row_limit, :row_limit])
            result_rows.append(torch.cat(result_row, dim=3))

        dec = torch.cat(result_rows, dim=2)
        if not return_dict:
            return (dec,)

        return DecoderOutput(sample=dec)
    return sync_tiled_decode
