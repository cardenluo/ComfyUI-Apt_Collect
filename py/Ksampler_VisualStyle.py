

#region-----------导入与全局函数-------------------------------------------------
# 内置库


# 第三方库

import torch
from dataclasses import dataclass
from einops import rearrange
import torch.nn as nn

# 本地库
import comfy
import comfy.controlnet
import comfy.lora
import comfy.sample
import comfy.samplers
import comfy.sd
import comfy.utils
import folder_paths
from comfy.ldm.modules.attention import default, optimized_attention, optimized_attention_masked
from comfy.model_patcher import ModelPatcher
import comfy.ops
import latent_preview
from nodes import CLIPTextEncode,  VAEDecode, VAEEncode, common_ksampler
from typing import Any, Dict, Optional, Tuple, Union, cast

# 相对导入
from ..def_unit import *
from ..nodes import *

T = torch.Tensor

SHARE_NORM_OPTIONS = ["both", "group", "layer", "disabled"]
SHARE_ATTN_OPTIONS = ["q+k", "q+k+v", "disabled"]


class VisualStyleProcessor(object):
    def __init__(self, 
        module_self, 
        keys_scale: float = 1.0,
        enabled: bool = True, 
        adain_queries: bool = True,
        adain_keys: bool = True,
        adain_values: bool = False 
    ):
        self.module_self = module_self
        self.keys_scale = keys_scale
        self.enabled = enabled
        self.adain_queries = adain_queries
        self.adain_keys = adain_keys
        self.adain_values = adain_values

    def visual_style_forward(self, x, context, value, mask=None):
        q = self.module_self.to_q(x)
        context = default(context, x)
        k = self.module_self.to_k(context)
        if value is not None:
            v = self.module_self.to_v(value)
            del value
        else:
            v = self.module_self.to_v(context)

        if self.enabled:
            if self.adain_queries:
                q = adain(q)
            if self.adain_keys:
                k = adain(k)
            if self.adain_values:
                v = adain(v)
            
            k = concat_first(k, -2, self.keys_scale)
            v = concat_first(v, -2)

        if mask is None:
            out = optimized_attention(q, k, v, self.module_self.heads)
        else:
            out = optimized_attention_masked(q, k, v, self.module_self.heads, mask)
        return self.module_self.to_out(out)



@dataclass(frozen=True)
class StyleAlignedArgs:
    share_group_norm: bool = True
    share_layer_norm: bool = True,
    share_attention: bool = True
    adain_queries: bool = True
    adain_keys: bool = True
    adain_values: bool = False
    full_attention_share: bool = False
    keys_scale: float = 1.
    only_self_level: float = 0.

def expand_first(feat: T, scale=1., ) -> T:
    b = feat.shape[0]
    feat_style = torch.stack((feat[0], feat[b // 2])).unsqueeze(1)
    if scale == 1:
        feat_style = feat_style.expand(2, b // 2, *feat.shape[1:])
    else:
        feat_style = feat_style.repeat(1, b // 2, 1, 1, 1)
        feat_style = torch.cat([feat_style[:, :1], scale * feat_style[:, 1:]], dim=1)
    return feat_style.reshape(*feat.shape)


def concat_first(feat: T, dim=2, scale=1.) -> T:
    feat_style = expand_first(feat, scale=scale)
    return torch.cat((feat, feat_style), dim=dim)


def calc_mean_std(feat, eps: float = 1e-5) -> tuple[T, T]:
    feat_std = (feat.var(dim=-2, keepdims=True) + eps).sqrt()
    feat_mean = feat.mean(dim=-2, keepdims=True)
    return feat_mean, feat_std


def adain(feat: T) -> T:
    feat_mean, feat_std = calc_mean_std(feat)
    feat_style_mean = expand_first(feat_mean)
    feat_style_std = expand_first(feat_std)
    feat = (feat - feat_mean) / feat_std
    feat = feat * feat_style_std + feat_style_mean
    return feat

def swapping_attention(key, value, chunk_size=2):
    chunk_length = key.size()[0] // chunk_size  # [text-condition, null-condition]
    reference_image_index = [0] * chunk_length  # [0 0 0 0 0]
    key = rearrange(key, "(b f) d c -> b f d c", f=chunk_length)
    key = key[:, reference_image_index]  # ref to all
    key = rearrange(key, "b f d c -> (b f) d c")
    value = rearrange(value, "(b f) d c -> b f d c", f=chunk_length)
    value = value[:, reference_image_index]  # ref to all
    value = rearrange(value, "b f d c -> (b f) d c")

    return key, value


def exists(val):
    return val is not None

def default(val, d):
    if exists(val):
        return val
    return d


class StyleAlignedArgs:
    def __init__(self, share_attn: str) -> None:
        self.adain_keys = "k" in share_attn
        self.adain_values = "v" in share_attn
        self.adain_queries = "q" in share_attn

    share_attention: bool = True
    adain_queries: bool = True
    adain_keys: bool = True
    adain_values: bool = True


def expand_first(
    feat: T,
    scale=1.0,
) -> T:

    b = feat.shape[0]
    feat_style = torch.stack((feat[0], feat[b // 2])).unsqueeze(1)
    if scale == 1:
        feat_style = feat_style.expand(2, b // 2, *feat.shape[1:])
    else:
        feat_style = feat_style.repeat(1, b // 2, 1, 1, 1)
        feat_style = torch.cat([feat_style[:, :1], scale * feat_style[:, 1:]], dim=1)
    return feat_style.reshape(*feat.shape)


def concat_first(feat: T, dim=2, scale=1.0) -> T:

    feat_style = expand_first(feat, scale=scale)
    return torch.cat((feat, feat_style), dim=dim)


def calc_mean_std(feat, eps: float = 1e-5) -> "tuple[T, T]":
    feat_std = (feat.var(dim=-2, keepdims=True) + eps).sqrt()
    feat_mean = feat.mean(dim=-2, keepdims=True)
    return feat_mean, feat_std

def adain(feat: T) -> T:
    feat_mean, feat_std = calc_mean_std(feat)
    feat_style_mean = expand_first(feat_mean)
    feat_style_std = expand_first(feat_std)
    feat = (feat - feat_mean) / feat_std
    feat = feat * feat_style_std + feat_style_mean
    return feat


class SharedAttentionProcessor:
    def __init__(self, args: StyleAlignedArgs, scale: float):
        self.args = args
        self.scale = scale

    def __call__(self, q, k, v, extra_options):
        if self.args.adain_queries:
            q = adain(q)
        if self.args.adain_keys:
            k = adain(k)
        if self.args.adain_values:
            v = adain(v)
        if self.args.share_attention:
            k = concat_first(k, -2, scale=self.scale)
            v = concat_first(v, -2)

        return q, k, v


def get_norm_layers(
    layer: nn.Module,
    norm_layers_: "dict[str, list[Union[nn.GroupNorm, nn.LayerNorm]]]",
    share_layer_norm: bool,
    share_group_norm: bool,
):
    if isinstance(layer, nn.LayerNorm) and share_layer_norm:
        norm_layers_["layer"].append(layer)
    if isinstance(layer, nn.GroupNorm) and share_group_norm:
        norm_layers_["group"].append(layer)
    else:
        for child_layer in layer.children():
            get_norm_layers(
                child_layer, norm_layers_, share_layer_norm, share_group_norm
            )


def register_norm_forward(
    norm_layer: Union[nn.GroupNorm, nn.LayerNorm],
) -> Union[nn.GroupNorm, nn.LayerNorm]:
    if not hasattr(norm_layer, "orig_forward"):
        setattr(norm_layer, "orig_forward", norm_layer.forward)
    orig_forward = norm_layer.orig_forward

    def forward_(hidden_states: T) -> T:
        n = hidden_states.shape[-2]
        hidden_states = concat_first(hidden_states, dim=-2)
        hidden_states = orig_forward(hidden_states)  # type: ignore
        return hidden_states[..., :n, :]

    norm_layer.forward = forward_  # type: ignore
    return norm_layer


def register_shared_norm(
    model: ModelPatcher,
    share_group_norm: bool = True,
    share_layer_norm: bool = True,
):
    norm_layers = {"group": [], "layer": []}
    get_norm_layers(model.model, norm_layers, share_layer_norm, share_group_norm)
    print(
        f"Patching {len(norm_layers['group'])} group norms, {len(norm_layers['layer'])} layer norms."
    )
    return [register_norm_forward(layer) for layer in norm_layers["group"]] + [
        register_norm_forward(layer) for layer in norm_layers["layer"]
    ]


class StyleAlignedBatchAlign:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "share_norm": (SHARE_NORM_OPTIONS,),
                "share_attn": (SHARE_ATTN_OPTIONS,),
                "scale": ("FLOAT", {"default": 1, "min": 0, "max": 1.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "Style"
    def patch(
        self,
        model: ModelPatcher,
        share_norm: str,
        share_attn: str,
        scale: float,
    ):
        m = model.clone()
        share_group_norm = share_norm in ["group", "both"]
        share_layer_norm = share_norm in ["layer", "both"]
        register_shared_norm(model, share_group_norm, share_layer_norm)
        args = StyleAlignedArgs(share_attn)
        m.set_model_attn1_patch(SharedAttentionProcessor(args, scale))
        return (m,)


class chx_Ksampler_VisualStyle:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "context": ("RUN_CONTEXT",),
                "reference_image": ("IMAGE",),
                "reference_image_text": ("STRING", {"multiline": True}),
                "positive_prompt": ("STRING", {"default": "", "multiline": True}),
                "enabled": ("BOOLEAN", {"default": True}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "share_norm": (SHARE_NORM_OPTIONS,),
                "share_attn": (SHARE_ATTN_OPTIONS,),
                "scale": ("FLOAT", {"default": 1, "min": 0, "max": 1.0, "step": 0.1}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096, "step": 2}),
            }
        }

    RETURN_TYPES = ("RUN_CONTEXT", "IMAGE")
    RETURN_NAMES = ("context", "images")
    
    CATEGORY = "Apt_Preset/ksampler"

    FUNCTION = "run"
    
    def run(
        self, 
        reference_image, 
        reference_image_text,
        positive_prompt,
        seed,
        denoise,
        enabled,
        share_norm: str,
        share_attn: str,
        scale: float,
        batch_size=1,
        context=None, 
    ):
        # 从 context 中获取必要的值
        vae = context.get("vae")
        negative = context.get("negative")
        clip = context.get("clip")
        model2: comfy.model_patcher.ModelPatcher = context.get("model")  # 从 context 中获取 model2

        # 检查 model2 是否为 ModelPatcher 类型
        if not isinstance(model2, comfy.model_patcher.ModelPatcher):
            raise TypeError(f"Expected model2 to be of type ModelPatcher, got {type(model2)}")

        # 编码参考图像文本
        tokens = clip.tokenize(reference_image_text)
        cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
        reference_image_prompt = [[cond, {"pooled_output": pooled}]]

        # 重复 reference_image 以匹配 batch_size
        reference_image = reference_image.repeat(((batch_size + 1) // 2, 1, 1, 1))

        # 编码参考图像为 latent
        self.model = model2  # 使用 model2
        reference_latent = vae.encode(reference_image[:, :, :, :3])
        
        # 修改 CrossAttention 层
        for n, m in model2.model.diffusion_model.named_modules():  # 使用 model2
            if m.__class__.__name__ == "CrossAttention":
                processor = VisualStyleProcessor(m, enabled=enabled)
                setattr(m, 'forward', processor.visual_style_forward)

        # 编码 positive_prompt
        positive, = CLIPTextEncode().encode(clip, positive_prompt)
        positive = reference_image_prompt + positive
        negative = negative * 2 

        # 创建 latent 张量
        latents = torch.zeros_like(reference_latent) 
        latents = torch.cat([latents] * 2)

        # 设置第一张图片的 latent 为 reference_latent
        latents[::2] = reference_latent  # 设为默认 denoise1=1

        # 创建 denoise_mask
        denoise_mask = torch.ones_like(latents)[:, :1, ...] 
        denoise_mask[0] = 0.  # 第一张图片不需要 denoise

        # 修改 model2
        model_patched = StyleAlignedBatchAlign().patch(model2, share_norm, share_attn, scale)  # 使用 model2
        model2 = model_patched[0]  # 从 tuple 中提取 ModelPatcher 对象

        # 创建 latent 字典
        latent = {"samples": latents, "noise_mask": denoise_mask}

        # 获取采样参数
        steps = context.get("steps", None)
        cfg = context.get("cfg", None)
        sampler = context.get("sampler", None)
        scheduler = context.get("scheduler", None)
        
        # 调用 common_ksampler
        latent = common_ksampler(model2, seed, steps, cfg, sampler, scheduler,  # 使用 model2
                                positive, negative, latent, denoise=denoise)[0]

        # 解码 latent 为图像
        output_image = VAEDecode().decode(vae, latent)[0]
        
        # 更新 context
        context = new_context(context, positive=positive, negative=negative, model=model2, latent=latent)  # 使用 model2

        # 返回 context 和 output_image
        return (context, output_image)



