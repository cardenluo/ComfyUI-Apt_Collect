
from torch import Tensor
from math import cos, sin, pi
from random import random
import torch
import comfy.model_management
from typing import Callable
import comfy.latent_formats
import comfy.sample
from dataclasses import dataclass
import torch
import torch.nn as nn
from comfy.model_patcher import ModelPatcher
import comfy.ops
from typing import Union
import latent_preview
import comfy.utils
import comfy.sd
import comfy.utils
import comfy.lora
import comfy.samplers
import comfy.controlnet 
import node_helpers
T = torch.Tensor






#region---------------style
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
    """
    Expand the first element so it has the same shape as the rest of the batch.
    """
    b = feat.shape[0]
    feat_style = torch.stack((feat[0], feat[b // 2])).unsqueeze(1)
    if scale == 1:
        feat_style = feat_style.expand(2, b // 2, *feat.shape[1:])
    else:
        feat_style = feat_style.repeat(1, b // 2, 1, 1, 1)
        feat_style = torch.cat([feat_style[:, :1], scale * feat_style[:, 1:]], dim=1)
    return feat_style.reshape(*feat.shape)


def concat_first(feat: T, dim=2, scale=1.0) -> T:
    """
    concat the the feature and the style feature expanded above
    """
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


SHARE_NORM_OPTIONS = ["both", "group", "layer", "disabled"]
SHARE_ATTN_OPTIONS = ["q+k", "q+k+v", "disabled"]

#endregion


class model_diff_inpaint:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", ),
                "positive": ("CONDITIONING", ),
                "negative": ("CONDITIONING", ),
                "vae": ("VAE", ),
                "pixels": ("IMAGE", ),
                "mask": ("MASK", ),
                "noise_mask": ("BOOLEAN", {"default": True, }),

            }
        }

    RETURN_TYPES = ( "MODEL", "CONDITIONING", "CONDITIONING", "LATENT",)
    RETURN_NAMES = ("model", "positive", "negative", "latent", )
    FUNCTION = "process"

    CATEGORY = "Apt_Preset/tool"

    def process(self, positive, negative, pixels, vae, mask, noise_mask, model):
        
        
        # To_inpaint functionality
        x = (pixels.shape[1] // 8) * 8
        y = (pixels.shape[2] // 8) * 8
        mask = torch.nn.functional.interpolate(mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])), size=(pixels.shape[1], pixels.shape[2]), mode="bilinear")

        orig_pixels = pixels
        pixels = orig_pixels.clone()
        if pixels.shape[1] != x or pixels.shape[2] != y:
            x_offset = (pixels.shape[1] % 8) // 2
            y_offset = (pixels.shape[2] % 8) // 2
            pixels = pixels[:, x_offset:x + x_offset, y_offset:y + y_offset, :]
            mask = mask[:, :, x_offset:x + x_offset, y_offset:y + y_offset]

        m = (1.0 - mask.round()).squeeze(1)
        for i in range(3):
            pixels[:, :, :, i] -= 0.5
            pixels[:, :, :, i] *= m
            pixels[:, :, :, i] += 0.5
        concat_latent = vae.encode(pixels)
        orig_latent = vae.encode(orig_pixels)

        out_latent = {"samples": orig_latent}
        if noise_mask:
            out_latent["noise_mask"] = mask

        out = []
        for conditioning in [positive, negative]:
            c = node_helpers.conditioning_set_values(conditioning, {"concat_latent_image": concat_latent, "concat_mask": mask})
            out.append(c)

        # To_Differ functionality
        model = model.clone()
        model.set_model_denoise_mask_function(self.forward)

        return (out[0], out[1], out_latent, model)

    def forward(self, sigma: torch.Tensor, denoise_mask: torch.Tensor, extra_options: dict):
        model = extra_options["model"]
        step_sigmas = extra_options["sigmas"]
        sigma_to = model.inner_model.model_sampling.sigma_min
        if step_sigmas[-1] > sigma_to:
            sigma_to = step_sigmas[-1]
        sigma_from = step_sigmas[0]

        ts_from = model.inner_model.model_sampling.timestep(sigma_from)
        ts_to = model.inner_model.model_sampling.timestep(sigma_to)
        current_ts = model.inner_model.model_sampling.timestep(sigma[0])

        threshold = (current_ts - ts_to) / (ts_from - ts_to)

        return (denoise_mask >= threshold).to(denoise_mask.dtype)



#region----模型调色---------


def apply_scaling(
    alg: str,
    current_step: int,
    total_steps: int,
    bri: float,
    con: float,
    sat: float,
    r: float,
    g: float,
    b: float,
):

    if alg == "Flat":
        mod = 1.0

    else:
        ratio = float(current_step / total_steps)
        rad = ratio * pi / 2

        match alg:
            case "Cos":
                mod = cos(rad)
            case "Sin":
                mod = sin(rad)
            case "1 - Cos":
                mod = 1.0 - cos(rad)
            case "1 - Sin":
                mod = 1.0 - sin(rad)
            case _:
                mod = 1.0

    return (bri * mod, con * mod, (sat - 1.0) * mod + 1.0, r * mod, g * mod, b * mod)


def RGB_2_CbCr(r: float, g: float, b: float) -> tuple[float, float]:
    """Convert RGB channels into YCbCr for SDXL"""
    cb = -0.17 * r - 0.33 * g + 0.5 * b
    cr = 0.5 * r - 0.42 * g - 0.08 * b

    return cb, cr


class NoiseMethods:
    @staticmethod
    def get_delta(latent: Tensor) -> Tensor:
        mean = torch.mean(latent)
        return torch.sub(latent, mean)

    @staticmethod
    def to_abs(latent: Tensor) -> Tensor:
        return torch.abs(latent)

    @staticmethod
    def zeros(latent: Tensor) -> Tensor:
        return torch.zeros_like(latent)

    @staticmethod
    def ones(latent: Tensor) -> Tensor:
        return torch.ones_like(latent)

    @staticmethod
    def gaussian_noise(latent: Tensor) -> Tensor:
        return torch.rand_like(latent)

    @staticmethod
    def normal_noise(latent: Tensor) -> Tensor:
        return torch.randn_like(latent)

    @staticmethod
    def multires_noise(latent: Tensor, use_zero: bool, iterations: int = 8, discount: float = 0.4):
        noise = NoiseMethods.zeros(latent) if use_zero else NoiseMethods.ones(latent)
        batchSize, c, w, h = noise.shape

        device = comfy.model_management.get_torch_device()
        upsampler = torch.nn.Upsample(size=(w, h), mode="bilinear").to(device)

        for b in range(batchSize):
            for i in range(iterations):
                r = random() * 2 + 2

                wn = max(1, int(w / (r**i)))
                hn = max(1, int(h / (r**i)))

                noise[b] += (upsampler(torch.randn(1, c, hn, wn).to(device)) * discount**i)[0]

                if wn == 1 or hn == 1:
                    break

        return noise / noise.std()


def normalize_tensor(x: Tensor, r):
    ratio = r / max(abs(float(x.min())), abs(float(x.max())))
    x *= max(ratio, 1.0)
    return x


class Model_adjust_color:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "alt": ("BOOLEAN", {"default": False}),
                "brightness": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.05}),
                "contrast": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.05}),
                "saturation": ("FLOAT", {"default": 1.0, "min": 0.05, "max": 3.0, "step": 0.05}),
                "r": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.05}),
                "g": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.05}),
                "b": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.05}),
                "method": (
                    ["Straight", "Straight Abs.", "Cross", "Cross Abs.", "Ones", "N.Random", "U.Random", "Multi-Res", "Multi-Res Abs."],
                    {"default": "Straight Abs."},
                ),
                "scaling": (["Flat", "Cos", "Sin", "1 - Cos", "1 - Sin"], {"default": "Flat"}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "hook"
    CATEGORY = "Apt_Preset/tool"

    PARAMS_NAME = "vectorscope_cc"

    def hook(
        self,
        model: ModelPatcher,
        alt: bool,
        brightness: float,
        contrast: float,
        saturation: float,
        r: float,
        g: float,
        b: float,
        method: str,
        scaling: str,
    ):
        m = model.clone()
        latent_format = type(model.model.latent_format)

        m.model_options[self.PARAMS_NAME] = (
            latent_format,
            alt,
            brightness,
            contrast,
            saturation,
            r,
            g,
            b,
            method,
            scaling,
        )

        return (m,)

    @staticmethod
    def callback(params: tuple):
        def _callback(step: int, x0: Tensor, x: Tensor, total_steps: int):
            (
                latent_format,
                alt,
                brightness,
                contrast,
                saturation,
                r,
                g,
                b,
                method,
                scaling,
            ) = params

            brightness /= total_steps
            contrast /= total_steps
            saturation = pow(saturation, 1.0 / total_steps)
            r /= total_steps
            g /= total_steps
            b /= total_steps

            latent, cross_latent = (x, x0) if alt else (x0, x)

            if "Straight" in method:
                target = latent.detach().clone()
            elif "Cross" in method:
                target = cross_latent.detach().clone()
            elif "Multi-Res" in method:
                target = NoiseMethods.multires_noise(latent, "Abs" in method)
            elif method == "Ones":
                target = NoiseMethods.ones(latent)
            elif method == "N.Random":
                target = NoiseMethods.normal_noise(latent)
            elif method == "U.Random":
                target = NoiseMethods.gaussian_noise(latent)
            else:
                raise ValueError

            if "Abs" in method:
                target = NoiseMethods.to_abs(target)

            brightness, contrast, saturation, r, g, b = apply_scaling(scaling, step, total_steps, brightness, contrast, saturation, r, g, b)
            bs, _, _, _ = latent.shape

            match latent_format:
                case comfy.latent_formats.SD15:
                    for b in range(bs):
                        # Brightness
                        latent[b][0] += target[b][0] * brightness
                        # Contrast
                        latent[b][0] += NoiseMethods.get_delta(latent[b][0]) * contrast

                        # RGB
                        latent[b][2] -= target[b][2] * r
                        latent[b][1] += target[b][1] * g
                        latent[b][3] -= target[b][3] * b

                        # Saturation
                        latent[b][2] *= saturation
                        latent[b][1] *= saturation
                        latent[b][3] *= saturation

                case comfy.latent_formats.SDXL:
                    cb, cr = RGB_2_CbCr(r, g, b)

                    for b in range(bs):
                        # Brightness
                        latent[b][0] += target[b][0] * brightness
                        # Contrast
                        latent[b][0] += NoiseMethods.get_delta(latent[b][0]) * contrast

                        # CbCr
                        latent[b][1] -= target[b][1] * cr
                        latent[b][2] -= target[b][2] * cb

                        # Saturation
                        latent[b][1] *= saturation
                        latent[b][2] *= saturation

        return _callback


class CallbackManager:
    def __init__(self):
        self.callbacks: dict[str, tuple[Callable[[tuple], Callable], int]] = {}

    def hijack_samplers(self):
        import comfy.sample
        if not hasattr(comfy.sample, "sample_original"):
            comfy.sample.sample_original = comfy.sample.sample
            comfy.sample.sample = self.sample_wrapper(comfy.sample.sample_original)
        if not hasattr(comfy.sample, "sample_custom_original"):
            comfy.sample.sample_custom_original = comfy.sample.sample_custom
            comfy.sample.sample_custom = self.sample_wrapper(comfy.sample.sample_custom_original)

    def register_callback(self, params_name: str, callback_func: Callable[[tuple], Callable], priority: int):
        self.callbacks[params_name] = callback_func, priority

    def sample_wrapper(self, original_sample: Callable):
        def sample(*args, **kwargs):
            model = args[0]

            original_cb = kwargs["callback"]
            original_cb_priority = 1000

            callbacks = []

            def add_cb(cb, priority):
                if cb is not None:
                    callbacks.append((priority, cb))

            for params_name, (cb_wrapper, priority) in self.callbacks.items():
                params = model.model_options.get(params_name, None)
                if params:
                    cb = cb_wrapper(params)
                    add_cb(cb, priority)
            add_cb(original_cb, original_cb_priority)

            callbacks.sort()

            def callback(step: int, x0: torch.Tensor, x: torch.Tensor, total_steps: int):
                for _, cb in callbacks:
                    cb(step, x0, x, total_steps)

            kwargs["callback"] = callback
            return original_sample(*args, **kwargs)

        return sample



cb_manager = CallbackManager()
cb_manager.hijack_samplers()
cb_manager.register_callback(Model_adjust_color.PARAMS_NAME, Model_adjust_color.callback, 210)




#endregion----模型调色------------