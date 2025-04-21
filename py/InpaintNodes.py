
#region---------------依赖----------------


from ..def_unit import new_context
from typing import Any
import numpy as np
import torch
import torch.nn.functional as F

from torch import Tensor
from tqdm import trange

from comfy.model_patcher import ModelPatcher
from comfy.model_base import BaseModel
from comfy.model_management import cast_to_device, get_torch_device
import comfy.utils
import comfy.lora
import folder_paths
import nodes
import kornia.filters
from comfy_extras.nodes_differential_diffusion import DifferentialDiffusion



def mask_unsqueeze(mask: Tensor):
    if len(mask.shape) == 3:  # BHW -> B1HW
        mask = mask.unsqueeze(1)
    elif len(mask.shape) == 2:  # HW -> B1HW
        mask = mask.unsqueeze(0).unsqueeze(0)
    return mask


def to_torch(image: Tensor, mask: Tensor | None = None):
    if len(image.shape) == 3:
        image = image.unsqueeze(0)
    image = image.permute(0, 3, 1, 2)  # BHWC -> BCHW
    if mask is not None:
        mask = mask_unsqueeze(mask)
    if image.shape[2:] != mask.shape[2:]:
        raise ValueError(
            f"Image and mask must be the same size. {image.shape[2:]} != {mask.shape[2:]}"
        )
    return image, mask


def to_comfy(image: Tensor):
    return image.permute(0, 2, 3, 1)  # BCHW -> BHWC


def mask_floor(mask: Tensor, threshold: float = 0.99):
    return (mask >= threshold).to(mask.dtype)


# torch pad does not support padding greater than image size with "reflect" mode
def pad_reflect_once(x: Tensor, original_padding: tuple[int, int, int, int]):
    _, _, h, w = x.shape
    padding = np.array(original_padding)
    size = np.array([w, w, h, h])

    initial_padding = np.minimum(padding, size - 1)
    additional_padding = padding - initial_padding

    x = F.pad(x, tuple(initial_padding), mode="reflect")
    if np.any(additional_padding > 0):
        x = F.pad(x, tuple(additional_padding), mode="constant")
    return x


def resize_square(image: Tensor, mask: Tensor, size: int):
    _, _, h, w = image.shape
    pad_w, pad_h, prev_size = 0, 0, w
    if w == size and h == size:
        return image, mask, (pad_w, pad_h, prev_size)

    if w < h:
        pad_w = h - w
        prev_size = h
    elif h < w:
        pad_h = w - h
        prev_size = w
    image = pad_reflect_once(image, (0, pad_w, 0, pad_h))
    mask = pad_reflect_once(mask, (0, pad_w, 0, pad_h))

    if image.shape[-1] != size:
        image = F.interpolate(image, size=size, mode="nearest-exact")
        mask = F.interpolate(mask, size=size, mode="nearest-exact")

    return image, mask, (pad_w, pad_h, prev_size)


def undo_resize_square(image: Tensor, original_size: tuple[int, int, int]):
    _, _, h, w = image.shape
    pad_w, pad_h, prev_size = original_size
    if prev_size != w or prev_size != h:
        image = F.interpolate(image, size=prev_size, mode="bilinear")
    return image[:, :, 0 : prev_size - pad_h, 0 : prev_size - pad_w]


def gaussian_blur(image: Tensor, radius: int, sigma: float = 0):
    c = image.shape[-3]
    if sigma <= 0:
        sigma = 0.3 * (radius - 1) + 0.8
    return kornia.filters.gaussian_blur2d(image, (radius, radius), (sigma, sigma))


def binary_erosion(mask: Tensor, radius: int):
    kernel = torch.ones(1, 1, radius * 2 + 1, radius * 2 + 1, device=mask.device)
    mask = F.pad(mask, (radius, radius, radius, radius), mode="constant", value=1)
    mask = F.conv2d(mask, kernel, groups=1)
    mask = (mask == kernel.numel()).to(mask.dtype)
    return mask


def binary_dilation(mask: Tensor, radius: int):
    kernel = torch.ones(1, radius * 2 + 1, device=mask.device)
    mask = kornia.filters.filter2d_separable(mask, kernel, kernel, border_type="constant")
    mask = (mask > 0).to(mask.dtype)
    return mask


def make_odd(x):
    if x > 0 and x % 2 == 0:
        return x + 1
    return x


















class InpaintHead(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.head = torch.nn.Parameter(torch.empty(size=(320, 5, 3, 3), device="cpu"))

    def __call__(self, x):
        x = F.pad(x, (1, 1, 1, 1), "replicate")
        return F.conv2d(x, weight=self.head)


def load_fooocus_patch(lora: dict, to_load: dict):
    patch_dict = {}
    loaded_keys = set()
    for key in to_load.values():
        if value := lora.get(key, None):
            patch_dict[key] = ("fooocus", value)
            loaded_keys.add(key)

    not_loaded = sum(1 for x in lora if x not in loaded_keys)
    if not_loaded > 0:
        print(
            f"[ApplyFooocusInpaint] {len(loaded_keys)} Lora keys loaded, {not_loaded} remaining keys not found in model."
        )
    return patch_dict


if not hasattr(comfy.lora, "calculate_weight") and hasattr(ModelPatcher, "calculate_weight"):
    too_old_msg = "comfyui-inpaint-nodes requires a newer version of ComfyUI (v0.1.1 or later), please update!"
    raise RuntimeError(too_old_msg)


original_calculate_weight = comfy.lora.calculate_weight
injected_model_patcher_calculate_weight = False


def calculate_weight_patched(patches, weight, key, intermediate_dtype=torch.float32):
    remaining = []

    for p in patches:
        alpha = p[0]
        v = p[1]

        is_fooocus_patch = isinstance(v, tuple) and len(v) == 2 and v[0] == "fooocus"
        if not is_fooocus_patch:
            remaining.append(p)
            continue

        if alpha != 0.0:
            v = v[1]
            w1 = cast_to_device(v[0], weight.device, torch.float32)
            if w1.shape == weight.shape:
                w_min = cast_to_device(v[1], weight.device, torch.float32)
                w_max = cast_to_device(v[2], weight.device, torch.float32)
                w1 = (w1 / 255.0) * (w_max - w_min) + w_min
                weight += alpha * cast_to_device(w1, weight.device, weight.dtype)
            else:
                print(
                    f"[ApplyFooocusInpaint] Shape mismatch {key}, weight not merged ({w1.shape} != {weight.shape})"
                )

    if len(remaining) > 0:
        return original_calculate_weight(remaining, weight, key, intermediate_dtype)
    return weight


def inject_patched_calculate_weight():
    global injected_model_patcher_calculate_weight
    if not injected_model_patcher_calculate_weight:
        print(
            "[comfyui-inpaint-nodes] Injecting patched comfy.model_patcher.ModelPatcher.calculate_weight"
        )
        comfy.lora.calculate_weight = calculate_weight_patched
        injected_model_patcher_calculate_weight = True







#endregion------------------------old--------------------------------



#-------------------new-----------------------------------------------





import os






MODELS_DIR = os.path.join(folder_paths.models_dir, "inpaint")
if "inpaint" not in folder_paths.folder_names_and_paths:
    current_paths = [MODELS_DIR]
else:
    current_paths, _ = folder_paths.folder_names_and_paths["inpaint"]
folder_paths.folder_names_and_paths["inpaint"] = (
    current_paths,
    folder_paths.supported_pt_extensions,
)


class pre_inpaint:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                    "context": ("RUN_CONTEXT",),
                "pixels": ("IMAGE",),
                "mask": ("MASK",),
                "head": (folder_paths.get_filename_list("inpaint"), {"default": "fooocus_inpaint_head.pth"}),
                "patch": (folder_paths.get_filename_list("inpaint"), {"default": "inpaint_v26.fooocus.patch"}),
            }
        }

    RETURN_TYPES = ("RUN_CONTEXT",)
    RETURN_NAMES = ("context",)
    CATEGORY = "Apt_Preset/ksampler"
    FUNCTION = "patch"

    _inpaint_head_feature: Tensor | None = None
    _inpaint_block: Tensor | None = None


    def _input_block_patch(self, h: Tensor, transformer_options: dict):
        if transformer_options["block"][1] == 0:
            if self._inpaint_block is None or self._inpaint_block.shape != h.shape:
                assert self._inpaint_head_feature is not None
                batch = h.shape[0] // self._inpaint_head_feature.shape[0]
                self._inpaint_block = self._inpaint_head_feature.to(h).repeat(batch, 1, 1, 1)
            h = h + self._inpaint_block
        return h


    def patch(self, head: str, patch: str,  pixels, mask,context):

        model: ModelPatcher=context.get("model")
        positive=context.get("positive")
        negative=context.get("negative")
        vae=context.get("vae")



        if isinstance(head, list):
            if len(head) > 0:
                head = head[0]  # 选择列表中的第一个元素
            else:
                raise ValueError("The 'head' list is empty.")
        if isinstance(patch, list):
            if len(patch) > 0:
                patch = patch[0]  # 选择列表中的第一个元素
            else:
                raise ValueError("The 'patch' list is empty.")

        # 加载文件的逻辑
        head_file = folder_paths.get_full_path("inpaint", head)
        inpaint_head_model = InpaintHead()
        sd = torch.load(head_file, map_location="cpu", weights_only=True)
        inpaint_head_model.load_state_dict(sd)

        patch_file = folder_paths.get_full_path("inpaint", patch)
        inpaint_lora = comfy.utils.load_torch_file(patch_file, safe_load=True)

        positive, negative, latent = nodes.InpaintModelConditioning().encode(positive, negative, pixels, vae, mask)
        latent0=latent
        
        latent = dict(samples=positive[0][1]["concat_latent_image"], noise_mask=latent["noise_mask"].round())

        base_model: BaseModel = model.model
        latent_pixels = base_model.process_latent_in(latent["samples"])
        noise_mask = latent["noise_mask"].round()

        latent_mask = F.max_pool2d(noise_mask, (8, 8)).round().to(latent_pixels)

        feed = torch.cat([latent_mask, latent_pixels], dim=1)
        inpaint_head_model.to(device=feed.device, dtype=feed.dtype)
        self._inpaint_head_feature = inpaint_head_model(feed)
        self._inpaint_block = None

        lora_keys = comfy.lora.model_lora_keys_unet(model.model, {})
        lora_keys.update({x: x for x in base_model.state_dict().keys()})
        loaded_lora = load_fooocus_patch(inpaint_lora, lora_keys)

        m = model.clone()
        m.set_model_input_block_patch(self._input_block_patch)
        patched = m.add_patches(loaded_lora, 1.0)

        not_patched_count = sum(1 for x in loaded_lora if x not in patched)
        if not_patched_count > 0:
            print(f"[ApplyFooocusInpaint] Failed to patch {not_patched_count} keys")

        inject_patched_calculate_weight()
        
        model = DifferentialDiffusion().apply(m)
        
        context = new_context(context, latent=latent0, positive=positive,negative=negative,vae=vae,model=m,)
        return (context,  )       

