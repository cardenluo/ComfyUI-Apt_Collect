import numpy as np
import torch
import os
import comfy.sd
import comfy.utils
import comfy.lora
import folder_paths


from comfy.cli_args import args
import comfy.samplers
import comfy.controlnet 
import nodes



import comfy.ops
import torch.nn as nn
from torch import nn
from torch.nn import functional as F
from einops import rearrange
from comfy.ldm.modules.attention import optimized_attention
from comfy.ldm.modules.diffusionmodules.mmdit import (RMSNorm,JointBlock,)
import math
from diffusers.models.embeddings import Timesteps, TimestepEmbedding



#region----------动能组-------condi_IPAdapterSD3-----------------------------------


class AdaLayerNorm(nn.Module):


    def __init__(self, embedding_dim: int, time_embedding_dim=None, mode="normal"):
        super().__init__()

        self.silu = nn.SiLU()
        num_params_dict = dict(
            zero=6,
            normal=2,
        )
        num_params = num_params_dict[mode]
        self.linear = nn.Linear(
            time_embedding_dim or embedding_dim, num_params * embedding_dim, bias=True
        )
        self.norm = nn.LayerNorm(embedding_dim, elementwise_affine=False, eps=1e-6)
        self.mode = mode

    def forward(
        self,
        x,
        hidden_dtype=None,
        emb=None,
    ):
        emb = self.linear(self.silu(emb))
        if self.mode == "normal":
            shift_msa, scale_msa = emb.chunk(2, dim=1)
            x = self.norm(x) * (1 + scale_msa[:, None]) + shift_msa[:, None]
            return x

        elif self.mode == "zero":
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = emb.chunk(
                6, dim=1
            )
            x = self.norm(x) * (1 + scale_msa[:, None]) + shift_msa[:, None]
            return x, gate_msa, shift_mlp, scale_mlp, gate_mlp


class IPAttnProcessor(nn.Module):

    def __init__(
        self,
        hidden_size=None,
        cross_attention_dim=None,
        ip_hidden_states_dim=None,
        ip_encoder_hidden_states_dim=None,
        head_dim=None,
        timesteps_emb_dim=1280,
    ):
        super().__init__()

        self.norm_ip = AdaLayerNorm(
            ip_hidden_states_dim, time_embedding_dim=timesteps_emb_dim
        )
        self.to_k_ip = nn.Linear(ip_hidden_states_dim, hidden_size, bias=False)
        self.to_v_ip = nn.Linear(ip_hidden_states_dim, hidden_size, bias=False)
        self.norm_q = RMSNorm(head_dim, 1e-6)
        self.norm_k = RMSNorm(head_dim, 1e-6)
        self.norm_ip_k = RMSNorm(head_dim, 1e-6)

    def forward(
        self,
        ip_hidden_states,
        img_query,
        img_key=None,
        img_value=None,
        t_emb=None,
        n_heads=1,
    ):
        if ip_hidden_states is None:
            return None

        if not hasattr(self, "to_k_ip") or not hasattr(self, "to_v_ip"):
            return None

        # norm ip input
        norm_ip_hidden_states = self.norm_ip(ip_hidden_states, emb=t_emb)

        # to k and v
        ip_key = self.to_k_ip(norm_ip_hidden_states)
        ip_value = self.to_v_ip(norm_ip_hidden_states)

        # reshape
        img_query = rearrange(img_query, "b l (h d) -> b h l d", h=n_heads)
        img_key = rearrange(img_key, "b l (h d) -> b h l d", h=n_heads)
        # note that the image is in a different shape: b l h d
        # so we transpose to b h l d
        # or do we have to transpose here?
        img_value = torch.transpose(img_value, 1, 2)
        ip_key = rearrange(ip_key, "b l (h d) -> b h l d", h=n_heads)
        ip_value = rearrange(ip_value, "b l (h d) -> b h l d", h=n_heads)

        # norm
        img_query = self.norm_q(img_query)
        img_key = self.norm_k(img_key)
        ip_key = self.norm_ip_k(ip_key)

        # cat img
        key = torch.cat([img_key, ip_key], dim=2)
        value = torch.cat([img_value, ip_value], dim=2)

        #
        ip_hidden_states = F.scaled_dot_product_attention(
            img_query, key, value, dropout_p=0.0, is_causal=False
        )
        ip_hidden_states = rearrange(ip_hidden_states, "b h l d -> b l (h d)")
        ip_hidden_states = ip_hidden_states.to(img_query.dtype)
        return ip_hidden_states


class JointBlockIPWrapper:
    """To be used as a patch_replace with Comfy"""

    def __init__(
        self,
        original_block: JointBlock,
        adapter: IPAttnProcessor,
        ip_options=None,
    ):
        self.original_block = original_block
        self.adapter = adapter
        if ip_options is None:
            ip_options = {}
        self.ip_options = ip_options

    def block_mixing(self, context, x, context_block, x_block, c):
        """
        Comes from mmdit.py. Modified to add ipadapter attention.
        """
        context_qkv, context_intermediates = context_block.pre_attention(context, c)

        if x_block.x_block_self_attn:
            x_qkv, x_qkv2, x_intermediates = x_block.pre_attention_x(x, c)
        else:
            x_qkv, x_intermediates = x_block.pre_attention(x, c)

        qkv = tuple(torch.cat((context_qkv[j], x_qkv[j]), dim=1) for j in range(3))

        attn = optimized_attention(
            qkv[0],
            qkv[1],
            qkv[2],
            heads=x_block.attn.num_heads,
        )
        context_attn, x_attn = (
            attn[:, : context_qkv[0].shape[1]],
            attn[:, context_qkv[0].shape[1] :],
        )
        # if the current timestep is not in the ipadapter enabling range, then the resampler wasn't run
        # and the hidden states will be None
        if (
            self.ip_options["hidden_states"] is not None
            and self.ip_options["t_emb"] is not None
        ):
            # IP-Adapter
            ip_attn = self.adapter(
                self.ip_options["hidden_states"],
                *x_qkv,
                self.ip_options["t_emb"],
                x_block.attn.num_heads,
            )
            x_attn = x_attn + ip_attn * self.ip_options["weight"]

        # Everything else is unchanged
        if not context_block.pre_only:
            context = context_block.post_attention(context_attn, *context_intermediates)

        else:
            context = None
        if x_block.x_block_self_attn:
            attn2 = optimized_attention(
                x_qkv2[0],
                x_qkv2[1],
                x_qkv2[2],
                heads=x_block.attn2.num_heads,
            )
            x = x_block.post_attention_x(x_attn, attn2, *x_intermediates)
        else:
            x = x_block.post_attention(x_attn, *x_intermediates)
        return context, x

    def __call__(self, args, _):

        #   ```
        c, x = self.block_mixing(
            args["txt"],
            args["img"],
            self.original_block.context_block,
            self.original_block.x_block,
            c=args["vec"],
        )
        return {"txt": c, "img": x}



import math
import torch
import torch.nn as nn
from diffusers.models.embeddings import Timesteps, TimestepEmbedding


# FFN
def FeedForward(dim, mult=4):
    inner_dim = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim, bias=False),
        nn.GELU(),
        nn.Linear(inner_dim, dim, bias=False),
    )


def reshape_tensor(x, heads):
    bs, length, width = x.shape
    # (bs, length, width) --> (bs, length, n_heads, dim_per_head)
    x = x.view(bs, length, heads, -1)
    # (bs, length, n_heads, dim_per_head) --> (bs, n_heads, length, dim_per_head)
    x = x.transpose(1, 2)
    # (bs, n_heads, length, dim_per_head) --> (bs*n_heads, length, dim_per_head)
    x = x.reshape(bs, heads, length, -1)
    return x


class PerceiverAttention(nn.Module):
    def __init__(self, *, dim, dim_head=64, heads=8):
        super().__init__()
        self.scale = dim_head**-0.5
        self.dim_head = dim_head
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x, latents, shift=None, scale=None):
        """
        Args:
            x (torch.Tensor): image features
                shape (b, n1, D)
            latent (torch.Tensor): latent features
                shape (b, n2, D)
        """
        x = self.norm1(x)
        latents = self.norm2(latents)

        if shift is not None and scale is not None:
            latents = latents * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

        b, l, _ = latents.shape

        q = self.to_q(latents)
        kv_input = torch.cat((x, latents), dim=-2)
        k, v = self.to_kv(kv_input).chunk(2, dim=-1)

        q = reshape_tensor(q, self.heads)
        k = reshape_tensor(k, self.heads)
        v = reshape_tensor(v, self.heads)

        # attention
        scale = 1 / math.sqrt(math.sqrt(self.dim_head))
        weight = (q * scale) @ (k * scale).transpose(
            -2, -1
        )  # More stable with f16 than dividing afterwards
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        out = weight @ v

        out = out.permute(0, 2, 1, 3).reshape(b, l, -1)

        return self.to_out(out)


class Resampler(nn.Module):
    def __init__(
        self,
        dim=1024,
        depth=8,
        dim_head=64,
        heads=16,
        num_queries=8,
        embedding_dim=768,
        output_dim=1024,
        ff_mult=4,
        *args,
        **kwargs,
    ):
        super().__init__()

        self.latents = nn.Parameter(torch.randn(1, num_queries, dim) / dim**0.5)

        self.proj_in = nn.Linear(embedding_dim, dim)

        self.proj_out = nn.Linear(dim, output_dim)
        self.norm_out = nn.LayerNorm(output_dim)

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PerceiverAttention(dim=dim, dim_head=dim_head, heads=heads),
                        FeedForward(dim=dim, mult=ff_mult),
                    ]
                )
            )

    def forward(self, x):

        latents = self.latents.repeat(x.size(0), 1, 1)

        x = self.proj_in(x)

        for attn, ff in self.layers:
            latents = attn(x, latents) + latents
            latents = ff(latents) + latents

        latents = self.proj_out(latents)
        return self.norm_out(latents)


class TimeResampler(nn.Module):
    def __init__(
        self,
        dim=1024,
        depth=8,
        dim_head=64,
        heads=16,
        num_queries=8,
        embedding_dim=768,
        output_dim=1024,
        ff_mult=4,
        timestep_in_dim=320,
        timestep_flip_sin_to_cos=True,
        timestep_freq_shift=0,
    ):
        super().__init__()

        self.latents = nn.Parameter(torch.randn(1, num_queries, dim) / dim**0.5)

        self.proj_in = nn.Linear(embedding_dim, dim)

        self.proj_out = nn.Linear(dim, output_dim)
        self.norm_out = nn.LayerNorm(output_dim)

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        # msa
                        PerceiverAttention(dim=dim, dim_head=dim_head, heads=heads),
                        # ff
                        FeedForward(dim=dim, mult=ff_mult),
                        # adaLN
                        nn.Sequential(nn.SiLU(), nn.Linear(dim, 4 * dim, bias=True)),
                    ]
                )
            )

        # time
        self.time_proj = Timesteps(
            timestep_in_dim, timestep_flip_sin_to_cos, timestep_freq_shift
        )
        self.time_embedding = TimestepEmbedding(timestep_in_dim, dim, act_fn="silu")


    def forward(self, x, timestep, need_temb=False):
        timestep_emb = self.embedding_time(x, timestep)  # bs, dim

        latents = self.latents.repeat(x.size(0), 1, 1)

        x = self.proj_in(x)
        x = x + timestep_emb[:, None]

        for attn, ff, adaLN_modulation in self.layers:
            shift_msa, scale_msa, shift_mlp, scale_mlp = adaLN_modulation(
                timestep_emb
            ).chunk(4, dim=1)
            latents = attn(x, latents, shift_msa, scale_msa) + latents

            res = latents
            for idx_ff in range(len(ff)):
                layer_ff = ff[idx_ff]
                latents = layer_ff(latents)
                if idx_ff == 0 and isinstance(layer_ff, nn.LayerNorm):  # adaLN
                    latents = latents * (
                        1 + scale_mlp.unsqueeze(1)
                    ) + shift_mlp.unsqueeze(1)
            latents = latents + res

            # latents = ff(latents) + latents

        latents = self.proj_out(latents)
        latents = self.norm_out(latents)

        if need_temb:
            return latents, timestep_emb
        else:
            return latents

    def embedding_time(self, sample, timestep):

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            # This would be a good case for the `match` statement (Python 3.10+)
            is_mps = sample.device.type == "mps"
            if isinstance(timestep, float):
                dtype = torch.float32 if is_mps else torch.float64
            else:
                dtype = torch.int32 if is_mps else torch.int64
            timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])

        t_emb = self.time_proj(timesteps)

        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=sample.dtype)

        emb = self.time_embedding(t_emb, None)
        return emb




























MODELS_DIR = os.path.join(folder_paths.models_dir, "ipadapter")
if "ipadapter" not in folder_paths.folder_names_and_paths:
    current_paths = [MODELS_DIR]
else:
    current_paths, _ = folder_paths.folder_names_and_paths["ipadapter"]
folder_paths.folder_names_and_paths["ipadapter"] = (
    current_paths,
    folder_paths.supported_pt_extensions,
)


def patch(
    patcher,
    ip_procs,
    resampler: TimeResampler,
    clip_embeds,
    weight=1.0,
    start=0.0,
    end=1.0,
):
    """
    Patches a model_sampler to add the ipadapter
    """
    mmdit = patcher.model.diffusion_model
    timestep_schedule_max = patcher.model.model_config.sampling_settings.get(
        "timesteps", 1000
    )
    # hook the model's forward function
    # so that when it gets called, we can grab the timestep and send it to the resampler
    ip_options = {
        "hidden_states": None,
        "t_emb": None,
        "weight": weight,
    }

    def ddit_wrapper(forward, args):
        # this is between 0 and 1, so the adapters can calculate start_point and end_point
        # actually, do we need to get the sigma value instead?
        t_percent = 1 - args["timestep"].flatten()[0].cpu().item()
        if start <= t_percent <= end:
            batch_size = args["input"].shape[0] // len(args["cond_or_uncond"])
            # if we're only doing cond or only doing uncond, only pass one of them through the resampler
            embeds = clip_embeds[args["cond_or_uncond"]]
            # slight efficiency optimization todo: pass the embeds through and then afterwards
            # repeat to the batch size
            embeds = torch.repeat_interleave(embeds, batch_size, dim=0)
            # the resampler wants between 0 and MAX_STEPS
            timestep = args["timestep"] * timestep_schedule_max
            image_emb, t_emb = resampler(embeds, timestep, need_temb=True)
            # these will need to be accessible to the IPAdapters
            ip_options["hidden_states"] = image_emb
            ip_options["t_emb"] = t_emb
        else:
            ip_options["hidden_states"] = None
            ip_options["t_emb"] = None

        return forward(args["input"], args["timestep"], **args["c"])

    patcher.set_model_unet_function_wrapper(ddit_wrapper)
    # patch each dit block
    for i, block in enumerate(mmdit.joint_blocks):
        wrapper = JointBlockIPWrapper(block, ip_procs[i], ip_options)
        patcher.set_model_patch_replace(wrapper, "dit", "double_block", i)


class SD3IPAdapter:
    def __init__(self, checkpoint: str, device):
        self.device = device
        # load the checkpoint right away
        self.state_dict = torch.load(
            os.path.join(MODELS_DIR, checkpoint),
            map_location=self.device,
            weights_only=True,
        )
        # todo: infer some of the params from the checkpoint instead of hardcoded
        self.resampler = TimeResampler(
            dim=1280,
            depth=4,
            dim_head=64,
            heads=20,
            num_queries=64,
            embedding_dim=1152,
            output_dim=2432,
            ff_mult=4,
            timestep_in_dim=320,
            timestep_flip_sin_to_cos=True,
            timestep_freq_shift=0,
        )
        self.resampler.eval()
        self.resampler.to(self.device, dtype=torch.float16)
        self.resampler.load_state_dict(self.state_dict["image_proj"])

        # now we'll create the attention processors
        # ip_adapter.keys looks like [0.proj, 0.to_k, ..., 1.proj, 1.to_k, ...]
        n_procs = len(
            set(x.split(".")[0] for x in self.state_dict["ip_adapter"].keys())
        )
        self.procs = torch.nn.ModuleList(
            [
                # this is hardcoded for SD3.5L
                IPAttnProcessor(
                    hidden_size=2432,
                    cross_attention_dim=2432,
                    ip_hidden_states_dim=2432,
                    ip_encoder_hidden_states_dim=2432,
                    head_dim=64,
                    timesteps_emb_dim=1280,
                ).to(self.device, dtype=torch.float16)
                for _ in range(n_procs)
            ]
        )
        self.procs.load_state_dict(self.state_dict["ip_adapter"])















class Stack_IPA_SD3:
    def __init__(self):
        self.unfold_batch = False

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "ipadapter": (folder_paths.get_filename_list("ipadapter"), {"default": "SD3.5-Large-IP-Adapter.bin"}),
                "image_embed": (folder_paths.get_filename_list("clip_vision"), {"default": "sigclip_vision_patch14_384.safetensors"}),
                "weight": ("FLOAT",{"default": 1.0, "min": -1.0, "max": 5.0, "step": 0.05},),
            },
            "optional": {
                "mask": ("MASK", ),
            }
        }

    RETURN_TYPES = ("IPA3_STACK",)
    RETURN_NAMES = ("ipa3_stack",)
    FUNCTION = "ipa_stack"
    CATEGORY = "Apt_Preset/stack"

    def ipa_stack(self, image, ipadapter, image_embed, weight, mask=None,):
        # 初始化ipa_list
        ipa_list = []
        # 将当前IPA的相关信息打包成一个元组，并添加到ipa_list中
        ipa_info = (
            image,
            ipadapter,
            image_embed,
            weight,
            mask,
        )
        ipa_list.append(ipa_info)

        return (ipa_list,)


class Apply_IPA_SD3:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "ipa3_stack": ("IPA3_STACK",),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "apply_ipa_stack"
    CATEGORY = "Apt_Preset/stack"

    def apply_ipa_stack(self, model, ipa3_stack):
        if not ipa3_stack:
            raise ValueError("ipa3_stack 不能为空")

        # 初始化变量
        work_model = model.clone()

        # 遍历 ipa_stack 中的每个 IPA 配置
        for ipa_info in ipa3_stack:
            (
                image,
                ipadapter_name,
                image_embed_name,
                weight,
                mask,
            ) = ipa_info

            # 创建IPAdapter实例
            ipadapter = SD3IPAdapter(ipadapter_name, "cuda")
            
            # 加载并处理图像嵌入
            clip_path = folder_paths.get_full_path_or_raise("clip_vision", image_embed_name)
            image_embed = comfy.clip_vision.load(clip_path)
            image_embed = image_embed.encode_image(image, crop=True)
            
            # 设置模型
            image_embed = image_embed.penultimate_hidden_states
            embeds = torch.cat([image_embed, torch.zeros_like(image_embed)], dim=0).to(
                ipadapter.device, dtype=torch.float16
            )
            
            # 应用patch
            patch(
                work_model,
                ipadapter.procs,
                ipadapter.resampler,
                embeds,
                weight=weight,
                start=0,
                end=1,
            )

        return (work_model,)



class IPA_dapterSD3LOAD:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "ipadapter": (folder_paths.get_filename_list("ipadapter"), {"default": "SD3.5-Large-IP-Adapter.bin"}),

                "image_embed": (folder_paths.get_filename_list("clip_vision"), {"default": "sigclip_vision_patch14_384.safetensors"}), 
                "image": ("IMAGE",),
                "weight": ("FLOAT",{"default": 1.0, "min": -1.0, "max": 5.0, "step": 0.05},),
            },
            "optional": { 
                "mask": ("MASK", ), 
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    
    FUNCTION = "apply_ipadapter"
    CATEGORY = "Apt_Preset/IPA"

    def apply_ipadapter( self, model, ipadapter, image, image_embed, weight, mask=None ):
        
        ipadapter = SD3IPAdapter(ipadapter, "cuda")
        
        clip_path = folder_paths.get_full_path_or_raise("clip_vision", image_embed)
        image_embed = comfy.clip_vision.load(clip_path)

        image_embed = image_embed.encode_image(image, crop=True)




        # set model
        new_model = model.clone()
        # add uncond embedding
        image_embed = image_embed.penultimate_hidden_states
        embeds = torch.cat([image_embed, torch.zeros_like(image_embed)], dim=0).to(
            ipadapter.device, dtype=torch.float16
        )
        patch(
            new_model,
            ipadapter.procs,
            ipadapter.resampler,
            embeds,
            weight=weight,
            start=0,
            end=1,
        )
        return (new_model,)

#endregion

