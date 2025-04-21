
# 内置库

import math
# 第三方库
import torch
from PIL import Image, ImageFilter
# 本地库
import comfy
import comfy.sd
import comfy.utils
import comfy.samplers
import comfy.sample
import node_helpers
import latent_preview
from nodes import CLIPTextEncode, VAEDecode, VAEEncode
import math
import nodes
from comfy.k_diffusion import sampling as k_diffusion_sampling
from comfy import samplers
import comfy.model_management as mm
from comfy_extras.nodes_custom_sampler import Noise_EmptyNoise, Noise_RandomNoise
from comfy_extras import nodes_custom_sampler

# 相对导入
from ..def_unit import *




def calculate_sigmas(model, sampler, scheduler, steps):
    discard_penultimate_sigma = False
    if sampler in ['dpm_2', 'dpm_2_ancestral', 'uni_pc', 'uni_pc_bh2']:
        steps += 1
        discard_penultimate_sigma = True

    if scheduler.startswith('AYS'):
        sigmas = nodes.NODE_CLASS_MAPPINGS['AlignYourStepsScheduler']().get_sigmas(scheduler[4:], steps, denoise=1.0)[0]
    elif scheduler.startswith('GITS[coeff='):
        sigmas = nodes.NODE_CLASS_MAPPINGS['GITSScheduler']().get_sigmas(float(scheduler[11:-1]), steps, denoise=1.0)[0]
    elif scheduler == 'LTXV[default]':
        sigmas = nodes.NODE_CLASS_MAPPINGS['LTXVScheduler']().get_sigmas(20, 2.05, 0.95, True, 0.1)[0]
    else:
        sigmas = samplers.calculate_sigmas(model.get_model_object("model_sampling"), scheduler, steps)

    if discard_penultimate_sigma:
        sigmas = torch.cat([sigmas[:-2], sigmas[-1:]])
    return sigmas


def get_noise_sampler(x, cpu, total_sigmas, **kwargs):
    if 'extra_args' in kwargs and 'seed' in kwargs['extra_args']:
        sigma_min, sigma_max = total_sigmas[total_sigmas > 0].min(), total_sigmas.max()
        seed = kwargs['extra_args'].get("seed", None)
        return k_diffusion_sampling.BrownianTreeNoiseSampler(x, sigma_min, sigma_max, seed=seed, cpu=cpu)
    return None


def ksampler(sampler_name, total_sigmas, extra_options={}, inpaint_options={}):
    if sampler_name == "dpmpp_sde":
        def sample_dpmpp_sde(model, x, sigmas, **kwargs):
            noise_sampler = get_noise_sampler(x, True, total_sigmas, **kwargs)
            if noise_sampler is not None:
                kwargs['noise_sampler'] = noise_sampler

            return k_diffusion_sampling.sample_dpmpp_sde(model, x, sigmas, **kwargs)

        sampler_function = sample_dpmpp_sde

    elif sampler_name == "dpmpp_sde_gpu":
        def sample_dpmpp_sde(model, x, sigmas, **kwargs):
            noise_sampler = get_noise_sampler(x, False, total_sigmas, **kwargs)
            if noise_sampler is not None:
                kwargs['noise_sampler'] = noise_sampler

            return k_diffusion_sampling.sample_dpmpp_sde_gpu(model, x, sigmas, **kwargs)

        sampler_function = sample_dpmpp_sde

    elif sampler_name == "dpmpp_2m_sde":
        def sample_dpmpp_sde(model, x, sigmas, **kwargs):
            noise_sampler = get_noise_sampler(x, True, total_sigmas, **kwargs)
            if noise_sampler is not None:
                kwargs['noise_sampler'] = noise_sampler

            return k_diffusion_sampling.sample_dpmpp_2m_sde(model, x, sigmas, **kwargs)

        sampler_function = sample_dpmpp_sde

    elif sampler_name == "dpmpp_2m_sde_gpu":
        def sample_dpmpp_sde(model, x, sigmas, **kwargs):
            noise_sampler = get_noise_sampler(x, False, total_sigmas, **kwargs)
            if noise_sampler is not None:
                kwargs['noise_sampler'] = noise_sampler

            return k_diffusion_sampling.sample_dpmpp_2m_sde_gpu(model, x, sigmas, **kwargs)

        sampler_function = sample_dpmpp_sde

    elif sampler_name == "dpmpp_3m_sde":
        def sample_dpmpp_sde(model, x, sigmas, **kwargs):
            noise_sampler = get_noise_sampler(x, True, total_sigmas, **kwargs)
            if noise_sampler is not None:
                kwargs['noise_sampler'] = noise_sampler

            return k_diffusion_sampling.sample_dpmpp_3m_sde(model, x, sigmas, **kwargs)

        sampler_function = sample_dpmpp_sde

    elif sampler_name == "dpmpp_3m_sde_gpu":
        def sample_dpmpp_sde(model, x, sigmas, **kwargs):
            noise_sampler = get_noise_sampler(x, False, total_sigmas, **kwargs)
            if noise_sampler is not None:
                kwargs['noise_sampler'] = noise_sampler

            return k_diffusion_sampling.sample_dpmpp_3m_sde_gpu(model, x, sigmas, **kwargs)

        sampler_function = sample_dpmpp_sde

    else:
        return comfy.samplers.sampler_object(sampler_name)

    return samplers.KSAMPLER(sampler_function, extra_options, inpaint_options)


# modified version of SamplerCustom.sample
def sample_with_custom_noise(model, add_noise, noise_seed, cfg, positive, negative, sampler, sigmas, latent_image, noise=None, callback=None):
    latent = latent_image
    latent_image = latent["samples"]

    if hasattr(comfy.sample, 'fix_empty_latent_channels'):
        latent_image = comfy.sample.fix_empty_latent_channels(model, latent_image)

    out = latent.copy()
    out['samples'] = latent_image

    if noise is None:
        if not add_noise:
            noise = Noise_EmptyNoise().generate_noise(out)
        else:
            noise = Noise_RandomNoise(noise_seed).generate_noise(out)

    noise_mask = None
    if "noise_mask" in latent:
        noise_mask = latent["noise_mask"]

    x0_output = {}
    preview_callback = latent_preview.prepare_callback(model, sigmas.shape[-1] - 1, x0_output)

    if callback is not None:
        def touched_callback(step, x0, x, total_steps):
            callback(step, x0, x, total_steps)
            preview_callback(step, x0, x, total_steps)
    else:
        touched_callback = preview_callback

    disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED

    device = mm.get_torch_device()

    noise = noise.to(device)
    latent_image = latent_image.to(device)
    if noise_mask is not None:
        noise_mask = noise_mask.to(device)

    if negative != 'NegativePlaceholder':
        # This way is incompatible with Advanced ControlNet, yet.
        # guider = comfy.samplers.CFGGuider(model)
        # guider.set_conds(positive, negative)
        # guider.set_cfg(cfg)
        samples = comfy.sample.sample_custom(model, noise, cfg, sampler, sigmas, positive, negative, latent_image,
                                             noise_mask=noise_mask, callback=touched_callback,
                                             disable_pbar=disable_pbar, seed=noise_seed)
    else:
        guider = nodes_custom_sampler.Guider_Basic(model)
        positive = node_helpers.conditioning_set_values(positive, {"guidance": cfg})
        guider.set_conds(positive)
        samples = guider.sample(noise, latent_image, sampler, sigmas, denoise_mask=noise_mask, callback=touched_callback, disable_pbar=disable_pbar, seed=noise_seed)

    samples = samples.to(comfy.model_management.intermediate_device())

    out["samples"] = samples
    if "x0" in x0_output:
        out_denoised = latent.copy()
        out_denoised["samples"] = model.model.process_latent_out(x0_output["x0"].cpu())
    else:
        out_denoised = out
    return out, out_denoised


# When sampling one step at a time, it mitigates the problem. (especially for _sde series samplers)
def separated_sample(model, add_noise, seed, steps, cfg, sampler_name, scheduler, positive, negative,
                     latent_image, start_at_step, end_at_step, return_with_leftover_noise, sigma_ratio=1.0, sampler_opt=None, noise=None, callback=None, scheduler_func=None):

    if scheduler_func is not None:
        total_sigmas = scheduler_func(model, sampler_name, steps)
    else:
        if sampler_opt is None:
            total_sigmas = calculate_sigmas(model, sampler_name, scheduler, steps)
        else:
            total_sigmas = calculate_sigmas(model, "", scheduler, steps)

    sigmas = total_sigmas

    if end_at_step is not None and end_at_step < (len(total_sigmas) - 1):
        sigmas = total_sigmas[:end_at_step + 1]
        if not return_with_leftover_noise:
            sigmas[-1] = 0

    if start_at_step is not None:
        if start_at_step < (len(sigmas) - 1):
            sigmas = sigmas[start_at_step:] * sigma_ratio
        else:
            if latent_image is not None:
                return latent_image
            else:
                return {'samples': torch.zeros_like(noise)}

    if sampler_opt is None:
        impact_sampler = ksampler(sampler_name, total_sigmas)
    else:
        impact_sampler = sampler_opt

    if len(sigmas) == 0 or (len(sigmas) == 1 and sigmas[0] == 0):
        return latent_image
    
    res = sample_with_custom_noise(model, add_noise, seed, cfg, positive, negative, impact_sampler, sigmas, latent_image, noise=noise, callback=callback)

    if return_with_leftover_noise:
        return res[0]
    else:
        return res[1]


def impact_sample(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise=1.0, sigma_ratio=1.0, sampler_opt=None, noise=None, scheduler_func=None):
    advanced_steps = math.floor(steps / denoise)
    start_at_step = advanced_steps - steps
    end_at_step = start_at_step + steps
    return separated_sample(model, True, seed, advanced_steps, cfg, sampler_name, scheduler, positive, negative, latent_image,
                            start_at_step, end_at_step, False, scheduler_func=scheduler_func)


def ksampler_wrapper(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise,
                     refiner_ratio=None, refiner_model=None, refiner_clip=None, refiner_positive=None, refiner_negative=None, sigma_factor=1.0, noise=None, scheduler_func=None):

    if refiner_ratio is None or refiner_model is None or refiner_clip is None or refiner_positive is None or refiner_negative is None:
        # Use separated_sample instead of KSampler for `AYS scheduler`
        # refined_latent = nodes.KSampler().sample(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise * sigma_factor)[0]

        advanced_steps = math.floor(steps / denoise)
        start_at_step = advanced_steps - steps
        end_at_step = start_at_step + steps

        refined_latent = separated_sample(model, True, seed, advanced_steps, cfg, sampler_name, scheduler,
                                          positive, negative, latent_image, start_at_step, end_at_step, False,
                                          sigma_ratio=sigma_factor, noise=noise, scheduler_func=scheduler_func)
    else:
        advanced_steps = math.floor(steps / denoise)
        start_at_step = advanced_steps - steps
        end_at_step = start_at_step + math.floor(steps * (1.0 - refiner_ratio))

        # print(f"pre: {start_at_step} .. {end_at_step} / {advanced_steps}")
        temp_latent = separated_sample(model, True, seed, advanced_steps, cfg, sampler_name, scheduler,
                                       positive, negative, latent_image, start_at_step, end_at_step, True,
                                       sigma_ratio=sigma_factor, noise=noise, scheduler_func=scheduler_func)

        if 'noise_mask' in latent_image:
            # noise_latent = \
            #     impact_sampling.separated_sample(refiner_model, "enable", seed, advanced_steps, cfg, sampler_name,
            #                                      scheduler, refiner_positive, refiner_negative, latent_image, end_at_step,
            #                                      end_at_step, "enable")

            latent_compositor = nodes.NODE_CLASS_MAPPINGS['LatentCompositeMasked']()
            temp_latent = latent_compositor.composite(latent_image, temp_latent, 0, 0, False, latent_image['noise_mask'])[0]

        # print(f"post: {end_at_step} .. {advanced_steps + 1} / {advanced_steps}")
        refined_latent = separated_sample(refiner_model, False, seed, advanced_steps, cfg, sampler_name, scheduler,
                                          refiner_positive, refiner_negative, temp_latent, end_at_step, advanced_steps + 1, False,
                                          sigma_ratio=sigma_factor, scheduler_func=scheduler_func)

    return refined_latent



class KSamplerWrapper:
    params = None

    def __init__(self, model, seed, steps, cfg, sampler_name, scheduler, positive, negative, denoise, scheduler_func=None):
        self.params = model, seed, steps, cfg, sampler_name, scheduler, positive, negative, denoise
        self.scheduler_func = scheduler_func

    def sample(self, latent_image, hook=None):
        model, seed, steps, cfg, sampler_name, scheduler, positive, negative, denoise = self.params

        if hook is not None:
            model, seed, steps, cfg, sampler_name, scheduler, positive, negative, upscaled_latent, denoise = \
                hook.pre_ksample(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise)

        return impact_sample(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise, scheduler_func=self.scheduler_func)


class KSamplerProvider:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "tooltip": "Random seed to use for generating CPU noise for sampling."}),
                                "steps": ("INT", {"default": 20, "min": 1, "max": 10000, "tooltip": "total sampling steps"}),
                                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "tooltip": "classifier free guidance value"}),
                                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "The amount of noise to remove. This amount is the noise added at the start, and the higher it is, the more the input latent will be modified before being returned."}),
                                "basic_pipe": ("BASIC_PIPE", {"tooltip": "basic_pipe input for sampling"})
                            },
                "optional": {
                    "scheduler_func_opt": ("SCHEDULER_FUNC", {"tooltip": "[OPTIONAL] Noise schedule generation function. If this is set, the scheduler widget will be ignored."}),
                    }
                }

    OUTPUT_TOOLTIPS = ("sampler wrapper. (Can be used when generating a regional_prompt.)",)

    RETURN_TYPES = ("KSAMPLER",)
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Sampler"

    @staticmethod
    def doit(seed, steps, cfg, sampler_name, scheduler, denoise, basic_pipe, scheduler_func_opt=None):
        model, _, _, positive, negative = basic_pipe
        sampler = KSamplerWrapper(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, denoise, scheduler_func=scheduler_func_opt)
        return (sampler, )




class TwoSamplersForMask:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                     "latent_image": ("LATENT", {"tooltip": "input latent image"}),
                     "base_sampler": ("KSAMPLER", {"tooltip": "Sampler to apply to the region outside the mask."}),
                     "mask_sampler": ("KSAMPLER", {"tooltip": "Sampler to apply to the masked region."}),
                     "mask": ("MASK", {"tooltip": "region mask"})
                     },
                }

    OUTPUT_TOOLTIPS = ("result latent", )

    RETURN_TYPES = ("LATENT", )
    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Sampler"

    @staticmethod
    def doit(latent_image, base_sampler, mask_sampler, mask):
        inv_mask = torch.where(mask != 1.0, torch.tensor(1.0), torch.tensor(0.0))

        latent_image['noise_mask'] = inv_mask
        new_latent_image = base_sampler.sample(latent_image)

        new_latent_image['noise_mask'] = mask
        new_latent_image = mask_sampler.sample(new_latent_image)

        del new_latent_image['noise_mask']

        return (new_latent_image, )

#--------------------------------chx_Ksampler_dual_area----------------------


class chx_Ksampler_dual_area:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "context": ("RUN_CONTEXT",),
                
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "image_cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, }),
                "mask_cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, }),
                "image_denoise": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "mask_denoise": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            
            "optional": {

            "model_img": ("MODEL",),
            "model_mask": ("MODEL",),
            "prompt_img": ("CONDITIONING",),
            "prompt_mask": ("CONDITIONING",),
            "image": ("IMAGE",),
            "mask": ("MASK",),
            "smoothness":("INT", {"default": 1,  "min":0, "max": 150, "step": 1,"display": "slider"}),
            "image_pos": ("STRING", {"default": "","multiline": True}),
            "mask_pos": ("STRING", {"default": "","multiline": True}),

            },
            
            
        }

    RETURN_TYPES = ("RUN_CONTEXT", "IMAGE","MASK")
    RETURN_NAMES = ("context", "image","mask")
    FUNCTION = "sample"
    CATEGORY = "Apt_Preset/ksampler"


    def sample(self, seed, image, image_denoise, mask, mask_denoise, image_cfg, mask_cfg, model_img=None,model_mask=None, prompt_img=None,prompt_mask=None,
                context=None,scheduler_func_opt=None, mask_pos="",image_pos="",smoothness=1 ):

        vae = context.get("vae")
        steps = context.get("steps")
        sampler = context.get("sampler")
        scheduler = context.get("scheduler")
        negative= context.get("negative")

        clip = context.get("clip")
        
        if image_pos != None and mask_pos != '':       
            positive1, =CLIPTextEncode().encode(clip, image_pos)  
        
        if mask_pos != None and mask_pos != '':       
            positive2, =CLIPTextEncode().encode(clip, mask_pos)  


        if model_img!= None:
            model1=model_img
        else:
            model1=context.get("model")
        
        if model_mask!= None:
            model2=model_mask
        else:
            model2=context.get("model")

        if prompt_img!= None:
            positive1=prompt_img
        else:
            positive1=context.get("positive")

        if prompt_mask!= None:
            positive2=prompt_mask
        else:
            positive2=context.get("positive")

        latent = VAEEncode().encode(vae, image)[0]

        image_sampler = KSamplerWrapper(model1, seed, steps, image_cfg, sampler, scheduler, positive1, negative, image_denoise, scheduler_func=scheduler_func_opt)
        mask_sampler  = KSamplerWrapper(model2, seed, steps, mask_cfg, sampler, scheduler, positive2, negative, mask_denoise, scheduler_func=scheduler_func_opt)
        
        
        #----------------------add smooth
        mask=tensor2pil(mask)
        feathered_image = mask.filter(ImageFilter.GaussianBlur(smoothness))
        mask=pil2tensor(feathered_image)
            
        
        latent = TwoSamplersForMask().doit(latent, image_sampler, mask_sampler, mask)[0]


        output_image = VAEDecode().decode(vae, latent)[0]
        context = new_context(context,  latent=latent, images=output_image,)
        
        
        return  (context, output_image, mask)







