import nodes
import node_helpers
import torch
import comfy.model_management
import comfy.utils
import comfy.latent_formats
import comfy.clip_vision
import folder_paths
from ..def_unit import *



def clip_vision_output_encode(clip_vision_name, image):
    clip_path = folder_paths.get_full_path_or_raise("clip_vision", clip_vision_name)
    clip_vision = comfy.clip_vision.load(clip_path)
    output = clip_vision.encode_image(image, crop=True)
    return (output,)




class WanImageToVideo:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"positive": ("CONDITIONING", ),
                             "negative": ("CONDITIONING", ),
                             "vae": ("VAE", ),
                             "width": ("INT", {"default": 832, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 16}),
                             "height": ("INT", {"default": 480, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 16}),
                             "length": ("INT", {"default": 81, "min": 1, "max": nodes.MAX_RESOLUTION, "step": 4}),
                             "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
                },
                "optional": {"clip_vision_output": ("CLIP_VISION_OUTPUT", ),
                             "start_image": ("IMAGE", ),
                }}

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "negative", "latent")
    FUNCTION = "encode"

    CATEGORY = "conditioning/video_models"

    def encode(self, positive, negative, vae, width, height, length, batch_size, start_image=None, clip_vision_output=None):
        latent = torch.zeros([batch_size, 16, ((length - 1) // 4) + 1, height // 8, width // 8], device=comfy.model_management.intermediate_device())
        if start_image is not None:
            start_image = comfy.utils.common_upscale(start_image[:length].movedim(-1, 1), width, height, "bilinear", "center").movedim(1, -1)
            image = torch.ones((length, height, width, start_image.shape[-1]), device=start_image.device, dtype=start_image.dtype) * 0.5
            image[:start_image.shape[0]] = start_image

            concat_latent_image = vae.encode(image[:, :, :, :3])
            mask = torch.ones((1, 1, latent.shape[2], concat_latent_image.shape[-2], concat_latent_image.shape[-1]), device=start_image.device, dtype=start_image.dtype)
            mask[:, :, :((start_image.shape[0] - 1) // 4) + 1] = 0.0

            positive = node_helpers.conditioning_set_values(positive, {"concat_latent_image": concat_latent_image, "concat_mask": mask})
            negative = node_helpers.conditioning_set_values(negative, {"concat_latent_image": concat_latent_image, "concat_mask": mask})

        if clip_vision_output is not None:
            positive = node_helpers.conditioning_set_values(positive, {"clip_vision_output": clip_vision_output})
            negative = node_helpers.conditioning_set_values(negative, {"clip_vision_output": clip_vision_output})

        out_latent = {}
        out_latent["samples"] = latent
        return (positive, negative, out_latent)


class WanFunControlToVideo:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"positive": ("CONDITIONING", ),
                             "negative": ("CONDITIONING", ),
                             "vae": ("VAE", ),
                             "width": ("INT", {"default": 832, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 16}),
                             "height": ("INT", {"default": 480, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 16}),
                             "length": ("INT", {"default": 81, "min": 1, "max": nodes.MAX_RESOLUTION, "step": 4}),
                             "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
                },
                "optional": {"clip_vision_output": ("CLIP_VISION_OUTPUT", ),
                             "start_image": ("IMAGE", ),
                             "control_video": ("IMAGE", ),
                }}

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "negative", "latent")
    FUNCTION = "encode"

    CATEGORY = "conditioning/video_models"

    def encode(self, positive, negative, vae, width, height, length, batch_size, start_image=None, clip_vision_output=None, control_video=None):
        latent = torch.zeros([batch_size, 16, ((length - 1) // 4) + 1, height // 8, width // 8], device=comfy.model_management.intermediate_device())
        concat_latent = torch.zeros([batch_size, 16, ((length - 1) // 4) + 1, height // 8, width // 8], device=comfy.model_management.intermediate_device())
        concat_latent = comfy.latent_formats.Wan21().process_out(concat_latent)
        concat_latent = concat_latent.repeat(1, 2, 1, 1, 1)

        if start_image is not None:
            start_image = comfy.utils.common_upscale(start_image[:length].movedim(-1, 1), width, height, "bilinear", "center").movedim(1, -1)
            concat_latent_image = vae.encode(start_image[:, :, :, :3])
            concat_latent[:,16:,:concat_latent_image.shape[2]] = concat_latent_image[:,:,:concat_latent.shape[2]]

        if control_video is not None:
            control_video = comfy.utils.common_upscale(control_video[:length].movedim(-1, 1), width, height, "bilinear", "center").movedim(1, -1)
            concat_latent_image = vae.encode(control_video[:, :, :, :3])
            concat_latent[:,:16,:concat_latent_image.shape[2]] = concat_latent_image[:,:,:concat_latent.shape[2]]

        positive = node_helpers.conditioning_set_values(positive, {"concat_latent_image": concat_latent})
        negative = node_helpers.conditioning_set_values(negative, {"concat_latent_image": concat_latent})

        if clip_vision_output is not None:
            positive = node_helpers.conditioning_set_values(positive, {"clip_vision_output": clip_vision_output})
            negative = node_helpers.conditioning_set_values(negative, {"clip_vision_output": clip_vision_output})

        out_latent = {}
        out_latent["samples"] = latent
        return (positive, negative, out_latent)


class WanFirstLastFrameToVideo:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"positive": ("CONDITIONING", ),
                             "negative": ("CONDITIONING", ),
                             "vae": ("VAE", ),
                             "width": ("INT", {"default": 832, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 16}),
                             "height": ("INT", {"default": 480, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 16}),
                             "length": ("INT", {"default": 81, "min": 1, "max": nodes.MAX_RESOLUTION, "step": 4}),
                             "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
                },
                "optional": {"clip_vision_start_image": ("CLIP_VISION_OUTPUT", ),
                             "clip_vision_end_image": ("CLIP_VISION_OUTPUT", ),
                             "start_image": ("IMAGE", ),
                             "end_image": ("IMAGE", ),
                }}

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "negative", "latent")
    FUNCTION = "encode"

    CATEGORY = "conditioning/video_models"

    def encode(self, positive, negative, vae, width, height, length, batch_size, start_image=None, end_image=None, clip_vision_start_image=None, clip_vision_end_image=None):
        if vae is None:
            raise Exception("Missing VAE model.")
        latent = torch.zeros([batch_size, 16, ((length - 1) // 4) + 1, height // 8, width // 8], device=comfy.model_management.intermediate_device())
        if start_image is not None:
            start_image = comfy.utils.common_upscale(start_image[:length].movedim(-1, 1), width, height, "bilinear", "center").movedim(1, -1)
        if end_image is not None:
            end_image = comfy.utils.common_upscale(end_image[-length:].movedim(-1, 1), width, height, "bilinear", "center").movedim(1, -1)

        image = torch.ones((length, height, width, 3)) * 0.5
        mask = torch.ones((1, 1, latent.shape[2] * 4, latent.shape[-2], latent.shape[-1]))

        if start_image is not None:
            image[:start_image.shape[0]] = start_image
            mask[:, :, :start_image.shape[0] + 3] = 0.0

        if end_image is not None:
            image[-end_image.shape[0]:] = end_image
            mask[:, :, -end_image.shape[0]:] = 0.0

        concat_latent_image = vae.encode(image[:, :, :, :3])
        mask = mask.view(1, mask.shape[2] // 4, 4, mask.shape[3], mask.shape[4]).transpose(1, 2)
        positive = node_helpers.conditioning_set_values(positive, {"concat_latent_image": concat_latent_image, "concat_mask": mask})
        negative = node_helpers.conditioning_set_values(negative, {"concat_latent_image": concat_latent_image, "concat_mask": mask})

        if clip_vision_start_image is not None:
            clip_vision_output = clip_vision_start_image

        if clip_vision_end_image is not None:
            if clip_vision_output is not None:
                states = torch.cat([clip_vision_output.penultimate_hidden_states, clip_vision_end_image.penultimate_hidden_states], dim=-2)
                clip_vision_output = comfy.clip_vision.Output()
                clip_vision_output.penultimate_hidden_states = states
            else:
                clip_vision_output = clip_vision_end_image

        if clip_vision_output is not None:
            positive = node_helpers.conditioning_set_values(positive, {"clip_vision_output": clip_vision_output})
            negative = node_helpers.conditioning_set_values(negative, {"clip_vision_output": clip_vision_output})

        out_latent = {}
        out_latent["samples"] = latent
        return (positive, negative, out_latent)


class WanFunInpaintToVideo:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"positive": ("CONDITIONING", ),
                             "negative": ("CONDITIONING", ),
                             "vae": ("VAE", ),
                             "width": ("INT", {"default": 832, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 16}),
                             "height": ("INT", {"default": 480, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 16}),
                             "length": ("INT", {"default": 81, "min": 1, "max": nodes.MAX_RESOLUTION, "step": 4}),
                             "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
                },
                "optional": {"clip_vision_output": ("CLIP_VISION_OUTPUT", ),
                             "start_image": ("IMAGE", ),
                             "end_image": ("IMAGE", ),
                }}

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "negative", "latent")
    FUNCTION = "encode"

    CATEGORY = "conditioning/video_models"

    def encode(self, positive, negative, vae, width, height, length, batch_size, start_image=None, end_image=None, clip_vision_output=None):
        flfv = WanFirstLastFrameToVideo()
        return flfv.encode(positive, negative, vae, width, height, length, batch_size, start_image=start_image, end_image=end_image, clip_vision_start_image=clip_vision_output)





class Stack_WanImageToVideo:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                "clip_vision_name": (['None'] +folder_paths.get_filename_list("clip_vision"),{"default": 'clip_vision_h.safetensors'}),
                "width": ("INT", {"default": 832, "min": 16, "max": 4096, "step": 16}),
                "height": ("INT", {"default": 480, "min": 16, "max": 4096, "step": 16}),
                "length": ("INT", {"default": 40, "min": 1, "max": 4096, "step": 4}),
                                    
                },
                "optional": {    
                    "clip_img": ("IMAGE", ),
                    
                    "start_image": ("IMAGE", ),
                }}

    RETURN_TYPES = ("IMAGE_2V",)
    RETURN_NAMES = ("Image_2V",)
    FUNCTION = "encode"
    CATEGORY = "Apt_Preset/stack"
    
    def encode(self, width, height, length, start_image=None, clip_img=None, clip_vision_name=None):

        return ((width, height, length, start_image, clip_img, clip_vision_name),)


class Stack_WanFunControlToVideo:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                "clip_vision_name": (['None'] +folder_paths.get_filename_list("clip_vision"),{"default": 'clip_vision_h.safetensors'}),
                "width": ("INT", {"default": 832, "min": 16, "max": 4096, "step": 16}),
                "height": ("INT", {"default": 480, "min": 16, "max": 4096, "step": 16}),
                "length": ("INT", {"default": 40, "min": 1, "max": 4096, "step": 4}),

                },
                "optional": {                    
                    "clip_img": ("IMAGE", ),

                    "start_image": ("IMAGE", ),
                    "control_video": ("IMAGE", ),
                }}

    RETURN_TYPES = ("FCTL_2V",)
    RETURN_NAMES = ("FunControl_2V",)
    FUNCTION = "encode"
    CATEGORY = "Apt_Preset/stack"

    def encode(self, width, height, length, start_image=None, control_video=None, clip_img=None, clip_vision_name=None): 

        return ((width, height, length, start_image, control_video, clip_img, clip_vision_name),)


class Stack_WanFirstLastFrameToVideo:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                    "clip_vision_start": (['None'] +folder_paths.get_filename_list("clip_vision"),{"default": 'clip_vision_h.safetensors'}),
                    "clip_vision_end": (['None'] +folder_paths.get_filename_list("clip_vision"),{"default": 'clip_vision_h.safetensors'}),
                "width": ("INT", {"default": 832, "min": 16, "max": 4096, "step": 16}),
                "height": ("INT", {"default": 480, "min": 16, "max": 4096, "step": 16}),
                "length": ("INT", {"default": 40, "min": 1, "max": 4096, "step": 4}),

                },
                "optional": {
                    "clip_start_img": ("IMAGE", ),
                    "clip_end_img": ("IMAGE", ),

                    "start_image": ("IMAGE", ),
                    "end_image": ("IMAGE", ),
                }
        }

    RETURN_TYPES = ("FFL_2V",)
    RETURN_NAMES = ("FirstLast_2V",)
    FUNCTION = "encode"
    CATEGORY = "Apt_Preset/stack"

    def encode(self, width, height, length, start_image=None, end_image=None, clip_start_img=None, clip_end_img=None, clip_vision_start=None, clip_vision_end=None):

        return ((width, height, length, start_image, end_image, clip_start_img, clip_end_img, clip_vision_start, clip_vision_end),)


class Stack_WanFunInpaintToVideo:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip_vision_name": (['None'] +folder_paths.get_filename_list("clip_vision"),{"default": 'clip_vision_h.safetensors'}),
                "width": ("INT", {"default": 832, "min": 16, "max": 4096, "step": 16}),
                "height": ("INT", {"default": 480, "min": 16, "max": 4096, "step": 16}),
                "length": ("INT", {"default": 40, "min": 1, "max": 4096, "step": 4}),
                                    
                },
                "optional": {                    
                    "clip_img": ("IMAGE", ),

                    "start_image": ("IMAGE", ),
                    "end_image": ("IMAGE", ),
                }
        }

    RETURN_TYPES = ("FIP_2V",)
    RETURN_NAMES = ("FunInpaint_2V",)
    FUNCTION = "encode"
    CATEGORY = "Apt_Preset/stack"

    def encode(self, width, height, length, start_image=None, end_image=None, clip_img=None, clip_vision_name=None):
        return ((width, height, length, start_image, end_image, clip_img, clip_vision_name),)


class sum_stack_Wan:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "context": ("RUN_CONTEXT",),
            },
            "optional": {
                "model":("MODEL", ),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),

                "Image_2V": ("IMAGE_2V", ),
                "FunControl_2V": ("FCTL_2V", ),
                "FirstLast_2V": ("FFL_2V", ),
                "FunInpaint_2V": ("FIP_2V", ),

            },
            "hidden": {},
        }
        
    RETURN_TYPES = ("RUN_CONTEXT", "MODEL", "CONDITIONING", "CONDITIONING","LATENT",)
    RETURN_NAMES = ("context", "model", "positive", "negative", "latent",)
    FUNCTION = "merge"
    CATEGORY = "Apt_Preset/load"


    def merge(self, context=None, model=None, positive=None, negative=None, Image_2V=None, FunControl_2V=None, FirstLast_2V=None, FunInpaint_2V=None):
        
        clip = context.get("clip", None)
        vae = context.get("vae")

        if model is None:
            model = context.get("model")

        if positive is None:
            positive = context.get("positive")
        if negative is None:
            negative = context.get("negative")


        if Image_2V is not None:
            width, height, length, start_image, clip_img, clip_vision_name = Image_2V
            clip_vision_output = None
            if clip_img is not None and clip_vision_name is not None:
                # 提取元组中的对象
                clip_vision_output = clip_vision_output_encode(clip_vision_name, clip_img)[0]

            positive, negative, latent = WanImageToVideo().encode(positive, negative, vae, width, height, length, 1, start_image=start_image, clip_vision_output=clip_vision_output)

        if FunControl_2V is not None:
            width, height, length, start_image, control_video, clip_img, clip_vision_name = FunControl_2V
            clip_vision_output = None
            if clip_img is not None and clip_vision_name is not None:
                # 提取元组中的对象
                clip_vision_output = clip_vision_output_encode(clip_vision_name, clip_img)[0]

            positive, negative, latent = WanFunControlToVideo().encode(positive, negative, vae, width, height, length, 1, start_image=start_image, clip_vision_output=clip_vision_output, control_video=control_video)


        if FirstLast_2V is not None:
            width, height, length, start_image, end_image, clip_start_img, clip_end_img, clip_vision_start, clip_vision_end = FirstLast_2V
            clip_vision_start_image = None  # 初始化 clip_vision_start_image
            clip_vision_end_image = None  # 初始化 clip_vision_end_image
            if clip_start_img is not None and clip_vision_start is not None:
                clip_vision_start_image = clip_vision_output_encode(clip_vision_start, clip_start_img)[0]
            if clip_end_img is not None and clip_vision_end is not None:
                clip_vision_end_image = clip_vision_output_encode(clip_vision_end, clip_end_img)[0]


            positive, negative, latent = WanFirstLastFrameToVideo().encode(positive, negative, vae, width, height, length, 1, start_image=start_image, end_image=end_image, clip_vision_start_image=clip_vision_start_image, clip_vision_end_image=clip_vision_end_image)


        if FunInpaint_2V is not None:
            width, height, length, start_image, end_image, clip_img, clip_vision_name = FunInpaint_2V
            clip_vision_output = None  # 初始化 clip_vision_output
            if clip_img is not None and clip_vision_name is not None:
                clip_vision_output = clip_vision_output_encode(clip_vision_name, clip_img)[0]

            positive, negative, latent = WanFunInpaintToVideo().encode(positive, negative, vae, width, height, length, 1, start_image=start_image, end_image=end_image, clip_vision_output=clip_vision_output)

        context = new_context(context, clip=clip, positive=positive, latent=latent, negative=negative, model=model)

        return (context, model, positive, negative, latent )

