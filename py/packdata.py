
import folder_paths




class param_preset:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "optional": {
                "name": ("STRING", {"default": "","multiline": False}),
                "Float_min": ("FLOAT", {"default": 0.0, "min": -90000.0, "max": 90000.0, "step": 0.05, "round": 0.01}),
                "Float_max": ("FLOAT", {"default": 0.0, "min": -90000.0, "max": 90000.0, "step": 0.05, "round": 0.01}),
                "Float_data": ("FLOAT", {"default": 0.0, "min": -90000.0, "max": 90000.0, "step": 0.05, "round": 0.01}),
                "Num_min": ("INT", {"default": 512, "min": 1, "max": 10000}),
                "Num_max": ("INT", {"default": 512, "min": 1, "max": 10000}),
                "Num_width": ("INT", {"default": 512, "min": 1, "max": 10000}),
                "Num_height": ("INT", {"default": 512, "min": 1, "max": 10000}),
                "Num_data": ("INT", {"default": 1, "min": 1, "max": 10000}),
                "Float_Strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001, "round": 0.001}),
                "Float_start": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001, "round": 0.001}),
                "Float_end": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001, "round": 0.001}),
                "Float_scale": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 10.0, "step": 0.5, "round": 0.1}),
            }
        }

    RETURN_TYPES = ("DPIPE",)
    RETURN_NAMES = ("Drun",)
    FUNCTION = "data"
    CATEGORY = "Apt_Preset/unpack"

    def data(self, name, Float_min, Float_max, Float_data, Num_min, Num_max, Num_width, Num_height, Num_data, Float_Strength, Float_start, Float_end, Float_scale):
        
        Drun = name, Float_min,Float_max, Float_data,Num_min,Num_max,Num_width,Num_height,Num_data,Float_Strength,Float_start,Float_end,Float_scale
        
        return (Drun,)


class Unpack_param:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "Drun": ("DPIPE",)
            },
        }

    RETURN_TYPES = ("STRING", "FLOAT", "FLOAT", "FLOAT", "INT", "INT", "INT", "INT", "INT", "FLOAT", "FLOAT", "FLOAT", "FLOAT")
    RETURN_NAMES = ("name", "Float_min", "Float_max", "Float_data", "Num_min", "Num_max", "Num_width", "Num_height", "Num_data", "Float_Strength", "Float_start", "Float_end", "Float_scale")

    FUNCTION = "unpack_pipe"
    CATEGORY = "Apt_Preset/unpack"

    def unpack_pipe(self, Drun):

        name,Float_min,Float_max, Float_data,Num_min,Num_max,Num_width,Num_height,Num_data,Float_Strength,Float_start,Float_end,Float_scale = Drun 

        return (name, Float_min, Float_max, Float_data, Num_min, Num_max, Num_width, Num_height, Num_data, Float_Strength, Float_start, Float_end, Float_scale)


class Model_Preset:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "ckpt_name": (folder_paths.get_filename_list("checkpoints"), {"default": "flux1-dev-fp8.safetensors"}),
                "unet_name": (folder_paths.get_filename_list("unet"), {"default": "flux1-dev.safetensors"}),
                "clip_name1": (folder_paths.get_filename_list("clip"), {"default": "t5xxl_fp8_e4m3fn.safetensors"}),
                "clip_name2": (folder_paths.get_filename_list("clip"), {"default": "clip_g.safetensors"}),
                "clip_name3": (folder_paths.get_filename_list("clip"), {"default": "clip_l.safetensors"}),
                "clip_vision": (folder_paths.get_filename_list("clip_vision"), {"default": "CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors"}),
                "control_net_name": (folder_paths.get_filename_list("controlnet"), {"default": "Flux.1-dev-ControlNet-Union-Pro-fp8.safetensors"}),
                "style_model_name": (folder_paths.get_filename_list("style_models"), {"default": "flux1-redux-dev.safetensors"}),
                "loras_name": (folder_paths.get_filename_list("loras"), {"default": "add_detail.safetensors"}),
            }
        }

    RETURN_TYPES = ("EPIPE",)
    RETURN_NAMES = ("Erun",)
    FUNCTION = "data"
    CATEGORY = "Apt_Preset/unpack"

    def data(self, ckpt_name, unet_name, clip_name1, clip_name2, clip_name3, clip_vision, control_net_name, style_model_name, loras_name):
        Erun = {
            "ckpt_name": ckpt_name,
            "unet_name": unet_name,
            "clip_name1": clip_name1,
            "clip_name2": clip_name2,
            "clip_name3": clip_name3,
            "clip_vision": clip_vision,
            "control_net_name": control_net_name,
            "style_model_name": style_model_name,
            "loras_name": loras_name,
        }
        return (Erun,)


class Unpack_Model:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "Erun": ("EPIPE",),
            }
        }

    RETURN_TYPES = (
        folder_paths.get_filename_list("checkpoints"),
        folder_paths.get_filename_list("unet"),
        folder_paths.get_filename_list("clip"),
        folder_paths.get_filename_list("clip"),
        folder_paths.get_filename_list("clip"),
        folder_paths.get_filename_list("clip_vision"),
        folder_paths.get_filename_list("controlnet"),
        folder_paths.get_filename_list("style_models"),
        folder_paths.get_filename_list("loras"),
    )
    RETURN_NAMES = ("ckpt_name", "unet_name", "clip_name1", "clip_name2", "clip_name3", "clip_vision", "control_net_name", "style_model_name", "loras_name")
    FUNCTION = "unpack_pipe"
    CATEGORY = "Apt_Preset/unpack"

    def unpack_pipe(self, Erun):
        ckpt_name = Erun["ckpt_name"]
        unet_name = Erun["unet_name"]
        clip_name1 = Erun["clip_name1"]
        clip_name2 = Erun["clip_name2"]
        clip_name3 = Erun["clip_name3"]
        clip_vision = Erun["clip_vision"]
        control_net_name = Erun["control_net_name"]
        style_model_name = Erun["style_model_name"]
        loras_name = Erun["loras_name"]

        return ckpt_name, unet_name, clip_name1, clip_name2, clip_name3, clip_vision, control_net_name, style_model_name, loras_name


class CN_preset1:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "image": ("IMAGE",),
                "control_net": (folder_paths.get_filename_list("controlnet"), {"default": "control_v11p_sd15_canny.pth"}),
                "Float_Strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.001, "round": 0.001}),
                "Float_start": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001, "round": 0.001}),
                "Float_end": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.001, "round": 0.001}),
            }
        }

    RETURN_TYPES = ("FPIPE1",)
    RETURN_NAMES = ("Frun1",)
    FUNCTION = "data"
    CATEGORY = "Apt_Preset/unpack"

    def data(self, control_net, Float_Strength, Float_start, Float_end, image=None):

        Frun1 = {
            "controlnet": control_net,
            "image": image,
            "Float_Strength": Float_Strength,
            "Float_start": Float_start,
            "Float_end": Float_end,
        }
        return (Frun1,)


class Unpack_CN:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"Frun1": ("FPIPE1",)
            },
        }

    RETURN_TYPES = (folder_paths.get_filename_list("controlnet"), "IMAGE", "FLOAT", "FLOAT", "FLOAT",)
    RETURN_NAMES = ("control_net", "image", "Float_Strength", "Float_start", "Float_end", )
    FUNCTION = "unpack_pipe"
    CATEGORY = "Apt_Preset/unpack"

    def unpack_pipe(self, Frun1):
        return (Frun1["controlnet"], Frun1["image"], Frun1["Float_Strength"], Frun1["Float_start"], Frun1["Float_end"], )





class photoshop_preset:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
            },
            "optional": {               
                "Canvas": ("IMAGE",),
                "Mask": ("MASK",),
                "Slider": ("FLOAT", {"forceInput": True,"default": 1,}),
                "Seed": ("INT",{"forceInput": True,"default": 1}),
                "pos": ("STRING", {"forceInput": True,"default": "", }),
                "neg": ("STRING",{"forceInput": True,"default": "",} ),
                "W": ("INT",{"forceInput": True,"default": 512,}),
                "H": ("INT",{"forceInput": True,"default": 512,})}
        }

    RETURN_TYPES = ("FPIPE2",)
    RETURN_NAMES = ("ps_pack",)
    FUNCTION = "pspack"
    CATEGORY = "Apt_Preset/unpack"

    def pspack(self, Canvas=None, Mask=None, Slider=1, Seed=1, pos="", neg="", W=512, H=512):
        ps_pack = Canvas, Mask, Slider, Seed, pos, neg, W, H
        
        return (ps_pack,)      # 打包与解包，逗号----数据类型




class Unpack_photoshop:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ps_pack": ("FPIPE2",)},
            
            "optional": {}
        }

    RETURN_TYPES = ("IMAGE", "MASK", "FLOAT", "INT", "STRING", "STRING", "INT", "INT",)
    RETURN_NAMES = ("Canvas", "Mask", "Slider", "Seed", "pos", "neg", "W", "H",)
    FUNCTION = "unpack"
    CATEGORY = "Apt_Preset/unpack"

    def unpack(self, ps_pack):
        Canvas, Mask, Slider, Seed, pos, neg, W, H = ps_pack
        return ( Canvas, Mask, Slider, Seed, pos, neg, W, H )
