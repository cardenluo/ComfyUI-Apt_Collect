
from nodes import MAX_RESOLUTION
import torch
import comfy
import cv2
import numpy as np
from torchvision.transforms.functional import to_pil_image, to_tensor
from importlib import import_module
import comfy.utils
import comfy.model_management
from torchvision.transforms import Resize, CenterCrop
import torchvision.transforms.functional as TF
from typing import Literal, Any
from PIL import Image, ImageDraw, ImageFilter
import math
import comfy.sd
import comfy.lora
import folder_paths
import comfy.samplers
import comfy.controlnet 
from nodes import NODE_CLASS_MAPPINGS as ALL_NODE
from PIL import Image, ImageOps, ImageDraw, ImageFont, ImageChops
from comfy_extras.nodes_mask import composite

import torch
import numpy as np
import cv2


if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")



#--------------------------------------------------------------------------------------#

class pad_uv_fill:  #画布扩展算法填充
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "add_Height": ("INT", {
                    "default": 64,
                    "min": 0,
                }),
                "add_Width": ("INT", {
                    "default": 64,
                    "min": 0,
                }),
                "method": (["reflect", "edge", "constant"],),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "node"
    CATEGORY = "Apt_Collect/image"

    def node(self, images, add_Height, add_Width, method):
        def transpose_tensor(image):
            tensor = image.clone().detach()
            tensor_pad = TF.pad(tensor.permute(2, 0, 1), [add_Width, add_Height], padding_mode=method).permute(1, 2, 0)

            return tensor_pad

        return (torch.stack([
            transpose_tensor(images[i]) for i in range(len(images))
        ]),)


class AddPaddingBase:
     def __init__(self):
        pass
     
     FUNCTION = "resize"
     CATEGORY = "Apt_Collect/image"

     def add_padding(self, image, left, top, right, bottom, color="#ffffff", transparent=False):
            padded_images = []
            image = [self.tensor2pil(img) for img in image]
            for img in image:
                padded_image = Image.new("RGBA" if transparent else "RGB", 
                     (img.width + left + right, img.height + top + bottom), 
                     (0, 0, 0, 0) if transparent else self.hex_to_tuple(color))
                padded_image.paste(img, (left, top))
                padded_images.append(self.pil2tensor(padded_image))
            return torch.cat(padded_images, dim=0)
     
     def create_mask(self, image, left, top, right, bottom):
            masks = []
            image = [self.tensor2pil(img) for img in image]
            for img in image:
                shape = (left, top, img.width + left, img.height + top)
                mask_image = Image.new("L", (img.width + left + right, img.height + top + bottom), 255)
                draw = ImageDraw.Draw(mask_image)
                draw.rectangle(shape, fill=0)
                masks.append(self.pil2tensor(mask_image))
            return torch.cat(masks, dim=0)
     
     def hex_to_float(self, color):
        if not isinstance(color, str):
            raise ValueError("Color must be a hex string")
        color = color.strip("#")
        return int(color, 16) / 255.0
     
     def hex_to_tuple(self, color):
        if not isinstance(color, str):
            raise ValueError("Color must be a hex string")
        color = color.strip("#")
        return tuple([int(color[i:i + 2], 16) for i in range(0, len(color), 2)])
     
     # Tensor to PIL
     def tensor2pil(self, image):
        return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))
    
     # PIL to Tensor
     def pil2tensor(self, image):
        return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)


class pad_color_fill(AddPaddingBase):

    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "left": ("INT", {"default": 0, "step": 1, "min": 0, "max": 4096}),
                "top": ("INT", {"default": 0, "step": 1, "min": 0, "max": 4096}),
                "right": ("INT", {"default": 0, "step": 1, "min": 0, "max": 4096}),
                "bottom": ("INT", {"default": 0, "step": 1, "min": 0, "max": 4096}),
                "color": ("STRING", {"default": "#ffffff"}),
                "transparent": ("BOOLEAN", {"default": False}),
            },
        }
    
    RETURN_TYPES = ("IMAGE", "MASK")

    def resize(self, image, left, top, right, bottom, color, transparent):
        return (self.add_padding(image, left, top, right, bottom, color, transparent),
                self.create_mask(image, left, top, right, bottom),)


class Image_keep_OneColorr:  #保留一色
    NAME = "Color Stylizer"
    
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "target_r": ("INT", {
                    "default": 255,
                    "min": 0,
                    "max": 255,
                    "step": 1,
                    "display": "number"
                }),
                "target_g": ("INT", {
                    "default": 255,
                    "min": 0,
                    "max": 255,
                    "step": 1,
                    "display": "number"
                }),
                "target_b": ("INT", {
                    "default": 255,
                    "min": 0,
                    "max": 255,
                    "step": 1,
                    "display": "number"
                }),
                "falloff": ("FLOAT", {
                    "default": 30.0,
                    "min": 0.0,
                    "max": 100.0,
                    "step": 1.0,
                    "display": "number"
                }),
                "gain": ("FLOAT", {
                    "default": 1.5,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.5,
                    "display": "number"
                })
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "stylize"
    OUTPUT_NODE = False
    CATEGORY = "Apt_Collect/image"

    def stylize(self, image, target_r, target_g, target_b, falloff, gain):
        target_color = (target_b, target_g, target_r)
        image = image.squeeze(0)
        image = image.mul(255).byte().numpy()
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
        falloff_mask = self.create_falloff_mask(image, target_color, falloff)
        image_amplified = image.copy()
        image_amplified[:, :, 2] = np.clip(image_amplified[:, :, 2] * gain, 0, 255).astype(np.uint8)
        stylized_img = (image_amplified * falloff_mask + gray_img * (1 - falloff_mask)).astype(np.uint8)
        stylized_img = cv2.cvtColor(stylized_img, cv2.COLOR_BGR2RGB)
        stylized_img_tensor = to_tensor(stylized_img).float()
        stylized_img_tensor = stylized_img_tensor.permute(1, 2, 0).unsqueeze(0)
        return (stylized_img_tensor,)

    def create_falloff_mask(self, img, target_color, falloff):
        target_color = np.array(target_color, dtype=np.uint8)
        print("img shape:", img.shape)
        print("img dtype:", img.dtype)
        print("target_color shape:", target_color.shape)
        print("target_color dtype:", target_color.dtype)
        target_color = np.full_like(img, target_color)
        print(111)
        diff = cv2.absdiff(img, target_color)
        print(222)
        diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        print(333)
        _, mask = cv2.threshold(diff, falloff, 255, cv2.THRESH_BINARY_INV)
        mask = cv2.GaussianBlur(mask, (0, 0), falloff / 2)
        mask = mask / 255.0
        mask = mask.reshape(*mask.shape, 1)
        return mask


class Image_Normal_light:  #法向光源图

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "diffuse_map": ("IMAGE",),
                "normal_map": ("IMAGE",),
                "specular_map": ("IMAGE",),
                "light_yaw": ("FLOAT", {"default": 45, "min": -180, "max": 180, "step": 1}),
                "light_pitch": ("FLOAT", {"default": 30, "min": -90, "max": 90, "step": 1}),
                "specular_power": ("FLOAT", {"default": 32, "min": 1, "max": 200, "step": 1}),
                "ambient_light": ("FLOAT", {"default": 0.50, "min": 0, "max": 1, "step": 0.01}),
                "NormalDiffuseStrength": ("FLOAT", {"default": 1.00, "min": 0, "max": 5.0, "step": 0.01}),
                "SpecularHighlightsStrength": ("FLOAT", {"default": 1.00, "min": 0, "max": 5.0, "step": 0.01}),
                "TotalGain": ("FLOAT", {"default": 1.00, "min": 0, "max": 2.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("IMAGE",)

    FUNCTION = "execute"
    CATEGORY = "Apt_Collect/image"

    def execute(self, diffuse_map, normal_map, specular_map, light_yaw, light_pitch, specular_power, ambient_light,NormalDiffuseStrength,SpecularHighlightsStrength,TotalGain):

        # 入力画像をテンソルに変換
        diffuse_tensor = diffuse_map.permute(0, 3, 1, 2)  # (1, 512, 512, 3) -> (1, 3, 512, 512)
        normal_tensor = normal_map.permute(0, 3, 1, 2) * 2.0 - 1.0   # (1, 512, 512, 3) -> (1, 3, 512, 512)
        specular_tensor = specular_map.permute(0, 3, 1, 2)  # (1, 512, 512, 3) -> (1, 3, 512, 512)

        # 法線ベクトルを正規化
        normal_tensor = torch.nn.functional.normalize(normal_tensor, dim=1)

        # light_directionをブロードキャスト用に正しくリシェイプ
        light_direction = self.euler_to_vector(light_yaw, light_pitch, 0 )
        light_direction = light_direction.view(1, 3, 1, 1)  # [1, 3, 1, 1]にリシェイプしてブロードキャストを可能にする

        # camera_directionをブロードキャスト用に正しくリシェイプ
        camera_direction = self.euler_to_vector(0,0,0)
        camera_direction = camera_direction.view(1, 3, 1, 1) # [1, 3, 1, 1]にリシェイプしてブロードキャストを可能にする

        # 乗算のための既存のコード...
        diffuse = torch.sum(normal_tensor * light_direction, dim=1, keepdim=True)
        diffuse = torch.clamp(diffuse, 0, 1)

        # 鏡面反射の計算
        half_vector = torch.nn.functional.normalize(light_direction + camera_direction, dim=1)
        specular = torch.sum(normal_tensor * half_vector, dim=1, keepdim=True)
        specular = torch.pow(torch.clamp(specular, 0, 1), specular_power)

        # 拡散反射と鏡面反射の結果を合成
        output_tensor = ( diffuse_tensor * (ambient_light + diffuse * NormalDiffuseStrength ) + specular_tensor * specular * SpecularHighlightsStrength) * TotalGain

        # テンソルを出力用の画像に変換
        output_tensor = output_tensor.permute(0, 2, 3, 1)  # (1, 3, 512, 512) -> (1, 512, 512, 3)

        return (output_tensor,)


    def euler_to_vector(self, yaw, pitch, roll):
        yaw_rad = np.radians(yaw)
        pitch_rad = np.radians(pitch)
        roll_rad = np.radians(roll)

        cos_pitch = np.cos(pitch_rad)
        sin_pitch = np.sin(pitch_rad)
        cos_yaw = np.cos(yaw_rad)
        sin_yaw = np.sin(yaw_rad)
        cos_roll = np.cos(roll_rad)
        sin_roll = np.sin(roll_rad)

        direction = np.array([
            sin_yaw * cos_pitch,
            sin_pitch,
            cos_pitch * cos_yaw
        ])


        return torch.from_numpy(direction).float()

    def convert_tensor_to_image(self, tensor):
        # PyTorchのテンソルをPIL.Imageに変換
        tensor = tensor.squeeze(0)  # (1, 512, 512, 3) -> (512, 512, 3)
        tensor = tensor.clamp(0, 1)  # テンソルの値を0から1の範囲に制限
        image = Image.fromarray((tensor.detach().cpu().numpy() * 255).astype(np.uint8))
        return image


#region------Image_Adjust------图像尺寸调整
import torch
import numpy as np
from PIL import Image, ImageOps
from math import gcd
import torch.nn.functional as F


def resize_image(image, target_width, target_height, method='contain', background_color='#000000'):

    # 将ComfyUI的图像Tensor转换为PIL图像对象
    img = Image.fromarray(np.clip(255. * image.cpu().numpy(), 0, 255).astype(np.uint8))
    img_width, img_height = img.size

    if method == 'contain':
        # Contain: 缩放图像以适应目标尺寸，保持宽高比，可能出现背景
        img_ratio = img_width / img_height
        target_ratio = target_width / target_height
        if img_ratio > target_ratio:
            new_width = target_width
            new_height = int(target_width / img_ratio)
        else:
            new_height = target_height
            new_width = int(target_height * img_ratio)
        resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # 解析背景颜色
        try:
            color = tuple(int(background_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
        except ValueError:
            color = (0, 0, 0)  # 默认黑色
            
        padded_img = Image.new('RGB', (target_width, target_height), color)
        x_offset = (target_width - new_width) // 2
        y_offset = (target_height - new_height) // 2
        padded_img.paste(resized_img, (x_offset, y_offset))
        
        # 生成mask
        mask = calculate_mask((img_width, img_height), (target_width, target_height), method, scale_factor=1.0)
        return np.array(padded_img).astype(np.float32)/255.0, mask, target_width, target_height

    elif method == 'cover':
        # Cover: 缩放图像以覆盖目标尺寸，保持宽高比，可能裁剪
        img_ratio = img_width / img_height
        target_ratio = target_width / target_height
        if img_ratio > target_ratio:
            new_height = target_height
            new_width = int(target_height * img_ratio)
        else:
            new_width = target_width
            new_height = int(target_width / img_ratio)
        resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        x_offset = (new_width - target_width) // 2
        y_offset = (new_height - target_height) // 2
        cropped_img = resized_img.crop((x_offset, y_offset, x_offset + target_width, y_offset + target_height))
        
        # 生成mask
        mask = calculate_mask((img_width, img_height), (target_width, target_height), method, scale_factor=1.0)
        return np.array(cropped_img).astype(np.float32)/255.0, mask, target_width, target_height

    elif method == 'fill':
        # Fill: 拉伸图像以填充目标尺寸，忽略宽高比，可能变形
        resized_img = img.resize((target_width, target_height), Image.Resampling.LANCZOS)
        # 生成mask
        mask = calculate_mask((img_width, img_height), (target_width, target_height), method, scale_factor=1.0)
        return np.array(resized_img).astype(np.float32)/255.0, mask, target_width, target_height

    elif method == 'inside':
        # Inside: 和 contain 相同，保持宽高比，缩小或不改变图像使其完全适合容器
        img_ratio = img_width / img_height
        target_ratio = target_width / target_height
        if img_ratio > target_ratio:
            new_width = target_width
            new_height = int(target_width / img_ratio)
        else:
            new_height = target_height
            new_width = int(target_height * img_ratio)
        resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # 解析背景颜色
        try:
            color = tuple(int(background_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
        except ValueError:
            color = (0, 0, 0)  # 默认黑色
            
        padded_img = Image.new('RGB', (target_width, target_height), color)
        x_offset = (target_width - new_width) // 2
        y_offset = (target_height - new_height) // 2
        padded_img.paste(resized_img, (x_offset, y_offset))
        
        # 生成mask
        mask = calculate_mask((img_width, img_height), (target_width, target_height), method, scale_factor=1.0)
        return np.array(padded_img).astype(np.float32)/255.0, mask, target_width, target_height

    elif method == 'outside':
        # Outside: 和 cover 相同，保持宽高比，放大或不改变图像使其完全覆盖容器
        img_ratio = img_width / img_height
        target_ratio = target_width / target_height
        if img_ratio > target_ratio:
            new_height = target_height
            new_width = int(target_height * img_ratio)
        else:
            new_width = target_width
            new_height = int(target_width / img_ratio)
        resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        x_offset = (new_width - target_width) // 2
        y_offset = (new_height - target_height) // 2
        cropped_img = resized_img.crop((x_offset, y_offset, x_offset + target_width, y_offset + target_height))
        
        # 生成mask
        mask = calculate_mask((img_width, img_height), (target_width, target_height), method, scale_factor=1.0)
        return np.array(cropped_img).astype(np.float32)/255.0, mask, target_width, target_height

def pad_image(image, target_width, target_height, position='center', background_color='#000000'):
    """Pad an image to the target dimensions with specified background color."""
    # 将ComfyUI的图像Tensor转换为PIL图像对象
    img = Image.fromarray(np.clip(255. * image.cpu().numpy(), 0, 255).astype(np.uint8))
    img_width, img_height = img.size

    # 解析十六进制颜色
    try:
        color = tuple(int(background_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
    except ValueError:
        print(f"Invalid color format: {background_color}, using black")
        color = (0, 0, 0)

    # 创建指定颜色的背景图像
    padded_img = Image.new('RGB', (target_width, target_height), color)

    # 计算粘贴位置
    if position == 'center':
        x_offset = (target_width - img_width) // 2
        y_offset = (target_height - img_height) // 2
    elif position == 'top':
        x_offset = (target_width - img_width) // 2
        y_offset = 0
    elif position == 'bottom':
        x_offset = (target_width - img_width) // 2
        y_offset = target_height - img_height
    elif position == 'left':
        x_offset = 0
        y_offset = (target_height - img_height) // 2
    elif position == 'right':
        x_offset = target_width - img_width
        y_offset = (target_height - img_height) // 2
    else:
        raise ValueError(f"Invalid pad position: {position}")

    # 将原图粘贴到背景上
    padded_img.paste(img, (x_offset, y_offset))
    
    # 生成mask
    mask = calculate_mask((img_width, img_height), (target_width, target_height), position, scale_factor=1.0)
    
    # 转换回ComfyUI需要的格式
    return np.array(padded_img).astype(np.float32) / 255.0, mask, target_width, target_height

def calculate_resolution(aspect_ratio, scale_factor, max_width, max_height, min_width, min_height):
    """Calculate the target resolution based on aspect ratio and scale factor."""
    
    # SDXL 最佳分辨率对照表
    base_resolutions = {
        "1:1": (1024, 1024),
        "9:7": (1152, 896),
        "7:9": (896, 1152),
        "3:2": (1216, 832),
        "2:3": (832, 1216),
        "7:4": (1344, 768),
        "4:7": (768, 1344),
        "12:5": (1536, 640),
        "5:12": (640, 1536),
    }
    
    # 从输入中提取比例部分
    ratio = aspect_ratio.split(" ")[0]
    
    # 查找对应的基础分辨率
    if ratio in base_resolutions:
        base_width, base_height = base_resolutions[ratio]
    else:
        raise ValueError(f"Invalid aspect ratio: {ratio}")

    # 计算初始目标尺寸
    target_width = int(base_width * scale_factor)
    target_height = int(base_height * scale_factor)
    
    # 应用最大分辨率约束
    current_max = max(target_width, target_height)
    if current_max > max_width or current_max > max_height:
        scale = min(max_width / current_max, max_height / current_max)
        target_width = int(target_width * scale)
        target_height = int(target_height * scale)
    
    # 应用最小分辨率约束
    current_min = min(target_width, target_height)
    if current_min < min_width or current_min < min_height:
        scale = max(min_width / current_min, min_height / current_min)
        target_width = int(target_width * scale)
        target_height = int(target_height * scale)
    
    return target_width, target_height, base_width, base_height

def get_aspect_ratio_string(width, height):
    """Get the aspect ratio string from width and height, maintaining SDXL standard ratios"""
    # SDXL 标准比例映射
    sdxl_ratios = {
        (384, 512): "3:4",
        (512, 384): "4:3",
        (512, 512): "1:1",
        (512, 768): "2:3",
        (576, 1024): "9:16",
        (768, 512): "3:2",
        (768, 768): "1:1",
        (768, 1280): "3:5",
        (896, 1152): "7:9",
        (960, 1280): "3:4",
        (1024, 576): "16:9",
        (1024, 768): "4:3",
        (1024, 1024): "1:1",
        (1088, 1350): "272:337",
        (1080, 1920): "9:16",
        (1200, 675): "16:9",
        (1200, 1800): "2:3",
        (1280, 768): "5:3",
        (1280, 960): "4:3",
        (1344, 768): "7:4",
        (1440, 2560): "9:16",
        (1500, 2100): "5:7",
        (1536, 768): "2:1",
        (1920, 1080): "16:9"
    }


    # 如果是标准 SDXL 分辨率，直接返回对应的比例
    if (width, height) in sdxl_ratios:
        return sdxl_ratios[(width, height)]
    
    # 如果不是标准分辨率，则使用最大公约数计算
    common_divisor = gcd(width, height)
    aspect_width = width // common_divisor
    aspect_height = height // common_divisor
    return f"{aspect_width}:{aspect_height}"

def create_outline(image, background_color):
    """给图片添加1像素的描边，使用背景色的反色"""
    # 将背景色转换为RGB元组
    try:
        bg_color = tuple(int(background_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
    except ValueError:
        bg_color = (0, 0, 0)
    
    # 计算反色
    outline_color = tuple(255 - c for c in bg_color)
    
    # 将图像转换为PIL图像
    img = Image.fromarray(np.clip(255. * image, 0, 255).astype(np.uint8))
    width, height = img.size
    
    # 创建新图像，比原图大2像素
    outlined = Image.new('RGB', (width + 2, height + 2), outline_color)
    # 将原图粘贴到中心
    outlined.paste(img, (1, 1))
    
    # 转换回tensor格式
    return np.array(outlined).astype(np.float32) / 255.0

def calculate_mask(original_size, target_size, extend_mode, feather=0, scale_factor=1.0):
    """计算填充区域的mask
    Args:
        original_size: (width, height) 原始图像尺寸
        target_size: (width, height) 目标尺寸
        extend_mode: 扩展模式
        feather: 羽化程度
        scale_factor: 缩放因子
    """
    orig_w, orig_h = original_size
    target_w, target_h = target_size
    
    # 创建目标尺寸的mask（默认全黑，表示背景）
    mask = torch.zeros((target_h, target_w))
    
    if extend_mode == "fill":
        mask.fill_(1.0)
        
    elif extend_mode in ["cover", "outside"]:
        mask.fill_(1.0)
        
    elif extend_mode in ["contain", "inside"]:
        ratio = min(target_w/orig_w, target_h/orig_h)
        new_w = int(orig_w * ratio)
        new_h = int(orig_h * ratio)
        
        x_offset = (target_w - new_w) // 2
        y_offset = (target_h - new_h) // 2
        
        mask[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = 1.0
        
    elif extend_mode in ["top", "bottom", "left", "right", "center"]:
        # 1. 只应用 scale_factor 缩放，其他情况不缩放
        if scale_factor != 1.0:
            new_w = int(orig_w * scale_factor)
            new_h = int(orig_h * scale_factor)
        else:
            new_w = orig_w
            new_h = orig_h

        # 2. 计算有效的偏移量，确保图片完全在目标区域内
        if extend_mode == "center":
            # 居中对齐：在两个方向上都居中
            x_offset = max(0, (target_w - new_w) // 2)
            y_offset = max(0, (target_h - new_h) // 2)
        elif extend_mode == "top":
            # 顶部对齐：水平居中，垂直靠上
            x_offset = max(0, (target_w - new_w) // 2)
            y_offset = 0
        elif extend_mode == "bottom":
            # 底部对齐：水平居中，垂直靠下
            x_offset = max(0, (target_w - new_w) // 2)
            y_offset = max(0, target_h - new_h)
        elif extend_mode == "left":
            # 左对齐：垂直居中，水平靠左
            x_offset = 0
            y_offset = max(0, (target_h - new_h) // 2)
        else:  # right
            # 右对齐：垂直居中，水平靠右
            x_offset = max(0, target_w - new_w)
            y_offset = max(0, (target_h - new_h) // 2)

        # 3. 确保图片区域不超出目标范围
        actual_w = min(new_w, target_w - x_offset)
        actual_h = min(new_h, target_h - y_offset)

        # 4. 设置mask区域（有图的地方是白色(1)，没图的地方是黑色(0)）
        mask.fill_(0.0)  # 先将整个区域设为0（未填充）
        if actual_w > 0 and actual_h > 0:  # 确保有效的图像区域
            mask[y_offset:y_offset + actual_h, x_offset:x_offset + actual_w] = 1.0

    return mask

class Image_Adjust:
    def __init__(self):
        self.selected_color = "#000000"
    
    @classmethod
    def get_resolution_options(cls):

        base_resolutions = [
            (384, 512),  # 3:4
            (512, 384),  # 4:3
            (512, 512),  # 1:1
            (512, 768),  # 2:3
            (576, 1024),  # 9:16
            (768, 512),  # 3:2
            (768, 768),  # 1:1
            (768, 1280),  # 3:5
            (896, 1152),  # 7:9
            (960, 1280),  # 3:4
            (1024, 576),  # 16:9
            (1024, 768),  # 4:3
            (1024, 1024),  # 1:1
            (1088, 1350),  # 272:337
            (1080, 1920),  # 9:16
            (1200, 675),  # 16:9
            (1200, 1800),  # 2:3
            (1280, 768),  # 5:3
            (1280, 960),  # 4:3
            (1344, 768),  # 7:4
            (1440, 2560),  # 9:16
            (1500, 2100),  # 5:7
            (1536, 768),  # 2:1
            (1920, 1080),  # 16:9
        ]





        options = []
        for width, height in base_resolutions:
            ratio = get_aspect_ratio_string(width, height)
            options.append(f"{ratio} ({width}x{height})")
        
        return options

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "target_resolution": (s.get_resolution_options(),),
                "extend_mode": (["contain", "cover", "fill", "inside", "outside", "top", "bottom", "left", "right", "center"],),
                "scale_factor": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1}),
                "max_width": ("INT", {"default": 2048, "min": 1, "max": 8192, "step": 1}),
                "max_height": ("INT", {"default": 2048, "min": 1, "max": 8192, "step": 1}),
                "min_width": ("INT", {"default": 640, "min": 1, "max": 8192, "step": 1}),
                "min_height": ("INT", {"default": 640, "min": 1, "max": 8192, "step": 1}),
                "background_color": ("COLOR", {"default": "#000000"}),

            },
            "hidden": {"color_widget": "COMBO"}
        }

    CATEGORY = "Apt_Collect/image"
    RETURN_TYPES = ("IMAGE", "MASK", "INT", "INT")
    RETURN_NAMES = ("images", "mask", "width", "height")
    FUNCTION = "adjust_resolution"

    def adjust_resolution(self, images, target_resolution, extend_mode, background_color, scale_factor, max_width, max_height, min_width, min_height, ):
        output_images = []
        output_masks = []
        feather=0
        # 从目标分辨率字符串中提取宽高比
        aspect_ratio = target_resolution.split(" ")[0]
        
        # 计算目标分辨率
        target_width, target_height, base_width, base_height = calculate_resolution(
            aspect_ratio, scale_factor, max_width, max_height, min_width, min_height
        )
        
        for image in images:
            if extend_mode in ["contain", "cover", "fill", "inside", "outside"]:
                scaled_image, mask, width, height = resize_image(image, target_width, target_height, method=extend_mode, background_color=background_color)
            elif extend_mode in ["top", "bottom", "left", "right", "center"]:
                scaled_image, mask, width, height = pad_image(image, target_width, target_height, 
                                                      position=extend_mode, 
                                                      background_color=background_color)
            else:
                raise ValueError(f"Invalid extend_mode: {extend_mode}")
            
            
            output_images.append(torch.from_numpy(scaled_image).unsqueeze(0))
            output_masks.append(mask.unsqueeze(0))
        
        # 合并所有图像和mask
        output_images = torch.cat(output_images, dim=0)
        output_masks = torch.cat(output_masks, dim=0)
        
        return (output_images, output_masks, width, height)

    @classmethod
    def VALIDATE_INPUTS(s, **kwargs):
        if "background_color" in kwargs:
            color = kwargs["background_color"]
            # 验证颜色格式
            if not color.startswith('#') or len(color) != 7:
                return False
            try:
                # 尝试解析十六进制颜色
                int(color[1:], 16)
            except ValueError:
                return False
        return True

    # 添加 Widget 定义
    @classmethod
    def WIDGETS(s):
        return {"color_widget": {"widget_type": "color_picker", "target": "background_color"}}

#endregion


class Image_transform:  #图像与遮罩同步变换

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "up": ("INT", {"default": 0, "min": -32768, "max": 32767}),
                "down": ("INT", {"default": 0, "min": -32768, "max": 32767}),
                "left": ("INT", {"default": 0, "min": -32768, "max": 32767}),
                "right": ("INT", {"default": 0, "min": -32768, "max": 32767}),
                "Background": (["White", "Black", "Mirror", "Tile", "Extend"],),
                "InvertMask": ("BOOLEAN", {"default": False}),
                "InvertBackMask": ("BOOLEAN", {"default": False})
            },
            "optional": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
            },
        }
    CATEGORY = "Apt_Collect/image"
    RETURN_TYPES = ("IMAGE", "MASK", "MASK")
    RETURN_NAMES = ("image", "mask", "bg_mask")
    FUNCTION = "adv_crop"

    def adv_crop(self, up, down, left, right, Background, InvertMask, InvertBackMask, image=None, mask=None):
        Background_mapping = {
            "White": "White",
            "Black": "Black",
            "Mirror": "reflect",
            "Tile": "circular",
            "Extend": "replicate"
        }
        Background = Background_mapping[Background]

        back_mask = None
        crop_data = np.array([left, right, up, down])
        if image is not None:
            image, back_mask = self.data_processing(
                image, crop_data, back_mask, Background)
        if mask is not None:
            mask, back_mask = self.data_processing(
                mask, crop_data, back_mask, Background)
            if InvertMask:
                mask = 1.0 - mask
        if InvertBackMask and back_mask is not None:
            back_mask = 1.0 - back_mask
        return (image, mask, back_mask)

    def data_processing(self, image, crop_data, back_mask, Background):
    # Obtain image data 获取图像数据
        n, h, w, c, dim, image = self.get_image_data(image)
        shape = np.array([h, h, w, w])

        # Set the crop data value that exceeds the boundary to the boundary value of -1
        # 将超出边界的crop_data值设为边界值-1
        for i in range(crop_data.shape[0]):
            if crop_data[i] >= h:
                crop_data[i] = shape[i]-1

        # Determine whether the height exceeds the boundary 判断高是否超出边界
        if crop_data[0]+crop_data[1] >= h:
            raise ValueError(
                f"The height {crop_data[0]+crop_data[1]} of the cropped area exceeds the size of image {image.shape[2]}")
        # Determine if the width exceeds the boundary 判断宽是否超出边界
        elif crop_data[2]+crop_data[3] >= w:
            raise ValueError(
                f"The width {crop_data[2]+crop_data[3]} of the cropped area exceeds the size of image {image.shape[3]}")

        # Separate into cropped and expanded data 分离为裁剪和扩展数据
        extend_data = np.array([0, 0, 0, 0])
        for i in range(crop_data.shape[0]):
            if crop_data[i] < 0:
                extend_data[i] = abs(crop_data[i])
                crop_data[i] = 0

        # Expand the image and mask 扩展背景遮罩
        back_mask_run = False
        if back_mask is None:
            back_mask_run = True
            back_mask = torch.ones(
                (n, h, w), dtype=torch.float32, device=device)
            back_mask = torch.nn.functional.pad(
                back_mask, tuple(extend_data), mode='constant', value=0.0)

        # Expand the image and mask 扩展图像和背景遮罩
            # Filling method during expansion
            # 扩展时的图像填充方式
        fill_color = 1.0
        if Background == "White":
            Background = "constant"
        elif Background == "Black":
            Background = "constant"
            fill_color = 0.0

            # Extended data varies depending on the image or mask
            # 扩展数据因图像或遮罩而异
        if dim == 4:
            extend_data = tuple(np.concatenate(
                (np.array([0, 0]), extend_data)))
        else:
            extend_data = tuple(extend_data)

            # run Expand the image and mask 运行扩展图像和背景遮罩
        if Background == "constant":
            image = torch.nn.functional.pad(
                image, extend_data, mode=Background, value=fill_color)
        else:
            image = torch.nn.functional.pad(
                image, extend_data, mode=Background)

        # Crop the image and mask 裁剪图像和背景遮罩
        if dim == 4:
            n, h, w, c = image.shape
            image = image[:,
                          crop_data[2]:h-crop_data[3],
                          crop_data[0]:w-crop_data[1],
                          :]
        else:
            n, h, w = image.shape
            image = image[:,
                          crop_data[2]:h-crop_data[3],
                          crop_data[0]:w-crop_data[1]
                          ]
        if back_mask_run:
            back_mask = back_mask[:,
                                  crop_data[2]:h-crop_data[3],
                                  crop_data[0]:w-crop_data[1]
                                  ]
        return [image, back_mask]

    # Obtaining and standardizing image data 获取并标准化图像数据
    def get_image_data(self, image):
        shape = image.shape
        dim = image.dim()
        n, h, w, c = 1, 1, 1, 1
        if dim == 4:
            n, h, w, c = shape
            if c == 1:  # 最后一维为单通道时应为遮罩
                image = image.squeeze(3)
                dim = 3
                print(f"""warning: Due to the input not being a standard image tensor,
                      it has been detected that it may be a mask.
                      We are currently converting {shape} to {image.shape} for processing""")
            elif h == 1 and (w != 1 or c != 1):  # 第2维为单通道时应为遮罩
                image = image.squeeze(1)
                dim = 3
                print(f"""warning: Due to the input not being a standard image/mask tensor,
                      it has been detected that it may be a mask.
                      We are currently converting {shape} to {image.shape} for processing""")
            else:
                print(f"Processing standard images:{shape}")
        elif dim == 3:
            n, h, w = shape
            print(f"Processing standard mask:{shape}")
        elif dim == 5:
            n, c, c1, h, w = shape
            if c == 1 and c1 == 1:  # was插件生成的mask批次可能会有此问题
                image = image.squeeze(1)
                image = image.squeeze(1)  # 移除mask批次多余的维度
                dim = 3
                print(f"""warning: Due to the input not being a standard mask tensor,
                      it has been detected that it may be a mask.
                      We are currently converting {shape} to {image.shape} for processing""")
        else:  # The image dimension is incorrect 图像维度不正确
            raise ValueError(
                f"The shape of the input image or mask data is incorrect, requiring image n, h, w, c mask n, h, w \nWhat was obtained is{shape}")
        return [n, h, w, c, dim,image]


class Image_cutResize:  #图像与遮罩同步裁切
    def __init__(self):
        pass


    ACTION_TYPE_RESIZE = "resize only"
    ACTION_TYPE_CROP = "crop to ratio"
    ACTION_TYPE_PAD = "pad to ratio"
    RESIZE_MODE_DOWNSCALE = "reduce size only"
    RESIZE_MODE_UPSCALE = "increase size only"
    RESIZE_MODE_ANY = "any"
    RETURN_TYPES = ("IMAGE", "MASK",)
    FUNCTION = "resize"
    CATEGORY = "Apt_Collect/image"


    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pixels": ("IMAGE",),
                "action": ([s.ACTION_TYPE_RESIZE, s.ACTION_TYPE_CROP, s.ACTION_TYPE_PAD],),
                "smaller_side": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 8}),
                "larger_side": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 8}),
                "scale_factor": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                "resize_mode": ([s.RESIZE_MODE_DOWNSCALE, s.RESIZE_MODE_UPSCALE, s.RESIZE_MODE_ANY],),
                "side_ratio": ("STRING", {"default": "4:3"}),
                "crop_pad_position": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "pad_feathering": ("INT", {"default": 20, "min": 0, "max": 8192, "step": 1}),
            },
            "optional": {
                "mask_optional": ("MASK",),
            },
        }


    @classmethod
    def VALIDATE_INPUTS(s, action, smaller_side, larger_side, scale_factor, resize_mode, side_ratio, **_):
        if side_ratio is not None:
            if action != s.ACTION_TYPE_RESIZE and s.parse_side_ratio(side_ratio) is None:
                return f"Invalid side ratio: {side_ratio}"

        if smaller_side is not None and larger_side is not None and scale_factor is not None:
            if int(smaller_side > 0) + int(larger_side > 0) + int(scale_factor > 0) > 1:
                return f"At most one scaling rule (smaller_side, larger_side, scale_factor) should be enabled by setting a non-zero value"

        if scale_factor is not None:
            if resize_mode == s.RESIZE_MODE_DOWNSCALE and scale_factor > 1.0:
                return f"For resize_mode {s.RESIZE_MODE_DOWNSCALE}, scale_factor should be less than one but got {scale_factor}"
            if resize_mode == s.RESIZE_MODE_UPSCALE and scale_factor > 0.0 and scale_factor < 1.0:
                return f"For resize_mode {s.RESIZE_MODE_UPSCALE}, scale_factor should be larger than one but got {scale_factor}"

        return True


    @classmethod
    def parse_side_ratio(s, side_ratio):
        try:
            x, y = map(int, side_ratio.split(":", 1))
            if x < 1 or y < 1:
                raise Exception("Ratio factors have to be positive numbers")
            return float(x) / float(y)
        except:
            return None


    def resize(self, pixels, action, smaller_side, larger_side, scale_factor, resize_mode, side_ratio, crop_pad_position, pad_feathering, mask_optional=None):
        validity = self.VALIDATE_INPUTS(action, smaller_side, larger_side, scale_factor, resize_mode, side_ratio)
        if validity is not True:
            raise Exception(validity)

        height, width = pixels.shape[1:3]
        if mask_optional is None:
            mask = torch.zeros(1, height, width, dtype=torch.float32)
        else:
            mask = mask_optional
            if mask.shape[1] != height or mask.shape[2] != width:
                mask = torch.nn.functional.interpolate(mask.unsqueeze(0), size=(height, width), mode="bicubic").squeeze(0).clamp(0.0, 1.0)

        crop_x, crop_y, pad_x, pad_y = (0.0, 0.0, 0.0, 0.0)
        if action == self.ACTION_TYPE_CROP:
            target_ratio = self.parse_side_ratio(side_ratio)
            if height * target_ratio < width:
                crop_x = width - height * target_ratio
            else:
                crop_y = height - width / target_ratio
        elif action == self.ACTION_TYPE_PAD:
            target_ratio = self.parse_side_ratio(side_ratio)
            if height * target_ratio > width:
                pad_x = height * target_ratio - width
            else:
                pad_y = width / target_ratio - height

        if smaller_side > 0:
            if width + pad_x - crop_x > height + pad_y - crop_y:
                scale_factor = float(smaller_side) / (height + pad_y - crop_y)
            else:
                scale_factor = float(smaller_side) / (width + pad_x - crop_x)
        if larger_side > 0:
            if width + pad_x - crop_x > height + pad_y - crop_y:
                scale_factor = float(larger_side) / (width + pad_x - crop_x)
            else:
                scale_factor = float(larger_side) / (height + pad_y - crop_y)

        if (resize_mode == self.RESIZE_MODE_DOWNSCALE and scale_factor >= 1.0) or (resize_mode == self.RESIZE_MODE_UPSCALE and scale_factor <= 1.0):
            scale_factor = 0.0

        if scale_factor > 0.0:
            pixels = torch.nn.functional.interpolate(pixels.movedim(-1, 1), scale_factor=scale_factor, mode="bicubic", antialias=True).movedim(1, -1).clamp(0.0, 1.0)
            mask = torch.nn.functional.interpolate(mask.unsqueeze(0), scale_factor=scale_factor, mode="bicubic", antialias=True).squeeze(0).clamp(0.0, 1.0)
            height, width = pixels.shape[1:3]

            crop_x *= scale_factor
            crop_y *= scale_factor
            pad_x *= scale_factor
            pad_y *= scale_factor

        if crop_x > 0.0 or crop_y > 0.0:
            remove_x = (round(crop_x * crop_pad_position), round(crop_x * (1 - crop_pad_position))) if crop_x > 0.0 else (0, 0)
            remove_y = (round(crop_y * crop_pad_position), round(crop_y * (1 - crop_pad_position))) if crop_y > 0.0 else (0, 0)
            pixels = pixels[:, remove_y[0]:height - remove_y[1], remove_x[0]:width - remove_x[1], :]
            mask = mask[:, remove_y[0]:height - remove_y[1], remove_x[0]:width - remove_x[1]]
        elif pad_x > 0.0 or pad_y > 0.0:
            add_x = (round(pad_x * crop_pad_position), round(pad_x * (1 - crop_pad_position))) if pad_x > 0.0 else (0, 0)
            add_y = (round(pad_y * crop_pad_position), round(pad_y * (1 - crop_pad_position))) if pad_y > 0.0 else (0, 0)

            new_pixels = torch.zeros(pixels.shape[0], height + add_y[0] + add_y[1], width + add_x[0] + add_x[1], pixels.shape[3], dtype=torch.float32)
            new_pixels[:, add_y[0]:height + add_y[0], add_x[0]:width + add_x[0], :] = pixels
            pixels = new_pixels

            new_mask = torch.ones(mask.shape[0], height + add_y[0] + add_y[1], width + add_x[0] + add_x[1], dtype=torch.float32)
            new_mask[:, add_y[0]:height + add_y[0], add_x[0]:width + add_x[0]] = mask
            mask = new_mask

            if pad_feathering > 0:
                for i in range(mask.shape[0]):
                    for j in range(pad_feathering):
                        feather_strength = (1 - j / pad_feathering) * (1 - j / pad_feathering)
                        if add_x[0] > 0 and j < width:
                            for k in range(height):
                                mask[i, k, add_x[0] + j] = max(mask[i, k, add_x[0] + j], feather_strength)
                        if add_x[1] > 0 and j < width:
                            for k in range(height):
                                mask[i, k, width + add_x[0] - j - 1] = max(mask[i, k, width + add_x[0] - j - 1], feather_strength)
                        if add_y[0] > 0 and j < height:
                            for k in range(width):
                                mask[i, add_y[0] + j, k] = max(mask[i, add_y[0] + j, k], feather_strength)
                        if add_y[1] > 0 and j < height:
                            for k in range(width):
                                mask[i, height + add_y[0] - j - 1, k] = max(mask[i, height + add_y[0] - j - 1, k], feather_strength)

        return (pixels, mask)


class Image_Extract_Channel:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "channel": (["R", "G", "B", "A"],),
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("channel_data",)
    CATEGORY = "Apt_Collect/image"
    FUNCTION = "image_extract_alpha"

    def image_extract_alpha(self, images: torch.Tensor, channel):
        # images in shape (N, H, W, C)

        if len(images.shape) < 4:
            images = images.unsqueeze(3).repeat(1, 1, 1, 3)

        if channel == "A" and images.shape[3] < 4:
            raise Exception("Image does not have an alpha channel")

        channel_index = ["R", "G", "B", "A"].index(channel)
        mask = images[:, :, :, channel_index].cpu().clone()

        return (mask,)


class Image_Apply_Channel:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "channel_data": ("MASK",),
                "channel": (["R", "G", "B", "A"],),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    CATEGORY = "Apt_Collect/image"
    FUNCTION = "image_apply_channel"

    def image_apply_channel(self, images: torch.Tensor, channel_data: torch.Tensor, channel):
        merged_images = []

        for image in images:
            image = image.cpu().clone()

            if channel == "A":
                if image.shape[2] < 4:
                    image = torch.cat([image, torch.ones((image.shape[0], image.shape[1], 1))], dim=2)

                image[:, :, 3] = channel_data
            elif channel == "R":
                image[:, :, 0] = channel_data
            elif channel == "G":
                image[:, :, 1] = channel_data
            else:
                image[:, :, 2] = channel_data

            merged_images.append(image)

        return (torch.stack(merged_images),)


#region --------Image_LightShape-------灯光形状图

def draw_shape(shape, size=(200, 200), offset=(0, 0), scale=1.0, rotation=0, bg_color=(255, 255, 255),
               shape_color=(0, 0, 0), opacity=1.0, blur_radius=0, base_image=None):
    width, height = size
    offset_x, offset_y = offset
    center_x, center_y = width // 2 + offset_x, height // 2 + offset_y
    max_dim = min(width, height) * scale

    diagonal = int(math.sqrt(width ** 2 + height ** 2))
    img_tmp = Image.new('RGBA', (diagonal, diagonal), (0, 0, 0, 0))
    draw_tmp = ImageDraw.Draw(img_tmp)

    tmp_center = diagonal // 2

    alpha = int(opacity * 255)
    shape_color = shape_color + (alpha,)

    if shape == 'circle':
        bbox = (tmp_center - max_dim / 2, tmp_center - max_dim / 2, tmp_center + max_dim / 2, tmp_center + max_dim / 2)
        draw_tmp.ellipse(bbox, fill=shape_color)

    elif shape == 'semicircle':
        bbox = (tmp_center - max_dim / 2, tmp_center - max_dim / 2, tmp_center + max_dim / 2, tmp_center + max_dim / 2)
        draw_tmp.pieslice(bbox, start=0, end=180, fill=shape_color)

    elif shape == 'quarter_circle':
        bbox = (tmp_center - max_dim / 2, tmp_center - max_dim / 2, tmp_center + max_dim / 2, tmp_center + max_dim / 2)
        draw_tmp.pieslice(bbox, start=0, end=90, fill=shape_color)

    elif shape == 'ellipse':
        bbox = (tmp_center - max_dim / 2, tmp_center - max_dim / 4, tmp_center + max_dim / 2, tmp_center + max_dim / 4)
        draw_tmp.ellipse(bbox, fill=shape_color)

    elif shape == 'square':
        bbox = (tmp_center - max_dim / 2, tmp_center - max_dim / 2, tmp_center + max_dim / 2, tmp_center + max_dim / 2)
        draw_tmp.rectangle(bbox, fill=shape_color)

    elif shape == 'triangle':
        points = [
            (tmp_center, tmp_center - max_dim / 2),
            (tmp_center - max_dim / 2, tmp_center + max_dim / 2),
            (tmp_center + max_dim / 2, tmp_center + max_dim / 2)
        ]
        draw_tmp.polygon(points, fill=shape_color)

    elif shape == 'cross':
        vertical = [(tmp_center - max_dim / 6, tmp_center - max_dim / 2),
                    (tmp_center + max_dim / 6, tmp_center - max_dim / 2),
                    (tmp_center + max_dim / 6, tmp_center + max_dim / 2),
                    (tmp_center - max_dim / 6, tmp_center + max_dim / 2)]
        horizontal = [(tmp_center - max_dim / 2, tmp_center - max_dim / 6),
                      (tmp_center + max_dim / 2, tmp_center - max_dim / 6),
                      (tmp_center + max_dim / 2, tmp_center + max_dim / 6),
                      (tmp_center - max_dim / 2, tmp_center + max_dim / 6)]
        draw_tmp.polygon(vertical, fill=shape_color)
        draw_tmp.polygon(horizontal, fill=shape_color)

    elif shape == 'star':
        points = []
        for i in range(10):
            angle = i * 36 * math.pi / 180
            radius = max_dim / 2 if i % 2 == 0 else max_dim / 4
            points.append((tmp_center + radius * math.sin(angle), tmp_center - radius * math.cos(angle)))
        draw_tmp.polygon(points, fill=shape_color)

    elif shape == 'radial':
        num_rays = 12
        for i in range(num_rays):
            angle = i * (360 / num_rays) * math.pi / 180
            x1 = tmp_center + max_dim / 4 * math.cos(angle)
            y1 = tmp_center + max_dim / 4 * math.sin(angle)
            x2 = tmp_center + max_dim / 2 * math.cos(angle)
            y2 = tmp_center + max_dim / 2 * math.sin(angle)
            draw_tmp.line([(x1, y1), (x2, y2)], fill=shape_color, width=int(max_dim / 20))

    img_tmp = img_tmp.rotate(rotation, resample=Image.BICUBIC, expand=True)
    if base_image is None:
        img = Image.new('RGBA', size, bg_color + (255,))
    else:
        img = base_image.copy()
        if img.mode != 'RGBA':
            img = img.convert('RGBA')

    paste_x = center_x - img_tmp.width // 2
    paste_y = center_y - img_tmp.height // 2

    img.alpha_composite(img_tmp, (paste_x, paste_y))

    if blur_radius > 0:
        img = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    return img


def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return (r, g, b)


class Image_LightShape:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "wide": ("INT", {"default": 512, "min": 0, "max": 5000, "step": 1}),
                "height": ("INT", {"default": 512, "min": 0, "max": 5000, "step": 1}),
                "shape": (
                    [
                        'circle',
                        'square',
                        'semicircle',
                        'quarter_circle',
                        'ellipse',
                        'triangle',
                        'cross',
                        'star',
                        'radial',
                    ],
                    {"default": "circle"},
                ),
                "X_offset": ("INT", {"default": 0, "min": -10000, "max": 10000, "step": 1}),
                "Y_offset": ("INT", {"default": 0, "min": -10000, "max": 10000, "step": 1}),
                "scale": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1}),
                "rotation": ("INT", {"default": 0, "min": 0, "max": 360, "step": 1}),
                "opacity": ("FLOAT", {"default": 1.0, "min": 0, "max": 1.0, "step": 0.1}),
                "blur_radius": ("INT", {"default": 0, "min": 0, "max": 1000, "step": 1}),
                "background_color": ("STRING", {"default": "#000000"}),
                "shape_color": ("STRING", {"default": "#FFFFFF"}),
            },
            "optional": {
                "base_image": ("IMAGE", {"default": None}),
            },
        }

    CATEGORY = "Apt_Collect/image"

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "drew_light_shape"

    def drew_light_shape(self, wide, height, shape, X_offset, Y_offset, scale, rotation, opacity, blur_radius, background_color,
                         shape_color, base_image=None):

        if base_image is None:
            img = draw_shape(shape, size=(wide, height), offset=(X_offset, Y_offset), scale=scale,
                             rotation=rotation,
                             bg_color=hex_to_rgb(background_color), shape_color=hex_to_rgb(shape_color),
                             opacity=opacity, blur_radius=blur_radius)
        else:
            img_cv = Image.fromarray((base_image.squeeze().cpu().numpy() * 255).astype(np.uint8)).convert("RGBA")
            img = draw_shape(shape, size=(wide, height), offset=(X_offset, Y_offset), scale=scale,
                             rotation=rotation,
                             bg_color=hex_to_rgb(background_color), shape_color=hex_to_rgb(shape_color),
                             opacity=opacity, blur_radius=blur_radius, base_image=img_cv)

        rst = torch.from_numpy(np.array(img).astype(np.float32) / 255.0).unsqueeze(0)
        return (rst,)
    
#endregion


class lay_edge_cut:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "rows": ("INT", {"default": 2, "min": 1, "max": 10}),
                "cols": ("INT", {"default": 3, "min": 1, "max": 10}),
                "row_split_method": (["uniform", "edge_detection"],),
                "col_split_method": (["uniform", "edge_detection"],),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    FUNCTION = "split_image"
    CATEGORY = "Apt_Collect/layout"

    def find_split_positions(self, image, num_splits, is_vertical, split_method):
        if split_method == "uniform":
            size = image.shape[1] if is_vertical else image.shape[0]
            return [i * size // (num_splits + 1) for i in range(1, num_splits + 1)]
        else:  # edge_detection
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges, axis=0) if is_vertical else np.sum(edges, axis=1)
            
            window_size = len(edge_density) // (num_splits + 1) // 2
            smoothed_density = np.convolve(edge_density, np.ones(window_size)/window_size, mode='same')
            
            split_positions = []
            for i in range(1, num_splits + 1):
                start = i * len(smoothed_density) // (num_splits + 1) - window_size
                end = i * len(smoothed_density) // (num_splits + 1) + window_size
                split = start + np.argmin(smoothed_density[start:end])
                split_positions.append(split)
            return split_positions

    def split_image(self, image, rows, cols, row_split_method, col_split_method):
        
        # 确保输入图像是 4D tensor (B, H, W, C)
        if len(image.shape) == 3:
            image = image.unsqueeze(0)
        
        # 将图像从 PyTorch tensor 转换为 NumPy 数组
        img_np = (image[0].cpu().numpy() * 255).astype(np.uint8)
        
        height, width = img_np.shape[:2]
        print(f"Original image size: {width}x{height}")

        # 找到分割位置
        vertical_splits = self.find_split_positions(img_np, cols - 1, True, col_split_method)
        horizontal_splits = self.find_split_positions(img_np, rows - 1, False, row_split_method)

        # 创建预览图
        preview_img = image.clone()
        
        # 在预览图上画线
        green_line = torch.tensor([0.0, 1.0, 0.0]).view(1, 1, 1, 3)
        for x in vertical_splits:
            preview_img[:, :, x:x+2, :] = green_line
        for y in horizontal_splits:
            preview_img[:, y:y+2, :, :] = green_line
        
        # 分割图片
        split_images = []
        h_splits = [0] + horizontal_splits + [height]
        v_splits = [0] + vertical_splits + [width]
        
        # 计算最大的分割尺寸
        max_height = max([h_splits[i+1] - h_splits[i] for i in range(len(h_splits) - 1)])
        max_width = max([v_splits[i+1] - v_splits[i] for i in range(len(v_splits) - 1)])
        
        for i in range(len(h_splits) - 1):
            for j in range(len(v_splits) - 1):
                top = h_splits[i]
                bottom = h_splits[i+1]
                left = v_splits[j]
                right = v_splits[j+1]
                
                cell = image[:, top:bottom, left:right, :]
                
                # 填充到最大尺寸
                pad_bottom = max_height - (bottom - top)
                pad_right = max_width - (right - left)
                cell = torch.nn.functional.pad(cell, (0, 0, 0, pad_right, 0, pad_bottom), mode='constant')
                
                split_images.append(cell)

        split_images = torch.cat(split_images, dim=0)  
        return (preview_img, split_images)


class Image_RemoveAlpha:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "Apt_Collect/image"

    def execute(self, image):
        if image.shape[3] == 4:
            image = image[..., :3]
        return (image,)


#region --------image_sumTransform-------图像变换

from math import ceil, sqrt
from typing import cast
import torch
import torchvision.transforms.functional as TF
from PIL import Image
import numpy as np
import numpy.typing as npt
from collections.abc import Callable, Sequence

def to_numpy(image: torch.Tensor) -> npt.NDArray[np.uint8]:
    np_array = np.clip(255.0 * image.cpu().numpy(), 0, 255).astype(np.uint8)
    return np_array

def hex_to_rgb(hex_color):
    try:
        hex_color = hex_color.lstrip("#")
        return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
    except ValueError:
        return (0, 0, 0)

def handle_batch(
    tensor: torch.Tensor,
    func: Callable[[torch.Tensor], Image.Image | npt.NDArray[np.uint8]],
) -> list[Image.Image] | list[npt.NDArray[np.uint8]]:
    """Handles batch processing for a given tensor and conversion function."""
    return [func(tensor[i]) for i in range(tensor.shape[0])]

def tensor2pil(tensor: torch.Tensor) -> list[Image.Image]:
    def single_tensor2pil(t: torch.Tensor) -> Image.Image:
        np_array = to_numpy(t)
        if np_array.ndim == 2:  # (H, W) for masks
            return Image.fromarray(np_array, mode="L")
        elif np_array.ndim == 3:  # (H, W, C) for RGB/RGBA
            if np_array.shape[2] == 3:
                return Image.fromarray(np_array, mode="RGB")
            elif np_array.shape[2] == 4:
                return Image.fromarray(np_array, mode="RGBA")
        raise ValueError(f"Invalid tensor shape: {t.shape}")
    return handle_batch(tensor, single_tensor2pil)

def pil2tensor(images: Image.Image | list[Image.Image]) -> torch.Tensor:
    def single_pil2tensor(image: Image.Image) -> torch.Tensor:
        np_image = np.array(image).astype(np.float32) / 255.0
        if np_image.ndim == 2:  # Grayscale
            return torch.from_numpy(np_image).unsqueeze(0)  # (1, H, W)
        else:  # RGB or RGBA
            return torch.from_numpy(np_image).unsqueeze(0)  # (1, H, W, C)
    if isinstance(images, Image.Image):
        return single_pil2tensor(images)
    else:
        return torch.cat([single_pil2tensor(img) for img in images], dim=0)

class image_sumTransform:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "x": ("FLOAT", {"default": 0, "step": 1, "min": -4096, "max": 4096}),
                "y": ("FLOAT", {"default": 0, "step": 1, "min": -4096, "max": 4096}),
                "zoom": ("FLOAT", {"default": 1.0, "min": 0.001, "step": 0.01}),
                "angle": ("FLOAT", {"default": 0, "step": 1, "min": -360, "max": 360}),
                "shear": ("FLOAT", {"default": 0, "step": 1, "min": -4096, "max": 4096}),
                "border_handling": (["edge", "constant", "reflect", "symmetric"], {"default": "edge"}),
                "constant_color": ("COLOR", {"default": "#000000"}),
            },
            "optional": {
                "filter_type": (["nearest", "box", "bilinear", "hamming", "bicubic", "lanczos"], {"default": "bilinear"}),
            },
        }

    FUNCTION = "transform"
    RETURN_TYPES = ("IMAGE",)
    CATEGORY = "Apt_Collect/image"

    def transform(
        self,
        image: torch.Tensor,
        x: float,
        y: float,
        zoom: float,
        angle: float,
        shear: float,
        border_handling="edge",
        constant_color=None,
        filter_type="nearest",
    ):
        filter_map = {
            "nearest": Image.NEAREST,
            "box": Image.BOX,
            "bilinear": Image.BILINEAR,
            "hamming": Image.HAMMING,
            "bicubic": Image.BICUBIC,
            "lanczos": Image.LANCZOS,
        }
        resampling_filter = filter_map[filter_type]

        x = int(x)
        y = int(y)
        angle = int(angle)

        if image.size(0) == 0:
            return (torch.zeros(0),)
        transformed_images = []
        frames_count, frame_height, frame_width, frame_channel_count = image.size()

        new_height, new_width = int(frame_height * zoom), int(frame_width * zoom)

        diagonal = sqrt(frame_width**2 + frame_height**2)
        max_padding = ceil(diagonal * zoom - min(frame_width, frame_height))
        pw = int(frame_width - new_width)
        ph = int(frame_height - new_height)

        pw += abs(max_padding)
        ph += abs(max_padding)

        padding = [
            max(0, pw + x),
            max(0, ph + y),
            max(0, pw - x),
            max(0, ph - y),
        ]

        constant_color = hex_to_rgb(constant_color)

        for img in tensor2pil(image):
            img = TF.pad(
                img,
                padding=padding,
                padding_mode=border_handling,
                fill=constant_color or 0,
            )

            img = cast(
                Image.Image,
                TF.affine(
                    img,
                    angle=angle,
                    scale=zoom,
                    translate=[x, y],
                    shear=shear,
                    interpolation=resampling_filter,
                ),
            )

            left = abs(padding[0])
            upper = abs(padding[1])
            right = img.width - abs(padding[2])
            bottom = img.height - abs(padding[3])

            img = img.crop((left, upper, right, bottom))

            transformed_images.append(img)

        return (pil2tensor(transformed_images),)

#endregion--------image_Transform-------图像变换


#region---------image_selction--------图像选择

import numpy as np
import ast
import re


class str_edit:
    def __init__(self):
        pass
    @classmethod
    def convert_list(cls, string_input,arrangement=True):
        if string_input == "":
            return ([],)
        if arrangement:
            string_input = cls.tolist_v1(string_input)
        if string_input[0] != "[":
            string_input = "[" + string_input + "]"
            return (ast.literal_eval(string_input),)
        else:
            return (ast.literal_eval(string_input),)
        
    def tolist_v1(cls,user_input):#转换为简单的带负数多维数组格式
        user_input = user_input.replace('{', '[').replace('}', ']')# 替换大括号
        user_input = user_input.replace('(', '[').replace(')', ']')# 替换小括号
        user_input = user_input.replace('，', ',')# 替换中文逗号
        user_input = re.sub(r'\s+', '', user_input)#去除空格和换行符
        user_input = re.sub(r'[^\d,.\-[\]]', '', user_input)#去除非数字字符，但不包括,.-[]
        return user_input
    @classmethod
    def tolist_v2(cls,str_input,to_list=True,to_oneDim=False,to_int=False,positive=False):#转换为数组格式
        if str_input == "":
            if to_list:return ([],)
            else:return ""
        else:
            str_input = str_input.replace('，', ',')# 替换中文逗号
            if to_oneDim:
                str_input = re.sub(r'[\(\)\[\]\{\}（）【】｛｝]', "" , str_input)
                str_input = "[" + str_input + "]"
            else:
                text=re.sub(r'[\(\[\{（【｛]', '[', text)#替换括号
                text=re.sub(r'[\)\]\}）】｝]', ']', text)#替换反括号
                if str_input[0] != "[":str_input = "[" + str_input + "]"
            str_input = re.sub(r'[^\d,.\-[\]]', '', str_input)#去除非数字字符，但不包括,.-[]
            str_input = re.sub(r'(?<![0-9])[,]', '', str_input)#如果,前面不是数字则去除
            #str_input = re.sub(r'(-{2,}|\.{2,})', '', str_input)#去除多余的.和-
            str_input = re.sub(r'\.{2,}', '.', str_input)#去除多余的.
            if positive:
                str_input = re.sub(r'-','', str_input)#移除-
            else:
                str_input = re.sub(r'-{2,}', '-', str_input)#去除多余的-
            list1=np.array(ast.literal_eval(str_input))
            if to_int:
                list1=list1.astype(int)
            if to_list:
                return list1.tolist()
            else:
                return str_input
            
    def repair_brackets(cls,str_input):#括号补全(待开发)
        pass


class image_selct_batch:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "indexes": ("STRING", {"default": "1,2,"}),
            },
        }
    CATEGORY = "Apt_Collect/image"
    RETURN_TYPES = ("IMAGE", "IMAGE",)
    RETURN_NAMES = ("select_img", "exclude_img")
    FUNCTION = "SelectImages"

    def SelectImages(self, images, indexes):
        select_list = np.array(str_edit.tolist_v2(
            indexes, to_oneDim=True, to_int=True, positive=True))
        select_list1 = select_list[(select_list >= 1) & (
            select_list <= len(images))]-1
        if len(select_list1) < 1:  # 若输入的编号全部不在范围内则返回原输入
            print(
                "Warning:The input value is out of range, return to the original input.")
            return (images, [])
        else:
            exclude_list = np.arange(1, len(images) + 1)-1
            exclude_list = np.setdiff1d(exclude_list, select_list1)  # 排除的图像
            if len(select_list1) < len(select_list):  # 若输入的编号超出范围则仅输出符合编号的图像
                n = abs(len(select_list)-len(select_list1))
                print(
                    f"Warning:The maximum value entered is greater than the batch number range, {n} maximum values have been removed.")
            print(f"Selected the first {select_list1} image")
            return (images[torch.tensor(select_list1, dtype=torch.long)], images[torch.tensor(exclude_list, dtype=torch.long)])




#endregion----------select image----------



#region----------scale_match----------

def fit_resize_image(image:Image, target_width:int, target_height:int, fit:str, resize_sampler:str, background_color:str = '#000000') -> Image:
    image = image.convert('RGB')
    orig_width, orig_height = image.size
    if image is not None:
        if fit == 'letterbox':
            if orig_width / orig_height > target_width / target_height:  # 更宽，上下留黑
                fit_width = target_width
                fit_height = int(target_width / orig_width * orig_height)
            else:  # 更瘦，左右留黑
                fit_height = target_height
                fit_width = int(target_height / orig_height * orig_width)
            fit_image = image.resize((fit_width, fit_height), resize_sampler)
            ret_image = Image.new('RGB', size=(target_width, target_height), color=background_color)
            ret_image.paste(fit_image, box=((target_width - fit_width)//2, (target_height - fit_height)//2))
        elif fit == 'crop':
            if orig_width / orig_height > target_width / target_height:  # 更宽，裁左右
                fit_width = int(orig_height * target_width / target_height)
                fit_image = image.crop(
                    ((orig_width - fit_width)//2, 0, (orig_width - fit_width)//2 + fit_width, orig_height))
            else:   # 更瘦，裁上下
                fit_height = int(orig_width * target_height / target_width)
                fit_image = image.crop(
                    (0, (orig_height-fit_height)//2, orig_width, (orig_height-fit_height)//2 + fit_height))
            ret_image = fit_image.resize((target_width, target_height), resize_sampler)
        else:
            ret_image = image.resize((target_width, target_height), resize_sampler)
    return  ret_image

def tensor2pil(t_image: torch.Tensor)  -> Image:
    return Image.fromarray(np.clip(255.0 * t_image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))



def image2mask(image:Image) -> torch.Tensor:
    if image.mode == 'L':
        return torch.tensor([pil2tensor(image)[0, :, :].tolist()])
    else:
        image = image.convert('RGB').split()[0]
        return torch.tensor([pil2tensor(image)[0, :, :].tolist()])


class AnyType(str):
    def __eq__(self, _) -> bool:
        return True
    def __ne__(self, __value: object) -> bool:
        return False

ANY_TYPE = AnyType("*")


class Image_scale_match:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(self):

        fit_mode = ['letterbox', 'crop', 'fill']
        method_mode = ['lanczos', 'bicubic', 'hamming', 'bilinear', 'box', 'nearest']

        return {
            "required": {
                "scale_as": (ANY_TYPE, {}),
                "fit": (fit_mode,),
                "method": (method_mode,),
            },
            "optional": {
                "image": ("IMAGE",),  #
                "mask": ("MASK",),  #
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "BOX", )
    RETURN_NAMES = ("image", "mask", "ori_size",)
    FUNCTION = 'image_mask_scale_as'
    CATEGORY = "Apt_Collect/image"
    def image_mask_scale_as(self, scale_as, fit, method,
                            image=None, mask = None,
                            ):
        if scale_as.shape[0] > 0:
            _asimage = tensor2pil(scale_as[0])
        else:
            _asimage = tensor2pil(scale_as)
        target_width, target_height = _asimage.size
        _mask = Image.new('L', size=_asimage.size, color='black')
        _image = Image.new('RGB', size=_asimage.size, color='black')
        orig_width = 4
        orig_height = 4
        resize_sampler = Image.LANCZOS
        if method == "bicubic":
            resize_sampler = Image.BICUBIC
        elif method == "hamming":
            resize_sampler = Image.HAMMING
        elif method == "bilinear":
            resize_sampler = Image.BILINEAR
        elif method == "box":
            resize_sampler = Image.BOX
        elif method == "nearest":
            resize_sampler = Image.NEAREST

        ret_images = []
        ret_masks = []
        
        if image is not None:
            for i in image:
                i = torch.unsqueeze(i, 0)
                _image = tensor2pil(i).convert('RGB')
                orig_width, orig_height = _image.size
                _image = fit_resize_image(_image, target_width, target_height, fit, resize_sampler)
                ret_images.append(pil2tensor(_image))
        if mask is not None:
            if mask.dim() == 2:
                mask = torch.unsqueeze(mask, 0)
            for m in mask:
                m = torch.unsqueeze(m, 0)
                _mask = tensor2pil(m).convert('L')
                orig_width, orig_height = _mask.size
                _mask = fit_resize_image(_mask, target_width, target_height, fit, resize_sampler).convert('L')
                ret_masks.append(image2mask(_mask))
        if len(ret_images) > 0 and len(ret_masks) >0:

            return (torch.cat(ret_images, dim=0), torch.cat(ret_masks, dim=0), [orig_width, orig_height],target_width, target_height,)
        elif len(ret_images) > 0 and len(ret_masks) == 0:

            return (torch.cat(ret_images, dim=0), None, [orig_width, orig_height],target_width, target_height,)
        elif len(ret_images) == 0 and len(ret_masks) > 0:

            return (None, torch.cat(ret_masks, dim=0), [orig_width, orig_height], target_width, target_height,)
        else:

            return (None, None, [orig_width, orig_height], 0, 0,)

#endregion----------scale_match----------


class Image_overlay:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "image_overlay": ("IMAGE",),
                "width": ("INT", {"default": 512, "min": -48000, "max": 48000, "step": 1}),
                "height": ("INT", {"default": 512, "min": -48000, "max": 48000, "step": 1}),
                "X": ("INT", {"default": 0, "min": -48000, "max": 48000, "step": 1}),
                "Y": ("INT", {"default": 0, "min": -48000, "max": 48000, "step": 1}),
                "rotation": ("INT", {"default": 0, "min": -360, "max": 360, "step": 1}),
                "feathering": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "image_transpose"
    CATEGORY = "Apt_Collect/image"


    def image_transpose(self, image: torch.Tensor, image_overlay: torch.Tensor, width: int, height: int, X: int, Y: int, rotation: int, feathering: int = 0):
        return (pil2tensor(self.apply_transpose_image(tensor2pil(image), tensor2pil(image_overlay), (width, height), (X, Y), rotation, feathering)), )

    def apply_transpose_image(self, image_bg, image_element, size, loc, rotate=0, feathering=0):

        image_element = image_element.rotate(rotate, expand=True)
        image_element = image_element.resize(size)

        if feathering > 0:
            mask = Image.new('L', image_element.size, 255)  
            draw = ImageDraw.Draw(mask)
            for i in range(feathering):
                alpha_value = int(255 * (i + 1) / feathering)  
                draw.rectangle((i, i, image_element.size[0] - i, image_element.size[1] - i), fill=alpha_value)
            alpha_mask = Image.merge('RGBA', (mask, mask, mask, mask))
            image_element = Image.composite(image_element, Image.new('RGBA', image_element.size, (0, 0, 0, 0)), alpha_mask)

        new_image = Image.new('RGBA', image_bg.size, (0, 0, 0, 0))
        new_image.paste(image_element, loc)

        image_bg = image_bg.convert('RGBA')
        image_bg.paste(new_image, (0, 0), new_image)

        return image_bg



class Image_overlay_mask:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_image": ("IMAGE",),
                "overlay_image": ("IMAGE",),
                "overlay_resize": (["None", "Fit", "Resize by rescale_factor", "Resize to width & heigth"],),
                "resize_method": (["nearest-exact", "bilinear", "area"],),
                "rescale_factor": ("FLOAT", {"default": 1, "min": 0.01, "max": 16.0, "step": 0.1}),
                "width": ("INT", {"default": 512, "min": 0, "max": 48000, "step": 64}),
                "height": ("INT", {"default": 512, "min": 0, "max": 48000, "step": 64}),
                "x_offset": ("INT", {"default": 0, "min": -48000, "max": 48000, "step": 10}),
                "y_offset": ("INT", {"default": 0, "min": -48000, "max": 48000, "step": 10}),
                "rotation": ("INT", {"default": 0, "min": -180, "max": 180, "step": 5}),
                "opacity": ("FLOAT", {"default": 0, "min": 0, "max": 100, "step": 5}),
            },
            "optional": {"optional_mask": ("MASK",),}
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_overlay_image"
    CATEGORY = "Apt_Collect/image"

    def apply_overlay_image(self, base_image, overlay_image, overlay_resize, resize_method, rescale_factor,
                            width, height, x_offset, y_offset, rotation, opacity, optional_mask=None):

        size = width, height
        location = x_offset, y_offset
        mask = optional_mask

        if overlay_resize != "None":
            overlay_image_size = overlay_image.size()
            overlay_image_size = (overlay_image_size[2], overlay_image_size[1])
            if overlay_resize == "Fit":
                h_ratio = base_image.size()[1] / overlay_image_size[1]
                w_ratio = base_image.size()[2] / overlay_image_size[0]
                ratio = min(h_ratio, w_ratio)
                overlay_image_size = tuple(round(dimension * ratio) for dimension in overlay_image_size)
            elif overlay_resize == "Resize by rescale_factor":
                overlay_image_size = tuple(int(dimension * rescale_factor) for dimension in overlay_image_size)
            elif overlay_resize == "Resize to width & heigth":
                overlay_image_size = (size[0], size[1])

            samples = overlay_image.movedim(-1, 1)
            overlay_image = comfy.utils.common_upscale(samples, overlay_image_size[0], overlay_image_size[1], resize_method, False)
            overlay_image = overlay_image.movedim(1, -1)
            
        overlay_image = tensor2pil(overlay_image)

        overlay_image = overlay_image.convert('RGBA')
        overlay_image.putalpha(Image.new("L", overlay_image.size, 255))

        if mask is not None:
            mask = tensor2pil(mask)
            mask = mask.resize(overlay_image.size)
            overlay_image.putalpha(ImageOps.invert(mask))

        # Rotate the overlay image
        overlay_image = overlay_image.rotate(rotation, expand=True)

        r, g, b, a = overlay_image.split()
        a = a.point(lambda x: max(0, int(x * (1 - opacity / 100))))
        overlay_image.putalpha(a)

        base_image_list = torch.unbind(base_image, dim=0)

        processed_base_image_list = []
        for tensor in base_image_list:
            # Convert tensor to PIL Image
            image = tensor2pil(tensor)

            if mask is None:
                image.paste(overlay_image, location)
            else:
                image.paste(overlay_image, location, overlay_image)

            processed_tensor = pil2tensor(image)
            processed_base_image_list.append(processed_tensor)
        base_image = torch.stack([tensor.squeeze() for tensor in processed_base_image_list])

        return (base_image,)



class Image_overlay_composite:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "bg_image": ("IMAGE",),
                "batch_image": ("IMAGE",),
                "x": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 1}),
                "y": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 1}),
                "resize_source": ("BOOLEAN", {"default": False}),
                "Invert": ("BOOLEAN", {"default": False}),

            },
            "optional": {
                "batch_mask": ("MASK",),
            }
        }
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "composite"
    CATEGORY = "Apt_Collect/image"

    def composite(self, batch_image, bg_image, x, y, resize_source, Invert, batch_mask = None):
        if Invert and batch_mask is not None:
            batch_mask = 1 - batch_mask
        batch_image = batch_image.clone().movedim(-1, 1)


        output = composite(batch_image, bg_image.movedim(-1, 1), x, y, batch_mask, 1, resize_source).movedim(1, -1)
        return (output,)





class Image_overlay_transform:


    _blend_methods = [
        "add", "subtract", "multiply", "screen", "overlay", "difference", "hard_light", "soft_light",
        "add_modulo", "blend", "darker", "duplicate", "lighter", "subtract_modulo"
    ]

    def __init__(self): pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "bg_image": ("IMAGE", ), "overlay_image": ("IMAGE", ), "mask_image": ("MASK", ),
                "opacity": ("FLOAT", {"default": 100.0, "min": 0.0, "max": 100.0, "step": 0.5}),
                "overlay_x": ("INT", {"default": 0, "min": -250, "max": 250, "step": 1}),
                "overlay_y": ("INT", {"default": 0, "min": -250, "max": 250, "step": 1}),
                "overlay_fit": (["left", "right", "center", "top", "top left", "top right", "bottom", "bottom left", "bottom right"], {"default": "center"}),
                "mask_x": ("INT", {"default": 0, "min": -250, "max": 250, "step": 1}),
                "mask_y": ("INT", {"default": 0, "min": -250, "max": 250, "step": 1}),
                "mask_fit": (["left", "right", "center", "top", "top left", "top right", "bottom", "bottom left", "bottom right", "zoom_left", "zoom_center", "zoom_right", "fit"], {"default": "center"}),
                "mask_zoom": ("INT", {"default": 0, "min": -250, "max": 250, "step": 1}),
                "mask_stretch_x": ("INT", {"default": 0, "min": -250, "max": 250, "step": 1}),
                "mask_stretch_y": ("INT", {"default": 0, "min": -250, "max": 250, "step": 1}),
                "outline": ("BOOLEAN", {"default": False}),
                "outline_thickness": ("INT", {"default": 2, "min": 1, "max": 50, "step": 1}),
                "outline_color": ("COLOR",),
                "outline_opacity": ("FLOAT", {"default": 100.0, "min": 0.0, "max": 100.0, "step": 0.5}),
                "outline_position": (["center", "inside", "outside"], {"default": "center"}),
                "blend": ("BOOLEAN", {"default": False}),
                "blend_method": (cls._blend_methods, {"default": "add"}),
                "blend_strength": ("FLOAT", {"default": 50.0, "min": 0.0, "max": 100.0, "step": 0.5}),
                "blend_area": (["all", "inside", "outside", "outline"], {"default": "all"})
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", )
    RETURN_NAMES = ("image", "invert_image", "contour_image", )
    FUNCTION = "paste_with_mask"
    CATEGORY = "Apt_Collect/image"

    def hex_to_rgb(self, hex_color): 
        hex_color = hex_color.lstrip("#")
        return tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))

    def hex_to_rgba(self, hex_color, alpha): 
        r, g, b = self.hex_to_rgb(hex_color)
        return (r, g, b, int(alpha * 255 / 100))

    def blend_images(self, img1, img2, method, strength):
        blended = None
        if method == "add":
            blended = ImageChops.add(img1, img2)
        elif method == "subtract":
            blended = ImageChops.subtract(img1, img2)
        elif method == "multiply":
            blended = ImageChops.multiply(img1, img2)
        elif method == "screen":
            blended = ImageChops.screen(img1, img2)
        elif method == "overlay":
            blended = ImageChops.overlay(img1, img2)
        elif method == "difference":
            blended = ImageChops.difference(img1, img2)
        elif method == "hard_light":
            blended = ImageChops.hard_light(img1, img2)
        elif method == "soft_light":
            blended = ImageChops.soft_light(img1, img2)
        elif method == "add_modulo":
            blended = ImageChops.add_modulo(img1, img2)
        elif method == "blend":
            blended = ImageChops.blend(img1, img2, strength / 100.0)
        elif method == "darker":
            blended = ImageChops.darker(img1, img2)
        elif method == "duplicate":
            blended = ImageChops.duplicate(img1)
        elif method == "lighter":
            blended = ImageChops.lighter(img1, img2)
        elif method == "subtract_modulo":
            blended = ImageChops.subtract_modulo(img1, img2)
        else:
            raise ValueError("Unsupported blend method")
        
        return Image.blend(img1, blended, strength / 100.0)

    def resize_and_fit_image(self, bg_image, image, fit_option, x_offset, y_offset, mask_zoom, mask_stretch_x, mask_stretch_y, is_mask=False):
        bg_h, bg_w = bg_image.shape[1], bg_image.shape[2]
        image_pil = Image.fromarray((image.squeeze().cpu().numpy() * 255).astype(np.uint8))
        if is_mask:
            image_pil = image_pil.convert("L")
        image_w, image_h = image_pil.size

        # Apply mask zoom
        if mask_zoom != 0:
            new_width = image_w + mask_zoom
            new_height = image_h + mask_zoom
            image_pil = image_pil.resize((new_width, new_height), Image.LANCZOS)
            image_w, image_h = new_width, new_height

        # Apply mask stretch x
        if mask_stretch_x != 0:
            new_width = image_w + mask_stretch_x
            image_pil = image_pil.resize((new_width, image_h), Image.LANCZOS)
            image_w = new_width

        # Apply mask stretch y
        if mask_stretch_y != 0:
            new_height = image_h + mask_stretch_y
            image_pil = image_pil.resize((image_w, new_height), Image.LANCZOS)
            image_h = new_height

        if is_mask:
            if fit_option == "fit":
                # Convert image to binary mask
                mask_np = np.array(image_pil) > 0
                # Find bounding box
                coords = np.column_stack(np.where(mask_np))
                y_min, x_min = coords.min(axis=0)
                y_max, x_max = coords.max(axis=0)
                bbox = (x_min, y_min, x_max, y_max)
                # Crop the mask to the bounding box
                image_pil = image_pil.crop(bbox)
                image_w, image_h = image_pil.size

                # Resize to fit the background image while keeping proportions
                scale = min(bg_w / image_w, bg_h / image_h)
                new_size = (int(image_w * scale), int(image_h * scale))
                image_pil = image_pil.resize(new_size, Image.LANCZOS)

                # Center the mask on the background
                left = (bg_w - new_size[0]) // 2 + x_offset
                top = (bg_h - new_size[1]) // 2 + y_offset
                new_image = Image.new("L", (bg_w, bg_h))
                new_image.paste(image_pil, (left, top))
                image_pil = new_image
            elif fit_option in ["zoom_left", "zoom_center", "zoom_right"]:
                if bg_h > bg_w:
                    new_height = bg_h
                    new_width = int((new_height / image_h) * image_w)
                    image_pil = image_pil.resize((new_width, new_height), Image.LANCZOS)
                    if fit_option == "zoom_left":
                        image_pil = image_pil.crop((0 + x_offset, 0 + y_offset, bg_w + x_offset, bg_h + y_offset))
                    elif fit_option == "zoom_right":
                        image_pil = image_pil.crop((new_width - bg_w + x_offset, 0 + y_offset, new_width + x_offset, bg_h + y_offset))
                    else:  # zoom_center
                        left = (new_width - bg_w) // 2
                        image_pil = image_pil.crop((left + x_offset, 0 + y_offset, left + bg_w + x_offset, bg_h + y_offset))
                else:
                    scale = min(bg_w / image_w, bg_h / image_h)
                    new_size = (int(image_w * scale), int(image_h * scale))
                    image_pil = image_pil.resize(new_size, Image.LANCZOS)
                    left = top = 0
                    if "top" in fit_option:
                        top = 0
                    elif "bottom" in fit_option:
                        top = image_pil.size[1] - bg_h
                    else:
                        top = (image_pil.size[1] - bg_h) // 2

                    if "left" in fit_option:
                        left = 0
                    elif "right" in fit_option:
                        left = image_pil.size[0] - bg_w
                    else:
                        left = (image_pil.size[0] - bg_w) // 2

                    image_pil = image_pil.crop((left + x_offset, top + y_offset, left + bg_w + x_offset, top + bg_h + y_offset))
            else:
                if bg_h < bg_w and image_h > image_w:
                    new_width = bg_w
                    new_height = int((new_width / image_w) * image_h)
                    image_pil = image_pil.resize((new_width, new_height), Image.LANCZOS)
                    top = (new_height - bg_h) // 2 if fit_option == "center" else 0 if "top" in fit_option else new_height - bg_h
                    left = 0
                    if "left" in fit_option:
                        left = 0
                    elif "right" in fit_option:
                        left = new_width - bg_w
                    else:
                        left = (new_width - bg_w) // 2
                    image_pil = image_pil.crop((left + x_offset, top + y_offset, left + bg_w + x_offset, top + bg_h + y_offset))
                elif bg_h > bg_w and image_h < image_w:
                    new_width = bg_w
                    new_height = int((new_width / image_w) * image_h)
                    image_pil = image_pil.resize((new_width, new_height), Image.LANCZOS)
                    top = (bg_h - new_height) // 2 if fit_option == "center" else 0 if "top" in fit_option else bg_h - new_height
                    left = 0
                    if "left" in fit_option:
                        left = 0
                    elif "right" in fit_option:
                        left = bg_w - new_width
                    else:
                        left = (bg_w - new_width) // 2
                    new_image = Image.new("L", (bg_w, bg_h))
                    new_image.paste(image_pil, (left + x_offset, top + y_offset))
                    image_pil = new_image
                else:
                    scale = min(bg_w / image_w, bg_h / image_h)
                    new_size = (int(image_w * scale), int(image_h * scale))
                    image_pil = image_pil.resize(new_size, Image.LANCZOS)
                    left = top = 0
                    if "top" in fit_option:
                        top = 0
                    elif "bottom" in fit_option:
                        top = image_pil.size[1] - bg_h
                    else:
                        top = (image_pil.size[1] - bg_h) // 2

                    if "left" in fit_option:
                        left = 0
                    elif "right" in fit_option:
                        left = image_pil.size[0] - bg_w
                    else:
                        left = (image_pil.size[0] - bg_w) // 2

                    image_pil = image_pil.crop((left + x_offset, top + y_offset, left + bg_w + x_offset, top + bg_h + y_offset))
        else:
            if bg_h < bg_w and image_h > image_w:
                new_width = bg_w
                new_height = int((new_width / image_w) * image_h)
                image_pil = image_pil.resize((new_width, new_height), Image.LANCZOS)
            elif bg_h > bg_w and image_h < image_w:
                new_height = bg_h
                new_width = int((new_height / image_h) * image_w)
                image_pil = image_pil.resize((new_width, new_height), Image.LANCZOS)
            else:
                scale = min(bg_w / image_w, bg_h / image_h)
                new_size = (int(image_w * scale), int(image_h * scale))
                image_pil = image_pil.resize(new_size, Image.LANCZOS)

            left = top = 0
            if "top" in fit_option:
                top = 0
            elif "bottom" in fit_option:
                top = image_pil.size[1] - bg_h
            else:
                top = (image_pil.size[1] - bg_h) // 2

            if "left" in fit_option:
                left = 0
            elif "right" in fit_option:
                left = image_pil.size[0] - bg_w
            else:
                left = (image_pil.size[0] - bg_w) // 2

            image_pil = image_pil.crop((left + x_offset, top + y_offset, left + bg_w + x_offset, top + bg_h + y_offset))

        return torch.tensor(np.array(image_pil).astype(np.float32) / 255.0).unsqueeze(0)

    def paste_with_mask(self, bg_image, overlay_image, mask_image, opacity, overlay_x, overlay_y, overlay_fit, mask_x, mask_y, mask_fit, mask_zoom, mask_stretch_x, mask_stretch_y, outline, outline_thickness, outline_color, outline_opacity, outline_position, blend, blend_method, blend_strength, blend_area):
        overlay_image = self.resize_and_fit_image(bg_image, overlay_image, overlay_fit, overlay_x, overlay_y, 0, 0, 0, is_mask=False)
        mask_image = self.resize_and_fit_image(bg_image, mask_image, mask_fit, mask_x, mask_y, mask_zoom, mask_stretch_x, mask_stretch_y, is_mask=True)

        mask_image = mask_image.unsqueeze(1)
        mask_image = mask_image.expand(bg_image.shape[0], bg_image.shape[3], bg_image.shape[1], bg_image.shape[2]).permute(0, 2, 3, 1)
        opacity /= 100.0
        output_image = (bg_image * (1 - mask_image * opacity) + overlay_image * (mask_image * opacity)).clamp(0, 1)
        inverted_mask_image = (bg_image * mask_image * opacity + overlay_image * (1 - mask_image * opacity)).clamp(0, 1)

        if blend:
            output_image_pil = Image.fromarray((output_image.squeeze().cpu().numpy() * 255).astype(np.uint8))
            overlay_image_pil = Image.fromarray((overlay_image.squeeze().cpu().numpy() * 255).astype(np.uint8))
            inverted_mask_image_pil = Image.fromarray((inverted_mask_image.squeeze().cpu().numpy() * 255).astype(np.uint8))

            if blend_area == "all":
                output_image_pil = self.blend_images(output_image_pil, overlay_image_pil, blend_method, blend_strength)
                inverted_mask_image_pil = self.blend_images(inverted_mask_image_pil, overlay_image_pil, blend_method, blend_strength)
            elif blend_area == "inside":
                mask_np = (mask_image.squeeze().cpu().numpy() * 255).astype(np.uint8)
                mask_pil = Image.fromarray(mask_np).convert("L")
                blended_inside = self.blend_images(output_image_pil, overlay_image_pil, blend_method, blend_strength)
                output_image_pil = Image.composite(blended_inside, output_image_pil, mask_pil)
                blended_inside_inverted = self.blend_images(inverted_mask_image_pil, overlay_image_pil, blend_method, blend_strength)
                inverted_mask_image_pil = Image.composite(blended_inside_inverted, inverted_mask_image_pil, mask_pil)
            elif blend_area == "outside":
                mask_np = ((1 - mask_image).squeeze().cpu().numpy() * 255).astype(np.uint8)
                mask_pil = Image.fromarray(mask_np).convert("L")
                blended_outside = self.blend_images(output_image_pil, overlay_image_pil, blend_method, blend_strength)
                output_image_pil = Image.composite(blended_outside, output_image_pil, mask_pil)
                blended_outside_inverted = self.blend_images(inverted_mask_image_pil, overlay_image_pil, blend_method, blend_strength)
                inverted_mask_image_pil = Image.composite(blended_outside_inverted, inverted_mask_image_pil, mask_pil)
            elif blend_area == "outline":
                mask_np = (mask_image.squeeze().cpu().numpy() * 255).astype(np.uint8)
                if len(mask_np.shape) > 2: mask_np = mask_np[:, :, 0]
                contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                outline_mask = Image.new("L", (mask_np.shape[1], mask_np.shape[0]), 0)
                outline_draw = ImageDraw.Draw(outline_mask)
                for contour in contours:
                    contour_list = [tuple(point[0]) for point in contour]
                    if len(contour_list) > 2:
                        offset = outline_thickness // 2
                        offset_contour = [tuple(np.array(point) - offset if outline_position == "inside" else np.array(point) + offset if outline_position == "outside" else point) for point in contour_list]
                        outline_draw.line(offset_contour + [offset_contour[0]], fill=255, width=outline_thickness)
                outline_mask = outline_mask.convert("L")
                blended_outline = self.blend_images(output_image_pil, overlay_image_pil, blend_method, blend_strength)
                output_image_pil = Image.composite(blended_outline, output_image_pil, outline_mask)
                blended_outline_inverted = self.blend_images(inverted_mask_image_pil, overlay_image_pil, blend_method, blend_strength)
                inverted_mask_image_pil = Image.composite(blended_outline_inverted, inverted_mask_image_pil, outline_mask)

            output_image = torch.tensor(np.array(output_image_pil).astype(np.float32) / 255.0).unsqueeze(0)
            inverted_mask_image = torch.tensor(np.array(inverted_mask_image_pil).astype(np.float32) / 255.0).unsqueeze(0)

        contour_image = torch.zeros_like(output_image)
        if outline:
            mask_np = (mask_image.squeeze().cpu().numpy() > 0.5).astype(np.uint8)
            if len(mask_np.shape) > 2: mask_np = mask_np[:, :, 0]
            contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            outline_colour_rgba = self.hex_to_rgba(outline_color, outline_opacity)
            output_image_np = (output_image.squeeze().cpu().numpy() * 255).astype(np.uint8)
            output_image_pil, contour_image_pil = Image.fromarray(output_image_np), Image.fromarray(np.zeros_like(output_image_np))
            draw, contour_draw = ImageDraw.Draw(output_image_pil, "RGBA"), ImageDraw.Draw(contour_image_pil, "RGBA")
            for contour in contours:
                contour_list = [tuple(point[0]) for point in contour]
                if len(contour_list) > 2:
                    offset = outline_thickness // 2
                    offset_contour = [tuple(np.array(point) - offset if outline_position == "inside" else np.array(point) + offset if outline_position == "outside" else point) for point in contour_list]
                    draw.line(offset_contour + [offset_contour[0]], fill=outline_colour_rgba, width=outline_thickness)
                    contour_draw.line(contour_list + [contour_list[0]], fill=outline_colour_rgba, width=outline_thickness)
            
            inverted_mask_image_np = (inverted_mask_image.squeeze().cpu().numpy() * 255).astype(np.uint8)
            inverted_mask_image_pil = Image.fromarray(inverted_mask_image_np)
            inverted_draw = ImageDraw.Draw(inverted_mask_image_pil, "RGBA")
            for contour in contours:
                contour_list = [tuple(point[0]) for point in contour]
                if len(contour_list) > 2:
                    offset = outline_thickness // 2
                    offset_contour = [tuple(np.array(point) - offset if outline_position == "inside" else np.array(point) + offset if outline_position == "outside" else point) for point in contour_list]
                    inverted_draw.line(offset_contour + [offset_contour[0]], fill=outline_colour_rgba, width=outline_thickness)

            output_image = torch.tensor(np.array(output_image_pil).astype(np.float32) / 255.0).unsqueeze(0)
            contour_image = torch.tensor(np.array(contour_image_pil).astype(np.float32) / 255.0).unsqueeze(0)
            inverted_mask_image = torch.tensor(np.array(inverted_mask_image_pil).astype(np.float32) / 255.0).unsqueeze(0)

        #image_dimensions = f"Width: {bg_image.shape[2]}, Height: {bg_image.shape[1]}"
        
        return output_image, inverted_mask_image, contour_image



#region------overlay sum----------------
import numpy as np
from rembg import remove, new_session
from PIL import Image, ImageOps, ImageFilter, ImageEnhance, ImageDraw, ImageFont
import torch
import logging
import cv2
from tqdm import tqdm
import onnxruntime as ort
from transformers import pipeline, AutoModelForImageSegmentation
from enum import Enum
import random
import math
import os
from torchvision import transforms

logging.basicConfig(level=logging.INFO)


class BlendingMode(Enum):
    NORMAL = "normal"
    MULTIPLY = "multiply"
    SCREEN = "screen"
    OVERLAY = "overlay"
    DARKEN = "darken"
    LIGHTEN = "lighten"
    COLOR_DODGE = "color_dodge"
    COLOR_BURN = "color_burn"
    LINEAR_DODGE = "linear_dodge"
    LINEAR_BURN = "linear_burn"
    HARD_LIGHT = "hard_light"
    SOFT_LIGHT = "soft_light"
    VIVID_LIGHT = "vivid_light"
    LINEAR_LIGHT = "linear_light"
    PIN_LIGHT = "pin_light"
    DIFFERENCE = "difference"
    EXCLUSION = "exclusion"
    SUBTRACT = "subtract"

class AnimationType(Enum):
    NONE = "none"
    BOUNCE = "bounce"
    TRAVEL_LEFT = "travel_left"
    TRAVEL_RIGHT = "travel_right"
    ROTATE = "rotate"
    FADE_IN = "fade_in"
    FADE_OUT = "fade_out"
    ZOOM_IN = "zoom_in"
    ZOOM_OUT = "zoom_out"

def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

def pil2tensor(image):
    np_image = np.array(image).astype(np.float32) / 255.0
    if np_image.ndim == 2:
        np_image = np_image[None, None, ...]
    elif np_image.ndim == 3:
        np_image = np_image[None, ...]
    return torch.from_numpy(np_image)

class Image_overlay_sum:
    def __init__(self):
        self.session = None
        self.bria_pipeline = None
        self.birefnet = None
        self.use_gpu = 'CUDAExecutionProvider' in ort.get_available_providers()
        self.custom_model_path = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "foreground": ("IMAGE",),
                                "output_format": (["RGBA", "RGB"],),
                "enable_background_removal": ("BOOLEAN", {"default": True}),
                "model": (["u2net", "u2netp", "u2net_human_seg", "u2net_cloth_seg", "silueta", "isnet-general-use", "isnet-anime", "bria", "inspyrenet", "tracer", "basnet", "deeplab", "ormbg", "u2net_custom", "birefnet"],),
                "alpha_matting": ("BOOLEAN", {"default": False}),
                "alpha_matting_foreground_threshold": ("INT", {"default": 240, "min": 0, "max": 255, "step": 1}),
                "alpha_matting_background_threshold": ("INT", {"default": 10, "min": 0, "max": 255, "step": 1}),
                "post_process_mask": ("BOOLEAN", {"default": False}),

                "background_mode": (["transparent", "color", "image",],),
                "background_color": ("COLOR", {"default": "#000000"}),
                "blending_mode": ([mode.value for mode in BlendingMode],),
                "blend_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),

                "foreground_scale": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 5.0, "step": 0.1}),
                "x_position": ("INT", {"default": 0, "min": -10000, "max": 10000, "step": 1}),
                "y_position": ("INT", {"default": 0, "min": -10000, "max": 10000, "step": 1}),
                "rotation": ("FLOAT", {"default": 0, "min": -360, "max": 360, "step": 0.1}),
                "opacity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "flip_horizontal": ("BOOLEAN", {"default": False}),
                "flip_vertical": ("BOOLEAN", {"default": False}),
                "invert_mask": ("BOOLEAN", {"default": False}),
                "mask_blur": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1}),
                "mask_expansion": ("INT", {"default": 0, "min": -100, "max": 100, "step": 1}),
                "edge_color": ("COLOR", {"default": "#FFFFFF"}),
                "edge_detection": ("BOOLEAN", {"default": False}),
                "edge_thickness": ("INT", {"default": 1, "min": 1, "max": 10, "step": 1}),


            },
            "optional": {
                "background": ("IMAGE",),
                "input_masks": ("MASK",),


            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "process_image"
    CATEGORY = "Apt_Collect/image"

    def ensure_image_format(self, image):
        img_np = np.array(image)
        if img_np.dtype != np.uint8:
            img_np = (img_np * 255).astype(np.uint8)
        if len(img_np.shape) == 2:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
        elif img_np.shape[2] == 4:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB)
        return img_np

    def apply_chroma_key(self, image, color, threshold, color_tolerance=20):
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        if color == "green":
            lower = np.array([40 - color_tolerance, 40, 40])
            upper = np.array([80 + color_tolerance, 255, 255])
        elif color == "blue":
            lower = np.array([90 - color_tolerance, 40, 40])
            upper = np.array([130 + color_tolerance, 255, 255])
        elif color == "red":
            lower = np.array([0, 40, 40])
            upper = np.array([20 + color_tolerance, 255, 255])
        else:
            return np.zeros(image.shape[:2], dtype=np.uint8)

        mask = cv2.inRange(hsv, lower, upper)
        mask = 255 - cv2.threshold(mask, threshold, 255, cv2.THRESH_BINARY)[1]
        return mask

    def process_mask(self, mask, invert_mask, feather_amount, mask_blur, mask_expansion):
        if invert_mask:
            mask = 255 - mask
        if mask_expansion != 0:
            kernel = np.ones((abs(mask_expansion), abs(mask_expansion)), np.uint8)
            if mask_expansion > 0:
                mask = cv2.dilate(mask, kernel, iterations=1)
            else:
                mask = cv2.erode(mask, kernel, iterations=1)
        if feather_amount > 0:
            mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=feather_amount)
        if mask_blur > 0:
            mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=mask_blur)
        return mask

    def apply_color_adjustments(self, image, brightness, contrast, saturation, hue, sharpness):
        if brightness != 1.0:
            image = ImageEnhance.Brightness(image).enhance(brightness)
        if contrast != 1.0:
            image = ImageEnhance.Contrast(image).enhance(contrast)
        if saturation != 1.0:
            image = ImageEnhance.Color(image).enhance(saturation)
        if hue != 0.0:
            r, g, b = image.split()
            image = Image.merge("RGB", (
                r.point(lambda x: x + hue * 255),
                g.point(lambda x: x - hue * 255),
                b
            ))
        if sharpness != 1.0:
            image = ImageEnhance.Sharpness(image).enhance(sharpness)
        return image

    def apply_filter(self, image, filter_type, strength):
        if filter_type == "blur":
            return image.filter(ImageFilter.GaussianBlur(radius=strength * 2))
        elif filter_type == "sharpen":
            percent = int(100 + (strength * 100))
            return image.filter(ImageFilter.UnsharpMask(radius=2, percent=percent, threshold=3))
        elif filter_type == "edge_enhance":
            return image.filter(ImageFilter.EDGE_ENHANCE_MORE)
        elif filter_type == "emboss":
            return image.filter(ImageFilter.EMBOSS)
        elif filter_type == "toon":
            return self.apply_toon_filter(image, strength)
        elif filter_type == "sepia":
            return self.apply_sepia_filter(image)
        elif filter_type == "film_grain":
            return self.apply_film_grain(image, strength)
        elif filter_type == "matrix":
            return self.apply_matrix_filter(image, strength)
        else:
            return image

    def apply_toon_filter(self, image, strength):
        img_np = self.ensure_image_format(image)
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
        color = cv2.bilateralFilter(img_np, d=9, sigmaColor=300, sigmaSpace=300)
        cartoon = cv2.bitwise_and(color, color, mask=edges)
        
        # Adjust the strength of the effect
        result = cv2.addWeighted(img_np, 1 - strength, cartoon, strength, 0)
        return Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))

    def apply_sepia_filter(self, image):
        img_np = self.ensure_image_format(image)
        sepia_kernel = np.array([
            [0.393, 0.769, 0.189],
            [0.349, 0.686, 0.168],
            [0.272, 0.534, 0.131]
        ])
        sepia_img = cv2.transform(img_np, sepia_kernel)
        sepia_img = np.clip(sepia_img, 0, 255).astype(np.uint8)
        return Image.fromarray(sepia_img)

    def apply_film_grain(self, image, strength):
        img_np = self.ensure_image_format(image)
        h, w, c = img_np.shape
        noise = np.random.randn(h, w) * 10 * strength
        noise = np.dstack([noise] * 3)
        grain_img = np.clip(img_np + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(grain_img)

    def apply_matrix_filter(self, image, strength):
        img_np = self.ensure_image_format(image)
        h, w, _ = img_np.shape
        
        # Create a black background
        matrix = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Generate matrix rain effect
        for x in range(w):
            for y in range(h):
                if random.random() < strength * 0.1:
                    value = random.choice([0, 1])
                    color = (0, 255, 0) if value == 1 else (0, 100, 0)  # Bright green for 1, dark green for 0
                    matrix[y, x] = color
        
        # Add a motion blur effect
        kernel_size = int(strength * 5)
        kernel = np.zeros((kernel_size, kernel_size))
        kernel[:, int((kernel_size-1)/2)] = np.ones(kernel_size)
        kernel /= kernel_size
        matrix = cv2.filter2D(matrix, -1, kernel)
        
        # Blend the matrix with the original image
        matrix_img = cv2.addWeighted(img_np, 1-strength, matrix, strength, 0)
        return Image.fromarray(matrix_img)

    def create_matrix_background(self, width, height, foreground_pattern, custom_text, density, fall_speed, glow, glow_intensity):
        image = Image.new('RGB', (width, height), color='black')
        draw = ImageDraw.Draw(image)
        
        try:
            font = ImageFont.truetype("arial.ttf", 12)
        except IOError:
            font = ImageFont.load_default()
        
        if foreground_pattern == "BINARY":
            chars = "01"
        elif foreground_pattern == "RANDOM":
            chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789!@#$%^&*()_+-=[]{}|;:,.<>?"
        else:  # CUSTOM
            chars = custom_text
        
        drops = [0 for _ in range(width // 10)]
        for _ in range(int(height * density / 10)):
            for i in range(len(drops)):
                if random.random() < 0.1:
                    x = i * 10
                    y = drops[i] * 15
                    char = random.choice(chars)
                    color = (0, 255, 0)
                    draw.text((x, y), char, font=font, fill=color)
                    
                    if glow == "ENABLED":
                        for offset in range(1, 4):
                            alpha = int(255 * (1 - offset / 4) * glow_intensity)
                            glow_color = (0, 255, 0, alpha)
                            draw.text((x-offset, y), char, font=font, fill=glow_color)
                            draw.text((x+offset, y), char, font=font, fill=glow_color)
                            draw.text((x, y-offset), char, font=font, fill=glow_color)
                            draw.text((x, y+offset), char, font=font, fill=glow_color)
                    
                    drops[i] += fall_speed
                    if drops[i] * 15 > height:
                        drops[i] = 0
        
        return image

    def apply_blending_mode(self, bg, fg, mode, strength):
        bg_np = np.array(bg).astype(np.float32) / 255.0
        fg_np = np.array(fg).astype(np.float32) / 255.0

        if bg_np.shape[-1] == 3:
            bg_np = np.dstack([bg_np, np.ones(bg_np.shape[:2], dtype=np.float32)])
        if fg_np.shape[-1] == 3:
            fg_np = np.dstack([fg_np, np.ones(fg_np.shape[:2], dtype=np.float32)])

        bg_rgb, bg_a = bg_np[..., :3], bg_np[..., 3]
        fg_rgb, fg_a = fg_np[..., :3], fg_np[..., 3]

        if mode == BlendingMode.NORMAL.value:
            blended_rgb = fg_rgb
        elif mode == BlendingMode.MULTIPLY.value:
            blended_rgb = bg_rgb * fg_rgb
        elif mode == BlendingMode.SCREEN.value:
            blended_rgb = 1 - (1 - bg_rgb) * (1 - fg_rgb)
        elif mode == BlendingMode.OVERLAY.value:
            blended_rgb = np.where(bg_rgb <= 0.5, 2 * bg_rgb * fg_rgb, 1 - 2 * (1 - bg_rgb) * (1 - fg_rgb))
        elif mode == BlendingMode.DARKEN.value:
            blended_rgb = np.minimum(bg_rgb, fg_rgb)
        elif mode == BlendingMode.LIGHTEN.value:
            blended_rgb = np.maximum(bg_rgb, fg_rgb)
        elif mode == BlendingMode.COLOR_DODGE.value:
            blended_rgb = np.where(fg_rgb == 1, 1, np.minimum(1, bg_rgb / (1 - fg_rgb)))
        elif mode == BlendingMode.COLOR_BURN.value:
            blended_rgb = np.where(fg_rgb == 0, 0, np.maximum(0, 1 - (1 - bg_rgb) / fg_rgb))
        elif mode == BlendingMode.LINEAR_DODGE.value:
            blended_rgb = np.minimum(1, bg_rgb + fg_rgb)
        elif mode == BlendingMode.LINEAR_BURN.value:
            blended_rgb = np.maximum(0, bg_rgb + fg_rgb - 1)
        elif mode == BlendingMode.HARD_LIGHT.value:
            blended_rgb = np.where(fg_rgb <= 0.5, 2 * bg_rgb * fg_rgb, 1 - 2 * (1 - bg_rgb) * (1 - fg_rgb))
        elif mode == BlendingMode.SOFT_LIGHT.value:
            blended_rgb = np.where(fg_rgb <= 0.5,
                                   bg_rgb - (1 - 2 * fg_rgb) * bg_rgb * (1 - bg_rgb),
                                   bg_rgb + (2 * fg_rgb - 1) * (np.sqrt(bg_rgb) - bg_rgb))
        elif mode == BlendingMode.VIVID_LIGHT.value:
            blended_rgb = np.where(fg_rgb <= 0.5,
                                   np.where(fg_rgb == 0, 0, np.maximum(0, 1 - (1 - bg_rgb) / (2 * fg_rgb))),
                                   np.where(fg_rgb == 1, 1, np.minimum(1, bg_rgb / (2 * (1 - fg_rgb)))))
        elif mode == BlendingMode.LINEAR_LIGHT.value:
            blended_rgb = np.clip(bg_rgb + 2 * fg_rgb - 1, 0, 1)
        elif mode == BlendingMode.PIN_LIGHT.value:
            blended_rgb = np.where(fg_rgb <= 0.5, np.minimum(bg_rgb, 2 * fg_rgb), np.maximum(bg_rgb, 2 * fg_rgb - 1))
        elif mode == BlendingMode.DIFFERENCE.value:
            blended_rgb = np.abs(bg_rgb - fg_rgb)
        elif mode == BlendingMode.EXCLUSION.value:
            blended_rgb = bg_rgb + fg_rgb - 2 * bg_rgb * fg_rgb
        elif mode == BlendingMode.SUBTRACT.value:
            blended_rgb = np.maximum(0, bg_rgb - fg_rgb)
        else:
            blended_rgb = fg_rgb

        blended_rgb = blended_rgb * strength + bg_rgb * (1 - strength)
        blended_a = fg_a * strength + bg_a * (1 - strength)
        blended = np.dstack([blended_rgb, blended_a])

        return Image.fromarray((blended * 255.0).astype(np.uint8))

    def animate_element(self, element, element_mask, animation_type, animation_speed, frame_number, total_frames,
                        x_start, y_start, x_end, y_end, canvas_width, canvas_height):
        progress = frame_number / total_frames
        
        if animation_type == AnimationType.BOUNCE.value:
            y_offset = int(math.sin(progress * 2 * math.pi) * animation_speed * 50)
            x = x_start
            y = y_start + y_offset
        elif animation_type == AnimationType.TRAVEL_LEFT.value:
            x = int(x_start + (x_end - x_start) * (1 - progress))
            y = y_start
        elif animation_type == AnimationType.TRAVEL_RIGHT.value:
            x = int(x_start + (x_end - x_start) * progress)
            y = y_start
        elif animation_type == AnimationType.ROTATE.value:
            angle = progress * 360 * animation_speed
            element = element.rotate(angle, resample=Image.BICUBIC, expand=True)
            if element_mask:
                element_mask = element_mask.rotate(angle, resample=Image.BICUBIC, expand=True)
            x = x_start
            y = y_start
        elif animation_type == AnimationType.FADE_IN.value:
            x, y = x_start, y_start
            opacity = progress
            element = Image.blend(Image.new('RGBA', element.size, (0, 0, 0, 0)), element, opacity)
        elif animation_type == AnimationType.FADE_OUT.value:
            x, y = x_start, y_start
            opacity = 1 - progress
            element = Image.blend(Image.new('RGBA', element.size, (0, 0, 0, 0)), element, opacity)
        elif animation_type == AnimationType.ZOOM_IN.value:
            scale = 1 + progress * animation_speed
            new_size = (int(element.width * scale), int(element.height * scale))
            element = element.resize(new_size, Image.LANCZOS)
            if element_mask:
                element_mask = element_mask.resize(new_size, Image.LANCZOS)
            x = x_start - (new_size[0] - element.width) // 2
            y = y_start - (new_size[1] - element.height) // 2
        elif animation_type == AnimationType.ZOOM_OUT.value:
            scale = 1 + (1 - progress) * animation_speed
            new_size = (int(element.width * scale), int(element.height * scale))
            element = element.resize(new_size, Image.LANCZOS)
            if element_mask:
                element_mask = element_mask.resize(new_size, Image.LANCZOS)
            x = x_start - (new_size[0] - element.width) // 2
            y = y_start - (new_size[1] - element.height) // 2
        else:  # NONE
            x = x_start
            y = y_start

        return element, element_mask, x, y

    def process_image(self, foreground, enable_background_removal, model, alpha_matting, alpha_matting_foreground_threshold, 
                      alpha_matting_background_threshold, post_process_mask, 
                      background_mode, background_color, blending_mode,
                      blend_strength, foreground_scale, x_position, y_position,
                      rotation, opacity, flip_horizontal, flip_vertical, invert_mask,  edge_detection,
                      edge_thickness, edge_color, 
                      mask_blur, mask_expansion,
                      
                      background=None, input_masks=None, output_format="RGBA", only_mask=False, custom_model_path=""):

        feather_amount = 1
        animation_frames= 1





        if enable_background_removal:
            if model == "bria":
                if self.bria_pipeline is None:
                    self.bria_pipeline = pipeline("image-segmentation", model="briaai/RMBG-1.4", trust_remote_code=True)
            elif model == "birefnet":
                if self.birefnet is None:
                    self.birefnet = AutoModelForImageSegmentation.from_pretrained('zhengpeng7/BiRefNet', trust_remote_code=True)
                    self.birefnet.to(self.device)
                    self.birefnet.eval()
            elif self.session is None or self.session.model_name != model:
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if self.use_gpu else ['CPUExecutionProvider']
                if model == 'u2net_custom' and custom_model_path:
                    self.session = new_session('u2net_custom', model_path=custom_model_path, providers=providers)
                else:
                    self.session = new_session(model, providers=providers)

        bg_color = background_color
        edge_color = edge_color

        def process_single_image(fg_image, bg_image=None, input_mask=None, frame_number=0):
            fg_pil = tensor2pil(fg_image)
            bg_pil = tensor2pil(bg_image) if bg_image is not None else None
            original_fg = np.array(fg_pil)
            
            if input_mask is not None:
                input_mask_np = input_mask.squeeze().cpu().numpy().astype(np.uint8) * 255
            else:
                input_mask_np = None

            if enable_background_removal:

                if model == "bria":
                    removed_bg = self.bria_pipeline(fg_pil, return_mask=True)
                    rembg_mask = np.array(removed_bg)
                elif model == "birefnet":
                    transform_image = transforms.Compose([
                        transforms.Resize((1024, 1024)),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    ])
                    input_tensor = transform_image(fg_pil).unsqueeze(0).to(self.device)
                    with torch.no_grad():
                        pred = self.birefnet(input_tensor)[-1].sigmoid().cpu()
                    rembg_mask = (pred[0].squeeze().numpy() * 255).astype(np.uint8)
                else:
                    removed_bg = remove(
                        fg_pil,
                        session=self.session,
                        alpha_matting=alpha_matting,
                        alpha_matting_foreground_threshold=alpha_matting_foreground_threshold,
                        alpha_matting_background_threshold=alpha_matting_background_threshold,
                        post_process_mask=post_process_mask,
                    )
                    rembg_mask = np.array(removed_bg)[:,:,3]

                if input_mask_np is not None:
                    final_mask = cv2.bitwise_and(rembg_mask, input_mask_np)
                else:
                    final_mask = rembg_mask

                final_mask = self.process_mask(final_mask, invert_mask, feather_amount, mask_blur, mask_expansion)
            else:
                if input_mask_np is not None:
                    final_mask = input_mask_np
                else:
                    final_mask = np.full(fg_pil.size[::-1], 255, dtype=np.uint8)

            if only_mask:
                return pil2tensor(Image.fromarray(final_mask)), pil2tensor(Image.fromarray(final_mask))

            orig_width, orig_height = fg_pil.size
            new_width, new_height = orig_width, orig_height


            if background_mode == "transparent":
                result = Image.new("RGBA", (new_width, new_height), (0, 0, 0, 0))
            elif background_mode == "color":
                result = Image.new("RGBA", (new_width, new_height), bg_color)

            elif background_mode == "image" and bg_pil is not None:
                result = bg_pil.convert("RGBA").resize((new_width, new_height), Image.LANCZOS)
            else:
                result = Image.new("RGBA", (new_width, new_height), (0, 0, 0, 0))

            fg_width = int(new_width * foreground_scale)
            fg_height = int(new_height * foreground_scale)

            fg_pil = fg_pil.resize((fg_width, fg_height), Image.LANCZOS)
            fg_mask = Image.fromarray(final_mask).resize((fg_width, fg_height), Image.LANCZOS)

            if flip_horizontal:
                fg_pil = fg_pil.transpose(Image.FLIP_LEFT_RIGHT)
                fg_mask = fg_mask.transpose(Image.FLIP_LEFT_RIGHT)
            if flip_vertical:
                fg_pil = fg_pil.transpose(Image.FLIP_TOP_BOTTOM)
                fg_mask = fg_mask.transpose(Image.FLIP_TOP_BOTTOM)

            fg_pil = fg_pil.rotate(rotation, resample=Image.BICUBIC, expand=True)
            fg_mask = fg_mask.rotate(rotation, resample=Image.BICUBIC, expand=True)



            # 使用 x_position 和 y_position 来确定粘贴位置
            paste_x = x_position
            paste_y = y_position



            bg_subset = result.crop((paste_x, paste_y, paste_x + fg_pil.width, paste_y + fg_pil.height))
            blended = self.apply_blending_mode(bg_subset, fg_pil, blending_mode, blend_strength)

            blended_rgba = blended.convert("RGBA")
            fg_with_opacity = Image.new("RGBA", blended_rgba.size, (0, 0, 0, 0))
            fg_data = blended_rgba.getdata()
            new_data = [(r, g, b, int(a * opacity)) for r, g, b, a in fg_data]
            fg_with_opacity.putdata(new_data)

            fg_mask_with_opacity = fg_mask.point(lambda p: int(p * opacity))

            result.paste(fg_with_opacity, (paste_x, paste_y), fg_mask_with_opacity)

            if edge_detection:
                edge_mask = cv2.Canny(np.array(fg_mask), 100, 200)
                edge_mask = cv2.dilate(edge_mask, np.ones((edge_thickness, edge_thickness), np.uint8), iterations=1)
                edge_overlay = Image.new("RGBA", (new_width, new_height), (0, 0, 0, 0))
                edge_overlay.paste(Image.new("RGB", fg_pil.size, edge_color), (paste_x, paste_y), Image.fromarray(edge_mask))
                result = Image.alpha_composite(result, edge_overlay)

            if output_format == "RGB":
                result = result.convert("RGB")

            return pil2tensor(result), pil2tensor(fg_mask)

        try:
            fg_batch_size = foreground.shape[0]
            bg_batch_size = background.shape[0] if background is not None else 0
            max_batch_size = max(fg_batch_size, bg_batch_size)

            processed_images = []
            processed_masks = []
            
            for frame_number in tqdm(range(animation_frames), desc="Processing frames"):
                fg_index = frame_number % fg_batch_size
                bg_index = frame_number % bg_batch_size if bg_batch_size > 0 else None

                single_fg = foreground[fg_index:fg_index+1]
                single_bg = background[bg_index:bg_index+1] if bg_index is not None else None
                single_input_mask = input_masks[fg_index:fg_index+1] if input_masks is not None else None
                
                processed_image, processed_mask = process_single_image(single_fg, single_bg, single_input_mask, frame_number)
                
                processed_images.append(processed_image)
                processed_masks.append(processed_mask)

            logging.info("Finished processing all frames")
            
            stacked_images = torch.cat(processed_images, dim=0)
            stacked_masks = torch.cat(processed_masks, dim=0)
            
            if len(stacked_masks.shape) == 4 and stacked_masks.shape[1] == 1:
                stacked_masks = stacked_masks.squeeze(1)
            
            return (stacked_images, stacked_masks)
        
        except Exception as e:
            logging.error(f"Error during image processing: {str(e)}")
            raise

#endregion-------------------------------------------------------#
