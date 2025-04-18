
#region---------
import torch
import torch.nn.functional as F
from typing import Tuple, Dict, Any
import numpy as np
from torchvision import transforms
from scipy.ndimage import binary_dilation, binary_erosion
import folder_paths as comfy_paths
from PIL import Image, ImageEnhance, ImageFilter, ImageDraw, ImageChops
from comfy.utils import common_upscale
from nodes import MAX_RESOLUTION, SaveImage, common_ksampler
import os,sys
import comfy.utils
import copy
import cv2
from pymatting import estimate_alpha_cf, estimate_foreground_ml, fix_trimap
from PIL import Image, ImageOps, ImageFilter
import math
import scipy.ndimage
from torchvision.transforms import functional as TF
from comfy_extras.nodes_mask import GrowMask





MODELS_DIR =  comfy_paths.models_dir

def upscale(image, upscale_method, width,height):
    samples = image.movedim(-1,1)
    s = comfy.utils.common_upscale(samples, width, height, upscale_method, "disabled")
    s = s.movedim(1,-1)
    return (s,)

def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)


def resize_mask(mask_pil, target_size):
    return mask_pil.resize(target_size, Image.LANCZOS)
def image2mask(image_pil):
    # Convert image to grayscale
    image_pil = image_pil.convert("L")
    # Convert grayscale image to binary mask
    threshold = 128
    mask_array = np.array(image_pil) > threshold
    return Image.fromarray((mask_array * 255).astype(np.uint8))

def pil2mask(image):
    image_np = np.array(image.convert("L")).astype(np.float32) / 255.0
    mask = torch.from_numpy(image_np)
    return 1.0 - mask




#endregion-------------------------------------------------------------------------------#


class cstr(str):
    class color:
        END = '\33[0m'
        BOLD = '\33[1m'
        ITALIC = '\33[3m'
        UNDERLINE = '\33[4m'
        BLINK = '\33[5m'
        BLINK2 = '\33[6m'
        SELECTED = '\33[7m'

        BLACK = '\33[30m'
        RED = '\33[31m'
        GREEN = '\33[32m'
        YELLOW = '\33[33m'
        BLUE = '\33[34m'
        VIOLET = '\33[35m'
        BEIGE = '\33[36m'
        WHITE = '\33[37m'

        BLACKBG = '\33[40m'
        REDBG = '\33[41m'
        GREENBG = '\33[42m'
        YELLOWBG = '\33[43m'
        BLUEBG = '\33[44m'
        VIOLETBG = '\33[45m'
        BEIGEBG = '\33[46m'
        WHITEBG = '\33[47m'

        GREY = '\33[90m'
        LIGHTRED = '\33[91m'
        LIGHTGREEN = '\33[92m'
        LIGHTYELLOW = '\33[93m'
        LIGHTBLUE = '\33[94m'
        LIGHTVIOLET = '\33[95m'
        LIGHTBEIGE = '\33[96m'
        LIGHTWHITE = '\33[97m'

        GREYBG = '\33[100m'
        LIGHTREDBG = '\33[101m'
        LIGHTGREENBG = '\33[102m'
        LIGHTYELLOWBG = '\33[103m'
        LIGHTBLUEBG = '\33[104m'
        LIGHTVIOLETBG = '\33[105m'
        LIGHTBEIGEBG = '\33[106m'
        LIGHTWHITEBG = '\33[107m'

        @staticmethod
        def add_code(name, code):
            if not hasattr(cstr.color, name.upper()):
                setattr(cstr.color, name.upper(), code)
            else:
                raise ValueError(f"'cstr' object already contains a code with the name '{name}'.")

    def __new__(cls, text):
        return super().__new__(cls, text)

    def __getattr__(self, attr):
        if attr.lower().startswith("_cstr"):
            code = getattr(self.color, attr.upper().lstrip("_cstr"))
            modified_text = self.replace(f"__{attr[1:]}__", f"{code}")
            return cstr(modified_text)
        elif attr.upper() in dir(self.color):
            code = getattr(self.color, attr.upper())
            modified_text = f"{code}{self}{self.color.END}"
            return cstr(modified_text)
        elif attr.lower() in dir(cstr):
            return getattr(cstr, attr.lower())
        else:
            raise AttributeError(f"'cstr' object has no attribute '{attr}'")

    def print(self, **kwargs):
        print(self, **kwargs)


#region---------LightSource-----------------------------------------------------------------------------------------------#

import os
import logging
from typing import TypedDict
import torch
import numpy as np
import folder_paths
from nodes import MAX_RESOLUTION
from enum import Enum

# logger
logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

# contents
MODEL_TYPE_ICLIGHT = "iclight"

# set the models directory
if MODEL_TYPE_ICLIGHT not in folder_paths.folder_names_and_paths:
    current_paths = [os.path.join(folder_paths.models_dir, MODEL_TYPE_ICLIGHT)]
else:
    current_paths, _ = folder_paths.folder_names_and_paths[MODEL_TYPE_ICLIGHT]
folder_paths.folder_names_and_paths[MODEL_TYPE_ICLIGHT] = (current_paths, folder_paths.supported_pt_extensions)


class UnetParams(TypedDict):
    input: torch.Tensor
    timestep: torch.Tensor
    c: dict
    cond_or_uncond: torch.Tensor



class LightPosition(Enum):
    LEFT = "Left Light"
    RIGHT = "Right Light"
    TOP = "Top Light"
    BOTTOM = "Bottom Light"
    TOP_LEFT = "Top Left Light"
    TOP_RIGHT = "Top Right Light"
    BOTTOM_LEFT = "Bottom Left Light"
    BOTTOM_RIGHT = "Bottom Right Light"


def generate_gradient_image(width: int, height: int, start_color: tuple, end_color: tuple, multiplier: float,
                            lightPosition: LightPosition):

    # Create a gradient from 0 to 1 and apply multiplier
    if lightPosition == LightPosition.LEFT:
        gradient = np.tile(np.linspace(0, 1, width) ** multiplier, (height, 1))
    elif lightPosition == LightPosition.RIGHT:
        gradient = np.tile(np.linspace(1, 0, width) ** multiplier, (height, 1))
    elif lightPosition == LightPosition.TOP:
        gradient = np.tile(np.linspace(0, 1, height) ** multiplier, (width, 1)).T
    elif lightPosition == LightPosition.BOTTOM:
        gradient = np.tile(np.linspace(1, 0, height) ** multiplier, (width, 1)).T
    elif lightPosition == LightPosition.BOTTOM_RIGHT:
        x = np.linspace(1, 0, width) ** multiplier
        y = np.linspace(1, 0, height) ** multiplier
        x_mesh, y_mesh = np.meshgrid(x, y)
        gradient = np.sqrt(x_mesh ** 2 + y_mesh ** 2) / np.sqrt(2.0)
    elif lightPosition == LightPosition.BOTTOM_LEFT:
        x = np.linspace(0, 1, width) ** multiplier
        y = np.linspace(1, 0, height) ** multiplier
        x_mesh, y_mesh = np.meshgrid(x, y)
        gradient = np.sqrt(x_mesh ** 2 + y_mesh ** 2) / np.sqrt(2.0)
    elif lightPosition == LightPosition.TOP_RIGHT:
        x = np.linspace(1, 0, width) ** multiplier
        y = np.linspace(0, 1, height) ** multiplier
        x_mesh, y_mesh = np.meshgrid(x, y)
        gradient = np.sqrt(x_mesh ** 2 + y_mesh ** 2) / np.sqrt(2.0)
    elif lightPosition == LightPosition.TOP_LEFT:
        x = np.linspace(0, 1, width) ** multiplier
        y = np.linspace(0, 1, height) ** multiplier
        x_mesh, y_mesh = np.meshgrid(x, y)
        gradient = np.sqrt(x_mesh ** 2 + y_mesh ** 2) / np.sqrt(2.0)
    else:
        raise ValueError(f"Unsupported position. Choose from {', '.join([member.value for member in LightPosition])}.")

    # Interpolate between start_color and end_color based on the gradient
    gradient_img = np.zeros((height, width, 3), dtype=np.float32)
    for i in range(3):
        gradient_img[..., i] = start_color[i] + (end_color[i] - start_color[i]) * gradient

    gradient_img = np.clip(gradient_img, 0, 255).astype(np.uint8)
    return gradient_img


class Mask_lightSource:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "light_position": ([member.value for member in LightPosition],),
                "multiplier": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.001}),
                "start_color": ("STRING", {"default": "#FFFFFF"}),
                "end_color": ("STRING", {"default": "#000000"}),
                "width": ("INT", {"default": 512, "min": 0, "max": MAX_RESOLUTION, "step": 8, }),
                "height": ("INT", {"default": 512, "min": 0, "max": MAX_RESOLUTION, "step": 8, })
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "Apt_Collect/mask"


    def execute(self, light_position, multiplier, start_color, end_color, width, height):
        def toRgb(color):
            if color.startswith('#') and len(color) == 7:  # e.g. "#RRGGBB"
                color_rgb = tuple(int(color[i:i + 2], 16) for i in (1, 3, 5))
            else:  # e.g. "255,255,255"
                color_rgb = tuple(int(i) for i in color.split(','))
            return color_rgb

        lightPosition = LightPosition(light_position)
        start_color_rgb = toRgb(start_color)
        end_color_rgb = toRgb(end_color)
        image = generate_gradient_image(width, height, start_color_rgb, end_color_rgb, multiplier, lightPosition)

        image = image.astype(np.float32) / 255.0
        image = torch.from_numpy(image)[None,]
        return (image,)

#endregion--------------------------------------------------------------------------------------------------------#


#region---------DetectByLabel检测遮罩--------------------------------------
sys.path.append(os.path.join(__file__,'../../'))
                
from ultralytics import YOLO,settings,YOLOWorld
# from ultralytics import YOLOWorld

# Update a setting
settings.update({'weights_dir':os.path.join(folder_paths.models_dir,'ultralytics')})


def get_files_with_extension(directory, extension):
    file_list = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(extension):
                file_name = os.path.splitext(file)[0]
                file_list.append(file_name)
    return file_list


def createMask(image,x,y,w,h):
    mask = Image.new("L", image.size)
    pixels = mask.load()
    # 遍历指定区域的像素，将其设置为黑色（0 表示黑色）
    for i in range(int(x), int(x + w)):
        for j in range(int(y), int(y + h)):
            pixels[i, j] = 255
    # mask.save("mask.png")
    return mask


def grow(mask, expand, tapered_corners):
    c = 0 if tapered_corners else 1
    kernel = np.array([[c, 1, c],
                            [1, 1, 1],
                            [c, 1, c]])
    mask = mask.reshape((-1, mask.shape[-2], mask.shape[-1]))
    out = []
    for m in mask:
        output = m.numpy()
        for _ in range(abs(expand)):
            if expand < 0:
                output = scipy.ndimage.grey_erosion(output, footprint=kernel)
            else:
                output = scipy.ndimage.grey_dilation(output, footprint=kernel)
        output = torch.from_numpy(output)
        out.append(output)
    return torch.stack(out, dim=0)


def combine(destination, source, x, y):
    output = destination.reshape((-1, destination.shape[-2], destination.shape[-1])).clone()
    source = source.reshape((-1, source.shape[-2], source.shape[-1]))

    left, top = (x, y,)
    right, bottom = (min(left + source.shape[-1], destination.shape[-1]), min(top + source.shape[-2], destination.shape[-2]))
    visible_width, visible_height = (right - left, bottom - top,)

    source_portion = source[:, :visible_height, :visible_width]
    destination_portion = destination[:, top:bottom, left:right]

    #operation == "subtract":
    output[:, top:bottom, left:right] = destination_portion - source_portion
        
    output = torch.clamp(output, 0.0, 1.0)

    return output

#endregion-----------------



class Mask_Detect_label:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "image": ("IMAGE",),
            "confidence":("FLOAT", {"default": 0.1,
                                                                "min": 0.0,
                                                                "max": 1,
                                                                "step":0.01,
                                                                "display": "number"}),
            "model":(get_files_with_extension(os.path.join(folder_paths.models_dir,'ultralytics'),'.pt'),),

                            },
                "optional":{ 
                    #"target_label": ("STRING",{"forceInput": True,"default": "","multiline": False,"dynamicPrompts": False}), 
                    #"debug":(["on","off"],),
                }

                }
    
    RETURN_TYPES = ("MASK","IMAGE",)
    RETURN_NAMES = ("masks","image",)

    FUNCTION = "run"

    CATEGORY = "Apt_Collect/mask"

    INPUT_IS_LIST = False
    OUTPUT_IS_LIST = (True,True,True,True,)

    def run(self,image,confidence,model,target_label="",debug="on"):
        # print(model)
        target_labels=target_label.split('\n')
        target_labels=[t.strip() for t in target_labels if t.strip()!='']

        model = YOLO(model+'.pt')  
        image=tensor2pil(image)
        image=image.convert('RGB')
        images=[image]

        # Run batched inference on a list of images
        results = model(images)  # return a list of Results objects
        
        masks=[]
        names=[]
        grids=[]
        images_debug=[]
        # Process results list
        for i in range(len(results)):
            result=results[i]
            img=images[i]
            boxes = result.boxes  # Boxes object for bbox outputs
            bb=boxes.xyxy.cpu().numpy()
            confs=boxes.conf.cpu().numpy()

            # Plot results image
            if debug=='on':
                im_bgr = result.plot()  # BGR-order numpy array
                im_rgb = Image.fromarray(im_bgr[..., ::-1])  # RGB-order PIL image
                # im = result.plot(pil=True)
                images_debug.append(pil2tensor(im_rgb))
    
            for j in range(len(bb)):
                name=result.names[boxes[j].cls.item()]
                
                # 判断是否是目标label
                is_target=True
                if len(target_labels)>0:
                    is_target=False
                    for t in target_labels:
                        if t==name:
                            is_target=True

                if is_target==True:
                    b=bb[j]
                    conf=confs[j]
                    if debug=='on':
                        print('#confidence',name,conf)
                    if conf >= confidence:
                        x,y,xw,yh=b
                        w=xw-x
                        h=yh-y
                        mask=createMask(img,x,y,w,h)
                        mask=pil2tensor(mask)
                        masks.append(mask)

                        names.append(name)

                        grids.append((x,y,w,h))

        if len(masks)==0:
            mask = Image.new("L", image.size)
            mask=pil2tensor(mask)
            masks.append(mask)
            grids.append((0,0,image.size[0],image.size[1]))
            names.append(['-'])


        return (masks,images_debug,)



class Mask_inpaint_Grey:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "multiplier": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.001}),
                "brighter": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "smoothness": ("INT", {"default": 1, "min":0, "max": 150, "step": 1}),

            }
        }

    CATEGORY = "Apt_Collect/mask"
    # 修改返回类型，添加 "MASK"
    RETURN_TYPES = ("IMAGE", "IMAGE", "MASK")
    # 修改返回名称，添加 "original_mask"
    RETURN_NAMES = ("multi_img", "bright_img", "mask")
    FUNCTION = "apply"

    def apply(self, image: torch.Tensor, mask: torch.Tensor, multiplier, brighter, smoothness ):
        # Clone the original image and mask to avoid modifying them
        image1_input = image.clone()  # Clone for image1 processing
        mask1_input = mask.clone()    # Clone for image1 processing

        # GrayScaler Logic for Image1 (use clones, not the original)
        if image1_input.ndim != 4 or mask1_input.ndim not in [3, 4]:
            raise ValueError("image must be a 4D tensor, and mask must be a 3D or 4D tensor.")

        if mask1_input.ndim == 3:
            mask1_input = mask1_input.unsqueeze(-1)

        grey_value = 0.5 * multiplier
        image1 = image1_input * mask1_input + (1 - mask1_input) * grey_value

        # Clone the original image and mask again to ensure independence for image2
        image2_input = image.clone()  # Clone for image2 processing
        mask2_input = mask.clone()    # Clone for image2 processing

        # CZeSTGrayoutSubject Logic for Image2 (use clones, not the original)
        subject_mask = mask2_input.reshape((-1, 1, mask2_input.shape[-2], mask2_input.shape[-1])).movedim(1, -1).expand(-1, -1, -1, 3)

        target = Image.fromarray((255. * image2_input[0]).cpu().numpy().astype(np.uint8))
        subjectmask = Image.fromarray((255. * subject_mask[0]).cpu().numpy().astype(np.uint8))

        grayall_target = target.convert('L').convert('RGB')
        grayall_target = ImageEnhance.Brightness(grayall_target)
        grayall_target = grayall_target.enhance(brighter)

        graysubject = ImageChops.darker(grayall_target, subjectmask)
        colorbackground = ImageChops.darker(target, ImageChops.invert(subjectmask))

        grayoutsubject = ImageChops.lighter(colorbackground, graysubject)

        image2 = np.array(grayoutsubject).astype(np.float32) / 255.0
        image2 = torch.from_numpy(image2)[None,]


        mask=tensor2pil(mask)
        feathered_image = mask.filter(ImageFilter.GaussianBlur(smoothness))
        mask=pil2tensor(feathered_image)

        return (image1, image2, mask)



class Mask_mulcolor_img:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", {"default": 512, "min": 4, "max": 99999, "step": 1}),
                "height": ("INT", {"default": 512, "min": 4, "max": 99999, "step": 1}),
            },
            "optional": {
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("red","green","blue","cyan","magenta","yellow","black","white",)
    FUNCTION = 'color_images'
    CATEGORY = "Apt_Collect/mask"

    def color_images(self, width, height):
        red_image = Image.new('RGB', (width, height), color=(255, 0, 0))  # RGB 值对应 #FFF000
        green_image = Image.new('RGB', (width, height), color=(0, 255, 0))  # RGB 值对应 #00FF00
        blue_image = Image.new('RGB', (width, height), color=(0, 0, 255))  # RGB 值对应 #0000FF
        cyan_image = Image.new('RGB', (width, height), color=(0, 255, 255))  # RGB 值对应 #00FFFF
        magenta_image = Image.new('RGB', (width, height), color=(255, 0, 255))  # RGB 值对应 #FF00FF
        yellow_image = Image.new('RGB', (width, height), color=(255, 255, 0))  # RGB 值对应 #FFFF00
        black_image = Image.new('RGB', (width, height), color=(0, 0, 0))  # RGB 值对应 #000000
        white_image = Image.new('RGB', (width, height), color=(255, 255, 255))  # RGB 值对应 #FFFFFF

        return (
            pil2tensor(red_image),
            pil2tensor(green_image),
            pil2tensor(blue_image),
            pil2tensor(cyan_image),
            pil2tensor(magenta_image),
            pil2tensor(yellow_image),
            pil2tensor(black_image),
            pil2tensor(white_image)
        )



class Mask_mulcolor_mask:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE", ),
                "threshold_r": ("FLOAT", { "default": 0.15, "min": 0.0, "max": 1, "step": 0.01, }),
                "threshold_g": ("FLOAT", { "default": 0.15, "min": 0.0, "max": 1, "step": 0.01, }),
                "threshold_b": ("FLOAT", { "default": 0.15, "min": 0.0, "max": 1, "step": 0.01, }),
            }
        }

    RETURN_TYPES = ("MASK","MASK","MASK","MASK","MASK","MASK","MASK","MASK",)
    RETURN_NAMES = ("red","green","blue","cyan","magenta","yellow","black","white",)
    FUNCTION = "execute"
    CATEGORY = "Apt_Collect/mask"

    def execute(self, image, threshold_r, threshold_g, threshold_b):
        red = ((image[..., 0] >= 1-threshold_r) & (image[..., 1] < threshold_g) & (image[..., 2] < threshold_b)).float()
        green = ((image[..., 0] < threshold_r) & (image[..., 1] >= 1-threshold_g) & (image[..., 2] < threshold_b)).float()
        blue = ((image[..., 0] < threshold_r) & (image[..., 1] < threshold_g) & (image[..., 2] >= 1-threshold_b)).float()

        cyan = ((image[..., 0] < threshold_r) & (image[..., 1] >= 1-threshold_g) & (image[..., 2] >= 1-threshold_b)).float()
        magenta = ((image[..., 0] >= 1-threshold_r) & (image[..., 1] < threshold_g) & (image[..., 2] > 1-threshold_b)).float()
        yellow = ((image[..., 0] >= 1-threshold_r) & (image[..., 1] >= 1-threshold_g) & (image[..., 2] < threshold_b)).float()

        black = ((image[..., 0] <= threshold_r) & (image[..., 1] <= threshold_g) & (image[..., 2] <= threshold_b)).float()
        white = ((image[..., 0] >= 1-threshold_r) & (image[..., 1] >= 1-threshold_g) & (image[..., 2] >= 1-threshold_b)).float()
        
        return (red, green, blue, cyan, magenta, yellow, black, white,)



class Mask_math:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mask1":("MASK",),
                "mask2":("MASK",),
                "operation":(["-","+","*","&"],{"+":"subtract"}),
                "algorithm":(["cv2","torch"],{"default":"cv2"}),
                "invert_mask1":("BOOLEAN",{"default":False}),
                "invert_mask2":("BOOLEAN",{"default":False}),
            }
        }
    CATEGORY = "Apt_Collect/mask"
    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "mask_math"
    def mask_math(self, mask1, mask2, operation, algorithm, invert_mask1, invert_mask2):
        #invert mask
        if invert_mask1:
            mask1 = 1-mask1
        if invert_mask2:
            mask2 = 1-mask2

        #repeat mask
        if mask1.dim() == 2:
            mask1 = mask1.unsqueeze(0)
        if mask2.dim() == 2:
            mask2 = mask2.unsqueeze(0)
        if mask1.shape[0] == 1 and mask2.shape[0] != 1:
            mask1 = mask1.repeat(mask2.shape[0],1,1)
        elif mask1.shape[0] != 1 and mask2.shape[0] == 1:
            mask2 = mask2.repeat(mask1.shape[0],1,1)

        #check cv2
        if algorithm == "cv2":
            try:
                import cv2
            except:
                print("prompt-mask_and_mask_math: cv2 is not installed, Using Torch")
                print("prompt-mask_and_mask_math: cv2 未安装, 使用torch")
                algorithm = "torch"

        #algorithm
        if algorithm == "cv2":
            if operation == "-":
                return (self.subtract_masks(mask1, mask2),)
            elif operation == "+":
                return (self.add_masks(mask1, mask2),)
            elif operation == "*":
                return (self.multiply_masks(mask1, mask2),)
            elif operation == "&":
                return (self.and_masks(mask1, mask2),)
        elif algorithm == "torch":
            if operation == "-":
                return (torch.clamp(mask1 - mask2, min=0, max=1),)
            elif operation == "+":
                return (torch.clamp(mask1 + mask2, min=0, max=1),)
            elif operation == "*":
                return (torch.clamp(mask1 * mask2, min=0, max=1),)
            elif operation == "&":
                mask1 = torch.round(mask1).bool()
                mask2 = torch.round(mask2).bool()
                return (mask1 & mask2, )

    def subtract_masks(self, mask1, mask2):
        mask1 = mask1.cpu()
        mask2 = mask2.cpu()
        cv2_mask1 = np.array(mask1) * 255
        cv2_mask2 = np.array(mask2) * 255
        import cv2
        if cv2_mask1.shape == cv2_mask2.shape:
            cv2_mask = cv2.subtract(cv2_mask1, cv2_mask2)
            return torch.clamp(torch.from_numpy(cv2_mask) / 255.0, min=0, max=1)
        else:
            # do nothing - incompatible mask shape: mostly empty mask
            print("Warning-mask_math: The two masks have different shapes")
            return mask1

    def add_masks(self, mask1, mask2):
        mask1 = mask1.cpu()
        mask2 = mask2.cpu()
        cv2_mask1 = np.array(mask1) * 255
        cv2_mask2 = np.array(mask2) * 255
        import cv2
        if cv2_mask1.shape == cv2_mask2.shape:
            cv2_mask = cv2.add(cv2_mask1, cv2_mask2)
            return torch.clamp(torch.from_numpy(cv2_mask) / 255.0, min=0, max=1)
        else:
            # do nothing - incompatible mask shape: mostly empty mask
            print("Warning-mask_math: The two masks have different shapes")
            return mask1
    
    def multiply_masks(self, mask1, mask2):
        mask1 = mask1.cpu()
        mask2 = mask2.cpu()
        cv2_mask1 = np.array(mask1) * 255
        cv2_mask2 = np.array(mask2) * 255
        import cv2
        if cv2_mask1.shape == cv2_mask2.shape:
            cv2_mask = cv2.multiply(cv2_mask1, cv2_mask2)
            return torch.clamp(torch.from_numpy(cv2_mask) / 255.0, min=0, max=1)
        else:
            # do nothing - incompatible mask shape: mostly empty mask
            print("Warning-mask_math: The two masks have different shapes")
            return mask1
    
    def and_masks(self, mask1, mask2):
        mask1 = mask1.cpu()
        mask2 = mask2.cpu()
        cv2_mask1 = np.array(mask1)
        cv2_mask2 = np.array(mask2)
        import cv2
        if cv2_mask1.shape == cv2_mask2.shape:
            cv2_mask = cv2.bitwise_and(cv2_mask1, cv2_mask2)
            return torch.from_numpy(cv2_mask)
        else:
            # do nothing - incompatible mask shape: mostly empty mask
            print("Warning-mask_math: The two masks have different shapes")
            return mask1



class Mask_AD_generate:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "width": ("INT", { "default": 512, "min": 1, "max": 5120, "step": 1, }),
                "height": ("INT", { "default": 512, "min": 1, "max": 5120, "step": 1, }),
                "frames": ("INT", { "default": 16, "min": 1, "max": 9999, "step": 1, }),
                "start_frame": ("INT", { "default": 0, "min": 0, "step": 1, }),
                "end_frame": ("INT", { "default": 9999, "min": 0, "step": 1, }),
                "transition_type": (["horizontal slide", "vertical slide", "horizontal bar", "vertical bar", "center box", "horizontal door", "vertical door", "circle", "fade"],),
                "method": (["linear", "in", "out", "in-out"],),
                
                "invertMask": ("BOOLEAN", {"default": False})
            }
        }

    FUNCTION = "run"
    CATEGORY = "Apt_Preset/AD"
    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)

    def linear(self, i, t):
        return i/t
    def ease_in(self, i, t):
        return pow(i/t, 2)
    def ease_out(self, i, t):
        return 1 - pow(1 - i/t, 2)
    def ease_in_out(self, i, t):
        if i < t/2:
            return pow(i/(t/2), 2) / 2
        else:
            return 1 - pow(1 - (i - t/2)/(t/2), 2) / 2

    def run(self, width, height, frames, start_frame, end_frame, transition_type, method, invertMask):
        if method == 'in':
            method = self.ease_in
        elif method == 'out':
            method = self.ease_out
        elif method == 'in-out':
            method = self.ease_in_out
        else:
            method = self.linear

        out = []

        end_frame = min(frames, end_frame)
        transition = end_frame - start_frame

        if start_frame > 0:
            out = out + [torch.full((height, width), 0.0, dtype=torch.float32, device="cpu")] * start_frame

        for i in range(transition):
            frame = torch.full((height, width), 0.0, dtype=torch.float32, device="cpu")
            progress = method(i, transition-1)

            if "horizontal slide" in transition_type:
                pos = round(width*progress)
                frame[:, :pos] = 1.0
            elif "vertical slide" in transition_type:
                pos = round(height*progress)
                frame[:pos, :] = 1.0
            elif "box" in transition_type:
                box_w = round(width*progress)
                box_h = round(height*progress)
                x1 = (width - box_w) // 2
                y1 = (height - box_h) // 2
                x2 = x1 + box_w
                y2 = y1 + box_h
                frame[y1:y2, x1:x2] = 1.0
            elif "circle" in transition_type:
                radius = math.ceil(math.sqrt(pow(width,2)+pow(height,2))*progress/2)
                c_x = width // 2
                c_y = height // 2
                # is this real life? Am I hallucinating?
                x = torch.arange(0, width, dtype=torch.float32, device="cpu")
                y = torch.arange(0, height, dtype=torch.float32, device="cpu")
                y, x = torch.meshgrid((y, x), indexing="ij")
                circle = ((x - c_x) ** 2 + (y - c_y) ** 2) <= (radius ** 2)
                frame[circle] = 1.0
            elif "horizontal bar" in transition_type:
                bar = round(height*progress)
                y1 = (height - bar) // 2
                y2 = y1 + bar
                frame[y1:y2, :] = 1.0
            elif "vertical bar" in transition_type:
                bar = round(width*progress)
                x1 = (width - bar) // 2
                x2 = x1 + bar
                frame[:, x1:x2] = 1.0
            elif "horizontal door" in transition_type:
                bar = math.ceil(height*progress/2)
                if bar > 0:
                    frame[:bar, :] = 1.0
                    frame[-bar:, :] = 1.0
            elif "vertical door" in transition_type:
                bar = math.ceil(width*progress/2)
                if bar > 0:
                    frame[:, :bar] = 1.0
                    frame[:, -bar:] = 1.0
            elif "fade" in transition_type:
                frame[:,:] = progress

            out.append(frame)

        if end_frame < frames:
            out = out + [torch.full((height, width), 1.0, dtype=torch.float32, device="cpu")] * (frames - end_frame)

        out = torch.stack(out, dim=0)
        
        if invertMask:
            out = 1.0 - out


        return (out, )




#region-------------image2mask--------------#


class Mask_image2mask:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask_type": (["White_Balance", "color", "channel", "depth"],),
                "WB_low_threshold": ("INT", {"default": 1, "min": 1, "max": 255, "step": 1}),
                "WB_high_threshold": ("INT", {"default": 255, "min": 1, "max": 255, "step": 1}),

                "color": ("COLOR",),
                "color_threshold": ("INT", {"default": 10, "min": 0, "max": 255, "step": 1}),
                "channel": (["red", "green", "blue", "alpha"],),
                "depth": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.001, "display": "number"}),


                "blur_radius": ("INT", {"default": 1, "min": 1, "max": 32768, "step": 1}),
                "expand": ("INT", {"default": 0, "min": -150, "max": 150, "step": 1}),

            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "MASK", "MASK")
    RETURN_NAMES = ("mask2_img", "invert_mask2_img", "mask", "invert_mask")
    FUNCTION = "image_to_mask"
    CATEGORY = "Apt_Collect/mask"

    def image_to_mask(self, image, WB_low_threshold, WB_high_threshold, blur_radius, mask_type, depth, color, color_threshold, channel, expand):
        
        tapered_corners = True
        
        if isinstance(image, torch.Tensor):
            image_np = image.cpu().numpy()
        else:
            image_np = image

        out_image = image

        if mask_type == "White_Balance":
            image = 255. * image_np[0]
            image = Image.fromarray(np.clip(image, 0, 255).astype(np.uint8))
            image = ImageOps.grayscale(image)
            threshold_filter = lambda x: 255 if x > WB_high_threshold else 0 if x < WB_low_threshold else x
            image = image.convert("L").point(threshold_filter, mode="L")
            image = np.array(image).astype(np.float32) / 255.0
            mask = 1- torch.from_numpy(image)
            #mask2_img = torch.from_numpy(image)[None,]


        elif mask_type == "color":
            images = 255. * image_np
            images = np.clip(images, 0, 255).astype(np.uint8)
            images = [Image.fromarray(img) for img in images]
            images = [np.array(img) for img in images]

            black = [0, 0, 0]
            white = [255, 255, 255]
            new_images = []

            # 将十六进制颜色字符串转换为 RGB 值
            if isinstance(color, str) and color.startswith('#'):
                color = color.lstrip('#')
                rgb = tuple(int(color[i:i+2], 16) for i in (0, 2, 4))
                color = np.array(rgb, dtype=np.float32)

            for img in images:
                new_image = np.full_like(img, black)

                color_distances = np.linalg.norm(img - color, axis=-1)
                complement_indexes = color_distances <= color_threshold
                new_image[complement_indexes] = white
                new_images.append(new_image)

            new_images = np.array(new_images).astype(np.float32) / 255.0
            new_images = torch.from_numpy(new_images).permute(3, 0, 1, 2)
            mask = new_images[0]

        elif mask_type == "channel":
            channels = ["red", "green", "blue", "alpha"]
            mask = image[:, :, :, channels.index(channel)]


        elif mask_type == "depth":
            bs = image.size()[0]
            width = image.size()[2]
            height = image.size()[1]
            mask1 = torch.zeros((bs, height, width))
            image = upscale(image, 'lanczos', width, height)[0]
            for k in range(bs):
                for i in range(width):
                    for j in range(height):
                        now_depth = image[k][j][i][0].item()
                        if now_depth < depth:
                            mask1[k][j][i] = 1
            mask = mask1


        if blur_radius > 0:
            mask=tensor2pil(mask)
            feathered_image = mask.filter(ImageFilter.GaussianBlur(blur_radius))
            mask=pil2tensor(feathered_image)


        mask = GrowMask().expand_mask(mask, expand, tapered_corners)[0]
        mask2_img = mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])).movedim(1, -1).expand(-1, -1, -1, 3)
        invert_mask = 1.0 - mask
        
        invert_mask2_img =1 - mask2_img

        return (mask2_img, invert_mask2_img, mask, invert_mask)



#endregion-------------image2mask--------------#



#region-------------mask2mask--------------#


class Mask_Remap:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mask": ("MASK",),
                "min": ("FLOAT", {"default": 0.0,"min": -10.0, "max": 1.0, "step": 0.01}),
                "max": ("FLOAT", {"default": 1.0,"min": 0.0, "max": 10.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "remap"
    CATEGORY = "Apt_Collect/mask"

    def remap(self, mask, min, max):

        mask_max = torch.max(mask)
        mask_max = mask_max if mask_max > 0 else 1
        scaled_mask = (mask / mask_max) * (max - min) + min
        scaled_mask = torch.clamp(scaled_mask, min=0.0, max=1.0)
        return (scaled_mask, )



class Mask_Outline:

    @classmethod
    def INPUT_TYPES(s):
        return {
                "required": {
                                "mask": ("MASK",),
                                "outline_width":("INT", {"default": 10,"min": 1, "max": 50, "step": 1}),
                                "tapered_corners": ("BOOLEAN", {"default": True}),
                            }
            }
    


    FUNCTION = "run"
    CATEGORY = "Apt_Collect/mask"
    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)

    # 运行的函数
    def run(self, mask, outline_width, tapered_corners):

        m1=grow(mask,outline_width,tapered_corners)
        m2=grow(mask,-outline_width,tapered_corners)

        m3=combine(m1,m2,0,0)

        return (m3,)



class Mask_Smooth:
    @classmethod
    def INPUT_TYPES(s):
        return {
                "required": {
                                "mask": ("MASK",),
                                "smoothness":("INT", {"default": 1, 
                                                        "min":0, 
                                                        "max": 150, 
                                                        "step": 1,
                                                        "display": "slider"})
                            }
            }
    
    RETURN_TYPES = ('MASK',)
    FUNCTION = "run"
    CATEGORY = "Apt_Collect/mask"
    INPUT_IS_LIST = False
    OUTPUT_IS_LIST = (False,)

    # 运行的函数
    def run(self,mask,smoothness):
        mask=tensor2pil(mask)
        feathered_image = mask.filter(ImageFilter.GaussianBlur(smoothness))
        mask=pil2tensor(feathered_image)
        return (mask,)



class Mask_mask2mask:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "expand": ("INT", {"default": 0, "min": -150, "max": 150, "step": 1}),
                "tapered_corners": ("BOOLEAN", {"default": True}),
                "blur_radius": ("FLOAT", {"default": 0.0,"min": 0.0,"max": 100,"step": 0.1}),

                "min": ("FLOAT", {"default": 0.0,"min": -10.0, "max": 1.0, "step": 0.01}),
                "max": ("FLOAT", {"default": 1.0,"min": 0.0, "max": 10.0, "step": 0.01}),
                "outline_width":("INT", {"default": 10,"min": 1, "max": 50, "step": 1}),


            },
            "optional": {
                "fill_holes": ("BOOLEAN", {"default": False}),
            },
        }

    CATEGORY = "Apt_Collect/mask"
    RETURN_TYPES = ("MASK", "MASK", "MASK",)
    RETURN_NAMES = ("mask", "invert_mask", "outline_mask")
    FUNCTION = "expand_mask"

    
    def expand_mask(self, mask, expand, tapered_corners, blur_radius, min, max, outline_width,  fill_holes=False):


        mask_max = torch.max(mask)
        mask_max = mask_max if mask_max > 0 else 1
        scaled_mask = (mask / mask_max) * (max - min) + min
        scaled_mask = torch.clamp(scaled_mask, min=0.0, max=1.0)

        mask = scaled_mask


        c = 0 if tapered_corners else 1
        kernel = np.array([[c, 1, c],
                        [1, 1, 1],
                        [c, 1, c]])
        growmask = mask.reshape((-1, mask.shape[-2], mask.shape[-1])).cpu()
        out = []

        current_expand = expand
        for m in growmask:
            output = m.numpy().astype(np.float32)
            for _ in range(abs(round(current_expand))):
                if current_expand < 0:
                    output = scipy.ndimage.grey_erosion(output, footprint=kernel)
                else:
                    output = scipy.ndimage.grey_dilation(output, footprint=kernel)
            if fill_holes:
                binary_mask = output > 0
                output = scipy.ndimage.binary_fill_holes(binary_mask)
                output = output.astype(np.float32) * 255
            output = torch.from_numpy(output)

            out.append(output)

        if blur_radius != 0:

            for idx, tensor in enumerate(out):
                pil_image = tensor2pil(tensor.cpu().detach())
                pil_image = pil_image.filter(ImageFilter.GaussianBlur(blur_radius))

                out[idx] = pil2tensor(pil_image)
            blurred = torch.cat(out, dim=0)

            mask = blurred
            invert_mask = 1.0 - mask

        else:
            mask = torch.stack(out, dim=0)
            invert_mask = 1.0 - mask
        


        m1=grow(mask,outline_width,tapered_corners)
        m2=grow(mask,-outline_width,tapered_corners)
        outline_mask=combine(m1,m2,0,0)



        return (mask, invert_mask, outline_mask)


#endregion-------------mask2mask--------------#



class Mask_mask2img:

    @classmethod
    def INPUT_TYPES(s):
        return {
                "required": {
                    "mask": ("MASK",),
                "blur_radius": ("INT", {"default": 1, "min": 1, "max": 32768, "step": 1}),
                "expand": ("INT", {"default": 0, "min": -150, "max": 150, "step": 1}),
                }
        }
    CATEGORY = "Apt_Collect/mask"

    RETURN_TYPES = ("IMAGE","IMAGE",)
    RETURN_NAMES = ("image","invert_img")
    FUNCTION = "mask_to_image"

    def mask_to_image(self, mask, blur_radius,expand):

        if blur_radius > 0:
            mask=tensor2pil(mask)
            feathered_image = mask.filter(ImageFilter.GaussianBlur(blur_radius))
            mask=pil2tensor(feathered_image)

        mask = GrowMask().expand_mask(mask, expand, True)[0]

        image = mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])).movedim(1, -1).expand(-1, -1, -1, 3)

        invert_mask = 1.0 - mask
        invert_img = invert_mask.reshape((-1, 1, invert_mask.shape[-2], invert_mask.shape[-1])).movedim(1, -1).expand(-1, -1, -1, 3)

        return (image,invert_img)





class Mask_Offset:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mask": ("MASK",),
                "x": ("INT", { "default": 0, "min": -4096, "max": 10240, "step": 1, "display": "number" }),
                "y": ("INT", { "default": 0, "min": -4096, "max": 10240, "step": 1, "display": "number" }),
                "angle": ("INT", { "default": 0, "min": -360, "max": 360, "step": 1, "display": "number" }),
                "duplication_factor": ("INT", { "default": 1, "min": 1, "max": 1000, "step": 1, "display": "number" }),
                "roll": ("BOOLEAN", { "default": False }),
                "incremental": ("BOOLEAN", { "default": False }),
                "padding_mode": (
            [   
                'empty',
                'border',
                'reflection',
                
            ], {
            "default": 'empty'
            }),
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "offset"
    CATEGORY = "Apt_Collect/mask"


    def offset(self, mask, x, y, angle, roll=False, incremental=False, duplication_factor=1, padding_mode="empty"):
        # Create duplicates of the mask batch
        mask = mask.repeat(duplication_factor, 1, 1).clone()

        batch_size, height, width = mask.shape

        if angle != 0 and incremental:
            for i in range(batch_size):
                rotation_angle = angle * (i+1)
                mask[i] = TF.rotate(mask[i].unsqueeze(0), rotation_angle).squeeze(0)
        elif angle > 0:
            for i in range(batch_size):
                mask[i] = TF.rotate(mask[i].unsqueeze(0), angle).squeeze(0)

        if roll:
            if incremental:
                for i in range(batch_size):
                    shift_x = min(x*(i+1), width-1)
                    shift_y = min(y*(i+1), height-1)
                    if shift_x != 0:
                        mask[i] = torch.roll(mask[i], shifts=shift_x, dims=1)
                    if shift_y != 0:
                        mask[i] = torch.roll(mask[i], shifts=shift_y, dims=0)
            else:
                shift_x = min(x, width-1)
                shift_y = min(y, height-1)
                if shift_x != 0:
                    mask = torch.roll(mask, shifts=shift_x, dims=2)
                if shift_y != 0:
                    mask = torch.roll(mask, shifts=shift_y, dims=1)
        else:
            
            for i in range(batch_size):
                if incremental:
                    temp_x = min(x * (i+1), width-1)
                    temp_y = min(y * (i+1), height-1)
                else:
                    temp_x = min(x, width-1)
                    temp_y = min(y, height-1)
                if temp_x > 0:
                    if padding_mode == 'empty':
                        mask[i] = torch.cat([torch.zeros((height, temp_x)), mask[i, :, :-temp_x]], dim=1)
                    elif padding_mode in ['replicate', 'reflect']:
                        mask[i] = F.pad(mask[i, :, :-temp_x], (0, temp_x), mode=padding_mode)
                elif temp_x < 0:
                    if padding_mode == 'empty':
                        mask[i] = torch.cat([mask[i, :, :temp_x], torch.zeros((height, -temp_x))], dim=1)
                    elif padding_mode in ['replicate', 'reflect']:
                        mask[i] = F.pad(mask[i, :, -temp_x:], (temp_x, 0), mode=padding_mode)

                if temp_y > 0:
                    if padding_mode == 'empty':
                        mask[i] = torch.cat([torch.zeros((temp_y, width)), mask[i, :-temp_y, :]], dim=0)
                    elif padding_mode in ['replicate', 'reflect']:
                        mask[i] = F.pad(mask[i, :-temp_y, :], (0, temp_y), mode=padding_mode)
                elif temp_y < 0:
                    if padding_mode == 'empty':
                        mask[i] = torch.cat([mask[i, :temp_y, :], torch.zeros((-temp_y, width))], dim=0)
                    elif padding_mode in ['replicate', 'reflect']:
                        mask[i] = F.pad(mask[i, -temp_y:, :], (temp_y, 0), mode=padding_mode)
        
        return mask,


class Mask_cut_mask:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),  
            },
            "optional": {
                "up": ("INT", {"default": 0, "min": 0}),
                "down": ("INT", {"default": 0, "min": 0}),
                "left": ("INT", {"default": 0, "min": 0}),
                "right": ("INT", {"default": 0, "min": 0}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", )
    RETURN_NAMES = ("cut_img", "cut_mask", )
    FUNCTION = "mask_crop"
    CATEGORY = "Apt_Collect/mask"

    def mask_crop(self, image, mask, up=0, down=0, left=0, right=0):
        
        image_pil = tensor2pil(image)
        mask_pil = tensor2pil(mask)

        mask_array = np.array(mask_pil) > 0
        
        coords = np.where(mask_array)
        if coords[0].size == 0 or coords[1].size == 0:
            
            return (image, None)

        x0, y0, x1, y1 = coords[1].min(), coords[0].min(), coords[1].max(), coords[0].max()

        
        x0 -= left
        y0 -= up
        x1 += right
        y1 += down
        
        x0 = max(x0, 0)
        y0 = max(y0, 0)
        x1 = min(x1, image_pil.width)
        y1 = min(y1, image_pil.height)
        
        cropped_image_pil = image_pil.crop((x0, y0, x1, y1))
        
        cropped_mask_pil = mask_pil.crop((x0, y0, x1, y1))
        
        cropped_image_tensor = pil2tensor(cropped_image_pil)
        cropped_mask_tensor = pil2tensor(cropped_mask_pil)
        
        return (cropped_image_tensor, cropped_mask_tensor)






import cv2
import numpy as np
import torch

def tensor2cv2img(tensor) -> np.ndarray:
    # Move the tensor to the CPU if needed,and revmoe the batch dimension
    tensor = tensor.cpu().squeeze(0)
    array = tensor.numpy()
    array = (array * 255).astype(np.uint8)
    # Convert the color space from RGB to GRAY
    return cv2.cvtColor(array, cv2.COLOR_RGB2GRAY)

def tensorMask2cv2img(tensor) -> np.ndarray:
    # Move the tensor to the CPU if needed,and revmoe the batch dimension
    tensor = tensor.cpu().squeeze(0)
    array = tensor.numpy()
    array = (array * 255).astype(np.uint8)
    return array





class Mask_splitMask:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "ignore_threshold": ("INT", {"default": 0, "min": 0, "max": 2048, "step": 1}),
                "index": ("INT", {"default": 0, "min": 0, "max": 99, "step": 1}),

            },
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("masks", )
    FUNCTION = "separate"

    CATEGORY = "Apt_Collect/mask"

    def separate(self, mask, ignore_threshold=100, index=0):
        # 将pythorch的tensor转换为opencv需要的图像格式
        opencv_gray_image = tensorMask2cv2img(mask)
        # 阈值处理，将图像二值化
        _, binary_mask = cv2.threshold(
            opencv_gray_image, 1, 255, cv2.THRESH_BINARY)
        # 寻找轮廓
        contours, _ = cv2.findContours(
            binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # 创建一个空白图像，用于绘制分割结果
        segmented_masks = []
        for contour in contours:
            # 计算轮廓的面积
            area = cv2.contourArea(contour)
            # 如果面积小于忽略阈值，则跳过
            if area < ignore_threshold:
                continue

            # 创建一个空白图像，用于绘制分割结果
            segmented_mask = np.zeros_like(binary_mask)
            # 绘制轮廓
            cv2.drawContours(segmented_mask, [
                            contour], 0, (255, 255, 255), thickness=cv2.FILLED)
            # 将当前分割好的遮罩添加到列表中
            segmented_masks.append(segmented_mask)
        output_masks = []

        for segmented_mask in segmented_masks:
            numpy_mask = np.array(segmented_mask).astype(np.float32) / 255.0
            i_mask = torch.from_numpy(numpy_mask)
            # 将图像和遮罩添加到输出列表
            output_masks.append(i_mask.unsqueeze(0))

        mask = output_masks

        if isinstance(mask, list):
            # 如果是数组，则直接返回数组中的第一个元素
            result = mask[index]
        else:
            # 如果不是数组，则直接返回 mask
            result = mask
        if result is None:
            result = torch.zeros((64, 64), dtype=torch.float32, device="cpu")
        return (result,)

