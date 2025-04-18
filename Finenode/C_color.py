
from PIL import Image, ImageOps, ImageChops, ImageDraw, ImageFilter, ImageEnhance
import numpy as np
import torch
import cv2
from color_matcher import ColorMatcher
import matplotlib.pyplot as plt
import io

#region------------------------def--------------
def hex_to_rgb(hex_color: str, bgr: bool = False):
    hex_color = hex_color.lstrip("#")
    if bgr:
        return tuple(int(hex_color[i : i + 2], 16) for i in (4, 2, 0))

    return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))

def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

color_mapping = {
    "white": (255, 255, 255),
    "black": (0, 0, 0),
    "red": (255, 0, 0),
    "green": (0, 255, 0),
    "blue": (0, 0, 255),
    "yellow": (255, 255, 0),
    "cyan": (0, 255, 255),
    "magenta": (255, 0, 255),
    "orange": (255, 165, 0),
    "purple": (128, 0, 128),
    "pink": (255, 192, 203),
    "brown": (165, 42, 42),
    "gray": (128, 128, 128),
    "lightgray": (211, 211, 211),
    "darkgray": (169, 169, 169),
    "olive": (128, 128, 0),
    "lime": (0, 128, 0),
    "teal": (0, 128, 128),
    "navy": (0, 0, 128),
    "maroon": (128, 0, 0),
    "fuchsia": (255, 0, 128),
    "aqua": (0, 255, 128),
    "silver": (192, 192, 192),
    "gold": (255, 215, 0),
    "turquoise": (64, 224, 208),
    "lavender": (230, 230, 250),
    "violet": (238, 130, 238),
    "coral": (255, 127, 80),
    "indigo": (75, 0, 130),    
}

COLORS = ["custom", "white", "black", "red", "green", "blue", "yellow",
          "cyan", "magenta", "orange", "purple", "pink", "brown", "gray",
          "lightgray", "darkgray", "olive", "lime", "teal", "navy", "maroon",
          "fuchsia", "aqua", "silver", "gold", "turquoise", "lavender",
          "violet", "coral", "indigo"]

STYLES = ["Accent","afmhot","autumn","binary","Blues","bone","BrBG","brg",
    "BuGn","BuPu","bwr","cividis","CMRmap","cool","coolwarm","copper","cubehelix","Dark2","flag",
    "gist_earth","gist_gray","gist_heat","gist_rainbow","gist_stern","gist_yarg","GnBu","gnuplot","gnuplot2","gray","Greens",
    "Greys","hot","hsv","inferno","jet","magma","nipy_spectral","ocean","Oranges","OrRd",
    "Paired","Pastel1","Pastel2","pink","PiYG","plasma","PRGn","prism","PuBu","PuBuGn",
    "PuOr","PuRd","Purples","rainbow","RdBu","RdGy","RdPu","RdYlBu","RdYlGn","Reds","seismic",
    "Set1","Set2","Set3","Spectral","spring","summer","tab10","tab20","tab20b","tab20c","terrain",
    "turbo","twilight","twilight_shifted","viridis","winter","Wistia","YlGn","YlGnBu","YlOrBr","YlOrRd"]






#endregion----------------------def-------------------------


#region----------------old------------

class color_hex2color:
    """Hex to RGB"""

    def __init__(self) -> None:
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "hex_string": ("STRING",),
            },
        }

    CATEGORY = "Apt_Collect/color"
    RETURN_TYPES = ("COLOR","INT","INT","INT",)
    RETURN_NAMES = ("color","R","G","B",)

    FUNCTION = "execute"

    def execute(self, hex_string):  # 修改参数名和输入类型定义一致
        hex_color = hex_string.lstrip("#")
        r, g, b = hex_to_rgb(hex_color)
        return ('#' + hex_color, r, g, b)  # 返回值格式和 RETURN_TYPES 一致


class color_color2hex:
    """Color to RGB and HEX"""

    def __init__(self) -> None:
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "color": ("COLOR",),
            },
        }

    CATEGORY = "Apt_Collect/color"
    RETURN_TYPES = ("STRING", "INT", "INT", "INT")
    RETURN_NAMES = ("hex_string", "R", "G", "B")
    FUNCTION = "execute"

    def execute(self, color):
        hex_color = color  # 假设输入的 color 本身就是十六进制字符串
        r, g, b = hex_to_rgb(hex_color)
        return (hex_color, r, g, b)


class color_input:
    """Returns to inverse of a color"""

    def __init__(self) -> None:
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "color": ("COLOR",),
            },
        }

    CATEGORY = "Apt_Collect/color"
    RETURN_TYPES = ("COLOR","COLOR",)
    RETURN_NAMES = ("color","Inver_color",)

    FUNCTION = "execute"

    def execute(self, color):

        color2 = color
        color = color.lstrip("#")
        table = str.maketrans('0123456789abcdef', 'fedcba9876543210')
        return (color2, '#' + color.lower().translate(table))


class ImageReplaceColor:
    """Replace Color in an Image"""
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "target_color": ("COLOR",),
                "replace_color": ("COLOR",),
                "clip_threshold": ("INT", {"default": 10, "min": 0, "max": 255, "step": 1}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "image_remove_color"

    CATEGORY = "Apt_Collect/color"

    def image_remove_color(self, image, clip_threshold=10, target_color='#ffffff',replace_color='#ffffff'):
        return (pil2tensor(self.apply_remove_color(tensor2pil(image), clip_threshold, hex_to_rgb(target_color), hex_to_rgb(replace_color))), )

    def apply_remove_color(self, image, threshold=10, color=(255, 255, 255), rep_color=(0, 0, 0)):
        # Create a color image with the same size as the input image
        color_image = Image.new('RGB', image.size, color)

        # Calculate the difference between the input image and the color image
        diff_image = ImageChops.difference(image, color_image)

        # Convert the difference image to grayscale
        gray_image = diff_image.convert('L')

        # Apply a threshold to the grayscale difference image
        mask_image = gray_image.point(lambda x: 255 if x > threshold else 0)

        # Invert the mask image
        mask_image = ImageOps.invert(mask_image)

        # Apply the mask to the original image
        result_image = Image.composite(
            Image.new('RGB', image.size, rep_color), image, mask_image)

        return result_image



class color_Match:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_ref": ("IMAGE",),
                "image_target": ("IMAGE",),
                "method": (
            [   
                'mkl',
                'hm', 
                'reinhard', 
                'mvgd', 
                'hm-mvgd-hm', 
                'hm-mkl-hm',
            ], {
            "default": 'mkl'
            }),
            },
            "optional": {
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
            }
        }
    
    FUNCTION = "colormatch"

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    CATEGORY = "Apt_Collect/color"
    
    
    def colormatch(self, image_ref, image_target, method, strength=1.0):
        cm = ColorMatcher()
        image_ref = image_ref.cpu()
        image_target = image_target.cpu()
        batch_size = image_target.size(0)
        out = []
        images_target = image_target.squeeze()
        images_ref = image_ref.squeeze()

        image_ref_np = images_ref.numpy()
        images_target_np = images_target.numpy()

        if image_ref.size(0) > 1 and image_ref.size(0) != batch_size:
            raise ValueError("ColorMatch: Use either single reference image or a matching batch of reference images.")

        for i in range(batch_size):
            image_target_np = images_target_np if batch_size == 1 else images_target[i].numpy()
            image_ref_np_i = image_ref_np if image_ref.size(0) == 1 else images_ref[i].numpy()
            try:
                image_result = cm.transfer(src=image_target_np, ref=image_ref_np_i, method=method)  #method选择不同的方法
            except BaseException as e:
                print(f"Error occurred during transfer: {e}")
                break
            # Apply the strength multiplier
            image_result = image_target_np + strength * (image_result - image_target_np)
            out.append(torch.from_numpy(image_result))
            
        out = torch.stack(out, dim=0).to(torch.float32)
        out.clamp_(0, 1)
        return (out,)


class color_adjust:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "brightness": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01}),
                "contrast": ("FLOAT", {"default": 1.0, "min": -1.0, "max": 2.0, "step": 0.01}),
                "saturation": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.01}),
                "sharpness": ("FLOAT", {"default": 1.0, "min": -5.0, "max": 5.0, "step": 0.01}),
                "blur": ("INT", {"default": 0, "min": 0, "max": 16, "step": 1}),
                "gaussian_blur": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1024.0, "step": 0.1}),
                "edge_enhance": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "detail_enhance": (["false", "true"],),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "Color_adjust"

    CATEGORY = "Apt_Collect/color"

    def Color_adjust(self, image, brightness, contrast, saturation, sharpness, blur, gaussian_blur, edge_enhance, detail_enhance):


        tensors = []
        if len(image) > 1:
            for img in image:

                pil_image = None

                # Apply NP Adjustments
                if brightness > 0.0 or brightness < 0.0:
                    # Apply brightness
                    img = np.clip(img + brightness, 0.0, 1.0)

                if contrast > 1.0 or contrast < 1.0:
                    # Apply contrast
                    img = np.clip(img * contrast, 0.0, 1.0)

                # Apply PIL Adjustments
                if saturation > 1.0 or saturation < 1.0:
                    # PIL Image
                    pil_image = tensor2pil(img)
                    # Apply saturation
                    pil_image = ImageEnhance.Color(pil_image).enhance(saturation)

                if sharpness > 1.0 or sharpness < 1.0:
                    # Assign or create PIL Image
                    pil_image = pil_image if pil_image else tensor2pil(img)
                    # Apply sharpness
                    pil_image = ImageEnhance.Sharpness(pil_image).enhance(sharpness)

                if blur > 0:
                    # Assign or create PIL Image
                    pil_image = pil_image if pil_image else tensor2pil(img)
                    # Apply blur
                    for _ in range(blur):
                        pil_image = pil_image.filter(ImageFilter.BLUR)

                if gaussian_blur > 0.0:
                    # Assign or create PIL Image
                    pil_image = pil_image if pil_image else tensor2pil(img)
                    # Apply Gaussian blur
                    pil_image = pil_image.filter(
                        ImageFilter.GaussianBlur(radius=gaussian_blur))

                if edge_enhance > 0.0:
                    # Assign or create PIL Image
                    pil_image = pil_image if pil_image else tensor2pil(img)
                    # Edge Enhancement
                    edge_enhanced_img = pil_image.filter(ImageFilter.EDGE_ENHANCE_MORE)
                    # Blend Mask
                    blend_mask = Image.new(
                        mode="L", size=pil_image.size, color=(round(edge_enhance * 255)))
                    # Composite Original and Enhanced Version
                    pil_image = Image.composite(
                        edge_enhanced_img, pil_image, blend_mask)
                    # Clean-up
                    del blend_mask, edge_enhanced_img

                if detail_enhance == "true":
                    pil_image = pil_image if pil_image else tensor2pil(img)
                    pil_image = pil_image.filter(ImageFilter.DETAIL)

                # Output image
                out_image = (pil2tensor(pil_image) if pil_image else img)

                tensors.append(out_image)

            tensors = torch.cat(tensors, dim=0)

        else:

            pil_image = None
            img = image

            # Apply NP Adjustments
            if brightness > 0.0 or brightness < 0.0:
                # Apply brightness
                img = np.clip(img + brightness, 0.0, 1.0)

            if contrast > 1.0 or contrast < 1.0:
                # Apply contrast
                img = np.clip(img * contrast, 0.0, 1.0)

            # Apply PIL Adjustments
            if saturation > 1.0 or saturation < 1.0:
                # PIL Image
                pil_image = tensor2pil(img)
                # Apply saturation
                pil_image = ImageEnhance.Color(pil_image).enhance(saturation)

            if sharpness > 1.0 or sharpness < 1.0:
                # Assign or create PIL Image
                pil_image = pil_image if pil_image else tensor2pil(img)
                # Apply sharpness
                pil_image = ImageEnhance.Sharpness(pil_image).enhance(sharpness)

            if blur > 0:
                # Assign or create PIL Image
                pil_image = pil_image if pil_image else tensor2pil(img)
                # Apply blur
                for _ in range(blur):
                    pil_image = pil_image.filter(ImageFilter.BLUR)

            if gaussian_blur > 0.0:
                # Assign or create PIL Image
                pil_image = pil_image if pil_image else tensor2pil(img)
                # Apply Gaussian blur
                pil_image = pil_image.filter(
                    ImageFilter.GaussianBlur(radius=gaussian_blur))

            if edge_enhance > 0.0:
                # Assign or create PIL Image
                pil_image = pil_image if pil_image else tensor2pil(img)
                # Edge Enhancement
                edge_enhanced_img = pil_image.filter(ImageFilter.EDGE_ENHANCE_MORE)
                # Blend Mask
                blend_mask = Image.new(
                    mode="L", size=pil_image.size, color=(round(edge_enhance * 255)))
                # Composite Original and Enhanced Version
                pil_image = Image.composite(
                    edge_enhanced_img, pil_image, blend_mask)
                # Clean-up
                del blend_mask, edge_enhanced_img

            if detail_enhance == "true":
                pil_image = pil_image if pil_image else tensor2pil(img)
                pil_image = pil_image.filter(ImageFilter.DETAIL)

            # Output image
            out_image = (pil2tensor(pil_image) if pil_image else img)

            tensors = out_image


        return (tensors, )


class color_RadialGradient:
    @classmethod
    def INPUT_TYPES(s):
    
        return {"required": {
                    "width": ("INT", {"default": 512, "min": 64, "max": 4096}),
                    "height": ("INT", {"default": 512, "min": 64, "max": 4096}),
                    
                    "gradient_distance": ("FLOAT", {"default": 1, "min": 0, "max": 2, "step": 0.05}),
                    "radial_center_x": ("FLOAT", {"default": 0.5, "min": 0, "max": 1, "step": 0.05}),
                    "radial_center_y": ("FLOAT", {"default": 0.5, "min": 0, "max": 1, "step": 0.05}),
                    
                    "start_color_hex": ("COLOR", {"default": "#000000"}),
                    "end_color_hex": ("COLOR", {"default": "#ffffff"}),
                    
                    },
                "optional": {

                }
        }

    RETURN_TYPES = ("IMAGE",  )
    RETURN_NAMES = ("IMAGE", )
    FUNCTION = "draw"
    CATEGORY = "Apt_Collect/color"

    def draw(self, width, height, 
            radial_center_x=0.5, radial_center_y=0.5, gradient_distance=1,
            start_color_hex='#000000', end_color_hex='#ffffff'): # Default to .5 if the value is not found

        color1_rgb = hex_to_rgb(start_color_hex)

        color2_rgb = hex_to_rgb(end_color_hex)

        # Create a blank canvas
        canvas = np.zeros((height, width, 3), dtype=np.uint8)

        center_x = int(radial_center_x * width)
        center_y = int(radial_center_y * height)                
        # Computation for max_distance
        max_distance = (np.sqrt(max(center_x, width - center_x)**2 + max(center_y, height - center_y)**2))*gradient_distance

        for i in range(width):
            for j in range(height):
                distance_to_center = np.sqrt((i - center_x) ** 2 + (j - center_y) ** 2)
                t = distance_to_center / max_distance
                # Ensure t is between 0 and 1
                t = max(0, min(t, 1))
                interpolated_color = [int(c1 * (1 - t) + c2 * t) for c1, c2 in zip(color1_rgb, color2_rgb)]
                canvas[j, i] = interpolated_color 

        fig, ax = plt.subplots(figsize=(width / 100, height / 100))

        ax.imshow(canvas)
        plt.axis('off')
        plt.tight_layout(pad=0, w_pad=0, h_pad=0)
        plt.autoscale(tight=True)

        img_buf = io.BytesIO()
        plt.savefig(img_buf, format='png')
        img = Image.open(img_buf)

        image_out = pil2tensor(img.convert("RGB"))


        return (image_out, )


class color_Gradient:
    

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        
        return {"required": {
                    "width": ("INT", {"default": 512, "min": 64, "max": 4096}),
                    "height": ("INT", {"default": 512, "min": 64, "max": 4096}),
                    "gradient_distance": ("FLOAT", {"default": 1, "min": 0, "max": 2, "step": 0.05}),
                    "linear_transition": ("FLOAT", {"default": 0.5, "min": 0, "max": 1, "step": 0.05}),
                    "orientation": ("INT", {"default": 0, "min": 0, "max": 360, "step": 10}),
                    "start_color_hex": ("COLOR", {"default": "#000000"}),
                    "end_color_hex": ("COLOR", {"default": "#ffffff"}),
                    
                    },
                "optional": {
                }
        }

    RETURN_TYPES = ("IMAGE", )
    RETURN_NAMES = ("IMAGE", )
    FUNCTION = "draw"
    CATEGORY = "Apt_Collect/color"

    def draw(self, width, height, orientation, start_color_hex='#000000', end_color_hex='#ffffff', 
            linear_transition=0.5, gradient_distance=1,): 
        

            
        color1_rgb = hex_to_rgb(start_color_hex)

        color2_rgb = hex_to_rgb(end_color_hex)

        # Create a blank canvas
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        # Convert orientation angle to radians
        angle = np.radians(orientation)
        
        # Create gradient based on angle
        x = np.linspace(-1, 1, width)
        y = np.linspace(-1, 1, height)
        X, Y = np.meshgrid(x, y)
        
        # Calculate gradient based on angle
        gradient = X * np.cos(angle) + Y * np.sin(angle)
        
        # Normalize gradient to 0-1 range
        gradient = (gradient - gradient.min()) / (gradient.max() - gradient.min())
        
        # Apply gradient distance and transition
        gradient = (gradient - (linear_transition - gradient_distance/2)) / gradient_distance
        gradient = np.clip(gradient, 0, 1)
        
        # Apply gradient to colors
        for j in range(height):
            for i in range(width):
                t = gradient[j, i]
                interpolated_color = [int(c1 * (1 - t) + c2 * t) for c1, c2 in zip(color1_rgb, color2_rgb)]
                canvas[j, i] = interpolated_color
                    
        fig, ax = plt.subplots(figsize=(width / 100, height / 100))

        ax.imshow(canvas)
        plt.axis('off')
        plt.tight_layout(pad=0, w_pad=0, h_pad=0)
        plt.autoscale(tight=True)

        img_buf = io.BytesIO()
        plt.savefig(img_buf, format='png')
        img = Image.open(img_buf)
        
        image_out = pil2tensor(img.convert("RGB"))         


        return (image_out,  )


class color_pure_img:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "width": ("INT", {"default": 512, "min": 64, "max": 4096}),
                "height": ("INT", {"default": 512, "min": 64, "max": 4096}),
                "color_hex": ("COLOR", {"default": "#000000"}),
            },
            "optional": {}
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE",)
    FUNCTION = "draw"
    CATEGORY = "Apt_Collect/color"

    def draw(self, width, height, color_hex='#000000'):
        # 将十六进制颜色转换为 RGB 格式
        color_rgb = hex_to_rgb(color_hex)
        # 使用 PIL 创建纯色图片
        img = Image.new('RGB', (width, height), color_rgb)
        # 将 PIL 图片转换为张量
        image_out = pil2tensor(img)
        return (image_out,)




#region----------------------color_transfer


import numpy as np
from collections import namedtuple
import cv2
import torch
import folder_paths as comfy_paths
from torchvision.ops import masks_to_boxes
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from PIL import Image, ImageFilter, ImageOps

# Check for CUDA availability
device = 'cuda' if torch.cuda.is_available() else 'cpu'

ARRAY_DATATYPE = torch.int32  # Corresponding to 'l'

Rgb = namedtuple('Rgb', ('r', 'g', 'b'))
Hsl = namedtuple('Hsl', ('h', 's', 'l'))
    
MODELS_DIR =  comfy_paths.models_dir

def tensor2rgb(t: torch.Tensor) -> torch.Tensor:
    size = t.size()
    if (len(size) < 4):
        return t.unsqueeze(3).repeat(1, 1, 1, 3)
    if size[3] == 1:
        return t.repeat(1, 1, 1, 3)
    elif size[3] == 4:
        return t[:, :, :, :3]
    else:
        return t
    
def tensor2rgba(t: torch.Tensor) -> torch.Tensor:
    size = t.size()
    if (len(size) < 4):
        return t.unsqueeze(3).repeat(1, 1, 1, 4)
    elif size[3] == 1:
        return t.repeat(1, 1, 1, 4)
    elif size[3] == 3:
        alpha_tensor = torch.ones((size[0], size[1], size[2], 1))
        return torch.cat((t, alpha_tensor), dim=3)
    else:
        return t

def tensor2mask(t: torch.Tensor) -> torch.Tensor:
    size = t.size()
    if (len(size) < 4):
        return t
    if size[3] == 1:
        return t[:,:,:,0]
    elif size[3] == 4:
        # Not sure what the right thing to do here is. Going to try to be a little smart and use alpha unless all alpha is 1 in case we'll fallback to RGB behavior
        if torch.min(t[:, :, :, 3]).item() != 1.:
            return t[:,:,:,3]

    return TF.rgb_to_grayscale(tensor2rgb(t).permute(0,3,1,2), num_output_channels=1)[:,0,:,:]


# PIL to Tensor
def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

# PIL to Tensor
def pil2tensor_stacked(image):
    if isinstance(image, Image.Image):
        return torch.from_numpy(np.array(image).astype(np.float32) / 255.0)
    elif isinstance(image, torch.Tensor):
        return image
    else:
        raise ValueError(f"Unexpected datatype for input to 'pil2tensor_stacked'. Expected a PIL Image or tensor, but received type: {type(image)}")



def sample(image, mask=None):
    top_two_bits = 0b11000000

    sides = 1 << 2
    cubes = sides ** 7

    samples = torch.zeros((cubes,), dtype=torch.float32, device=device)  # Make sure samples is of float32 type

    # Handle mask
    if mask is not None:
        mask_values = (torch.rand_like(mask, dtype=torch.float32) * 255).int()
        active_pixels = mask_values > mask
    else:
        active_pixels = torch.ones_like(image[:, :, 0], dtype=torch.bool)

    # Calculate RGB, HSL, and Y
    r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    h, s, l = hsl(r, g, b)  # We need to convert the hsl function to use PyTorch
    Y = (r * 0.2126 + g * 0.7152 + b * 0.0722).int()

    # Packing
    packed = ((Y & top_two_bits) << 4) | ((h & top_two_bits) << 2) | (l & top_two_bits)
    packed *= 4

    # Accumulate samples
    packed_active = packed[active_pixels]
    r_active, g_active, b_active = r[active_pixels], g[active_pixels], b[active_pixels]

    samples.index_add_(0, packed_active, r_active)
    samples.index_add_(0, packed_active + 1, g_active)
    samples.index_add_(0, packed_active + 2, b_active)
    samples.index_add_(0, packed_active + 3, torch.ones_like(packed_active, dtype=torch.float32))

    return samples

def pick_used(samples):
    # Find indices where count (every 4th value) is non-zero
    non_zero_indices = torch.arange(0, samples.size(0), 4, device=samples.device)[samples[3::4] > 0]

    # Get counts for non-zero indices
    counts = samples[non_zero_indices + 3]

    # Combine counts and indices
    used = torch.stack((counts, non_zero_indices), dim=-1)

    # Convert torch tensors to list of tuples on CPU
    used_tuples = [(int(count.item()), int(idx.item())) for count, idx in zip(used[:, 0], used[:, 1])]

    return used_tuples

def get_colors(samples, used, number_of_colors):
    number_of_colors = min(number_of_colors, len(used))
    used = used[:number_of_colors]
    
    # Extract counts and indices
    counts, indices = zip(*used)
    counts = torch.tensor(counts, dtype=torch.long, device=device)
    indices = torch.tensor(indices, dtype=torch.long, device=device)

    # Calculate total pixels
    total_pixels = torch.sum(counts)

    # Get RGB values
    r_vals = samples[indices] // counts
    g_vals = samples[indices + 1] // counts
    b_vals = samples[indices + 2] // counts

    # Convert Torch tensors to lists
    r_vals_list = r_vals.tolist()
    g_vals_list = g_vals.tolist()
    b_vals_list = b_vals.tolist()
    counts_list = counts.tolist()

    # Create Color objects
    colors = [Color(r, g, b, count) for r, g, b, count in zip(r_vals_list, g_vals_list, b_vals_list, counts_list)]

    # Update proportions
    for color in colors:
        color.proportion /= total_pixels.item()

    return colors



class Color(object):
    def __init__(self, r, g, b, proportion):
        self.rgb = Rgb(r, g, b)
        self.proportion = proportion
    
    def __repr__(self):
        return "<colorgram.py Color: {}, {}%>".format(
            str(self.rgb), str(self.proportion * 100))

    @property
    def hsl(self):
        try:
            return self._hsl
        except AttributeError:
            self._hsl = Hsl(*hsl(*self.rgb))
            return self._hsl

def extract(image_np, number_of_colors, mask_np=None):
    # Check and convert the image if needed
    if len(image_np.shape) == 2 or image_np.shape[2] != 3:  # If grayscale or not RGB
        image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
    
    samples = sample(image_np, mask_np)
    used = pick_used(samples)
    used.sort(key=lambda x: x[0], reverse=True)
    return get_colors(samples, used, number_of_colors)



def hsl(r, g, b):
    r, g, b = r / 255.0, g / 255.0, b / 255.0
    
    max_val, _ = torch.max(torch.stack([r, g, b]), dim=0)
    min_val, _ = torch.min(torch.stack([r, g, b]), dim=0)
    diff = max_val - min_val
    
    # Luminance
    l = (max_val + min_val) / 2.0

    # Saturation
    s = torch.where(
        (max_val == min_val) | (l == 0),
        torch.zeros_like(l),
        torch.where(l < 0.5, diff / (max_val + min_val), diff / (2.0 - max_val - min_val))
    )
    
    # Hue
    conditions = [
        max_val == r,
        max_val == g,
        max_val == b
    ]

    values = [
        ((g - b) / diff) % 6,
        ((b - r) / diff) + 2,
        ((r - g) / diff) + 4
    ]

    h = torch.zeros_like(r)
    for condition, value in zip(conditions, values):
        h = torch.where(condition, value, h)
    h /= 6.0

    return (h * 255).int(), (s * 255).int(), (l * 255).int()



def segment_image(image_torch, palette_colors, mask_torch=None, threshold=128):
    """
    Segment the image based on the color similarity of each color in the palette using PyTorch.
    """
    if mask_torch is None:
        mask_torch = torch.ones(image_torch.shape[:2], device='cuda') * 255

    output_image_torch = torch.zeros_like(image_torch)

    # Convert palette colors to PyTorch tensor
    palette_torch = torch.tensor([list(color.rgb) for color in palette_colors], device='cuda').float()

    distances = torch.norm(image_torch.unsqueeze(-2) - palette_torch, dim=-1)
    closest_color_indices = torch.argmin(distances, dim=-1)

    for idx, palette_color in enumerate(palette_torch):
        output_image_torch[closest_color_indices == idx] = palette_color

    output_image_torch[mask_torch < threshold] = image_torch[mask_torch < threshold]

    # Convert the PyTorch tensor back to a numpy array for saving or further operations
    output_image_np = output_image_torch.cpu().numpy().astype('uint8')
    return output_image_np

def calculate_luminance_vectorized(colors):
    """Calculate the luminance of an array of RGB colors using PyTorch."""
    R, G, B = colors[:, 0], colors[:, 1], colors[:, 2]
    return 0.299 * R + 0.587 * G + 0.114 * B

def luminance_match(palette1, palette2):
    # Convert palettes to PyTorch tensors
    palette1_rgb = torch.tensor([color.rgb for color in palette1], device='cuda').float()
    palette2_rgb = torch.tensor([color.rgb for color in palette2], device='cuda').float()

    luminance1 = calculate_luminance_vectorized(palette1_rgb)
    luminance2 = calculate_luminance_vectorized(palette2_rgb)

    # Sort luminances and get the sorted indices
    sorted_indices1 = torch.argsort(luminance1)
    sorted_indices2 = torch.argsort(luminance2)

    reordered_palette2 = [None] * len(palette2)

    # Match colors based on sorted luminance order
    for idx1, idx2 in zip(sorted_indices1.cpu().numpy(), sorted_indices2.cpu().numpy()):
        print(f"idx1: {idx1}, idx2: {idx2}")  # Add this to debug
        reordered_palette2[idx1] = palette2[idx2]

    return reordered_palette2

def apply_blur(image_torch, blur_radius, blur_amount):
    image_torch = image_torch.float().div(255.0)
    channels = image_torch.shape[2]

    kernel_size = int(6 * blur_radius + 1)
    kernel_size += 1 if kernel_size % 2 == 0 else 0
    
    # Calculate the padding required to keep the output size the same
    padding = kernel_size // 2
    
    # Create a Gaussian kernel
    x = torch.linspace(-blur_amount, blur_amount, kernel_size).to(image_torch.device)
    x = torch.exp(-x**2 / (2 * blur_radius**2))
    x /= x.sum()
    kernel = x[:, None] * x[None, :]
    
    # Apply the kernel using depthwise convolution
    channels = image_torch.shape[-1]
    kernel = kernel[None, None, ...].repeat(channels, 1, 1, 1)
    blurred = F.conv2d(image_torch.permute(2, 0, 1)[None, ...], kernel, groups=channels, padding=padding)
    
    # Convert the tensor back to byte and de-normalize
    blurred = (blurred * 255.0).byte().squeeze(0).permute(1, 2, 0)
    return blurred

def refined_replace_and_blend_colors(Source_np, img_np, palette1, modified_palette2, blur_radius=0, blur_amount=0, mask_torch=None):
    # Convert numpy arrays to torch tensors on GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    Source = torch.from_numpy(Source_np).float().to(device)
    img_torch = torch.tensor(img_np, device=device).float()

    palette1_rgb = torch.stack([torch.tensor(color.rgb, device=device).float() if hasattr(color, 'rgb') else torch.tensor(color, device=device).float() for color in palette1])
    modified_palette2_rgb = torch.stack([torch.tensor(color.rgb, device=device).float() if hasattr(color, 'rgb') else torch.tensor(color, device=device).float() for color in modified_palette2])

    # Direct color replacement using broadcasting
    distances = torch.norm(img_torch[:, :, None] - palette1_rgb, dim=-1)
    closest_indices = torch.argmin(distances, dim=-1)
    intermediate_output = modified_palette2_rgb[closest_indices]
    
    # Convert to uint8 if not already
    intermediate_output = torch.clamp(intermediate_output, 0, 255).byte()

    # Apply blur if needed
    if blur_radius > 0 and blur_amount > 0:
        blurred_output = apply_blur(intermediate_output, blur_radius, blur_amount)
    else:
        blurred_output = intermediate_output

    # Blend based on the mask's intensity values if provided
    if mask_torch is not None:
        three_channel_mask = mask_torch[:, :, None].expand_as(Source)
        output_torch = Source * (1 - three_channel_mask) + blurred_output.float() * three_channel_mask
    else:
        output_torch = blurred_output
    
    output_np = output_torch.cpu().numpy().astype(np.uint8)

    return output_np


def retain_luminance_hsv_swap(img1_np, img2_np, strength):
    """
    Blend two images while retaining the luminance of the first.
    The blending is controlled by the strength parameter.
    Assumes img1_np and img2_np are numpy arrays in BGR format.
    """

    # Convert BGR to RGB
    img1_rgb_np = cv2.cvtColor(img1_np, cv2.COLOR_BGR2RGB).astype(float) / 255.0
    img2_rgb_np = cv2.cvtColor(img2_np, cv2.COLOR_BGR2RGB).astype(float) / 255.0

    # Blend the two RGB images linearly based on the strength
    blended_rgb_np = (1 - strength) * img1_rgb_np + strength * img2_rgb_np

    # Convert the blended RGB image and the original RGB image to YUV
    blended_yuv_np = cv2.cvtColor((blended_rgb_np * 255).astype(np.uint8), cv2.COLOR_RGB2YUV)
    img1_yuv_np = cv2.cvtColor(img1_np, cv2.COLOR_BGR2YUV)

    # Replace the Y channel (luminance) of the blended image with the original image's luminance
    blended_yuv_np[:,:,0] = img1_yuv_np[:,:,0]

    # Convert back to BGR
    result_bgr_np = cv2.cvtColor(blended_yuv_np, cv2.COLOR_YUV2BGR)

    return result_bgr_np

def adjust_gamma_contrast(image_np, gamma, contrast, brightness, mask_np=None):
    # Ensure CUDA is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Transfer data to PyTorch tensors and move to the appropriate device
    image_torch = torch.tensor(image_np, dtype=torch.float32).to(device)
    
    # Gamma correction using a lookup table
    inv_gamma = 1.0 / gamma
    table = torch.tensor([(i / 255.0) ** inv_gamma * 255 for i in range(256)], device=device).float()
    gamma_corrected = torch.index_select(table, 0, image_torch.long().flatten()).reshape_as(image_torch)
    
    # Contrast and brightness adjustment
    contrast_adjusted = contrast * gamma_corrected + brightness
    contrast_adjusted = torch.clamp(contrast_adjusted, 0, 255).byte()
    
    # If mask is provided, blend the original and adjusted images
    if mask_np is not None:
        mask_torch = torch.tensor(mask_np, device=device).float() / 255.0
        three_channel_mask = mask_torch.unsqueeze(-1).expand_as(image_torch)
        contrast_adjusted = image_torch * (1 - three_channel_mask) + contrast_adjusted.float() * three_channel_mask

    # Transfer data back to numpy array
    result_np = contrast_adjusted.cpu().numpy()

    return result_np




class color_transfer:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "edit_image": ("IMAGE",),
                "ref_color": ("IMAGE",),
                "mask": ("MASK",),
                "no_of_colors": ("INT", {"default": 6, "min": 0, "max": 256, "step": 1}),
                "blur_radius": ("INT", {"default": 2, "min": 0, "max": 100, "step": 1}),
                "blur_amount": ("INT", {"default": 2, "min": 0, "max": 100, "step": 1}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1}),
                "gamma": ("FLOAT", {"default": 1.0, "min": 0.10, "max": 2.0, "step": 0.1}),
                "contrast": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.1}),
                "brightness": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1}),
            },
            "optional": {
            },
        }

    CATEGORY = "Apt_Collect/color"

    RETURN_TYPES = ("IMAGE", )
    RETURN_NAMES = ("image", )
    FUNCTION = "ColorXfer2"

    def ColorXfer2(cls, edit_image, ref_color, no_of_colors, blur_radius, blur_amount, strength, gamma, contrast, brightness, mask=None):   
        

        
        if mask is not None:
            if torch.is_tensor(mask):
                # Convert to grayscale if it's a 3-channel image
                if mask.shape[-1] == 3:
                    mask = torch.mean(mask, dim=-1)

                # Remove batch dimension if present
                if mask.dim() == 3:
                    mask = mask.squeeze(0)

                mask_np1 = (mask.cpu().numpy() * 255).astype(np.uint8)
            else:
                mask_np1 = (mask * 255).astype(np.uint8)

            mask_np = mask_np1 / 255.0
            mask_torch = torch.tensor(mask_np).to(device)

        # If the edit_image is a tensor, convert it to a numpy array
        if torch.is_tensor(edit_image):
            Source_np = (edit_image[0].cpu().numpy() * 255).astype(np.uint8)
        else:
            Source_np = (edit_image * 255).astype(np.uint8)

        # If the edit_image is a tensor, convert it to a numpy array
        if torch.is_tensor(ref_color):
            Target_np = (ref_color[0].cpu().numpy() * 255).astype(np.uint8)
        else:
            Target_np = (ref_color * 255).astype(np.uint8)

        # Load the source image and convert to torch tensor
        Source_np = cv2.cvtColor(Source_np, cv2.COLOR_BGR2RGB)
        Source = torch.from_numpy(Source_np).float().to(device)

        # Extract colors from the source image
        colors1 = extract(Source, no_of_colors, mask_np=mask_torch)

        # Load the target image
        Target_np = cv2.cvtColor(Target_np, cv2.COLOR_BGR2RGB)
        Target = torch.from_numpy(Target_np).float().to(device=device)

        # Extract colors from the target image
        colors2 = extract(Target, no_of_colors)     

        min_length = min(len(colors1), len(colors2))
        colors1 = colors1[:min_length]
        colors2 = colors2[:min_length]

        # Segment the image
        segmented_np = segment_image(Source, colors1, mask_torch=mask_torch, threshold=1)       

        matched_pairs = luminance_match(colors1, colors2)

        result_rgb = refined_replace_and_blend_colors(Source.cpu().numpy(), segmented_np, colors1, matched_pairs, blur_radius, blur_amount, mask_torch=mask_torch)

        luminance_np = retain_luminance_hsv_swap(Source.cpu().numpy(), result_rgb, strength)

        gamma_contrast_np = adjust_gamma_contrast(luminance_np, gamma, contrast, brightness, mask_np=mask_np1)

        final_img_np_rgb = cv2.cvtColor(gamma_contrast_np, cv2.COLOR_BGR2RGB)

        # Convert the numpy array back to a PyTorch tensor
        final_img_tensor = torch.tensor(final_img_np_rgb).float().to(device)

        final_img_tensor = final_img_tensor.unsqueeze(0)

        if final_img_tensor.max() > 1.0:
            final_img_tensor /= 255.0

        return (final_img_tensor, )








#endregion----------------------color_transfer------


#endregion----------------old------------