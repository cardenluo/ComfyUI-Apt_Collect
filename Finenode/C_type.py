
import torch
import comfy

class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False
any_type = AnyType("*")

class type_BasiPIPE:
    
    @classmethod
    def INPUT_TYPES(self):
        return {
            "optional": {
                "context": ("RUN_CONTEXT", ),
            },
        }
    RETURN_TYPES = ("BASIC_PIPE",)
    RETURN_NAMES = ("basic_pipe",)
    CATEGORY = "Apt_Collect/utils"
    
    FUNCTION = "fn"

    def fn(self, context):
        pipe = (context['model'], context['clip'], context['vae'], context['positive'], context['negative'])
        return pipe,



class type_text_list2batch :
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "text_list": (any_type,),
                    "delimiter":(["newline","comma","backslash","space"],),
                            },
                }
    
    RETURN_TYPES = ("STRING",) 
    RETURN_NAMES = ("text",) 
    FUNCTION = "run"
    CATEGORY = "Apt_Collect/utils"

    INPUT_IS_LIST = True # 当true的时候，输入时list，当false的时候，如果输入是list，则会自动包一层for循环调用
    OUTPUT_IS_LIST = (False,)

    def run(self,text_list,delimiter):
        delimiter=delimiter[0]
        if delimiter =='newline':
            delimiter='\n'
        elif delimiter=='comma':
            delimiter=','
        elif delimiter=='backslash':
            delimiter='\\'
        elif delimiter=='space':
            delimiter=' '
        t=''
        if isinstance(text_list, list):
            t=delimiter.join(text_list)
        return (t,)



class type_make_maskBatch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "count": ("INT", {"default": 2, "min": 1, "max": 50, "step": 1}),
            }
        }
    
    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "stack_image"
    CATEGORY = "Apt_Collect/utils"

    def stack_image(self, count, **kwargs):
        mask_list = []
        
        for i in range(1, count + 1):
            mask = kwargs.get(f"mask_{i}")
            if mask is not None:
                mask_list.append(mask)
        if len(mask_list) > 0:
            mask_batch = torch.cat(mask_list, dim=0)
            return (mask_batch,)
        return (None,)



import torch
import numpy as np
from PIL import Image, ImageOps



class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False
any_type = AnyType("*")

class type_BasiPIPE:
    
    @classmethod
    def INPUT_TYPES(self):
        return {
            "optional": {
                "context": ("RUN_CONTEXT", ),
            },
        }
    RETURN_TYPES = ("BASIC_PIPE",)
    RETURN_NAMES = ("basic_pipe",)
    CATEGORY = "Apt_Collect/utils"
    
    FUNCTION = "fn"

    def fn(self, context):
        pipe = (context['model'], context['clip'], context['vae'], context['positive'], context['negative'])
        return pipe,


class type_make_imagesBatch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "count": ("INT", {"default": 3, "min": 1, "max": 50, "step": 1}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "stack_image"
    CATEGORY = "Apt_Collect/utils"

    def stack_image(self, count, **kwargs):
        image_list = []
        
        for i in range(1, count + 1):
            image = kwargs.get(f"image_{i}")
            if image is not None:
                image_list.append(image)
        if len(image_list) > 0:
            image_batch = torch.cat(image_list, dim=0)
            return (image_batch,)
        return (None,)


def join_with_(text_list,delimiter):
    joined_text = delimiter.join(text_list)
    return joined_text


class type_text_list2batch :
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "text_list": (any_type,),
                    "delimiter":(["newline","comma","backslash","space"],),
                            },
                }
    
    RETURN_TYPES = ("STRING",) 
    RETURN_NAMES = ("text",) 
    FUNCTION = "run"
    CATEGORY = "Apt_Collect/utils"

    INPUT_IS_LIST = True # 当true的时候，输入时list，当false的时候，如果输入是list，则会自动包一层for循环调用
    OUTPUT_IS_LIST = (False,)

    def run(self,text_list,delimiter):
        delimiter=delimiter[0]
        if delimiter =='newline':
            delimiter='\n'
        elif delimiter=='comma':
            delimiter=','
        elif delimiter=='backslash':
            delimiter='\\'
        elif delimiter=='space':
            delimiter=' '
        t=''
        if isinstance(text_list, list):
            t=join_with_(text_list,delimiter)
        return (t,)






#region ---------------lay_image_match_W_and_Hh----------------


def combine_images(images, layout_direction='horizontal'):

    if layout_direction == 'horizontal':
        combined_width = sum(image.width for image in images)
        combined_height = max(image.height for image in images)
    else:
        combined_width = max(image.width for image in images)
        combined_height = sum(image.height for image in images)

    combined_image = Image.new('RGB', (combined_width, combined_height))

    x_offset = 0
    y_offset = 0  # Initialize y_offset for vertical layout
    for image in images:
        combined_image.paste(image, (x_offset, y_offset))
        if layout_direction == 'horizontal':
            x_offset += image.width
        else:
            y_offset += image.height

    return combined_image

def apply_outline_and_border(images, outline_thickness, outline_color, border_thickness, border_color):
    for i, image in enumerate(images):
        # Apply the outline
        if outline_thickness > 0:
            image = ImageOps.expand(image, outline_thickness, fill=outline_color)

        if border_thickness > 0:
            image = ImageOps.expand(image, border_thickness, fill=border_color)
        images[i] = image
    return images

def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')  # Remove the '#' character, if present
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return (r, g, b)

def get_color_values(color, color_hex, color_mapping):
    

    if color == "custom":
        color_rgb = hex_to_rgb(color_hex)
    else:
        color_rgb = color_mapping.get(color, (0, 0, 0))  # Default to black if the color is not found

    return color_rgb 

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
    "brown": (160, 85, 15),
    "gray": (128, 128, 128),
    "lightgray": (211, 211, 211),
    "darkgray": (102, 102, 102),
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



class lay_image_match_W_and_H:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "count": ("INT", {"default": 2, "min": 1, "max": 50, "step": 1}),
                "border_thickness": ("INT", {"default": 0, "min": 0, "max": 1024}),
                "border_color": ("COLOR", {"default": "#000000"}),
                "outline_thickness": ("INT", {"default": 0, "min": 0, "max": 1024}),
                "outline_color": ("COLOR", {"default": "#000000"}),
                "rows": ("INT", {"default": 1, "min": 1, "max": 16, "step": 1}),
                "cols": ("INT", {"default": 1, "min": 1, "max": 16, "step": 1}),
            },
            "optional": {
                # Optional inputs can be added here if needed
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "process_images"
    CATEGORY = "Apt_Collect/layout"

    def process_images(self, count,
                      border_thickness,
                      outline_thickness, 
                      rows, cols, 
                      outline_color='#000000', 
                      border_color='#000000', **kwargs):
        
        # Get all images
        images = []
        for i in range(1, count + 1):
            image = kwargs.get(f"image_{i}")
            if image is not None:
                images.append(tensor2pil(image))

        # Resize images to match first image size
        if len(images) > 0:
            first_size = images[0].size
            for i in range(1, len(images)):
                if images[i].size != first_size:
                    images[i] = images[i].resize(first_size)

        # Apply borders and outlines
        images = apply_outline_and_border(images, outline_thickness, outline_color, border_thickness, border_color)

        # Combine images into a grid
        combined_image = self.combine_images_grid(images, rows, cols)
        return (pil2tensor(combined_image),)

    def combine_images_grid(self, images, rows, cols):
        if not images:
            return Image.new('RGB', (0, 0))

        # Calculate grid size
        img_width, img_height = images[0].size
        grid_width = img_width * cols
        grid_height = img_height * rows

        # Create blank canvas
        grid_image = Image.new('RGB', (grid_width, grid_height))

        # Paste images into grid
        for i, img in enumerate(images):
            if i >= rows * cols:
                break
            row = i // cols
            col = i % cols
            x = col * img_width
            y = row * img_height
            grid_image.paste(img, (x, y))

        return grid_image


#endregion---------------------------



class type_text_2_UTF8:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING",),
            }
        }


    RETURN_TYPES = ("STRING",) 
    RETURN_NAMES = ("text",) 
    CATEGORY = "Apt_Collect/utils"
    FUNCTION = "encode_utf8"

    def encode_utf8(self, text):
        try:
            encoded_bytes = text.encode('utf-8', 'ignore')
            encoded_text = encoded_bytes.decode('utf-8', 'replace')
            return (encoded_text,)
        except Exception as e:
            return (f"Error during encoding: {e}",)
