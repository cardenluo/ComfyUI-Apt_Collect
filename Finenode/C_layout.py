from PIL import Image
import torch
import numpy as np
import nodes
from .def_util import *
from comfy.utils import  common_upscale


class lay_ImageGrid:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "rows": ("INT", {"default": 2, "min": 1, "max": 16, "step": 1}),
                "cols": ("INT", {"default": 2, "min": 1, "max": 16, "step": 1}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "grid_images"
    CATEGORY = "Apt_Collect/layout"

    def grid_images(self, images, rows, cols):
        # Convert tensor to numpy array
        images = images.cpu().numpy()
        batch_size, height, width, channels = images.shape
        
        # Calculate grid size
        grid_width = width * cols
        grid_height = height * rows
        
        # Create blank canvas
        grid_image = Image.new('RGB', (grid_width, grid_height))
        
        # Paste images into grid
        for i in range(min(rows * cols, batch_size)):
            row = i // cols
            col = i % cols
            
            # Convert numpy array to PIL Image
            img = Image.fromarray((images[i] * 255).astype(np.uint8))
            
            # Calculate position
            x = col * width
            y = row * height
            
            # Paste image
            grid_image.paste(img, (x, y))
        
        # Convert back to tensor
        grid_image = np.array(grid_image).astype(np.float32) / 255.0
        grid_image = torch.from_numpy(grid_image)[None,]
        
        return (grid_image,)



class lay_MaskGrid:
    @classmethod 
    def INPUT_TYPES(cls):
        return {
            "required": {
                "masks": ("MASK",),
                "rows": ("INT", {"default": 2, "min": 1, "max": 16, "step": 1}),
                "cols": ("INT", {"default": 2, "min": 1, "max": 16, "step": 1}),
            },
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "grid_masks"
    CATEGORY = "Apt_Collect/layout"

    def grid_masks(self, masks, rows, cols):
        # Convert tensor to numpy array
        masks = masks.cpu().numpy()
        batch_size, height, width = masks.shape
        
        # Calculate grid size
        grid_width = width * cols
        grid_height = height * rows
        
        # Create blank canvas
        grid_mask = Image.new('L', (grid_width, grid_height))
        
        # Paste masks into grid
        for i in range(min(rows * cols, batch_size)):
            row = i // cols
            col = i % cols
            
            # Convert numpy array to PIL Image
            mask = Image.fromarray((masks[i] * 255).astype(np.uint8))
            
            # Calculate position
            x = col * width
            y = row * height
            
            # Paste mask
            grid_mask.paste(mask, (x, y))
        
        # Convert back to tensor
        grid_mask = np.array(grid_mask).astype(np.float32) / 255.0
        grid_mask = torch.from_numpy(grid_mask)[None,]
        
        return (grid_mask,)



class lay_text:
    @classmethod
    def INPUT_TYPES(s):

        font_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "fonts")       
        file_list = [f for f in os.listdir(font_dir) if os.path.isfile(os.path.join(font_dir, f)) and f.lower().endswith(".ttf")]

        return {"required": {

                    "image_bg": ("IMAGE",),
                    "text": ("STRING", {"multiline": True, "default": "text"}),
                    "font_name": (file_list,),
                    "font_size": ("INT", {"default": 50, "min": 1, "max": 1024}),
                    
                "text_color": ("COLOR",), 
                
                    "align": (ALIGN_OPTIONS,),
                    "justify": (JUSTIFY_OPTIONS,),
                    "margins": ("INT", {"default": 0, "min": -1024, "max": 1024}),
                    "line_spacing": ("INT", {"default": 0, "min": -1024, "max": 1024}),
                    "position_x": ("INT", {"default": 0, "min": -4096, "max": 4096}),
                    "position_y": ("INT", {"default": 0, "min": -4096, "max": 4096}),
                    "rotation_angle": ("FLOAT", {"default": 0.0, "min": -360.0, "max": 360.0, "step": 0.1}),
                    "rotation_options": (ROTATE_OPTIONS,),
                } ,   
                
            "optional": { 
                "text_bg": ("IMAGE",),},
                
    }

    RETURN_TYPES = ("IMAGE", )
    RETURN_NAMES = ("image", )
    FUNCTION = "composite_text"
    CATEGORY = "Apt_Collect/layout"
    
    def composite_text(self, image_bg, text,
                    font_name, font_size, 
                    margins, line_spacing,
                    position_x, position_y,
                    align, justify,text_color,
                    rotation_angle, rotation_options,text_bg=None, ):

        image_3d = image_bg[0, :, :, :]
        back_image = tensor2pil(image_3d)
        text_image = Image.new('RGB', back_image.size, text_color)
        
        if text_bg is not None:
            text_image = tensor2pil(text_bg[0, :, :, :])
        

        text_mask = Image.new('L', back_image.size)
    
        rotated_text_mask = draw_masked_text(text_mask, text, font_name, font_size,
                                            margins, line_spacing, 
                                            position_x, position_y,
                                            align, justify,
                                            rotation_angle, rotation_options)

        image_out = Image.composite(text_image, back_image, rotated_text_mask)       

        return (pil2tensor(image_out), )



class lay_image_match_W_or_H:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "image1": ("IMAGE",),
            "image2": ("IMAGE",),
            "direction": (
            [   'right',
                'down',
                'left',
                'up',
            ],
            {
            "default": 'right'
            }),

        }}
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "concatenate"
    CATEGORY = "Apt_Collect/layout"


    def concatenate(self, image1, image2, direction, first_image_shape=None):

        batch_size1 = image1.shape[0]
        batch_size2 = image2.shape[0]

        if batch_size1 != batch_size2:
            # Calculate the number of repetitions needed
            max_batch_size = max(batch_size1, batch_size2)
            repeats1 = max_batch_size - batch_size1
            repeats2 = max_batch_size - batch_size2
            
            # Repeat the last image to match the largest batch size
            if repeats1 > 0:
                last_image1 = image1[-1].unsqueeze(0).repeat(repeats1, 1, 1, 1)
                image1 = torch.cat([image1.clone(), last_image1], dim=0)
            if repeats2 > 0:
                last_image2 = image2[-1].unsqueeze(0).repeat(repeats2, 1, 1, 1)
                image2 = torch.cat([image2.clone(), last_image2], dim=0)

        target_shape = first_image_shape if first_image_shape is not None else image1.shape
        original_height = image2.shape[1]
        original_width = image2.shape[2]
        original_aspect_ratio = original_width / original_height

        if direction in ['left', 'right']:
            target_height = target_shape[1]  # B, H, W, C format
            target_width = int(target_height * original_aspect_ratio)
        elif direction in ['up', 'down']:

            target_width = target_shape[2]  # B, H, W, C format
            target_height = int(target_width / original_aspect_ratio)
        
        image2_for_upscale = image2.movedim(-1, 1) 
        image2_resized = common_upscale(image2_for_upscale, target_width, target_height, "lanczos", "disabled")
        image2_resized = image2_resized.movedim(1, -1)


        channels_image1 = image1.shape[-1]
        channels_image2 = image2_resized.shape[-1]

        if channels_image1 != channels_image2:
            if channels_image1 < channels_image2:
                # Add alpha channel to image1 if image2 has it
                alpha_channel = torch.ones((*image1.shape[:-1], channels_image2 - channels_image1), device=image1.device)
                image1 = torch.cat((image1, alpha_channel), dim=-1)
            else:
                # Add alpha channel to image2 if image1 has it
                alpha_channel = torch.ones((*image2_resized.shape[:-1], channels_image1 - channels_image2), device=image2_resized.device)
                image2_resized = torch.cat((image2_resized, alpha_channel), dim=-1)

        if direction == 'right':
            concatenated_image = torch.cat((image1, image2_resized), dim=2)  # Concatenate along width
        elif direction == 'down':
            concatenated_image = torch.cat((image1, image2_resized), dim=1)  # Concatenate along height
        elif direction == 'left':
            concatenated_image = torch.cat((image2_resized, image1), dim=2)  # Concatenate along width
        elif direction == 'up':
            concatenated_image = torch.cat((image2_resized, image1), dim=1)  # Concatenate along height
        return concatenated_image,

