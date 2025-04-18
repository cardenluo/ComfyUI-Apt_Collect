
#region---------全局------------------------------------------------------------------------#
import torch
import numpy as np
import os
import folder_paths
from typing import Any, Dict, List, Tuple
import math
from PIL import Image, ImageDraw, ImageFilter, ImageOps, ImageChops, ImageEnhance
import random





def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0) 
def join_with_(text_list,delimiter):
    joined_text = delimiter.join(text_list)
    return joined_text

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

class AnyType(str):
    """A special type that can be connected to any other types. Credit to pythongosssss"""

    def __ne__(self, __value: object) -> bool:
        return False

any_type = AnyType("*")

def get_input_folder(input_folder, input_path):
    # Set the input path
    if input_path != '' and input_path is not None:
        if not os.path.exists(input_path):
            print(f"[Warning] CR Image List: The input_path `{input_path}` does not exist")
            return ("",)  
        in_path = input_path
    else:
        input_dir = folder_paths.input_directory
        in_path = os.path.join(input_dir, input_folder)
    return in_path    

#endregion---------------全局-----------


#region-----------------收纳--------------------

class AD_ImageExpandBatch:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "size": ("INT", { "default": 16, "min": 1, "step": 1, }),
                "method": (["expand", "repeat all", "repeat first", "repeat last"],)
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "Apt_Preset/AD"

    def execute(self, image, size, method):
        orig_size = image.shape[0]

        if orig_size == size:
            return (image,)

        if size <= 1:
            return (image[:size],)

        if 'expand' in method:
            out = torch.empty([size] + list(image.shape)[1:], dtype=image.dtype, device=image.device)
            if size < orig_size:
                scale = (orig_size - 1) / (size - 1)
                for i in range(size):
                    out[i] = image[min(round(i * scale), orig_size - 1)]
            else:
                scale = orig_size / size
                for i in range(size):
                    out[i] = image[min(math.floor((i + 0.5) * scale), orig_size - 1)]
        elif 'all' in method:
            out = image.repeat([math.ceil(size / image.shape[0])] + [1] * (len(image.shape) - 1))[:size]
        elif 'first' in method:
            if size < image.shape[0]:
                out = image[:size]
            else:
                out = torch.cat([image[:1].repeat(size-image.shape[0], 1, 1, 1), image], dim=0)
        elif 'last' in method:
            if size < image.shape[0]:
                out = image[:size]
            else:
                out = torch.cat((image, image[-1:].repeat((size-image.shape[0], 1, 1, 1))), dim=0)

        return (out,)



class AD_MaskExpandBatch:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mask": ("MASK",),
                "size": ("INT", { "default": 16, "min": 1, "step": 1, }),
                "method": (["expand", "repeat all", "repeat first", "repeat last"],)
            }
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "execute"
    CATEGORY = "Apt_Preset/AD"

    def execute(self, mask, size, method):
        orig_size = mask.shape[0]

        if orig_size == size:
            return (mask,)

        if size <= 1:
            return (mask[:size],)

        if 'expand' in method:
            out = torch.empty([size] + list(mask.shape)[1:], dtype=mask.dtype, device=mask.device)
            if size < orig_size:
                scale = (orig_size - 1) / (size - 1)
                for i in range(size):
                    out[i] = mask[min(round(i * scale), orig_size - 1)]
            else:
                scale = orig_size / size
                for i in range(size):
                    out[i] = mask[min(math.floor((i + 0.5) * scale), orig_size - 1)]
        elif 'all' in method:
            out = mask.repeat([math.ceil(size / mask.shape[0])] + [1] * (len(mask.shape) - 1))[:size]
        elif 'first' in method:
            if size < mask.shape[0]:
                out = mask[:size]
            else:
                out = torch.cat([mask[:1].repeat(size-mask.shape[0], 1, 1), mask], dim=0)
        elif 'last' in method:
            if size < mask.shape[0]:
                out = mask[:size]
            else:
                out = torch.cat((mask, mask[-1:].repeat((size-mask.shape[0], 1, 1))), dim=0)

        return (out,)



class AD_Dynamic_MASK:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        base_animation_types = ["LeftToRight", "RightToLeft", "TopDown", "BottomToTop", "GrowingCircle", "ShrinkingCircle",
                           "DiagonalTopLeft-BottomRight", "DiagonalBottomRight-TopLeft", "DiagonalTopRight-BottomLeft",
                           "DiagonalBottomLeft-TopRight", "Fade", "SqSpinCw", "SqSpinCCW", "VenetianBlindsHorizontal", 
                           "VenetianBlindsVertical", "DiagonalVenetianBlinds1", "DiagonalVenetianBlinds2"]
        animation_types = base_animation_types + ["Random", "RandomNoVenetian"]
        easing_options = ["ease_in", "ease_out", "ease_in_out", "false"]
        generation_options = ["Only Transitions", "Generate QR", "Generate Edge-FX", "Generate All"]
        return {
            "required": {
                "animation_type_1": (animation_types, {"default": cls.random_animation()}),
                "animation_type_2": (animation_types, {"default": cls.random_animation()}),
                "animation_type_3": (animation_types, {"default": cls.random_animation()}),
                "animation_type_4": (animation_types, {"default": cls.random_animation()}),

                "transition_easing": (easing_options, {"default": "false"}),
                "blur_easing": (easing_options, {"default": "false"}),
                "frame_width": ("INT", {"default": 512, "min": 1, "step": 1, "display": "number"}),
                "frame_height": ("INT", {"default": 512, "min": 1, "step": 1, "display": "number"}),
                "hold_frames": ("INT", {"default": 8, "min": 1, "step": 1, "display": "number"}),
                "transition_frames": ("INT", {"default": 20, "min": 1, "step": 1, "display": "number"}),
                "padding_frames": ("INT", {"default": 6, "min": 0, "step": 1, "display": "number"}),
                "input_frames": ("INT", {"default": 5, "min": 1, "step": 1, "display": "number"}),
                "gaussian_blur_amount": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10.0, "step": 0.1, "display": "slider"}),
                "edge_fx_thickness": ("INT", {"default": 1, "min": 1, "max": 10, "step": 1, "display": "number"}),
                "push_fx": ("INT", {"default": 0, "min": 0, "max": 30, "step": 1, "display": "number"}),
                "retract_fx": ("INT", {"default": 0, "min": 0, "max": 30, "step": 1, "display": "number"}),
                "fx_cull_white_frames": ("FLOAT", {"default": 10.0, "min": 0.0, "max": 100.0, "step": 0.1, "display": "slider"}),
                "qr_greyness": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "display": "slider"}),
                "random_seed": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10.0, "step": 0.1, "display": "slider"}),
                "edgeFade_contrast": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 6.0, "step": 0.1, "display": "slider"}),  # Increased contrast effectiveness
                "edgeFade_blur": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10.0, "step": 0.1, "display": "slider"}),  # Final blur adjustment for Edge FX Fade
                "generation_mode": (generation_options, {"default": "Only Transitions"}),
                "edge_fx_fade_balance": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 0.9, "step": 0.05, "display": "slider"}),
                "venetian_bars": ("INT", {"default": 4, "min": 1, "step": 1, "display": "number"})
            }
        }

    @classmethod
    def random_animation(cls, exclude_venetian=False):
        venetian_types = ["VenetianBlindsHorizontal", "VenetianBlindsVertical", "DiagonalVenetianBlinds1", "DiagonalVenetianBlinds2"]
        base_animation_types = ["LeftToRight", "RightToLeft", "TopDown", "BottomToTop", "GrowingCircle", "ShrinkingCircle",
                           "DiagonalTopLeft-BottomRight", "DiagonalBottomRight-TopLeft", "DiagonalTopRight-BottomLeft",
                           "DiagonalBottomLeft-TopRight", "Fade", "SqSpinCw", "SqSpinCCW", "VenetianBlindsHorizontal", 
                           "VenetianBlindsVertical", "DiagonalVenetianBlinds1", "DiagonalVenetianBlinds2"]
        available_types = base_animation_types
        if exclude_venetian:
            available_types = [t for t in available_types if t not in venetian_types]
        return random.choice(available_types)

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE", "INT")
    RETURN_NAMES = ("transitions", "qr_mode", "edge_fx", "edge_fx_fade", "total_frames")
    FUNCTION = "run"
    CATEGORY = "Apt_Preset/AD"

    def run(self, **kwargs):
        animation_types = [kwargs[f'animation_type_{i}'] for i in range(1, 5)]
        frame_width = kwargs['frame_width']
        frame_height = kwargs['frame_height']
        hold_frames = kwargs['hold_frames']
        transition_frames = kwargs['transition_frames']
        padding_frames = kwargs['padding_frames']
        input_frames = kwargs['input_frames']
        gaussian_blur_amount = kwargs['gaussian_blur_amount'] * 20  # Amplify the blur amount here
        edgeFade_blur = kwargs['edgeFade_blur'] * 10  # Amplify the final blur amount by 10
        edgeFade_contrast = kwargs['edgeFade_contrast']
        transition_easing = kwargs['transition_easing']
        blur_easing = kwargs['blur_easing']
        qr_greyness = kwargs['qr_greyness']
        random_seed = int(kwargs['random_seed'] * 10)
        edge_fx_thickness = kwargs['edge_fx_thickness']
        push_fx = kwargs['push_fx']
        retract_fx = kwargs['retract_fx']
        fx_cull_white_frames = kwargs['fx_cull_white_frames']
        generation_mode = kwargs['generation_mode']
        edge_fx_fade_balance = kwargs['edge_fx_fade_balance']
        venetian_bars = kwargs['venetian_bars']

        random.seed(random_seed)

        images = []
        qr_images = []
        edge_fx_frames = []
        edge_fx_fade_frames = []
        total_frames = 0
        for i in range(input_frames):
            animation_type = animation_types[i % 4]
            if animation_type == "Random":
                animation_type = self.random_animation()
            elif animation_type == "RandomNoVenetian":
                animation_type = self.random_animation(exclude_venetian=True)

            frames, frame_count = self.generate_animation(animation_type, frame_width, frame_height, hold_frames, transition_frames, gaussian_blur_amount, transition_easing, blur_easing, venetian_bars)

            if generation_mode in ["Generate QR", "Generate All"]:
                qr_frames = self.apply_qr_mode(frames, i, input_frames, hold_frames + transition_frames)
                qr_images.extend(qr_frames)
            
            images.extend(frames)
            total_frames += frame_count

        final_padding_color_qr = 'black' if input_frames % 2 == 0 else 'white'
        for _ in range(padding_frames):
            frame = self.create_frame(frame_width, frame_height, is_black=False)
            images.append(frame)
            if generation_mode in ["Generate QR", "Generate All"]:
                qr_frame = self.create_frame(frame_width, frame_height, is_black=(final_padding_color_qr == 'black'))
                qr_images.append(qr_frame)
            total_frames += 1

        if generation_mode in ["Generate QR", "Generate All"]:
            qr_images = self.apply_qr_greyness(qr_images, qr_greyness, frame_width, frame_height)

        if generation_mode in ["Generate Edge-FX", "Generate All"]:
            edge_fx_frames = self.generate_edge_fx(images, edge_fx_thickness, frame_width, frame_height, fx_cull_white_frames)
            edge_fx_frames = self.apply_push_retract_fx(edge_fx_frames, push_fx, retract_fx, frame_width, frame_height)
            edge_fx_frames = self.check_and_correct_white_frames(edge_fx_frames, frame_width, frame_height, fx_cull_white_frames)

        if generation_mode == "Generate All":
            unblurred_edge_fx = edge_fx_frames.copy()
            final_edge_fx = self.composite_edge_fx(unblurred_edge_fx, edge_fx_frames)
            edge_fx_fade_frames = self.apply_fade_to_edge_fx(final_edge_fx, transition_frames, hold_frames, edge_fx_fade_balance)
            edge_fx_fade_frames = self.apply_contrast_to_frames(edge_fx_fade_frames, edgeFade_contrast)
            edge_fx_fade_frames = self.apply_blur_to_frames(edge_fx_fade_frames, edgeFade_blur)

        image_batch = torch.cat([self.process_image_for_output(frame) for frame in images], dim=0)
        qr_image_batch = torch.cat([self.process_image_for_output(frame) for frame in (qr_images if qr_images else images)], dim=0)
        edge_fx_batch = torch.cat([self.process_image_for_output(frame) for frame in (edge_fx_frames if edge_fx_frames else images)], dim=0)
        edge_fx_fade_batch = torch.cat([self.process_image_for_output(frame) for frame in (edge_fx_fade_frames if edge_fx_fade_frames else images)], dim=0)

        return (image_batch, qr_image_batch, edge_fx_batch, edge_fx_fade_batch, total_frames)

    def generate_animation(self, animation_type, frame_width, frame_height, hold_frames, transition_frames, gaussian_blur_amount, transition_easing, blur_easing, venetian_bars):
        if animation_type == "LeftToRight":
            return self.generate_left_to_right_animation(frame_width, frame_height, hold_frames, transition_frames, gaussian_blur_amount, transition_easing, blur_easing)
        elif animation_type == "RightToLeft":
            return self.generate_rotated_animation(self.generate_left_to_right_animation, frame_width, frame_height, hold_frames, transition_frames, gaussian_blur_amount, transition_easing, blur_easing, 180)
        elif animation_type == "TopDown":
            return self.generate_rotated_animation(self.generate_left_to_right_animation, frame_width, frame_height, hold_frames, transition_frames, gaussian_blur_amount, transition_easing, blur_easing, 90, resize=True)
        elif animation_type == "BottomToTop":
            return self.generate_rotated_animation(self.generate_left_to_right_animation, frame_width, frame_height, hold_frames, transition_frames, gaussian_blur_amount, transition_easing, blur_easing, 270, resize=True)
        elif animation_type == "GrowingCircle":
            return self.generate_growing_circle_animation(frame_width, frame_height, hold_frames, transition_frames, gaussian_blur_amount, transition_easing, blur_easing)
        elif animation_type == "ShrinkingCircle":
            return self.generate_shrinking_circle_animation(frame_width, frame_height, hold_frames, transition_frames, gaussian_blur_amount, transition_easing, blur_easing)
        elif animation_type in ["DiagonalTopLeft-BottomRight", "DiagonalBottomRight-TopLeft", "DiagonalTopRight-BottomLeft", "DiagonalBottomLeft-TopRight"]:
            return self.generate_diagonal_animation(animation_type, frame_width, frame_height, hold_frames, transition_frames, gaussian_blur_amount, transition_easing, blur_easing)
        elif animation_type == "Fade":
            return self.generate_fade_animation(frame_width, frame_height, hold_frames, transition_frames, transition_easing, blur_easing)
        elif animation_type == "SqSpinCw":
            return self.generate_sq_spin_animation(frame_width, frame_height, hold_frames, transition_frames, gaussian_blur_amount, transition_easing, blur_easing, clockwise=True)
        elif animation_type == "SqSpinCCW":
            return self.generate_sq_spin_animation(frame_width, frame_height, hold_frames, transition_frames, gaussian_blur_amount, transition_easing, blur_easing, clockwise=False)
        elif animation_type == "VenetianBlindsHorizontal":
            return self.generate_venetian_blinds_animation(frame_width, frame_height, hold_frames, transition_frames, gaussian_blur_amount, transition_easing, blur_easing, vertical=False, venetian_bars=venetian_bars)
        elif animation_type == "VenetianBlindsVertical":
            return self.generate_venetian_blinds_animation(frame_width, frame_height, hold_frames, transition_frames, gaussian_blur_amount, transition_easing, blur_easing, vertical=True, venetian_bars=venetian_bars)
        elif animation_type == "DiagonalVenetianBlinds1":
            return self.generate_diagonal_venetian_blinds_animation(frame_width, frame_height, hold_frames, transition_frames, gaussian_blur_amount, transition_easing, blur_easing, angle=45, venetian_bars=venetian_bars)
        elif animation_type == "DiagonalVenetianBlinds2":
            return self.generate_diagonal_venetian_blinds_animation(frame_width, frame_height, hold_frames, transition_frames, gaussian_blur_amount, transition_easing, blur_easing, angle=135, venetian_bars=venetian_bars)

    def generate_edge_fx(self, frames, offset, frame_width, frame_height, fx_cull_white_frames):
        edge_fx_frames = []
        for i in range(len(frames)):
            current_frame = frames[i]
            if i + offset < len(frames):
                next_frame = frames[i + offset]
            else:
                next_frame = Image.new('RGB', (frame_width, frame_height), color='white')

            mask = ImageChops.difference(current_frame, next_frame)
            mask = mask.convert('L').point(lambda x: 255 if x > 0 else 0, mode='1')

            edge_fx_frame = Image.new('RGB', (frame_width, frame_height), color='black')
            edge_fx_frame.paste(Image.new('RGB', (frame_width, frame_height), color='white'), mask=mask)

            if ImageChops.invert(edge_fx_frame).getbbox() is None:
                edge_fx_frame = Image.new('RGB', (frame_width, frame_height), color='black')

            edge_fx_frames.append(edge_fx_frame)

        edge_fx_frames = self.check_and_correct_white_frames(edge_fx_frames, frame_width, frame_height, fx_cull_white_frames)

        return edge_fx_frames

    def check_and_correct_white_frames(self, frames, frame_width, frame_height, fx_cull_white_frames):
        corrected_frames = []
        for frame in frames:
            white_pixels = np.sum(np.array(frame) == 255)
            total_pixels = frame_width * frame_height * 3
            white_percentage = (white_pixels / total_pixels) * 100

            if white_percentage > fx_cull_white_frames:
                corrected_frame = Image.new('RGB', (frame_width, frame_height), color='black')
            else:
                corrected_frame = frame
            corrected_frames.append(corrected_frame)
        return corrected_frames

    def apply_push_retract_fx(self, frames, push_fx, retract_fx, frame_width, frame_height):
        black_frame = Image.new('RGB', (frame_width, frame_height), color='black')

        if push_fx > 0:
            frames = [black_frame] * push_fx + frames[:-push_fx]
        elif retract_fx > 0:
            frames = frames[retract_fx:] + [black_frame] * retract_fx

        return frames

    def apply_qr_greyness(self, frames, qr_greyness, frame_width, frame_height):
        grey_image_white = Image.new('RGB', (frame_width, frame_height), color=(255, 255, 255))
        grey_image_black = Image.new('RGB', (frame_width, frame_height), color=(235, 235, 235))
        grey_blend_white = lambda img: Image.blend(img, grey_image_white, qr_greyness)
        grey_blend_black = lambda img: Image.blend(img, grey_image_black, qr_greyness)
        frames = [grey_blend_black(grey_blend_white(frame)) for frame in frames]
        return frames

    def generate_diagonal_venetian_blinds_animation(self, frame_width, frame_height, hold_frames, transition_frames, gaussian_blur_amount, transition_easing, blur_easing, angle, venetian_bars):
        frames, _ = self.generate_venetian_blinds_animation(frame_width, frame_height, hold_frames, transition_frames, gaussian_blur_amount, transition_easing, blur_easing, vertical=True, venetian_bars=venetian_bars)

        enlarged_frames = []
        for frame in frames:
            large_canvas = Image.new('RGB', (frame_width * 2, frame_height * 2), 'black')
            enlarged_frame = frame.resize((int(frame_width * 1.5), int(frame_height * 1.5)), Image.LANCZOS)
            large_canvas.paste(enlarged_frame, ((large_canvas.width - enlarged_frame.width) // 2, (large_canvas.height - enlarged_frame.height) // 2))

            rotated_frame = large_canvas.rotate(angle, expand=True)

            cropped_frame = rotated_frame.crop(((rotated_frame.width - frame_width) // 2, (rotated_frame.height - frame_height) // 2,
                                                (rotated_frame.width + frame_width) // 2, (rotated_frame.height + frame_height) // 2))

            enlarged_frames.append(cropped_frame)

        total_frames = hold_frames + transition_frames
        return enlarged_frames, total_frames

    def generate_venetian_blinds_animation(self, frame_width, frame_height, hold_frames, transition_frames, gaussian_blur_amount, transition_easing, blur_easing, vertical=True, venetian_bars=4):
        images = []
        bar_size = frame_height // venetian_bars if vertical else frame_width // venetian_bars

        for _ in range(hold_frames):
            frame = self.create_frame(frame_width, frame_height, is_black=True)
            images.append(frame)

        for i in range(transition_frames):
            ease_factor = self.calculate_ease_factor(i, transition_frames, transition_easing)
            blur_factor = self.calculate_blur_factor(i, transition_frames, blur_easing, gaussian_blur_amount)

            frame = self.create_venetian_blinds_transition_frame(frame_width, frame_height, venetian_bars, bar_size, ease_factor, blur_factor['gaussian'], vertical)
            images.append(frame)

        total_frames = hold_frames + transition_frames
        return images, total_frames

    def create_venetian_blinds_transition_frame(self, frame_width, frame_height, num_bars, bar_size, ease_factor, gaussian_blur_amount, vertical=True):
        frame = Image.new('RGB', (frame_width, frame_height), color='black')
        draw = ImageDraw.Draw(frame)

        for j in range(num_bars):
            if vertical:
                left = j * bar_size
                right = left + int(bar_size * ease_factor)
                draw.rectangle([left, 0, right, frame_height], fill='white')
            else:
                top = j * bar_size
                bottom = top + int(bar_size * ease_factor)
                draw.rectangle([0, top, frame_width, bottom], fill='white')

        if gaussian_blur_amount > 0:
            frame = frame.filter(ImageFilter.GaussianBlur(gaussian_blur_amount))

        return frame

    def generate_sq_spin_animation(self, frame_width, frame_height, hold_frames, transition_frames, gaussian_blur_amount, transition_easing, blur_easing, clockwise=True):
        images = []
        initial_square_size = 2
        max_square_size = int(frame_width * 1.65)
        canvas_size = int(frame_width * 2.5)

        for _ in range(hold_frames):
            frame = self.create_frame(frame_width, frame_height, is_black=True)
            images.append(frame)

        for i in range(transition_frames):
            ease_factor = self.calculate_ease_factor(i, transition_frames, transition_easing)
            blur_factor = self.calculate_blur_factor(i, transition_frames, blur_easing, gaussian_blur_amount)

            square_size = initial_square_size + int((max_square_size - initial_square_size) * ease_factor)
            rotation_angle = (235 * ease_factor) if clockwise else (-235 * ease_factor)

            frame = self.create_sq_spin_transition_frame(canvas_size, frame_width, frame_height, square_size, rotation_angle, blur_factor['gaussian'])
            images.append(frame)

        total_frames = hold_frames + transition_frames
        return images, total_frames

    def create_sq_spin_transition_frame(self, canvas_size, frame_width, frame_height, square_size, rotation_angle, gaussian_blur_amount):
        frame = Image.new('RGB', (canvas_size, canvas_size), color='black')
        draw = ImageDraw.Draw(frame)

        top_left = ((canvas_size - square_size) // 2, (canvas_size - square_size) // 2)
        bottom_right = (top_left[0] + square_size, top_left[1] + square_size)
        draw.rectangle([top_left, bottom_right], fill='white')

        frame = frame.rotate(rotation_angle, expand=True)

        left = (frame.width - frame_width) // 2
        top = (frame.height - frame_height) // 2
        frame = frame.crop((left, top, left + frame_width, top + frame_height))

        if gaussian_blur_amount > 0:
            frame = frame.filter(ImageFilter.GaussianBlur(gaussian_blur_amount))

        return frame

    def generate_rotated_animation(self, animation_func, frame_width, frame_height, hold_frames, transition_frames, gaussian_blur_amount, transition_easing, blur_easing, rotation_angle, resize=False):
        images, total_frames = animation_func(frame_width, frame_height, hold_frames, transition_frames, gaussian_blur_amount, transition_easing, blur_easing)
        rotated_images = [img.rotate(rotation_angle, expand=True) for img in images]
        if resize:
            rotated_images = [img.resize((frame_width, frame_height), Image.LANCZOS) for img in rotated_images]
        return rotated_images, total_frames

    def generate_left_to_right_animation(self, frame_width, frame_height, hold_frames, transition_frames, gaussian_blur_amount, transition_easing, blur_easing):
        images = []

        for _ in range(hold_frames):
            frame = self.create_frame(frame_width, frame_height, is_black=True)
            images.append(frame)

        for i in range(transition_frames):
            ease_factor = self.calculate_ease_factor(i, transition_frames, transition_easing)
            blur_factor = self.calculate_blur_factor(i, transition_frames, blur_easing, gaussian_blur_amount)
            frame = self.create_left_to_right_transition_frame(frame_width, frame_height, blur_factor['gaussian'], transition_frames, frame_index=i, ease_factor=ease_factor)
            images.append(frame)

        total_frames = hold_frames + transition_frames
        return images, total_frames

    def generate_growing_circle_animation(self, frame_width, frame_height, hold_frames, transition_frames, gaussian_blur_amount, transition_easing, blur_easing):
        images = []
        circle_size = int(frame_width * 1.5)

        for _ in range(hold_frames):
            frame = self.create_frame(frame_width, frame_height, is_black=True)
            images.append(frame)

        for i in range(transition_frames):
            ease_factor = self.calculate_ease_factor(i, transition_frames, transition_easing)
            blur_factor = self.calculate_blur_factor(i, transition_frames, blur_easing, gaussian_blur_amount)
            frame = self.create_growing_circle_transition_frame(frame_width, frame_height, circle_size, blur_factor['gaussian'], transition_frames, frame_index=i, ease_factor=ease_factor)
            images.append(frame)

        total_frames = hold_frames + transition_frames
        return images, total_frames

    def generate_shrinking_circle_animation(self, frame_width, frame_height, hold_frames, transition_frames, gaussian_blur_amount, transition_easing, blur_easing):
        images = []
        circle_size = int(frame_width * 1.5)

        for _ in range(hold_frames):
            frame = self.create_frame(frame_width, frame_height, is_black=True)
            images.append(frame)

        for i in range(transition_frames):
            ease_factor = self.calculate_ease_factor(i, transition_frames, transition_easing)
            blur_factor = self.calculate_blur_factor(i, transition_frames, blur_easing, gaussian_blur_amount)
            frame = self.create_shrinking_circle_transition_frame(frame_width, frame_height, circle_size, blur_factor['gaussian'], transition_frames, frame_index=i, ease_factor=ease_factor)
            images.append(frame)

        total_frames = hold_frames + transition_frames
        return images, total_frames

    def generate_diagonal_animation(self, animation_type, frame_width, frame_height, hold_frames, transition_frames, gaussian_blur_amount, transition_easing, blur_easing):
        frames, _ = self.generate_left_to_right_animation(frame_width, frame_height, hold_frames, transition_frames, gaussian_blur_amount, transition_easing, blur_easing)

        enlarged_frames = []
        for frame in frames:
            large_canvas = Image.new('RGB', (frame_width * 2, frame_height * 2), 'black')
            enlarged_frame = frame.resize((int(frame_width * 1.5), int(frame_height * 1.5)), Image.LANCZOS)
            large_canvas.paste(enlarged_frame, ((large_canvas.width - enlarged_frame.width) // 2, (large_canvas.height - enlarged_frame.height) // 2))

            if animation_type == "DiagonalTopLeft-BottomRight":
                rotated_frame = large_canvas.rotate(45, expand=True)
            elif animation_type == "DiagonalBottomRight-TopLeft":
                rotated_frame = large_canvas.rotate(225, expand=True)
            elif animation_type == "DiagonalTopRight-BottomLeft":
                rotated_frame = large_canvas.rotate(135, expand=True)
            elif animation_type == "DiagonalBottomLeft-TopRight":
                rotated_frame = large_canvas.rotate(315, expand=True)

            cropped_frame = rotated_frame.crop(((rotated_frame.width - frame_width) // 2, (rotated_frame.height - frame_height) // 2,
                                                (rotated_frame.width + frame_width) // 2, (rotated_frame.height + frame_height) // 2))

            enlarged_frames.append(cropped_frame)

        total_frames = hold_frames + transition_frames
        return enlarged_frames, total_frames

    def generate_fade_animation(self, frame_width, frame_height, hold_frames, transition_frames, transition_easing, blur_easing):
        images = []

        for _ in range(hold_frames):
            frame = self.create_frame(frame_width, frame_height, is_black=True)
            images.append(frame)

        for i in range(transition_frames):
            ease_factor = self.calculate_ease_factor(i, transition_frames, transition_easing)
            blur_factor = self.calculate_blur_factor(i, transition_frames, blur_easing, 0)
            gray_value = int(255 * (i / transition_frames) * ease_factor)
            frame = Image.new('RGB', (frame_width, frame_height), color=(gray_value, gray_value, gray_value))
            images.append(frame)

        total_frames = hold_frames + transition_frames
        return images, total_frames

    def create_frame(self, frame_width, frame_height, is_black=True):
        color = 'black' if is_black else 'white'
        frame = Image.new('RGB', (frame_width, frame_height), color=color)
        return frame

    def create_left_to_right_transition_frame(self, frame_width, frame_height, blur_factor, transition_frames, frame_index, ease_factor):
        frame = Image.new('RGB', (frame_width, frame_height), color='black')
        draw = ImageDraw.Draw(frame)

        box_width = int(frame_width * ease_factor)
        draw.rectangle([0, 0, box_width, frame_height], fill='white')

        if blur_factor > 0:
            frame = frame.filter(ImageFilter.GaussianBlur(blur_factor))

        return frame

    def create_growing_circle_transition_frame(self, frame_width, frame_height, circle_size, blur_factor, transition_frames, frame_index, ease_factor):
        frame = Image.new('RGB', (circle_size, circle_size), color='black')
        draw = ImageDraw.Draw(frame)

        max_radius = circle_size // 2
        radius = int(max_radius * ease_factor)

        draw.ellipse([(circle_size // 2 - radius, circle_size // 2 - radius),
                      (circle_size // 2 + radius, circle_size // 2 + radius)], fill='white')

        frame = frame.crop((circle_size//2-frame_width//2, circle_size//2-frame_height//2, circle_size//2+frame_width//2, circle_size//2+frame_height//2))

        if blur_factor > 0:
            frame = frame.filter(ImageFilter.GaussianBlur(blur_factor))

        return frame

    def create_shrinking_circle_transition_frame(self, frame_width, frame_height, circle_size, blur_factor, transition_frames, frame_index, ease_factor):
        frame = Image.new('RGB', (circle_size, circle_size), color='white')
        draw = ImageDraw.Draw(frame)

        max_radius = circle_size // 2
        radius = int(max_radius * (1 - ease_factor))

        draw.ellipse([(circle_size // 2 - radius, circle_size // 2 - radius),
                      (circle_size // 2 + radius, circle_size // 2 + radius)], fill='black')

        frame = frame.crop((circle_size//2-frame_width//2, circle_size//2-frame_height//2, circle_size//2+frame_width//2, circle_size//2+frame_height//2))

        if blur_factor > 0:
            frame = frame.filter(ImageFilter.GaussianBlur(blur_factor))

        return frame

    def calculate_ease_factor(self, frame_index, transition_frames, transition_easing):
        t = frame_index / transition_frames
        if transition_easing == "ease_in":
            return self.ease_in_quad(t)
        elif transition_easing == "ease_out":
            return self.ease_out_quad(t)
        elif transition_easing == "ease_in_out":
            return self.ease_in_out_quad(t)
        else:  # "false" or any other value
            return t  # Linear easing

    def calculate_blur_factor(self, frame_index, transition_frames, blur_easing, gaussian_blur_amount):
        t = frame_index / transition_frames
        if blur_easing == "ease_in":
            blur_factor = self.ease_in_quad(t)
        elif blur_easing == "ease_out":
            blur_factor = 1 - self.ease_out_quad(1 - t)
        elif blur_easing == "ease_in_out":
            blur_factor = self.ease_in_out_quad_for_blur(t)
        else:  # "false" or any other value
            blur_factor = 1  # Constant blur amount for linear

        return {
            'gaussian': gaussian_blur_amount * blur_factor
        }

    def ease_in_quad(self, t):
        return t * t

    def ease_out_quad(self, t):
        return t * (2 - t)

    def ease_in_out_quad(self, t):
        return 2 * t * t if t < 0.5 else -1 + (4 - 2 * t) * t

    def ease_in_out_quad_for_blur(self, t):
        if t < 0.5:
            return 2 * t * t
        else:
            return 1 - pow(-2 * t + 2, 2) / 2

    def apply_qr_mode(self, frames, cycle_index, input_frames, cycle_length):
        if cycle_index % 2 == 1:  # Apply inversion for even cycles
            frames = [ImageOps.invert(frame) for frame in frames]
        return frames

    def process_image_for_output(self, image) -> torch.Tensor:
        i = ImageOps.exif_transpose(image)
        if i.mode == 'I':
            i = i.point(lambda i: i * (1 / 255))
        image = i.convert("RGB")
        image_np = np.array(image).astype(np.float32) / 255.0
        return torch.from_numpy(image_np)[None,]

    def apply_blur_to_frames(self, frames, gaussian_blur_amount):
        return [frame.filter(ImageFilter.GaussianBlur(gaussian_blur_amount)) for frame in frames]

    def apply_contrast_to_frames(self, frames, contrast_factor):
        return [ImageEnhance.Contrast(frame).enhance(contrast_factor * 2) for frame in frames]  # Apply 3x contrast factor

    def composite_edge_fx(self, unblurred_frames, blurred_frames):
        final_frames = []
        for unblurred, blurred in zip(unblurred_frames, blurred_frames):
            # Create a black background
            final_frame = Image.new('RGB', unblurred.size, color='black')

            # Check if the unblurred frame has an alpha channel
            if unblurred.mode == 'RGBA':
                mask = unblurred.split()[3]
            else:
                # If no alpha channel, use the image itself as a mask
                mask = unblurred.convert('L')

            # Use unblurred frame as mask to cut out blurred frame
            final_frame.paste(blurred, mask=mask)

            final_frames.append(final_frame)

        return final_frames

    def apply_fade_to_edge_fx(self, edge_fx_frames, transition_frames, hold_frames, fade_balance):
        faded_frames = []
        frames_per_cycle = transition_frames + hold_frames

        # Calculate fade-up and fade-down frames, ensuring they sum to transition_frames
        fade_up_frames = max(1, min(transition_frames - 1, round(transition_frames * fade_balance)))
        fade_down_frames = transition_frames - fade_up_frames

        for i, frame in enumerate(edge_fx_frames):
            cycle_position = i % frames_per_cycle

            if cycle_position < fade_up_frames:
                # Fade in
                fade_factor = cycle_position / fade_up_frames
            elif cycle_position >= frames_per_cycle - fade_down_frames:
                # Fade out
                fade_factor = (frames_per_cycle - cycle_position) / fade_down_frames
            else:
                # Hold at full opacity
                fade_factor = 1.0

            # Apply fade effect
            faded_frame = Image.new('RGB', frame.size, (0, 0, 0))
            faded_frame = Image.blend(faded_frame, frame, fade_factor)
            faded_frames.append(faded_frame)

        return faded_frames



class AD_batch_prompt:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "keyframe_interval": ("INT", {"default": 30, "min": 0, "max": 999, "step": 1}),
                "loops": ("INT", {"default": 1, "min": 1, "max": 1000}),
                "prompt_1": ("STRING", {"multiline": True, "default": "dancing,(neon:1.3)"}),
            },
            "optional": {
                "prompt_2": ("STRING", {"multiline": True, "default": "(dancing:`pw_a`)"}),
                "prompt_3": ("STRING", {"multiline": True, "default": "(smiling:`(0.5+0.5*sin(t/max_f))`"}),
                "prompt_4": ("STRING", {"multiline": True, "default": "(b:`(1.5*cos(1.57*(t-30)/30)*cos(1.57*(t-30)/30))`)"}),
                "prompt_5": ("STRING", {"multiline": True}),
                "prompt_6": ("STRING", {"multiline": True}),
                "prompt_7": ("STRING", {"multiline": True}),
                "prompt_8": ("STRING", {"multiline": True}),
                "prompt_9": ("STRING", {"multiline": True}),
                "prompt_10": ("STRING", {"multiline": True}),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("sch_prompt",)
    FUNCTION = "make_keyframes"
    CATEGORY = "Apt_Preset/AD"

    def make_keyframes(self, keyframe_interval, loops, prompt_1, prompt_2="", prompt_3="", prompt_4="", prompt_5="", prompt_6="", prompt_7="", prompt_8="", prompt_9="", prompt_10=""):
        prompts = []

        if prompt_1 != "":
            prompts.append(prompt_1)

        if prompt_2 != "":
            prompts.append(prompt_2)

        if prompt_3 != "":
            prompts.append(prompt_3)

        if prompt_4 != "":
            prompts.append(prompt_4)

        if prompt_5 != "":
            prompts.append(prompt_5)

        if prompt_6 != "":
            prompts.append(prompt_6)

        if prompt_7 != "":
            prompts.append(prompt_7)

        if prompt_8 != "":
            prompts.append(prompt_8)

        if prompt_9 != "":
            prompts.append(prompt_9)

        if prompt_10 != "":
            prompts.append(prompt_10)

        keyframe_list = []
        
        i = 0
        for j in range(1, loops + 1): 
            for index, prompt in enumerate(prompts):
                if i == 0:
                    keyframe_list.append(f"\"0\": \"{prompt}\",\n")
                    i += keyframe_interval  
                    continue
                
                new_keyframe = f"\"{i}\": \"{prompt}\",\n"
                keyframe_list.append(new_keyframe)
                i += keyframe_interval 
        
        sch_prompt = " ".join(keyframe_list)[:-2]
        
        return (sch_prompt,)


#endregion-----------------收纳--------------------


