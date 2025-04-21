import torch
import torch.nn.functional as F
from tqdm import tqdm
from typing import Tuple

class AD_InfiniteZoom:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image1": ("IMAGE",),  # First image (smallest)
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
                "image4": ("IMAGE",),
                "image5": ("IMAGE",),  # Last image (largest)
                "frames_per_intro": ("INT", {"default": 16, "min": 1, "step": 1}),
                "easing_function": (["ease_in_out_cubic", "linear"], {"default": "ease_in_out_cubic"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("zoomed_sequence",)
    FUNCTION = "make_zoom"
    CATEGORY = "Apt_Preset/AD"


    def make_zoom(self, image1: torch.Tensor, image2: torch.Tensor, image3: torch.Tensor, 
                 image4: torch.Tensor, image5: torch.Tensor, 
                 frames_per_intro: int = 16, 
                 easing_function: str = "ease_in_out_cubic") -> tuple[torch.Tensor]:
        """
        Creates a "layered infinite zoom" effect using at most 200% upscaling.
        Each new image i enters at 200% and shrinks to 100%, while older images
        keep shrinking by another factor of 2, all in a 2× big canvas. 
        Then center-crop to produce the final 1024×1024 frames.
        """
        # Convert individual BHWC images to NCHW format
        images = []
        for img in [image1, image2, image3, image4, image5]:
            # Remove batch dimension if present and convert to CHW
            if img.dim() == 4:  # [B,H,W,C]
                img = img.squeeze(0)  # Remove batch dimension -> [H,W,C]
            if img.shape[-1] <= 4:  # If channels last format
                img = img.permute(2, 0, 1)  # Convert to [C,H,W]
            images.append(img)
        
        # Stack into NCHW format
        images = torch.stack(images, dim=0)  # [N,C,H,W]
        channels_last = False

        N, C, H, W = images.shape
        if N < 2:
            if channels_last:
                images = images.permute(0,2,3,1)
            return (images,)

        # We'll have (N-1) segments. Each segment is frames_per_intro frames.
        num_frames = (N - 1) * frames_per_intro

        big_canvas_h = 2 * H
        big_canvas_w = 2 * W

        frames_out = []

        # Add initial frame showing the first image at 100% scale
        canvas = torch.zeros(
            (C, big_canvas_h, big_canvas_w),
            dtype=images.dtype,
            device=images.device
        )
        self._paste_in_2x_canvas(canvas, images[0], scale=1.0)
        start_y = (big_canvas_h - H) // 2
        start_x = (big_canvas_w - W) // 2
        frames_out.append(canvas[:, start_y:start_y+H, start_x:start_x+W])

        def ease_in_out_cubic(x: float) -> float:
            """
            Smooth easing function that creates natural acceleration and deceleration
            """
            if x < 0.5:
                return 4 * x * x * x
            else:
                return 1 - pow(-2 * x + 2, 3) / 2

        def linear(x: float) -> float:
            return x

        # Select easing function based on input
        ease_func = ease_in_out_cubic if easing_function == "ease_in_out_cubic" else linear

        for f in tqdm(range(num_frames), desc="Zooming"):
            seg_index = f // frames_per_intro
            local_t = (f % frames_per_intro) / (frames_per_intro - 1e-8)
            eased_t = ease_func(local_t)  # Use selected easing function

            canvas = torch.zeros(
                (C, big_canvas_h, big_canvas_w),
                dtype=images.dtype,
                device=images.device
            )

            for i in range(seg_index + 1, -1, -1):
                if i >= N:
                    continue

                dist_from_bottom = (seg_index + 1) - i

                if dist_from_bottom == 0:
                    scale_start = 2.0
                    scale_end = 1.0
                else:
                    scale_start = 1.0 / (2.0 ** (dist_from_bottom - 1))
                    scale_end = scale_start / 2.0

                # Use eased_t instead of local_t for smoother transitions
                scale_i = scale_start + (scale_end - scale_start) * eased_t
                self._paste_in_2x_canvas(canvas, images[i], scale_i)

            # Finally, center crop (H,W) from the big (2H,2W) canvas
            # center is (big_canvas_h//2, big_canvas_w//2)
            start_y = (big_canvas_h - H) // 2
            start_x = (big_canvas_w - W) // 2
            final_frame = canvas[:, start_y:start_y+H, start_x:start_x+W]

            frames_out.append(final_frame)

        # Stack frames
        frames_out = torch.stack(frames_out, dim=0)  # [F,C,H,W]

        # Convert back to BHWC format for ComfyUI compatibility
        frames_out = frames_out.permute(0, 2, 3, 1)  # [F,H,W,C]

        return (frames_out,)


    def _paste_in_2x_canvas(self, canvas: torch.Tensor, src: torch.Tensor, scale: float):
        """
        Paste src (C,H,W) into canvas (C,2H,2W) at center with `scale` in [0..2].
        """
        Cc, Hc, Wc = canvas.shape
        Cs, Hs, Ws = src.shape

       

        # compute scaled size and ensure it's even
        new_h = int(round(Hs * scale))
        new_w = int(round(Ws * scale))
        
        # Ensure dimensions are even to prevent half-pixel offsets
        new_h = new_h + (new_h % 2)
        new_w = new_w + (new_w % 2)

        if new_h < 1 or new_w < 1:
            return  # too small

        # resize the src with explicit align_corners=True for better precision
        img_resized = F.interpolate(
            src.unsqueeze(0), 
            size=(new_h, new_w),
            mode='bilinear',
            align_corners=False
        )[0]

        # Ensure center positioning is precise
        top = (Hc - new_h) // 2
        left = (Wc - new_w) // 2

        # Add small padding to prevent edge cases
        overlap_y1 = max(0, top)
        overlap_y2 = min(Hc, top + new_h )  # Add 1 pixel padding
        overlap_x1 = max(0, left)
        overlap_x2 = min(Wc, left + new_w )  # Add 1 pixel padding

        if overlap_y2 <= overlap_y1 or overlap_x2 <= overlap_x1:
            return

        subH = overlap_y2 - overlap_y1
        subW = overlap_x2 - overlap_x1

        # subregion in img_resized
        src_y1 = overlap_y1 - top
        src_x1 = overlap_x1 - left
        src_crop = img_resized[:, src_y1:src_y1+subH, src_x1:src_x1+subW]

        # Overwrite onto canvas
        canvas[:, overlap_y1:overlap_y2, overlap_x1:overlap_x2] = src_crop

