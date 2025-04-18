import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.v2 as T

import comfy.model_management



def channels_layer(images, channels, function):
    if channels == "rgb":
        img = images[:, :, :, :3]
    elif channels == "rgba":
        img = images[:, :, :, :4]
    elif channels == "rg":
        img = images[:, :, :, [0, 1]]
    elif channels == "rb":
        img = images[:, :, :, [0, 2]]
    elif channels == "ra":
        img = images[:, :, :, [0, 3]]
    elif channels == "gb":
        img = images[:, :, :, [1, 2]]
    elif channels == "ga":
        img = images[:, :, :, [1, 3]]
    elif channels == "ba":
        img = images[:, :, :, [2, 3]]
    elif channels == "r":
        img = images[:, :, :, 0]
    elif channels == "g":
        img = images[:, :, :, 1]
    elif channels == "b":
        img = images[:, :, :, 2]
    elif channels == "a":
        img = images[:, :, :, 3]
    else:
        raise ValueError("Unsupported channels")

    result = torch.from_numpy(function(img.numpy()))

    if channels == "rgb":
        images[:, :, :, :3] = result
    elif channels == "rgba":
        images[:, :, :, :4] = result
    elif channels == "rg":
        images[:, :, :, [0, 1]] = result
    elif channels == "rb":
        images[:, :, :, [0, 2]] = result
    elif channels == "ra":
        images[:, :, :, [0, 3]] = result
    elif channels == "gb":
        images[:, :, :, [1, 2]] = result
    elif channels == "ga":
        images[:, :, :, [1, 3]] = result
    elif channels == "ba":
        images[:, :, :, [2, 3]] = result
    elif channels == "r":
        images[:, :, :, 0] = result
    elif channels == "g":
        images[:, :, :, 1] = result
    elif channels == "b":
        images[:, :, :, 2] = result
    elif channels == "a":
        images[:, :, :, 3] = result

    return images



class latent_chx_noise:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "strength": ("FLOAT", {
                    "default": 0.5,
                    "step": 0.01
                }),
                "monochromatic": (["false", "true"],),
                "invert": (["false", "true"],),
                "channels": (["rgb", "rgba", "rg", "rb", "ra", "gb", "ga", "ba", "r", "g", "b", "a"],),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "node"
    CATEGORY = "Apt_Collect/latent"

    def noise(self, images, strength, monochromatic, invert):
        if monochromatic and images.shape[3] > 1:
            noise = np.random.normal(0, 1, images.shape[:3])
        else:
            noise = np.random.normal(0, 1, images.shape)

        noise = np.abs(noise)
        noise /= noise.max()

        if monochromatic and images.shape[3] > 1:
            noise = noise[..., np.newaxis].repeat(images.shape[3], -1)

        if invert:
            noise = images - noise * strength
        else:
            noise = images + noise * strength

        noise = np.clip(noise, 0.0, 1.0)
        noise = noise.astype(images.dtype)

        return noise

    def node(self, images, strength, monochromatic, invert, channels):
        tensor = images.clone().detach()

        monochromatic = True if monochromatic == "true" else False
        invert = True if invert == "true" else False

        return (channels_layer(tensor, channels, lambda x: self.noise(
            x, strength, monochromatic, invert
        )),)


class latent_Image2Noise:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "noise_strenght": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01 }),
                "noise_size": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01 }),
                "color_noise": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01 }),
                "mask_strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01 }),
                "mask_scale_diff": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01 }),
                "mask_contrast": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1 }),
                "saturation": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 100.0, "step": 0.1 }),
                "contrast": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1 }),
                "blur": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1 }),
            },
            "optional": {
                "noise_mask": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "Apt_Collect/latent"

    def execute(self, image, noise_size, color_noise, mask_strength, mask_scale_diff, mask_contrast, noise_strenght, saturation, contrast, blur, noise_mask=None):
        torch.manual_seed(0)

        elastic_alpha = max(image.shape[1], image.shape[2])# * noise_size
        elastic_sigma = elastic_alpha / 400 * noise_size

        blur_size = int(6 * blur+1)
        if blur_size % 2 == 0:
            blur_size+= 1

        if noise_mask is None:
            noise_mask = image
        
        # increase contrast of the mask
        if mask_contrast != 1:
            noise_mask = T.ColorJitter(contrast=(mask_contrast,mask_contrast))(noise_mask.permute([0, 3, 1, 2])).permute([0, 2, 3, 1])

        # Ensure noise mask is the same size as the image
        if noise_mask.shape[1:] != image.shape[1:]:
            noise_mask = F.interpolate(noise_mask.permute([0, 3, 1, 2]), size=(image.shape[1], image.shape[2]), mode='bicubic', align_corners=False)
            noise_mask = noise_mask.permute([0, 2, 3, 1])
        # Ensure we have the same number of masks and images
        if noise_mask.shape[0] > image.shape[0]:
            noise_mask = noise_mask[:image.shape[0]]
        else:
            noise_mask = torch.cat((noise_mask, noise_mask[-1:].repeat((image.shape[0]-noise_mask.shape[0], 1, 1, 1))), dim=0)

        # Convert mask to grayscale mask
        noise_mask = noise_mask.mean(dim=3).unsqueeze(-1)

        # add color noise
        imgs = image.clone().permute([0, 3, 1, 2])
        if color_noise > 0:
            color_noise = torch.normal(torch.zeros_like(imgs), std=color_noise)
            color_noise *= (imgs - imgs.min()) / (imgs.max() - imgs.min())

            imgs = imgs + color_noise
            imgs = imgs.clamp(0, 1)

        # create fine and coarse noise
        fine_noise = []
        for n in imgs:
            avg_color = n.mean(dim=[1,2])

            tmp_noise = T.ElasticTransform(alpha=elastic_alpha, sigma=elastic_sigma, fill=avg_color.tolist())(n)
            if blur > 0:
                tmp_noise = T.GaussianBlur(blur_size, blur)(tmp_noise)
            tmp_noise = T.ColorJitter(contrast=(contrast,contrast), saturation=(saturation,saturation))(tmp_noise)
            fine_noise.append(tmp_noise)

        imgs = None
        del imgs

        fine_noise = torch.stack(fine_noise, dim=0)
        fine_noise = fine_noise.permute([0, 2, 3, 1])
        #fine_noise = torch.stack(fine_noise, dim=0)
        #fine_noise = pb(fine_noise)
        mask_scale_diff = min(mask_scale_diff, 0.99)
        if mask_scale_diff > 0:
            coarse_noise = F.interpolate(fine_noise.permute([0, 3, 1, 2]), scale_factor=1-mask_scale_diff, mode='area')
            coarse_noise = F.interpolate(coarse_noise, size=(fine_noise.shape[1], fine_noise.shape[2]), mode='bilinear', align_corners=False)
            coarse_noise = coarse_noise.permute([0, 2, 3, 1])
        else:
            coarse_noise = fine_noise

        output = (1 - noise_mask) * coarse_noise + noise_mask * fine_noise

        if mask_strength < 1:
            noise_mask = noise_mask.pow(mask_strength)
            noise_mask = torch.nan_to_num(noise_mask).clamp(0, 1)
        output = noise_mask * output + (1 - noise_mask) * image

        # apply noise to image
        output = output * noise_strenght + image * (1 - noise_strenght)
        output = output.clamp(0, 1)

        return (output, )



class latent_mask:
    def __init__(self):
        self.device = comfy.model_management.intermediate_device()

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "width": ("INT", {"default": 512, "min": 16, "max": 5120, "step": 8, "tooltip": "The width of the latent images in pixels."}),
                "height": ("INT", {"default": 512, "min": 16, "max": 5120, "step": 8, "tooltip": "The height of the latent images in pixels."}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096, "tooltip": "The number of latent images in the batch."}),

            },
            "optional": {
                "mask_op": ("MASK",)
            }
            
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "generate"
    CATEGORY = "Apt_Collect/latent"

    def generate(self, width, height, batch_size, mask_op=None):
        
        if mask_op is None:
            latent = torch.zeros([batch_size, 4, height // 8, width // 8], device=self.device)
            return ({"samples":latent}, )
        
        latent = torch.zeros([batch_size, 4, height // 8, width // 8], device=self.device)
        latent_dict = {"samples": latent}
        latent_dict["noise_mask"] = mask_op.reshape((-1, 1, mask_op.shape[-2], mask_op.shape[-1]))
        
        return (latent_dict,)


