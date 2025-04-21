
from .AdvancedControlNet.utils import *
from .AdvancedControlNet.nodes_main import AdvancedControlNetApply
from .AdvancedControlNet.utils import StrengthInterpolation as SI
from collections.abc import Iterable
from .ScheduledNodes import batch_get_inbetweens,batch_parse_key_frames


defaultValue = """0:(0),
30:(1),
60:(0),
90:(0),
120:(0)
"""



class AD_sch_adv_CN:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "conditioning": ("CONDITIONING", ),
                "control_net": ("CONTROL_NET", ),
                "image": ("IMAGE", ),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
            },
            "optional": {
                "mask_optional": ("MASK", ),
                "latent_kf_override": ("LATENT_KEYFRAME", ),
                #"timestep_kf": ("TIMESTEP_KEYFRAME", ),
                #"weights_override": ("CONTROL_NET_WEIGHTS", ),
                #"model_optional": ("MODEL",),
                #"vae_optional": ("VAE",),
            },
            "hidden": {
                "autosize": ("ACNAUTOSIZE", {"padding": 0}),
            }
        }

    DEPRECATED = True
    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("CONDITIONING",)
    FUNCTION = "apply_controlnet"

    CATEGORY = "Apt_Preset/AD"

    def apply_controlnet(self, conditioning, control_net, image, strength, 
                        mask_optional=None, model_optional=None, vae_optional=None,
                        timestep_kf: TimestepKeyframeGroup=None, latent_kf_override=None,
                        weights_override: ControlWeights=None):
        values = AdvancedControlNetApply.apply_controlnet(self, positive=conditioning, negative=None, control_net=control_net, image=image,
                                                        strength=strength, start_percent=0, end_percent=1,
                                                        mask_optional=mask_optional, vae_optional=vae_optional,
                                                        timestep_kf=timestep_kf, latent_kf_override=latent_kf_override, weights_override=weights_override,
                                                        control_apply_to_uncond=True)
        return (values[0], model_optional)



class AD_latent_keyframe:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "batch_index_from": ("INT", {"default": 0, "min": BIGMIN, "max": BIGMAX, "step": 1}),
                "batch_index_to_excl": ("INT", {"default": 0, "min": BIGMIN, "max": BIGMAX, "step": 1}),
                "strength_from": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001}, ),
                "strength_to": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001}, ),
                "interpolation": (SI._LIST, ),
            },
            "optional": {
                "prev_latent_kf": ("LATENT_KEYFRAME", ),
                #"print_keyframes": ("BOOLEAN", {"default": False}),
            },
            "hidden": {
                "autosize": ("ACNAUTOSIZE", {"padding": 0}),
            }
        }

    RETURN_NAMES = ("LATENT_KF", )
    RETURN_TYPES = ("LATENT_KEYFRAME", )
    FUNCTION = "load_keyframe"
    CATEGORY = "Apt_Preset/AD"

    def load_keyframe(self,
                        batch_index_from: int,
                        strength_from: float,
                        batch_index_to_excl: int,
                        strength_to: float,
                        interpolation: str,
                        prev_latent_kf: LatentKeyframeGroup=None,
                        prev_latent_keyframe: LatentKeyframeGroup=None, # old name
                        print_keyframes=False):

        if (batch_index_from > batch_index_to_excl):
            raise ValueError("batch_index_from must be less than or equal to batch_index_to.")

        if (batch_index_from < 0 and batch_index_to_excl >= 0):
            raise ValueError("batch_index_from and batch_index_to must be either both positive or both negative.")

        prev_latent_keyframe = prev_latent_keyframe if prev_latent_keyframe else prev_latent_kf
        if not prev_latent_keyframe:
            prev_latent_keyframe = LatentKeyframeGroup()
        else:
            prev_latent_keyframe = prev_latent_keyframe.clone()
        curr_latent_keyframe = LatentKeyframeGroup()

        steps = batch_index_to_excl - batch_index_from
        diff = strength_to - strength_from
        if interpolation == SI.LINEAR:
            weights = np.linspace(strength_from, strength_to, steps)
        elif interpolation == SI.EASE_IN:
            index = np.linspace(0, 1, steps)
            weights = diff * np.power(index, 2) + strength_from
        elif interpolation == SI.EASE_OUT:
            index = np.linspace(0, 1, steps)
            weights = diff * (1 - np.power(1 - index, 2)) + strength_from
        elif interpolation == SI.EASE_IN_OUT:
            index = np.linspace(0, 1, steps)
            weights = diff * ((1 - np.cos(index * np.pi)) / 2) + strength_from

        for i in range(steps):
            keyframe = LatentKeyframe(batch_index_from + i, float(weights[i]))
            curr_latent_keyframe.add(keyframe)
        
        if print_keyframes:
            for keyframe in curr_latent_keyframe.keyframes:
                logger.info(f"LatentKeyframe {keyframe.batch_index}={keyframe.strength}")

        # replace values with prev_latent_keyframes
        for latent_keyframe in prev_latent_keyframe.keyframes:
            curr_latent_keyframe.add(latent_keyframe)

        return (curr_latent_keyframe,)



class AD_latent_kfGroup:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "index_strengths": ("STRING", {"multiline": True, "default": "0=1.0, 1:10=0.6, 10=0.5, 11=0.3"}),
            },
            "optional": {
                "prev_latent_kf": ("LATENT_KEYFRAME", ),
                "latent_optional": ("LATENT", ),
                #"print_keyframes": ("BOOLEAN", {"default": False}),
            },
            "hidden": {
                "autosize": ("ACNAUTOSIZE", {"padding": 0}),
            }
        }
    
    RETURN_NAMES = ("LATENT_KF", )
    RETURN_TYPES = ("LATENT_KEYFRAME", )
    FUNCTION = "load_keyframes"

    CATEGORY = "Apt_Preset/AD"

    def validate_index(self, index: int, latent_count: int = 0, is_range: bool = False, allow_negative = False) -> int:
        # if part of range, do nothing
        if is_range:
            return index
        # otherwise, validate index
        # validate not out of range - only when latent_count is passed in
        if latent_count > 0 and index > latent_count-1:
            raise IndexError(f"Index '{index}' out of range for the total {latent_count} latents.")
        # if negative, validate not out of range
        if index < 0:
            if not allow_negative:
                raise IndexError(f"Negative indeces not allowed, but was {index}.")
            conv_index = latent_count+index
            if conv_index < 0:
                raise IndexError(f"Index '{index}', converted to '{conv_index}' out of range for the total {latent_count} latents.")
            index = conv_index
        return index

    def convert_to_index_int(self, raw_index: str, latent_count: int = 0, is_range: bool = False, allow_negative = False) -> int:
        try:
            return self.validate_index(int(raw_index), latent_count=latent_count, is_range=is_range, allow_negative=allow_negative)
        except ValueError as e:
            raise ValueError(f"index '{raw_index}' must be an integer.", e)

    def convert_to_latent_keyframes(self, latent_indeces: str, latent_count: int) -> set[LatentKeyframe]:
        if not latent_indeces:
            return set()
        int_latent_indeces = [i for i in range(0, latent_count)]
        allow_negative = latent_count > 0
        chosen_indeces = set()
        # parse string - allow positive ints, negative ints, and ranges separated by ':'
        groups = latent_indeces.split(",")
        groups = [g.strip() for g in groups]
        for g in groups:
            # parse strengths - default to 1.0 if no strength given
            strength = 1.0
            if '=' in g:
                g, strength_str = g.split("=", 1)
                g = g.strip()
                try:
                    strength = float(strength_str.strip())
                except ValueError as e:
                    raise ValueError(f"strength '{strength_str}' must be a float.", e)
                if strength < 0:
                    raise ValueError(f"Strength '{strength}' cannot be negative.")
            # parse range of indeces (e.g. 2:16)
            if ':' in g:
                index_range = g.split(":", 1)
                index_range = [r.strip() for r in index_range]
                start_index = self.convert_to_index_int(index_range[0], latent_count=latent_count, is_range=True, allow_negative=allow_negative)
                end_index = self.convert_to_index_int(index_range[1], latent_count=latent_count, is_range=True, allow_negative=allow_negative)
                # if latents were passed in, base indeces on known latent count
                if len(int_latent_indeces) > 0:
                    for i in int_latent_indeces[start_index:end_index]:
                        chosen_indeces.add(LatentKeyframe(i, strength))
                # otherwise, assume indeces are valid
                else:
                    for i in range(start_index, end_index):
                        chosen_indeces.add(LatentKeyframe(i, strength))
            # parse individual indeces
            else:
                chosen_indeces.add(LatentKeyframe(self.convert_to_index_int(g, latent_count=latent_count, allow_negative=allow_negative), strength))
        return chosen_indeces

    def load_keyframes(self,
                    index_strengths: str,
                    prev_latent_kf: LatentKeyframeGroup=None,
                    prev_latent_keyframe: LatentKeyframeGroup=None, # old name
                    latent_image_opt=None,
                    print_keyframes=False):
        prev_latent_keyframe = prev_latent_keyframe if prev_latent_keyframe else prev_latent_kf
        if not prev_latent_keyframe:
            prev_latent_keyframe = LatentKeyframeGroup()
        else:
            prev_latent_keyframe = prev_latent_keyframe.clone()
        curr_latent_keyframe = LatentKeyframeGroup()

        latent_count = -1
        if latent_image_opt:
            latent_count = latent_image_opt['samples'].size()[0]
        latent_keyframes = self.convert_to_latent_keyframes(index_strengths, latent_count=latent_count)

        for latent_keyframe in latent_keyframes:
            curr_latent_keyframe.add(latent_keyframe)
        
        if print_keyframes:
            for keyframe in curr_latent_keyframe.keyframes:
                logger.info(f"LatentKeyframe {keyframe.batch_index}={keyframe.strength}")

        # replace values with prev_latent_keyframes
        for latent_keyframe in prev_latent_keyframe.keyframes:
            curr_latent_keyframe.add(latent_keyframe)

        return (curr_latent_keyframe,)



class AD_sch_latent:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": defaultValue}),
                "max_frames": ("INT", {"default": 300, "min": 1, "max": 999999, "step": 1}),
            },
            "optional": {
                "prev_latent_kf": ("LATENT_KEYFRAME", ),
            },
            "hidden": {
                "autosize": ("ACNAUTOSIZE", {"padding": 0}),
            }
        }

    RETURN_NAMES = ("LATENT_KF", )
    RETURN_TYPES = ("LATENT_KEYFRAME", )
    FUNCTION = "load_keyframe"
    CATEGORY = "Apt_Preset/AD"

    def load_keyframe(self, text, max_frames, print_output=None,
                    prev_latent_kf: LatentKeyframeGroup=None,
                    prev_latent_keyframe: LatentKeyframeGroup=None, # old name
                    ):
        print_output=None
        
        t, _ = self.animate(text, max_frames, print_output)
        float_strengths = t

        prev_latent_keyframe = prev_latent_keyframe if prev_latent_keyframe else prev_latent_kf
        if not prev_latent_keyframe:
            prev_latent_keyframe = LatentKeyframeGroup()
        else:
            prev_latent_keyframe = prev_latent_keyframe.clone()
        curr_latent_keyframe = LatentKeyframeGroup()

        # if received a normal float input, do nothing
        if type(float_strengths) in (float, int):
            logger.info("No batched float_strengths passed into Latent Keyframe Batch Group node; will not create any new keyframes.")
        # if iterable, attempt to create LatentKeyframes with chosen strengths
        elif isinstance(float_strengths, Iterable):
            for idx, strength in enumerate(float_strengths):
                keyframe = LatentKeyframe(idx, strength)
                curr_latent_keyframe.add(keyframe)
        else:
            raise ValueError(f"Expected strengths to be an iterable input, but was {type(float_strengths).__repr__}.")    

        # replace values with prev_latent_keyframes
        for latent_keyframe in prev_latent_keyframe.keyframes:
            curr_latent_keyframe.add(latent_keyframe)

        return (curr_latent_keyframe,)

    def animate(self, text, max_frames, print_output=None):
        t = batch_get_inbetweens(batch_parse_key_frames(text, max_frames), max_frames)
        return (t, list(map(int,t)),)