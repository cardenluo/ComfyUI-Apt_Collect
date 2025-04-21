
import numexpr
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import re
import json
import comfy.samplers
from ..def_unit import *


#region---------------------def----------------------


class ScheduleSettings:
    def __init__(
            self,
            text_g: str,
            pre_text_G: str,
            app_text_G: str,
            text_L: str,
            pre_text_L: str,
            app_text_L: str,
            max_frames: int,
            current_frame: int,
            print_output: bool,
            pw_a: float,
            pw_b: float,
            pw_c: float,
            pw_d: float,
            start_frame: int,
            end_frame:int,
            width: int,
            height: int,
            crop_w: int,
            crop_h: int,
            target_width: int,
            target_height: int,
    ):
        self.text_g=text_g
        self.pre_text_G=pre_text_G
        self.app_text_G=app_text_G
        self.text_l=text_L
        self.pre_text_L=pre_text_L
        self.app_text_L=app_text_L
        self.max_frames=max_frames
        self.current_frame=current_frame
        self.print_output=print_output
        self.pw_a=pw_a
        self.pw_b=pw_b
        self.pw_c=pw_c
        self.pw_d=pw_d
        self.start_frame=start_frame
        self.end_frame=end_frame
        self.width=width
        self.height=height
        self.crop_w=crop_w
        self.crop_h=crop_h
        self.target_width=target_width
        self.target_height=target_height

    def set_sync_option(self, sync_option: bool):
        self.sync_context_to_pe = sync_option


defaultPrompt = """"0" :"",
"30" :"",
"60" :"",
"90" :"",
"120" :""
"""


defaultValue = """0:(0),
30:(0),
60:(0),
90:(0),
120:(0)
"""


def batch_parse_key_frames(string, max_frames):
    # because math functions (i.e. sin(t)) can utilize brackets
    # it extracts the value in form of some stuff
    # which has previously been enclosed with brackets and
    # with a comma or end of line existing after the closing one
    string = re.sub(r',\s*$', '', string)
    frames = dict()
    for match_object in string.split(","):
        frameParam = match_object.split(":")
        max_f = max_frames - 1  # needed for numexpr even though it doesn't look like it's in use.
        frame = int(sanitize_value(frameParam[0])) if check_is_number(
            sanitize_value(frameParam[0].strip())) else int(numexpr.evaluate(
            frameParam[0].strip().replace("'", "", 1).replace('"', "", 1)[::-1].replace("'", "", 1).replace('"', "",1)[::-1]))
        frames[frame] = frameParam[1].strip()
    if frames == {} and len(string) != 0:
        raise RuntimeError('Key Frame string not correctly formatted')
    return frames


def sanitize_value(value):
    # Remove single quotes, double quotes, and parentheses
    value = value.replace("'", "").replace('"', "").replace('(', "").replace(')', "")
    return value


def batch_get_inbetweens(key_frames, max_frames, integer=False, interp_method='Linear', is_single_string=False):
    key_frame_series = pd.Series([np.nan for a in range(max_frames)])
    max_f = max_frames - 1  # needed for numexpr even though it doesn't look like it's in use.
    value_is_number = False
    for i in range(0, max_frames):
        if i in key_frames:
            value = str(key_frames[i])  # Convert to string to ensure it's treated as an expression
            value_is_number = check_is_number(sanitize_value(value))
            if value_is_number:
                key_frame_series[i] = sanitize_value(value)
        if not value_is_number:
            t = i
            # workaround for values formatted like 0:("I am test") //used for sampler schedules
            key_frame_series[i] = numexpr.evaluate(value) if not is_single_string else sanitize_value(value)
        elif is_single_string:  # take previous string value and replicate it
            key_frame_series[i] = key_frame_series[i - 1]
    key_frame_series = key_frame_series.astype(float) if not is_single_string else key_frame_series  # as string

    if interp_method == 'Cubic' and len(key_frames.items()) <= 3:
        interp_method = 'Quadratic'
    if interp_method == 'Quadratic' and len(key_frames.items()) <= 2:
        interp_method = 'Linear'

    key_frame_series[0] = key_frame_series[key_frame_series.first_valid_index()]
    key_frame_series[max_frames - 1] = key_frame_series[key_frame_series.last_valid_index()]
    key_frame_series = key_frame_series.interpolate(method=interp_method.lower(), limit_direction='both')

    if integer:
        return key_frame_series.astype(int)
    return key_frame_series

#--------------------------------------------------------------------------------------
def batch_prompt_schedule(settings:ScheduleSettings,clip):
    # Clear whitespace and newlines from json
    animation_prompts = process_input_text(settings.text_g)

    # Add pre_text and app_text then split the combined prompt into positive and negative prompts
    pos, neg = batch_split_weighted_subprompts(animation_prompts, settings.pre_text_G, settings.app_text_G)

    # Interpolate the positive prompt weights over frames
    pos_cur_prompt, pos_nxt_prompt, weight = interpolate_prompt_seriesA(pos, settings)
    neg_cur_prompt, neg_nxt_prompt, weight = interpolate_prompt_seriesA(neg, settings)

    # Apply composable diffusion across the batch
    p = BatchPoolAnimConditioning(pos_cur_prompt, pos_nxt_prompt, weight, clip, settings)
    n = BatchPoolAnimConditioning(neg_cur_prompt, neg_nxt_prompt, weight, clip, settings)

    # return positive and negative conditioning as well as the current and next prompts for each
    return (p, n,)


def process_input_text(text: str) -> dict:
    input_text = text.replace('\n', '')
    input_text = "{" + input_text + "}"
    input_text = re.sub(r',\s*}', '}', input_text)
    animation_prompts = json.loads(input_text.strip())
    return animation_prompts


def BatchPoolAnimConditioning(cur_prompt_series, nxt_prompt_series, weight_series, clip, settings:ScheduleSettings):
    pooled_out = []
    cond_out = []
    max_size = 0

    if settings.end_frame == 0:
        settings.end_frame = settings.max_frames
        print("end_frame at 0, using max_frames instead!")

    if settings.start_frame >= settings.end_frame:
        settings.start_frame = 0
        print("start_frame larger than or equal to end_frame, using max_frames instead!")

    if max_size == 0:
        for i in range(0, settings.end_frame):
            tokens = clip.tokenize(str(cur_prompt_series[i]))
            cond_to, pooled_to = clip.encode_from_tokens(tokens, return_pooled=True)
            max_size = max(max_size, cond_to.shape[1])
    for i in range(settings.start_frame, settings.end_frame):
        tokens = clip.tokenize(str(cur_prompt_series[i]))
        cond_to, pooled_to = clip.encode_from_tokens(tokens, return_pooled=True)

        if i < len(nxt_prompt_series):
            tokens = clip.tokenize(str(nxt_prompt_series[i]))
            cond_from, pooled_from = clip.encode_from_tokens(tokens, return_pooled=True)
        else:
            cond_from, pooled_from = torch.zeros_like(cond_to), torch.zeros_like(pooled_to)

        interpolated_conditioning = addWeighted([[cond_to, {"pooled_output": pooled_to}]],
                                                [[cond_from, {"pooled_output": pooled_from}]],
                                                weight_series[i],max_size)

        interpolated_cond = interpolated_conditioning[0][0]
        interpolated_pooled = interpolated_conditioning[0][1].get("pooled_output", pooled_from)

        cond_out.append(interpolated_cond)
        pooled_out.append(interpolated_pooled)

    final_pooled_output = torch.cat(pooled_out, dim=0)
    final_conditioning = torch.cat(cond_out, dim=0)

    return [[final_conditioning, {"pooled_output": final_pooled_output}]]


def addWeighted(conditioning_to, conditioning_from, conditioning_to_strength, max_size=0):
    out = []

    if len(conditioning_from) > 1:
        print("Warning: ConditioningAverage conditioning_from contains more than 1 cond, only the first one will actually be applied to conditioning_to.")

    cond_from = conditioning_from[0][0]
    pooled_output_from = conditioning_from[0][1].get("pooled_output", None)

    for i in range(len(conditioning_to)):
        t1 = conditioning_to[i][0]
        pooled_output_to = conditioning_to[i][1].get("pooled_output", pooled_output_from)
        if max_size == 0:
            max_size = max(t1.shape[1], cond_from.shape[1])
        t0, max_size = pad_with_zeros(cond_from, max_size)
        t1, max_size = pad_with_zeros(t1, t0.shape[1])  # Padding t1 to match max_size
        t0, max_size = pad_with_zeros(t0, t1.shape[1])

        tw = torch.mul(t1, conditioning_to_strength) + torch.mul(t0, (1.0 - conditioning_to_strength))
        t_to = conditioning_to[i][1].copy()

        t_to["pooled_output"] = pooled_output_from
        n = [tw, t_to]
        out.append(n)

    return out


def pad_with_zeros(tensor, target_length):
    current_length = tensor.shape[1]

    if current_length < target_length:
        # Calculate the required padding length
        pad_length = target_length - current_length

        # Calculate padding on both sides to maintain the tensor's original shape
        left_pad = pad_length // 2
        right_pad = pad_length - left_pad

        # Pad the tensor along the second dimension
        tensor = F.pad(tensor, (0, 0, left_pad, right_pad))

    return tensor, target_length


def batch_split_weighted_subprompts(text, pre_text, app_text):
    pos = {}
    neg = {}
    pre_text = str(pre_text)
    app_text = str(app_text)

    if "--neg" in pre_text:
        pre_pos, pre_neg = pre_text.split("--neg")
    else:
        pre_pos, pre_neg = pre_text, ""

    if "--neg" in app_text:
        app_pos, app_neg = app_text.split("--neg")
    else:
        app_pos, app_neg = app_text, ""

    for frame, prompt in text.items():
        negative_prompts = ""
        positive_prompts = ""
        prompt_split = prompt.split("--neg")

        if len(prompt_split) > 1:
            positive_prompts, negative_prompts = prompt_split[0], prompt_split[1]
        else:
            positive_prompts = prompt_split[0]

        pos[frame] = ""
        neg[frame] = ""
        pos[frame] += (str(pre_pos) + " " + positive_prompts + " " + str(app_pos))
        neg[frame] += (str(pre_neg) + " " + negative_prompts + " " + str(app_neg))
        if pos[frame].endswith('0'):
            pos[frame] = pos[frame][:-1]
        if neg[frame].endswith('0'):
            neg[frame] = neg[frame][:-1]
    return pos, neg


def check_is_number(value):
    float_pattern = r'^(?=.)([+-]?([0-9]*)(\.([0-9]+))?)$'
    return re.match(float_pattern, value)


def convert_pw_to_tuples(settings):
    if isinstance(settings.pw_a, (int, float, np.float64)):
        settings.pw_a = tuple([settings.pw_a] * settings.max_frames)
    if isinstance(settings.pw_b, (int, float, np.float64)):
        settings.pw_b = tuple([settings.pw_b] * settings.max_frames)
    if isinstance(settings.pw_c, (int, float, np.float64)):
        settings.pw_c = tuple([settings.pw_c] * settings.max_frames)
    if isinstance(settings.pw_d, (int, float, np.float64)):
        settings.pw_d = tuple([settings.pw_d] * settings.max_frames)


def interpolate_prompt_seriesA(animation_prompts, settings:ScheduleSettings):

    max_f = settings.max_frames  # needed for numexpr even though it doesn't look like it's in use.
    parsed_animation_prompts = {}


    for key, value in animation_prompts.items():
        if check_is_number(key):  # default case 0:(1 + t %5), 30:(5-t%2)
            parsed_animation_prompts[key] = value
        else:  # math on the left hand side case 0:(1 + t %5), maxKeyframes/2:(5-t%2)
            parsed_animation_prompts[int(numexpr.evaluate(key))] = value

    sorted_prompts = sorted(parsed_animation_prompts.items(), key=lambda item: int(item[0]))

    # Automatically set the first keyframe to 0 if it's missing
    if sorted_prompts[0][0] != "0":
        sorted_prompts.insert(0, ("0", sorted_prompts[0][1]))

    # Automatically set the last keyframe to the maximum number of frames
    if sorted_prompts[-1][0] != str(settings.max_frames):
        sorted_prompts.append((str(settings.max_frames), sorted_prompts[-1][1]))

    # Setup containers for interpolated prompts
    nan_list = [np.nan for a in range(settings.max_frames)]
    cur_prompt_series = pd.Series(nan_list,dtype=object)
    nxt_prompt_series = pd.Series(nan_list,dtype=object)

    # simple array for strength values
    weight_series = [np.nan] * settings.max_frames

    # in case there is only one keyed prompt, set all prompts to that prompt
    if settings.max_frames == 1:
        for i in range(0, len(cur_prompt_series) - 1):
            current_prompt = sorted_prompts[0][1]
            cur_prompt_series[i] = str(current_prompt)
            nxt_prompt_series[i] = str(current_prompt)

    #make sure prompt weights are tuples and convert them if not
    convert_pw_to_tuples(settings)

    # Initialized outside of loop for nan check
    current_key = 0
    next_key = 0

    # For every keyframe prompt except the last
    for i in range(0, len(sorted_prompts) - 1):
        # Get current and next keyframe
        current_key = int(sorted_prompts[i][0])
        next_key = int(sorted_prompts[i + 1][0])

        # Ensure there's no weird ordering issues or duplication in the animation prompts
        # (unlikely because we sort above, and the json parser will strip dupes)
        if current_key >= next_key:
            print(
                f"WARNING: Sequential prompt keyframes {i}:{current_key} and {i + 1}:{next_key} are not monotonously increasing; skipping interpolation.")
            continue

        # Get current and next keyframes' positive and negative prompts (if any)
        current_prompt = sorted_prompts[i][1]
        next_prompt = sorted_prompts[i + 1][1]

        # Calculate how much to shift the weight from current to next prompt at each frame.
        weight_step = 1 / (next_key - current_key)

        for f in range(max(current_key, 0), min(next_key, len(cur_prompt_series))):
            next_weight = weight_step * (f - current_key)
            current_weight = 1 - next_weight

            # add the appropriate prompts and weights to their respective containers.
            weight_series[f] = 0.0
            cur_prompt_series[f] = str(current_prompt)
            nxt_prompt_series[f] = str(next_prompt)

            weight_series[f] += current_weight

        current_key = next_key
        next_key = settings.max_frames
        current_weight = 0.0

    index_offset = 0

    # Evaluate the current and next prompt's expressions
    for i in range(settings.start_frame, min(settings.end_frame,len(cur_prompt_series))):
        cur_prompt_series[i] = prepare_batch_promptA(cur_prompt_series[i], settings, i)
        nxt_prompt_series[i] = prepare_batch_promptA(nxt_prompt_series[i], settings, i)
        if settings.print_output == True:
            # Show the to/from prompts with evaluated expressions for transparency.
            if(settings.start_frame >= i):
                if(settings.end_frame > 0):
                    if(settings.end_frame > i):
                        print("\n", "Max Frames: ", settings.max_frames, "\n", "frame index: ", (settings.start_frame + i),
                              "\n", "Current Prompt: ",
                              cur_prompt_series[i], "\n", "Next Prompt: ", nxt_prompt_series[i], "\n", "Strength : ",
                              weight_series[i], "\n")
                else:
                    print("\n", "Max Frames: ", settings.max_frames, "\n", "frame index: ", (settings.start_frame + i), "\n", "Current Prompt: ",
                          cur_prompt_series[i], "\n", "Next Prompt: ", nxt_prompt_series[i], "\n", "Strength : ",
                          weight_series[i], "\n")
        index_offset = index_offset + 1

    # Output methods depending if the prompts are the same or if the current frame is a keyframe.
    # if it is an in-between frame and the prompts differ, composable diffusion will be performed.
    return (cur_prompt_series, nxt_prompt_series, weight_series)


def prepare_batch_promptA(prompt, settings:ScheduleSettings, index):
    max_f = settings.max_frames - 1
    pattern = r'`.*?`'  # set so the expression will be read between two backticks (``)
    regex = re.compile(pattern)
    prompt_parsed = str(prompt)

    for match in regex.finditer(prompt_parsed):
        matched_string = match.group(0)
        parsed_string = matched_string.replace(
            't',
            f'{index}').replace("pw_a",
            f"{settings.pw_a[index]}").replace("pw_b",
            f"{settings.pw_b[index]}").replace("pw_c",
            f"{settings.pw_c[index]}").replace("pw_d",
            f"{settings.pw_d[index]}").replace("max_f",
            f"{max_f}").replace('`', '')  # replace t, max_f and `` respectively
        parsed_value = numexpr.evaluate(parsed_string)
        prompt_parsed = prompt_parsed.replace(matched_string, str(parsed_value))
    return prompt_parsed.strip()


#endregion--------------------------------def-------------------------------------------------



class AD_sch_Value:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"text": ("STRING", {"multiline": True, "default": defaultValue}),
                            "max_frames": ("INT", {"default": 120.0, "min": 1.0, "max": 999999.0, "step": 1.0}),
                            "print_output": ("BOOLEAN", {"default": False})}}

    RETURN_TYPES = ("FLOAT", "INT")
    FUNCTION = "animate"
    CATEGORY = "Apt_Preset/AD"

    def animate(self, text, max_frames, print_output):
        t = batch_get_inbetweens(batch_parse_key_frames(text, max_frames), max_frames)
        if print_output is True:
            print("ValueSchedule: ", t)
        return (t, list(map(int,t)),)


class AD_prompt_Schedule:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "text": ("STRING", {"multiline": True, "default": defaultPrompt}),
                    "clip": ("CLIP",),
                    "max_frames": ("INT", {"default": 120.0, "min": 1.0, "max": 999999.0, "step": 1.0}),

                },
                "optional": {
                    "pre_text": ("STRING", {"multiline": False, }),
                    "app_text": ("STRING", {"multiline": False, }),

                    "pw_a": ("FLOAT", {"default": 0.0, "min": -9999.0, "max": 9999.0, "step": 0.1, }),
                    "pw_b": ("FLOAT", {"default": 0.0, "min": -9999.0, "max": 9999.0, "step": 0.1, }),
                    "pw_c": ("FLOAT", {"default": 0.0, "min": -9999.0, "max": 9999.0, "step": 0.1, }),
                    "pw_d": ("FLOAT", {"default": 0.0, "min": -9999.0, "max": 9999.0, "step": 0.1, }),
                }
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING",)
    RETURN_NAMES = ("positive", "negative",)
    FUNCTION = "animate"
    CATEGORY = "Apt_Preset/AD"
    
    def animate(self, text, max_frames, clip, pw_a=0, pw_b=0, pw_c=0, pw_d=0, pre_text='', app_text=''):
        

        settings = ScheduleSettings(
            text_g=text,
            pre_text_G=pre_text,
            app_text_G=app_text,
            text_L=None,
            pre_text_L=None,
            app_text_L=None,
            max_frames=max_frames,
            current_frame=None,
            print_output=None,
            pw_a=pw_a,
            pw_b=pw_b,
            pw_c=pw_c,
            pw_d=pw_d,
            start_frame=0,
            end_frame=max_frames,
            width=None,
            height=None,
            crop_w=None,
            crop_h=None,
            target_width=None,
            target_height=None,
        )
        
        positive, negative = batch_prompt_schedule(settings, clip)
        
        return (positive, negative)


class chx_prompt_Schedule:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "context": ("RUN_CONTEXT",),
                    "text": ("STRING", {"multiline": True, "default": defaultPrompt}),
                    "max_frames": ("INT", {"default": 120.0, "min": 1.0, "max": 999999.0, "step": 1.0}),
                },
                "optional": {
                    "pre_text": ("STRING", {"multiline": False, }),
                    "app_text": ("STRING", {"multiline": False, }),
                    "pw_a": ("FLOAT", {"default": 0.0, "min": -9999.0, "max": 9999.0, "step": 0.1, }),
                    "pw_b": ("FLOAT", {"default": 0.0, "min": -9999.0, "max": 9999.0, "step": 0.1, }),
                    "pw_c": ("FLOAT", {"default": 0.0, "min": -9999.0, "max": 9999.0, "step": 0.1, }),
                    "pw_d": ("FLOAT", {"default": 0.0, "min": -9999.0, "max": 9999.0, "step": 0.1, }),
                }
        }

    RETURN_TYPES = ("RUN_CONTEXT",)
    RETURN_NAMES = ("context",)
    FUNCTION = "animate"
    CATEGORY = "Apt_Preset/tool"
    def animate(self, text, max_frames, pw_a=0, pw_b=0, pw_c=0, pw_d=0, pre_text='', app_text='',context=None):
        
        clip=context.get("clip")
        
        positive=None
        negative=None
        
        settings = ScheduleSettings(
            text_g=text,
            pre_text_G=pre_text,
            app_text_G=app_text,
            text_L=None,
            pre_text_L=None,
            app_text_L=None,
            max_frames=max_frames,
            current_frame=None,
            print_output=None,
            pw_a=pw_a,
            pw_b=pw_b,
            pw_c=pw_c,
            pw_d=pw_d,
            start_frame=0,
            end_frame=max_frames,
            width=None,
            height=None,
            crop_w=None,
            crop_h=None,
            target_width=None,
            target_height=None,
        )
        
        positive, negative = batch_prompt_schedule(settings, clip)
        context = new_context(context, positive=positive, negative=negative,)  
        
        return (context,)


class AD_sch_mask:
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "points_string": ("STRING", {"default": "0:(0.0),\n7:(1.0),\n15:(0.0)\n", "multiline": True}),
                "invert": ("BOOLEAN", {"default": False}),
                "frames": ("INT", {"default": 16,"min": 2, "max": 255, "step": 1}),
                "width": ("INT", {"default": 512,"min": 1, "max": 4096, "step": 1}),
                "height": ("INT", {"default": 512,"min": 1, "max": 4096, "step": 1}),
                "interpolation": (["linear", "ease_in", "ease_out", "ease_in_out"],),
        },
    } 
    
    
    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "createfademask"
    CATEGORY = "Apt_Preset/AD"


    def createfademask(self, frames, width, height, invert, points_string, interpolation):
        
        def ease_in(t):
            return t * t
        
        def ease_out(t):
            return 1 - (1 - t) * (1 - t)

        def ease_in_out(t):
            return 3 * t * t - 2 * t * t * t
        
        # Parse the input string into a list of tuples
        points = []
        points_string = points_string.rstrip(',\n')
        for point_str in points_string.split(','):
            frame_str, color_str = point_str.split(':')
            frame = int(frame_str.strip())
            color = float(color_str.strip()[1:-1])  # Remove parentheses around color
            points.append((frame, color))

        # Check if the last frame is already in the points
        if len(points) == 0 or points[-1][0] != frames - 1:
            # If not, add it with the color of the last specified frame
            points.append((frames - 1, points[-1][1] if points else 0))

        # Sort the points by frame number
        points.sort(key=lambda x: x[0])

        batch_size = frames
        out = []
        image_batch = np.zeros((batch_size, height, width), dtype=np.float32)

        # Index of the next point to interpolate towards
        next_point = 1

        for i in range(batch_size):
            while next_point < len(points) and i > points[next_point][0]:
                next_point += 1

            # Interpolate between the previous point and the next point
            prev_point = next_point - 1
            t = (i - points[prev_point][0]) / (points[next_point][0] - points[prev_point][0])
            if interpolation == "ease_in":
                t = ease_in(t)
            elif interpolation == "ease_out":
                t = ease_out(t)
            elif interpolation == "ease_in_out":
                t = ease_in_out(t)
            elif interpolation == "linear":
                pass  # No need to modify `t` for linear interpolation

            color = points[prev_point][1] - t * (points[prev_point][1] - points[next_point][1])
            color = np.clip(color, 0, 255)
            image = np.full((height, width), color, dtype=np.float32)
            image_batch[i] = image

        output = torch.from_numpy(image_batch)
        mask = output
        out.append(mask)

        if invert:
            return (1.0 - torch.cat(out, dim=0),)
        return (torch.cat(out, dim=0),)

