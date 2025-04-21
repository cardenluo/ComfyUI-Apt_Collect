
#region-------------------------原版收纳--------------------
from ..def_unit import *
from copy import deepcopy
import keyframed as kf
import logging
import numpy as np
import torch


logging.basicConfig(level=logging.DEBUG, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

CATEGORY="Keyframe" + "/schedule"



class KfKeyframedCondition:

    CATEGORY=CATEGORY
    FUNCTION = 'main'
    RETURN_TYPES = ("KEYFRAMED_CONDITION",)
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "conditioning": ("CONDITIONING", {}),
                "time": ("FLOAT", {"default": 0, "step": 1}), 
                "interpolation_method": (list(kf.interpolation.EASINGS.keys()), {"default":"linear"}),
            },
        }
    
    def main(self, conditioning, time, interpolation_method):


        cond_tensor, cond_dict = conditioning[0] 
        cond_tensor = cond_tensor.clone()
        kf_cond_t = kf.Keyframe(t=time, value=cond_tensor, interpolation_method=interpolation_method)

        cond_pooled = cond_dict.get("pooled_output")
        cond_dict = deepcopy(cond_dict)
        kf_cond_pooled = None
        if cond_pooled is not None:
            cond_pooled = cond_pooled.clone()
            kf_cond_pooled = kf.Keyframe(t=time, value=cond_pooled, interpolation_method=interpolation_method)
            cond_dict["pooled_output"] = cond_pooled
        
        return ({"kf_cond_t":kf_cond_t, "kf_cond_pooled":kf_cond_pooled, "cond_dict":cond_dict},)


def set_keyframed_condition(keyframed_condition, schedule=None):
    keyframed_condition = deepcopy(keyframed_condition)
    cond_dict = keyframed_condition.pop("cond_dict")
    #cond_dict = deepcopy(cond_dict)

    if schedule is None:
        # get a new copy of the tensor
        kf_cond_t = keyframed_condition["kf_cond_t"]
        #kf_cond_t.value = kf_cond_t.value.clone() # should be redundant with the deepcopy
        curve_tokenized = kf.Curve([kf_cond_t], label="kf_cond_t")
        curves = [curve_tokenized]
        if keyframed_condition["kf_cond_pooled"] is not None:
            kf_cond_pooled = keyframed_condition["kf_cond_pooled"]
            curve_pooled = kf.Curve([kf_cond_pooled], label="kf_cond_pooled")
            curves.append(curve_pooled)
        schedule = (kf.ParameterGroup(curves), cond_dict)
    else:
        schedule = deepcopy(schedule)
        schedule, old_cond_dict = schedule
        for k, v in keyframed_condition.items():
            if (v is not None):
                # for now, assume we already have a schedule for k.
                # Not sure how to handle new conditioning type appearing.
                schedule.parameters[k][v.t] = v
        old_cond_dict.update(cond_dict) # NB: mutating this is probably bad
        schedule = (schedule, old_cond_dict)
    return schedule





class KfSetKeyframe:
    CATEGORY=CATEGORY
    FUNCTION = 'main'
    RETURN_TYPES = ("SCHEDULE",)
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "keyframed_condition": ("KEYFRAMED_CONDITION", {}),
            },
            "optional": {
                "schedule": ("SCHEDULE", {}), 
            }
        }
    def main(self, keyframed_condition, schedule=None):
        schedule = set_keyframed_condition(keyframed_condition, schedule)
        return (schedule,)


def evaluate_schedule_at_time(schedule, time):
    schedule = deepcopy(schedule)
    schedule, cond_dict = schedule
    #cond_dict = deepcopy(cond_dict)
    values = schedule[time]
    cond_t = values.get("kf_cond_t")
    cond_pooled = values.get("kf_cond_pooled")
    if cond_pooled is not None:
        #cond_dict = deepcopy(cond_dict)
        cond_dict["pooled_output"] = cond_pooled #.clone()
    #return [(cond_t.clone(), cond_dict)]
    return [(cond_t, cond_dict)]


class KfGetScheduleConditionAtTime:
    CATEGORY=CATEGORY
    FUNCTION = 'main'
    RETURN_TYPES = ("CONDITIONING",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "schedule": ("SCHEDULE",{}),
                "time": ("FLOAT",{}),
            }
        }
    
    def main(self, schedule, time):
        lerped_cond = evaluate_schedule_at_time(schedule, time)
        return (lerped_cond,)


class KfGetScheduleConditionSlice:
    CATEGORY=CATEGORY
    FUNCTION = 'main'
    RETURN_TYPES = ("CONDITIONING",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "schedule": ("SCHEDULE",{}),
                "start": ("FLOAT",{"default":0}),
                #"stop": ("FLOAT",{"default":0}),
                "step": ("FLOAT",{"default":1}),
                "n": ("INT", {"default":24}),
                #"endpoint": ("BOOL", {"default":True})
            }
        }
    
    #def main(self, schedule, start, stop, n, endpoint):
    def main(self, schedule, start, step, n):
        stop = start+n*step
        times = np.linspace(start=start, stop=stop, num=n, endpoint=True)
        conds = [evaluate_schedule_at_time(schedule, time)[0] for time in times]
        lerped_tokenized = [c[0] for c in conds]
        lerped_pooled = [c[1]["pooled_output"] for c in conds]
        lerped_tokenized_t = torch.cat(lerped_tokenized, dim=0)
        out_dict = deepcopy(conds[0][1])
        if isinstance(lerped_pooled[0], torch.Tensor) and isinstance(lerped_pooled[-1], torch.Tensor):
            out_dict['pooled_output'] =  torch.cat(lerped_pooled, dim=0)
        return [[(lerped_tokenized_t, out_dict)]] #


class AD_Evaluate_Condi:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                
                "conditioning": ("CONDITIONING", {}),
                "interpolation_method": (list(kf.interpolation.EASINGS.keys()), {"default":"linear"}),
                "schedule": ("SCHEDULE",{}),
                "time": ("FLOAT", {"default": 0, "step": 1}), 
                "start": ("FLOAT",{"default":0}),
                "step": ("FLOAT",{"default":1}),
                "n": ("INT", {"default":60}),
                
                
            },

                "hidden": {
                
                    },
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("condition", )
    FUNCTION = "main"
    CATEGORY = "Apt_Collect/AD"

    def main(self,conditioning, interpolation_method, schedule, time,start, step, n):

        cond_tensor, cond_dict = conditioning[0] 
        cond_tensor = cond_tensor.clone()
        kf_cond_t = kf.Keyframe(t=time, value=cond_tensor, interpolation_method=interpolation_method)
        cond_pooled = cond_dict.get("pooled_output")
        cond_dict = deepcopy(cond_dict)
        kf_cond_pooled = None
        if cond_pooled is not None:
            cond_pooled = cond_pooled.clone()
            kf_cond_pooled = kf.Keyframe(t=time, value=cond_pooled, interpolation_method=interpolation_method)
            cond_dict["pooled_output"] = cond_pooled
        
        keyframed_condition= {"kf_cond_t":kf_cond_t, "kf_cond_pooled":kf_cond_pooled, "cond_dict":cond_dict}    #算出keyframed_condition
        
        
        schedule = set_keyframed_condition(keyframed_condition, schedule)    #算出schedule
        
        stop = start+n*step        #估值
        times = np.linspace(start=start, stop=stop, num=n, endpoint=True)
        conds = [evaluate_schedule_at_time(schedule, time)[0] for time in times]
        lerped_tokenized = [c[0] for c in conds]
        lerped_pooled = [c[1]["pooled_output"] for c in conds]
        lerped_tokenized_t = torch.cat(lerped_tokenized, dim=0)
        out_dict = deepcopy(conds[0][1])
        if isinstance(lerped_pooled[0], torch.Tensor) and isinstance(lerped_pooled[-1], torch.Tensor):
            out_dict['pooled_output'] =  torch.cat(lerped_pooled, dim=0)
        return [[(lerped_tokenized_t, out_dict)]] 


#endregion--------------------收纳---------------------



#-------------------------修改版本--------------

#region-------------------drawing schedule--------------------

from toolz.itertoolz import sliding_window

def schedule_to_weight_curves(schedule):
    schedule, _ = schedule
    schedule = schedule.parameters["kf_cond_t"]
    schedule = deepcopy(schedule)
    curves = []
    keyframes = list(schedule._data.values())
    
    if len(keyframes) == 1:
        keyframe = keyframes[0]
        curves = kf.ParameterGroup({keyframe.label: keyframe.Curve(1)})
        return curves
    
    for (frame_in, frame_curr, frame_out) in sliding_window(3, keyframes):
        frame_in.value, frame_curr.value, frame_out.value = 0, 1, 0
        c = kf.Curve({frame_in.t: frame_in, frame_curr.t: frame_curr, frame_out.t: frame_out}, 
                    label=frame_curr.label)
        c = deepcopy(c)
        curves.append(c)
    
    begin, end = keyframes[:2], keyframes[-2:]
    begin[0].value = 1
    begin[1].value = 0
    end[0].value = 0
    end[1].value = 1
    
    outv = [kf.Curve(begin, label=begin[0].label)]
    if len(keyframes) == 2:
        return kf.ParameterGroup({begin[0].label: outv[0]})
    
    outv += curves
    outv += [kf.Curve(end, label=end[1].label)]
    return kf.ParameterGroup({c.label: c for c in outv})


import matplotlib.pyplot as plt
import io
from PIL import Image
import torchvision.transforms as TT

def plot_curve(curve, n, show_legend, is_pgroup=False):
    fig, ax = plt.subplots()
    eps: float = 1e-9

    m = 3
    if n < m:
        n = curve.duration + 1
        n = max(m, n)
    
    xs_base = list(range(int(n))) + list(curve.keyframes)
    logger.debug(f"xs_base:{xs_base}")
    xs = set()
    for x in xs_base:
        xs.add(x)
        xs.add(x - eps)

    width, height = 12, 8  # inches
    plt.figure(figsize=(width, height))        

    xs = [x for x in list(set(xs)) if (x >= 0)]
    xs.sort()

    def draw_curve(curve):
        ys = [curve[x] for x in xs]
        line = plt.plot(xs, ys, label=curve.label)
        kfx = curve.keyframes
        kfy = [curve[x] for x in kfx]
        plt.scatter(kfx, kfy, color=line[0].get_color())

    if is_pgroup:
        for c in curve.parameters.values():
            draw_curve(c)
    else:
        draw_curve(curve)
    
    if show_legend:
        plt.legend()

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()  # no idea if this makes a difference
    buf.seek(0)

    pil_image = Image.open(buf).convert('RGB')
    img_tensor = TT.ToTensor()(pil_image)
    img_tensor = img_tensor.unsqueeze(0)
    img_tensor = img_tensor.permute([0, 2, 3, 1])
    return img_tensor


class AD_DrawSchedule:
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "main"
    CATEGORY = "Apt_Preset/AD"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "schedule": ("SCHEDULE", {"forceInput": True}),
                "n": ("INT", {"default": 64}),
                "show_legend": ("BOOLEAN", {"default": True}),
            }
        }

    def main(self, schedule, n, show_legend):
        curves = schedule_to_weight_curves(schedule)
        img_tensor = plot_curve(curves, n, show_legend, is_pgroup=True)
        return (img_tensor,)
#endregion----------------------drawing schedule--------------------


class AD_slice_Condi:

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("positive", )

    FUNCTION = "main"
    CATEGORY = "Apt_Preset/AD"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "keyframed_condition": ("KEYFRAMED_CONDITION", {}),
                "schedule": ("SCHEDULE",{}),
                "offset": ("INT",{"default":1}),
                #"step": ("FLOAT",{"default":1}),
                "total_flame": ("INT", {"default":30}),
            },
            "optional": {
                "schedule": ("SCHEDULE", {}), 
            }
        }
    

    def main(self, keyframed_condition, offset, schedule, total_flame):
        
        
        step=1
        n=total_flame
        
        schedule = set_keyframed_condition(keyframed_condition, schedule) #kfsetkeyframe
        stop = offset+n*step
        times = np.linspace(start=offset, stop=stop, num=n, endpoint=True)
        conds = [evaluate_schedule_at_time(schedule, time)[0] for time in times]
        lerped_tokenized = [c[0] for c in conds]
        lerped_pooled = [c[1]["pooled_output"] for c in conds]
        lerped_tokenized_t = torch.cat(lerped_tokenized, dim=0)
        out_dict = deepcopy(conds[0][1])
        if isinstance(lerped_pooled[0], torch.Tensor) and isinstance(lerped_pooled[-1], torch.Tensor):
            out_dict['pooled_output'] =  torch.cat(lerped_pooled, dim=0)
        
        positive= [[(lerped_tokenized_t, out_dict)]]
        
        
        return  positive






class AD_sch_prompt_adv(KfKeyframedCondition):

    CATEGORY = "Apt_Preset/AD"
    FUNCTION = 'main'
    RETURN_TYPES = ("KEYFRAMED_CONDITION", "SCHEDULE",)
    RETURN_NAMES = ("keyframed_condition", "schedule", )

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                
                "context": ("RUN_CONTEXT",),

                "text1": ("STRING", {"multiline": True, "default": "hill"}),
                "keyframe1": ("FLOAT", {"default": 1, "step": 1}),
                "interpolation_method1": (list(kf.interpolation.EASINGS.keys()), {"default": "linear"}),

                "text2": ("STRING", {"multiline": True, "default": "sea"}),
                "keyframe2": ("FLOAT", {"default": 30, "step": 1}),
                "interpolation_method2": (list(kf.interpolation.EASINGS.keys()), {"default": "linear"}),

                "text3": ("STRING", {"multiline": True, "default": ""}),
                "keyframe3": ("FLOAT", {"default": 0, "step": 1}),
                "interpolation_method3": (list(kf.interpolation.EASINGS.keys()), {"default": "linear"}),

                "text4": ("STRING", {"multiline": True, "default": ""}),
                "keyframe4": ("FLOAT", {"default": 0, "step": 1}),
                "interpolation_method4": (list(kf.interpolation.EASINGS.keys()), {"default": "linear"}),

                "text5": ("STRING", {"multiline": True, "default": ""}),
                "keyframe5": ("FLOAT", {"default": 0, "step": 1}),
                "interpolation_method5": (list(kf.interpolation.EASINGS.keys()), {"default": "linear"}),

                "text6": ("STRING", {"multiline": True, "default": ""}),
                "keyframe6": ("FLOAT", {"default": 0, "step": 1}),
                "interpolation_method6": (list(kf.interpolation.EASINGS.keys()), {"default": "linear"}),
            },
            "optional": {
                "schedule": ("SCHEDULE", {}),
            }
        }

    def main(self, context, text1, keyframe1, interpolation_method1, text2, keyframe2, interpolation_method2,
                text3, keyframe3, interpolation_method3, text4, keyframe4, interpolation_method4,
                text5, keyframe5, interpolation_method5, text6, keyframe6, interpolation_method6, schedule=None):
        
        
        clip= context.get("clip")
        
        # 处理第一个关键帧
        tokens1 = clip.tokenize(text1)
        cond1, pooled1 = clip.encode_from_tokens(tokens1, return_pooled=True)
        conditioning = [[cond1, {"pooled_output": pooled1}]]
        keyframed_condition = super().main(conditioning, keyframe1, interpolation_method1)[0]  
        keyframed_condition["kf_cond_t"].label = text1
        schedule = set_keyframed_condition(keyframed_condition, schedule)

        # 处理第二个关键帧
        tokens2 = clip.tokenize(text2)
        cond2, pooled2 = clip.encode_from_tokens(tokens2, return_pooled=True)
        conditioning = [[cond2, {"pooled_output": pooled2}]]
        keyframed_condition = super().main(conditioning, keyframe2, interpolation_method2)[0]  
        keyframed_condition["kf_cond_t"].label = text2
        schedule = set_keyframed_condition(keyframed_condition, schedule)

        # 处理第三个关键帧
        tokens3 = clip.tokenize(text3)
        cond3, pooled3 = clip.encode_from_tokens(tokens3, return_pooled=True)
        conditioning = [[cond3, {"pooled_output": pooled3}]]
        keyframed_condition = super().main(conditioning, keyframe3, interpolation_method3)[0]  
        keyframed_condition["kf_cond_t"].label = text3
        schedule = set_keyframed_condition(keyframed_condition, schedule)

        # 处理第四个关键帧
        tokens4 = clip.tokenize(text4)
        cond4, pooled4 = clip.encode_from_tokens(tokens4, return_pooled=True)
        conditioning = [[cond4, {"pooled_output": pooled4}]]
        keyframed_condition = super().main(conditioning, keyframe4, interpolation_method4)[0]  
        keyframed_condition["kf_cond_t"].label = text4
        schedule = set_keyframed_condition(keyframed_condition, schedule)

        # 处理第五个关键帧
        tokens5 = clip.tokenize(text5)
        cond5, pooled5 = clip.encode_from_tokens(tokens5, return_pooled=True)
        conditioning = [[cond5, {"pooled_output": pooled5}]]
        keyframed_condition = super().main(conditioning, keyframe5, interpolation_method5)[0]  
        keyframed_condition["kf_cond_t"].label = text5
        schedule = set_keyframed_condition(keyframed_condition, schedule)

        # 处理第六个关键帧
        tokens6 = clip.tokenize(text6)
        cond6, pooled6 = clip.encode_from_tokens(tokens6, return_pooled=True)
        conditioning = [[cond6, {"pooled_output": pooled6}]]
        keyframed_condition = super().main(conditioning, keyframe6, interpolation_method6)[0]  
        keyframed_condition["kf_cond_t"].label = text6
        schedule = set_keyframed_condition(keyframed_condition, schedule)
        
        return (keyframed_condition, schedule, conditioning)


