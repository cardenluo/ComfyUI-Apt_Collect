

#region-----------------------ADdiff

from typing import Dict, List
from comfy.model_patcher import ModelPatcher
from .AnimateDiffEvolved.ad_settings import AnimateDiffSettings
from .AnimateDiffEvolved.context import ContextOptionsGroup

from .AnimateDiffEvolved.utils_model import Folders, BetaSchedules, get_available_motion_models
from .AnimateDiffEvolved.utils_motion import ADKeyframeGroup
from .AnimateDiffEvolved.motion_lora import MotionLoraList
from .AnimateDiffEvolved.model_injection import (ModelPatcherHelper, InjectionParams, MotionModelGroup, get_mm_attachment, load_motion_module_gen1)
from .AnimateDiffEvolved.sampling import outer_sample_wrapper, sliding_calc_cond_batch
from .AnimateDiffEvolved.sample_settings import SampleSettings



#原始
class AD_AnimateDiff:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "model_name": (get_available_motion_models(),),
                "beta_schedule": (BetaSchedules.ALIAS_LIST, {"default": BetaSchedules.AUTOSELECT}),
            },
            "optional": {
                "context_options": ("CONTEXT_OPTIONS",),
                "motion_lora": ("MOTION_LORA",),
                #"ad_settings": ("AD_SETTINGS",),
                #"sample_settings": ("SAMPLE_SETTINGS",),
                "motion_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "step": 0.001}),
                "apply_v2_models_properly": ("BOOLEAN", {"default": True}),
                #"ad_keyframes": ("AD_KEYFRAMES",),
            }
        }
    
    DEPRECATED = True
    RETURN_TYPES = ("MODEL",)
    CATEGORY = "Animate Diff 🎭🅐🅓/① Gen1 nodes ①"
    FUNCTION = "load_mm_and_inject_params"

    def load_mm_and_inject_params(self,
        model: ModelPatcher,
        model_name: str, beta_schedule: str,# apply_mm_groupnorm_hack: bool,
        context_options: ContextOptionsGroup=None, motion_lora: MotionLoraList=None, ad_settings: AnimateDiffSettings=None, motion_model_settings: AnimateDiffSettings=None,
        sample_settings: SampleSettings=None, motion_scale: float=1.0, apply_v2_models_properly: bool=False, ad_keyframes: ADKeyframeGroup=None,
    ):
        if ad_settings is not None:
            motion_model_settings = ad_settings
        # load motion module
        motion_model = load_motion_module_gen1(model_name, model, motion_lora=motion_lora, motion_model_settings=motion_model_settings)
        # set injection params
        params = InjectionParams(
                unlimited_area_hack=False,
                apply_v2_properly=apply_v2_models_properly,
        )
        if context_options:
            params.set_context(context_options)
        if not motion_model_settings:
            motion_model_settings = AnimateDiffSettings()
        motion_model_settings.attn_scale = motion_scale
        params.set_motion_model_settings(motion_model_settings)

        attachment = get_mm_attachment(motion_model)
        if params.motion_model_settings.mask_attn_scale is not None:
            attachment.scale_multival = params.motion_model_settings.mask_attn_scale * params.motion_model_settings.attn_scale
        else:
            attachment.scale_multival = params.motion_model_settings.attn_scale

        attachment.keyframes = ad_keyframes.clone() if ad_keyframes else ADKeyframeGroup()

        model = model.clone()
        helper = ModelPatcherHelper(model)
        helper.set_all_properties(
            outer_sampler_wrapper=outer_sample_wrapper,
            calc_cond_batch_wrapper=sliding_calc_cond_batch,
            params=params,
            sample_settings=sample_settings,
            motion_models=MotionModelGroup(motion_model),
        )

        sample_settings = helper.get_sample_settings()

        if sample_settings.sigma_schedule is not None:
            model.add_object_patch("model_sampling", sample_settings.sigma_schedule.clone().model_sampling)
        else:

            if beta_schedule == BetaSchedules.AUTOSELECT and helper.get_motion_models():
                beta_schedule = helper.get_motion_models()[0].model.get_best_beta_schedule(log=True)
            new_model_sampling = BetaSchedules.to_model_sampling(beta_schedule, model)
            if new_model_sampling is not None:
                model.add_object_patch("model_sampling", new_model_sampling)
        del motion_model
        return (model,)


#简化
class AD_AD:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "model_name": (get_available_motion_models(),),
                "beta_schedule": (BetaSchedules.ALIAS_LIST, {"default": BetaSchedules.AUTOSELECT}),
            },
            "optional": {
                "context_options": ("CONTEXT_OPTIONS",),
                "motion_lora": ("MOTION_LORA",),
                "motion_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "step": 0.001}),
                "apply_v2_models_properly": ("BOOLEAN", {"default": True}),
            }
        }
    
    DEPRECATED = True
    RETURN_TYPES = ("MODEL",)
    CATEGORY = "Apt_Preset/AD"
    FUNCTION = "load_mm_and_inject_params"

    def load_mm_and_inject_params(self,
        model: ModelPatcher,
        model_name: str, 
        beta_schedule: str,
        context_options: ContextOptionsGroup=None, 
        motion_lora: MotionLoraList=None, 
        motion_scale: float=1.0,
        apply_v2_models_properly: bool=False,
    ):

        
        motion_model = load_motion_module_gen1(model_name, model, motion_lora=motion_lora)
        # set injection params
        params = InjectionParams(
                unlimited_area_hack=False,
                apply_v2_properly=apply_v2_models_properly,
        )
        if context_options:
            params.set_context(context_options)
        
        motion_model_settings = AnimateDiffSettings()
        motion_model_settings.attn_scale = motion_scale
        params.set_motion_model_settings(motion_model_settings)

        attachment = get_mm_attachment(motion_model)
        attachment.scale_multival = motion_model_settings.attn_scale

        model = model.clone()
        helper = ModelPatcherHelper(model)
        helper.set_all_properties(
            outer_sampler_wrapper=outer_sample_wrapper,
            calc_cond_batch_wrapper=sliding_calc_cond_batch,
            params=params,
            motion_models=MotionModelGroup(motion_model),
        )

        if beta_schedule == BetaSchedules.AUTOSELECT and helper.get_motion_models():
            beta_schedule = helper.get_motion_models()[0].model.get_best_beta_schedule(log=True)
        new_model_sampling = BetaSchedules.to_model_sampling(beta_schedule, model)
        if new_model_sampling is not None:
            model.add_object_patch("model_sampling", new_model_sampling)
        
        del motion_model
        return (model,)

#---------------------------------------------------------------

class stack_AD_diff:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_name": (get_available_motion_models(),),
                "beta_schedule": (BetaSchedules.ALIAS_LIST, {"default": BetaSchedules.AUTOSELECT}),
            },
            "optional": {
                "context_options": ("CONTEXT_OPTIONS",),
                "motion_lora": ("MOTION_LORA",),
                "motion_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "step": 0.001}),
                "apply_v2_models_properly": ("BOOLEAN", {"default": True}),
            }
        }
    
    RETURN_TYPES = ("AD_STACK",)
    RETURN_NAMES = ("ad_stack",)
    FUNCTION = "pack_ad_params"
    CATEGORY = "Apt_Preset/AD"

    def pack_ad_params(self,
        model_name: str, 
        beta_schedule: str,
        context_options: ContextOptionsGroup=None, 
        motion_lora: MotionLoraList=None, 
        motion_scale: float=1.0,
        apply_v2_models_properly: bool=False,
    ):

        # 初始化ipa_list
        ad_stack = []
        
        ipa_info = (
            model_name,
            beta_schedule,
            context_options,
            motion_lora,
            motion_scale,
            apply_v2_models_properly,
        )
        ad_stack.append(ipa_info)
        return (ad_stack,)


class Apply_AD_diff:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "ad_stack": ("AD_STACK",),
            }
        }
    
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "apply_ad_params"
    CATEGORY = "Apt_Preset/AD"

    def apply_ad_params(self, model: ModelPatcher, ad_stack):

        if ad_stack is not None:

            for ad_params in ad_stack:
                (
            model_name,
            beta_schedule,
            context_options,
            motion_lora,
            motion_scale,
            apply_v2_models_properly,
            ) = ad_params


        motion_model = load_motion_module_gen1(model_name, model, motion_lora=motion_lora)
        # set injection params
        params = InjectionParams(
                unlimited_area_hack=False,
                apply_v2_properly=apply_v2_models_properly,
        )
        if context_options:
            params.set_context(context_options)
        
        motion_model_settings = AnimateDiffSettings()
        motion_model_settings.attn_scale = motion_scale
        params.set_motion_model_settings(motion_model_settings)

        attachment = get_mm_attachment(motion_model)
        attachment.scale_multival = motion_model_settings.attn_scale

        model = model.clone()
        helper = ModelPatcherHelper(model)
        helper.set_all_properties(
            outer_sampler_wrapper=outer_sample_wrapper,
            calc_cond_batch_wrapper=sliding_calc_cond_batch,
            params=params,
            motion_models=MotionModelGroup(motion_model),
        )

        if beta_schedule == BetaSchedules.AUTOSELECT and helper.get_motion_models():
            beta_schedule = helper.get_motion_models()[0].model.get_best_beta_schedule(log=True)
        new_model_sampling = BetaSchedules.to_model_sampling(beta_schedule, model)
        if new_model_sampling is not None:
            model.add_object_patch("model_sampling", new_model_sampling)
        
        del motion_model
        return (model,)

#endregion-----------------------ADdiff-------------

