WEB_DIRECTORY = "./web"
#__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']


from .nodes import *
from .stack import *
from .py.packdata import *
from .py.AdvancedCN import AD_sch_adv_CN,AD_latent_keyframe,AD_latent_kfGroup,AD_sch_latent
from .py.IPAdapterPlus import chx_IPA_basic,chx_IPA_faceID,chx_IPA_XL_adv,chx_IPA_region_combine, chx_IPA_apply_combine
from .py.ScheduledNodes import AD_sch_Value,chx_prompt_Schedule,AD_sch_mask
from .py.Keyframe_schedule import AD_slice_Condi,AD_sch_prompt_adv,AD_DrawSchedule
from .py.InpaintNodes import pre_inpaint
from .py.Ksampler_VisualStyle import chx_Ksampler_VisualStyle
from .py.Ksampler_dual_area import chx_Ksampler_dual_area
from .py.IPAdapterSD3 import IPA_dapterSD3LOAD,Stack_IPA_SD3,Apply_IPA_SD3
from .py.layered_infinite_zoom import AD_InfiniteZoom
from .py.video_node import *


NODE_CLASS_MAPPINGS = {

#-sum-------------------------------------------#
"sum_load": sum_load,
"sum_editor": sum_editor,
"sum_latent": sum_latent,
"sum_lora": sum_lora,
"sum_controlnet": sum_controlnet,
"sum_stack_AD": sum_stack_AD,
"sum_stack_image": sum_stack_image,
"sum_stack_all": sum_stack_all,
"sum_stack_Wan": sum_stack_Wan,
"sum_text": sum_text,   


"load_FLUX": load_FLUX,
"load_basic": load_basic,
"load_SD35": load_SD35,


"Data_chx_Merge":Data_chx_Merge,
"Data_chx_MergeBig":Data_chx_MergeBig,
"Data_presetData":Data_presetData,
"Data_basic": Data_basic,
"Data_basic_easy": Data_basic_easy,
"Data_sample": Data_sample,
"Data_select": Data_select,
"Data_preset_save": Data_preset_save,



#--------AD------------------------------------------
"AD_sch_prompt": AD_sch_prompt,
"AD_sch_prompt_adv": AD_sch_prompt_adv,


"AD_sch_mask":AD_sch_mask,
"AD_sch_Value": AD_sch_Value,
"AD_sch_latent": AD_sch_latent,
"AD_sch_IPA": AD_sch_IPA,

"AD_sch_adv_CN":AD_sch_adv_CN,
"AD_latent_keyframe": AD_latent_keyframe,
"AD_latent_kfGroup": AD_latent_kfGroup,
"AD_DrawSchedule": AD_DrawSchedule,
"AD_slice_Condi": AD_slice_Condi,
"AD_InfiniteZoom": AD_InfiniteZoom,


#-sample-------------------------------------------#
"pre_sample_data":  pre_sample_data,  
"pre_make_context": pre_make_context,
"pre_inpaint": pre_inpaint,

"basic_Ksampler_simple": basic_Ksampler_simple,  
"basic_Ksampler_mid": basic_Ksampler_mid,    
"basic_Ksampler_full": basic_Ksampler_full,     
"basic_Ksampler_custom": basic_Ksampler_custom,

"chx_Ksampler_VisualStyle": chx_Ksampler_VisualStyle,  
"chx_Ksampler_adv": chx_Ksampler_adv,
"chx_Ksampler_mix": chx_Ksampler_mix,
"chx_Ksampler_texture": chx_Ksampler_texture,
"chx_ksampler_Deforum": chx_ksampler_Deforum,
"chx_Ksampler_dual_area": chx_Ksampler_dual_area,
"chx_Ksampler_refine": chx_Ksampler_refine,
"chx_ksampler_tile": chx_ksampler_tile,   
"chx_Ksampler_inpaint": chx_Ksampler_inpaint,


#-stack-------------------------------------------#
"Apply_ControlNetStack": Apply_ControlNetStack,
"Apply_LoRAStack": Apply_LoRAStack,
"Apply_IPA": Apply_IPA,
"Apply_prompt_Schedule": Apply_prompt_Schedule,
"Apply_adv_CN": Apply_adv_CN,
"Apply_AD_diff": Apply_AD_diff,
"Apply_condiStack": Apply_condiStack,
"Apply_textStack": Apply_textStack,
"Apply_Redux": Apply_Redux,
"Apply_latent": Apply_latent,
"Apply_IPA_SD3":Apply_IPA_SD3,
"Apply_CN_union":Apply_CN_union,

"Stack_latent": Stack_latent,
"Stack_IPA_SD3":Stack_IPA_SD3,
"Stack_Redux":Stack_Redux,
"stack_AD_diff": stack_AD_diff,
"Stack_adv_CN_easy": Stack_adv_CN_easy,
"Stack_adv_CN": Stack_adv_CN,
"Stack_IPA": Stack_IPA,
"Stack_text": Stack_text,
"Stack_condi": Stack_condi,
"Stack_LoRA": Stack_LoRA,
"Stack_ControlNet": Stack_ControlNet,
"Stack_ControlNet1":Stack_ControlNet1,
"Stack_CN_union":Stack_CN_union,


"Stack_WanImageToVideo": Stack_WanImageToVideo,
"Stack_WanFunControlToVideo": Stack_WanFunControlToVideo,
"Stack_WanFirstLastFrameToVideo": Stack_WanFirstLastFrameToVideo,
"Stack_WanFunInpaintToVideo": Stack_WanFunInpaintToVideo,




#-IPA-------------------------------------------#
"chx_IPA_basic": chx_IPA_basic,
"chx_IPA_faceID": chx_IPA_faceID,
"chx_IPA_XL_adv": chx_IPA_XL_adv,
"chx_IPA_region_combine": chx_IPA_region_combine,
"chx_IPA_apply_combine": chx_IPA_apply_combine,
"IPA_dapterSD3LOAD":IPA_dapterSD3LOAD,
"chx_YC_LG_Redux": chx_YC_LG_Redux,
"chx_Style_Redux": chx_Style_Redux,
"chx_StyleModelApply":chx_StyleModelApply,


#---------tool------------------------------------------#

"chx_re_fluxguide": chx_re_fluxguide,
"chx_vae_encode": chx_vae_encode,
"chx_controlnet": chx_controlnet, 
"chx_controlnet_union": chx_controlnet_union,
"chx_mask_Mulcondi": mask_Mulcondition,
"chx_Upscale_simple": chx_Upscale_simple,
"chx_prompt_Schedule": chx_prompt_Schedule,


#--unpack------------------------------------------#
"param_preset_pack": param_preset,
"param_preset_Unpack": Unpack_param,
"Model_Preset_pack":Model_Preset,
"Model_Preset_Unpack": Unpack_Model,

"CN_preset1_pack": CN_preset1,
"CN_preset1_Unpack": Unpack_CN,
"photoshop_preset_pack":photoshop_preset,
"photoshop_preset_Unpack":Unpack_photoshop,

"stack_sum_pack": stack_sum_pack,










}








NODE_DISPLAY_NAME_MAPPINGS = {

"pre_inpaint":" pre_inpaint(xl)",
"chx_ksampler_tile": "chx_ksampler_tile(sole)",   
"chx_Ksampler_inpaint": "chx_Ksampler_inpaint(sole)",

"chx_IPA_basic": "IPA_basic",
"chx_IPA_faceID": "IPA_faceID",
"chx_IPA_XL_adv": "IPA_XL_adv",


}


