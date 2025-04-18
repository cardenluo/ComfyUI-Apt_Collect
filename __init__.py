
WEB_DIRECTORY = "./web"





from .Finenode.CSV_loader import *
from .Finenode.C_math import *
from .Finenode.C_model import *
from .Finenode.C_mask import *
from .Finenode.C_latent import *
from .Finenode.C_viewIO import *
from .Finenode.C_AD import *
from .Finenode.C_sampler import *
from .Finenode.C_image import *
from .Finenode.C_promp import *
from .Finenode.C_utils import *
from .Finenode.C_imgEffect import *
from .Finenode.C_color import *
from .Finenode.C_type import *
from .Finenode.C_test import *
from .Finenode.C_layout import *
from .Finenode.C_GPT import *


from .web_node.edit_mask import *
from .web_node.inpaint_cropandstitch import *
from .web_node.info import *
from .web_node.latent_Selector import *
from .web_node.image_color_analyzer import *
from .web_node.TranslationNode import *
from .web_node.stack_Wildcards import *





#--------------------------------------------------------------------------------------------


NODE_CLASS_MAPPINGS = {  


#-------------view-IO-------------------
"IO_input": C_input,  
"IO_inputbasic": C_inputbasic,
"IO_Textlist": Output_Textlist,
"IO_Valuelist": Output_Valuelist,
"IO_load_anyimage":C_load_anyimage,
"IO_Free_Memory": IO_Free_Memory,


"view_Data": view_Data,  #wed
"view_bridge_image": CEditMask,  #wed
"view_bridge_Text": view_bridge_Text, #wed
"view_mask": CMaskPreview,
"view_latent": PreviewLatentAdvanced,
"view_combo": view_combo,#wed
"view_node_Script": view_node_Script,
"view_GetLength": GetLength, #wed----utils
"view_GetShape": GetShape, #wed----utils



#-------------utils-------------------
"pack_Pack": Pack, #wed
"pack_Unpack": Unpack, #wed

"list_ListGetByIndex": ListGetByIndex,
"list_ListSlice": ListSlice,
"list_ListToBatch": ListToBatch, #wed
"list_CreateList": CreateList,#wed
"list_MergeList": MergeList, #wed   
"batch_BatchGetByIndex": BatchGetByIndex,
"batch_BatchSlice": BatchSlice,
"batch_BatchToList": BatchToList,
"batch_CreateBatch": CreateBatch,  #wed
"batch_MergeBatch": MergeBatch, #wed
"batch_MakeImageBatch": MakeImageBatch, 
"batch_MakeMaskBatch": MakeMaskBatch, 

"text_SplitString": SplitString,

"type_AnyCast": AnyCast, #wed
"type_BatchItemCast": BatchItemCast, 
"view_GetWidgetsValues": GetWidgetsValues, #wed----utils
"math_Exec": Exec,#wed----utils

"type_make_maskBatch": type_make_maskBatch,
"type_make_condition": type_make_condition,
"type_Anyswitch": type_Anyswitch,
"type_BasiPIPE": type_BasiPIPE,
"type_Image_List2Batch":type_Image_List2Batch,
"type_Image_Batch2List":type_Image_Batch2List,
"type_Mask_Batch2List":type_Mask_Batch2List,
"type_Mask_List2Batch":type_Mask_List2Batch,
"type_text_list2batch ": type_text_list2batch ,  
"type_text_2_UTF8": type_text_2_UTF8 ,  





#---------math------------------
"math_Float_Op": CFloatUnaryOperation,  
"math_Float_Condi": CFloatUnaryCondition,  
"math_Float_Binary_Op": CFloatBinaryOperation,  
"math_Float_Binary_Condin": CFloatBinaryCondition,  
"math_Int_Unary_Op": CIntUnaryOperation,
"math_Int_Unary_Condi": CIntUnaryCondition,
"math_Int_Binary_Op": CIntBinaryOperation,
"math_Int_Binary_Condi": CIntBinaryCondition,
"math_Remap_Data": C_Remap_DataRange,  
"math_CreateRange": CreateRange,
"math_CreateArange": CreateArange,
"math_CreateLinspace": CreateLinspace,

"Batch_Gradient Float":Gradient_Float,
"Batch_Gradient Integer":Gradient_Integer,
"Batch_Increment Float":Increment_Float,        
"Batch_Increment Integer":Increment_Integer,



#---------------model--------
"model_adjust_color": Model_adjust_color,
"model_diff_inpaint": model_diff_inpaint,




#----------------image------------------------
"pad_uv_fill": pad_uv_fill,
"pad_color_fill": pad_color_fill,
"Image_LightShape": Image_LightShape,    
"Image_Normal_light": Image_Normal_light,
"Image_keep_OneColorr": Image_keep_OneColorr,  

"Image_transform": Image_transform,    
"Image_cutResize": Image_cutResize,
"Image_Adjust": Image_Adjust,
"image_sumTransform": image_sumTransform,
"Image_overlay": Image_overlay,
"Image_overlay_mask": Image_overlay_mask,
"Image_overlay_composite": Image_overlay_composite,
"Image_overlay_transform": Image_overlay_transform,
"Image_overlay_sum": Image_overlay_sum,

"Image_Extract_Channel": Image_Extract_Channel,
"Image_Apply_Channel": Image_Apply_Channel,

"Image_RemoveAlpha": Image_RemoveAlpha,
"image_selct_batch": image_selct_batch,
"Image_scale_match": Image_scale_match,





#-----------------mask----------------------
"Mask_inpaint_Grey": Mask_inpaint_Grey,
"Mask_lightSource": Mask_lightSource,
"Mask_math": Mask_math,
"Mask_Detect_label": Mask_Detect_label,
"Mask_mulcolor_img": Mask_mulcolor_img,   
"Mask_mulcolor_mask": Mask_mulcolor_mask,   


"Mask_Outline": Mask_Outline,
"Mask_Remap": Mask_Remap,
"Mask_Smooth": Mask_Smooth,
"Mask_Offset": Mask_Offset,
"Mask_cut_mask": Mask_cut_mask,
"Mask_image2mask": Mask_image2mask,
"Mask_mask2mask": Mask_mask2mask,
"Mask_mask2img": Mask_mask2img,
"Mask_splitMask": Mask_splitMask,



#------------latent---------------------
"latent_chx_noise": latent_chx_noise,
"latent_Image2Noise": latent_Image2Noise,
"latent_ratio": latent_ratio,
"latent_mask":latent_mask,


#----------prompt----------------

"text_mul_replace": text_mul_replace,
"text_mul_remove": text_mul_remove,
"text_free_wildcards": text_free_wildcards,
"text_SuperPrompter": text_SuperPrompter,
"text_selectOutput": text_selectOutput,
"text_CSV_load": text_CSV_load,


"stack_Wildcards": stack_Wildcards,
"stack_text_combine": stack_text_combine,



#---------Gpt modle---------------
"ChineseToEnglish": ChineseToEnglish,
"EnglishToChinese": EnglishToChinese,

"deepseek_api_text": deepseek_api_text,
"Janus_img_2_text": Janus_img_2_text,
"Janus_generate_img": Janus_generate_img,



#-------sample----------------------------------------
"sampler_InpaintCrop": InpaintCrop,  #wed
"sampler_InpaintStitch": InpaintStitch,  #wed

"sampler_DynamicTileSplit": DynamicTileSplit, 
"sampler_DynamicTileMerge": DynamicTileMerge,

"sampler_enhance": sampler_enhance,
"sampler_sigmas": sampler_sigmas,



#---------AD--------------------
"AD_batch_prompt": AD_batch_prompt,
"AD_MaskExpandBatch": AD_MaskExpandBatch, 
"AD_ImageExpandBatch": AD_ImageExpandBatch,
"AD_Dynamic_MASK": AD_Dynamic_MASK,

"AD_Mask_generate": Mask_AD_generate,

#-----------------Color----------------

"color_Analyzer_Image": Image_Color_Analyzer,
"color_adjust": color_adjust,
"color_Match": color_Match,
"color_transfer":color_transfer,

"color_input": color_input,
"color_color2hex":color_color2hex,
"color_hex2color":color_hex2color,
"color_image_Replace": ImageReplaceColor,

"color_pure_img": color_pure_img,
"color_Gradient": color_Gradient,
"color_RadialGradient": color_RadialGradient,



#----------------imgEffect--------
"img_Loadeffect": img_Loadeffect,
"img_Upscaletile": img_Upscaletile,
"img_Remove_bg": img_Remove_bg,
"img_CircleWarp": img_CircleWarp,
"img_Stretch": img_Stretch,
"img_WaveWarp": img_WaveWarp,
"img_Liquify": img_Liquify,
"img_Seam_adjust_size": img_Seam_adjust_size,
"img_texture_Offset": img_texture_Offset,
"img_White_balance":img_White_balance,  #白平衡重定向
"img_HDR": img_HDR,



#-----------------layout----------------
"lay_ImageGrid": lay_ImageGrid,
"lay_MaskGrid": lay_MaskGrid,
"lay_text":lay_text,
"lay_edge_cut": lay_edge_cut,   
"lay_image_match_W_or_H": lay_image_match_W_or_H,
"lay_image_match_W_and_H": lay_image_match_W_and_H,



}



NODE_DISPLAY_NAME_MAPPINGS = {    }


