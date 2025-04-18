
#import numpy as np

import json
import folder_paths
import os
import torch





#region-----------latent_Selector-----

def read_ratios():

    current_dir = os.path.dirname(os.path.abspath(__file__))
    p = os.path.dirname(current_dir)
    file_path = os.path.join(p, 'web', 'ratios.json')
    with open(file_path, 'r') as file:
        data = json.load(file)
    ratio_sizes = list(data['ratios'].keys())
    ratio_dict = data['ratios']
    return ratio_sizes, ratio_dict


class latent_ratio:
    
    @classmethod
    def INPUT_TYPES(s):
        s.ratio_sizes, s.ratio_dict = read_ratios()
        return {'required': {'ratio_selected': (s.ratio_sizes,),
                            "batch_size": ("INT", {"default": 1, "min": 1, "max": 64})}}

    RETURN_TYPES = ('LATENT',)
    FUNCTION = 'generate'
    CATEGORY = "Apt_Collect/latent"

    def generate(self, ratio_selected, batch_size=1):
        width = self.ratio_dict[ratio_selected]["width"]
        height = self.ratio_dict[ratio_selected]["height"]
        latent = torch.zeros([batch_size, 4, height // 8, width // 8])
        return ({"samples": latent}, )

#endregion-------------------------------



