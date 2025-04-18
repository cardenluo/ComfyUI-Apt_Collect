
from nodes import MAX_RESOLUTION, SaveImage, common_ksampler
import torch
import os
import sys
import comfy.controlnet
import comfy.sd
import folder_paths
import random
from PIL import Image, ImageOps
from PIL.PngImagePlugin import PngInfo
from comfy import latent_formats
import json
import latent_preview
import comfy.utils
from comfy.cli_args import args
import numpy as np
import comfy.controlnet
import inspect
import logging
import traceback





sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "comfy"))


#-------------------------------------------------------------------------------------------#

class AnyType(str):
    def __eq__(self, _) -> bool:
        return True
    def __ne__(self, __value: object) -> bool:
        return False
ANY_TYPE = AnyType("*")


#-------------------------------------------------------------------------------------------#

class PreviewLatentAdvanced:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required":
                    {"latent": ("LATENT",),
                    "base_model": (["SD15","SDXL"],),
                    "preview_method": (["auto","taesd","latent2rgb"],),
                    },
            "hidden": {"prompt": "PROMPT",
                        "extra_pnginfo": "EXTRA_PNGINFO",
                        "my_unique_id": "UNIQUE_ID",},
                }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    OUTPUT_NODE = True
    FUNCTION = "lpreview"
    CATEGORY = "Apt_Collect/view_IO"

    def lpreview(self, latent, base_model, preview_method, prompt=None, extra_pnginfo=None, my_unique_id=None):
        previous_preview_method = args.preview_method
        if preview_method == "taesd":
            temp_previewer = latent_preview.LatentPreviewMethod.TAESD
        elif preview_method == "latent2rgb":
            temp_previewer = latent_preview.LatentPreviewMethod.Latent2RGB
        else:
            temp_previewer = latent_preview.LatentPreviewMethod.Auto

        results = list()

        try:
            args.preview_method=temp_previewer
            preview_format = "PNG"
            load_device=comfy.model_management.vae_offload_device()
            latent_format = {"SD15":latent_formats.SD15,
                            "SDXL":latent_formats.SDXL}[base_model]()

            result=[]
            for i in range(len(latent["samples"])):
                x=latent.copy()
                x["samples"] = latent["samples"][i:i+1].clone()
                x_sample = x["samples"]
                x_sample = x_sample /  {"SD15":6,"SDXL":7.5}[base_model]

                img = latent_preview.get_previewer(load_device, latent_format).decode_latent_to_preview(x_sample)
                full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path("",folder_paths.get_temp_directory(), img.height, img.width)
                metadata = None
                if not args.disable_metadata:
                    metadata = PngInfo()
                    if prompt is not None:
                        metadata.add_text("prompt", json.dumps(prompt))
                    if extra_pnginfo is not None:
                        for x in extra_pnginfo:
                            metadata.add_text(x, json.dumps(extra_pnginfo[x]))

                file = "latent_"+"".join(random.choice("0123456789") for x in range(8))+".png"
                img.save(os.path.join(full_output_folder, file), pnginfo=metadata, compress_level=4)
                results.append({"filename": file, "subfolder": subfolder, "type": "temp"})

        finally:
            # Restore global changes
            args.preview_method=previous_preview_method

        return {"result": (latent,), "ui": { "images": results } }


class PreviewLatent(PreviewLatentAdvanced):
    @classmethod
    def INPUT_TYPES(cls):
        return {"required":
                    {"latent": ("LATENT",),
                    },
                "hidden": {"prompt": "PROMPT",
                        "extra_pnginfo": "EXTRA_PNGINFO",
                        "my_unique_id": "UNIQUE_ID",},
                }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    OUTPUT_NODE = True
    FUNCTION = "lpreview_basic"
    CATEGORY = "Apt_Collect/view_IO"

    def lpreview_basic(self, latent, prompt=None, extra_pnginfo=None, my_unique_id=None):
        return PreviewLatentAdvanced().lpreview(latent=latent, base_model="SD15", preview_method="auto", prompt=prompt, extra_pnginfo=extra_pnginfo, my_unique_id=my_unique_id)


class CMaskPreview(SaveImage):
    
    def __init__(self):
        pass

    def __init__(self):
        self.output_dir = folder_paths.get_temp_directory()
        self.type = "temp"
        self.prefix_append = "_temp_" + ''.join(random.choice("abcdefghijklmnopqrstupvxyz") for x in range(5))
        self.compress_level = 4

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {"mask": ("MASK",), },  
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    FUNCTION = "execute"
    CATEGORY = "Apt_Collect/view_IO"

    def execute(self, mask, filename_prefix="ComfyUI", prompt=None, extra_pnginfo=None):
        preview = mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])).movedim(1, -1).expand(-1, -1, -1, 3)
        return self.save_images(preview, filename_prefix, prompt, extra_pnginfo)


class C_load_anyimage:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "file_path": ("STRING", {}),
                "fill_color": (["None", "white", "gray", "black"], {}),
                "smooth": ("BOOLEAN", {"default": True})
            },
            "optional": {
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff})
            }
        }

        
        
        
        
    RETURN_TYPES = ('IMAGE', 'MASK',)
    FUNCTION = "get_transparent_image"
    CATEGORY = "Apt_Collect/view_IO"
    
    def get_transparent_image(self, file_path, smooth, seed, fill_color):
        try:
            if os.path.isdir(file_path):
                images = []
                for filename in os.listdir(file_path):
                    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                        img_path = os.path.join(file_path, filename)
                        image = Image.open(img_path).convert('RGBA')
                        images.append(image)
                
                if not images:
                    return None, None
                
                target_size = images[0].size
                
                resized_images = []
                for image in images:
                    if image.size != target_size:
                        image = image.resize(target_size, Image.BILINEAR)
                    resized_images.append(image)
                
                batch_images = np.stack([np.array(img) for img in resized_images], axis=0).astype(np.float32) / 255.0
                batch_tensor = torch.from_numpy(batch_images)
                
                mask_tensor = None
                
                return batch_tensor, mask_tensor        
            else:
                file_path = file_path.strip('"')
                image = Image.open(file_path)
                if image is not None:
                    image_rgba = image.convert('RGBA')
                    image_rgba.save(file_path.rsplit('.', 1)[0] + '.png')
            
                    mask = np.array(image_rgba.getchannel('A')).astype(np.float32) / 255.0
                    if smooth:
                        mask = 1.0 - mask
                    mask_tensor = torch.from_numpy(mask)[None, None, :, :]
            
                    if fill_color == 'white':
                        for y in range(image_rgba.height):
                            for x in range(image_rgba.width):
                                if image_rgba.getpixel((x, y))[3] == 0:
                                    image_rgba.putpixel((x, y), (255, 255, 255, 255))
                    elif fill_color == 'gray':
                        for y in range(image_rgba.height):
                            for x in range(image_rgba.width):
                                if image_rgba.getpixel((x, y))[3] == 0:
                                    image_rgba.putpixel((x, y), (128, 128, 128))
                    elif fill_color == 'black':
                        for y in range(image_rgba.height):
                            for x in range(image_rgba.width):
                                if image_rgba.getpixel((x, y))[3] == 0:
                                    image_rgba.putpixel((x, y), (0, 0, 0))
                    elif fill_color == 'None':
                        pass
                    else:
                        raise ValueError("Invalid fill color specified.")
            
                    image_np = np.array(image_rgba).astype(np.float32) / 255.0
                    image_tensor = torch.from_numpy(image_np)[None, :, :, :]
            
                    return (image_tensor, mask_tensor)
            
        except Exception as e:
            print(f"出错请重置节点：{e}")
        return None, None


class C_inputbasic:
    def __init__(self):
        pass
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input": ("STRING", {
                    "multiline": True,
                    "default": ""
                }),
            }
        }
    RETURN_TYPES = ("INT", "FLOAT", "STRING")
    RETURN_NAMES = ("INT", "FLOAT", "STRING")
    FUNCTION = "convert_number_types"
    CATEGORY = "Apt_Collect/view_IO"
    def convert_number_types(self, input):
        try:
            float_num = float(input)
            int_num = int(float_num)
            str_num = input
        except ValueError:
            return (None, None, input)
        return (int_num, float_num, str_num)


class C_input:
    def __init__(self):
        pass
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # 新增布尔类型
                "data_type": (["float","int","text","Valuelist","Textlist", "bool"], ),
                "input": ("STRING", {"multiline": True,"default": ""}),
            }
        }
    
    RETURN_TYPES = (ANY_TYPE,)
    RETURN_NAMES = ("data",)
    FUNCTION = "execute"
    CATEGORY = "Apt_Collect/view_IO"

    # 根据不同的输入类型返回不同的输出类型
    def get_return_types(self, data_type):
        if data_type == "float":
            return ("FLOAT",)
        elif data_type == "int":
            return ("INT",)
        elif data_type == "text":
            return ("STRING",)
        elif data_type == "Valuelist":
            return ("FLOAT",)
        elif data_type == "Textlist":
            return ("STRING",)
        # 新增布尔类型判断
        elif data_type == "bool":
            return ("BOOLEAN",)
        return (ANY_TYPE,)

    def convert_types(self, data_type, input):
        if data_type == "float":
            try:
                return (float(input),)
            except ValueError:
                return (None,)
        elif data_type == "int":
            try:
                return (int(input),)
            except ValueError:
                return (None,)
        elif data_type == "text":
            return (input,)
        elif data_type == "Valuelist":
            values = []
            for line in input.split('\n'):
                try:
                    values.append(float(line))
                except ValueError:
                    continue
            return (values,)
        elif data_type == "Textlist":
            return (input.split('\n'),)
        # 新增布尔类型转换
        elif data_type == "bool":
            lower_input = input.lower()
            if lower_input in ['true', '1', 'yes']:
                return (True,)
            elif lower_input in ['false', '0', 'no']:
                return (False,)
            return (None,)
        return (None,)

    def execute(self, data_type, input):
        RETURN_TYPES = self.get_return_types(data_type)
        result = self.convert_types(data_type, input)
        return result


class Output_Textlist:
    
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "text": ("STRING", {"multiline": True, "default": ""}),
            "repeats": ("INT", {"default": 1, "min": 1, "max": 99999}),
            "loops": ("INT", {"default": 1, "min": 1, "max": 99999}),
            }
        }

    RETURN_TYPES = (ANY_TYPE, )
    RETURN_NAMES = ("STRING", )
    OUTPUT_IS_LIST = (True, )
    FUNCTION = "cycle"
    CATEGORY = "Apt_Collect/view_IO"

    def cycle(self, text, repeats, loops=1):
    

    
        lines = text.split('\n')
        list_out = []

        for i in range(loops):
            for text_item in lines:
                for _ in range(repeats):
                    list_out.append(text_item)
        
        return (list_out,)


class Output_Valuelist:
    
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "values": ("STRING", {"multiline": True, "default": ""}),
            "repeats": ("INT", {"default": 1, "min": 1, "max": 99999}),
            "loops": ("INT", {"default": 1, "min": 1, "max": 99999}),
            }
        }

    RETURN_TYPES = ("FLOAT", "INT", )
    RETURN_NAMES = ("FLOAT", "INT", )
    OUTPUT_IS_LIST = (True, True, )
    FUNCTION = "cycle"
    CATEGORY = "Apt_Collect/view_IO" 

    def cycle(self, values, repeats, loops=1):
    
    
        lines = values.split('\n')
        float_list_out = []
        int_list_out = []

        # add check if valid number

        for i in range(loops):
            for _ in range(repeats):
                for text_item in lines:
                    if all(char.isdigit() or char == '.' for char in text_item.strip()):
                        float_list_out.append(float(text_item))
                        int_list_out.append(int(float(text_item)))  # Convert to int after parsing as float

        return (float_list_out, int_list_out, )    


class AlwaysEqualProxy(str):
    def __eq__(self, _):
        return True

    def __ne__(self, _):
        return False


class view_combo:

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "prompt": ("STRING", {"multiline": True, "default": "text"}),
                    "start_index": ("INT", {"default": 0, "min": 0, "max": 9999}),
                    "max_rows": ("INT", {"default": 1000, "min": 1, "max": 9999}),
                    },
            "hidden":{
                "workflow_prompt": "PROMPT", "my_unique_id": "UNIQUE_ID"
            }
        }

    RETURN_TYPES = ("STRING", AlwaysEqualProxy('*'))
    RETURN_NAMES = ("STRING", "COMBO")
    OUTPUT_IS_LIST = (True, True)
    FUNCTION = "generate_strings"
    CATEGORY = "Apt_Collect/view_IO"

    def generate_strings(self, prompt, start_index, max_rows, workflow_prompt=None, my_unique_id=None):
        lines = prompt.split('\n')

        start_index = max(0, min(start_index, len(lines) - 1))
        end_index = min(start_index + max_rows, len(lines))
        rows = lines[start_index:end_index]

        return (rows, rows)


class view_node_Script:
    def __init__(self):
        self.node_list = []
        self.custom_node_list = []
        self.update_node_list()

    def update_node_list(self):
        """Scan and update the list of available nodes in ComfyUI"""
        try:
            import nodes
            self.node_list = []
            self.custom_node_list = []
            
            for node_name, node_class in nodes.NODE_CLASS_MAPPINGS.items():
                try:
                    # Determine if it's a custom node
                    module = inspect.getmodule(node_class)
                    module_path = getattr(module, '__file__', '')
                    is_custom = 'custom_nodes' in module_path

                    node_info = {
                        'name': node_name,
                        'class_name': node_class.__name__,
                        'category': getattr(node_class, 'CATEGORY', 'Uncategorized'),
                        'description': getattr(node_class, 'DESCRIPTION', ''),
                        'is_custom': is_custom
                    }
                    
                    self.node_list.append(node_info)
                    if is_custom:
                        self.custom_node_list.append(node_info)
                except Exception as e:
                    logging.error(f"Error processing node {node_name}: {str(e)}")
                    continue
            
            # Sort nodes alphabetically
            self.node_list.sort(key=lambda x: x['name'])
            self.custom_node_list.sort(key=lambda x: x['name'])
            
        except Exception as e:
            logging.error(f"Error updating node list: {str(e)}")
            traceback.print_exc()

    @classmethod
    def INPUT_TYPES(cls):
        try:
            import nodes
            node_names = sorted(list(nodes.NODE_CLASS_MAPPINGS.keys()))
            if not node_names:
                node_names = ["No nodes found"]
                
            return {
                "required": {
                    "mode": (["All Nodes", "Custom Nodes Only", "Built-in Nodes Only"], {
                        "default": "All Nodes"
                    }),
                    "view_mode": (["List Nodes", "View Source Code", "Usage Guide"], {
                        "default": "List Nodes"
                    }),
                    "selected_node": (node_names, {
                        "default": node_names[0]
                    }),
                    "search": ("STRING", {
                        "default": "",
                        "multiline": False
                    }),
                    "show_all": ("BOOLEAN", {
                        "default": True,
                        "label": "Show All Nodes"
                    }),
                    "refresh_list": ("BOOLEAN", {
                        "default": False,
                        "label": "Refresh Node List"
                    })
                }
            }
        except Exception as e:
            print(f"Error in INPUT_TYPES: {str(e)}")
            return {
                "required": {
                    "mode": (["All Nodes", "Custom Nodes Only", "Built-in Nodes Only"], {"default": "All Nodes"}),
                    "view_mode": (["List Nodes", "View Source Code", "Usage Guide"], {"default": "List Nodes"}),
                    "search": ("STRING", {"default": "", "multiline": False}),
                    "show_all": ("BOOLEAN", {"default": True, "label": "Show All Nodes"}),
                    "refresh_list": ("BOOLEAN", {"default": False, "label": "Refresh Node List"})
                }
            }

    RETURN_TYPES = ("STRING", "STRING",)
    RETURN_NAMES = ("node_info", "node_source",)
    FUNCTION = "find_script"
    CATEGORY = "Apt_Collect/view_IO"

    def get_node_source_code(self, node_name):
        """Get the source code of a node"""
        try:
            import nodes
            import inspect
            import os

            # Get the node class
            node_class = nodes.NODE_CLASS_MAPPINGS.get(node_name)
            if not node_class:
                return f"Node '{node_name}' not found"

            # Get the module
            module = inspect.getmodule(node_class)
            if not module:
                return f"Could not find module for {node_name}"

            # Get file path
            try:
                file_path = inspect.getfile(module)
            except TypeError:
                return f"Could not determine file path for {node_name}"

            # Read entire file content
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    file_content = f.read()
            except Exception as e:
                return f"Error reading file: {str(e)}"

            # Find the class definition
            class_def = f"class {node_class.__name__}:"
            class_start = file_content.find(class_def)
            
            if class_start == -1:
                return f"Could not find class definition for {node_name}"

            # Extract class source code
            lines = file_content[class_start:].split('\n')
            class_lines = []
            indent_level = None

            for line in lines:
                # Determine initial indent level
                if indent_level is None:
                    if line.strip().startswith('class'):
                        indent_level = len(line) - len(line.lstrip())
                    continue

                # Check if we've reached the end of the class
                current_indent = len(line) - len(line.lstrip())
                if current_indent <= indent_level and line.strip():
                    break

                class_lines.append(line)

            # Construct formatted output
            source_output = f"=== Node: {node_name} ===\n"
            source_output += f"File: {file_path}\n\n"
            source_output += "=== Source Code ===\n"
            source_output += "\n".join(class_lines)

            return source_output

        except Exception as e:
            return f"Error retrieving source code: {str(e)}"

    def find_script(self, mode, view_mode, selected_node, search, show_all, refresh_list):
        """Main function to find and return selected node"""
        try:
            # Refresh node list if requested
            if refresh_list:
                self.update_node_list()

            # Handle source code view mode
            if view_mode == "View Source Code":
                if selected_node:
                    source_code = self.get_node_source_code(selected_node)
                    return f"Source Code for {selected_node}", source_code
                return "No node selected", "Please select a node to view its source code"

            # Default fallback
            return "Node Source Finder", "Select a node and choose 'View Source Code' mode"

        except Exception as e:
            logging.error(f"Error in find_script: {str(e)}")
            traceback.print_exc()
            return f"Error: {str(e)}", traceback.format_exc()







#region--------------------------------------------

import torch
import gc
import psutil
import os
import ctypes
import comfy.model_management as mm

class MemoryBase:
    def free_memory(self, aggressive=False):
        print("Attempting to free GPU VRAM and system RAM...")
        self.free_gpu_vram(aggressive)
        self.free_system_ram(aggressive)

    def free_gpu_vram(self, aggressive):
        if torch.cuda.is_available():
            initial_memory = torch.cuda.memory_allocated()
            
            if aggressive:
                mm.unload_all_models()
                mm.soft_empty_cache()
            
            torch.cuda.empty_cache()
            
            final_memory = torch.cuda.memory_allocated()
            memory_freed = initial_memory - final_memory
            print(f"GPU VRAM: Initial usage: {initial_memory/1e9:.2f} GB, "
                  f"Final usage: {final_memory/1e9:.2f} GB, "
                  f"Freed: {memory_freed/1e9:.2f} GB")
        else:
            print("CUDA is not available. No GPU VRAM to free.")

    def free_system_ram(self, aggressive):
        initial_memory = psutil.virtual_memory().percent
        
        collected = gc.collect()
        print(f"Garbage collector: collected {collected} objects.")

        if aggressive:
            if os.name == 'posix':  # Unix/Linux
                try:
                    os.system('sync')
                    with open('/proc/sys/vm/drop_caches', 'w') as f:
                        f.write('3')
                    print("Cleared system caches on Linux.")
                except Exception as e:
                    print(f"Failed to clear system caches on Linux: {str(e)}")
            elif os.name == 'nt':  # Windows
                try:
                    ctypes.windll.psapi.EmptyWorkingSet(ctypes.windll.kernel32.GetCurrentProcess())
                    print("Attempted to clear working set on Windows.")
                except Exception as e:
                    print(f"Failed to clear working set on Windows: {str(e)}")

        final_memory = psutil.virtual_memory().percent
        memory_freed = initial_memory - final_memory
        print(f"System RAM: Initial usage: {initial_memory:.2f}%, "
              f"Final usage: {final_memory:.2f}%, "
              f"Freed: {memory_freed:.2f}%")

class IO_Free_Memory(MemoryBase):
    @classmethod
    def INPUT_TYPES(s):
        return {
            
            "optional": {
            "image": ("IMAGE",),
            "latent": ("LATENT",),
            "model": ("MODEL",),
            "clip": ("CLIP",),
            "aggressive": ("BOOLEAN", {"default": False})
        }}
    
    RETURN_TYPES = ("IMAGE", "LATENT", "MODEL", "CLIP")
    
    FUNCTION = "free_memory_combined"
    CATEGORY = "Apt_Collect/view_IO"

    def free_memory_combined(self, image=None, latent=None, model=None, clip=None, aggressive=False):
        self.free_memory(aggressive)
        return (image, latent, model, clip)





#endregion--------------------------------------------