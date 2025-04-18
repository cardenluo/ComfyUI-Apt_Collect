
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, TRANSFORMERS_CACHE
import os
import shutil
import folder_paths




class BaseTranslatorNode:
    def __init__(self, model_name):
        self.use_gpu = True  # 默认使用GPU
        self.model_name = model_name
        self.model_path = os.path.join(folder_paths.base_path, "models", "translator", model_name)
        
        
        self.set_device()
        self.load_or_download_model()

    def set_device(self):
        self.device = torch.device("cuda" if self.use_gpu and torch.cuda.is_available() else "cpu")

    def load_or_download_model(self):
        try:
            # 尝试从本地加载模型
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_path, local_files_only=True).to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, local_files_only=True)
        except Exception as e:
            print(f"本地模型加载失败: {e}")
            print("尝试下载模型")
            
            try:
                # 从Hugging Face下载模型
                model_id = f"Helsinki-NLP/{self.model_name}"
                self.model = AutoModelForSeq2SeqLM.from_pretrained(model_id).to(self.device)
                self.tokenizer = AutoTokenizer.from_pretrained(model_id)
                
                # 保存模型到本地
                self.model.save_pretrained(self.model_path)
                self.tokenizer.save_pretrained(self.model_path)            
                
                self.clear_cache() # 清理缓存
            except Exception as e:
                print(f"模型下载或保存失败: {e}")
                raise  # 重新抛出异常，让调用者处理

    def clear_cache(self):
        if os.path.exists(TRANSFORMERS_CACHE):
            try:
                shutil.rmtree(TRANSFORMERS_CACHE)
                print(f"缓存已清除: {TRANSFORMERS_CACHE}")
            except Exception as e:
                print(f"清除缓存时出错: {e}")
        else:
            print(f"缓存目录不存在: {TRANSFORMERS_CACHE}")  

    @classmethod
    def INPUT_TYPES(s):
        return {            
            "required": {
                "use_gpu": ("BOOLEAN", {"default": True, "label": "使用GPU"}),
                "input_text": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "输入要翻译的文本"
                }),                
            },
            "optional": {
                "optional_input_text": ("STRING", {
                    "forceInput": True,
                    "default": "",
                    "placeholder": "连接的输入（如果提供则优先使用）"
                }),
            }
        }

    OUTPUT_NODE = True
    RETURN_TYPES = ("STRING",)
    FUNCTION = "translate"
    CATEGORY = "Apt_Collect/GPT"

    def translate(self, input_text, use_gpu, optional_input_text=""):
        if self.use_gpu != use_gpu:
            self.use_gpu = use_gpu
            self.set_device()
            self.model = self.model.to(self.device)

        text_to_translate = optional_input_text if optional_input_text.strip() else input_text
        
        if not text_to_translate.strip():
            return {"ui": {"text": ""}, "result": ("",)}
        
        inputs = self.tokenizer(text_to_translate, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(**inputs)
        
        translated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)  

        return {"ui": {"text": translated_text}, "result": (translated_text,)}

class ChineseToEnglish(BaseTranslatorNode):
    def __init__(self):
        super().__init__("opus-mt-zh-en")

class EnglishToChinese(BaseTranslatorNode):
    def __init__(self):
        super().__init__("opus-mt-en-zh")

