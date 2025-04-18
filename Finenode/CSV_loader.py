

import os
import re
import folder_paths

class text_CSV_load:
    # 获取当前脚本所在目录
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    CSV_DIR = os.path.join(BASE_DIR, "CSV")

    @staticmethod
    def load_csv(csv_path: str):
        data = {"Error loading CSV, check the console": ["", ""]}
        if not os.path.exists(csv_path):
            return data
        try:
            with open(csv_path, "r", encoding="utf-8") as f:
                data = [[x.replace('"', '').replace('\n', '') for x in re.split(',(?=(?:[^"]*"[^"]*")*[^"]*$)', line)] for line in f.readlines()[1:]]
                data = {x[0]: [x[1], x[2]] for x in data}
        except Exception as e:
            print(f"Error loading CSV: {csv_path}. Error: {e}")
        return data

    @classmethod
    def INPUT_TYPES(cls):
        # 使用构建好的绝对路径加载 CSV 文件
        cls.artists_csv = cls.load_csv(os.path.join(cls.CSV_DIR, "A_artists.csv"))
        cls.material_csv = cls.load_csv(os.path.join(cls.CSV_DIR, "A_material.csv"))
        cls.nature_csv = cls.load_csv(os.path.join(cls.CSV_DIR, "E_nature.csv"))
        cls.scene_csv = cls.load_csv(os.path.join(cls.CSV_DIR, "E_scene.csv"))
        cls.building_csv = cls.load_csv(os.path.join(cls.CSV_DIR, "M_building.csv"))
        cls.cosplay_csv = cls.load_csv(os.path.join(cls.CSV_DIR, "M_cosplay.csv"))
        cls.camera_csv = cls.load_csv(os.path.join(cls.CSV_DIR, "T_camera.csv"))
        cls.cameraEffect_csv = cls.load_csv(os.path.join(cls.CSV_DIR, "T_cameraEffect.csv"))
        cls.detail_csv = cls.load_csv(os.path.join(cls.CSV_DIR, "T_detail.csv"))
        cls.light_csv = cls.load_csv(os.path.join(cls.CSV_DIR, "T_light.csv"))

        return {
            "required": {
                "A_artists": (list(cls.artists_csv.keys()),),
                "A_material": (list(cls.material_csv.keys()),),
                "M_cosplay": (list(cls.cosplay_csv.keys()),),
                "T_detail": (list(cls.detail_csv.keys()),),
                "T_light": (list(cls.light_csv.keys()),),
                "E_nature": (list(cls.nature_csv.keys()),),
                "E_scene": (list(cls.scene_csv.keys()),),
                "M_building1": (list(cls.building_csv.keys()),),
                "M_building2": (list(cls.building_csv.keys()),),
                "M_building3": (list(cls.building_csv.keys()),),
                "T_camera": (list(cls.camera_csv.keys()),),
                "T_cameraEffect": (list(cls.cameraEffect_csv.keys()),),
                "subject": ("STRING", {"multiline": True}),
            },
        }

    # ... 其他代码保持不变 ...


    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("pos", "neg")
    FUNCTION = "execute"
    CATEGORY = "Apt_Collect/prompt"

    def execute(self, A_artists, M_cosplay, T_detail, T_light, E_nature, E_scene, M_building1, M_building2, M_building3, A_material, T_camera, T_cameraEffect,subject=""):
        # Combine all positive and negative prompts
        positive_prompt = ""
        negative_prompt = ""

        # Aggregating all positive and negative prompts from the different CSV files
        for csv_dict in [self.artists_csv, self.building_csv, self.cosplay_csv, self.nature_csv, self.scene_csv,
                        self.material_csv, self.camera_csv, self.cameraEffect_csv, self.detail_csv, self.light_csv]:
            if A_artists in csv_dict:
                positive_prompt += csv_dict[A_artists][0] + " "
                negative_prompt += csv_dict[A_artists][1] + " "
            if M_cosplay in csv_dict:
                positive_prompt += csv_dict[M_cosplay][0] + " "
                negative_prompt += csv_dict[M_cosplay][1] + " "
            if T_detail in csv_dict:
                positive_prompt += csv_dict[T_detail][0] + " "
                negative_prompt += csv_dict[T_detail][1] + " "
            if T_light in csv_dict:
                positive_prompt += csv_dict[T_light][0] + " "
                negative_prompt += csv_dict[T_light][1] + " "
            if E_nature in csv_dict:
                positive_prompt += csv_dict[E_nature][0] + " "
                negative_prompt += csv_dict[E_nature][1] + " "
            if E_scene in csv_dict:
                positive_prompt += csv_dict[E_scene][0] + " "
                negative_prompt += csv_dict[E_scene][1] + " "
            if M_building1 in csv_dict:
                positive_prompt += csv_dict[M_building1][0] + " "
                negative_prompt += csv_dict[M_building1][1] + " "
            if M_building2 in csv_dict:
                positive_prompt += csv_dict[M_building2][0] + " "
                negative_prompt += csv_dict[M_building2][1] + " "
            if M_building3 in csv_dict:
                positive_prompt += csv_dict[M_building3][0] + " "
                negative_prompt += csv_dict[M_building3][1] + " "
            if A_material in csv_dict:
                positive_prompt += csv_dict[A_material][0] + " "
                negative_prompt += csv_dict[A_material][1] + " "
            if T_camera in csv_dict:
                positive_prompt += csv_dict[T_camera][0] + " "
                negative_prompt += csv_dict[T_camera][1] + " "
            if T_cameraEffect in csv_dict:
                positive_prompt += csv_dict[T_cameraEffect][0] + " "
                negative_prompt += csv_dict[T_cameraEffect][1] + " "

        pos = positive_prompt.strip()
        neg = negative_prompt.strip()

        prompt = "{prompt}"
        if prompt not in pos:
            pos = subject + " " + pos
        else:
            pos = pos.replace(prompt, subject)
        return (pos, neg)

