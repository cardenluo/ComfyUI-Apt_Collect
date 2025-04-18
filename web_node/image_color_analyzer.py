# Standard library # Standard library imports
import os
import json

# Third-party imports
import numpy as np
import torch
from PIL import Image
from sklearn.cluster import KMeans
from skimage import color

# Local imports
import folder_paths

# Add new imports for specific exceptions
from typing import Tuple, List, Dict, Optional
from PIL import UnidentifiedImageError


class Image_Color_Analyzer:
    """
    ComfyUI node for analyzing image colors and generating SD-friendly descriptions
    """
    
    def __init__(self):
        """Initialize the color analyzer"""
        self.data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "web", "color")
        os.makedirs(self.data_dir, exist_ok=True)
        
        self.color_data = {}
        self.theme_data = {}
        
        # Color analysis parameters
        self.min_color_distance = 15.0  # Minimum Delta-E distance between colors
        self.color_weights = {
            'presence': 0.4,
            'saturation': 0.3,
            'position': 0.3
        }
        
        self.load_color_data()
        self.load_theme_data()

    def load_color_data(self):
        """Load and merge color definitions from all color JSON files"""
        try:
            self.color_data = {}
            color_files = ["color_names.json", "html_colors.json"]
            
            for filename in color_files:
                file_path = os.path.join(self.data_dir, filename)
                if os.path.exists(file_path):
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            for category, colors in data.items():
                                if category not in self.color_data:
                                    self.color_data[category] = {}
                                self.color_data[category].update(colors)
                    except json.JSONDecodeError as e:
                        print(f"Invalid JSON format in {filename}: {e}")
                    except UnicodeDecodeError as e:
                        print(f"Encoding error in {filename}: {e}")
                    except IOError as e:
                        print(f"IO error reading {filename}: {e}")
                        
            if not self.color_data:
                print("Warning: No color data loaded, using basic colors")
                self.color_data = self.get_basic_colors()
                
        except Exception as e:
            print(f"Critical error in load_color_data: {str(e)}")
            self.color_data = self.get_basic_colors()

    def load_theme_data(self):
        """Load color theme definitions"""
        try:
            theme_file = os.path.join(self.data_dir, "color_themes.json")
            if os.path.exists(theme_file):
                with open(theme_file, 'r', encoding='utf-8') as f:
                    self.theme_data = json.load(f)
            else:
                print("Warning: No theme data found, using basic themes")
                self.theme_data = self.get_basic_themes()
        except Exception as e:
            print(f"Error loading theme data: {e}")
            self.theme_data = self.get_basic_themes()

    @classmethod
    def INPUT_TYPES(cls):
        # Get list of input images from directory
        input_dir = folder_paths.get_input_directory()
        files = []
        for root, dirs, filenames in os.walk(input_dir):
            for filename in filenames:
                full_path = os.path.join(root, filename)
                relative_path = os.path.relpath(full_path, input_dir)
                relative_path = relative_path.replace("\\", "/")
                files.append(relative_path)

        return {
            "required": {
                "image": (sorted(files), {"image_upload": True}),
                "num_colors": ("INT", {
                    "default": 5,
                    "min": 3,
                    "max": 16,
                    "step": 1
                }),
                "color_sample_width": ("INT", {
                    "default": 512,
                    "min": 8,
                    "max": 4096
                }),
                "color_sample_height": ("INT", {
                    "default": 512,
                    "min": 8,
                    "max": 4096
                })
            },
            "optional": {
                "pipe_input": ("IMAGE",)
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "STRING", "STRING",)
    RETURN_NAMES = ("color_panels", "color_names", "color_theme", "detailed_info", "hex_values",)
    FUNCTION = "analyze_image"
    CATEGORY = "Apt_Collect/color"

    def create_color_panel(self, width, height, hex_color):
        """Create a color panel for a given hex color"""
        try:
            # Convert hex to RGB
            hex_color = hex_color.lstrip('#')
            rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
            
            # Create new image with specified color
            panel = Image.new('RGB', (width, height), rgb)
            
            # Convert to tensor
            img_array = np.array(panel)
            img_tensor = torch.from_numpy(img_array).float() / 255.0
            img_tensor = img_tensor.unsqueeze(0)
            return img_tensor
        except Exception as e:
            print(f"Error creating color panel: {e}")
            return None

    def rgb_to_lab(self, rgb):
        """Convert RGB to L*a*b* color space"""
        try:
            # Normalize RGB values to [0, 1]
            rgb_norm = np.array(rgb) / 255.0
            # Reshape for skimage
            rgb_norm = rgb_norm.reshape(1, 1, 3)
            # Convert to L*a*b*
            lab = color.rgb2lab(rgb_norm)
            return lab[0, 0]
        except Exception as e:
            print(f"Error converting RGB to LAB: {e}")
            return np.array([0, 0, 0])

    def calculate_color_distance(self, color1, color2):
        """Calculate Delta-E color difference in L*a*b* space"""
        try:
            lab1 = self.rgb_to_lab(color1)
            lab2 = self.rgb_to_lab(color2)
            
            # Calculate Euclidean distance in L*a*b* space
            delta_L = lab1[0] - lab2[0]
            delta_a = lab1[1] - lab2[1]
            delta_b = lab1[2] - lab2[2]
            
            # Calculate Delta-E
            delta_E = np.sqrt(delta_L**2 + delta_a**2 + delta_b**2)
            return delta_E
        except Exception as e:
            print(f"Error calculating color distance: {e}")
            return float('inf')

    def tensor_to_pil(self, img_tensor):
        """Convert a PyTorch tensor to PIL Image"""
        try:
            if torch.is_tensor(img_tensor):
                img_tensor = img_tensor.cpu()
                img_array = img_tensor.numpy()
                img_array = (img_array * 255).astype(np.uint8)
                if len(img_array.shape) == 4:
                    img_array = img_array[0]
                return Image.fromarray(img_array)
            return img_tensor
        except Exception as e:
            print(f"Error converting tensor to PIL: {e}")
            return None

    def rgb_to_hex(self, rgb):
        """Convert RGB tuple to HEX string"""
        try:
            return '#{:02x}{:02x}{:02x}'.format(int(rgb[0]), int(rgb[1]), int(rgb[2]))
        except Exception as e:
            print(f"Error converting RGB to HEX: {e}")
            return "#000000"

    def get_color_name(self, rgb):
        """Find the closest matching color name using Delta-E"""
        try:
            min_distance = float('inf')
            color_name = None
            sd_name = None
            
            # Iterate through all categories and colors
            for category, colors in self.color_data.items():
                for name, data in colors.items():
                    if 'rgb' not in data:
                        continue
                        
                    distance = self.calculate_color_distance(rgb, data['rgb'])
                    if distance < min_distance:
                        min_distance = distance
                        color_name = name
                        sd_name = data.get('sd_names', [name])[0]
            
            # Always return the closest match, don't use a threshold
            if color_name is None:
                return "undefined", "undefined"
            
            return sd_name, color_name
            
        except Exception as e:
            print(f"Error finding color name: {e}")
            return "undefined", "undefined"

    def detect_color_theme(self, colors, percentages):
        """Detect the overall color theme"""
        try:
            if not colors:
                return "undefined theme"

            # Remove duplicates while preserving order
            unique_colors = []
            seen = set()
            for color in colors:
                if color.lower() not in seen:
                    unique_colors.append(color)
                    seen.add(color.lower())
            
            # Join colors with "and" and append "color palette"
            return " and ".join(unique_colors) + " color palette"
            
        except Exception as e:
            print(f"Error detecting theme: {e}")
            return "undefined theme"

    def generate_basic_theme_description(self, colors, percentages):
        """Generate a basic theme description"""
        try:
            if not colors:
                return "undefined theme"
            
            primary_color = colors[0]
            if percentages[0] > 50:
                return f"dominant {primary_color} color scheme"
            else:
                # Use 'and' for the last color instead of comma
                if len(colors) > 2:
                    color_list = colors[:3]
                    return f"color palette of {', '.join(color_list[:-1])} and {color_list[-1]}"
                elif len(colors) == 2:
                    return f"color palette of {colors[0]} and {colors[1]}"
                else:
                    return f"color palette of {colors[0]}"
        except Exception as e:
            print(f"Error generating theme description: {e}")
            return "undefined theme"

    def analyze_image(self, image, num_colors=5, color_sample_width=512, color_sample_height=512, pipe_input=None):
        """Analyze image colors from either pipe input or file upload"""
        try:
            if pipe_input is not None:
                if not isinstance(pipe_input, (torch.Tensor, Image.Image)):
                    raise ValueError("Pipe input must be a tensor or PIL Image")
                # Handle piped image input
                if isinstance(pipe_input, torch.Tensor):
                    # Convert tensor to PIL Image
                    if pipe_input.ndim == 4:
                        pipe_input = pipe_input.squeeze(0)  # Remove batch dimension if present
                    img_array = (pipe_input * 255).byte().cpu().numpy()
                    if img_array.shape[0] == 3:  # If channels first
                        img_array = np.transpose(img_array, (1, 2, 0))
                    img = Image.fromarray(img_array)
                else:
                    img = pipe_input
            else:
                try:
                    image_path = folder_paths.get_annotated_filepath(image)
                    if not os.path.exists(image_path):
                        raise FileNotFoundError(f"Image file not found: {image_path}")
                    img = Image.open(image_path)
                except UnidentifiedImageError as e:
                    raise ValueError(f"Invalid or corrupted image file: {e}")
                except FileNotFoundError as e:
                    raise FileNotFoundError(f"Image file not found: {e}")

            # Input validation
            if not (3 <= num_colors <= 16):
                raise ValueError("num_colors must be between 3 and 16")
            if not (8 <= color_sample_width <= 4096):
                raise ValueError("color_sample_width must be between 8 and 4096")
            if not (8 <= color_sample_height <= 4096):
                raise ValueError("color_sample_height must be between 8 and 4096")

            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')

            # Convert to numpy array for processing
            img_array = np.array(img)

            # Resize for processing
            img = img.resize((150, 150))
            
            # Convert to numpy array for K-means
            img_array = np.array(img)
            pixels = img_array.reshape(-1, 3)
            
            # Apply K-means clustering
            kmeans = KMeans(n_clusters=num_colors, random_state=42)
            kmeans.fit(pixels)
            colors = kmeans.cluster_centers_
            
            # Calculate color percentages
            labels = kmeans.labels_
            total_pixels = len(labels)
            percentages = [(np.sum(labels == i) / total_pixels) * 100 
                          for i in range(num_colors)]
            
            # Sort colors by percentage
            color_info = sorted(zip(colors, percentages), 
                              key=lambda x: x[1], 
                              reverse=True)
            
            # Process colors
            color_names = []
            detailed_info = []
            sd_colors = []
            hex_values = []
            color_panels = []
            
            # Generate color information
            for i, (color, percentage) in enumerate(color_info[:num_colors]):
                rgb = tuple(map(int, color))
                hex_value = self.rgb_to_hex(rgb)
                sd_name, internal_name = self.get_color_name(rgb)
                
                if sd_name == "undefined":
                    sd_name = f"Color {i+1}"
                
                color_names.append(sd_name)
                sd_colors.append(sd_name)
                hex_values.append(hex_value)
                detailed_info.append(
                    f"{sd_name}  |  {percentage:.1f}%  |  "
                    f"r={rgb[0]} g={rgb[1]} b={rgb[2]}  |  {hex_value}"
                )
                
                # Create color panel
                panel = self.create_color_panel(color_sample_width, color_sample_height, hex_value)
                if panel is not None:
                    color_panels.append(panel)
            
            # Combine color panels
            if color_panels:
                final_panels = torch.cat(color_panels, dim=0)
            else:
                final_panels = torch.zeros(1, 3, color_sample_height, color_sample_width)
            
            # Generate theme
            theme = self.detect_color_theme(sd_colors, percentages)
            
            # Remove duplicates while preserving order for color_names
            unique_color_names = []
            seen_names = set()
            for name in color_names:
                if name.lower() not in seen_names:
                    unique_color_names.append(name)
                    seen_names.add(name.lower())

            return (
                final_panels,
                "\n".join(unique_color_names),
                theme,
                "\n".join(detailed_info),
                "\n".join(hex_values)
            )
            
        except (ValueError, FileNotFoundError) as e:
            print(f"Input error: {e}")
            return self.error_output(color_sample_width, color_sample_height, str(e))
        except MemoryError as e:
            print(f"Memory error during image analysis: {e}")
            return self.error_output(color_sample_width, color_sample_height, "Out of memory")
        except Exception as e:
            print(f"Unexpected error during image analysis: {e}")
            return self.error_output(color_sample_width, color_sample_height, str(e))

    def error_output(self, color_sample_width, color_sample_height, error_message):
        """Generate error output in the expected format"""
        return (
            torch.zeros(1, 3, color_sample_height, color_sample_width),
            "Error analyzing colors",
            "Error detecting theme",
            f"Error analyzing image: {error_message}",
            "Error"
        )

    def get_basic_colors(self):
        """Fallback basic color definitions"""
        return {
            "basic": {
                "red": {
                    "rgb": [255, 0, 0],
                    "sd_names": ["red"],
                    "category": "basic"
                },
                "green": {
                    "rgb": [0, 255, 0],
                    "sd_names": ["green"],
                    "category": "basic"
                },
                "blue": {
                    "rgb": [0, 0, 255],
                    "sd_names": ["blue"],
                    "category": "basic"
                }
            }
        }

    def get_basic_themes(self):
        """Fallback basic theme definitions"""
        return {
            "basic": {
                "default": {
                    "sd_prompt": "color scheme",
                    "compatible_colors": ["red", "green", "blue"]
                }
            }
        }


    def calculate_color_importance(self, color, position, size, frequency):
        """Calculate the importance of a color based on multiple factors"""
        # Normalize position
        x, y = position
        w, h = size
        norm_x, norm_y = x/w, y/h
        
        # Calculate center weight with even less penalty for off-center colors
        center_dist = np.sqrt((norm_x - 0.5)**2 + (norm_y - 0.5)**2)
        position_weight = 1 / (1 + center_dist * 0.5)  # Even softer distance penalty
        
        # Calculate saturation and vibrancy
        r, g, b = color/255
        max_val = max(r, g, b)
        min_val = min(r, g, b)
        
        # Enhanced saturation calculation
        saturation = (max_val - min_val) / max_val if max_val != 0 else 0
        
        # Calculate color distinctiveness
        color_std = np.std([r, g, b])
        
        # Special handling for vibrant colors (like red mushrooms)
        is_vibrant = False
        if r > max(g, b) * 1.5:  # Red channel significantly higher
            is_vibrant = True
        
        # Calculate relative channel dominance
        channel_dominance = max(r, g, b) / (np.mean([r, g, b]) + 0.001)
        
        # Boost importance for small but vibrant/distinct colors
        if frequency < 0.1:  # For colors that occupy less than 10% of the image
            if is_vibrant or saturation > 0.5 or channel_dominance > 1.3:
                frequency = frequency * 4  # Quadruple their effective presence
        
        # Calculate distinctiveness with more weight to saturated colors
        distinctiveness = (color_std + saturation) / 2
        
        # Combine factors with adjusted weights
        importance = (
            frequency * 0.3 +                    # Reduced weight for frequency
            saturation * 0.3 +                   # Increased weight for saturation
            distinctiveness * 0.2 +              # Added distinctiveness factor
            position_weight * 0.1 +              # Reduced position influence
            (1.0 if is_vibrant else 0.0) * 0.1  # Bonus for vibrant colors
        )
        
        return importance
