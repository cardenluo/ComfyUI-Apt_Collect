import math
import torch
from typing import Any, Callable, Mapping
import numpy as np




class AnyType(str):
    def __eq__(self, _) -> bool:
        return True
    def __ne__(self, __value: object) -> bool:
        return False
ANY_TYPE = AnyType("*")




#region  ----缓动函数-------------------------------------------------------------#
def apply_easing(value, easing_type):
    if easing_type == "Linear":
        return value

    # Back easing functions
    def easeInBack(t):
        s = 1.70158
        return t * t * ((s + 1) * t - s)

    def easeOutBack(t):
        s = 1.70158
        return ((t - 1) * t * ((s + 1) * t + s)) + 1

    def easeInOutBack(t):
        s = 1.70158 * 1.525
        if t < 0.5:
            return (t * t * (t * (s + 1) - s)) * 2
        return ((t - 2) * t * ((s + 1) * t + s) + 2) * 2

    # Elastic easing functions
    def easeInElastic(t):
        if t == 0:
            return 0
        if t == 1:
            return 1
        p = 0.3
        s = p / 4
        return -(
            math.pow(2, 10 * (t - 1))
            * math.sin((t - 1 - s) * (2 * math.pi) / p)
        )

    def easeOutElastic(t):
        if t == 0:
            return 0
        if t == 1:
            return 1
        p = 0.3
        s = p / 4
        return math.pow(2, -10 * t) * math.sin((t - s) * (2 * math.pi) / p) + 1

    def easeInOutElastic(t):
        if t == 0:
            return 0
        if t == 1:
            return 1
        p = 0.3 * 1.5
        s = p / 4
        t = t * 2
        if t < 1:
            return -0.5 * (
                math.pow(2, 10 * (t - 1))
                * math.sin((t - 1 - s) * (2 * math.pi) / p)
            )
        return (
            0.5
            * math.pow(2, -10 * (t - 1))
            * math.sin((t - 1 - s) * (2 * math.pi) / p)
            + 1
        )

    # Bounce easing functions
    def easeInBounce(t):
        return 1 - easeOutBounce(1 - t)

    def easeOutBounce(t):
        if t < (1 / 2.75):
            return 7.5625 * t * t
        elif t < (2 / 2.75):
            t -= 1.5 / 2.75
            return 7.5625 * t * t + 0.75
        elif t < (2.5 / 2.75):
            t -= 2.25 / 2.75
            return 7.5625 * t * t + 0.9375
        else:
            t -= 2.625 / 2.75
            return 7.5625 * t * t + 0.984375

    def easeInOutBounce(t):
        if t < 0.5:
            return easeInBounce(t * 2) * 0.5
        return easeOutBounce(t * 2 - 1) * 0.5 + 0.5

    # Quart easing functions
    def easeInQuart(t):
        return t * t * t * t

    def easeOutQuart(t):
        t -= 1
        return -(t**2 * t * t - 1)

    def easeInOutQuart(t):
        t *= 2
        if t < 1:
            return 0.5 * t * t * t * t
        t -= 2
        return -0.5 * (t**2 * t * t - 2)

    # Cubic easing functions
    def easeInCubic(t):
        return t * t * t

    def easeOutCubic(t):
        t -= 1
        return t**2 * t + 1

    def easeInOutCubic(t):
        t *= 2
        if t < 1:
            return 0.5 * t * t * t
        t -= 2
        return 0.5 * (t**2 * t + 2)

    # Circ easing functions
    def easeInCirc(t):
        return -(math.sqrt(1 - t * t) - 1)

    def easeOutCirc(t):
        t -= 1
        return math.sqrt(1 - t**2)

    def easeInOutCirc(t):
        t *= 2
        if t < 1:
            return -0.5 * (math.sqrt(1 - t**2) - 1)
        t -= 2
        return 0.5 * (math.sqrt(1 - t**2) + 1)

    # Sine easing functions
    def easeInSine(t):
        return -math.cos(t * (math.pi / 2)) + 1

    def easeOutSine(t):
        return math.sin(t * (math.pi / 2))

    def easeInOutSine(t):
        return -0.5 * (math.cos(math.pi * t) - 1)

    easing_functions = {
        "Sine In": easeInSine,
        "Sine Out": easeOutSine,
        "Sine In+Out": easeInOutSine,
        "Quart In": easeInQuart,
        "Quart Out": easeOutQuart,
        "Quart In+Out": easeInOutQuart,
        "Cubic In": easeInCubic,
        "Cubic Out": easeOutCubic,
        "Cubic In+Out": easeInOutCubic,
        "Circ In": easeInCirc,
        "Circ Out": easeOutCirc,
        "Circ In+Out": easeInOutCirc,
        "Back In": easeInBack,
        "Back Out": easeOutBack,
        "Back In+Out": easeInOutBack,
        "Elastic In": easeInElastic,
        "Elastic Out": easeOutElastic,
        "Elastic In+Out": easeInOutElastic,
        "Bounce In": easeInBounce,
        "Bounce Out": easeOutBounce,
        "Bounce In+Out": easeInOutBounce,
    }

    function_ease = easing_functions.get(easing_type)
    if function_ease:
        return function_ease(value)

    raise ValueError(f"Unknown easing type: {easing_type}")




DEFAULT_FLOAT = ("FLOAT", {"default": 1.0, "min": 0.0, "max": 9999.0, "step": 0.001,})

FLOAT_UNARY_OPERATIONS: Mapping[str, Callable[[float], float]] = {
    "Neg": lambda a: -a,
    "Inc": lambda a: a + 1,
    "Dec": lambda a: a - 1,
    "Abs": lambda a: abs(a),
    "Sqr": lambda a: a * a,
    "Cube": lambda a: a * a * a,
    "Sqrt": lambda a: math.sqrt(a),
    "Exp": lambda a: math.exp(a),
    "Ln": lambda a: math.log(a),
    "Log10": lambda a: math.log10(a),
    "Log2": lambda a: math.log2(a),
    "Sin": lambda a: math.sin(a),
    "Cos": lambda a: math.cos(a),
    "Tan": lambda a: math.tan(a),
    "Asin": lambda a: math.asin(a),
    "Acos": lambda a: math.acos(a),
    "Atan": lambda a: math.atan(a),
    "Sinh": lambda a: math.sinh(a),
    "Cosh": lambda a: math.cosh(a),
    "Tanh": lambda a: math.tanh(a),
    "Asinh": lambda a: math.asinh(a),
    "Acosh": lambda a: math.acosh(a),
    "Atanh": lambda a: math.atanh(a),
    "Round": lambda a: round(a),
    "Floor": lambda a: math.floor(a),
    "Ceil": lambda a: math.ceil(a),
    "Trunc": lambda a: math.trunc(a),
    "Erf": lambda a: math.erf(a),
    "Erfc": lambda a: math.erfc(a),
    "Gamma": lambda a: math.gamma(a),
    "Radians": lambda a: math.radians(a),
    "Degrees": lambda a: math.degrees(a),
}

FLOAT_UNARY_CONDITIONS: Mapping[str, Callable[[float], bool]] = {
    "IsZero": lambda a: a == 0.0000,
    "IsPositive": lambda a: a > 0.000,
    "IsNegative": lambda a: a < 0.000,
    "IsNonZero": lambda a: a != 0.000,
    "IsPositiveInfinity": lambda a: math.isinf(a) and a > 0.000,
    "IsNegativeInfinity": lambda a: math.isinf(a) and a < 0.000,
    "IsNaN": lambda a: math.isnan(a),
    "IsFinite": lambda a: math.isfinite(a),
    "IsInfinite": lambda a: math.isinf(a),
    "IsEven": lambda a: a % 2 == 0.000,
    "IsOdd": lambda a: a % 2 != 0.000,
}

FLOAT_BINARY_OPERATIONS: Mapping[str, Callable[[float, float], float]] = {
    "Add": lambda a, b: a + b,
    "Sub": lambda a, b: a - b,
    "Mul": lambda a, b: a * b,
    "Div": lambda a, b: a / b,
    "Ceil": lambda a, b: math.ceil(a / b),
    "Mod": lambda a, b: a % b,
    "Pow": lambda a, b: a**b,
    "FloorDiv": lambda a, b: a // b,
    "Max": lambda a, b: max(a, b),
    "Min": lambda a, b: min(a, b),
    "Log": lambda a, b: math.log(a, b),
    "Atan2": lambda a, b: math.atan2(a, b),
}

FLOAT_BINARY_CONDITIONS: Mapping[str, Callable[[float, float], bool]] = {
    "Eq": lambda a, b: a == b,
    "Neq": lambda a, b: a != b,
    "Gt": lambda a, b: a > b,
    "Gte": lambda a, b: a >= b,
    "Lt": lambda a, b: a < b,
    "Lte": lambda a, b: a <= b,
}


DEFAULT_INT = ("INT", {"default": 0})

INT_UNARY_OPERATIONS: Mapping[str, Callable[[int], int]] = {
    "Abs": lambda a: abs(a),
    "Neg": lambda a: -a,
    "Inc": lambda a: a + 1,
    "Dec": lambda a: a - 1,
    "Sqr": lambda a: a * a,
    "Cube": lambda a: a * a * a,
    "Not": lambda a: ~a,
    "Factorial": lambda a: math.factorial(a),
}

INT_UNARY_CONDITIONS: Mapping[str, Callable[[int], bool]] = {
    "IsZero": lambda a: a == 0,
    "IsNonZero": lambda a: a != 0,
    "IsPositive": lambda a: a > 0,
    "IsNegative": lambda a: a < 0,
    "IsEven": lambda a: a % 2 == 0,
    "IsOdd": lambda a: a % 2 == 1,
}

INT_BINARY_OPERATIONS: Mapping[str, Callable[[int, int], int]] = {
    "Add": lambda a, b: a + b,
    "Sub": lambda a, b: a - b,
    "Mul": lambda a, b: a * b,
    "Div": lambda a, b: a // b,
    "Ceil": lambda a, b: math.ceil(a / b),
    "Mod": lambda a, b: a % b,
    "Pow": lambda a, b: a**b,
    "And": lambda a, b: a & b,
    "Nand": lambda a, b: ~a & b,
    "Or": lambda a, b: a | b,
    "Nor": lambda a, b: ~a & b,
    "Xor": lambda a, b: a ^ b,
    "Xnor": lambda a, b: ~a ^ b,
    "Shl": lambda a, b: a << b,
    "Shr": lambda a, b: a >> b,
    "Max": lambda a, b: max(a, b),
    "Min": lambda a, b: min(a, b),
}

INT_BINARY_CONDITIONS: Mapping[str, Callable[[int, int], bool]] = {
    "Eq": lambda a, b: a == b,
    "Neq": lambda a, b: a != b,
    "Gt": lambda a, b: a > b,
    "Lt": lambda a, b: a < b,
    "Geq": lambda a, b: a >= b,
    "Leq": lambda a, b: a <= b,
}

# endregion


class CFloatUnaryOperation:
    @classmethod
    def INPUT_TYPES(cls) -> Mapping[str, Any]:
        return {
            "required": {
                "op": (list(FLOAT_UNARY_OPERATIONS.keys()),),
                "a": DEFAULT_FLOAT,
            }
        }

    RETURN_TYPES = ("FLOAT",)
    FUNCTION = "op"
    CATEGORY = "Apt_Collect/math"

    def op(self, op: str, a: float) -> tuple[float]:
        return (FLOAT_UNARY_OPERATIONS[op](a),)


class CFloatUnaryCondition:
    @classmethod
    def INPUT_TYPES(cls) -> Mapping[str, Any]:
        return {
            "required": {
                "op": (list(FLOAT_UNARY_CONDITIONS.keys()),),
                "a": DEFAULT_FLOAT,
            }
        }

    RETURN_TYPES = ("BOOL",)
    FUNCTION = "op"
    CATEGORY = "Apt_Collect/math"

    def op(self, op: str, a: float) -> tuple[bool]:
        return (FLOAT_UNARY_CONDITIONS[op](a),)


class CFloatBinaryOperation:
    @classmethod
    def INPUT_TYPES(cls) -> Mapping[str, Any]:
        return {
            "required": {
                "op": (list(FLOAT_BINARY_OPERATIONS.keys()),),
                "a": DEFAULT_FLOAT,
                "b": DEFAULT_FLOAT,
            }
        }

    RETURN_TYPES = ("FLOAT",)
    FUNCTION = "op"
    CATEGORY = "Apt_Collect/math"

    def op(self, op: str, a: float, b: float) -> tuple[float]:
        return (FLOAT_BINARY_OPERATIONS[op](a, b),)


class CFloatBinaryCondition:
    @classmethod
    def INPUT_TYPES(cls) -> Mapping[str, Any]:
        return {
            "required": {
                "op": (list(FLOAT_BINARY_CONDITIONS.keys()),),
                "a": DEFAULT_FLOAT,
                "b": DEFAULT_FLOAT,
            }
        }

    RETURN_TYPES = ("BOOL",)
    FUNCTION = "op"
    CATEGORY = "Apt_Collect/math"

    def op(self, op: str, a: float, b: float) -> tuple[bool]:
        return (FLOAT_BINARY_CONDITIONS[op](a, b),)


class CIntUnaryOperation:
    @classmethod
    def INPUT_TYPES(cls) -> Mapping[str, Any]:
        return {
            "required": {"op": (list(INT_UNARY_OPERATIONS.keys()),), "a": DEFAULT_INT}
        }

    RETURN_TYPES = ("INT",)
    FUNCTION = "op"
    CATEGORY = "Apt_Collect/math"

    def op(self, op: str, a: int) -> tuple[int]:
        return (INT_UNARY_OPERATIONS[op](a),)


class CIntUnaryCondition:
    @classmethod
    def INPUT_TYPES(cls) -> Mapping[str, Any]:
        return {
            "required": {"op": (list(INT_UNARY_CONDITIONS.keys()),), "a": DEFAULT_INT}
        }

    RETURN_TYPES = ("BOOL",)
    FUNCTION = "op"
    CATEGORY = "Apt_Collect/math"

    def op(self, op: str, a: int) -> tuple[bool]:
        return (INT_UNARY_CONDITIONS[op](a),)


class CIntBinaryOperation:
    @classmethod
    def INPUT_TYPES(cls) -> Mapping[str, Any]:
        return {
            "required": {
                "op": (list(INT_BINARY_OPERATIONS.keys()),),
                "a": DEFAULT_INT,
                "b": DEFAULT_INT,
            }
        }

    RETURN_TYPES = ("INT","FLOAT")
    FUNCTION = "op"
    CATEGORY = "Apt_Collect/math"

    def op(self, op: str, a: int, b: int) -> tuple[int]:
        return (INT_BINARY_OPERATIONS[op](a, b),FLOAT_BINARY_OPERATIONS[op](a, b),)


class CIntBinaryCondition:
    @classmethod
    def INPUT_TYPES(cls) -> Mapping[str, Any]:
        return {
            "required": {
                "op": (list(INT_BINARY_CONDITIONS.keys()),),
                "a": DEFAULT_INT,
                "b": DEFAULT_INT,
            }
        }

    RETURN_TYPES = ("BOOL",)
    FUNCTION = "op"
    CATEGORY = "Apt_Collect/math"

    def op(self, op: str, a: int, b: int) -> tuple[bool]:
        return (INT_BINARY_CONDITIONS[op](a, b),)
    
    
    
class C_Remap_DataRange:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                
                "value": (ANY_TYPE,),
                "clamp": ("BOOLEAN", {"default": False}),
                "source_min": ("FLOAT", {"default": 0.0, "min": -999, "max": 999, "step": 0.01}),
                "source_max": ("FLOAT", {"default": 1.0, "min": -999, "max": 999, "step": 0.01}),
                "target_min": ("FLOAT", {"default": 0.0, "min": -999, "max": 999, "step": 0.01}),
                "target_max": ("FLOAT", {"default": 1.0, "min": -999, "max": 999, "step": 0.01}),
                "easing": (
                    [
                        "Linear",
                        "Sine In",
                        "Sine Out",
                        "Sine In+Out",
                        "Quart In",
                        "Quart Out",
                        "Quart In+Out",
                        "Cubic In",
                        "Cubic Out",
                        "Cubic In+Out",
                        "Circ In",
                        "Circ Out",
                        "Circ In+Out",
                        "Back In",
                        "Back Out",
                        "Back In+Out",
                        "Elastic In",
                        "Elastic Out",
                        "Elastic In+Out",
                        "Bounce In",
                        "Bounce Out",
                        "Bounce In+Out",
                    ],
                    {"default": "Linear"},
                ),
            },
            "optional": {

            },
        }

    FUNCTION = "set_range"
    RETURN_TYPES = ("FLOAT", "INT",)
    RETURN_NAMES = ("float", "int",)
    CATEGORY = "Apt_Collect/math"

    def set_range(
        self,
        clamp,
        source_min,
        source_max,
        target_min,
        target_max,
        easing,
        value,
    ):
        
        
        try:
            float_value = float(value)
        except ValueError:
            raise ValueError("Invalid value for conversion to float")
        
        if source_min == source_max:
            normalized_value = 0
        else:
            normalized_value = (float_value - source_min) / (source_max - source_min)
        if clamp:
            normalized_value = max(min(normalized_value, 1), 0)
        eased_value = apply_easing(normalized_value, easing)
        res_float = target_min + (target_max - target_min) * eased_value
        res_int = int(res_float)

        return (res_float, res_int)




class CreateRange:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "start": ("INT", {"default": 0, "min": -9007199254740991}),
                "stop": ("INT", {"default": 1, "min": -9007199254740991}),
                "step": ("INT", {"default": 1, "min": -9007199254740991}),
            },
        }
    
    TITLE = "Create Range"
    RETURN_TYPES = ("INT", "LIST", "INT")
    RETURN_NAMES = ("INT", "LIST", "length")
    OUTPUT_IS_LIST = (True, False, False, )
    FUNCTION = "run"
    CATEGORY = "Apt_Collect/math"

    def run(self, start: int, stop: int, step: int):
        range_list = list(range(start, stop, step))
        return (range_list, range_list, len(range_list))


class CreateArange:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "start": ("FLOAT", {"default": 0}),
                "stop": ("FLOAT", {"default": 1}),
                "step": ("FLOAT", {"default": 1}),
            },
        }
    
    TITLE = "Create Arange"
    RETURN_TYPES = ("FLOAT", "LIST", "INT")
    RETURN_NAMES = ("FLOAT", "LIST", "length")
    OUTPUT_IS_LIST = (True, False, False, )
    FUNCTION = "run"
    CATEGORY = "Apt_Collect/math"

    def run(self, start: float, stop: float, step: float):
        range_list = list(np.arange(start, stop, step))
        return (range_list, range_list, len(range_list))


class CreateLinspace:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "start": ("FLOAT", {"default": 0}),
                "stop": ("FLOAT", {"default": 1}),
                "num": ("INT", {"default": 10, "min": 2}),
            },
        }
    
    TITLE = "Create Linspace"
    RETURN_TYPES = ("FLOAT", "LIST", "INT")
    RETURN_NAMES = ("FLOAT", "LIST", "length")
    OUTPUT_IS_LIST = (True, False, False, )
    FUNCTION = "run"
    CATEGORY = "Apt_Collect/math"

    def run(self, start: float, stop: float, num: int):
        range_list = list(np.linspace(start, stop, num))
        return (range_list, range_list, len(range_list))


class Gradient_Float:

    @classmethod
    def INPUT_TYPES(s):
        # 定义缓动函数的可选项
        easing_types = [
            "Linear",  # 默认线性渐变
            "Sine In", "Sine Out", "Sine In/Out",
            "Quart In", "Quart Out", "Quart In/Out",
            "Cubic In", "Cubic Out", "Cubic In/Out",
            "Circ In", "Circ Out", "Circ In/Out",
            "Back In", "Back Out", "Back In/Out",
            "Elastic In", "Elastic Out", "Elastic In/Out",
            "Bounce In", "Bounce Out", "Bounce In/Out",
        ]
        
        return {
            "required": {
                "start_value": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 9999.0, "step": 0.001}),
                "end_value": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 9999.0, "step": 0.001}),
                "start_frame": ("INT", {"default": 0, "min": 0, "max": 9999, "step": 1}),
                "frame_duration": ("INT", {"default": 1, "min": 0, "max": 9999, "step": 1}),
                "current_frame": ("INT", {"default": 0, "min": 0, "max": 9999, "step": 1}),
                "gradient_profile": (easing_types, {"default": "Linear"}),  # 添加缓动函数选项
            },
        }
    
    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("FLOAT",)    
    FUNCTION = "gradient"
    CATEGORY = "Apt_Collect/math"

    def gradient(self, start_value, end_value, start_frame, frame_duration, current_frame, gradient_profile):
        if current_frame < start_frame:
            return (start_value,)

        if current_frame > start_frame + frame_duration:
            return (end_value,)
            
        # 计算当前进度（0 到 1 之间的值）
        progress = (current_frame - start_frame) / frame_duration
        
        # 应用缓动函数
        if gradient_profile != "Linear":
            progress = apply_easing(progress, gradient_profile)  # 调用缓动函数
        
        # 计算最终值
        float_out = start_value + (end_value - start_value) * progress
        
        return (float_out,)


class Increment_Float:

    @classmethod
    def INPUT_TYPES(s):
    
        return {"required": {"start_value": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 9999.0, "step": 0.001,}),
                             "step": ("FLOAT", {"default": 0.1, "min": -9999.0, "max": 9999.0, "step": 0.001,}),
                             "start_frame": ("INT", {"default": 0.0, "min": 0.0, "max": 9999.0, "step": 1.0,}),
                             "frame_duration": ("INT", {"default": 1.0, "min": 0.0, "max": 9999.0, "step": 1.0,}),
                             "current_frame": ("INT", {"default": 0.0, "min": 0.0, "max": 9999.0, "step": 1.0,}),
                },
        }
    
    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("FLOAT",)
    OUTPUT_NODE = True    
    FUNCTION = "increment"
    CATEGORY = "Apt_Collect/math"

    def increment(self, start_value, step, start_frame, frame_duration, current_frame):

        #print(f"current frame {current_frame}")
        if current_frame < start_frame:
            return (start_value,)

        current_value = start_value + (current_frame - start_frame) * step
        if current_frame <= start_frame + frame_duration:
            current_value += step
            #print(f"<current value {current_value}")    
            return (current_value, )
                
        return (current_value, )


class Increment_Integer:

    @classmethod
    def INPUT_TYPES(s):
    
        return {"required": {"start_value": ("INT", {"default": 1.0, "min": 0.0, "max": 9999.0, "step": 1.0,}),
                            "step": ("INT", {"default": 1.0, "min": -9999.0, "max": 9999.0, "step": 1.0,}),
                            "start_frame": ("INT", {"default": 0.0, "min": 0.0, "max": 9999.0, "step": 1.0,}),
                            "frame_duration": ("INT", {"default": 1.0, "min": 0.0, "max": 9999.0, "step": 1.0,}),
                            "current_frame": ("INT", {"default": 0.0, "min": 0.0, "max": 9999.0, "step": 1.0,}),
                },
        }
    
    RETURN_TYPES = ("INT", "STRING", )
    RETURN_NAMES = ("INT", "show_help", )
    OUTPUT_NODE = True    
    FUNCTION = "increment"
    CATEGORY = "Apt_Collect/math"

    def increment(self, start_value, step, start_frame, frame_duration, current_frame):

        #print(f"current frame {current_frame}")
        if current_frame < start_frame:
            return (start_value,)

        current_value = start_value + (current_frame - start_frame) * step
        if current_frame <= start_frame + frame_duration:
            current_value += step
            #print(f"<current value {current_value}")    
            return (current_value,)
                
        return (current_value, )


class Gradient_Integer:

    @classmethod
    def INPUT_TYPES(s):
        # 定义缓动函数的可选项
        easing_types = [
            "Linear",  # 默认线性渐变
            "Sine In", "Sine Out", "Sine In/Out",
            "Quart In", "Quart Out", "Quart In/Out",
            "Cubic In", "Cubic Out", "Cubic In/Out",
            "Circ In", "Circ Out", "Circ In/Out",
            "Back In", "Back Out", "Back In/Out",
            "Elastic In", "Elastic Out", "Elastic In/Out",
            "Bounce In", "Bounce Out", "Bounce In/Out",
        ]
        
        return {
            "required": {
                "start_value": ("INT", {"default": 1, "min": 0, "max": 9999, "step": 1}),
                "end_value": ("INT", {"default": 1, "min": 0, "max": 9999, "step": 1}),
                "start_frame": ("INT", {"default": 0, "min": 0, "max": 9999, "step": 1}),
                "frame_duration": ("INT", {"default": 1, "min": 0, "max": 9999, "step": 1}),
                "current_frame": ("INT", {"default": 0, "min": 0, "max": 9999, "step": 1}),
                "gradient_profile": (easing_types, {"default": "Linear"}),  # 添加缓动函数选项
            },
        }
    
    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("INT",)
    FUNCTION = "gradient"
    CATEGORY = "Apt_Collect/math"

    def gradient(self, start_value, end_value, start_frame, frame_duration, current_frame, gradient_profile):
        if current_frame < start_frame:
            return (start_value,)

        if current_frame > start_frame + frame_duration:
            return (end_value,)
            
        # 计算当前进度（0 到 1 之间的值）
        progress = (current_frame - start_frame) / frame_duration
        
        # 应用缓动函数
        if gradient_profile != "Linear":
            progress = apply_easing(progress, gradient_profile)  # 调用缓动函数
        
        # 计算最终值
        int_out = start_value + int((end_value - start_value) * progress)
        
        return (int_out,)