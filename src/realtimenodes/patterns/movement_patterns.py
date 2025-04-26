import math
import random

from ..control_base import Pattern


class StaticPattern(Pattern):
    def get_name(self) -> str:
        return "static"
        
    def calculate(self, phase, min_val, max_val):
        return min_val

class SinePattern(Pattern):
    def get_name(self) -> str:
        return "sine"
        
    def calculate(self, phase, min_val, max_val):
        amplitude = (max_val - min_val) / 2
        center = min_val + amplitude
        return center + amplitude * math.sin(2 * math.pi * phase)

class TrianglePattern(Pattern):
    def get_name(self) -> str:
        return "triangle"
        
    def calculate(self, phase, min_val, max_val):
        if phase < 0.5:
            return min_val + (max_val - min_val) * (2 * phase)
        return max_val - (max_val - min_val) * (2 * (phase - 0.5))

class SawtoothPattern(Pattern):
    def get_name(self) -> str:
        return "sawtooth"
        
    def calculate(self, phase, min_val, max_val):
        return min_val + (max_val - min_val) * phase

class SquarePattern(Pattern):
    def get_name(self) -> str:
        return "square"
        
    def calculate(self, phase, min_val, max_val):
        return max_val if phase < 0.5 else min_val

class BouncePattern(Pattern):
    def get_name(self) -> str:
        return "bounce"
        
    def calculate(self, phase, min_val, max_val):
        bounce = abs(math.sin(math.pi * phase))
        return min_val + (max_val - min_val) * bounce

class ExponentialPattern(Pattern):
    def get_name(self) -> str:
        return "exponential"
        
    def calculate(self, phase, min_val, max_val):
        exp_phase = math.exp(4 * phase) - 1
        exp_max = math.exp(4) - 1
        return min_val + (max_val - min_val) * (exp_phase / exp_max)

class LogarithmicPattern(Pattern):
    def get_name(self) -> str:
        return "logarithmic"
        
    def calculate(self, phase, min_val, max_val):
        log_phase = math.log(1 + 99 * phase) / math.log(100)
        return min_val + (max_val - min_val) * log_phase

class PulsePattern(Pattern):
    def get_name(self) -> str:
        return "pulse"
        
    def calculate(self, phase, min_val, max_val):
        pulse = math.exp(-10 * ((phase - 0.5) ** 2))
        return min_val + (max_val - min_val) * pulse

class RandomWalkPattern(Pattern):
    def get_name(self) -> str:
        return "random_walk"
        
    def __init__(self):
        self.last_delta = 0
        self.current_pos = 0.5

    def calculate(self, phase, min_val, max_val):
        momentum = 0.7
        random_component = (2 * random.random() - 1) * 0.1
        self.last_delta = momentum * self.last_delta + random_component
        
        self.current_pos += self.last_delta
        self.current_pos = max(0, min(1, self.current_pos))
        
        return min_val + (max_val - min_val) * self.current_pos

class SmoothNoisePattern(Pattern):
    def get_name(self) -> str:
        return "smooth_noise"
        
    def calculate(self, phase, min_val, max_val):
        t = phase * 10  # Scale phase for more interesting variation
        noise = (math.sin(t) + math.sin(2.2*t + 5.52) + math.sin(3.6*t + 4.12)) / 3
        return min_val + (max_val - min_val) * (noise + 1) / 2

# Pattern registry
MOVEMENT_PATTERNS = {
    pattern().get_name(): pattern() for pattern in [
        StaticPattern,
        SinePattern,
        TrianglePattern,
        SawtoothPattern,
        SquarePattern,
        BouncePattern,
        ExponentialPattern,
        LogarithmicPattern,
        PulsePattern,
        RandomWalkPattern,
        SmoothNoisePattern
    ]
} 