import math
import time

class ValueControlBase:
    """Base class for float and integer control nodes"""
    # Class level storage for persistent values between executions
    instances = {}
    @classmethod
    def INPUT_TYPES(cls):
        """Base input types that are common to both Float and Int controls"""
        return {
            "required": {
                "steps_per_cycle": ("INT", {
                    "default": 30, 
                    "min": 1, 
                    "max": 1000, 
                    "step": 1,
                    "tooltip": "Number of steps to complete one full cycle of the movement pattern"
                }),
                "movement_type": ([
                    "static", 
                    "sine_wave", 
                    "triangle_wave", 
                    "sawtooth",
                    "square_wave",
                    "bounce",
                    "exponential",
                    "logarithmic",
                    "pulse",
                    "random_walk",
                    "smooth_noise"
                ], {
                    "default": "sine_wave",
                    "tooltip": "Pattern of value changes over time:\nstatic: No change\nsine_wave: Smooth sinusoidal pattern\ntriangle_wave: Linear up and down\nsawtooth: Linear up, instant reset\nsquare_wave: Instant switch between min/max\nbounce: Bouncing ball effect\nexponential: Accelerating increase\nlogarithmic: Decelerating increase\npulse: Single peak per cycle\nrandom_walk: Brownian motion\nsmooth_noise: Smooth random pattern"
                }),
                "always_execute": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "When enabled, the node updates every execution. When disabled, only updates when inputs change"
                }),
            },
        }
    
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        if kwargs.get('always_execute', True):
            return float(time.time())  # Return unique timestamp when always_execute is True
        return False  # Otherwise return consistent value
    
    def __init__(self):
        self.instance_id = str(id(self))
        if self.instance_id not in self.__class__.instances:
            self.__class__.instances[self.instance_id] = {
                'current_value': None,
                'iteration': 0 
            }

    def update_value_base(self, maximum_value, minimum_value, starting_value, steps_per_cycle, movement_type, always_execute=True):
        instance = self.__class__.instances[self.instance_id]
        # Initialize if this is the first run
        if instance['current_value'] is None:
            instance['current_value'] = starting_value
            instance['iteration'] = 0
            return (starting_value,)

        # Increment iteration counter
        instance['iteration'] += 1
        current_iteration = instance['iteration']
        
        # Calculate phase (0 to 1)
        phase = (current_iteration % steps_per_cycle) / steps_per_cycle

        # Update value based on movement type
        if movement_type == "static":
            new_value = instance['current_value']
        
        elif movement_type == "sine_wave":
            amplitude = (maximum_value - minimum_value) / 2
            center = minimum_value + amplitude
            new_value = center + amplitude * math.sin(2 * math.pi * phase)
        
        elif movement_type == "triangle_wave":
            if phase < 0.5:
                new_value = minimum_value + (maximum_value - minimum_value) * (2 * phase)
            else:
                new_value = maximum_value - (maximum_value - minimum_value) * (2 * (phase - 0.5))
        
        elif movement_type == "sawtooth":
            new_value = minimum_value + (maximum_value - minimum_value) * phase

        elif movement_type == "square_wave":
            new_value = maximum_value if phase < 0.5 else minimum_value

        elif movement_type == "bounce":
            bounce = abs(math.sin(math.pi * phase))
            new_value = minimum_value + (maximum_value - minimum_value) * bounce

        elif movement_type == "exponential":
            exp_phase = math.exp(4 * phase) - 1
            exp_max = math.exp(4) - 1
            new_value = minimum_value + (maximum_value - minimum_value) * (exp_phase / exp_max)

        elif movement_type == "logarithmic":
            log_phase = math.log(1 + 99 * phase) / math.log(100)
            new_value = minimum_value + (maximum_value - minimum_value) * log_phase

        elif movement_type == "pulse":
            pulse = math.exp(-10 * ((phase - 0.5) ** 2))
            new_value = minimum_value + (maximum_value - minimum_value) * pulse

        elif movement_type == "random_walk":
            if not hasattr(instance, 'last_delta'):
                instance['last_delta'] = 0
            
            momentum = 0.7
            random_component = (2 * (hash(str(current_iteration)) / 2**32) - 1) * 0.1
            instance['last_delta'] = momentum * instance['last_delta'] + random_component
            
            current_pos = (instance['current_value'] - minimum_value) / (maximum_value - minimum_value)
            new_pos = current_pos + instance['last_delta']
            new_pos = max(0, min(1, new_pos))  # Clamp between 0 and 1
            new_value = minimum_value + (maximum_value - minimum_value) * new_pos

        elif movement_type == "smooth_noise":
            t = current_iteration * 0.1
            noise = (math.sin(t) + math.sin(2.2*t + 5.52) + math.sin(3.6*t + 4.12)) / 3
            new_value = minimum_value + (maximum_value - minimum_value) * (noise + 1) / 2

        # Clamp value to min/max range
        new_value = max(minimum_value, min(maximum_value, new_value))
        instance['current_value'] = new_value
        
        return (new_value,)

class FloatControl(ValueControlBase):
    """A control node that outputs a floating point value that can change over time."""
    
    DESCRIPTION = "Generates a floating point value that changes over time according to various patterns. Useful for animating parameters or creating dynamic effects."

    @classmethod
    def INPUT_TYPES(cls):
        input_types = super().INPUT_TYPES()
        input_types["required"].update({
            "maximum_value": ("FLOAT", {
                "default": 1.0, 
                "min": -10000.0, 
                "max": 10000.0, 
                "step": 0.1,
                "tooltip": "Maximum value that can be output"
            }),
            "minimum_value": ("FLOAT", {
                "default": 0.0, 
                "min": -10000.0, 
                "max": 10000.0, 
                "step": 0.1,
                "tooltip": "Minimum value that can be output"
            }),
            "starting_value": ("FLOAT", {
                "default": 0.5, 
                "min": -10000.0, 
                "max": 10000.0, 
                "step": 0.1,
                "tooltip": "Initial value when the node first executes"
            }),
        })
        return input_types

    RETURN_TYPES = ("FLOAT",)
    FUNCTION = "update_value"
    CATEGORY = "control"
    
    def update_value(self, maximum_value, minimum_value, starting_value, steps_per_cycle, movement_type, always_execute=True):
        return self.update_value_base(maximum_value, minimum_value, starting_value, steps_per_cycle, movement_type, always_execute)

class IntControl(ValueControlBase):
    """A control node that outputs an integer value that can change over time."""
    
    DESCRIPTION = "Generates an integer value that changes over time according to various patterns. Useful for animating discrete parameters or creating stepped animations."

    @classmethod
    def INPUT_TYPES(cls):
        input_types = super().INPUT_TYPES()
        input_types["required"].update({
            "maximum_value": ("INT", {
                "default": 100, 
                "min": -10000, 
                "max": 10000, 
                "step": 1,
                "tooltip": "Maximum value that can be output"
            }),
            "minimum_value": ("INT", {
                "default": 0, 
                "min": -10000, 
                "max": 10000, 
                "step": 1,
                "tooltip": "Minimum value that can be output"
            }),
            "starting_value": ("INT", {
                "default": 50, 
                "min": -10000, 
                "max": 10000, 
                "step": 1,
                "tooltip": "Initial value when the node first executes"
            }),
        })
        return input_types

    RETURN_TYPES = ("INT",)
    FUNCTION = "update_value"
    CATEGORY = "control"
    
    def update_value(self, maximum_value, minimum_value, starting_value, steps_per_cycle, movement_type, always_execute=True):
        result = self.update_value_base(maximum_value, minimum_value, starting_value, steps_per_cycle, movement_type, always_execute)
        return (int(round(result[0])),)