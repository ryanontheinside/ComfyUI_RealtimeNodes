from ....src.realtimenodes.control_base import ControlNodeBase
from ....src.realtimenodes.patterns.movement_patterns import MOVEMENT_PATTERNS

class ValueControlBase(ControlNodeBase):
    """Base class for float and integer control nodes"""
    
    @classmethod
    def INPUT_TYPES(cls):
        inputs = super().INPUT_TYPES()
        inputs["required"].update({
            "steps_per_cycle": ("INT", {
                "default": 30, 
                "min": 1, 
                "max": 1000, 
                "step": 1,
                "tooltip": "Number of steps to complete one full cycle"
            }),
            "movement_type": (list(MOVEMENT_PATTERNS.keys()), {
                "default": "sine",
                "tooltip": "Pattern of value changes over time"
            }),
        })
        return inputs

    def update(self, *args, **kwargs):
        """Implement abstract method from ControlNodeBase"""
        return self.update_value_base(*args, **kwargs)

    def update_value_base(self, maximum_value, minimum_value, starting_value, steps_per_cycle, movement_type, always_execute=True):
        state = self.get_state({
            "current_value": None,
            "phase": 0.0
        })
        
        # Initialize if this is the first run
        if state["current_value"] is None:
            state["current_value"] = starting_value
            self.set_state(state)
            return (starting_value,)

        # Update phase
        state["phase"] = (state["phase"] + 1/steps_per_cycle) % 1.0
        
        # Calculate new value using pattern
        pattern = MOVEMENT_PATTERNS[movement_type]
        new_value = pattern.calculate(state["phase"], minimum_value, maximum_value)
        
        # Update state
        state["current_value"] = new_value
        self.set_state(state)
        
        return (new_value,)

class FloatControl(ValueControlBase):
    
    DESCRIPTION = "Generates a floating point value that changes over time according to various patterns."

    @classmethod
    def INPUT_TYPES(cls):
        inputs = super().INPUT_TYPES()
        inputs["required"].update({
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
        return inputs

    RETURN_TYPES = ("FLOAT",)
    FUNCTION = "update_value"
    CATEGORY = "real-time/control/value"
    
    def update_value(self, maximum_value, minimum_value, starting_value, steps_per_cycle, movement_type, always_execute=True):
        return self.update_value_base(maximum_value, minimum_value, starting_value, steps_per_cycle, movement_type, always_execute)

class IntControl(ValueControlBase):    
    
    DESCRIPTION = "Generates an integer value that changes over time according to various patterns."

    @classmethod
    def INPUT_TYPES(cls):
        inputs = super().INPUT_TYPES()
        inputs["required"].update({
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
        return inputs

    RETURN_TYPES = ("INT",)
    FUNCTION = "update_value"
    CATEGORY = "real-time/control/value"
    
    def update_value(self, maximum_value, minimum_value, starting_value, steps_per_cycle, movement_type, always_execute=True):
        result = self.update_value_base(maximum_value, minimum_value, starting_value, steps_per_cycle, movement_type, always_execute)
        return (int(round(result[0])),)

class StringControl(ControlNodeBase):
    
    DESCRIPTION = "Generates string outputs that change over time according to various patterns."

    @classmethod
    def INPUT_TYPES(cls):
        inputs = super().INPUT_TYPES()
        inputs["required"].update({
            "strings": ("STRING", {
                "multiline": True,
                "default": "first string\nsecond string\nthird string",
                "tooltip": "List of strings to cycle through (one per line)"
            }),
            "steps_per_cycle": ("INT", {
                "default": 30, 
                "min": 1, 
                "max": 1000, 
                "step": 1,
                "tooltip": "Number of steps to complete one full cycle"
            }),
            "movement_type": (list(MOVEMENT_PATTERNS.keys()), {
                "default": "sine",
                "tooltip": "Pattern of value changes over time"
            }),
        })
        return inputs

    RETURN_TYPES = ("STRING",)
    FUNCTION = "update"
    CATEGORY = "real-time/control/value"
    
    def update(self, strings, steps_per_cycle, movement_type, always_execute=True):
        # Split the input strings into a list
        string_list = [s.strip() for s in strings.split('\n') if s.strip()]
        if not string_list:
            return ("",)
            
        state = self.get_state({
            "phase": 0.0
        })
        
        # Update phase
        state["phase"] = (state["phase"] + 1/steps_per_cycle) % 1.0
        
        # Calculate index using pattern
        pattern = MOVEMENT_PATTERNS[movement_type]
        index_float = pattern.calculate(state["phase"], 0, len(string_list) - 1)
        index = int(round(index_float)) % len(string_list)
        
        # Update state
        self.set_state(state)
        
        return (string_list[index],)