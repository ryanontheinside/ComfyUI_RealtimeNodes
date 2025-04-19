from ....src.realtimenodes.control_base import ControlNodeBase

class SequenceControlBase(ControlNodeBase):
    """Base class for sequence-based controls"""
    
    @classmethod
    def INPUT_TYPES(cls):
        inputs = super().INPUT_TYPES()
        inputs["required"].update({
            "steps_per_item": ("INT", {
                "default": 30,
                "min": 1,
                "max": 1000,
                "step": 1,
                "tooltip": "Number of frames to show each item"
            }),
            "sequence_mode": (["forward", "reverse", "pingpong", "random"], {
                "default": "forward",
                "tooltip": "How to move through the sequence"
            })
        })
        return inputs

    def update_sequence_base(self, values: list, steps_per_item: int, sequence_mode: str, always_execute: bool = True):
        # Initialize or get state
        state = self.get_state({
            "current_index": 0,
            "step_counter": 0,
            "direction": 1  # 1 for forward, -1 for reverse (used in pingpong)
        })
        
        if not values:
            return (values[0] if values else None,)
            
        # Update step counter
        state["step_counter"] += 1
        
        # Check if we should move to next item
        if state["step_counter"] >= steps_per_item:
            state["step_counter"] = 0
            
            # Update index based on sequence mode
            if sequence_mode == "forward":
                state["current_index"] = (state["current_index"] + 1) % len(values)
            elif sequence_mode == "reverse":
                state["current_index"] = (state["current_index"] - 1) % len(values)
            elif sequence_mode == "pingpong":
                next_index = state["current_index"] + state["direction"]
                if next_index >= len(values) or next_index < 0:
                    state["direction"] *= -1  # Reverse direction
                    next_index = state["current_index"] + state["direction"]
                state["current_index"] = next_index
            elif sequence_mode == "random":
                import random
                state["current_index"] = random.randint(0, len(values) - 1)
        
        # Save state
        self.set_state(state)
        
        # Return current value
        return (values[state["current_index"]],)

class FloatSequence(SequenceControlBase):
    """Cycles through a sequence of float values"""
    
    RETURN_TYPES = ("FLOAT",)
    FUNCTION = "update"
    CATEGORY = "real-time/control/sequence"

    @classmethod
    def INPUT_TYPES(cls):
        inputs = super().INPUT_TYPES()
        inputs["required"].update({
            "values": ("STRING", {
                "default": "0.0, 0.5, 1.0",
                "tooltip": "List of float values (comma separated)"
            })
        })
        return inputs
    
    def update(self, values: str, steps_per_item: int, sequence_mode: str, always_execute: bool = True):
        # Parse comma-separated float values
        float_values = [float(v.strip()) for v in values.split(',') if v.strip()]
        return self.update_sequence_base(float_values, steps_per_item, sequence_mode, always_execute)

class IntSequence(SequenceControlBase):
    """Cycles through a sequence of integer values"""
    
    RETURN_TYPES = ("INT",)
    FUNCTION = "update"
    CATEGORY = "real-time/control/sequence"

    @classmethod
    def INPUT_TYPES(cls):
        inputs = super().INPUT_TYPES()
        inputs["required"].update({
            "values": ("STRING", {
                "default": "0, 5, 10",
                "tooltip": "List of integer values (comma separated)"
            })
        })
        return inputs
    
    def update(self, values: str, steps_per_item: int, sequence_mode: str, always_execute: bool = True):
        # Parse comma-separated integer values
        int_values = [int(v.strip()) for v in values.split(',') if v.strip()]
        return self.update_sequence_base(int_values, steps_per_item, sequence_mode, always_execute)

class StringSequence(SequenceControlBase):
    """Cycles through a sequence of strings"""
    
    RETURN_TYPES = ("STRING",)
    FUNCTION = "update"
    CATEGORY = "real-time/control/sequence"

    @classmethod
    def INPUT_TYPES(cls):
        inputs = super().INPUT_TYPES()
        inputs["required"].update({
            "values": ("STRING", {
                "multiline": True,
                "default": "first\nsecond\nthird",
                "tooltip": "List of strings (one per line)"
            })
        })
        return inputs
    
    def update(self, values: str, steps_per_item: int, sequence_mode: str, always_execute: bool = True):
        # Split into string values
        string_values = [v.strip() for v in values.split('\n') if v.strip()]
        return self.update_sequence_base(string_values, steps_per_item, sequence_mode, always_execute) 