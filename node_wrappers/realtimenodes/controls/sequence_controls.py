from ....src.realtimenodes.control_base import ControlNodeBase


class SequenceControlBase(ControlNodeBase):
    """Base class for sequence-based controls"""

    @classmethod
    def INPUT_TYPES(cls):
        inputs = super().INPUT_TYPES()
        inputs["required"].update(
            {
                "steps_per_item": (
                    "INT",
                    {"default": 30, "min": 1, "max": 1000, "step": 1, "tooltip": "Number of frames to show each item"},
                ),
                "sequence_mode": (
                    ["forward", "reverse", "pingpong", "random"],
                    {"default": "forward", "tooltip": "How to move through the sequence"},
                ),
                "batch_size": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 1000,
                        "step": 1,
                        "tooltip": "Number of values to generate in batch",
                    },
                ),
            }
        )
        return inputs

    def update_sequence_base(self, values: list, steps_per_item: int, sequence_mode: str, batch_size: int = 1, always_execute: bool = True):
        # Define default state
        default_state = {
            "current_index": 0,
            "step_counter": 0,
            "direction": 1,  # 1 for forward, -1 for reverse (used in pingpong)
        }
        
        # Initialize or get state - make sure to use the default state
        state = self.get_state(default_state)

        if not values:
            if batch_size > 1:
                return ([None] * batch_size,)
            return (None,)

        # Fast path for batch_size=1 (real-time case)
        if batch_size == 1:
            # Update step counter (ensure it exists first)
            if "step_counter" not in state:
                state["step_counter"] = 0
            state["step_counter"] += 1

            # Check if we should move to next item
            if state["step_counter"] >= steps_per_item:
                state["step_counter"] = 0

                # Ensure other state values exist
                if "current_index" not in state:
                    state["current_index"] = 0
                if "direction" not in state:
                    state["direction"] = 1

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
        
        # Batch processing path
        # Update step counter (ensure it exists first)
        if "step_counter" not in state:
            state["step_counter"] = 0
        state["step_counter"] += 1

        # Ensure other state values exist
        if "current_index" not in state:
            state["current_index"] = 0
        if "direction" not in state:
            state["direction"] = 1

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

        # Generate batch of values from current position
        batch_values = []
        for i in range(batch_size):
            if sequence_mode == "forward":
                idx = (state["current_index"] + i) % len(values)
            elif sequence_mode == "reverse":
                idx = (state["current_index"] - i) % len(values)
            elif sequence_mode == "pingpong":
                # For pingpong, we need to bounce when we hit the ends
                direction = state["direction"]
                idx = state["current_index"]
                for j in range(i):
                    idx += direction
                    if idx >= len(values) or idx < 0:
                        direction *= -1
                        idx += 2 * direction  # Move two steps to correct direction
                idx = max(0, min(idx, len(values) - 1))  # Safety bounds check
            elif sequence_mode == "random":
                import random
                idx = random.randint(0, len(values) - 1)
            
            batch_values.append(values[idx])
        
        return (batch_values,)


class FloatSequence(SequenceControlBase):
    """Cycles through a sequence of float values"""

    RETURN_TYPES = ("FLOAT",)
    FUNCTION = "update"
    CATEGORY = "Realtime Nodes/control/sequence"

    @classmethod
    def INPUT_TYPES(cls):
        inputs = super().INPUT_TYPES()
        inputs["required"].update({"values": ("STRING", {"default": "0.0, 0.5, 1.0", "tooltip": "List of float values (comma separated)"})})
        return inputs

    def update(self, values: str, steps_per_item: int, sequence_mode: str, batch_size: int = 1, always_execute: bool = True):
        # Parse comma-separated float values
        float_values = [float(v.strip()) for v in values.split(",") if v.strip()]
        return self.update_sequence_base(float_values, steps_per_item, sequence_mode, batch_size, always_execute)


class IntSequence(SequenceControlBase):
    """Cycles through a sequence of integer values"""

    RETURN_TYPES = ("INT",)
    FUNCTION = "update"
    CATEGORY = "Realtime Nodes/control/sequence"

    @classmethod
    def INPUT_TYPES(cls):
        inputs = super().INPUT_TYPES()
        inputs["required"].update({"values": ("STRING", {"default": "0, 5, 10", "tooltip": "List of integer values (comma separated)"})})
        return inputs

    def update(self, values: str, steps_per_item: int, sequence_mode: str, batch_size: int = 1, always_execute: bool = True):
        # Parse comma-separated integer values
        int_values = [int(v.strip()) for v in values.split(",") if v.strip()]
        return self.update_sequence_base(int_values, steps_per_item, sequence_mode, batch_size, always_execute)


class StringSequence(SequenceControlBase):
    """Cycles through a sequence of strings"""

    RETURN_TYPES = ("STRING",)
    FUNCTION = "update"
    CATEGORY = "Realtime Nodes/control/sequence"

    @classmethod
    def INPUT_TYPES(cls):
        inputs = super().INPUT_TYPES()
        inputs["required"].update(
            {
                "values": (
                    "STRING",
                    {"multiline": True, "default": "first\nsecond\nthird", "tooltip": "List of strings (one per line)"},
                )
            }
        )
        return inputs

    def update(self, values: str, steps_per_item: int, sequence_mode: str, batch_size: int = 1, always_execute: bool = True):
        # Split into string values
        string_values = [v.strip() for v in values.split("\n") if v.strip()]
        return self.update_sequence_base(string_values, steps_per_item, sequence_mode, batch_size, always_execute)
