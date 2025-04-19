from ....src.realtimenodes.control_base import ControlNodeBase

class StateResetNode(ControlNodeBase):
    """Node that resets all control node states when triggered"""
    
    OUTPUT_NODE = True
    
    @classmethod
    def INPUT_TYPES(cls):
        inputs = super().INPUT_TYPES()
        inputs["required"].update({
            "trigger": ("BOOLEAN", {
                "default": False,
                "tooltip": "Set to True to reset all states"
            })
        })
        return inputs

    RETURN_TYPES = ("BOOLEAN",)
    FUNCTION = "update"
    CATEGORY = "Realtime Nodes/control/utility"

    def update(self, trigger, always_execute=True):
        print(f"\n=== StateResetNode UPDATE - node_id: {self.node_id} ===")
        print(f"States before potential reset: {self.state_manager._states}")
        if trigger:
            print("RESETTING ALL STATES")
            self.state_manager.clear_all_states()
            print(f"States after reset: {self.state_manager._states}")
            return (True,)
        return (False,)

class StateTestNode(ControlNodeBase):
    """Simple node that maintains a counter to test state management"""
    
    @classmethod
    def INPUT_TYPES(cls):
        inputs = super().INPUT_TYPES()
        inputs["required"].update({
            "increment": ("INT", {
                "default": 1,
                "min": 1,
                "max": 100,
                "step": 1,
                "tooltip": "Amount to increment counter by"
            })
        })
        return inputs

    RETURN_TYPES = ("INT",)
    FUNCTION = "update"
    CATEGORY = "Realtime Nodes/control/utility"

    def update(self, increment, always_execute=True):
        print(f"\n=== StateTestNode UPDATE - node_id: {self.node_id} ===")
        print(f"All states before get: {self.state_manager._states}")
        
        state = self.get_state({
            "counter": 0
        })
        print(f"Retrieved state: {state}")
        
        state["counter"] += increment
        print(f"Updated state before save: {state}")
        
        self.set_state(state)
        print(f"All states after save: {self.state_manager._states}")
        
        return (state["counter"],)

