from ....src.realtimenodes.control_base import ControlNodeBase
import copy
from ....src.utils.general_utils import AlwaysEqualProxy

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
        if trigger:
            self.state_manager.clear_all_states()
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
        state = self.get_state({
            "counter": 0
        })
        
        state["counter"] += increment

        self.set_state(state)
        
        return (state["counter"],)



class GetStateNode(ControlNodeBase):
    """
    Node that retrieves a value from the global state using a user-specified key.
    """
    CATEGORY = "utils"
    RETURN_TYPES = (AlwaysEqualProxy("*"),)
    RETURN_NAMES = ("value",)
    FUNCTION = "update"
    DESCRIPTION = "Retrieve a value from the global state using the given key. If the key is not found, return the default value."
    @classmethod
    def INPUT_TYPES(cls):
        inputs = super().INPUT_TYPES()
        inputs["required"].update({
            "key": ("STRING", {"default": "default_key", "tooltip": "The key to retrieve the value from. If not provided, the default value will be returned."}),
            "default_value": (AlwaysEqualProxy("*"), {"tooltip": "The value to return if the key is not found."}),
            "use_default": ("BOOLEAN", {"default": False, "tooltip": "If True, the default value will be returned if the key is not found."}),
        })
        return inputs
    
    def update(self, key: str, default_value, use_default: bool, always_execute=True):
        """
        Retrieve a value from the global state using the given key.
        """
        if not key or use_default:
            return (default_value,)
        
        # Get the shared state dictionary
        shared_state = self.state_manager.get_state("__shared_keys__", {})
        
        # Check if the key exists
        if key in shared_state:
            return (shared_state[key],)
        
        # Return default value if key not found
        return (default_value,)
    
class SetStateNode(ControlNodeBase):
    """
    Node that stores a value in the global state with a user-specified key.
    The value will be accessible in future workflow runs through GetStateNode.
    """
    CATEGORY = "utils"
    RETURN_TYPES = (AlwaysEqualProxy("*"),)
    RETURN_NAMES = ("value",)
    FUNCTION = "update"
    OUTPUT_NODE = True
    DESCRIPTION = "Store a value in the global state with the given key. The value will be accessible in future workflow runs through GetStateNode."
    @classmethod
    def INPUT_TYPES(cls):
        inputs = super().INPUT_TYPES()
        inputs["required"].update({
            "key": ("STRING", {"default": "default_key", "tooltip": "The key to store the value under. If not provided, the value will not be stored."}),
            "value": (AlwaysEqualProxy("*"), {"tooltip": "The value to store in the global state."}),
        })
        return inputs
    
    def update(self, key: str, value, always_execute=True):
        """
        Store a value in the global state with the given key.
        """
        if not key:
            return (value,)
        
        try:
            shared_state = self.state_manager.get_state("__shared_keys__", {})
            shared_state[key] = copy.deepcopy(value)
            self.state_manager.set_state("__shared_keys__", shared_state)
        except Exception as e:
            print(f"[State Node] Error storing value: {str(e)}")
        
        return (value,)

