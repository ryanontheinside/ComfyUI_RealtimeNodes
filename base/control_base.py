from abc import ABC, abstractmethod
import time

class Pattern(ABC):
    """Base class for all patterns (movement, timing, etc)"""
    @abstractmethod
    def calculate(self, phase: float, min_val: float, max_val: float) -> float:
        """Calculate pattern value at given phase (0-1)"""
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Pattern name for UI"""
        pass

class StateManager:
    """Handles persistent state between executions"""
    def __init__(self):
        self._states = {}
    
    def get_state(self, node_id: str, default=None):
        return self._states.get(node_id, default)
    
    def set_state(self, node_id: str, state):
        self._states[node_id] = state

class ControlNodeBase(ABC):
    """Base class for all control nodes"""
    
    # Shared state manager
    state_manager = StateManager()
    
    @classmethod
    def INPUT_TYPES(cls):
        """Base input types common to all control nodes"""
        return {
            "required": {
                "always_execute": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "When enabled, the node updates every execution. This is primarily for use INSIDE of ComfyUI. Generally, you should set this to FALSE for real-time applications, like ComfyStream."
                }),
            },
        }
    
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        if kwargs.get('always_execute', True):
            return float(time.time())
        return False
    
    def __init__(self):
        self.node_id = str(id(self))
        
    def get_state(self, default=None):
        return self.state_manager.get_state(self.node_id, default)
        
    def set_state(self, state):
        self.state_manager.set_state(self.node_id, state)

    @abstractmethod
    def update(self, *args, **kwargs):
        """Main update method to be implemented by control nodes"""
        pass