import time
from abc import ABC, abstractmethod


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
        
    def clear_state(self, node_id: str) -> bool:
        """Clear state for a specific node. Returns True if state was found and cleared."""
        if node_id in self._states:
            del self._states[node_id]
            return True
        return False
        
    def clear_all_states(self):
        """Clear all states"""
        self._states.clear()

    def get_all_node_ids(self):
        """Get list of all node IDs with state"""
        return list(self._states.keys())
        
    def cleanup_orphaned_states(self, active_node_ids: set[str]):
        """Remove states for nodes that no longer exist"""
        orphaned = [nid for nid in self._states.keys() if nid not in active_node_ids]
        for nid in orphaned:
            del self._states[nid]
        return len(orphaned)

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
                    "default": True,
                    "tooltip": "When enabled, the node updates every execution"
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

    def cleanup(self):
        """Clean up node state when node is destroyed"""
        self.state_manager.clear_state(self.node_id)

    def __del__(self):
        self.cleanup()

    @abstractmethod
    def update(self, *args, **kwargs):
        """Main update method to be implemented by control nodes"""
        pass