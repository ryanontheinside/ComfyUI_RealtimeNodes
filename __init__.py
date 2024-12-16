from .control_nodes import FloatControl, IntControl
from .quick_shape_mask import QuickShapeMask

NODE_CLASS_MAPPINGS = {
    "FloatControl": FloatControl,
    "IntControl": IntControl,
    "QuickShapeMask": QuickShapeMask
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FloatControl": "Float Control (RyanOnTheInside)",
    "IntControl": "Int Control (RyanOnTheInside)"
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS'] 