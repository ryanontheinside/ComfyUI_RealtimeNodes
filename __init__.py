from .controls.value_controls import FloatControl, IntControl, StringControl
from .controls.sequence_controls import FloatSequence, IntSequence, StringSequence
from .controls.utility_controls import FPSMonitor
from .quick_shape_mask import QuickShapeMask
from .tiny_vae import TAESDVaeEncode, TAESDVaeDecode

NODE_CLASS_MAPPINGS = {
    "FloatControl": FloatControl,
    "IntControl": IntControl,
    "StringControl": StringControl,
    "FloatSequence": FloatSequence,
    "IntSequence": IntSequence,
    "StringSequence": StringSequence,
    "FPSMonitor": FPSMonitor,
    #"IntervalControl": IntervalCo  ntrol,
    #"DeltaControl": DeltaControl,
    "QuickShapeMask": QuickShapeMask,
    "TAESDVaeEncode": TAESDVaeEncode,
    "TAESDVaeDecode": TAESDVaeDecode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FloatControl": "Float Control (RyanOnTheInside)",
    "IntControl": "Int Control (RyanOnTheInside)",
    "StringControl": "String Control (RyanOnTheInside)",
    "FloatSequence": "Float Sequence (RyanOnTheInside)",
    "IntSequence": "Int Sequence (RyanOnTheInside)",
    "StringSequence": "String Sequence (RyanOnTheInside)",
    "FPSMonitor": "FPS Monitor (RyanOnTheInside)",
    #"IntervalControl": "Interval Control (RyanOnTheInside)",
    #"DeltaControl": "Delta Control (RyanOnTheInside)",
    "QuickShapeMask": "Quick Shape Mask (RyanOnTheInside)",
    "TAESDVaeEncode": "TAESD VAE Encode (RyanOnTheInside)",
    "TAESDVaeDecode": "TAESD VAE Decode (RyanOnTheInside)"
}

WEB_DIRECTORY = "./web/js"

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS'] 