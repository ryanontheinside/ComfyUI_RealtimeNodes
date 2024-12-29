from .controls.value_controls import FloatControl, IntControl, StringControl
from .controls.sequence_controls import FloatSequence, IntSequence, StringSequence
from .controls.utility_controls import FPSMonitor
from .controls.detector_controls import IntDetectionControl,FloatDetectionControl,StringDetectionControl
from .detectors.detector_nodes import MotionDetectorNode, RegionOfInterest, BrightnessDetectorNode
from .quick_shape_mask import QuickShapeMask
from .tiny_vae import TAESDVaeEncode, TAESDVaeDecode
from .misc_nodes import DTypeConverter

NODE_CLASS_MAPPINGS = {
    "FloatControl": FloatControl,
    "IntControl": IntControl,
    "StringControl": StringControl,
    
    
    "FloatSequence": FloatSequence,
    "IntSequence": IntSequence,
    "StringSequence": StringSequence,
    
    "FPSMonitor": FPSMonitor,
    "QuickShapeMask": QuickShapeMask,

    "MotionDetectorNode": MotionDetectorNode,
    "BrightnessDetectorNode": BrightnessDetectorNode,
    "RegionOfInterest": RegionOfInterest,
    "IntDetectionControl": IntDetectionControl,
    "FloatDetectionControl": FloatDetectionControl,
    "StringDetectionControl": StringDetectionControl,
    
    
    "TAESDVaeEncode": TAESDVaeEncode,
    "TAESDVaeDecode": TAESDVaeDecode,
    "DTypeConverter": DTypeConverter
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FloatControl": "Float Control (RyanOnTheInside)",
    "IntControl": "Int Control (RyanOnTheInside)",
    "StringControl": "String Control (RyanOnTheInside)",
    "FloatSequence": "Float Sequence (RyanOnTheInside)",
    "IntSequence": "Int Sequence (RyanOnTheInside)",
    "StringSequence": "String Sequence (RyanOnTheInside)",
    "FPSMonitor": "FPS Monitor (RyanOnTheInside)",
    "MotionDetectorNode": "Motion Detector (RyanOnTheInside)",
    "BrightnessDetectorNode": "Brightness Detector (RyanOnTheInside)",
    "RegionOfInterest": "Region of Interest (RyanOnTheInside)",
    "IntDetectionControl": "Detector Int Control (RyanOnTheInside)",
    "FloatDetectionControl": "Detector Float Control (RyanOnTheInside)",
    "StringDetectionControl": "Detector String Control (RyanOnTheInside)",
    "QuickShapeMask": "Quick Shape Mask (RyanOnTheInside)",
    "TAESDVaeEncode": "TAESD VAE Encode (RyanOnTheInside)",
    "TAESDVaeDecode": "TAESD VAE Decode (RyanOnTheInside)",
    "DTypeConverter": "DType Converter (RyanOnTheInside)"
}

WEB_DIRECTORY = "./web/js"

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS'] 