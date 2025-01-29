from .controls.value_controls import FloatControl, IntControl, StringControl
from .controls.sequence_controls import FloatSequence, IntSequence, StringSequence
from .controls.utility_controls import FPSMonitor, SimilarityFilter, LazyCondition
from .controls.motion_controls import MotionController, ROINode, IntegerMotionController
from .quick_shape_mask import QuickShapeMask
from .misc_nodes import DTypeConverter, FastWebcamCapture

NODE_CLASS_MAPPINGS = {
    "FloatControl": FloatControl,
    "IntControl": IntControl,
    "StringControl": StringControl,
    "FloatSequence": FloatSequence,
    "IntSequence": IntSequence,
    "StringSequence": StringSequence,
    "FPSMonitor": FPSMonitor,
    "SimilarityFilter": SimilarityFilter,
    "LazyCondition": LazyCondition,
    "MotionController": MotionController,
    "IntegerMotionController": IntegerMotionController,




    "ROINode": ROINode,
    #"IntervalControl": IntervalCo  ntrol,
    #"DeltaControl": DeltaControl,
    "QuickShapeMask": QuickShapeMask,
    "DTypeConverter": DTypeConverter,
    "FastWebcamCapture": FastWebcamCapture,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FloatControl": "Float Control (RyanOnTheInside)",
    "IntControl": "Int Control (RyanOnTheInside)",
    "StringControl": "String Control (RyanOnTheInside)",
    "FloatSequence": "Float Sequence (RyanOnTheInside)",
    "IntSequence": "Int Sequence (RyanOnTheInside)",
    "StringSequence": "String Sequence (RyanOnTheInside)",
    "FPSMonitor": "FPS Monitor (RyanOnTheInside)",
    "MotionController": "Float Motion Controller (RyanOnTheInside)",
    "ROINode": "ROI Node (RyanOnTheInside)",
    "IntegerMotionController": "Integer Motion Controller (RyanOnTheInside)",
    #"IntervalControl": "Interval Control (RyanOnTheInside)",
    #"DeltaControl": "Delta Control (RyanOnTheInside)",
    "QuickShapeMask": "Quick Shape Mask (RyanOnTheInside)",
    "TAESDVaeEncode": "TAESD VAE Encode (RyanOnTheInside)",
    "TAESDVaeDecode": "TAESD VAE Decode (RyanOnTheInside)",
    "DTypeConverter": "DType Converter (RyanOnTheInside)",
    "FastWebcamCapture": "Fast Webcam Capture (RyanOnTheInside)",
    "SimilarityFilter": "Similarity Filter (RyanOnTheInside)",
    "LazyCondition": "Lazy Condition (RyanOnTheInside)",


}


WEB_DIRECTORY = "./web/js"

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS'] 