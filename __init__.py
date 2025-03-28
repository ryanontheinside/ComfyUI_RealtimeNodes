from .controls.value_controls import FloatControl, IntControl, StringControl
from .controls.sequence_controls import FloatSequence, IntSequence, StringSequence
from .controls.utility_controls import FPSMonitor, SimilarityFilter, LazyCondition
from .controls.state_management_controls import StateResetNode, StateTestNode
from .controls.motion_controls import MotionController, ROINode, IntegerMotionController
from .misc_nodes import (
    DTypeConverter,
    FastWebcamCapture,
    YOLOSimilarityCompare,
    TextRenderer,
    QuickShapeMask,
    MultilineText,
    LoadImageFromPath_
)
from .stream_sampler import  StreamBatchSampler, StreamScheduler
from .stream_cfg import StreamCFG
from .stream_conditioning import StreamConditioning
from .media_pipe_nodes import HandTrackingNode, HandMaskNode
from .controls.mask_controls import RepulsiveMaskNode, ResizeMaskNode

import re


NODE_CLASS_MAPPINGS = {
    "FloatControl": FloatControl,
    "IntControl": IntControl,
    "StringControl": StringControl,
    "FloatSequence": FloatSequence,
    "IntSequence": IntSequence,
    "StringSequence": StringSequence,
    "FPSMonitor": FPSMonitor,
    "SimilarityFilter": SimilarityFilter,
    "StreamCFG": StreamCFG,
    "StreamConditioning": StreamConditioning,
    "StreamBatchSampler": StreamBatchSampler,
    "StreamScheduler": StreamScheduler,
    "LazyCondition": LazyCondition,
    "MotionController": MotionController,
    "IntegerMotionController": IntegerMotionController,
    "YOLOSimilarityCompare": YOLOSimilarityCompare,
    "TextRenderer": TextRenderer,


    "ROINode": ROINode,

    #"IntervalControl": IntervalCo  ntrol,
    #"DeltaControl": DeltaControl,
    "QuickShapeMask": QuickShapeMask,
    "DTypeConverter": DTypeConverter,
    "FastWebcamCapture": FastWebcamCapture,
    "MultilineText": MultilineText,
    "LoadImageFromPath_": LoadImageFromPath_,
    "HandTrackingNode": HandTrackingNode,
    "HandMaskNode": HandMaskNode,
    #"RepulsiveMaskNode": RepulsiveMaskNode,
    "ResizeMaskNode": ResizeMaskNode,
    "StateResetNode": StateResetNode,
    "StateTestNode": StateTestNode,
}






NODE_DISPLAY_NAME_MAPPINGS = {}

suffix = " 🕒🅡🅣🅝"

for node_name in NODE_CLASS_MAPPINGS.keys():
    # Convert camelCase or snake_case to Title Case
    if node_name not in NODE_DISPLAY_NAME_MAPPINGS:
        display_name = ' '.join(word.capitalize() for word in re.findall(r'[A-Z]?[a-z]+|[A-Z]+(?=[A-Z][a-z]|\d|\W|$)|\d+', node_name))
    else:
        display_name = NODE_DISPLAY_NAME_MAPPINGS[node_name]
    
    # Add the suffix if it's not already present
    if not display_name.endswith(suffix):
        display_name += suffix
    
    # Assign the final display name to the mappings
    NODE_DISPLAY_NAME_MAPPINGS[node_name] = display_name


WEB_DIRECTORY = "./web/js"

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS'] 
