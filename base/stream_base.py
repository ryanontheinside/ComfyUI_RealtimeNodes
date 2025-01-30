from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Any, Optional
import threading
from dataclasses import dataclass, field
from collections import defaultdict
from .control_base import ControlNodeBase

@dataclass
class FeatureSpec:
    """Specification for a stream feature"""
    type: str  # e.g. "FLOAT", "TENSOR", "INT"
    shape: tuple = field(default_factory=tuple)  # () for scalar, (N,) for vector, etc.
    range: tuple = (0.0, 1.0)  # min/max range for the feature
    description: str = ""

class StreamDataProcessor(ABC):
    """Base class for processing any type of streaming data"""
    
    def __init__(self):
        self._feature_cache = {}
        self._cache_lock = threading.Lock()
        self._registered_features = self.get_feature_specs()
    
    @abstractmethod
    def get_feature_specs(self) -> Dict[str, FeatureSpec]:
        """Return specifications for all available features"""
        pass
    
    @abstractmethod
    def process_buffer(self, buffer: Any) -> Dict[str, Any]:
        """Process incoming buffer and return computed features"""
        pass
    
    def get_feature(self, name: str) -> Optional[Any]:
        """Get a feature value from cache"""
        with self._cache_lock:
            return self._feature_cache.get(name)
    
    def update_cache(self, features: Dict[str, Any]):
        """Update feature cache with new values"""
        with self._cache_lock:
            self._feature_cache.update(features)

class StreamProcessorRegistry:
    """Global registry for stream processors"""
    _processors: Dict[str, StreamDataProcessor] = {}
    _feature_cache: Dict[str, Dict[str, Any]] = defaultdict(dict)
    _lock = threading.Lock()
    
    @classmethod
    def register_processor(cls, name: str, processor: StreamDataProcessor):
        with cls._lock:
            cls._processors[name] = processor
    
    @classmethod
    def get_processor(cls, name: str) -> Optional[StreamDataProcessor]:
        with cls._lock:
            return cls._processors.get(name)
    
    @classmethod
    def get_feature(cls, processor_name: str, feature_name: str) -> Optional[Any]:
        with cls._lock:
            return cls._feature_cache[processor_name].get(feature_name)
    
    @classmethod
    def update_features(cls, processor_name: str, features: Dict[str, Any]):
        with cls._lock:
            cls._feature_cache[processor_name].update(features)

class StreamControl(ControlNodeBase):
    """Base class for nodes that react to streaming data"""
    
    def __init__(self):
        super().__init__()
        self._last_update = 0.0
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                **super().INPUT_TYPES()["required"],
                "processor": ("STRING", {
                    "default": "audio",
                    "tooltip": "Type of stream processor to use"
                }),
                "feature": ("STRING", {
                    "default": "rms",
                    "tooltip": "Name of the feature to use"
                }),
                "smoothing": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Smoothing factor for feature values"
                })
            }
        }
    
    def get_feature_value(self, processor_name: str, feature_name: str, 
                         default: Any = 0.0) -> Any:
        """Get current value for a feature"""
        value = StreamProcessorRegistry.get_feature(processor_name, feature_name)
        return default if value is None else value
    
    def smooth_value(self, current: float, target: float, 
                    smoothing: float) -> float:
        """Apply smoothing to a value"""
        return current * smoothing + target * (1 - smoothing) 