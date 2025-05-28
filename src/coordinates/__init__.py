"""
Coordinates module for ComfyUI RealTimeNodes.

This package provides unified coordinate system handling for ComfyUI RealTimeNodes,
allowing seamless conversion between different coordinate spaces (normalized, pixel)
with support for single values, batches, and point collections.
"""

from .coordinate_system import CoordinateSystem, Point, PointList
from .drawing_engine import DrawingEngine
from . import coordinate_utils
from . import drawing_utils

__all__ = ["CoordinateSystem", "Point", "PointList", "DrawingEngine", "coordinate_utils", "drawing_utils"]
