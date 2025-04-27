import numpy as np


class MaskControlMixin:
    """Mixin providing common mask functionality"""

    def create_circle_mask(self, height, width, center_y, center_x, size):
        """Create a circular mask with anti-aliased edges"""
        y, x = np.ogrid[:height, :width]
        radius = size * min(height, width) / 2
        dist = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
        mask = np.clip(radius + 1 - dist, 0, 1).astype(np.float32)
        return mask

    def get_initial_state(self, x_pos, y_pos, size, min_size=None, max_size=None):
        """Get initial state dictionary"""
        state = {"x": x_pos, "y": y_pos, "size": size, "initialized": True}
        if min_size is not None:
            state["min_size"] = min_size
        if max_size is not None:
            state["max_size"] = max_size
        return state
