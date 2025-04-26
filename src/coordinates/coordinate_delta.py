import logging
from typing import Optional
import math
import collections
import typing # Add typing for Dict hint

#TODO: consider moving base nodes to src

# Import from the new consolidated utilities

logger = logging.getLogger(__name__)

MAX_POSITION_HISTORY = 50 # Matches MAX_LANDMARK_HISTORY for consistency
FLOAT_EQUALITY_TOLERANCE = 1e-6 # Same tolerance as base_delta_nodes


# --- Base Position Delta Logic (Handling Multiple Histories) ---
class BaseCoordinateDelta:
    """Base class for nodes calculating delta from position history.
       Handles single floats or lists of coordinates.
    """
    FUNCTION = "execute"

    def __init__(self):
        # Store history deques keyed by item index from input list
        # Key 0 is used for single float inputs
        self._histories: typing.Dict[int, collections.deque] = {}

    def _get_or_create_history(self, item_index: int) -> collections.deque:
        """Gets the deque for a specific index, creating it if needed."""
        if item_index not in self._histories:
            self._histories[item_index] = collections.deque(maxlen=MAX_POSITION_HISTORY)
        return self._histories[item_index]

    def _get_position_delta(self, x: float, y: float, z: float, window_size: int, item_index: int = 0) -> Optional[float]:
        """Calculates delta using the history deque for the given item_index."""
        history_deque = self._get_or_create_history(item_index)
        
        window_size = max(1, min(window_size, MAX_POSITION_HISTORY - 1))
        current_pos = (x, y, z)
        history_deque.append(current_pos)
        
        if len(history_deque) > window_size:
            pos_now = history_deque[-1]
            pos_past = history_deque[-(window_size + 1)] # Correct indexing
            if pos_now is None or pos_past is None:
                 logger.warning(f"{self.__class__.__name__}[{item_index}]: Missing position data in history window.")
                 return None
            delta = math.sqrt((pos_now[0] - pos_past[0])**2 +
                              (pos_now[1] - pos_past[1])**2 +
                              (pos_now[2] - pos_past[2])**2)
            return delta
        else:
            # logger.debug(f"{self.__class__.__name__}[{item_index}]: Not enough history ({len(history_deque)}/{window_size+1}).")
            return None # Not enough history yet
            
    # Remove _handle_input_coords, logic moved to execute
