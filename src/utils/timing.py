"""
Timing and timestamp utilities for RealTimeNodes.
"""

import time

class TimestampProvider:
    """
    Provides real-time, monotonic timestamps in milliseconds.

    This ensures timestamps are always increasing while staying aligned with real elapsed time.
    Useful for any process that requires monotonically increasing time values.
    """
    def __init__(self):
        self._start_time = time.time()
        self._last_timestamp = 0

    def next(self) -> int:
        """Get the next timestamp in milliseconds, guaranteed to be greater than the previous one."""
        now = int((time.time() - self._start_time) * 1000)
        if now <= self._last_timestamp:
            now = self._last_timestamp + 1
        self._last_timestamp = now
        return now 