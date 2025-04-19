import time

class TimestampProvider:
    """
    Provides real-time, monotonic timestamps in milliseconds.

    MediaPipe's RunningMode.VIDEO requires strictly increasing timestamp_ms values.
    Using time.time() alone can result in duplicates during tight loops, which causes
    MediaPipe to skip or freeze frames. This class ensures timestamps are always 
    increasing while staying aligned with real elapsed time.
    """
    def __init__(self):
        self._start_time = time.time()
        self._last_timestamp = 0

    def next(self) -> int:
        now = int((time.time() - self._start_time) * 1000)
        if now <= self._last_timestamp:
            now = self._last_timestamp + 1
        self._last_timestamp = now
        return now 