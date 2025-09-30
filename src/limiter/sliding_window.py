# rate_limiter.py
import time
import threading
import random
from collections import deque
from typing import Callable, Optional

class SlidingWindowRateLimiter:
    """
    Thread-safe sliding-window rate limiter.
    - Enforces at most `rpm * safety_ratio` calls per `window` seconds.
    - Uses monotonic clock to avoid wall-clock jumps.
    """
    def __init__(
        self,
        rpm: int,
        window: float = 60.0,
        safety_ratio: float = 0.9,
        min_sleep: float = 0.1,
        max_sleep: float = 1.0,
        jitter: float = 0.0,
        time_fn: Callable[[], float] = time.monotonic,
        sleep_fn: Callable[[float], None] = time.sleep,
    ):
        if rpm <= 0:
            raise ValueError("rpm must be > 0")
        if not (0.0 < safety_ratio <= 1.0):
            raise ValueError("safety_ratio must be in (0, 1]")
        self.window = float(window)
        self.capacity = max(1, int(rpm * safety_ratio))  # e.g., rpm=10 -> 9 if safety_ratio=0.9
        self.min_sleep = float(min_sleep)
        self.max_sleep = float(max_sleep)
        self.jitter = float(jitter)
        self._time = time_fn
        self._sleep = sleep_fn

        self._lock = threading.Lock()
        self._calls = deque()  # store float timestamps

    def update_limit(self, rpm: int, safety_ratio: Optional[float] = None):
        """Dynamically update RPM and optional safety ratio."""
        if rpm <= 0:
            raise ValueError("rpm must be > 0")
        with self._lock:
            if safety_ratio is not None:
                if not (0.0 < safety_ratio <= 1.0):
                    raise ValueError("safety_ratio must be in (0, 1]")
                self.capacity = max(1, int(rpm * safety_ratio))
            else:
                # keep current ratio by inferring from current capacity if possible
                # (conservative: set capacity=rpm if ratio not provided)
                self.capacity = max(1, int(rpm))

    def _purge_expired(self, now: float):
        """Remove timestamps older than `window` seconds."""
        while self._calls and ( (now - self._calls[0]) >= self.window ):
            self._calls.popleft()

    def acquire(self):
        """Block until a slot is available, then record the call."""
        while True:
            now = self._time()
            with self._lock:
                self._purge_expired(now)

                if len(self._calls) < self.capacity:
                    self._calls.append(now)
                    return

                # Compute how long until the oldest entry exits the window
                wait = self.window - (now - self._calls[0])

            # Clamp sleep and add optional jitter to reduce sync-thundering
            sleep_for = min(max(wait, self.min_sleep), self.max_sleep)
            if self.jitter > 0.0:
                sleep_for += random.uniform(0.0, self.jitter)
            self._sleep(sleep_for)
