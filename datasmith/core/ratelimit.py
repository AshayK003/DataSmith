"""Sliding window rate limiter — thread-safe, in-memory.

Usage:
    from datasmith.core.ratelimit import RateLimiter

    limiter = RateLimiter(max_requests=10, window_seconds=60)
    allowed, remaining = limiter.check("session-abc")
    if not allowed:
        logger.debug("rate limited")
"""

import logging
import time
import threading
from collections import defaultdict

logger = logging.getLogger(__name__)


class RateLimiter:
    """Sliding window rate limiter per key.

    Tracks request timestamps per key in a sliding window. Old entries
    outside the window are pruned on each check. Thread-safe via RLock.
    """

    def __init__(self, max_requests: int = 10, window_seconds: int = 60,
                 max_keys: int = 10_000):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.max_keys = max_keys
        self._windows: dict[str, list[float]] = defaultdict(list)
        self._lock = threading.RLock()

    def check(self, key: str) -> tuple[bool, int]:
        """Check if a request is allowed for *key*.

        Returns (allowed, remaining):
          allowed   — True if under limit
          remaining — number of requests still available in this window
        """
        now = time.time()
        cutoff = now - self.window_seconds
        with self._lock:
            self._evict_expired(now, cutoff)
            window = self._windows.get(key)
            if window is None:
                # Fresh key
                self._windows[key] = [now]
                return True, self.max_requests - 1

            # Prune expired entries
            fresh = [t for t in window if t > cutoff]
            self._windows[key] = fresh

            remaining = self.max_requests - len(fresh)
            if remaining <= 0:
                return False, 0

            fresh.append(now)
            self._windows[key] = fresh
            return True, remaining - 1

    def remaining(self, key: str) -> int:
        """Return how many requests *key* can still make in the current window."""
        now = time.time()
        cutoff = now - self.window_seconds
        with self._lock:
            window = self._windows.get(key)
            if window is None:
                return self.max_requests
            fresh = [t for t in window if t > cutoff]
            return max(0, self.max_requests - len(fresh))

    def _evict_expired(self, now: float, cutoff: float) -> None:
        """Drop fully-expired keys; cap total keys to bound memory."""
        expired = [k for k, ts in self._windows.items()
                   if not ts or max(ts) <= cutoff]
        for k in expired:
            del self._windows[k]
        # Leave room for the key about to be inserted.
        while len(self._windows) >= self.max_keys:
            self._windows.pop(next(iter(self._windows)))

    def reset(self, key: str) -> None:
        """Clear rate limit state for *key*."""
        with self._lock:
            self._windows.pop(key, None)

    @property
    def active_keys(self) -> int:
        """Number of distinct keys currently tracked (debug/metrics)."""
        with self._lock:
            return len(self._windows)
