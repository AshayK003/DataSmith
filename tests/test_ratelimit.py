"""Tests for the sliding window rate limiter."""

import time
from collections import defaultdict

import pytest

from datasmith.core.ratelimit import RateLimiter


class TestRateLimiter:
    def test_init(self):
        """Rate limiter initializes with defaults."""
        rl = RateLimiter(max_requests=5, window_seconds=10)
        assert rl.max_requests == 5
        assert rl.window_seconds == 10

    def test_allows_first_request(self):
        """First request is always allowed."""
        rl = RateLimiter(max_requests=3, window_seconds=60)
        allowed, remaining = rl.check("test-key")
        assert allowed is True
        assert remaining == 2  # 3 - 1 = 2

    def test_allows_up_to_limit(self):
        """All requests within the limit are allowed."""
        rl = RateLimiter(max_requests=3, window_seconds=60)
        for i in range(3):
            allowed, remaining = rl.check("test-key")
            assert allowed is True, f"Request {i} should be allowed"

    def test_blocks_over_limit(self):
        """Request beyond the limit is blocked."""
        rl = RateLimiter(max_requests=3, window_seconds=60)
        for i in range(3):
            rl.check("test-key")
        allowed, remaining = rl.check("test-key")
        assert allowed is False
        assert remaining == 0

    def test_different_keys_independent(self):
        """Rate limits are per key."""
        rl = RateLimiter(max_requests=2, window_seconds=60)
        rl.check("key-a")
        rl.check("key-a")
        allowed, remaining = rl.check("key-b")  # fresh key
        assert allowed is True

    def test_window_expires(self):
        """Old entries outside the window don't count."""
        rl = RateLimiter(max_requests=2, window_seconds=1)
        rl.check("test-key")
        rl.check("test-key")
        allowed, _ = rl.check("test-key")
        assert allowed is False  # window not expired yet
        time.sleep(1.1)
        allowed, _ = rl.check("test-key")
        assert allowed is True  # old entries expired

    def test_remaining_counts_correctly(self):
        """remaining() reports available requests."""
        rl = RateLimiter(max_requests=5, window_seconds=60)
        assert rl.remaining("test-key") == 5
        rl.check("test-key")
        assert rl.remaining("test-key") == 4
        rl.check("test-key")
        assert rl.remaining("test-key") == 3

    def test_reset(self):
        """Reset clears the rate limit for a key."""
        rl = RateLimiter(max_requests=2, window_seconds=60)
        rl.check("test-key")
        rl.check("test-key")
        assert rl.remaining("test-key") == 0
        rl.reset("test-key")
        assert rl.remaining("test-key") == 2

    def test_active_keys(self):
        """active_keys tracks distinct keys."""
        rl = RateLimiter(max_requests=5, window_seconds=60)
        rl.check("key-a")
        rl.check("key-b")
        assert rl.active_keys == 2

    def test_thread_safety(self):
        """Concurrent checks don't race."""
        import concurrent.futures

        rl = RateLimiter(max_requests=50, window_seconds=10)
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as exe:
            futures = [exe.submit(rl.check, "shared-key") for _ in range(50)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        allowed = sum(1 for a, _ in results if a)
        assert allowed == 50, f"Expected 50, got {allowed} (some requests were lost)"

        # Now one more should be blocked
        allowed, _ = rl.check("shared-key")
        assert allowed is False
