import time
import threading


class TTLCache:
    """
    Simple in-memory TTL cache shared across the engine's tick loop.

    Purpose: multiple modes (scalp/day/swing) often need the same
    (symbol, timeframe) data within the same tick. Without this cache each
    mode would trigger its own exchange API call for the same data,
    multiplying rate-limit usage. Callers fetch through get_or_fetch() with
    a cache key; the underlying fetch only runs once per TTL window.
    """

    def __init__(self, ttl_seconds=30):
        self.ttl_seconds = ttl_seconds
        self._store = {}
        self._lock = threading.Lock()

    def get_or_fetch(self, key, fetch_fn):
        now = time.time()
        with self._lock:
            entry = self._store.get(key)
            if entry is not None:
                value, expires_at = entry
                if now < expires_at:
                    return value

        # Run the actual fetch outside the lock so one slow network call
        # doesn't block lookups for other cache keys.
        value = fetch_fn()

        with self._lock:
            self._store[key] = (value, now + self.ttl_seconds)

        return value

    def invalidate(self, key=None):
        with self._lock:
            if key is None:
                self._store.clear()
            else:
                self._store.pop(key, None)
