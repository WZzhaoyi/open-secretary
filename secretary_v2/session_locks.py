"""Per-conversation serialization for history reads, runs, and compaction."""

import asyncio
from weakref import WeakValueDictionary


_locks: WeakValueDictionary[str, asyncio.Lock] = WeakValueDictionary()


def get_session_lock(session_key: str) -> asyncio.Lock:
    """Return the shared lock for one conversation within this event loop."""
    lock = _locks.get(session_key)
    if lock is None:
        lock = asyncio.Lock()
        _locks[session_key] = lock
    return lock
