from __future__ import annotations

import time
from typing import Callable, TypeVar


T = TypeVar("T")


def call_with_retries(
    fn: Callable[[], T],
    attempts: int = 3,
    base_delay: float = 1.0,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[T, int]:
    """Call ``fn`` until it succeeds, backing off exponentially; return the result and retry count.

    Use only for idempotent work such as model requests. Environment steps must not be retried
    because the action may already have been applied.
    """
    for attempt in range(attempts):
        try:
            return fn(), attempt
        except Exception:
            if attempt == attempts - 1:
                raise
            sleep(base_delay * (2 ** attempt))
    raise AssertionError("unreachable")
