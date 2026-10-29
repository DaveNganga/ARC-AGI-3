from __future__ import annotations

from typing import Protocol

import numpy as np

from ..compiler import Encoding
from ..models import ActionScore, SearchResult


class SearchBackend(Protocol):
    name: str

    def select(
        self,
        encoding: Encoding,
        action_scores: list[ActionScore],
        costs: np.ndarray,
        valid: np.ndarray,
    ) -> SearchResult: ...

