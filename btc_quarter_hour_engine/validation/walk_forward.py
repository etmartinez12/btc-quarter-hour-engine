from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np


@dataclass(slots=True)
class ExpandingWindowSplit:
    initial_train_size: int
    test_size: int
    step_size: int | None = None

    def split(self, X) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        n_samples = len(X)
        if self.initial_train_size <= 0:
            raise ValueError("initial_train_size must be positive")
        if self.test_size <= 0:
            raise ValueError("test_size must be positive")
        if self.step_size is None:
            step = self.test_size
        elif self.step_size <= 0:
            raise ValueError("step_size must be positive when provided")
        else:
            step = self.step_size
        train_end = self.initial_train_size

        while train_end + self.test_size <= n_samples:
            train_idx = np.arange(0, train_end)
            test_idx = np.arange(train_end, train_end + self.test_size)
            yield train_idx, test_idx
            train_end += step

    def get_n_splits(self, X) -> int:
        return sum(1 for _ in self.split(X))
