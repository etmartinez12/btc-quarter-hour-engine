from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
import pandas as pd


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

        if step < self.test_size:
            raise ValueError("step_size must be greater than or equal to test_size to prevent overlapping OOF test windows")

        train_end = self.initial_train_size

        while train_end + self.test_size <= n_samples:
            train_idx = np.arange(0, train_end)
            test_idx = np.arange(train_end, train_end + self.test_size)
            yield train_idx, test_idx
            train_end += step

    def get_n_splits(self, X) -> int:
        return sum(1 for _ in self.split(X))


@dataclass(slots=True)
class ExpandingTimeWindowSplit:
    initial_train_period: str | pd.Timedelta = "365D"
    test_period: str | pd.Timedelta = "30D"
    step_period: str | pd.Timedelta = "30D"

    @property
    def initial_window(self) -> str | pd.Timedelta:
        return self.initial_train_period

    @property
    def test_window(self) -> str | pd.Timedelta:
        return self.test_period

    @property
    def step_window(self) -> str | pd.Timedelta:
        return self.step_period

    @classmethod
    def from_config(cls, config) -> "ExpandingTimeWindowSplit":
        return cls(
            initial_train_period=getattr(config, "initial_train_period", "365D"),
            test_period=getattr(config, "test_period", "30D"),
            step_period=getattr(config, "step_period", "30D"),
        )

    def _as_timedelta(self, value: str | pd.Timedelta) -> pd.Timedelta:
        if isinstance(value, pd.Timedelta):
            td = value
        else:
            td = pd.to_timedelta(value)
        if td <= pd.Timedelta(0):
            raise ValueError("time windows must be positive")
        return td

    def split(self, timestamps) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        if timestamps is None or len(timestamps) == 0:
            return

        series = pd.Series(pd.to_datetime(timestamps, utc=True))
        if series.isna().any():
            raise ValueError("timestamps contain invalid values")
        if not series.is_monotonic_increasing:
            raise ValueError("timestamps must be sorted chronologically")

        initial = self._as_timedelta(self.initial_train_period)
        test_period = self._as_timedelta(self.test_period)
        step_period = self._as_timedelta(self.step_period)
        if step_period < test_period:
            raise ValueError("step_period must be greater than or equal to test_period to prevent overlapping OOF test windows")

        first_ts = series.iloc[0]
        last_ts = series.iloc[-1]
        train_end = first_ts + initial
        while train_end + test_period <= last_ts + pd.Timedelta(0):
            train_mask = series < train_end
            test_mask = (series >= train_end) & (series < train_end + test_period)
            if not train_mask.any() or not test_mask.any():
                break
            train_idx = np.flatnonzero(train_mask.to_numpy())
            test_idx = np.flatnonzero(test_mask.to_numpy())
            yield train_idx, test_idx
            train_end += step_period

    def get_n_splits(self, timestamps) -> int:
        return sum(1 for _ in self.split(timestamps))
