from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class FeatureConfig:
    momentum_windows_minutes: tuple[int, ...] = (1, 2, 3, 5, 10, 15, 30, 60)
    volatility_windows_minutes: tuple[int, ...] = (5, 15, 30, 60)
    volume_windows_minutes: tuple[int, ...] = (1, 5, 15, 30, 60)
    range_windows_minutes: tuple[int, ...] = (15, 30, 60)
    vwap_windows_minutes: tuple[int, ...] = (5, 15, 30, 60)
    regime_windows_minutes: tuple[int, ...] = (60, 240, 1440)
    microstructure_windows_minutes: tuple[int, ...] = (5, 15, 30)
    rolling_extrema_window_minutes: int = 60
    volume_zscore_window_minutes: int = 60
    ema_fast_window_minutes: int = 15
    ema_slow_window_minutes: int = 60


@dataclass(slots=True)
class ValidationConfig:
    initial_train_size: int = 64
    test_size: int = 16
    step_size: int = 16


@dataclass(slots=True)
class BaselineConfig:
    boundary_freq: str = "15min"
    feature: FeatureConfig = field(default_factory=FeatureConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    logistic_max_iter: int = 1000
    extra_trees_n_estimators: int = 300
    extra_trees_max_depth: int | None = 6
    lightgbm_n_estimators: int = 200
    lightgbm_learning_rate: float = 0.05
    lightgbm_num_leaves: int = 31
    xgboost_n_estimators: int = 250
    xgboost_learning_rate: float = 0.05
    xgboost_max_depth: int = 4
    random_state: int = 42
