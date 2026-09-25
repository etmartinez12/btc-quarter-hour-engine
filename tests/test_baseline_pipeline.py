import json

import pytest

from btc_quarter_hour_engine.config import BaselineConfig, ValidationConfig
from btc_quarter_hour_engine.run_baseline import _to_builtin, run_walk_forward_benchmark


def test_run_walk_forward_benchmark_raises_when_no_splits_are_possible():
    config = BaselineConfig(
        validation=ValidationConfig(initial_train_size=10_000, test_size=16, step_size=16)
    )

    with pytest.raises(ValueError, match="zero folds"):
        run_walk_forward_benchmark(config)


def test_walk_forward_benchmark_summary_is_json_serializable():
    summary = run_walk_forward_benchmark()

    serialized = json.dumps(_to_builtin(summary))

    assert "\"models\"" in serialized
