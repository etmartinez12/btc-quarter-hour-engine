import pandas as pd

from btc_quarter_hour_engine.targets import build_direction_target


def test_build_direction_target_labels_up_down_and_flat_moves():
    boundaries = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2026-01-01T00:00:00Z",
                    "2026-01-01T00:15:00Z",
                    "2026-01-01T00:30:00Z",
                    "2026-01-01T00:45:00Z",
                ],
                utc=True,
            ),
            "midpoint": [100.0, 101.0, 99.0, 99.0],
        }
    )

    actual = build_direction_target(boundaries)

    assert actual.loc[0, "target"] == 1
    assert actual.loc[1, "target"] == 0
    assert pd.isna(actual.loc[2, "target"])
    assert pd.isna(actual.loc[3, "target"])
