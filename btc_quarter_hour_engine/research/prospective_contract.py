from datetime import datetime, timezone


TRUSTED_MANIFEST_SHA256 = (
    "a52589a103a36a976adda49d65de38bee2c0ab656c2f4f13caa4c41067556d13"
)
RANGE_MODEL_SHA256 = (
    "db90d2eeb5b15a2a4b730cabcf8f9f2557e5816bcc02ed117a3acafc7a1f86e9"
)
FULL_V2_MODEL_SHA256 = (
    "c8d2bf2289faf567896d1f514a3daf5d8c17d7a8b6cacddc1a700cb517d47fed"
)
WINDOW_START = datetime(2026, 11, 15, tzinfo=timezone.utc)
WINDOW_END_EXCLUSIVE = datetime(2026, 12, 15, tzinfo=timezone.utc)
PREDICTION_DEADLINE_SECONDS = 60
PROBABILITY_THRESHOLD = 0.5
BASELINE_PREDICTION = 0
