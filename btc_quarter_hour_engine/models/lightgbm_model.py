from __future__ import annotations

from lightgbm import LGBMClassifier


class LightGBMBaselineModel:
    def __init__(
        self,
        n_estimators: int = 200,
        learning_rate: float = 0.05,
        num_leaves: int = 31,
        random_state: int = 42,
    ) -> None:
        self.model = LGBMClassifier(
            objective="binary",
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            random_state=random_state,
            verbosity=-1,
        )

    def fit(self, X, y) -> "LightGBMBaselineModel":
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)
