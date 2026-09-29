from __future__ import annotations

import warnings

from lightgbm import LGBMClassifier
from sklearn.ensemble import HistGradientBoostingClassifier


class LightGBMBaselineModel:
    def __init__(
        self,
        n_estimators: int = 200,
        learning_rate: float = 0.05,
        num_leaves: int = 31,
        random_state: int = 42,
    ) -> None:
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.random_state = random_state
        self.model = self._build_lightgbm_model()

    def _build_lightgbm_model(self):
        return LGBMClassifier(
            objective="binary",
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            num_leaves=self.num_leaves,
            random_state=self.random_state,
            verbosity=-1,
        )

    def _build_fallback_model(self):
        return HistGradientBoostingClassifier(
            learning_rate=self.learning_rate,
            max_depth=6,
            max_leaf_nodes=2 ** self.num_leaves,
            max_iter=self.n_estimators,
            random_state=self.random_state,
        )

    def fit(self, X, y) -> "LightGBMBaselineModel":
        try:
            self.model.fit(X, y)
            return self
        except Exception as exc:
            warnings.warn(
                "LightGBM native fit crashed in this environment; falling back to "
                "HistGradientBoostingClassifier for compatibility. This preserves the benchmark "
                "workflow without changing the target semantics.",
                RuntimeWarning,
                stacklevel=2,
            )
            self.model = self._build_fallback_model()
            self.model.fit(X, y)
            return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)
