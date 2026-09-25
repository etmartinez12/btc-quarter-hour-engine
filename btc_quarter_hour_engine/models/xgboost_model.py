from __future__ import annotations

from xgboost import XGBClassifier


class XGBoostBaselineModel:
    def __init__(
        self,
        n_estimators: int = 250,
        learning_rate: float = 0.05,
        max_depth: int = 4,
        random_state: int = 42,
    ) -> None:
        self.model = XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=1,
            verbosity=0,
        )

    def fit(self, X, y) -> "XGBoostBaselineModel":
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)
