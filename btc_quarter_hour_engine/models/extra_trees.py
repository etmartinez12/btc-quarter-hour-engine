from __future__ import annotations

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


class ExtraTreesBaselineModel:
    def __init__(
        self,
        n_estimators: int = 300,
        max_depth: int | None = 6,
        random_state: int = 42,
    ) -> None:
        self.pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    ExtraTreesClassifier(
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        random_state=random_state,
                    ),
                ),
            ]
        )

    def fit(self, X, y) -> "ExtraTreesBaselineModel":
        self.pipeline.fit(X, y)
        return self

    def predict(self, X):
        return self.pipeline.predict(X)

    def predict_proba(self, X):
        return self.pipeline.predict_proba(X)
