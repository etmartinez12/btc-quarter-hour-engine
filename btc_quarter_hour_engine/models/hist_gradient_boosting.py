from __future__ import annotations

from sklearn.ensemble import HistGradientBoostingClassifier


class HistGradientBoostingBaselineModel:
    def __init__(
        self,
        max_iter: int = 100,
        learning_rate: float = 0.1,
        max_leaf_nodes: int = 15,
        min_samples_leaf: int = 20,
        l2_regularization: float = 1.0,
        random_state: int = 42,
    ) -> None:
        self.model = HistGradientBoostingClassifier(
            max_iter=max_iter,
            learning_rate=learning_rate,
            max_leaf_nodes=max_leaf_nodes,
            min_samples_leaf=min_samples_leaf,
            l2_regularization=l2_regularization,
            random_state=random_state,
            early_stopping=False,
        )

    def fit(self, X, y) -> "HistGradientBoostingBaselineModel":
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)
