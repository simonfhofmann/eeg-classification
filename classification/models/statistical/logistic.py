# ----------------------------------------------------------------------
# logistic.py
#
# Logistic Regression classifier for EEG classification.
# ----------------------------------------------------------------------

import numpy as np
from typing import Dict, Any, Optional
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from models.base import BaseClassifier


class LogisticClassifier(BaseClassifier):
    """
    Logistic Regression classifier for EEG data.

    A simple but effective baseline classifier that:
    - Provides probability estimates
    - Supports L1/L2 regularization
    - Scales well with number of features
    """

    def __init__(
        self,
        C: float = 1.0,
        penalty: str = "l2",
        solver: str = "lbfgs",
        max_iter: int = 1000,
        class_weight: Optional[str] = "balanced",
        scale_features: bool = True,
        name: str = "LogisticRegression",
        **kwargs
    ):
        """
        Initialize Logistic Regression classifier.

        Args:
            C: Inverse regularization strength
            penalty: Regularization type ('l1', 'l2', 'elasticnet', 'none')
            solver: Optimization algorithm
            max_iter: Maximum iterations
            class_weight: Class weighting strategy
            scale_features: Whether to standardize features
            name: Model name
        """
        super().__init__(name=name, C=C, penalty=penalty, **kwargs)

        self.C = C
        self.penalty = penalty
        self.solver = solver
        self.max_iter = max_iter
        self.class_weight = class_weight
        self.scale_features = scale_features

        self._build_pipeline()

    def _build_pipeline(self):
        """Build sklearn pipeline."""
        steps = []

        if self.scale_features:
            steps.append(('scaler', StandardScaler()))

        lr = LogisticRegression(
            C=self.C,
            penalty=self.penalty,
            solver=self.solver,
            max_iter=self.max_iter,
            class_weight=self.class_weight,
            n_jobs=-1
        )
        steps.append(('lr', lr))

        self._model = Pipeline(steps)

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> 'LogisticClassifier':
        """Train the classifier."""
        if X.ndim == 3:
            X = X.reshape(X.shape[0], -1)

        self._model.fit(X, y)
        self.is_fitted = True

        self.history['n_samples'] = len(y)
        self.history['n_features'] = X.shape[1]
        self.history['classes'] = np.unique(y).tolist()

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        if X.ndim == 3:
            X = X.reshape(X.shape[0], -1)

        return self._model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        if X.ndim == 3:
            X = X.reshape(X.shape[0], -1)

        return self._model.predict_proba(X)

    def get_coefficients(self) -> np.ndarray:
        """Get model coefficients."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        return self._model.named_steps['lr'].coef_

    def get_feature_importance(self) -> np.ndarray:
        """
        Get feature importance (absolute coefficient values).

        Returns:
            Feature importance scores
        """
        coef = self.get_coefficients()
        return np.abs(coef).mean(axis=0)


class L1Logistic(LogisticClassifier):
    """Logistic Regression with L1 (Lasso) regularization for feature selection."""

    def __init__(self, C: float = 1.0, **kwargs):
        super().__init__(
            C=C,
            penalty="l1",
            solver="saga",
            name="L1-Logistic",
            **kwargs
        )
