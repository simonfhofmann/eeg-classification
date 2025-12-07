# ----------------------------------------------------------------------
# lda.py
#
# Linear Discriminant Analysis classifier for EEG classification.
# ----------------------------------------------------------------------

import numpy as np
from typing import Dict, Any, Optional
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from models.base import BaseClassifier
from config import LDA_PARAMS


class LDAClassifier(BaseClassifier):
    """
    Linear Discriminant Analysis classifier.

    LDA is a classic method for EEG classification that:
    - Finds linear combinations of features that best separate classes
    - Works well with limited training data
    - Provides interpretable results
    """

    def __init__(
        self,
        solver: str = "svd",
        shrinkage: Optional[str] = None,
        n_components: Optional[int] = None,
        scale_features: bool = True,
        name: str = "LDA",
        **kwargs
    ):
        """
        Initialize LDA classifier.

        Args:
            solver: Solver to use ('svd', 'lsqr', 'eigen')
            shrinkage: Shrinkage parameter ('auto', float, or None)
                      Only works with 'lsqr' and 'eigen' solvers
            n_components: Number of components for dimensionality reduction
            scale_features: Whether to standardize features first
            name: Model name
        """
        super().__init__(name=name, solver=solver, shrinkage=shrinkage, **kwargs)

        self.solver = solver
        self.shrinkage = shrinkage
        self.n_components = n_components
        self.scale_features = scale_features

        self._build_pipeline()

    def _build_pipeline(self):
        """Build sklearn pipeline."""
        steps = []

        if self.scale_features:
            steps.append(('scaler', StandardScaler()))

        lda = LinearDiscriminantAnalysis(
            solver=self.solver,
            shrinkage=self.shrinkage,
            n_components=self.n_components
        )
        steps.append(('lda', lda))

        self._model = Pipeline(steps)

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> 'LDAClassifier':
        """
        Train the LDA classifier.

        Args:
            X: Training features, shape (n_samples, n_features)
            y: Training labels, shape (n_samples,)

        Returns:
            self
        """
        if X.ndim == 3:
            X = X.reshape(X.shape[0], -1)

        self._model.fit(X, y)
        self.is_fitted = True

        # Store training info
        self.history['n_samples'] = len(y)
        self.history['n_features'] = X.shape[1]
        self.history['classes'] = np.unique(y).tolist()

        # Store explained variance ratio if available
        lda = self._model.named_steps['lda']
        if hasattr(lda, 'explained_variance_ratio_'):
            self.history['explained_variance_ratio'] = lda.explained_variance_ratio_.tolist()

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

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Project data to LDA space.

        Args:
            X: Features to transform

        Returns:
            Transformed features
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        if X.ndim == 3:
            X = X.reshape(X.shape[0], -1)

        return self._model.transform(X)

    def get_coefficients(self) -> np.ndarray:
        """
        Get LDA coefficients (feature weights).

        Returns:
            Coefficients array
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        return self._model.named_steps['lda'].coef_


class ShrinkageLDA(LDAClassifier):
    """LDA with automatic shrinkage (regularization)."""

    def __init__(self, **kwargs):
        super().__init__(
            solver="lsqr",
            shrinkage="auto",
            name="Shrinkage-LDA",
            **kwargs
        )
