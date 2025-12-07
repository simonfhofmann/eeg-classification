# ----------------------------------------------------------------------
# svm_classifier.py
#
# Support Vector Machine classifiers for EEG classification.
# ----------------------------------------------------------------------

import numpy as np
from typing import Dict, Any, Optional
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from models.base import BaseClassifier, SklearnWrapper
from config import SVM_PARAMS


class SVMClassifier(BaseClassifier):
    """
    SVM classifier with optional preprocessing pipeline.

    Features:
    - Automatic feature scaling
    - Multiple kernel options (linear, RBF, polynomial)
    - Hyperparameter configuration
    """

    def __init__(
        self,
        kernel: str = "rbf",
        C: float = 1.0,
        gamma: str = "scale",
        scale_features: bool = True,
        class_weight: Optional[str] = "balanced",
        name: str = "SVM",
        **kwargs
    ):
        """
        Initialize SVM classifier.

        Args:
            kernel: Kernel type ('linear', 'rbf', 'poly', 'sigmoid')
            C: Regularization parameter
            gamma: Kernel coefficient ('scale', 'auto', or float)
            scale_features: Whether to standardize features
            class_weight: Class weighting ('balanced' or None)
            name: Model name
            **kwargs: Additional SVC parameters
        """
        super().__init__(name=name, kernel=kernel, C=C, gamma=gamma, **kwargs)

        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.scale_features = scale_features
        self.class_weight = class_weight

        # Build pipeline
        self._build_pipeline()

    def _build_pipeline(self):
        """Build sklearn pipeline with optional scaler."""
        steps = []

        if self.scale_features:
            steps.append(('scaler', StandardScaler()))

        svm = SVC(
            kernel=self.kernel,
            C=self.C,
            gamma=self.gamma,
            class_weight=self.class_weight,
            probability=True,  # Enable predict_proba
            **{k: v for k, v in self.params.items()
               if k not in ['kernel', 'C', 'gamma', 'name']}
        )
        steps.append(('svm', svm))

        self._model = Pipeline(steps)

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> 'SVMClassifier':
        """
        Train the SVM classifier.

        Args:
            X: Training features, shape (n_samples, n_features)
            y: Training labels, shape (n_samples,)

        Returns:
            self
        """
        # Flatten if 3D
        if X.ndim == 3:
            X = X.reshape(X.shape[0], -1)

        self._model.fit(X, y)
        self.is_fitted = True

        # Store training info
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

    def get_support_vectors(self) -> np.ndarray:
        """Get support vectors from trained model."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        return self._model.named_steps['svm'].support_vectors_


class LinearSVM(SVMClassifier):
    """Linear SVM classifier (faster for high-dimensional data)."""

    def __init__(self, C: float = 1.0, **kwargs):
        super().__init__(kernel="linear", C=C, name="LinearSVM", **kwargs)


class RBFSVM(SVMClassifier):
    """RBF kernel SVM classifier."""

    def __init__(self, C: float = 1.0, gamma: str = "scale", **kwargs):
        super().__init__(kernel="rbf", C=C, gamma=gamma, name="RBF-SVM", **kwargs)


def grid_search_svm(
    X: np.ndarray,
    y: np.ndarray,
    param_grid: Optional[Dict] = None,
    cv: int = 5,
    scoring: str = "balanced_accuracy"
) -> Dict[str, Any]:
    """
    Perform grid search for SVM hyperparameters.

    Args:
        X: Features
        y: Labels
        param_grid: Parameter grid (uses default if None)
        cv: Number of cross-validation folds
        scoring: Scoring metric

    Returns:
        Dictionary with best parameters and results
    """
    from sklearn.model_selection import GridSearchCV

    if param_grid is None:
        param_grid = {
            'svm__C': [0.1, 1, 10, 100],
            'svm__gamma': ['scale', 'auto', 0.01, 0.1],
            'svm__kernel': ['rbf', 'linear']
        }

    # Create base model
    clf = SVMClassifier()

    grid_search = GridSearchCV(
        clf._model,
        param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        verbose=1
    )

    if X.ndim == 3:
        X = X.reshape(X.shape[0], -1)

    grid_search.fit(X, y)

    return {
        'best_params': grid_search.best_params_,
        'best_score': grid_search.best_score_,
        'cv_results': grid_search.cv_results_
    }
