# ----------------------------------------------------------------------
# base.py
#
# Abstract base class for all EEG classifiers.
# All model implementations should inherit from this class.
# ----------------------------------------------------------------------

from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
import json
import pickle


class BaseClassifier(ABC):
    """
    Abstract base class for EEG classification models.

    All classifiers (SVM, Deep Learning, Statistical) should inherit
    from this class to ensure a consistent interface across the project.

    This allows team members to implement different methods while
    maintaining compatibility with the evaluation pipeline.
    """

    def __init__(self, name: str = "BaseClassifier", **kwargs):
        """
        Initialize the classifier.

        Args:
            name: Human-readable name for the model
            **kwargs: Model-specific parameters
        """
        self.name = name
        self.params = kwargs
        self.is_fitted = False
        self._model = None

        # Store training history/metrics
        self.history: Dict[str, Any] = {}

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> 'BaseClassifier':
        """
        Train the model.

        Args:
            X: Training features, shape (n_samples, n_features) or
               (n_samples, n_channels, n_timepoints) for deep learning
            y: Training labels, shape (n_samples,)
            **kwargs: Additional training parameters

        Returns:
            self
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.

        Args:
            X: Features to predict on

        Returns:
            Predicted labels, shape (n_samples,)
        """
        pass

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.

        Args:
            X: Features to predict on

        Returns:
            Class probabilities, shape (n_samples, n_classes)

        Note:
            Not all classifiers support probability predictions.
            Override this method if your classifier supports it.
        """
        raise NotImplementedError(
            f"{self.name} does not support probability predictions"
        )

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute accuracy score.

        Args:
            X: Test features
            y: True labels

        Returns:
            Accuracy score (0-1)
        """
        predictions = self.predict(X)
        return np.mean(predictions == y)

    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        metrics: Optional[list] = None
    ) -> Dict[str, float]:
        """
        Comprehensive evaluation of the model.

        Args:
            X: Test features
            y: True labels
            metrics: List of metrics to compute. Default: ['accuracy', 'f1', 'auc']

        Returns:
            Dictionary of metric names to values
        """
        from sklearn.metrics import (
            accuracy_score, f1_score, roc_auc_score,
            precision_score, recall_score, balanced_accuracy_score
        )

        if metrics is None:
            metrics = ['accuracy', 'balanced_accuracy', 'f1', 'precision', 'recall']

        predictions = self.predict(X)

        results = {}

        if 'accuracy' in metrics:
            results['accuracy'] = accuracy_score(y, predictions)

        if 'balanced_accuracy' in metrics:
            results['balanced_accuracy'] = balanced_accuracy_score(y, predictions)

        if 'f1' in metrics:
            results['f1'] = f1_score(y, predictions, average='weighted')

        if 'precision' in metrics:
            results['precision'] = precision_score(y, predictions, average='weighted', zero_division=0)

        if 'recall' in metrics:
            results['recall'] = recall_score(y, predictions, average='weighted', zero_division=0)

        if 'auc' in metrics:
            try:
                proba = self.predict_proba(X)
                if proba.shape[1] == 2:
                    results['auc'] = roc_auc_score(y, proba[:, 1])
                else:
                    results['auc'] = roc_auc_score(y, proba, multi_class='ovr')
            except (NotImplementedError, ValueError):
                results['auc'] = np.nan

        return results

    def get_params(self) -> Dict[str, Any]:
        """
        Get model parameters.

        Returns:
            Dictionary of parameter names to values
        """
        return self.params.copy()

    def set_params(self, **params) -> 'BaseClassifier':
        """
        Set model parameters.

        Args:
            **params: Parameters to set

        Returns:
            self
        """
        self.params.update(params)
        return self

    def save(self, filepath: str) -> None:
        """
        Save model to disk.

        Args:
            filepath: Path to save model (without extension)
        """
        filepath = Path(filepath)

        # Save model state
        model_data = {
            'name': self.name,
            'params': self.params,
            'is_fitted': self.is_fitted,
            'history': self.history,
        }

        # Save metadata as JSON
        with open(filepath.with_suffix('.json'), 'w') as f:
            json.dump(model_data, f, indent=2, default=str)

        # Save model object as pickle
        with open(filepath.with_suffix('.pkl'), 'wb') as f:
            pickle.dump(self._model, f)

    @classmethod
    def load(cls, filepath: str) -> 'BaseClassifier':
        """
        Load model from disk.

        Args:
            filepath: Path to saved model (without extension)

        Returns:
            Loaded classifier instance
        """
        filepath = Path(filepath)

        # Load metadata
        with open(filepath.with_suffix('.json'), 'r') as f:
            model_data = json.load(f)

        # Load model object
        with open(filepath.with_suffix('.pkl'), 'rb') as f:
            model_obj = pickle.load(f)

        # Create instance
        instance = cls(name=model_data['name'], **model_data['params'])
        instance._model = model_obj
        instance.is_fitted = model_data['is_fitted']
        instance.history = model_data['history']

        return instance

    def __repr__(self) -> str:
        status = "fitted" if self.is_fitted else "not fitted"
        return f"{self.name}({status}, params={self.params})"


class SklearnWrapper(BaseClassifier):
    """
    Wrapper for sklearn-compatible classifiers.

    Use this as a base class for SVM, LDA, Logistic Regression, etc.
    """

    def __init__(self, sklearn_model, name: str = "SklearnModel", **kwargs):
        """
        Initialize with an sklearn model.

        Args:
            sklearn_model: An sklearn classifier instance
            name: Name for this model
            **kwargs: Additional parameters
        """
        super().__init__(name=name, **kwargs)
        self._model = sklearn_model

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> 'SklearnWrapper':
        """Train the sklearn model."""
        self._model.fit(X, y)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        return self._model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities if supported."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        if hasattr(self._model, 'predict_proba'):
            return self._model.predict_proba(X)
        raise NotImplementedError(f"{self.name} does not support predict_proba")

    def get_params(self) -> Dict[str, Any]:
        """Get sklearn model parameters."""
        params = super().get_params()
        if hasattr(self._model, 'get_params'):
            params.update(self._model.get_params())
        return params
