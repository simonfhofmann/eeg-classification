# ----------------------------------------------------------------------
# pipeline.py
#
# Feature extraction pipeline for combining multiple extractors.
# ----------------------------------------------------------------------

import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Any, Union
from pathlib import Path

from features.base import FeatureExtractor


class FeaturePipeline:
    """
    Pipeline for combining multiple feature extractors.

    The pipeline extracts features from each extractor and concatenates
    them into a single feature matrix. It also tracks feature names
    for interpretability.

    Args:
        extractors: List of FeatureExtractor instances
        normalize: Whether to normalize features ('zscore', 'minmax', or None)

    Example:
        >>> pipeline = FeaturePipeline([
        ...     BandPowerExtractor(bands=['alpha', 'beta']),
        ...     HjorthExtractor(),
        ...     ConnectivityExtractor(),
        ... ])
        >>> X_train = pipeline.fit_extract(eeg_train)
        >>> X_test = pipeline.extract(eeg_test)
    """

    def __init__(
        self,
        extractors: List[FeatureExtractor],
        normalize: Optional[str] = None
    ):
        self.extractors = extractors
        self.normalize = normalize

        # Normalization parameters (fitted on training data)
        self._mean: Optional[np.ndarray] = None
        self._std: Optional[np.ndarray] = None
        self._min: Optional[np.ndarray] = None
        self._max: Optional[np.ndarray] = None
        self._is_fitted = False

    def extract(self, eeg_data: np.ndarray, **kwargs) -> np.ndarray:
        """
        Extract features from all extractors.

        Args:
            eeg_data: EEG data array of shape (n_trials, n_channels, n_timepoints)
            **kwargs: Additional arguments passed to extractors

        Returns:
            Feature matrix of shape (n_trials, total_n_features)
        """
        all_features = []

        for extractor in self.extractors:
            features = extractor.extract(eeg_data, **kwargs)
            all_features.append(features)

        X = np.concatenate(all_features, axis=1)

        # Apply normalization if fitted
        if self._is_fitted and self.normalize:
            X = self._apply_normalization(X)

        return X

    def fit(self, eeg_data: np.ndarray, **kwargs) -> 'FeaturePipeline':
        """
        Fit the pipeline on training data.

        This fits all extractors and computes normalization parameters.

        Args:
            eeg_data: Training EEG data
            **kwargs: Additional arguments

        Returns:
            self (for method chaining)
        """
        # Fit all extractors
        for extractor in self.extractors:
            extractor.fit(eeg_data, **kwargs)

        # Extract features for normalization fitting
        X = self.extract(eeg_data, **kwargs)

        # Fit normalization
        if self.normalize == 'zscore':
            self._mean = X.mean(axis=0)
            self._std = X.std(axis=0)
            self._std[self._std == 0] = 1  # Prevent division by zero

        elif self.normalize == 'minmax':
            self._min = X.min(axis=0)
            self._max = X.max(axis=0)
            range_vals = self._max - self._min
            range_vals[range_vals == 0] = 1  # Prevent division by zero
            self._max = self._min + range_vals

        self._is_fitted = True
        return self

    def fit_extract(self, eeg_data: np.ndarray, **kwargs) -> np.ndarray:
        """
        Fit the pipeline and extract features in one step.

        Args:
            eeg_data: Training EEG data
            **kwargs: Additional arguments

        Returns:
            Feature matrix
        """
        self.fit(eeg_data, **kwargs)
        return self.extract(eeg_data, **kwargs)

    def _apply_normalization(self, X: np.ndarray) -> np.ndarray:
        """Apply fitted normalization to features."""
        if self.normalize == 'zscore':
            return (X - self._mean) / self._std
        elif self.normalize == 'minmax':
            return (X - self._min) / (self._max - self._min)
        return X

    def get_feature_names(self) -> List[str]:
        """
        Get names for all features in the pipeline.

        Returns:
            List of feature names
        """
        all_names = []
        for extractor in self.extractors:
            all_names.extend(extractor.get_feature_names())
        return all_names

    @property
    def n_features(self) -> int:
        """Total number of features from all extractors."""
        return sum(e.n_features for e in self.extractors)

    def get_feature_info(self) -> pd.DataFrame:
        """
        Get detailed information about all features.

        Returns:
            DataFrame with feature names, extractor sources, and indices
        """
        records = []
        idx = 0
        for extractor in self.extractors:
            for name in extractor.get_feature_names():
                records.append({
                    'index': idx,
                    'name': name,
                    'extractor': extractor.name,
                })
                idx += 1

        return pd.DataFrame(records)

    def add_extractor(self, extractor: FeatureExtractor) -> 'FeaturePipeline':
        """
        Add an extractor to the pipeline.

        Args:
            extractor: FeatureExtractor to add

        Returns:
            self (for method chaining)
        """
        self.extractors.append(extractor)
        self._is_fitted = False  # Need to refit
        return self

    def remove_extractor(self, name: str) -> 'FeaturePipeline':
        """
        Remove an extractor by name.

        Args:
            name: Name of the extractor to remove

        Returns:
            self (for method chaining)
        """
        self.extractors = [e for e in self.extractors if e.name != name]
        self._is_fitted = False
        return self

    def __repr__(self) -> str:
        extractor_names = [e.name for e in self.extractors]
        return f"FeaturePipeline({extractor_names}, n_features={self.n_features})"


def create_default_pipeline(
    ch_names: Optional[List[str]] = None,
    sfreq: int = 500,
    normalize: str = 'zscore'
) -> FeaturePipeline:
    """
    Create a default feature extraction pipeline.

    Includes time-domain, frequency-domain, and connectivity features.

    Args:
        ch_names: Channel names for feature naming
        sfreq: Sampling frequency
        normalize: Normalization method

    Returns:
        Configured FeaturePipeline
    """
    # Import here to avoid circular imports
    from features.time_domain import (
        extract_time_features,
        compute_variance,
        compute_hjorth_parameters
    )
    from features.frequency_domain import extract_frequency_features

    # Create wrapper extractors for existing functions
    # (In a full implementation, these would be proper FeatureExtractor subclasses)
    print("Warning: create_default_pipeline uses simplified extractors. "
          "Consider implementing full FeatureExtractor subclasses.")

    return FeaturePipeline([], normalize=normalize)


class FeatureSelector:
    """
    Select a subset of features based on various criteria.

    Useful for feature selection after extraction.

    Args:
        method: Selection method ('variance', 'correlation', 'importance', 'manual')
        n_features: Number of features to select (for automatic methods)
        feature_names: List of feature names to keep (for 'manual' method)
    """

    def __init__(
        self,
        method: str = 'variance',
        n_features: Optional[int] = None,
        feature_names: Optional[List[str]] = None
    ):
        self.method = method
        self.n_features = n_features
        self.feature_names = feature_names
        self._selected_indices: Optional[np.ndarray] = None
        self._is_fitted = False

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None,
            feature_names: Optional[List[str]] = None) -> 'FeatureSelector':
        """
        Fit the selector on training data.

        Args:
            X: Feature matrix
            y: Labels (needed for some methods)
            feature_names: Feature names (needed for 'manual' method)

        Returns:
            self
        """
        if self.method == 'variance':
            variances = X.var(axis=0)
            if self.n_features:
                self._selected_indices = np.argsort(variances)[-self.n_features:]
            else:
                # Keep features with non-zero variance
                self._selected_indices = np.where(variances > 0)[0]

        elif self.method == 'manual':
            if feature_names is None or self.feature_names is None:
                raise ValueError("Feature names required for manual selection")
            self._selected_indices = np.array([
                feature_names.index(name) for name in self.feature_names
                if name in feature_names
            ])

        elif self.method == 'correlation':
            if y is None:
                raise ValueError("Labels required for correlation-based selection")
            correlations = np.array([
                np.abs(np.corrcoef(X[:, i], y)[0, 1]) for i in range(X.shape[1])
            ])
            correlations = np.nan_to_num(correlations)
            if self.n_features:
                self._selected_indices = np.argsort(correlations)[-self.n_features:]
            else:
                self._selected_indices = np.where(correlations > 0.1)[0]

        self._is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Select features from the matrix.

        Args:
            X: Feature matrix

        Returns:
            Matrix with selected features only
        """
        if not self._is_fitted:
            raise ValueError("Selector not fitted. Call fit() first.")
        return X[:, self._selected_indices]

    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None,
                      feature_names: Optional[List[str]] = None) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(X, y, feature_names).transform(X)

    def get_selected_feature_names(self, feature_names: List[str]) -> List[str]:
        """Get names of selected features."""
        if not self._is_fitted:
            raise ValueError("Selector not fitted.")
        return [feature_names[i] for i in self._selected_indices]

    @property
    def n_selected(self) -> int:
        """Number of selected features."""
        if self._selected_indices is None:
            return 0
        return len(self._selected_indices)
