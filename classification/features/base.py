# ----------------------------------------------------------------------
# base.py
#
# Abstract base class for feature extractors.
# ----------------------------------------------------------------------

from abc import ABC, abstractmethod
import numpy as np
from typing import List, Optional, Dict, Any


class FeatureExtractor(ABC):
    """
    Abstract base class for all feature extractors.

    Feature extractors take EEG data and extract features that can be used
    for machine learning classification. All extractors must implement
    the `extract` and `get_feature_names` methods.

    Subclasses:
        - TimeDomainExtractor: Variance, Hjorth parameters, etc.
        - FrequencyDomainExtractor: Band power, spectral entropy, etc.
        - ConnectivityExtractor: Correlation, coherence, PLV
        - Custom extractors (e.g., AudioEEGCorrelationExtractor)

    Example:
        >>> class MyExtractor(FeatureExtractor):
        ...     def extract(self, eeg_data, **kwargs):
        ...         return my_feature_computation(eeg_data)
        ...
        ...     def get_feature_names(self):
        ...         return ['feature_1', 'feature_2']
    """

    def __init__(self, name: Optional[str] = None):
        """
        Initialize the feature extractor.

        Args:
            name: Optional name for this extractor instance
        """
        self.name = name or self.__class__.__name__

    @abstractmethod
    def extract(self, eeg_data: np.ndarray, **kwargs) -> np.ndarray:
        """
        Extract features from EEG data.

        Args:
            eeg_data: EEG data array of shape (n_trials, n_channels, n_timepoints)
            **kwargs: Additional arguments (e.g., stimulus_ids for audio features)

        Returns:
            Feature array of shape (n_trials, n_features)
        """
        pass

    @abstractmethod
    def get_feature_names(self) -> List[str]:
        """
        Get names for all extracted features.

        Returns:
            List of feature names (for interpretability and debugging)
        """
        pass

    @property
    def n_features(self) -> int:
        """Number of features extracted by this extractor."""
        return len(self.get_feature_names())

    def fit(self, eeg_data: np.ndarray, **kwargs) -> 'FeatureExtractor':
        """
        Fit the extractor to training data (if needed).

        Some extractors may need to learn parameters from training data
        (e.g., normalization statistics). Default implementation does nothing.

        Args:
            eeg_data: Training EEG data
            **kwargs: Additional arguments

        Returns:
            self (for method chaining)
        """
        return self

    def fit_extract(self, eeg_data: np.ndarray, **kwargs) -> np.ndarray:
        """
        Fit and extract in one step.

        Args:
            eeg_data: EEG data
            **kwargs: Additional arguments

        Returns:
            Feature array
        """
        return self.fit(eeg_data, **kwargs).extract(eeg_data, **kwargs)

    def __repr__(self) -> str:
        return f"{self.name}(n_features={self.n_features})"


class CompositeExtractor(FeatureExtractor):
    """
    Extractor that combines multiple extractors.

    This is useful when you want to use the same set of extractors
    as a single unit.

    Example:
        >>> extractor = CompositeExtractor([
        ...     VarianceExtractor(),
        ...     BandPowerExtractor(),
        ... ])
        >>> features = extractor.extract(eeg_data)
    """

    def __init__(self, extractors: List[FeatureExtractor], name: Optional[str] = None):
        """
        Initialize composite extractor.

        Args:
            extractors: List of FeatureExtractor instances to combine
            name: Optional name for this extractor
        """
        super().__init__(name=name or "CompositeExtractor")
        self.extractors = extractors

    def extract(self, eeg_data: np.ndarray, **kwargs) -> np.ndarray:
        """
        Extract features from all child extractors and concatenate.

        Args:
            eeg_data: EEG data array
            **kwargs: Additional arguments passed to all extractors

        Returns:
            Concatenated feature array
        """
        all_features = []
        for extractor in self.extractors:
            features = extractor.extract(eeg_data, **kwargs)
            all_features.append(features)

        return np.concatenate(all_features, axis=1)

    def get_feature_names(self) -> List[str]:
        """Get feature names from all child extractors."""
        all_names = []
        for extractor in self.extractors:
            names = extractor.get_feature_names()
            # Prefix with extractor name to avoid collisions
            prefixed = [f"{extractor.name}_{name}" for name in names]
            all_names.extend(prefixed)
        return all_names

    def fit(self, eeg_data: np.ndarray, **kwargs) -> 'CompositeExtractor':
        """Fit all child extractors."""
        for extractor in self.extractors:
            extractor.fit(eeg_data, **kwargs)
        return self

    def __repr__(self) -> str:
        extractor_names = [e.name for e in self.extractors]
        return f"CompositeExtractor({extractor_names})"


class ChannelWiseExtractor(FeatureExtractor):
    """
    Base class for extractors that compute features per channel.

    Many EEG features are computed independently for each channel
    (e.g., band power, variance). This base class handles the
    iteration over channels and naming conventions.

    Subclasses should implement `_extract_single_channel`.
    """

    def __init__(self, ch_names: Optional[List[str]] = None, name: Optional[str] = None):
        """
        Initialize channel-wise extractor.

        Args:
            ch_names: Channel names (for feature naming)
            name: Extractor name
        """
        super().__init__(name=name)
        self.ch_names = ch_names
        self._feature_names_per_channel: List[str] = []

    @abstractmethod
    def _extract_single_channel(self, channel_data: np.ndarray, **kwargs) -> np.ndarray:
        """
        Extract features from a single channel.

        Args:
            channel_data: Data for one channel, shape (n_trials, n_timepoints)
            **kwargs: Additional arguments

        Returns:
            Features for this channel, shape (n_trials, n_features_per_channel)
        """
        pass

    @abstractmethod
    def _get_feature_names_per_channel(self) -> List[str]:
        """
        Get feature names for a single channel (without channel prefix).

        Returns:
            List of feature names
        """
        pass

    def extract(self, eeg_data: np.ndarray, **kwargs) -> np.ndarray:
        """
        Extract features from all channels.

        Args:
            eeg_data: Shape (n_trials, n_channels, n_timepoints)
            **kwargs: Additional arguments

        Returns:
            Shape (n_trials, n_channels * n_features_per_channel)
        """
        n_trials, n_channels, _ = eeg_data.shape

        all_channel_features = []
        for ch_idx in range(n_channels):
            channel_data = eeg_data[:, ch_idx, :]
            features = self._extract_single_channel(channel_data, **kwargs)
            all_channel_features.append(features)

        return np.concatenate(all_channel_features, axis=1)

    def get_feature_names(self) -> List[str]:
        """Get feature names with channel prefixes."""
        base_names = self._get_feature_names_per_channel()

        if self.ch_names is None:
            # Use generic channel indices
            n_channels = len(self._feature_names_per_channel) // len(base_names) if self._feature_names_per_channel else 30
            ch_names = [f"ch{i}" for i in range(n_channels)]
        else:
            ch_names = self.ch_names

        all_names = []
        for ch_name in ch_names:
            for feat_name in base_names:
                all_names.append(f"{ch_name}_{feat_name}")

        return all_names
