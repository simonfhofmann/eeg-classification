# ----------------------------------------------------------------------
# preprocessor.py
#
# EEG data preprocessing utilities for classification.
# ----------------------------------------------------------------------

import numpy as np
from typing import Dict, Tuple, Optional, List, Union
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import SAMPLING_RATE, FREQ_BANDS, EPOCH_TMIN, EPOCH_TMAX


class Preprocessor:
    """
    EEG data preprocessor for classification.

    Handles:
    - Epoching (extracting trial segments)
    - Baseline correction
    - Normalization/standardization
    - Feature matrix construction

    Note:
        Heavy preprocessing (filtering, ICA, artifact rejection) is assumed
        to be done in MATLAB/EEGLAB. This class handles ML-specific preprocessing.
    """

    def __init__(
        self,
        sampling_rate: int = SAMPLING_RATE,
        epoch_tmin: float = EPOCH_TMIN,
        epoch_tmax: float = EPOCH_TMAX,
        baseline_correction: bool = True,
        baseline_window: Tuple[float, float] = (-0.2, 0.0),
        normalize: str = "zscore"
    ):
        """
        Initialize preprocessor.

        Args:
            sampling_rate: EEG sampling rate in Hz
            epoch_tmin: Epoch start time relative to event (seconds)
            epoch_tmax: Epoch end time relative to event (seconds)
            baseline_correction: Whether to apply baseline correction
            baseline_window: Time window for baseline (start, end) in seconds
            normalize: Normalization method ("zscore", "minmax", or None)
        """
        self.sampling_rate = sampling_rate
        self.epoch_tmin = epoch_tmin
        self.epoch_tmax = epoch_tmax
        self.baseline_correction = baseline_correction
        self.baseline_window = baseline_window
        self.normalize = normalize

        # Computed attributes
        self.n_samples_epoch = int((epoch_tmax - epoch_tmin) * sampling_rate)

        # Normalization parameters (fitted on training data)
        self._mean = None
        self._std = None
        self._min = None
        self._max = None
        self._is_fitted = False

    def extract_epochs(
        self,
        eeg_data: np.ndarray,
        event_samples: np.ndarray,
        tmin: Optional[float] = None,
        tmax: Optional[float] = None
    ) -> np.ndarray:
        """
        Extract epochs from continuous EEG data.

        Args:
            eeg_data: Continuous EEG data, shape (n_channels, n_samples)
            event_samples: Sample indices of events (e.g., stimulus onset)
            tmin: Epoch start (seconds relative to event), uses default if None
            tmax: Epoch end (seconds relative to event), uses default if None

        Returns:
            Epoched data, shape (n_epochs, n_channels, n_samples)
        """
        tmin = tmin if tmin is not None else self.epoch_tmin
        tmax = tmax if tmax is not None else self.epoch_tmax

        start_offset = int(tmin * self.sampling_rate)
        end_offset = int(tmax * self.sampling_rate)
        n_samples = end_offset - start_offset

        n_channels = eeg_data.shape[0]
        n_epochs = len(event_samples)

        epochs = np.zeros((n_epochs, n_channels, n_samples))

        for i, event_sample in enumerate(event_samples):
            start = event_sample + start_offset
            end = event_sample + end_offset

            # Handle boundary cases
            if start < 0 or end > eeg_data.shape[1]:
                print(f"Warning: Epoch {i} out of bounds, padding with zeros")
                # Pad with zeros if necessary
                epoch = np.zeros((n_channels, n_samples))
                valid_start = max(0, start)
                valid_end = min(eeg_data.shape[1], end)
                offset = valid_start - start
                epoch[:, offset:offset + (valid_end - valid_start)] = eeg_data[:, valid_start:valid_end]
                epochs[i] = epoch
            else:
                epochs[i] = eeg_data[:, start:end]

        return epochs

    def apply_baseline_correction(
        self,
        epochs: np.ndarray,
        baseline_window: Optional[Tuple[float, float]] = None
    ) -> np.ndarray:
        """
        Apply baseline correction to epochs.

        Args:
            epochs: Epoched data, shape (n_epochs, n_channels, n_samples)
            baseline_window: (start, end) in seconds relative to epoch start

        Returns:
            Baseline-corrected epochs
        """
        if baseline_window is None:
            baseline_window = self.baseline_window

        # Convert to samples (relative to epoch start at tmin)
        bl_start = int((baseline_window[0] - self.epoch_tmin) * self.sampling_rate)
        bl_end = int((baseline_window[1] - self.epoch_tmin) * self.sampling_rate)

        # Ensure valid indices
        bl_start = max(0, bl_start)
        bl_end = min(epochs.shape[2], bl_end)

        if bl_end <= bl_start:
            print("Warning: Invalid baseline window, skipping correction")
            return epochs

        # Compute baseline mean and subtract
        baseline_mean = epochs[:, :, bl_start:bl_end].mean(axis=2, keepdims=True)
        corrected = epochs - baseline_mean

        return corrected

    def fit(self, X: np.ndarray) -> 'Preprocessor':
        """
        Fit normalization parameters on training data.

        Args:
            X: Training data, shape (n_samples, n_features) or (n_epochs, n_channels, n_times)

        Returns:
            self
        """
        # Flatten if 3D
        if X.ndim == 3:
            X_flat = X.reshape(X.shape[0], -1)
        else:
            X_flat = X

        if self.normalize == "zscore":
            self._mean = X_flat.mean(axis=0)
            self._std = X_flat.std(axis=0)
            self._std[self._std == 0] = 1  # Prevent division by zero

        elif self.normalize == "minmax":
            self._min = X_flat.min(axis=0)
            self._max = X_flat.max(axis=0)
            self._max[self._max == self._min] = self._min[self._max == self._min] + 1

        self._is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Apply fitted normalization to data.

        Args:
            X: Data to transform

        Returns:
            Normalized data
        """
        if not self._is_fitted and self.normalize:
            raise ValueError("Preprocessor not fitted. Call fit() first.")

        original_shape = X.shape

        # Flatten if 3D
        if X.ndim == 3:
            X_flat = X.reshape(X.shape[0], -1)
        else:
            X_flat = X.copy()

        if self.normalize == "zscore":
            X_flat = (X_flat - self._mean) / self._std

        elif self.normalize == "minmax":
            X_flat = (X_flat - self._min) / (self._max - self._min)

        # Restore shape
        if len(original_shape) == 3:
            return X_flat.reshape(original_shape)
        return X_flat

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fit and transform in one step.
        """
        return self.fit(X).transform(X)

    def process_epochs(
        self,
        epochs: np.ndarray,
        apply_baseline: Optional[bool] = None,
        apply_normalization: bool = False
    ) -> np.ndarray:
        """
        Full preprocessing pipeline for epochs.

        Args:
            epochs: Epoched data, shape (n_epochs, n_channels, n_samples)
            apply_baseline: Whether to apply baseline correction (uses default if None)
            apply_normalization: Whether to apply normalization

        Returns:
            Processed epochs
        """
        processed = epochs.copy()

        # Baseline correction
        if apply_baseline is None:
            apply_baseline = self.baseline_correction

        if apply_baseline:
            processed = self.apply_baseline_correction(processed)

        # Normalization
        if apply_normalization and self.normalize:
            processed = self.transform(processed)

        return processed

    def flatten_epochs(self, epochs: np.ndarray) -> np.ndarray:
        """
        Flatten epochs to 2D feature matrix.

        Args:
            epochs: Shape (n_epochs, n_channels, n_samples)

        Returns:
            Shape (n_epochs, n_channels * n_samples)
        """
        return epochs.reshape(epochs.shape[0], -1)

    def get_time_vector(self) -> np.ndarray:
        """
        Get time vector for epochs in seconds.

        Returns:
            Array of time points relative to event onset
        """
        return np.linspace(
            self.epoch_tmin,
            self.epoch_tmax,
            self.n_samples_epoch
        )


def prepare_data_for_classification(
    epochs: np.ndarray,
    labels: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
    flatten: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare epoched data for classification with train/test split.

    Args:
        epochs: Shape (n_epochs, n_channels, n_samples)
        labels: Shape (n_epochs,)
        test_size: Fraction for test set
        random_state: Random seed
        flatten: Whether to flatten epochs to 2D

    Returns:
        X_train, X_test, y_train, y_test
    """
    from sklearn.model_selection import train_test_split

    X = epochs
    y = labels

    if flatten:
        X = X.reshape(X.shape[0], -1)

    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
