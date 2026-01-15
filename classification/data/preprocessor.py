# ----------------------------------------------------------------------
# preprocessor.py
#
# EEG data preprocessing utilities for classification.
# Includes trial filtering, label creation, data splitting, and scaling.
# ----------------------------------------------------------------------

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List, Union
from pathlib import Path
from sklearn.model_selection import train_test_split

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import SAMPLING_RATE, FREQ_BANDS, EPOCH_TMIN, EPOCH_TMAX, RANDOM_STATE
from data.containers import EEGDataContainer, SplitDataContainer


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


# ======================================================================
# New preprocessing functions for the refactored pipeline
# ======================================================================

def filter_trials_by_rating(
    data: EEGDataContainer,
    behavioral_df: pd.DataFrame,
    exclude_ratings: List[int] = [3],
    rating_column: str = "familiarity_rating"
) -> Tuple[EEGDataContainer, pd.DataFrame]:
    """
    Filter out trials with specific ratings (e.g., neutral ratings).

    Args:
        data: EEGDataContainer with EEG data
        behavioral_df: DataFrame with behavioral responses
        exclude_ratings: List of ratings to exclude (default: [3] for neutral)
        rating_column: Column name containing ratings

    Returns:
        Tuple of (filtered EEGDataContainer, filtered behavioral DataFrame)

    Example:
        >>> data, behavioral = filter_trials_by_rating(
        ...     data, behavioral_df, exclude_ratings=[3]
        ... )
        >>> print(f"Kept {data.n_trials} trials")
    """
    if rating_column not in behavioral_df.columns:
        raise ValueError(f"Column '{rating_column}' not found in behavioral data")

    ratings = behavioral_df[rating_column].values

    # Find valid trials (not in exclude list)
    valid_mask = ~np.isin(ratings, exclude_ratings)
    valid_indices = np.where(valid_mask)[0]

    # Filter EEG data
    filtered_data = data.select_trials(valid_indices)

    # Filter behavioral data
    filtered_behavioral = behavioral_df.iloc[valid_indices].reset_index(drop=True)

    # Print statistics
    n_excluded = len(ratings) - len(valid_indices)
    print(f"Trial filtering: kept {len(valid_indices)}/{len(ratings)} trials "
          f"(excluded {n_excluded} with ratings {exclude_ratings})")

    return filtered_data, filtered_behavioral


def create_binary_labels(
    behavioral_df: pd.DataFrame,
    rating_column: str = "familiarity_rating",
    threshold: int = 3,
    low_label: int = 0,
    high_label: int = 1
) -> np.ndarray:
    """
    Create binary labels from ratings.

    Ratings <= threshold -> low_label (e.g., unfamiliar)
    Ratings > threshold -> high_label (e.g., familiar)

    Args:
        behavioral_df: DataFrame with behavioral responses
        rating_column: Column name containing ratings
        threshold: Threshold for binarization
        low_label: Label for ratings <= threshold
        high_label: Label for ratings > threshold

    Returns:
        Binary labels array

    Example:
        >>> labels = create_binary_labels(behavioral_df, threshold=3)
        >>> # Ratings 1-3 -> 0 (unfamiliar), Ratings 4-5 -> 1 (familiar)
    """
    if rating_column not in behavioral_df.columns:
        raise ValueError(f"Column '{rating_column}' not found in behavioral data")

    ratings = behavioral_df[rating_column].values
    labels = np.where(ratings > threshold, high_label, low_label)

    # Print statistics
    n_low = np.sum(labels == low_label)
    n_high = np.sum(labels == high_label)
    print(f"Label distribution: {n_low} low ({low_label}), {n_high} high ({high_label})")

    return labels.astype(np.int64)


def create_binary_labels_exclude_neutral(
    behavioral_df: pd.DataFrame,
    rating_column: str = "familiarity_rating",
    low_ratings: List[int] = [1, 2],
    high_ratings: List[int] = [4, 5]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create binary labels excluding neutral ratings.

    This is the labeling scheme used in train_eegnet.py:
    - Ratings 1-2 -> 0 (unfamiliar)
    - Rating 3 -> excluded
    - Ratings 4-5 -> 1 (familiar)

    Args:
        behavioral_df: DataFrame with behavioral responses
        rating_column: Column name containing ratings
        low_ratings: Ratings to map to label 0
        high_ratings: Ratings to map to label 1

    Returns:
        Tuple of (labels for valid trials only, indices of valid trials)

    Example:
        >>> labels, valid_indices = create_binary_labels_exclude_neutral(behavioral_df)
        >>> X_filtered = X[valid_indices]
    """
    if rating_column not in behavioral_df.columns:
        raise ValueError(f"Column '{rating_column}' not found in behavioral data")

    ratings = behavioral_df[rating_column].values

    # Find valid trials
    low_mask = np.isin(ratings, low_ratings)
    high_mask = np.isin(ratings, high_ratings)
    valid_mask = low_mask | high_mask
    valid_indices = np.where(valid_mask)[0]

    # Create labels for valid trials
    valid_ratings = ratings[valid_indices]
    labels = np.where(np.isin(valid_ratings, high_ratings), 1, 0).astype(np.int64)

    # Print statistics
    n_low = np.sum(labels == 0)
    n_high = np.sum(labels == 1)
    n_excluded = len(ratings) - len(valid_indices)
    print(f"Label distribution:")
    print(f"  Low (ratings {low_ratings}): {n_low} trials")
    print(f"  High (ratings {high_ratings}): {n_high} trials")
    print(f"  Excluded: {n_excluded} trials")

    return labels, valid_indices


def stratified_train_val_test_split(
    data: EEGDataContainer,
    train_size: float = 0.7,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = RANDOM_STATE
) -> SplitDataContainer:
    """
    Perform stratified train/validation/test split on EEGDataContainer.

    Args:
        data: EEGDataContainer with X and y
        train_size: Fraction for training set
        val_size: Fraction for validation set
        test_size: Fraction for test set
        random_state: Random seed for reproducibility

    Returns:
        SplitDataContainer with train, val, test EEGDataContainers

    Raises:
        ValueError: If data has no labels or sizes don't sum to 1.0

    Example:
        >>> splits = stratified_train_val_test_split(data, 0.7, 0.15, 0.15)
        >>> print(f"Train: {splits.train.n_trials}, Val: {splits.val.n_trials}, Test: {splits.test.n_trials}")
    """
    if data.y is None:
        raise ValueError("Data must have labels (y) for stratified splitting")

    # Validate sizes
    total = train_size + val_size + test_size
    if not np.isclose(total, 1.0):
        raise ValueError(f"Split sizes must sum to 1.0, got {total}")

    X, y = data.X, data.y

    # First split: separate test set
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )

    # Second split: separate train and validation from remaining
    # Adjust val_size relative to remaining data
    val_size_adjusted = val_size / (train_size + val_size)

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val,
        test_size=val_size_adjusted,
        stratify=y_train_val,
        random_state=random_state
    )

    # Create containers for each split
    def make_container(X_split, y_split, split_name):
        return EEGDataContainer(
            X=X_split,
            y=y_split,
            sfreq=data.sfreq,
            ch_names=data.ch_names.copy(),
            participant_id=data.participant_id,
            metadata={**data.metadata, 'split': split_name}
        )

    train_data = make_container(X_train, y_train, 'train')
    val_data = make_container(X_val, y_val, 'val')
    test_data = make_container(X_test, y_test, 'test')

    # Print statistics
    print(f"\nData split (stratified):")
    print(f"  Train: {len(y_train)} trials "
          f"(class 0: {np.sum(y_train == 0)}, class 1: {np.sum(y_train == 1)})")
    print(f"  Val:   {len(y_val)} trials "
          f"(class 0: {np.sum(y_val == 0)}, class 1: {np.sum(y_val == 1)})")
    print(f"  Test:  {len(y_test)} trials "
          f"(class 0: {np.sum(y_test == 0)}, class 1: {np.sum(y_test == 1)})")

    return SplitDataContainer(train=train_data, val=val_data, test=test_data)


def stratified_train_test_split(
    data: EEGDataContainer,
    test_size: float = 0.2,
    random_state: int = RANDOM_STATE
) -> SplitDataContainer:
    """
    Perform stratified train/test split (no validation set).

    Args:
        data: EEGDataContainer with X and y
        test_size: Fraction for test set
        random_state: Random seed

    Returns:
        SplitDataContainer with train and test (val=None)
    """
    if data.y is None:
        raise ValueError("Data must have labels (y) for stratified splitting")

    X_train, X_test, y_train, y_test = train_test_split(
        data.X, data.y,
        test_size=test_size,
        stratify=data.y,
        random_state=random_state
    )

    def make_container(X_split, y_split, split_name):
        return EEGDataContainer(
            X=X_split,
            y=y_split,
            sfreq=data.sfreq,
            ch_names=data.ch_names.copy(),
            participant_id=data.participant_id,
            metadata={**data.metadata, 'split': split_name}
        )

    return SplitDataContainer(
        train=make_container(X_train, y_train, 'train'),
        val=None,
        test=make_container(X_test, y_test, 'test')
    )


def scale_to_microvolts(data: EEGDataContainer) -> EEGDataContainer:
    """
    Scale EEG data to microvolts.

    Neural network models often work better with data in microvolts
    rather than volts.

    Args:
        data: EEGDataContainer (assumed to be in volts)

    Returns:
        New EEGDataContainer with data in microvolts
    """
    return data.to_microvolts()


def scale_to_volts(data: EEGDataContainer) -> EEGDataContainer:
    """
    Scale EEG data to volts.

    MNE typically works with data in volts.

    Args:
        data: EEGDataContainer (assumed to be in microvolts)

    Returns:
        New EEGDataContainer with data in volts
    """
    return data.to_volts()


def compute_class_weights(y: np.ndarray) -> np.ndarray:
    """
    Compute balanced class weights for imbalanced datasets.

    Uses the formula: weight_i = n_samples / (n_classes * n_samples_i)

    Args:
        y: Labels array

    Returns:
        Array of class weights indexed by class label

    Example:
        >>> weights = compute_class_weights(y_train)
        >>> # For PyTorch: torch.tensor(weights, dtype=torch.float32)
    """
    classes = np.unique(y)
    n_samples = len(y)
    n_classes = len(classes)

    weights = np.zeros(n_classes)
    for i, c in enumerate(classes):
        n_samples_c = np.sum(y == c)
        weights[i] = n_samples / (n_classes * n_samples_c)

    return weights
