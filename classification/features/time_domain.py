# ----------------------------------------------------------------------
# time_domain.py
#
# Time-domain feature extraction for EEG signals.
# ----------------------------------------------------------------------

import numpy as np
from typing import Dict, List, Optional, Union
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import SAMPLING_RATE


def compute_variance(epochs: np.ndarray) -> np.ndarray:
    """
    Compute variance for each channel.

    Args:
        epochs: Shape (n_epochs, n_channels, n_samples)

    Returns:
        Shape (n_epochs, n_channels)
    """
    return np.var(epochs, axis=2)


def compute_mean(epochs: np.ndarray) -> np.ndarray:
    """
    Compute mean for each channel.

    Args:
        epochs: Shape (n_epochs, n_channels, n_samples)

    Returns:
        Shape (n_epochs, n_channels)
    """
    return np.mean(epochs, axis=2)


def compute_std(epochs: np.ndarray) -> np.ndarray:
    """
    Compute standard deviation for each channel.

    Args:
        epochs: Shape (n_epochs, n_channels, n_samples)

    Returns:
        Shape (n_epochs, n_channels)
    """
    return np.std(epochs, axis=2)


def compute_peak_to_peak(epochs: np.ndarray) -> np.ndarray:
    """
    Compute peak-to-peak amplitude for each channel.

    Args:
        epochs: Shape (n_epochs, n_channels, n_samples)

    Returns:
        Shape (n_epochs, n_channels)
    """
    return np.ptp(epochs, axis=2)


def compute_rms(epochs: np.ndarray) -> np.ndarray:
    """
    Compute root mean square for each channel.

    Args:
        epochs: Shape (n_epochs, n_channels, n_samples)

    Returns:
        Shape (n_epochs, n_channels)
    """
    return np.sqrt(np.mean(epochs ** 2, axis=2))


def compute_zero_crossings(epochs: np.ndarray) -> np.ndarray:
    """
    Compute number of zero crossings for each channel.

    Args:
        epochs: Shape (n_epochs, n_channels, n_samples)

    Returns:
        Shape (n_epochs, n_channels)
    """
    # Subtract mean to center around zero
    centered = epochs - np.mean(epochs, axis=2, keepdims=True)

    # Count sign changes
    zero_crossings = np.sum(np.diff(np.sign(centered), axis=2) != 0, axis=2)

    return zero_crossings


def compute_hjorth_parameters(epochs: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Compute Hjorth parameters (activity, mobility, complexity).

    These parameters characterize EEG signals in terms of amplitude,
    mean frequency, and bandwidth.

    Args:
        epochs: Shape (n_epochs, n_channels, n_samples)

    Returns:
        Dictionary with:
            - 'activity': variance of the signal
            - 'mobility': sqrt(var(derivative) / var(signal))
            - 'complexity': mobility of derivative / mobility of signal
    """
    # Activity: variance
    activity = np.var(epochs, axis=2)

    # First derivative
    d1 = np.diff(epochs, axis=2)
    var_d1 = np.var(d1, axis=2)

    # Second derivative
    d2 = np.diff(d1, axis=2)
    var_d2 = np.var(d2, axis=2)

    # Mobility
    mobility = np.sqrt(var_d1 / (activity + 1e-10))

    # Complexity
    mobility_d1 = np.sqrt(var_d2 / (var_d1 + 1e-10))
    complexity = mobility_d1 / (mobility + 1e-10)

    return {
        'activity': activity,
        'mobility': mobility,
        'complexity': complexity
    }


def compute_kurtosis(epochs: np.ndarray) -> np.ndarray:
    """
    Compute kurtosis for each channel.

    Args:
        epochs: Shape (n_epochs, n_channels, n_samples)

    Returns:
        Shape (n_epochs, n_channels)
    """
    from scipy.stats import kurtosis
    return kurtosis(epochs, axis=2)


def compute_skewness(epochs: np.ndarray) -> np.ndarray:
    """
    Compute skewness for each channel.

    Args:
        epochs: Shape (n_epochs, n_channels, n_samples)

    Returns:
        Shape (n_epochs, n_channels)
    """
    from scipy.stats import skew
    return skew(epochs, axis=2)


def extract_time_features(
    epochs: np.ndarray,
    features: Optional[List[str]] = None
) -> np.ndarray:
    """
    Extract multiple time-domain features.

    Args:
        epochs: Shape (n_epochs, n_channels, n_samples)
        features: List of features to extract. Options:
            - 'variance', 'mean', 'std', 'ptp', 'rms'
            - 'zero_crossings', 'hjorth', 'kurtosis', 'skewness'
            If None, extracts: ['variance', 'hjorth', 'zero_crossings']

    Returns:
        Feature matrix, shape (n_epochs, n_features)
        where n_features = n_channels * n_feature_types
    """
    if features is None:
        features = ['variance', 'hjorth', 'zero_crossings']

    feature_arrays = []

    for feat in features:
        if feat == 'variance':
            feature_arrays.append(compute_variance(epochs))
        elif feat == 'mean':
            feature_arrays.append(compute_mean(epochs))
        elif feat == 'std':
            feature_arrays.append(compute_std(epochs))
        elif feat == 'ptp':
            feature_arrays.append(compute_peak_to_peak(epochs))
        elif feat == 'rms':
            feature_arrays.append(compute_rms(epochs))
        elif feat == 'zero_crossings':
            feature_arrays.append(compute_zero_crossings(epochs))
        elif feat == 'hjorth':
            hjorth = compute_hjorth_parameters(epochs)
            feature_arrays.append(hjorth['activity'])
            feature_arrays.append(hjorth['mobility'])
            feature_arrays.append(hjorth['complexity'])
        elif feat == 'kurtosis':
            feature_arrays.append(compute_kurtosis(epochs))
        elif feat == 'skewness':
            feature_arrays.append(compute_skewness(epochs))
        else:
            raise ValueError(f"Unknown feature: {feat}")

    # Stack all features: (n_epochs, n_channels * n_features)
    all_features = np.concatenate(feature_arrays, axis=1)

    return all_features


def get_feature_names(
    n_channels: int,
    features: Optional[List[str]] = None,
    channel_names: Optional[List[str]] = None
) -> List[str]:
    """
    Generate feature names for interpretation.

    Args:
        n_channels: Number of EEG channels
        features: List of extracted features
        channel_names: Optional channel names

    Returns:
        List of feature names
    """
    if features is None:
        features = ['variance', 'hjorth', 'zero_crossings']

    if channel_names is None:
        channel_names = [f"ch{i}" for i in range(n_channels)]

    names = []
    for feat in features:
        if feat == 'hjorth':
            for subfeat in ['activity', 'mobility', 'complexity']:
                for ch in channel_names:
                    names.append(f"{ch}_{subfeat}")
        else:
            for ch in channel_names:
                names.append(f"{ch}_{feat}")

    return names
