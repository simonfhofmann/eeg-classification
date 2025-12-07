# ----------------------------------------------------------------------
# frequency_domain.py
#
# Frequency-domain feature extraction for EEG signals.
# ----------------------------------------------------------------------

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
from scipy import signal

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import SAMPLING_RATE, FREQ_BANDS


def compute_psd(
    epochs: np.ndarray,
    sampling_rate: int = SAMPLING_RATE,
    method: str = "welch",
    nperseg: Optional[int] = None,
    noverlap: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Power Spectral Density using Welch's method.

    Args:
        epochs: Shape (n_epochs, n_channels, n_samples)
        sampling_rate: Sampling rate in Hz
        method: PSD method ('welch' or 'multitaper')
        nperseg: Length of each segment for Welch's method
        noverlap: Overlap between segments

    Returns:
        freqs: Frequency vector
        psd: Shape (n_epochs, n_channels, n_freqs)
    """
    if nperseg is None:
        nperseg = min(epochs.shape[2], sampling_rate * 2)  # 2 second windows

    if noverlap is None:
        noverlap = nperseg // 2

    if method == "welch":
        freqs, psd = signal.welch(
            epochs,
            fs=sampling_rate,
            nperseg=nperseg,
            noverlap=noverlap,
            axis=2
        )
    else:
        raise ValueError(f"Unknown PSD method: {method}")

    return freqs, psd


def compute_band_power(
    epochs: np.ndarray,
    sampling_rate: int = SAMPLING_RATE,
    bands: Optional[Dict[str, Tuple[float, float]]] = None,
    relative: bool = False
) -> Dict[str, np.ndarray]:
    """
    Compute power in standard frequency bands.

    Args:
        epochs: Shape (n_epochs, n_channels, n_samples)
        sampling_rate: Sampling rate in Hz
        bands: Dictionary of band names to (low, high) frequency tuples
        relative: If True, compute relative band power (proportion of total)

    Returns:
        Dictionary with band names as keys, power arrays as values
        Each array has shape (n_epochs, n_channels)
    """
    if bands is None:
        bands = FREQ_BANDS

    freqs, psd = compute_psd(epochs, sampling_rate)

    # Frequency resolution
    freq_res = freqs[1] - freqs[0]

    band_power = {}

    for band_name, (low, high) in bands.items():
        # Find frequency indices
        idx = np.logical_and(freqs >= low, freqs <= high)

        # Integrate power in band
        power = np.trapz(psd[:, :, idx], dx=freq_res, axis=2)
        band_power[band_name] = power

    # Compute relative power if requested
    if relative:
        total_power = sum(band_power.values())
        band_power = {k: v / (total_power + 1e-10) for k, v in band_power.items()}

    return band_power


def compute_spectral_entropy(
    epochs: np.ndarray,
    sampling_rate: int = SAMPLING_RATE,
    normalize: bool = True
) -> np.ndarray:
    """
    Compute spectral entropy for each channel.

    Spectral entropy measures the complexity/irregularity of the signal
    in the frequency domain.

    Args:
        epochs: Shape (n_epochs, n_channels, n_samples)
        sampling_rate: Sampling rate in Hz
        normalize: If True, normalize by log(n_freqs)

    Returns:
        Shape (n_epochs, n_channels)
    """
    freqs, psd = compute_psd(epochs, sampling_rate)

    # Normalize PSD to probability distribution
    psd_norm = psd / (np.sum(psd, axis=2, keepdims=True) + 1e-10)

    # Compute entropy
    entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-10), axis=2)

    if normalize:
        entropy = entropy / np.log2(psd.shape[2])

    return entropy


def compute_peak_frequency(
    epochs: np.ndarray,
    sampling_rate: int = SAMPLING_RATE,
    band: Optional[Tuple[float, float]] = None
) -> np.ndarray:
    """
    Compute peak frequency for each channel.

    Args:
        epochs: Shape (n_epochs, n_channels, n_samples)
        sampling_rate: Sampling rate in Hz
        band: Optional (low, high) frequency range to search within

    Returns:
        Shape (n_epochs, n_channels)
    """
    freqs, psd = compute_psd(epochs, sampling_rate)

    if band is not None:
        idx = np.logical_and(freqs >= band[0], freqs <= band[1])
        freqs_band = freqs[idx]
        psd_band = psd[:, :, idx]
    else:
        freqs_band = freqs
        psd_band = psd

    # Find peak frequency
    peak_idx = np.argmax(psd_band, axis=2)
    peak_freq = freqs_band[peak_idx]

    return peak_freq


def compute_spectral_edge(
    epochs: np.ndarray,
    sampling_rate: int = SAMPLING_RATE,
    edge: float = 0.95
) -> np.ndarray:
    """
    Compute spectral edge frequency (frequency below which X% of power lies).

    Args:
        epochs: Shape (n_epochs, n_channels, n_samples)
        sampling_rate: Sampling rate in Hz
        edge: Proportion of power (0-1), default 0.95 (95%)

    Returns:
        Shape (n_epochs, n_channels)
    """
    freqs, psd = compute_psd(epochs, sampling_rate)

    # Compute cumulative power
    cumsum = np.cumsum(psd, axis=2)
    total = cumsum[:, :, -1:]

    # Normalize
    cumsum_norm = cumsum / (total + 1e-10)

    # Find edge frequency
    edge_idx = np.argmax(cumsum_norm >= edge, axis=2)
    edge_freq = freqs[edge_idx]

    return edge_freq


def compute_band_ratios(
    epochs: np.ndarray,
    sampling_rate: int = SAMPLING_RATE,
    bands: Optional[Dict[str, Tuple[float, float]]] = None
) -> Dict[str, np.ndarray]:
    """
    Compute common frequency band ratios.

    Args:
        epochs: Shape (n_epochs, n_channels, n_samples)
        sampling_rate: Sampling rate in Hz
        bands: Frequency band definitions

    Returns:
        Dictionary of ratio names to arrays (n_epochs, n_channels)
    """
    band_power = compute_band_power(epochs, sampling_rate, bands)

    ratios = {}

    # Theta/Beta ratio (often used in attention studies)
    if 'theta' in band_power and 'beta' in band_power:
        ratios['theta_beta'] = band_power['theta'] / (band_power['beta'] + 1e-10)

    # Alpha/Theta ratio
    if 'alpha' in band_power and 'theta' in band_power:
        ratios['alpha_theta'] = band_power['alpha'] / (band_power['theta'] + 1e-10)

    # Alpha/Beta ratio (relaxation vs alertness)
    if 'alpha' in band_power and 'beta' in band_power:
        ratios['alpha_beta'] = band_power['alpha'] / (band_power['beta'] + 1e-10)

    # (Theta + Alpha) / Beta
    if all(k in band_power for k in ['theta', 'alpha', 'beta']):
        ratios['slow_fast'] = (band_power['theta'] + band_power['alpha']) / (band_power['beta'] + 1e-10)

    return ratios


def extract_frequency_features(
    epochs: np.ndarray,
    sampling_rate: int = SAMPLING_RATE,
    features: Optional[List[str]] = None,
    bands: Optional[Dict[str, Tuple[float, float]]] = None
) -> np.ndarray:
    """
    Extract multiple frequency-domain features.

    Args:
        epochs: Shape (n_epochs, n_channels, n_samples)
        sampling_rate: Sampling rate in Hz
        features: List of features to extract. Options:
            - 'band_power', 'relative_band_power'
            - 'spectral_entropy', 'peak_freq', 'spectral_edge'
            - 'band_ratios'
            If None, extracts: ['band_power', 'spectral_entropy']
        bands: Frequency band definitions

    Returns:
        Feature matrix, shape (n_epochs, n_features)
    """
    if features is None:
        features = ['band_power', 'spectral_entropy']

    if bands is None:
        bands = FREQ_BANDS

    feature_arrays = []

    for feat in features:
        if feat == 'band_power':
            bp = compute_band_power(epochs, sampling_rate, bands, relative=False)
            for band_name in bands.keys():
                feature_arrays.append(bp[band_name])

        elif feat == 'relative_band_power':
            bp = compute_band_power(epochs, sampling_rate, bands, relative=True)
            for band_name in bands.keys():
                feature_arrays.append(bp[band_name])

        elif feat == 'spectral_entropy':
            feature_arrays.append(compute_spectral_entropy(epochs, sampling_rate))

        elif feat == 'peak_freq':
            feature_arrays.append(compute_peak_frequency(epochs, sampling_rate))

        elif feat == 'spectral_edge':
            feature_arrays.append(compute_spectral_edge(epochs, sampling_rate))

        elif feat == 'band_ratios':
            ratios = compute_band_ratios(epochs, sampling_rate, bands)
            for ratio_name in ratios.keys():
                feature_arrays.append(ratios[ratio_name])

        else:
            raise ValueError(f"Unknown feature: {feat}")

    # Stack all features
    all_features = np.concatenate(feature_arrays, axis=1)

    return all_features


def get_frequency_feature_names(
    n_channels: int,
    features: Optional[List[str]] = None,
    bands: Optional[Dict[str, Tuple[float, float]]] = None,
    channel_names: Optional[List[str]] = None
) -> List[str]:
    """
    Generate feature names for frequency features.

    Args:
        n_channels: Number of channels
        features: List of extracted features
        bands: Frequency band definitions
        channel_names: Optional channel names

    Returns:
        List of feature names
    """
    if features is None:
        features = ['band_power', 'spectral_entropy']

    if bands is None:
        bands = FREQ_BANDS

    if channel_names is None:
        channel_names = [f"ch{i}" for i in range(n_channels)]

    names = []

    for feat in features:
        if feat in ['band_power', 'relative_band_power']:
            prefix = 'rel_' if 'relative' in feat else ''
            for band_name in bands.keys():
                for ch in channel_names:
                    names.append(f"{ch}_{prefix}{band_name}_power")

        elif feat == 'spectral_entropy':
            for ch in channel_names:
                names.append(f"{ch}_spectral_entropy")

        elif feat == 'peak_freq':
            for ch in channel_names:
                names.append(f"{ch}_peak_freq")

        elif feat == 'spectral_edge':
            for ch in channel_names:
                names.append(f"{ch}_spectral_edge")

        elif feat == 'band_ratios':
            ratio_names = ['theta_beta', 'alpha_theta', 'alpha_beta', 'slow_fast']
            for ratio in ratio_names:
                for ch in channel_names:
                    names.append(f"{ch}_{ratio}_ratio")

    return names
