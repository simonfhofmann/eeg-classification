# ----------------------------------------------------------------------
# connectivity.py
#
# Connectivity-based feature extraction for EEG signals.
# (Optional advanced features)
# ----------------------------------------------------------------------

import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from scipy import signal

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import SAMPLING_RATE, FREQ_BANDS


def compute_correlation_matrix(epochs: np.ndarray) -> np.ndarray:
    """
    Compute correlation matrix between channels for each epoch.

    Args:
        epochs: Shape (n_epochs, n_channels, n_samples)

    Returns:
        Shape (n_epochs, n_channels, n_channels)
    """
    n_epochs, n_channels, _ = epochs.shape
    corr_matrices = np.zeros((n_epochs, n_channels, n_channels))

    for i in range(n_epochs):
        corr_matrices[i] = np.corrcoef(epochs[i])

    return corr_matrices


def compute_coherence(
    epochs: np.ndarray,
    sampling_rate: int = SAMPLING_RATE,
    band: Optional[Tuple[float, float]] = None,
    nperseg: Optional[int] = None
) -> np.ndarray:
    """
    Compute coherence between all channel pairs.

    Coherence measures the linear relationship between two signals
    in the frequency domain.

    Args:
        epochs: Shape (n_epochs, n_channels, n_samples)
        sampling_rate: Sampling rate in Hz
        band: Optional (low, high) to average coherence within
        nperseg: Segment length for coherence estimation

    Returns:
        Shape (n_epochs, n_channels, n_channels) - mean coherence matrix
    """
    n_epochs, n_channels, n_samples = epochs.shape

    if nperseg is None:
        nperseg = min(n_samples, sampling_rate)

    coherence_matrices = np.zeros((n_epochs, n_channels, n_channels))

    for ep in range(n_epochs):
        for i in range(n_channels):
            for j in range(i, n_channels):
                if i == j:
                    coherence_matrices[ep, i, j] = 1.0
                else:
                    freqs, coh = signal.coherence(
                        epochs[ep, i],
                        epochs[ep, j],
                        fs=sampling_rate,
                        nperseg=nperseg
                    )

                    if band is not None:
                        idx = np.logical_and(freqs >= band[0], freqs <= band[1])
                        coh_mean = np.mean(coh[idx])
                    else:
                        coh_mean = np.mean(coh)

                    coherence_matrices[ep, i, j] = coh_mean
                    coherence_matrices[ep, j, i] = coh_mean

    return coherence_matrices


def compute_phase_locking_value(
    epochs: np.ndarray,
    band: Tuple[float, float],
    sampling_rate: int = SAMPLING_RATE
) -> np.ndarray:
    """
    Compute Phase Locking Value (PLV) between channel pairs.

    PLV measures the consistency of phase difference between two signals
    across time, indicating functional connectivity.

    Args:
        epochs: Shape (n_epochs, n_channels, n_samples)
        band: (low, high) frequency band for filtering
        sampling_rate: Sampling rate in Hz

    Returns:
        Shape (n_epochs, n_channels, n_channels)
    """
    from scipy.signal import butter, filtfilt, hilbert

    n_epochs, n_channels, n_samples = epochs.shape

    # Design bandpass filter
    low, high = band
    nyq = sampling_rate / 2
    b, a = butter(4, [low / nyq, high / nyq], btype='band')

    plv_matrices = np.zeros((n_epochs, n_channels, n_channels))

    for ep in range(n_epochs):
        # Filter and get phase for each channel
        phases = np.zeros((n_channels, n_samples))
        for ch in range(n_channels):
            filtered = filtfilt(b, a, epochs[ep, ch])
            analytic = hilbert(filtered)
            phases[ch] = np.angle(analytic)

        # Compute PLV for each pair
        for i in range(n_channels):
            for j in range(i, n_channels):
                if i == j:
                    plv_matrices[ep, i, j] = 1.0
                else:
                    phase_diff = phases[i] - phases[j]
                    plv = np.abs(np.mean(np.exp(1j * phase_diff)))
                    plv_matrices[ep, i, j] = plv
                    plv_matrices[ep, j, i] = plv

    return plv_matrices


def extract_connectivity_features(
    epochs: np.ndarray,
    sampling_rate: int = SAMPLING_RATE,
    method: str = "correlation"
) -> np.ndarray:
    """
    Extract connectivity features (upper triangle of connectivity matrix).

    Args:
        epochs: Shape (n_epochs, n_channels, n_samples)
        sampling_rate: Sampling rate in Hz
        method: 'correlation', 'coherence', or 'plv'

    Returns:
        Feature matrix, shape (n_epochs, n_features)
        where n_features = n_channels * (n_channels - 1) / 2
    """
    if method == "correlation":
        conn_matrices = compute_correlation_matrix(epochs)
    elif method == "coherence":
        conn_matrices = compute_coherence(epochs, sampling_rate)
    elif method == "plv":
        # Use alpha band by default for PLV
        conn_matrices = compute_phase_locking_value(
            epochs, band=(8, 13), sampling_rate=sampling_rate
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    # Extract upper triangle (excluding diagonal)
    n_epochs, n_channels, _ = conn_matrices.shape
    triu_idx = np.triu_indices(n_channels, k=1)

    features = np.zeros((n_epochs, len(triu_idx[0])))
    for i in range(n_epochs):
        features[i] = conn_matrices[i][triu_idx]

    return features


def get_connectivity_feature_names(
    channel_names: List[str],
    method: str = "correlation"
) -> List[str]:
    """
    Generate feature names for connectivity features.

    Args:
        channel_names: List of channel names
        method: Connectivity method used

    Returns:
        List of feature names
    """
    n_channels = len(channel_names)
    triu_idx = np.triu_indices(n_channels, k=1)

    names = []
    for i, j in zip(triu_idx[0], triu_idx[1]):
        names.append(f"{channel_names[i]}_{channel_names[j]}_{method}")

    return names
