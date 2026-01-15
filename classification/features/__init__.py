# ----------------------------------------------------------------------
# Feature extraction module
# ----------------------------------------------------------------------

from .time_domain import (
    extract_time_features,
    compute_variance,
    compute_hjorth_parameters,
    compute_zero_crossings,
)
from .frequency_domain import (
    extract_frequency_features,
    compute_psd,
    compute_band_power,
    compute_spectral_entropy,
)
from .base import FeatureExtractor, CompositeExtractor, ChannelWiseExtractor
from .pipeline import FeaturePipeline, FeatureSelector

__all__ = [
    # Base classes
    "FeatureExtractor",
    "CompositeExtractor",
    "ChannelWiseExtractor",
    # Pipeline
    "FeaturePipeline",
    "FeatureSelector",
    # Time domain
    "extract_time_features",
    "compute_variance",
    "compute_hjorth_parameters",
    "compute_zero_crossings",
    # Frequency domain
    "extract_frequency_features",
    "compute_psd",
    "compute_band_power",
    "compute_spectral_entropy",
]
