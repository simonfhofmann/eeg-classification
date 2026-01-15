# ----------------------------------------------------------------------
# features/custom/__init__.py
#
# Custom feature extractors.
# ----------------------------------------------------------------------

from .audio_correlation import AudioEEGCorrelationExtractor

__all__ = [
    'AudioEEGCorrelationExtractor',
]
