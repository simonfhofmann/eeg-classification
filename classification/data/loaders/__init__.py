# ----------------------------------------------------------------------
# data/loaders/__init__.py
#
# EEG data loaders for different preprocessing pipelines.
# ----------------------------------------------------------------------

from .base import BaseLoader
from .matlab_loader import MatlabPreprocessedLoader
from .raw_loader import RawEEGLoader, load_raw_eeg

__all__ = [
    'BaseLoader',
    'MatlabPreprocessedLoader',
    'RawEEGLoader',
    'load_raw_eeg',
]
