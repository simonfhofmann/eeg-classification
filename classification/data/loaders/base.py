# ----------------------------------------------------------------------
# base.py
#
# Abstract base class for EEG data loaders.
# ----------------------------------------------------------------------

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Union
import pandas as pd

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from data.containers import EEGDataContainer


class BaseLoader(ABC):
    """
    Abstract base class for EEG data loaders.

    All loaders must implement the `load` method and return an EEGDataContainer.
    This ensures a common interface regardless of the data source or
    preprocessing pipeline used.

    Subclasses:
        - MatlabPreprocessedLoader: Load from MATLAB/EEGLAB preprocessed .mat files
        - RawEEGLoader: Load raw EEG and preprocess with Braindecode/MNE
    """

    @abstractmethod
    def load(
        self,
        eeg_path: Union[str, Path],
        participant_id: str,
        behavioral_path: Optional[Union[str, Path]] = None,
    ) -> EEGDataContainer:
        """
        Load EEG data and return as EEGDataContainer.

        Args:
            eeg_path: Path to EEG data file
            participant_id: Participant identifier
            behavioral_path: Optional path to behavioral data (for labels)

        Returns:
            EEGDataContainer with loaded and preprocessed data
        """
        pass

    @abstractmethod
    def load_behavioral(
        self,
        behavioral_path: Union[str, Path],
        participant_id: str
    ) -> pd.DataFrame:
        """
        Load behavioral data (ratings, responses).

        Args:
            behavioral_path: Path to behavioral data file or directory
            participant_id: Participant identifier

        Returns:
            DataFrame with behavioral data
        """
        pass

    def validate_paths(self, *paths: Union[str, Path]) -> None:
        """
        Validate that all provided paths exist.

        Args:
            *paths: Paths to validate

        Raises:
            FileNotFoundError: If any path does not exist
        """
        for path in paths:
            if path is not None:
                path = Path(path)
                if not path.exists():
                    raise FileNotFoundError(f"Path not found: {path}")
