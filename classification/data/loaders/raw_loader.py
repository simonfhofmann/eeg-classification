# ----------------------------------------------------------------------
# raw_loader.py
#
# Loader for raw EEG data with Braindecode/MNE preprocessing.
# ----------------------------------------------------------------------

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, Union, List, Tuple, Any
import mne

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from config import SAMPLING_RATE, EPOCH_TMIN, EPOCH_TMAX
from data.containers import EEGDataContainer
from data.loaders.base import BaseLoader


class RawEEGLoader(BaseLoader):
    """
    Load raw EEG data and preprocess using MNE/Braindecode.

    This loader is for raw EEG files (.edf, .bdf, .fif, .vhdr, etc.)
    that have not been preprocessed in MATLAB.

    The preprocessing pipeline includes:
    - Loading raw data
    - Filtering (bandpass, notch)
    - Resampling
    - Epoching based on events
    - Baseline correction
    - Optional: ICA artifact removal

    Args:
        preprocessing_config: Dictionary with preprocessing parameters:
            - l_freq: High-pass filter frequency (Hz)
            - h_freq: Low-pass filter frequency (Hz)
            - notch_freq: Notch filter frequency (Hz), None to skip
            - resample_freq: Target sampling frequency (Hz), None to keep original
            - epoch_tmin: Epoch start time relative to event (seconds)
            - epoch_tmax: Epoch end time relative to event (seconds)
            - baseline: Tuple (tmin, tmax) for baseline correction, None to skip
            - reject_bad_epochs: Whether to reject epochs with artifacts
            - reject_criteria: Dict with rejection thresholds

    Example:
        >>> config = {
        ...     'l_freq': 0.5,
        ...     'h_freq': 40.0,
        ...     'notch_freq': 50.0,
        ...     'resample_freq': 500,
        ...     'epoch_tmin': -0.5,
        ...     'epoch_tmax': 32.0,
        ...     'baseline': (-0.5, 0.0),
        ... }
        >>> loader = RawEEGLoader(preprocessing_config=config)
        >>> data = loader.load(
        ...     eeg_path="path/to/raw.edf",
        ...     participant_id="sub-001",
        ...     events_path="path/to/events.csv"
        ... )
    """

    DEFAULT_CONFIG = {
        'l_freq': 0.5,
        'h_freq': 45.0,
        'notch_freq': 50.0,
        'resample_freq': SAMPLING_RATE,
        'epoch_tmin': EPOCH_TMIN,
        'epoch_tmax': EPOCH_TMAX,
        'baseline': None,
        'reject_bad_epochs': False,
        'reject_criteria': {'eeg': 100e-6},  # 100 µV
        'event_id': None,  # Will use all events if None
    }

    def __init__(self, preprocessing_config: Optional[Dict[str, Any]] = None):
        self.config = {**self.DEFAULT_CONFIG}
        if preprocessing_config:
            self.config.update(preprocessing_config)

    def load(
        self,
        eeg_path: Union[str, Path],
        participant_id: str,
        behavioral_path: Optional[Union[str, Path]] = None,
        events: Optional[np.ndarray] = None,
        event_id: Optional[Dict[str, int]] = None,
    ) -> EEGDataContainer:
        """
        Load and preprocess raw EEG data.

        Args:
            eeg_path: Path to raw EEG file (.edf, .bdf, .fif, .vhdr)
            participant_id: Participant identifier
            behavioral_path: Optional path to behavioral data
            events: Optional events array (n_events, 3). If None, extracted from file.
            event_id: Optional event ID mapping. If None, uses all events.

        Returns:
            EEGDataContainer with preprocessed data
        """
        eeg_path = Path(eeg_path)
        self.validate_paths(eeg_path)

        # Load raw data
        raw = self._load_raw(eeg_path)

        # Apply preprocessing
        raw = self._preprocess_raw(raw)

        # Get or create events
        if events is None:
            events = self._extract_events(raw)

        # Create epochs
        epochs = self._create_epochs(raw, events, event_id)

        # Extract data
        X = epochs.get_data()  # Shape: (n_trials, n_channels, n_timepoints)

        # Build metadata
        metadata = {
            'source': 'raw_braindecode',
            'preprocessing_config': self.config.copy(),
            'raw_file': str(eeg_path),
            'n_events': len(events),
            'original_sfreq': raw.info['sfreq'],
        }

        container = EEGDataContainer(
            X=X.astype(np.float32),
            y=None,
            sfreq=int(epochs.info['sfreq']),
            ch_names=epochs.ch_names,
            participant_id=participant_id,
            metadata=metadata
        )

        return container

    def load_behavioral(
        self,
        behavioral_path: Union[str, Path],
        participant_id: str
    ) -> pd.DataFrame:
        """
        Load behavioral data.

        Args:
            behavioral_path: Path to behavioral data file
            participant_id: Participant identifier

        Returns:
            DataFrame with behavioral data
        """
        behavioral_path = Path(behavioral_path)
        self.validate_paths(behavioral_path)

        if behavioral_path.is_file():
            return pd.read_csv(behavioral_path)

        # Try to find file in directory
        possible_files = [
            behavioral_path / f"{participant_id}_data.csv",
            behavioral_path / f"{participant_id}.csv",
        ]

        for f in possible_files:
            if f.exists():
                return pd.read_csv(f)

        raise FileNotFoundError(
            f"Behavioral data not found for {participant_id} in {behavioral_path}"
        )

    def _load_raw(self, filepath: Path) -> mne.io.BaseRaw:
        """
        Load raw EEG data based on file extension.

        Args:
            filepath: Path to raw EEG file

        Returns:
            MNE Raw object
        """
        suffix = filepath.suffix.lower()

        if suffix == '.edf':
            raw = mne.io.read_raw_edf(filepath, preload=True, verbose=False)
        elif suffix == '.bdf':
            raw = mne.io.read_raw_bdf(filepath, preload=True, verbose=False)
        elif suffix in ['.fif', '.fif.gz']:
            raw = mne.io.read_raw_fif(filepath, preload=True, verbose=False)
        elif suffix == '.vhdr':
            raw = mne.io.read_raw_brainvision(filepath, preload=True, verbose=False)
        elif suffix == '.set':
            raw = mne.io.read_raw_eeglab(filepath, preload=True, verbose=False)
        else:
            raise ValueError(f"Unsupported file format: {suffix}")

        return raw

    def _preprocess_raw(self, raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
        """
        Apply preprocessing to raw data.

        Args:
            raw: MNE Raw object

        Returns:
            Preprocessed Raw object
        """
        # Pick only EEG channels
        raw = raw.pick_types(eeg=True, exclude='bads')

        # Apply bandpass filter
        if self.config['l_freq'] or self.config['h_freq']:
            raw = raw.filter(
                l_freq=self.config['l_freq'],
                h_freq=self.config['h_freq'],
                verbose=False
            )

        # Apply notch filter
        if self.config['notch_freq']:
            raw = raw.notch_filter(
                freqs=self.config['notch_freq'],
                verbose=False
            )

        # Resample if needed
        if self.config['resample_freq'] and raw.info['sfreq'] != self.config['resample_freq']:
            raw = raw.resample(self.config['resample_freq'], verbose=False)

        return raw

    def _extract_events(self, raw: mne.io.BaseRaw) -> np.ndarray:
        """
        Extract events from raw data.

        Args:
            raw: MNE Raw object

        Returns:
            Events array (n_events, 3)
        """
        try:
            events = mne.find_events(raw, verbose=False)
        except ValueError:
            # Try annotations if no stim channel
            events, _ = mne.events_from_annotations(raw, verbose=False)

        return events

    def _create_epochs(
        self,
        raw: mne.io.BaseRaw,
        events: np.ndarray,
        event_id: Optional[Dict[str, int]] = None
    ) -> mne.Epochs:
        """
        Create epochs from raw data and events.

        Args:
            raw: Preprocessed Raw object
            events: Events array
            event_id: Event ID mapping

        Returns:
            MNE Epochs object
        """
        # Use provided event_id or config or all events
        if event_id is None:
            event_id = self.config.get('event_id')

        # Set rejection criteria
        reject = None
        if self.config['reject_bad_epochs']:
            reject = self.config['reject_criteria']

        epochs = mne.Epochs(
            raw,
            events,
            event_id=event_id,
            tmin=self.config['epoch_tmin'],
            tmax=self.config['epoch_tmax'],
            baseline=self.config['baseline'],
            reject=reject,
            preload=True,
            verbose=False
        )

        return epochs


# Convenience function for quick loading
def load_raw_eeg(
    filepath: Union[str, Path],
    participant_id: str,
    config: Optional[Dict] = None,
    **kwargs
) -> EEGDataContainer:
    """
    Convenience function to load raw EEG data.

    Args:
        filepath: Path to raw EEG file
        participant_id: Participant identifier
        config: Preprocessing configuration
        **kwargs: Additional arguments passed to RawEEGLoader.load()

    Returns:
        EEGDataContainer with preprocessed data
    """
    loader = RawEEGLoader(preprocessing_config=config)
    return loader.load(filepath, participant_id, **kwargs)
