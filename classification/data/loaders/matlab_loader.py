# ----------------------------------------------------------------------
# matlab_loader.py
#
# Loader for MATLAB/EEGLAB preprocessed .mat files.
# ----------------------------------------------------------------------

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.io import loadmat
import h5py
import mne
from typing import Dict, Optional, Union, List, Tuple

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from config import (
    MAT_VARIABLE_NAME, DATA_SCALE_FACTOR,
    BASELINE_CORRECTION_TMIN, BASELINE_CORRECTION_TMAX,
    EPOCH_TMIN, EPOCH_TMAX
)
from data.containers import EEGDataContainer
from data.loaders.base import BaseLoader


class MatlabPreprocessedLoader(BaseLoader):
    """
    Load EEG data from MATLAB/EEGLAB preprocessed .mat files.

    This loader handles the specific structure from EEGLAB preprocessing:
    - cellArray contains the main struct with fields: data, srate, xmin, chanlocs
    - Data shape in file: (channels, timepoints, trials)
    - Output shape: (trials, channels, timepoints)

    Args:
        apply_baseline: Whether to apply baseline correction
        crop_epochs: Whether to crop epochs to stimulus period
        baseline_window: Tuple of (tmin, tmax) for baseline correction
        epoch_window: Tuple of (tmin, tmax) for epoch cropping
        convert_to_volts: Whether to convert from microvolts to volts

    Example:
        >>> loader = MatlabPreprocessedLoader()
        >>> data = loader.load(
        ...     eeg_path="path/to/data_eeg.mat",
        ...     participant_id="sub-001",
        ...     behavioral_path="path/to/logs"
        ... )
        >>> print(data.shape)  # (n_trials, n_channels, n_timepoints)
    """

    def __init__(
        self,
        apply_baseline: bool = True,
        crop_epochs: bool = True,
        baseline_window: Optional[Tuple[float, float]] = None,
        epoch_window: Optional[Tuple[float, float]] = None,
        convert_to_volts: bool = True
    ):
        self.apply_baseline = apply_baseline
        self.crop_epochs = crop_epochs
        self.baseline_window = baseline_window or (BASELINE_CORRECTION_TMIN, BASELINE_CORRECTION_TMAX)
        self.epoch_window = epoch_window or (EPOCH_TMIN, EPOCH_TMAX)
        self.convert_to_volts = convert_to_volts

    def load(
        self,
        eeg_path: Union[str, Path],
        participant_id: str,
        behavioral_path: Optional[Union[str, Path]] = None,
    ) -> EEGDataContainer:
        """
        Load EEG data from MATLAB .mat file.

        Args:
            eeg_path: Path to .mat file
            participant_id: Participant identifier
            behavioral_path: Optional path to behavioral logs directory

        Returns:
            EEGDataContainer with loaded data
        """
        eeg_path = Path(eeg_path)
        self.validate_paths(eeg_path)

        # Load raw EEG data from .mat file
        raw_eeg = self._load_mat_file(eeg_path)

        # Create MNE epochs and apply preprocessing
        epochs = self._create_mne_epochs(raw_eeg)

        # Extract processed data
        X = epochs.get_data()  # Shape: (n_trials, n_channels, n_timepoints)

        # Build metadata
        metadata = {
            'source': 'matlab_preprocessed',
            'original_tmin': raw_eeg['tmin'],
            'baseline_window': self.baseline_window if self.apply_baseline else None,
            'epoch_window': self.epoch_window if self.crop_epochs else None,
            'n_samples_original': raw_eeg['n_samples'],
            'mat_file': str(eeg_path),
        }

        container = EEGDataContainer(
            X=X.astype(np.float32),
            y=None,  # Labels set separately
            sfreq=raw_eeg['sfreq'],
            ch_names=raw_eeg['ch_names'],
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
        Load behavioral data (ratings) for a participant.

        Args:
            behavioral_path: Path to logs directory
            participant_id: Participant identifier

        Returns:
            DataFrame with behavioral data
        """
        behavioral_path = Path(behavioral_path)

        # Try different file naming conventions
        possible_files = [
            behavioral_path / f"{participant_id}_data.csv",
            behavioral_path / f"{participant_id}.csv",
            behavioral_path,  # If path is directly to file
        ]

        data_file = None
        for f in possible_files:
            if f.exists() and f.is_file():
                data_file = f
                break

        if data_file is None:
            raise FileNotFoundError(
                f"Behavioral data file not found for {participant_id} in {behavioral_path}"
            )

        df = pd.read_csv(data_file)

        # Validate expected columns
        expected_cols = ["trial_num", "familiarity_rating", "liking_rating"]
        missing = [col for col in expected_cols if col not in df.columns]
        if missing:
            print(f"Warning: Missing columns in behavioral data: {missing}")

        return df

    def _load_mat_file(self, filepath: Path) -> Dict:
        """
        Load and parse EEG data from preprocessed .mat file.

        Args:
            filepath: Path to .mat file

        Returns:
            Dictionary with data, sfreq, tmin, ch_names, etc.
        """
        try:
            # Try scipy first (works for v7.2 and earlier)
            mat = loadmat(str(filepath))
        except NotImplementedError:
            # v7.3 files need h5py
            mat = self._load_mat_v73(filepath)

        # Get the main struct from cellArray
        main_struct = mat[MAT_VARIABLE_NAME][0, 0]

        # Extract raw data - shape: (channels, timepoints, trials)
        raw_data = main_struct['data'][0, 0]

        # Extract sampling rate
        sfreq = int(main_struct['srate'][0, 0].item())

        # Extract epoch start time (tmin)
        tmin = main_struct['xmin'][0, 0].item()

        # Extract and clean channel names
        chan_structs = main_struct['chanlocs'][0, 0]
        raw_labels = [ch['labels'][0] for ch in chan_structs[0]]

        ch_names = []
        for label in raw_labels:
            if isinstance(label, np.ndarray):
                label = label.item()
            if isinstance(label, np.ndarray):
                label = label.item()
            ch_names.append(str(label))

        # Transpose to (trials, channels, timepoints) for MNE compatibility
        data = np.transpose(raw_data, (2, 0, 1))

        return {
            'data': data,
            'sfreq': sfreq,
            'tmin': tmin,
            'ch_names': ch_names,
            'n_trials': data.shape[0],
            'n_channels': data.shape[1],
            'n_samples': data.shape[2]
        }

    def _load_mat_v73(self, filepath: Path) -> Dict:
        """Load MATLAB v7.3 (HDF5) files using h5py."""
        data = {}
        with h5py.File(filepath, 'r') as f:
            for key in f.keys():
                if key.startswith('#'):
                    continue
                data[key] = self._h5py_to_numpy(f[key])
        return data

    def _h5py_to_numpy(self, item):
        """Recursively convert h5py objects to numpy arrays."""
        if isinstance(item, h5py.Dataset):
            arr = item[()]
            if arr.dtype == 'uint16':
                try:
                    return ''.join(chr(c) for c in arr.flatten())
                except:
                    pass
            return arr
        elif isinstance(item, h5py.Group):
            return {k: self._h5py_to_numpy(v) for k, v in item.items()}
        return item

    def _create_mne_epochs(self, eeg_data: Dict) -> mne.EpochsArray:
        """
        Create MNE EpochsArray from loaded EEG data.

        Applies baseline correction and cropping.

        Args:
            eeg_data: Dictionary from _load_mat_file()

        Returns:
            MNE EpochsArray object
        """
        # Convert to volts if needed
        if self.convert_to_volts:
            data = eeg_data['data'] * DATA_SCALE_FACTOR
        else:
            data = eeg_data['data']

        # Create MNE info object
        info = mne.create_info(
            ch_names=eeg_data['ch_names'],
            sfreq=eeg_data['sfreq'],
            ch_types='eeg'
        )

        # Create epochs with original tmin from the data
        epochs = mne.EpochsArray(data, info, tmin=eeg_data['tmin'], verbose=False)

        # Apply baseline correction
        if self.apply_baseline:
            epochs.apply_baseline(baseline=self.baseline_window, verbose=False)

        # Crop to stimulus period
        if self.crop_epochs:
            epochs = epochs.crop(
                tmin=self.epoch_window[0],
                tmax=self.epoch_window[1]
            )

        return epochs
