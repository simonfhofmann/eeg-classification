# ----------------------------------------------------------------------
# loader.py
#
# Functions for loading EEG data from .mat files and behavioral data from CSV.
# ----------------------------------------------------------------------

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.io import loadmat
import h5py
import mne
from typing import Dict, Tuple, Optional, Union, List

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import (
    LOGS_DIR, MAT_DATA_DIR, MARKERS, SAMPLING_RATE,
    MAT_VARIABLE_NAME, DATA_SCALE_FACTOR,
    RAW_EPOCH_TMIN, BASELINE_CORRECTION_TMIN, BASELINE_CORRECTION_TMAX,
    EPOCH_TMIN, EPOCH_TMAX
)


def load_mat_file(filepath: Union[str, Path], variable_name: Optional[str] = None) -> Dict:
    """
    Load a .mat file, handling both v7.3 (HDF5) and earlier formats.

    Args:
        filepath: Path to the .mat file
        variable_name: Specific variable to extract (optional)

    Returns:
        Dictionary containing the .mat file contents

    Note:
        MATLAB v7.3 files use HDF5 format and require h5py.
        The EEGLAB preprocessing script saves with '-v7.3' flag.
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"MAT file not found: {filepath}")

    try:
        # Try scipy first (works for v7.2 and earlier)
        data = loadmat(str(filepath), squeeze_me=True, struct_as_record=False)

        # Remove MATLAB metadata keys
        data = {k: v for k, v in data.items() if not k.startswith('__')}

    except NotImplementedError:
        # v7.3 files need h5py
        data = _load_mat_v73(filepath)

    if variable_name:
        if variable_name not in data:
            raise KeyError(f"Variable '{variable_name}' not found in {filepath}")
        return data[variable_name]

    return data


def _load_mat_v73(filepath: Path) -> Dict:
    """
    Load MATLAB v7.3 (HDF5) files using h5py.

    Args:
        filepath: Path to the .mat file

    Returns:
        Dictionary containing the data
    """
    data = {}

    with h5py.File(filepath, 'r') as f:
        for key in f.keys():
            if key.startswith('#'):  # Skip HDF5 metadata
                continue
            data[key] = _h5py_to_numpy(f[key])

    return data


def _h5py_to_numpy(item) -> Union[np.ndarray, Dict, str]:
    """
    Recursively convert h5py objects to numpy arrays or dicts.
    """
    if isinstance(item, h5py.Dataset):
        arr = item[()]
        # Handle MATLAB char arrays
        if arr.dtype == 'uint16':
            try:
                return ''.join(chr(c) for c in arr.flatten())
            except:
                pass
        return arr
    elif isinstance(item, h5py.Group):
        return {k: _h5py_to_numpy(v) for k, v in item.items()}
    else:
        return item


def load_eeg_from_mat(filepath: Union[str, Path]) -> Dict:
    """
    Load and parse EEG data from preprocessed .mat file.

    Handles the specific structure from EEGLAB preprocessing:
    - cellArray contains the main struct with fields: data, srate, xmin, chanlocs
    - data shape: (channels, timepoints, trials)
    - Channel names are nested in chanlocs struct

    Args:
        filepath: Path to the .mat file

    Returns:
        Dictionary containing:
            - 'data': numpy array (trials, channels, timepoints)
            - 'sfreq': sampling frequency (int)
            - 'tmin': epoch start time (float)
            - 'ch_names': list of channel names
    """
    filepath = Path(filepath)
    mat = loadmat(str(filepath))

    # Get the main struct from cellArray
    main_struct = mat[MAT_VARIABLE_NAME][0, 0]

    # Extract raw data - shape: (channels, timepoints, trials)
    raw_data = main_struct['data'][0, 0]

    # Extract sampling rate
    sfreq = main_struct['srate'][0, 0].item()

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


def create_mne_epochs(
    eeg_data: Dict,
    apply_baseline: bool = True,
    crop_epochs: bool = True,
    baseline: Optional[Tuple[float, float]] = None,
    tmin: Optional[float] = None,
    tmax: Optional[float] = None
) -> mne.EpochsArray:
    """
    Create MNE EpochsArray from loaded EEG data.

    Applies baseline correction and cropping to extract stimulus-related data.

    Args:
        eeg_data: Dictionary from load_eeg_from_mat()
        apply_baseline: Whether to apply baseline correction
        crop_epochs: Whether to crop to stimulus period
        baseline: Baseline window (tmin, tmax). Defaults to config values.
        tmin: Crop start time. Defaults to config EPOCH_TMIN.
        tmax: Crop end time. Defaults to config EPOCH_TMAX.

    Returns:
        MNE EpochsArray object
    """
    # Convert to volts (data is in microvolts)
    data_volts = eeg_data['data'] * DATA_SCALE_FACTOR

    # Create MNE info object
    info = mne.create_info(
        ch_names=eeg_data['ch_names'],
        sfreq=eeg_data['sfreq'],
        ch_types='eeg'
    )

    # Create epochs with original tmin from the data
    epochs = mne.EpochsArray(data_volts, info, tmin=eeg_data['tmin'], verbose=False)

    # Apply baseline correction
    if apply_baseline:
        bl = baseline or (BASELINE_CORRECTION_TMIN, BASELINE_CORRECTION_TMAX)
        epochs.apply_baseline(baseline=bl, verbose=False)

    # Crop to stimulus period (removes pre-stimulus baseline and hardware artifact)
    if crop_epochs:
        crop_tmin = tmin if tmin is not None else EPOCH_TMIN
        crop_tmax = tmax if tmax is not None else EPOCH_TMAX
        epochs = epochs.crop(tmin=crop_tmin, tmax=crop_tmax)

    return epochs


def load_subject_eeg(
    filepath: Union[str, Path],
    apply_baseline: bool = True,
    crop_epochs: bool = True
) -> Tuple[np.ndarray, List[str], int]:
    """
    Load EEG data for a subject and return processed numpy array.

    Convenience function that combines loading and MNE processing.

    Args:
        filepath: Path to .mat file
        apply_baseline: Whether to apply baseline correction
        crop_epochs: Whether to crop to stimulus period

    Returns:
        Tuple of:
            - data: numpy array (trials, channels, timepoints)
            - ch_names: list of channel names
            - sfreq: sampling frequency
    """
    # Load raw data
    eeg_data = load_eeg_from_mat(filepath)

    # Create MNE epochs with processing
    epochs = create_mne_epochs(
        eeg_data,
        apply_baseline=apply_baseline,
        crop_epochs=crop_epochs
    )

    # Extract processed data
    data = epochs.get_data()

    return data, eeg_data['ch_names'], eeg_data['sfreq']


def load_behavioral_data(participant_id: str, logs_dir: Optional[Path] = None) -> pd.DataFrame:
    """
    Load behavioral data (ratings) for a participant.

    Args:
        participant_id: Participant identifier (e.g., "sub-001")
        logs_dir: Path to logs directory (uses config default if None)

    Returns:
        DataFrame with columns: participant_id, trial_num, stimulus_name,
                               origin_pool, familiarity_rating, liking_rating
    """
    logs_dir = Path(logs_dir) if logs_dir else LOGS_DIR

    data_file = logs_dir / f"{participant_id}_data.csv"

    if not data_file.exists():
        raise FileNotFoundError(f"Behavioral data file not found: {data_file}")

    df = pd.read_csv(data_file)

    # Validate expected columns
    expected_cols = ["participant_id", "trial_num", "stimulus_name",
                     "origin_pool", "familiarity_rating", "liking_rating"]

    missing = set(expected_cols) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in behavioral data: {missing}")

    return df


def load_stimulus_order(participant_id: str, logs_dir: Optional[Path] = None) -> pd.DataFrame:
    """
    Load stimulus order file for a participant.

    Args:
        participant_id: Participant identifier
        logs_dir: Path to logs directory

    Returns:
        DataFrame with stimulus order and pool labels
    """
    logs_dir = Path(logs_dir) if logs_dir else LOGS_DIR

    order_file = logs_dir / f"{participant_id}_stimulus_order.csv"

    if not order_file.exists():
        raise FileNotFoundError(f"Stimulus order file not found: {order_file}")

    return pd.read_csv(order_file)


def load_subject_data(
    participant_id: str,
    mat_filepath: Union[str, Path],
    logs_dir: Optional[Path] = None,
    apply_baseline: bool = True,
    crop_epochs: bool = True,
    return_mne_epochs: bool = False,
    exclude_trials: Optional[List[int]] = None
) -> Dict:
    """
    Load and merge EEG data with behavioral data for a single subject.

    This is the main entry point for loading a subject's complete dataset.

    Args:
        participant_id: Participant identifier (e.g., "sub-001")
        mat_filepath: Path to preprocessed .mat file
        logs_dir: Path to behavioral logs directory
        apply_baseline: Whether to apply baseline correction
        crop_epochs: Whether to crop to stimulus period
        return_mne_epochs: If True, also return MNE EpochsArray object
        exclude_trials: List of trial indices to exclude (0-indexed).
                       Both EEG and behavioral data will be filtered.

    Returns:
        Dictionary containing:
            - 'eeg_data': Processed EEG data array (trials, channels, timepoints)
            - 'ch_names': List of channel names
            - 'sfreq': Sampling frequency
            - 'behavioral': DataFrame with behavioral responses
            - 'stimulus_order': DataFrame with stimulus order
            - 'participant_id': Participant ID string
            - 'metadata': Additional info extracted from the data
            - 'epochs': MNE EpochsArray (if return_mne_epochs=True)
            - 'excluded_trials': List of excluded trial indices (if any)
    """
    # Load and process EEG data using new functions
    raw_eeg = load_eeg_from_mat(mat_filepath)
    epochs = create_mne_epochs(
        raw_eeg,
        apply_baseline=apply_baseline,
        crop_epochs=crop_epochs
    )
    eeg_data = epochs.get_data()

    # Load behavioral data
    behavioral = load_behavioral_data(participant_id, logs_dir)

    # Load stimulus order
    try:
        stimulus_order = load_stimulus_order(participant_id, logs_dir)
    except FileNotFoundError:
        stimulus_order = None

    # Apply trial exclusion if specified
    n_trials_original = raw_eeg['n_trials']
    if exclude_trials is not None and len(exclude_trials) > 0:
        # Create mask for trials to keep
        mask = np.ones(n_trials_original, dtype=bool)
        mask[exclude_trials] = False

        # Filter EEG data
        eeg_data = eeg_data[mask]

        # Filter behavioral data
        behavioral = behavioral[mask].reset_index(drop=True)

        # Filter stimulus order if available
        if stimulus_order is not None:
            stimulus_order = stimulus_order[mask].reset_index(drop=True)

        # Filter MNE epochs if needed
        if return_mne_epochs:
            epochs = epochs[mask]

    # Extract metadata
    metadata = {
        'n_trials': eeg_data.shape[0],
        'n_trials_original': n_trials_original,
        'n_trials_excluded': len(exclude_trials) if exclude_trials else 0,
        'n_channels': raw_eeg['n_channels'],
        'n_samples_raw': raw_eeg['n_samples'],
        'n_samples_processed': eeg_data.shape[2],
        'original_tmin': raw_eeg['tmin'],
    }

    result = {
        'eeg_data': eeg_data,
        'ch_names': raw_eeg['ch_names'],
        'sfreq': raw_eeg['sfreq'],
        'behavioral': behavioral,
        'stimulus_order': stimulus_order,
        'participant_id': participant_id,
        'metadata': metadata,
    }

    if exclude_trials:
        result['excluded_trials'] = exclude_trials

    if return_mne_epochs:
        result['epochs'] = epochs

    return result


def create_labels(
    behavioral_df: pd.DataFrame,
    target_type: str = "familiarity_binary",
    threshold: int = 3
) -> np.ndarray:
    """
    Create classification labels from behavioral data.

    Args:
        behavioral_df: DataFrame with behavioral responses
        target_type: Type of labels to create:
            - "familiarity_binary": 0 (unfamiliar, <=threshold) vs 1 (familiar, >threshold)
            - "liking_binary": 0 (disliked) vs 1 (liked)
            - "origin_pool": 0 (unfamiliar pool) vs 1 (familiar pool)
            - "familiarity_multiclass": 0-4 (ratings 1-5)
        threshold: Threshold for binary classification (ratings <= threshold -> 0)

    Returns:
        numpy array of labels
    """
    if target_type == "familiarity_binary":
        labels = (behavioral_df["familiarity_rating"] > threshold).astype(int).values

    elif target_type == "liking_binary":
        labels = (behavioral_df["liking_rating"] > threshold).astype(int).values

    elif target_type == "origin_pool":
        # Assumes origin_pool column contains "familiar" or "unfamiliar"
        labels = (behavioral_df["origin_pool"].str.lower() == "familiar").astype(int).values

    elif target_type == "familiarity_multiclass":
        labels = (behavioral_df["familiarity_rating"] - 1).astype(int).values  # 0-4

    else:
        raise ValueError(f"Unknown target type: {target_type}")

    return labels


def list_available_subjects(mat_dir: Optional[Path] = None) -> List[str]:
    """
    List all available preprocessed subject files.

    Args:
        mat_dir: Directory containing .mat files

    Returns:
        List of subject IDs (extracted from filenames)
    """
    mat_dir = Path(mat_dir) if mat_dir else MAT_DATA_DIR

    if not mat_dir.exists():
        return []

    mat_files = list(mat_dir.glob("*.mat"))

    # Extract subject IDs from filenames (assumes format: sub-XXX_*.mat or similar)
    subjects = []
    for f in mat_files:
        # Try to extract subject ID from filename
        name = f.stem
        subjects.append(name)

    return sorted(subjects)
