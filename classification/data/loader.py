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
from typing import Dict, Tuple, Optional, Union, List

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import LOGS_DIR, MAT_DATA_DIR, MARKERS


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
    eeg_variable_name: str = "data_eeg",
    logs_dir: Optional[Path] = None
) -> Dict:
    """
    Load and merge EEG data with behavioral data for a single subject.

    This is the main entry point for loading a subject's complete dataset.

    Args:
        participant_id: Participant identifier (e.g., "sub-001")
        mat_filepath: Path to preprocessed .mat file
        eeg_variable_name: Name of the EEG data variable in the .mat file
        logs_dir: Path to behavioral logs directory

    Returns:
        Dictionary containing:
            - 'eeg_data': Raw EEG data from MATLAB (structure depends on preprocessing)
            - 'behavioral': DataFrame with behavioral responses
            - 'stimulus_order': DataFrame with stimulus order
            - 'participant_id': Participant ID string
            - 'metadata': Additional info extracted from the data
    """
    # Load EEG data
    mat_data = load_mat_file(mat_filepath)

    if eeg_variable_name not in mat_data:
        available = list(mat_data.keys())
        raise KeyError(
            f"Variable '{eeg_variable_name}' not found. Available: {available}"
        )

    eeg_data = mat_data[eeg_variable_name]

    # Load behavioral data
    behavioral = load_behavioral_data(participant_id, logs_dir)

    # Load stimulus order
    try:
        stimulus_order = load_stimulus_order(participant_id, logs_dir)
    except FileNotFoundError:
        stimulus_order = None

    # Extract metadata if available
    metadata = {
        'n_trials': len(behavioral),
        'mat_keys': list(mat_data.keys()),
    }

    return {
        'eeg_data': eeg_data,
        'behavioral': behavioral,
        'stimulus_order': stimulus_order,
        'participant_id': participant_id,
        'metadata': metadata,
    }


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
