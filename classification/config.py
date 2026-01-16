# ----------------------------------------------------------------------
# config.py
#
# Configuration for EEG classification pipeline.
# ----------------------------------------------------------------------

import os
from pathlib import Path

# --------------------------
# 1. PATHS
# --------------------------
BASE_DIR = Path(__file__).parent.parent
PROJECT_ROOT = BASE_DIR.parent

# Data paths
DATA_DIR = PROJECT_ROOT / "raw_eeg"
RESULTS_DIR = BASE_DIR / "results"
MODELS_DIR = BASE_DIR / "models"

# Path to preprocessed .mat files (adjust as needed)
MAT_DATA_DIR = PROJECT_ROOT / "preprocessed_data"


# Path to behavioral logs from experiment
LOGS_DIR = PROJECT_ROOT / "recording_logs"

# --------------------------
# 2. EEG PARAMETERS
# --------------------------
# Sampling rate after preprocessing (from MATLAB script - resampled to 500 Hz)
SAMPLING_RATE = 500  # Hz

# Channel count (after removing EOG, HR, GSR channels in MATLAB preprocessing)
N_CHANNELS = 30

# Stimulus duration from experiment
STIMULUS_DURATION = 32.0  # seconds

# Frequency bands for feature extraction
FREQ_BANDS = {
    "delta": (0.5, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 60),
}

# --------------------------
# 2.1 MAT FILE STRUCTURE
# --------------------------
# Variable name containing EEG data in .mat file
MAT_VARIABLE_NAME = "cellArray"

# Data is stored in microvolts, convert to volts for MNE
DATA_SCALE_FACTOR = 1e-6

# --------------------------
# 3. EXPERIMENT MARKERS
# --------------------------
# Marker codes from BrainVision annotations (Stimulus/S X -> code X)
MARKERS = {
    # --- Trial Phases ---
    "BASELINE_START": 1,      # S1: baseline period starts
    "STIMULUS_START": 2,      # S2: stimulus (music) starts - use this for epoching
    "STIMULUS_END": 5,        # S5: stimulus ends (default for most participants)
    "STIMULUS_END_ALT": 6,    # S6: stimulus ends (Sub01/Yannick only)

    # --- Experiment Control ---
    "BREAK": 3,               # S3: break marker
    "EXPERIMENT_RESUME": 14,  # S14: experiment resumed after crash
    "EXPERIMENT_END": 15,     # S15: experiment ended
}

# --------------------------
# 3.1 PARTICIPANT-SPECIFIC INFO
# --------------------------
# Information about recording issues, crashes, and special handling needed
PARTICIPANT_INFO = {
    "Sub01": {
        "name": "yannick",
        "eeg_file": "Yanick.vhdr",
        "log_file": "yannick_data.csv",
        "stimulus_end_marker": 6,  # Uses S6 instead of S5
        "crashes": [],  # No crashes
        "notes": "Different protocol - 60 trials, longer baseline (~8s)",
    },
    "Sub02": {
        "name": "daniel",
        "eeg_file": "daniel_1_eeg.vhdr",
        "log_file": "daniel_data.csv",
        "stimulus_end_marker": 5,
        "crashes": [],
        "notes": "Has duplicate stimuli in log (Avicii played 5x)",
    },
    "Sub03": {
        "name": "simon",
        "eeg_file": "Simon.vhdr",
        "log_file": "simon_data.csv",
        "stimulus_end_marker": 5,
        "crashes": [
            {
                "after_eeg_trial": 31,  # Crash occurred after this EEG trial
                "repeated_eeg_trial": 32,  # This EEG trial is a repeat (exclude)
                "resume_marker": 14,  # S14 marker indicates resume
            }
        ],
        "notes": "Program crashed at trial 31, trial repeated",
    },
    "Sub04": {
        "name": "karsten",
        "eeg_file": "karsten.vhdr",
        "log_file": "karsten_data.csv",
        "stimulus_end_marker": 5,
        "crashes": [],
        "notes": "",
    },
    "Sub05": {
        "name": "philipp",
        "eeg_file": "philipp.vhdr",
        "log_file": "philipp_data.csv",
        "stimulus_end_marker": 5,
        "crashes": [],
        "notes": "",
    },
}

# --------------------------
# 4. CLASSIFICATION SETTINGS
# --------------------------
# Target variable options
TARGET_TYPES = {
    "familiarity_binary": {
        "description": "Binary: familiar (4-5) vs unfamiliar (1-2)",
        "threshold": 3,
    },
    "liking_binary": {
        "description": "Binary: liked (4-5) vs disliked (1-2)",
        "threshold": 3,
    },
    "origin_pool": {
        "description": "Binary: from familiar vs unfamiliar genre pool",
    },
    "familiarity_multiclass": {
        "description": "5-class: ratings 1-5",
    },
}

# Default target
DEFAULT_TARGET = "familiarity_binary"

# --------------------------
# 5. PREPROCESSING SETTINGS
# --------------------------
# Epoch time window as stored in .mat file (relative to stimulus onset)
# Data includes 3s pre-stimulus baseline
RAW_EPOCH_TMIN = -3.0   # Start of epoch (baseline start)
RAW_EPOCH_TMAX = 32.0   # End of epoch (stimulus end)

# Epoch time window for model input (after cropping)
EPOCH_TMIN = 0.1    # Start slightly after stimulus onset (to avoid hardware artifact)
EPOCH_TMAX = 32.0   # End of epoch (stimulus duration)

# Baseline period (from marker reference: ~8 seconds jittered)
BASELINE_DURATION_MEAN = 8.0
BASELINE_DURATION_JITTER = 0.75

# Baseline correction window (relative to stimulus onset)
# Using pre-stimulus period, stopping before stimulus onset to avoid artifact
BASELINE_CORRECTION_TMIN = -3.0
BASELINE_CORRECTION_TMAX = -0.1

# --------------------------
# 6. CROSS-VALIDATION
# --------------------------
N_FOLDS = 5
RANDOM_STATE = 42

# --------------------------
# 7. MODEL HYPERPARAMETERS (Defaults)
# --------------------------
SVM_PARAMS = {
    "kernel": "rbf",
    "C": 1.0,
    "gamma": "scale",
}

LDA_PARAMS = {
    "solver": "svd",
}

# Deep learning
DL_PARAMS = {
    "batch_size": 32,
    "epochs": 100,
    "learning_rate": 1e-3,
    "early_stopping_patience": 10,
}
