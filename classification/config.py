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
LOGS_DIR = PROJECT_ROOT / "logs"

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
# Ensure they match recording markers!
MARKERS = {
    """
    # --- Experiment Control ---
    "EXPERIMENT_START": 1,
    "EXPERIMENT_END": 2,
    "BREAK_START": 3,
    "BREAK_END": 4,

    # --- Trial Phases ---
    "BASELINE_START": 10,
    "STIMULUS_START": 20,
    "STIMULUS_END": 30,
    """
    # --- Experiment Control ---
    "EXPERIMENT_START": 254,
    "EXPERIMENT_END": 255,
    "BREAK_START": 99, # not used in current setup
    "BREAK_END": 99, # not used in current setup

    # --- Trial Phases ---
    "BASELINE_START": 1,
    "STIMULUS_START": 2,
    "STIMULUS_END": 3,
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
