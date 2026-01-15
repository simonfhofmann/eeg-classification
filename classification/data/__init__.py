# ----------------------------------------------------------------------
# Data loading and preprocessing module
# ----------------------------------------------------------------------

from .loader import load_mat_file, load_behavioral_data, load_subject_data
from .preprocessor import (
    Preprocessor,
    filter_trials_by_rating,
    create_binary_labels,
    create_binary_labels_exclude_neutral,
    stratified_train_val_test_split,
    stratified_train_test_split,
    scale_to_microvolts,
    scale_to_volts,
    compute_class_weights,
)
from .dataset import (
    EEGDataset,
    TorchEEGDataset,
    BraindecodeDataset,
    combine_datasets,
    create_braindecode_datasets,
    reshape_for_braindecode,
)
from .containers import EEGDataContainer, SplitDataContainer
from .loaders import MatlabPreprocessedLoader, RawEEGLoader, BaseLoader

__all__ = [
    # Loader functions (legacy)
    "load_mat_file",
    "load_behavioral_data",
    "load_subject_data",
    # Loader classes (new)
    "BaseLoader",
    "MatlabPreprocessedLoader",
    "RawEEGLoader",
    # Containers
    "EEGDataContainer",
    "SplitDataContainer",
    # Preprocessor
    "Preprocessor",
    "filter_trials_by_rating",
    "create_binary_labels",
    "create_binary_labels_exclude_neutral",
    "stratified_train_val_test_split",
    "stratified_train_test_split",
    "scale_to_microvolts",
    "scale_to_volts",
    "compute_class_weights",
    # Datasets
    "EEGDataset",
    "TorchEEGDataset",
    "BraindecodeDataset",
    "combine_datasets",
    "create_braindecode_datasets",
    "reshape_for_braindecode",
]
