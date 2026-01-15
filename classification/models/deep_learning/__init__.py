# ----------------------------------------------------------------------
# Deep Learning classifiers
# ----------------------------------------------------------------------

from .eegnet import (
    create_eegnet,
    create_eegnet_classifier,
    create_eegnet_for_overfitting_test,
    compute_class_weights_tensor,
    save_model,
    load_model,
    get_model_summary,
)
from .trainer import EEGNetTrainer, train_eegnet_pipeline

__all__ = [
    # EEGNet factory functions
    "create_eegnet",
    "create_eegnet_classifier",
    "create_eegnet_for_overfitting_test",
    "compute_class_weights_tensor",
    "save_model",
    "load_model",
    "get_model_summary",
    # Trainer
    "EEGNetTrainer",
    "train_eegnet_pipeline",
]
