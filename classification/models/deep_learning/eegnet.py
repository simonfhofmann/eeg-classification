# ----------------------------------------------------------------------
# eegnet.py
#
# EEGNet model factory and utilities.
# Provides convenient functions for creating and configuring EEGNet models.
# ----------------------------------------------------------------------

import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Tuple
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from config import DL_PARAMS

# Braindecode imports
try:
    from braindecode.models import EEGNet
    from braindecode.classifier import EEGClassifier
    BRAINDECODE_AVAILABLE = True
except ImportError:
    BRAINDECODE_AVAILABLE = False
    print("Warning: Braindecode not installed. Some features will be unavailable.")


def create_eegnet(
    n_channels: int,
    n_classes: int,
    n_times: int,
    drop_prob: float = 0.5,
    device: Optional[str] = None
) -> nn.Module:
    """
    Create an EEGNet model.

    Args:
        n_channels: Number of EEG channels
        n_classes: Number of output classes
        n_times: Number of time points per trial
        drop_prob: Dropout probability (0.0 for no dropout)
        device: Device to place model on ('cuda', 'cpu', or None for auto)

    Returns:
        EEGNet model on specified device

    Example:
        >>> model = create_eegnet(30, 2, 15950, drop_prob=0.5)
        >>> print(model)
    """
    if not BRAINDECODE_AVAILABLE:
        raise ImportError("Braindecode is required for EEGNet. Install with: pip install braindecode")

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = EEGNet(
        n_chans=n_channels,
        n_outputs=n_classes,
        n_times=n_times,
        drop_prob=drop_prob
    )

    return model.to(device)


def create_eegnet_classifier(
    n_channels: int,
    n_classes: int,
    n_times: int,
    drop_prob: float = 0.5,
    learning_rate: float = 1e-3,
    weight_decay: float = 0.01,
    batch_size: int = 32,
    max_epochs: int = 100,
    patience: int = 15,
    class_weights: Optional[np.ndarray] = None,
    device: Optional[str] = None,
    verbose: int = 1
) -> 'EEGClassifier':
    """
    Create a fully configured EEGClassifier with EEGNet.

    This creates a Braindecode EEGClassifier which wraps EEGNet with
    training logic via Skorch. Includes:
    - AdamW optimizer with weight decay
    - Cosine annealing learning rate scheduler
    - Early stopping
    - Class weighting for imbalanced data

    Args:
        n_channels: Number of EEG channels
        n_classes: Number of output classes
        n_times: Number of time points per trial
        drop_prob: Dropout probability
        learning_rate: Initial learning rate
        weight_decay: L2 regularization strength
        batch_size: Training batch size
        max_epochs: Maximum number of training epochs
        patience: Early stopping patience
        class_weights: Optional class weights for imbalanced data
        device: Device ('cuda' or 'cpu')
        verbose: Verbosity level

    Returns:
        Configured EEGClassifier

    Example:
        >>> clf = create_eegnet_classifier(30, 2, 15950)
        >>> clf.fit(X_train, y_train)
        >>> y_pred = clf.predict(X_test)
    """
    if not BRAINDECODE_AVAILABLE:
        raise ImportError("Braindecode is required. Install with: pip install braindecode")

    from torch.optim import AdamW
    from skorch.callbacks import LRScheduler, EarlyStopping

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Create the model
    model = EEGNet(
        n_chans=n_channels,
        n_outputs=n_classes,
        n_times=n_times,
        drop_prob=drop_prob
    )

    # Set up criterion with class weights
    criterion_kwargs = {}
    if class_weights is not None:
        class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
        criterion_kwargs['weight'] = class_weights_tensor

    # Set up callbacks
    callbacks = [
        ('lr_scheduler', LRScheduler('CosineAnnealingLR', T_max=max_epochs - 1)),
        ('early_stopping', EarlyStopping(
            monitor='valid_loss',
            patience=patience,
            threshold=0.001,
            lower_is_better=True
        )),
    ]

    # Create classifier
    clf = EEGClassifier(
        model,
        criterion=nn.NLLLoss,
        optimizer=AdamW,
        optimizer__weight_decay=weight_decay,
        max_epochs=max_epochs,
        batch_size=batch_size,
        lr=learning_rate,
        device=device,
        callbacks=callbacks,
        verbose=verbose,
        **{f'criterion__{k}': v for k, v in criterion_kwargs.items()}
    )

    return clf


def create_eegnet_for_overfitting_test(
    n_channels: int,
    n_classes: int,
    n_times: int,
    learning_rate: float = 1e-3,
    batch_size: int = 16,
    max_epochs: int = 100,
    class_weights: Optional[np.ndarray] = None,
    device: Optional[str] = None
) -> 'EEGClassifier':
    """
    Create EEGNet classifier configured for overfitting test.

    This creates a model with all regularization disabled to test
    if the model has sufficient capacity to memorize the training data.
    Useful for debugging.

    Args:
        n_channels: Number of EEG channels
        n_classes: Number of output classes
        n_times: Number of time points per trial
        learning_rate: Learning rate
        batch_size: Batch size
        max_epochs: Maximum epochs
        class_weights: Optional class weights
        device: Device

    Returns:
        EEGClassifier with no regularization
    """
    if not BRAINDECODE_AVAILABLE:
        raise ImportError("Braindecode is required.")

    from torch.optim import AdamW
    from skorch.callbacks import LRScheduler

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Create model with NO dropout
    model = EEGNet(
        n_chans=n_channels,
        n_outputs=n_classes,
        n_times=n_times,
        drop_prob=0.0  # No dropout
    )

    # Set up criterion with class weights
    criterion_kwargs = {}
    if class_weights is not None:
        class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
        criterion_kwargs['weight'] = class_weights_tensor

    # Minimal callbacks - no early stopping
    callbacks = [
        ('lr_scheduler', LRScheduler('CosineAnnealingLR', T_max=max_epochs - 1)),
    ]

    clf = EEGClassifier(
        model,
        criterion=nn.NLLLoss,
        optimizer=AdamW,
        optimizer__weight_decay=0.0,  # No weight decay
        train_split=None,  # Don't use validation
        max_epochs=max_epochs,
        batch_size=batch_size,
        lr=learning_rate,
        device=device,
        callbacks=callbacks,
        verbose=1,
        **{f'criterion__{k}': v for k, v in criterion_kwargs.items()}
    )

    return clf


def compute_class_weights_tensor(
    y: np.ndarray,
    device: Optional[str] = None
) -> torch.Tensor:
    """
    Compute balanced class weights and return as PyTorch tensor.

    Args:
        y: Labels array
        device: Device to place tensor on

    Returns:
        Class weights tensor
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    classes = np.unique(y)
    n_samples = len(y)
    n_classes = len(classes)

    weights = np.zeros(n_classes)
    for i, c in enumerate(classes):
        n_samples_c = np.sum(y == c)
        weights[i] = n_samples / (n_classes * n_samples_c)

    return torch.tensor(weights, dtype=torch.float32).to(device)


def get_model_summary(model: nn.Module, input_shape: Tuple[int, ...]) -> str:
    """
    Get a summary of the model architecture.

    Args:
        model: PyTorch model
        input_shape: Expected input shape (n_channels, n_times, 1) for EEGNet

    Returns:
        Model summary string
    """
    try:
        from torchinfo import summary
        return str(summary(model, input_size=(1,) + input_shape, verbose=0))
    except ImportError:
        # Fallback to simple summary
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return (
            f"Model: {model.__class__.__name__}\n"
            f"Total parameters: {total_params:,}\n"
            f"Trainable parameters: {trainable_params:,}"
        )


def save_model(model: nn.Module, filepath: Path, metadata: Optional[Dict] = None):
    """
    Save model weights and optional metadata.

    Args:
        model: Model to save
        filepath: Path to save to
        metadata: Optional metadata to save alongside weights
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    save_dict = {
        'model_state_dict': model.state_dict(),
    }

    if metadata:
        save_dict['metadata'] = metadata

    torch.save(save_dict, filepath)
    print(f"Model saved to: {filepath}")


def load_model(
    filepath: Path,
    n_channels: int,
    n_classes: int,
    n_times: int,
    drop_prob: float = 0.5,
    device: Optional[str] = None
) -> Tuple[nn.Module, Optional[Dict]]:
    """
    Load model weights.

    Args:
        filepath: Path to saved model
        n_channels: Number of channels
        n_classes: Number of classes
        n_times: Number of time points
        drop_prob: Dropout probability
        device: Device to load to

    Returns:
        Tuple of (model, metadata)
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Create model architecture
    model = create_eegnet(n_channels, n_classes, n_times, drop_prob, device)

    # Load weights
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    metadata = checkpoint.get('metadata', None)

    print(f"Model loaded from: {filepath}")
    return model, metadata
