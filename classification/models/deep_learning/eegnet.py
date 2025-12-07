# ----------------------------------------------------------------------
# eegnet.py
#
# EEGNet implementation for EEG classification.
# Reference: Lawhern et al. 2018 - "EEGNet: A Compact Convolutional
#            Neural Network for EEG-based Brain-Computer Interfaces"
# ----------------------------------------------------------------------

import numpy as np
from typing import Dict, Any, Optional, Tuple
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from models.base import BaseClassifier
from config import DL_PARAMS, SAMPLING_RATE, N_CHANNELS


class EEGNetClassifier(BaseClassifier):
    """
    EEGNet-based classifier for EEG data.

    EEGNet is a compact CNN designed specifically for EEG classification.
    It uses depthwise and separable convolutions to reduce parameters
    while capturing both temporal and spatial features.

    Note:
        Requires PyTorch to be installed.
    """

    def __init__(
        self,
        n_channels: int = N_CHANNELS,
        n_samples: int = 500,
        n_classes: int = 2,
        dropout_rate: float = 0.5,
        kernel_length: int = 64,
        F1: int = 8,
        D: int = 2,
        F2: int = 16,
        learning_rate: float = DL_PARAMS['learning_rate'],
        batch_size: int = DL_PARAMS['batch_size'],
        epochs: int = DL_PARAMS['epochs'],
        early_stopping_patience: int = DL_PARAMS['early_stopping_patience'],
        device: str = "auto",
        name: str = "EEGNet",
        **kwargs
    ):
        """
        Initialize EEGNet classifier.

        Args:
            n_channels: Number of EEG channels
            n_samples: Number of time samples per trial
            n_classes: Number of output classes
            dropout_rate: Dropout probability
            kernel_length: Length of temporal convolution kernel
            F1: Number of temporal filters
            D: Depth multiplier for depthwise convolution
            F2: Number of pointwise filters
            learning_rate: Learning rate for optimizer
            batch_size: Training batch size
            epochs: Maximum number of training epochs
            early_stopping_patience: Epochs to wait before early stopping
            device: Device to use ('cpu', 'cuda', or 'auto')
            name: Model name
        """
        super().__init__(name=name, **kwargs)

        self.n_channels = n_channels
        self.n_samples = n_samples
        self.n_classes = n_classes
        self.dropout_rate = dropout_rate
        self.kernel_length = kernel_length
        self.F1 = F1
        self.D = D
        self.F2 = F2
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.early_stopping_patience = early_stopping_patience

        # Set device
        if device == "auto":
            try:
                import torch
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                self.device = "cpu"
        else:
            self.device = device

        self._model = None
        self._optimizer = None
        self._criterion = None

    def _build_model(self):
        """Build the EEGNet architecture."""
        try:
            import torch
            import torch.nn as nn
        except ImportError:
            raise ImportError(
                "PyTorch is required for EEGNet. "
                "Install with: pip install torch"
            )

        class EEGNet(nn.Module):
            def __init__(self, n_channels, n_samples, n_classes,
                         dropout_rate, kernel_length, F1, D, F2):
                super().__init__()

                # Layer 1: Temporal convolution
                self.conv1 = nn.Conv2d(1, F1, (1, kernel_length),
                                       padding=(0, kernel_length // 2), bias=False)
                self.bn1 = nn.BatchNorm2d(F1)

                # Layer 2: Depthwise convolution (spatial filtering)
                self.conv2 = nn.Conv2d(F1, F1 * D, (n_channels, 1),
                                       groups=F1, bias=False)
                self.bn2 = nn.BatchNorm2d(F1 * D)
                self.pool1 = nn.AvgPool2d((1, 4))
                self.dropout1 = nn.Dropout(dropout_rate)

                # Layer 3: Separable convolution
                self.conv3 = nn.Conv2d(F1 * D, F2, (1, 16),
                                       padding=(0, 8), bias=False)
                self.bn3 = nn.BatchNorm2d(F2)
                self.pool2 = nn.AvgPool2d((1, 8))
                self.dropout2 = nn.Dropout(dropout_rate)

                # Calculate flattened size
                self._to_linear = None
                self._get_conv_output((1, 1, n_channels, n_samples))

                # Classifier
                self.fc = nn.Linear(self._to_linear, n_classes)

            def _get_conv_output(self, shape):
                import torch
                with torch.no_grad():
                    x = torch.zeros(shape)
                    x = self._forward_features(x)
                    self._to_linear = x.numel()

            def _forward_features(self, x):
                import torch.nn.functional as F

                # Block 1
                x = self.conv1(x)
                x = self.bn1(x)

                # Block 2
                x = self.conv2(x)
                x = self.bn2(x)
                x = F.elu(x)
                x = self.pool1(x)
                x = self.dropout1(x)

                # Block 3
                x = self.conv3(x)
                x = self.bn3(x)
                x = F.elu(x)
                x = self.pool2(x)
                x = self.dropout2(x)

                return x

            def forward(self, x):
                x = self._forward_features(x)
                x = x.view(x.size(0), -1)
                x = self.fc(x)
                return x

        model = EEGNet(
            self.n_channels, self.n_samples, self.n_classes,
            self.dropout_rate, self.kernel_length,
            self.F1, self.D, self.F2
        )

        return model.to(self.device)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        verbose: bool = True,
        **kwargs
    ) -> 'EEGNetClassifier':
        """
        Train the EEGNet model.

        Args:
            X: Training data, shape (n_samples, n_channels, n_timepoints)
            y: Training labels, shape (n_samples,)
            X_val: Optional validation features
            y_val: Optional validation labels
            verbose: Whether to print training progress

        Returns:
            self
        """
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        # Update dimensions from data
        self.n_channels = X.shape[1]
        self.n_samples = X.shape[2]
        self.n_classes = len(np.unique(y))

        # Build model
        self._model = self._build_model()
        self._criterion = nn.CrossEntropyLoss()
        self._optimizer = torch.optim.Adam(
            self._model.parameters(),
            lr=self.learning_rate
        )

        # Prepare data
        X_tensor = torch.FloatTensor(X).unsqueeze(1)  # Add channel dim
        y_tensor = torch.LongTensor(y)

        train_dataset = TensorDataset(X_tensor, y_tensor)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True
        )

        # Validation data
        if X_val is not None and y_val is not None:
            X_val_tensor = torch.FloatTensor(X_val).unsqueeze(1)
            y_val_tensor = torch.LongTensor(y_val)
            val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size)
        else:
            val_loader = None

        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        self.history['train_loss'] = []
        self.history['val_loss'] = []
        self.history['train_acc'] = []
        self.history['val_acc'] = []

        for epoch in range(self.epochs):
            # Training
            self._model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0

            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

                self._optimizer.zero_grad()
                outputs = self._model(batch_X)
                loss = self._criterion(outputs, batch_y)
                loss.backward()
                self._optimizer.step()

                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += batch_y.size(0)
                train_correct += predicted.eq(batch_y).sum().item()

            train_loss /= len(train_loader)
            train_acc = train_correct / train_total

            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)

            # Validation
            if val_loader is not None:
                val_loss, val_acc = self._evaluate_loader(val_loader)
                self.history['val_loss'].append(val_loss)
                self.history['val_acc'].append(val_acc)

                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # Save best model state
                    self._best_state = self._model.state_dict().copy()
                else:
                    patience_counter += 1

                if patience_counter >= self.early_stopping_patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch + 1}")
                    self._model.load_state_dict(self._best_state)
                    break

                if verbose and (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch + 1}/{self.epochs} - "
                          f"Loss: {train_loss:.4f} - Acc: {train_acc:.4f} - "
                          f"Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}")
            else:
                if verbose and (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch + 1}/{self.epochs} - "
                          f"Loss: {train_loss:.4f} - Acc: {train_acc:.4f}")

        self.is_fitted = True
        return self

    def _evaluate_loader(self, loader) -> Tuple[float, float]:
        """Evaluate model on a data loader."""
        import torch

        self._model.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_X, batch_y in loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

                outputs = self._model(batch_X)
                loss = self._criterion(outputs, batch_y)

                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += batch_y.size(0)
                correct += predicted.eq(batch_y).sum().item()

        return total_loss / len(loader), correct / total

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        import torch

        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        self._model.eval()
        X_tensor = torch.FloatTensor(X).unsqueeze(1).to(self.device)

        with torch.no_grad():
            outputs = self._model(X_tensor)
            _, predicted = outputs.max(1)

        return predicted.cpu().numpy()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        import torch
        import torch.nn.functional as F

        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        self._model.eval()
        X_tensor = torch.FloatTensor(X).unsqueeze(1).to(self.device)

        with torch.no_grad():
            outputs = self._model(X_tensor)
            proba = F.softmax(outputs, dim=1)

        return proba.cpu().numpy()
