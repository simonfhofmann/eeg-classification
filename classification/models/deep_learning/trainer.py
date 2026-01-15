# ----------------------------------------------------------------------
# trainer.py
#
# Training utilities for deep learning EEG classification.
# Provides a clean training interface using the refactored pipeline.
# ----------------------------------------------------------------------

import numpy as np
import torch
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, Union
from datetime import datetime

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from config import DL_PARAMS, RANDOM_STATE
from data.containers import EEGDataContainer, SplitDataContainer
from data.dataset import BraindecodeDataset, reshape_for_braindecode
from data.preprocessor import compute_class_weights
from models.deep_learning.eegnet import (
    create_eegnet_classifier,
    create_eegnet_for_overfitting_test,
    compute_class_weights_tensor,
    save_model
)
from evaluation.evaluator import ModelEvaluator

# Braindecode imports
try:
    from braindecode.util import set_random_seeds
    BRAINDECODE_AVAILABLE = True
except ImportError:
    BRAINDECODE_AVAILABLE = False


class EEGNetTrainer:
    """
    Trainer for EEGNet models.

    Provides a high-level interface for training EEGNet on EEG data.
    Handles data preparation, model creation, training, and evaluation.

    Args:
        n_channels: Number of EEG channels
        n_classes: Number of output classes
        n_times: Number of time points per trial
        learning_rate: Learning rate
        batch_size: Batch size
        max_epochs: Maximum training epochs
        patience: Early stopping patience
        drop_prob: Dropout probability
        weight_decay: L2 regularization
        device: Device to train on
        random_seed: Random seed for reproducibility

    Example:
        >>> trainer = EEGNetTrainer(n_channels=30, n_classes=2, n_times=15950)
        >>> trainer.fit(train_data, val_data)
        >>> results = trainer.evaluate({'test': test_data})
    """

    def __init__(
        self,
        n_channels: int,
        n_classes: int = 2,
        n_times: int = 15950,
        learning_rate: float = 5e-4,
        batch_size: int = 16,
        max_epochs: int = 50,
        patience: int = 15,
        drop_prob: float = 0.5,
        weight_decay: float = 0.01,
        device: Optional[str] = None,
        random_seed: int = RANDOM_STATE,
        class_names: Optional[list] = None
    ):
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.n_times = n_times
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.drop_prob = drop_prob
        self.weight_decay = weight_decay
        self.random_seed = random_seed
        self.class_names = class_names or ['class_0', 'class_1']

        # Set device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        # Set random seeds
        if BRAINDECODE_AVAILABLE:
            set_random_seeds(seed=random_seed, cuda=(self.device == 'cuda'))

        self.model = None
        self.classifier = None
        self._is_fitted = False

    def fit(
        self,
        train_data: Union[EEGDataContainer, BraindecodeDataset, Tuple[np.ndarray, np.ndarray]],
        val_data: Optional[Union[EEGDataContainer, BraindecodeDataset, Tuple[np.ndarray, np.ndarray]]] = None,
        use_class_weights: bool = True,
        verbose: bool = True
    ) -> 'EEGNetTrainer':
        """
        Train the EEGNet model.

        Args:
            train_data: Training data (EEGDataContainer, BraindecodeDataset, or (X, y) tuple)
            val_data: Validation data (optional)
            use_class_weights: Whether to use class weights for imbalanced data
            verbose: Whether to print training progress

        Returns:
            self (for method chaining)
        """
        # Prepare training data
        X_train, y_train = self._prepare_data(train_data)

        # Prepare validation data if provided
        X_val, y_val = None, None
        if val_data is not None:
            X_val, y_val = self._prepare_data(val_data)

        # Compute class weights if requested
        class_weights = None
        if use_class_weights:
            class_weights = compute_class_weights(y_train)
            if verbose:
                print(f"\nClass weights: {class_weights}")

        # Update n_times based on actual data
        self.n_times = X_train.shape[2]

        if verbose:
            print(f"\n--- Training Configuration ---")
            print(f"Device: {self.device}")
            print(f"Input shape: ({self.n_channels}, {self.n_times})")
            print(f"Training samples: {len(y_train)}")
            if y_val is not None:
                print(f"Validation samples: {len(y_val)}")
            print(f"Batch size: {self.batch_size}")
            print(f"Learning rate: {self.learning_rate}")
            print(f"Max epochs: {self.max_epochs}")

        # Create classifier
        self.classifier = create_eegnet_classifier(
            n_channels=self.n_channels,
            n_classes=self.n_classes,
            n_times=self.n_times,
            drop_prob=self.drop_prob,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            batch_size=self.batch_size,
            max_epochs=self.max_epochs,
            patience=self.patience,
            class_weights=class_weights,
            device=self.device,
            verbose=1 if verbose else 0
        )

        # Reshape for Braindecode (4D input)
        X_train_4d = reshape_for_braindecode(X_train)

        if verbose:
            print(f"\n--- Starting Training ---")

        # Train
        self.classifier.fit(X_train_4d, y_train)

        self._is_fitted = True

        if verbose:
            print(f"\n--- Training Complete ---")

        return self

    def fit_overfitting_test(
        self,
        train_data: Union[EEGDataContainer, BraindecodeDataset, Tuple[np.ndarray, np.ndarray]],
        use_class_weights: bool = True,
        verbose: bool = True
    ) -> 'EEGNetTrainer':
        """
        Train with all regularization disabled (for testing model capacity).

        Args:
            train_data: Training data
            use_class_weights: Whether to use class weights
            verbose: Whether to print progress

        Returns:
            self
        """
        X_train, y_train = self._prepare_data(train_data)

        class_weights = None
        if use_class_weights:
            class_weights = compute_class_weights(y_train)

        self.n_times = X_train.shape[2]

        if verbose:
            print(f"\n--- OVERFITTING TEST MODE ---")
            print(f"All regularization disabled")
            print(f"Training samples: {len(y_train)}")

        self.classifier = create_eegnet_for_overfitting_test(
            n_channels=self.n_channels,
            n_classes=self.n_classes,
            n_times=self.n_times,
            learning_rate=self.learning_rate,
            batch_size=self.batch_size,
            max_epochs=self.max_epochs,
            class_weights=class_weights,
            device=self.device
        )

        X_train_4d = reshape_for_braindecode(X_train)
        self.classifier.fit(X_train_4d, y_train)

        self._is_fitted = True
        return self

    def predict(self, data: Union[EEGDataContainer, BraindecodeDataset, np.ndarray]) -> np.ndarray:
        """
        Make predictions on new data.

        Args:
            data: Data to predict on

        Returns:
            Predicted class labels
        """
        if not self._is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        X = self._prepare_X(data)
        X_4d = reshape_for_braindecode(X)
        return self.classifier.predict(X_4d)

    def predict_proba(self, data: Union[EEGDataContainer, BraindecodeDataset, np.ndarray]) -> np.ndarray:
        """
        Get prediction probabilities.

        Args:
            data: Data to predict on

        Returns:
            Class probabilities
        """
        if not self._is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        X = self._prepare_X(data)
        X_4d = reshape_for_braindecode(X)

        log_proba = self.classifier.predict_proba(X_4d)
        return np.exp(log_proba)  # Convert from log probabilities

    def evaluate(
        self,
        datasets: Dict[str, Union[EEGDataContainer, BraindecodeDataset, Tuple[np.ndarray, np.ndarray]]],
        verbose: bool = True
    ) -> Dict:
        """
        Evaluate model on multiple datasets.

        Args:
            datasets: Dictionary mapping split names to data
            verbose: Whether to print results

        Returns:
            Evaluation results dictionary
        """
        if not self._is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        # Convert all datasets to (X, y) format for evaluator
        prepared_datasets = {}
        for name, data in datasets.items():
            X, y = self._prepare_data(data)
            X_4d = reshape_for_braindecode(X)
            prepared_datasets[name] = (X_4d, y)

        evaluator = ModelEvaluator(
            model=self.classifier,
            class_names=self.class_names
        )

        return evaluator.evaluate(prepared_datasets, verbose=verbose)

    def save(self, filepath: Union[str, Path], include_metadata: bool = True):
        """
        Save the trained model.

        Args:
            filepath: Path to save to
            include_metadata: Whether to include training metadata
        """
        if not self._is_fitted:
            raise ValueError("Model not fitted. Nothing to save.")

        filepath = Path(filepath)

        metadata = None
        if include_metadata:
            metadata = {
                'n_channels': self.n_channels,
                'n_classes': self.n_classes,
                'n_times': self.n_times,
                'learning_rate': self.learning_rate,
                'batch_size': self.batch_size,
                'drop_prob': self.drop_prob,
                'saved_at': datetime.now().isoformat(),
            }

        save_model(self.classifier.module_, filepath, metadata)

    def _prepare_data(
        self,
        data: Union[EEGDataContainer, BraindecodeDataset, Tuple[np.ndarray, np.ndarray]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Convert various data formats to (X, y) tuple."""
        if isinstance(data, tuple):
            return data[0].astype(np.float32), data[1].astype(np.int64)
        elif hasattr(data, 'X') and hasattr(data, 'y'):
            return data.X.astype(np.float32), data.y.astype(np.int64)
        else:
            raise TypeError(f"Unsupported data type: {type(data)}")

    def _prepare_X(
        self,
        data: Union[EEGDataContainer, BraindecodeDataset, np.ndarray]
    ) -> np.ndarray:
        """Convert various data formats to X array."""
        if isinstance(data, np.ndarray):
            return data.astype(np.float32)
        elif hasattr(data, 'X'):
            return data.X.astype(np.float32)
        else:
            raise TypeError(f"Unsupported data type: {type(data)}")


def train_eegnet_pipeline(
    train_data: EEGDataContainer,
    val_data: Optional[EEGDataContainer] = None,
    test_data: Optional[EEGDataContainer] = None,
    learning_rate: float = 5e-4,
    batch_size: int = 16,
    max_epochs: int = 50,
    patience: int = 15,
    drop_prob: float = 0.5,
    class_names: Optional[list] = None,
    output_dir: Optional[Path] = None,
    verbose: bool = True
) -> Tuple['EEGNetTrainer', Dict]:
    """
    Complete EEGNet training pipeline.

    Convenience function that handles the full workflow:
    1. Create trainer
    2. Train model
    3. Evaluate on all provided datasets
    4. Save results

    Args:
        train_data: Training data container
        val_data: Validation data container (optional)
        test_data: Test data container (optional)
        learning_rate: Learning rate
        batch_size: Batch size
        max_epochs: Maximum epochs
        patience: Early stopping patience
        drop_prob: Dropout probability
        class_names: Class names for reporting
        output_dir: Directory to save results
        verbose: Whether to print progress

    Returns:
        Tuple of (trainer, results_dict)

    Example:
        >>> trainer, results = train_eegnet_pipeline(
        ...     train_data=splits.train,
        ...     val_data=splits.val,
        ...     test_data=splits.test,
        ...     output_dir=Path('results')
        ... )
    """
    # Create trainer
    trainer = EEGNetTrainer(
        n_channels=train_data.n_channels,
        n_classes=len(np.unique(train_data.y)),
        n_times=train_data.n_timepoints,
        learning_rate=learning_rate,
        batch_size=batch_size,
        max_epochs=max_epochs,
        patience=patience,
        drop_prob=drop_prob,
        class_names=class_names
    )

    # Train
    trainer.fit(train_data, val_data, verbose=verbose)

    # Prepare evaluation datasets
    eval_datasets = {'train': train_data}
    if val_data is not None:
        eval_datasets['val'] = val_data
    if test_data is not None:
        eval_datasets['test'] = test_data

    # Evaluate
    results = trainer.evaluate(eval_datasets, verbose=verbose)

    # Save results if output directory provided
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save model
        trainer.save(output_dir / 'model.pt')

        # Save evaluation results
        evaluator = ModelEvaluator(class_names=class_names, output_dir=output_dir)
        evaluator.save_results(results, 'evaluation_results.csv')

    return trainer, results
