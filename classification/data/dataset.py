# ----------------------------------------------------------------------
# dataset.py
#
# Dataset classes for EEG classification (sklearn and PyTorch compatible).
# ----------------------------------------------------------------------

import numpy as np
from typing import Dict, Tuple, Optional, List, Union, Callable
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import DEFAULT_TARGET


class EEGDataset:
    """
    EEG Dataset class compatible with sklearn-style APIs.

    Stores epoched EEG data with labels and provides easy access
    for training machine learning models.
    """

    def __init__(
        self,
        epochs: np.ndarray,
        labels: np.ndarray,
        participant_ids: Optional[np.ndarray] = None,
        trial_info: Optional[Dict] = None,
        channel_names: Optional[List[str]] = None,
        sampling_rate: int = 500
    ):
        """
        Initialize EEG dataset.

        Args:
            epochs: EEG data, shape (n_trials, n_channels, n_samples)
            labels: Classification labels, shape (n_trials,)
            participant_ids: Subject ID for each trial (for leave-one-subject-out CV)
            trial_info: Additional trial metadata (stimulus names, ratings, etc.)
            channel_names: List of channel names
            sampling_rate: Sampling rate in Hz
        """
        self.epochs = epochs
        self.labels = labels
        self.participant_ids = participant_ids
        self.trial_info = trial_info or {}
        self.channel_names = channel_names
        self.sampling_rate = sampling_rate

        # Validate shapes
        if len(epochs) != len(labels):
            raise ValueError(
                f"Number of epochs ({len(epochs)}) != number of labels ({len(labels)})"
            )

        if participant_ids is not None and len(participant_ids) != len(epochs):
            raise ValueError("participant_ids length must match epochs")

    @property
    def n_trials(self) -> int:
        return len(self.epochs)

    @property
    def n_channels(self) -> int:
        return self.epochs.shape[1]

    @property
    def n_samples(self) -> int:
        return self.epochs.shape[2]

    @property
    def shape(self) -> Tuple[int, int, int]:
        return self.epochs.shape

    @property
    def X(self) -> np.ndarray:
        """Return flattened feature matrix for sklearn."""
        return self.epochs.reshape(self.n_trials, -1)

    @property
    def y(self) -> np.ndarray:
        """Return labels."""
        return self.labels

    @property
    def classes(self) -> np.ndarray:
        """Return unique class labels."""
        return np.unique(self.labels)

    @property
    def n_classes(self) -> int:
        return len(self.classes)

    def get_class_distribution(self) -> Dict[int, int]:
        """Return count of samples per class."""
        unique, counts = np.unique(self.labels, return_counts=True)
        return dict(zip(unique, counts))

    def get_subject_ids(self) -> np.ndarray:
        """Return unique subject IDs."""
        if self.participant_ids is None:
            return np.array([])
        return np.unique(self.participant_ids)

    def subset(self, indices: np.ndarray) -> 'EEGDataset':
        """
        Create a subset of the dataset.

        Args:
            indices: Array of indices to include

        Returns:
            New EEGDataset with selected trials
        """
        subset_info = {k: v[indices] if isinstance(v, np.ndarray) else v
                       for k, v in self.trial_info.items()}

        return EEGDataset(
            epochs=self.epochs[indices],
            labels=self.labels[indices],
            participant_ids=self.participant_ids[indices] if self.participant_ids is not None else None,
            trial_info=subset_info,
            channel_names=self.channel_names,
            sampling_rate=self.sampling_rate
        )

    def filter_by_subject(self, subject_id: str) -> 'EEGDataset':
        """Return subset for a single subject."""
        if self.participant_ids is None:
            raise ValueError("No participant IDs available")

        mask = self.participant_ids == subject_id
        return self.subset(np.where(mask)[0])

    def exclude_subject(self, subject_id: str) -> 'EEGDataset':
        """Return subset excluding a single subject (for LOSO CV)."""
        if self.participant_ids is None:
            raise ValueError("No participant IDs available")

        mask = self.participant_ids != subject_id
        return self.subset(np.where(mask)[0])

    def __len__(self) -> int:
        return self.n_trials

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, int]:
        return self.epochs[idx], self.labels[idx]

    def __repr__(self) -> str:
        return (
            f"EEGDataset(n_trials={self.n_trials}, n_channels={self.n_channels}, "
            f"n_samples={self.n_samples}, n_classes={self.n_classes})"
        )


class TorchEEGDataset:
    """
    PyTorch-compatible Dataset for EEG data.

    Wraps EEGDataset for use with PyTorch DataLoader.
    Requires PyTorch to be installed.
    """

    def __init__(
        self,
        eeg_dataset: EEGDataset,
        transform: Optional[Callable] = None
    ):
        """
        Initialize PyTorch dataset.

        Args:
            eeg_dataset: EEGDataset instance
            transform: Optional transform to apply to each sample
        """
        self.dataset = eeg_dataset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple:
        """
        Get a sample.

        Returns:
            (data, label) tuple with data as torch tensor
        """
        try:
            import torch
        except ImportError:
            raise ImportError("PyTorch required for TorchEEGDataset")

        data, label = self.dataset[idx]

        if self.transform:
            data = self.transform(data)

        # Convert to torch tensors
        data = torch.FloatTensor(data)
        label = torch.LongTensor([label])[0]

        return data, label

    def get_dataloader(
        self,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 0
    ):
        """
        Create a PyTorch DataLoader.

        Args:
            batch_size: Batch size
            shuffle: Whether to shuffle data
            num_workers: Number of worker processes

        Returns:
            torch.utils.data.DataLoader
        """
        try:
            from torch.utils.data import DataLoader
        except ImportError:
            raise ImportError("PyTorch required for DataLoader")

        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers
        )


def combine_datasets(datasets: List[EEGDataset]) -> EEGDataset:
    """
    Combine multiple EEGDatasets into one.

    Useful for combining data from multiple subjects.

    Args:
        datasets: List of EEGDataset instances

    Returns:
        Combined EEGDataset
    """
    epochs = np.concatenate([d.epochs for d in datasets], axis=0)
    labels = np.concatenate([d.labels for d in datasets], axis=0)

    # Combine participant IDs
    participant_ids = None
    if all(d.participant_ids is not None for d in datasets):
        participant_ids = np.concatenate([d.participant_ids for d in datasets])

    # Use channel info from first dataset
    channel_names = datasets[0].channel_names
    sampling_rate = datasets[0].sampling_rate

    return EEGDataset(
        epochs=epochs,
        labels=labels,
        participant_ids=participant_ids,
        channel_names=channel_names,
        sampling_rate=sampling_rate
    )
