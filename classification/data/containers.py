# ----------------------------------------------------------------------
# containers.py
#
# Common data containers for EEG classification pipeline.
# Provides a unified interface regardless of preprocessing source.
# ----------------------------------------------------------------------

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from pathlib import Path


@dataclass
class EEGDataContainer:
    """
    Common container for preprocessed EEG data regardless of source.

    This container provides a unified interface for data that has been
    preprocessed either via MATLAB/EEGLAB or via Braindecode/MNE pipelines.

    Attributes:
        X: EEG data array of shape (n_trials, n_channels, n_timepoints)
        y: Labels array of shape (n_trials,). Can be None if labels not yet assigned.
        sfreq: Sampling frequency in Hz
        ch_names: List of channel names
        participant_id: Identifier for the participant
        metadata: Additional metadata (flexible dict for extra info)

    Example:
        >>> container = EEGDataContainer(
        ...     X=eeg_array,
        ...     y=labels,
        ...     sfreq=500,
        ...     ch_names=['Fp1', 'Fp2', ...],
        ...     participant_id='sub-001'
        ... )
        >>> print(f"Data shape: {container.shape}")
        >>> print(f"Number of trials: {container.n_trials}")
    """
    X: np.ndarray
    sfreq: int
    ch_names: List[str]
    participant_id: str = ""
    y: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate data dimensions and types."""
        if self.X.ndim != 3:
            raise ValueError(
                f"X must be 3D (n_trials, n_channels, n_timepoints), "
                f"got shape {self.X.shape}"
            )

        if len(self.ch_names) != self.X.shape[1]:
            raise ValueError(
                f"Number of channel names ({len(self.ch_names)}) must match "
                f"number of channels in X ({self.X.shape[1]})"
            )

        if self.y is not None and len(self.y) != self.X.shape[0]:
            raise ValueError(
                f"Length of y ({len(self.y)}) must match "
                f"number of trials in X ({self.X.shape[0]})"
            )

    @property
    def shape(self) -> tuple:
        """Return shape of EEG data (n_trials, n_channels, n_timepoints)."""
        return self.X.shape

    @property
    def n_trials(self) -> int:
        """Number of trials/epochs."""
        return self.X.shape[0]

    @property
    def n_channels(self) -> int:
        """Number of EEG channels."""
        return self.X.shape[1]

    @property
    def n_timepoints(self) -> int:
        """Number of time points per trial."""
        return self.X.shape[2]

    @property
    def duration(self) -> float:
        """Duration of each trial in seconds."""
        return self.n_timepoints / self.sfreq

    def copy(self) -> 'EEGDataContainer':
        """Create a deep copy of the container."""
        return EEGDataContainer(
            X=self.X.copy(),
            y=self.y.copy() if self.y is not None else None,
            sfreq=self.sfreq,
            ch_names=self.ch_names.copy(),
            participant_id=self.participant_id,
            metadata=self.metadata.copy()
        )

    def select_trials(self, indices: np.ndarray) -> 'EEGDataContainer':
        """
        Select a subset of trials by indices.

        Args:
            indices: Array of trial indices to keep

        Returns:
            New EEGDataContainer with selected trials
        """
        new_y = self.y[indices] if self.y is not None else None
        return EEGDataContainer(
            X=self.X[indices],
            y=new_y,
            sfreq=self.sfreq,
            ch_names=self.ch_names.copy(),
            participant_id=self.participant_id,
            metadata={**self.metadata, 'selected_from_n_trials': self.n_trials}
        )

    def select_channels(self, channel_names: List[str]) -> 'EEGDataContainer':
        """
        Select a subset of channels by name.

        Args:
            channel_names: List of channel names to keep

        Returns:
            New EEGDataContainer with selected channels
        """
        indices = [self.ch_names.index(ch) for ch in channel_names]
        return EEGDataContainer(
            X=self.X[:, indices, :],
            y=self.y.copy() if self.y is not None else None,
            sfreq=self.sfreq,
            ch_names=channel_names.copy(),
            participant_id=self.participant_id,
            metadata={**self.metadata, 'selected_from_n_channels': self.n_channels}
        )

    def set_labels(self, y: np.ndarray) -> 'EEGDataContainer':
        """
        Set or update labels.

        Args:
            y: New labels array

        Returns:
            Self (for method chaining)
        """
        if len(y) != self.n_trials:
            raise ValueError(
                f"Length of y ({len(y)}) must match number of trials ({self.n_trials})"
            )
        self.y = y
        return self

    def to_microvolts(self) -> 'EEGDataContainer':
        """
        Convert data to microvolts (multiply by 1e6 if in volts).

        Returns:
            New container with data in microvolts
        """
        # Check if data is likely in volts (values very small)
        if np.abs(self.X).max() < 1e-3:
            new_X = self.X * 1e6
            new_metadata = {**self.metadata, 'unit': 'microvolts'}
        else:
            new_X = self.X.copy()
            new_metadata = self.metadata.copy()

        return EEGDataContainer(
            X=new_X,
            y=self.y.copy() if self.y is not None else None,
            sfreq=self.sfreq,
            ch_names=self.ch_names.copy(),
            participant_id=self.participant_id,
            metadata=new_metadata
        )

    def to_volts(self) -> 'EEGDataContainer':
        """
        Convert data to volts (divide by 1e6 if in microvolts).

        Returns:
            New container with data in volts
        """
        # Check if data is likely in microvolts (values large)
        if np.abs(self.X).max() > 1e-3:
            new_X = self.X * 1e-6
            new_metadata = {**self.metadata, 'unit': 'volts'}
        else:
            new_X = self.X.copy()
            new_metadata = self.metadata.copy()

        return EEGDataContainer(
            X=new_X,
            y=self.y.copy() if self.y is not None else None,
            sfreq=self.sfreq,
            ch_names=self.ch_names.copy(),
            participant_id=self.participant_id,
            metadata=new_metadata
        )

    def __repr__(self) -> str:
        label_info = f", y={self.y.shape}" if self.y is not None else ", y=None"
        return (
            f"EEGDataContainer("
            f"X={self.shape}, "
            f"sfreq={self.sfreq}Hz, "
            f"duration={self.duration:.2f}s"
            f"{label_info}, "
            f"participant='{self.participant_id}')"
        )


@dataclass
class SplitDataContainer:
    """
    Container for train/validation/test splits.

    Attributes:
        train: Training data container
        val: Validation data container (optional)
        test: Test data container
    """
    train: EEGDataContainer
    test: EEGDataContainer
    val: Optional[EEGDataContainer] = None

    def __repr__(self) -> str:
        val_info = f", val={self.val.n_trials}" if self.val else ""
        return (
            f"SplitDataContainer("
            f"train={self.train.n_trials}, "
            f"test={self.test.n_trials}"
            f"{val_info})"
        )
