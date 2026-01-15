# ----------------------------------------------------------------------
# audio_correlation.py
#
# Custom feature extractor for EEG-audio correlation features.
# This is a template for computing correlation between EEG signals
# and audio envelope/features.
# ----------------------------------------------------------------------

import numpy as np
from pathlib import Path
from typing import List, Optional, Dict, Union
import warnings

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from features.base import FeatureExtractor


class AudioEEGCorrelationExtractor(FeatureExtractor):
    """
    Extract correlation features between EEG and audio signals.

    This extractor computes the correlation between EEG signals and
    audio features (e.g., envelope, spectral features) for each trial.
    This can capture how much the brain activity tracks the audio stimulus.

    The audio features should be pre-computed and stored in a directory
    or provided directly. Each audio feature file should correspond to
    a stimulus and be resampled to match the EEG sampling rate.

    Args:
        audio_features_dir: Directory containing audio feature files
        audio_features_dict: Alternative: dict mapping stimulus_id to features
        feature_types: Which correlation features to compute
        ch_names: EEG channel names for feature naming
        sfreq: Sampling frequency (should match EEG)

    Example:
        >>> extractor = AudioEEGCorrelationExtractor(
        ...     audio_features_dir="path/to/audio_features",
        ...     feature_types=['envelope', 'onset']
        ... )
        >>> features = extractor.extract(eeg_data, stimulus_ids=stimulus_ids)

    Note:
        This is a template implementation. You'll need to adapt the
        audio feature loading and correlation computation to your
        specific use case.
    """

    SUPPORTED_FEATURES = ['envelope', 'onset', 'spectral_flux', 'rms']

    def __init__(
        self,
        audio_features_dir: Optional[Union[str, Path]] = None,
        audio_features_dict: Optional[Dict[str, np.ndarray]] = None,
        feature_types: Optional[List[str]] = None,
        ch_names: Optional[List[str]] = None,
        sfreq: int = 500,
        name: str = "AudioEEGCorrelation"
    ):
        super().__init__(name=name)

        self.audio_features_dir = Path(audio_features_dir) if audio_features_dir else None
        self.audio_features_dict = audio_features_dict or {}
        self.feature_types = feature_types or ['envelope']
        self.ch_names = ch_names
        self.sfreq = sfreq

        # Validate feature types
        for ft in self.feature_types:
            if ft not in self.SUPPORTED_FEATURES:
                warnings.warn(f"Feature type '{ft}' not in supported list. "
                              f"Make sure you implement loading for it.")

        # Cache for loaded audio features
        self._audio_cache: Dict[str, Dict[str, np.ndarray]] = {}

    def extract(
        self,
        eeg_data: np.ndarray,
        stimulus_ids: Optional[List[str]] = None,
        **kwargs
    ) -> np.ndarray:
        """
        Extract EEG-audio correlation features.

        Args:
            eeg_data: EEG data of shape (n_trials, n_channels, n_timepoints)
            stimulus_ids: List of stimulus identifiers for each trial
                         (needed to load corresponding audio features)
            **kwargs: Additional arguments

        Returns:
            Feature array of shape (n_trials, n_features)
            where n_features = n_channels * len(feature_types)
        """
        n_trials, n_channels, n_timepoints = eeg_data.shape

        if stimulus_ids is None:
            warnings.warn("No stimulus_ids provided. Using placeholder features.")
            # Return zeros if no stimulus information
            n_features = n_channels * len(self.feature_types)
            return np.zeros((n_trials, n_features))

        if len(stimulus_ids) != n_trials:
            raise ValueError(
                f"Number of stimulus_ids ({len(stimulus_ids)}) must match "
                f"number of trials ({n_trials})"
            )

        # Extract correlation features for each trial
        all_features = []

        for trial_idx in range(n_trials):
            stimulus_id = stimulus_ids[trial_idx]
            trial_eeg = eeg_data[trial_idx]  # (n_channels, n_timepoints)

            # Load audio features for this stimulus
            audio_feats = self._load_audio_features(stimulus_id, n_timepoints)

            # Compute correlations
            trial_features = self._compute_correlations(trial_eeg, audio_feats)
            all_features.append(trial_features)

        return np.array(all_features)

    def _load_audio_features(
        self,
        stimulus_id: str,
        n_timepoints: int
    ) -> Dict[str, np.ndarray]:
        """
        Load audio features for a stimulus.

        Args:
            stimulus_id: Identifier for the stimulus
            n_timepoints: Expected number of timepoints (for validation)

        Returns:
            Dictionary mapping feature_type to feature array
        """
        # Check cache first
        if stimulus_id in self._audio_cache:
            return self._audio_cache[stimulus_id]

        audio_feats = {}

        for feature_type in self.feature_types:
            # Try to load from dict first
            if stimulus_id in self.audio_features_dict:
                feat = self.audio_features_dict[stimulus_id]
                if isinstance(feat, dict):
                    audio_feats[feature_type] = feat.get(feature_type, np.zeros(n_timepoints))
                else:
                    audio_feats[feature_type] = feat

            # Try to load from directory
            elif self.audio_features_dir is not None:
                feat_path = self.audio_features_dir / f"{stimulus_id}_{feature_type}.npy"
                if feat_path.exists():
                    audio_feats[feature_type] = np.load(feat_path)
                else:
                    # Try generic file
                    generic_path = self.audio_features_dir / f"{stimulus_id}.npy"
                    if generic_path.exists():
                        data = np.load(generic_path, allow_pickle=True)
                        if isinstance(data, dict):
                            audio_feats[feature_type] = data.get(feature_type, np.zeros(n_timepoints))
                        else:
                            audio_feats[feature_type] = data
                    else:
                        warnings.warn(f"Audio features not found for {stimulus_id}")
                        audio_feats[feature_type] = np.zeros(n_timepoints)
            else:
                audio_feats[feature_type] = np.zeros(n_timepoints)

            # Ensure correct length
            if len(audio_feats[feature_type]) != n_timepoints:
                # Resample or truncate/pad
                audio_feats[feature_type] = self._adjust_length(
                    audio_feats[feature_type], n_timepoints
                )

        # Cache the result
        self._audio_cache[stimulus_id] = audio_feats
        return audio_feats

    def _adjust_length(self, signal: np.ndarray, target_length: int) -> np.ndarray:
        """Adjust signal length to match target."""
        current_length = len(signal)

        if current_length == target_length:
            return signal
        elif current_length > target_length:
            return signal[:target_length]
        else:
            # Pad with zeros
            padded = np.zeros(target_length)
            padded[:current_length] = signal
            return padded

    def _compute_correlations(
        self,
        eeg: np.ndarray,
        audio_feats: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """
        Compute correlation between EEG channels and audio features.

        Args:
            eeg: EEG data for one trial, shape (n_channels, n_timepoints)
            audio_feats: Dictionary of audio feature arrays

        Returns:
            Correlation features, shape (n_channels * n_feature_types,)
        """
        n_channels = eeg.shape[0]
        features = []

        for feature_type in self.feature_types:
            audio = audio_feats[feature_type]

            for ch_idx in range(n_channels):
                eeg_channel = eeg[ch_idx]

                # Compute Pearson correlation
                corr = self._pearson_correlation(eeg_channel, audio)
                features.append(corr)

        return np.array(features)

    def _pearson_correlation(self, x: np.ndarray, y: np.ndarray) -> float:
        """Compute Pearson correlation coefficient."""
        if len(x) != len(y):
            raise ValueError("Arrays must have same length")

        # Handle constant arrays
        if np.std(x) == 0 or np.std(y) == 0:
            return 0.0

        return np.corrcoef(x, y)[0, 1]

    def get_feature_names(self) -> List[str]:
        """Get feature names."""
        if self.ch_names is None:
            # Use placeholder channel names
            ch_names = [f"ch{i}" for i in range(30)]  # Assume 30 channels
        else:
            ch_names = self.ch_names

        names = []
        for feature_type in self.feature_types:
            for ch_name in ch_names:
                names.append(f"{ch_name}_{feature_type}_corr")

        return names

    def clear_cache(self):
        """Clear the audio feature cache."""
        self._audio_cache.clear()

    def preload_audio_features(self, stimulus_ids: List[str], n_timepoints: int):
        """
        Preload audio features for a list of stimuli.

        Args:
            stimulus_ids: List of stimulus identifiers
            n_timepoints: Expected number of timepoints
        """
        for stimulus_id in set(stimulus_ids):
            self._load_audio_features(stimulus_id, n_timepoints)
        print(f"Preloaded audio features for {len(set(stimulus_ids))} stimuli")
