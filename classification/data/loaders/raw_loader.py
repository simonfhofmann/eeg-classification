# ----------------------------------------------------------------------
# raw_loader.py
#
# Loader for raw EEG data with Braindecode/MNE preprocessing.
# ----------------------------------------------------------------------

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, Union, List, Tuple, Any
import mne
from mne.preprocessing import ICA, create_eog_epochs

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from config import (
    SAMPLING_RATE,
    RAW_EPOCH_TMIN,
    RAW_EPOCH_TMAX,
    BASELINE_CORRECTION_TMIN,
    BASELINE_CORRECTION_TMAX,
    MARKERS,
    PARTICIPANT_INFO,
    LOGS_DIR,
    TARGET_TYPES,
    DEFAULT_TARGET,
)
from data.containers import EEGDataContainer
from data.loaders.base import BaseLoader


class RawEEGLoader(BaseLoader):
    """
    Load raw EEG data and preprocess using MNE/Braindecode.

    This loader is for raw EEG files (.edf, .bdf, .fif, .vhdr, etc.)
    that have not been preprocessed in MATLAB.

    The preprocessing pipeline includes:
    - Loading raw data
    - Filtering (bandpass, notch)
    - Resampling
    - Epoching based on events
    - Baseline correction
    - Optional: ICA artifact removal

    Args:
        preprocessing_config: Dictionary with preprocessing parameters:
            - l_freq: High-pass filter frequency (Hz)
            - h_freq: Low-pass filter frequency (Hz)
            - notch_freq: Notch filter frequency (Hz), None to skip
            - resample_freq: Target sampling frequency (Hz), None to keep original
            - epoch_tmin: Epoch start time relative to event (seconds)
            - epoch_tmax: Epoch end time relative to event (seconds)
            - baseline: Tuple (tmin, tmax) for baseline correction, None to skip
            - reject_bad_epochs: Whether to reject epochs with artifacts
            - reject_criteria: Dict with rejection thresholds
            - apply_ica: Whether to apply ICA for artifact removal
            - ica_n_components: Number of ICA components
            - ica_method: Algorithm ('fastica', 'infomax', 'picard')
            - ica_eog_channels: EOG channel names for auto-detection
            - ica_eog_threshold: Correlation threshold for EOG detection
            - ica_exclude_components: Manual list of components to exclude
            - interpolate_channels: Dict mapping trial indices to channels to interpolate

    Example:
        >>> config = {
        ...     'l_freq': 0.5,
        ...     'h_freq': 40.0,
        ...     'notch_freq': 50.0,
        ...     'resample_freq': 500,
        ...     'epoch_tmin': -0.5,
        ...     'epoch_tmax': 32.0,
        ...     'baseline': (-0.5, 0.0),
        ... }
        >>> loader = RawEEGLoader(preprocessing_config=config)
        >>> data = loader.load(
        ...     eeg_path="path/to/raw.edf",
        ...     participant_id="sub-001",
        ...     events_path="path/to/events.csv"
        ... )
    """

    DEFAULT_CONFIG = {
        # Filtering (matched to MATLAB/EEGLAB pipeline)
        'l_freq': 0.5,
        'h_freq': 60.0,               # Matched to MATLAB (was 45.0)
        'notch_freq': 50.0,
        'resample_freq': SAMPLING_RATE,
        # Epoching
        'epoch_tmin': RAW_EPOCH_TMIN,
        'epoch_tmax': RAW_EPOCH_TMAX,
        'baseline': (BASELINE_CORRECTION_TMIN, BASELINE_CORRECTION_TMAX),
        'reject_bad_epochs': False,
        'reject_criteria': {'eeg': 100e-6},  # 100 µV
        'event_id': {'stimulus': MARKERS['STIMULUS_START']},  # Lock to S2 (stimulus onset)
        # ICA settings
        'apply_ica': False,           # Whether to apply ICA artifact removal
        'ica_n_components': 20,       # Number of ICA components (None = max)
        'ica_method': 'infomax',      # ICA algorithm: 'infomax' (matches SOBI behavior), 'fastica', 'picard'
        'ica_random_state': 42,       # For reproducibility
        'ica_eog_channels': None,     # EOG channel names for automatic detection, None = use correlation
        'ica_eog_threshold': 0.4,     # Correlation threshold for EOG component detection
        'ica_exclude_components': None,  # Manual list of components to exclude (overrides auto)
        # Per-trial channel interpolation (matched to MATLAB)
        'interpolate_channels': None,  # Dict mapping trial indices to channel names/indices to interpolate
    }

    def __init__(self, preprocessing_config: Optional[Dict[str, Any]] = None):
        self.config = {**self.DEFAULT_CONFIG}
        if preprocessing_config:
            self.config.update(preprocessing_config)
        self._ica_info = None  # Will be populated if ICA is applied

    def load(
        self,
        eeg_path: Union[str, Path],
        participant_id: str,
        behavioral_path: Optional[Union[str, Path]] = None,
        events: Optional[np.ndarray] = None,
        event_id: Optional[Dict[str, int]] = None,
        exclude_trials: Optional[List[int]] = None,
        load_labels: bool = True,
        target_type: Optional[str] = None,
    ) -> EEGDataContainer:
        """
        Load and preprocess raw EEG data.

        Args:
            eeg_path: Path to raw EEG file (.edf, .bdf, .fif, .vhdr)
            participant_id: Participant identifier (e.g., "Sub01", "Sub02")
            behavioral_path: Optional path to behavioral data
            events: Optional events array (n_events, 3). If None, extracted from file.
            event_id: Optional event ID mapping. If None, uses config default.
            exclude_trials: Optional list of trial indices (1-based) to exclude.
                If None, automatically excludes repeated trials from crashes
                based on PARTICIPANT_INFO in config.
            load_labels: Whether to load familiarity/liking labels from log files.
                Default True.
            target_type: Type of target variable for labels. Options:
                - 'familiarity_binary': Familiar (4-5) vs unfamiliar (1-2)
                - 'liking_binary': Liked (4-5) vs disliked (1-2)
                - 'origin_pool': From familiar vs unfamiliar genre pool
                - 'familiarity_multiclass': 5-class ratings
                If None, uses DEFAULT_TARGET from config.

        Returns:
            EEGDataContainer with preprocessed data and labels (if load_labels=True)
        """
        eeg_path = Path(eeg_path)
        self.validate_paths(eeg_path)

        # Load raw data
        raw = self._load_raw(eeg_path)

        # Apply preprocessing
        raw = self._preprocess_raw(raw)

        # Get or create events
        if events is None:
            events = self._extract_events(raw)

        # Create epochs
        epochs = self._create_epochs(raw, events, event_id)

        # Apply per-trial channel interpolation if configured
        if self.config['interpolate_channels']:
            epochs = self._interpolate_channels_per_trial(epochs)

        # Get trials to exclude (from crashes or manual specification)
        if exclude_trials is None:
            exclude_trials = self._get_crash_trials(participant_id)

        # Track all trial numbers (1-based) - initially all epochs
        n_total_epochs = len(epochs)
        all_trial_nums = list(range(1, n_total_epochs + 1))

        # Load labels if requested
        y = None
        label_excluded_trials = []
        kept_trial_nums = all_trial_nums.copy()

        if load_labels:
            try:
                log_df = self._load_log_file(participant_id)
                y, kept_trial_nums = self._create_labels(
                    log_df,
                    target_type=target_type,
                    exclude_trials=exclude_trials
                )
                # Trials excluded due to ambiguous ratings (for binary targets)
                label_excluded_trials = [
                    t for t in all_trial_nums
                    if t not in kept_trial_nums and t not in (exclude_trials or [])
                ]
                if label_excluded_trials:
                    print(f"Excluded {len(label_excluded_trials)} trials due to ambiguous ratings: {label_excluded_trials}")
            except (ValueError, FileNotFoundError) as e:
                print(f"Warning: Could not load labels: {e}")
                load_labels = False

        # Determine final set of trials to keep
        if load_labels and kept_trial_nums:
            # Keep only trials that have valid labels
            keep_indices = [t - 1 for t in kept_trial_nums]  # Convert to 0-based
        else:
            # Just exclude crash trials
            exclude_set = set(exclude_trials) if exclude_trials else set()
            keep_indices = [i for i in range(n_total_epochs) if (i + 1) not in exclude_set]

        # Filter epochs
        epochs = epochs[keep_indices]
        n_final = len(epochs)

        if exclude_trials:
            print(f"Excluded {len(exclude_trials)} crash/manual trials: {exclude_trials}")
        print(f"Final epoch count: {n_final} (from {n_total_epochs} original)")

        # Extract data
        X = epochs.get_data()  # Shape: (n_trials, n_channels, n_timepoints)

        # Verify label count matches epoch count
        if y is not None and len(y) != len(X):
            raise ValueError(f"Label count ({len(y)}) doesn't match epoch count ({len(X)})")

        # Build metadata
        metadata = {
            'source': 'raw_braindecode',
            'preprocessing_config': self.config.copy(),
            'raw_file': str(eeg_path),
            'n_events_original': len(events),
            'n_epochs_original': n_total_epochs,
            'n_epochs_final': n_final,
            'excluded_trials_crash': exclude_trials,
            'excluded_trials_labels': label_excluded_trials,
            'kept_trial_nums': kept_trial_nums if load_labels else None,
            'original_sfreq': raw.info['sfreq'],
            'target_type': target_type if load_labels else None,
        }

        # Add ICA info if applied
        if self.config['apply_ica'] and hasattr(self, '_ica_info'):
            metadata['ica'] = self._ica_info

        # Add interpolation info if applied
        if self.config['interpolate_channels']:
            metadata['interpolated_channels'] = self.config['interpolate_channels']

        container = EEGDataContainer(
            X=X.astype(np.float32),
            y=y,
            sfreq=int(epochs.info['sfreq']),
            ch_names=epochs.ch_names,
            participant_id=participant_id,
            metadata=metadata
        )

        return container

    def _get_crash_trials(self, participant_id: str) -> List[int]:
        """
        Get trial indices to exclude based on crash information in config.

        Args:
            participant_id: Participant identifier (e.g., "Sub01", "Sub03")

        Returns:
            List of trial indices (1-based) to exclude
        """
        exclude = []

        # Look up participant info
        if participant_id in PARTICIPANT_INFO:
            info = PARTICIPANT_INFO[participant_id]
            for crash in info.get('crashes', []):
                # The repeated trial should be excluded
                repeated_trial = crash.get('repeated_eeg_trial')
                if repeated_trial:
                    exclude.append(repeated_trial)

        return exclude

    def _load_log_file(self, participant_id: str) -> pd.DataFrame:
        """
        Load the behavioral log file for a participant.

        Args:
            participant_id: Participant identifier (e.g., "Sub01", "Sub03")

        Returns:
            DataFrame with behavioral data including familiarity and liking ratings
        """
        if participant_id not in PARTICIPANT_INFO:
            raise ValueError(f"Unknown participant: {participant_id}")

        log_filename = PARTICIPANT_INFO[participant_id].get('log_file')
        if not log_filename:
            raise ValueError(f"No log file specified for {participant_id}")

        log_path = LOGS_DIR / log_filename
        if not log_path.exists():
            raise FileNotFoundError(f"Log file not found: {log_path}")

        df = pd.read_csv(log_path)

        # Sort by trial number to ensure correct order
        df = df.sort_values('trial_num').reset_index(drop=True)

        return df

    def _create_labels(
        self,
        log_df: pd.DataFrame,
        target_type: str = None,
        exclude_trials: Optional[List[int]] = None,
    ) -> Tuple[np.ndarray, List[int]]:
        """
        Create labels from log data based on target type.

        Args:
            log_df: DataFrame with behavioral data
            target_type: Type of target variable (from TARGET_TYPES in config).
                Options: 'familiarity_binary', 'liking_binary', 'origin_pool',
                'familiarity_multiclass'. If None, uses DEFAULT_TARGET.
            exclude_trials: List of trial indices (1-based) to exclude

        Returns:
            Tuple of (labels array, list of valid trial indices kept)
        """
        if target_type is None:
            target_type = DEFAULT_TARGET

        if target_type not in TARGET_TYPES:
            raise ValueError(f"Unknown target type: {target_type}. "
                           f"Available: {list(TARGET_TYPES.keys())}")

        # Get trials to keep (exclude crash trials etc.)
        exclude_set = set(exclude_trials) if exclude_trials else set()
        valid_mask = ~log_df['trial_num'].isin(exclude_set)
        df_valid = log_df[valid_mask].copy()

        # Create labels based on target type
        if target_type == 'familiarity_binary':
            threshold = TARGET_TYPES[target_type]['threshold']
            # Familiar (4-5) = 1, Unfamiliar (1-2) = 0, exclude middle (3)
            labels = []
            keep_indices = []
            for idx, row in df_valid.iterrows():
                rating = row['familiarity_rating']
                if rating >= threshold + 1:  # 4-5 = familiar
                    labels.append(1)
                    keep_indices.append(row['trial_num'])
                elif rating <= threshold - 1:  # 1-2 = unfamiliar
                    labels.append(0)
                    keep_indices.append(row['trial_num'])
                # Skip ratings == threshold (ambiguous)
            return np.array(labels, dtype=np.int64), keep_indices

        elif target_type == 'liking_binary':
            threshold = TARGET_TYPES[target_type]['threshold']
            labels = []
            keep_indices = []
            for idx, row in df_valid.iterrows():
                rating = row['liking_rating']
                if rating >= threshold + 1:  # 4-5 = liked
                    labels.append(1)
                    keep_indices.append(row['trial_num'])
                elif rating <= threshold - 1:  # 1-2 = disliked
                    labels.append(0)
                    keep_indices.append(row['trial_num'])
            return np.array(labels, dtype=np.int64), keep_indices

        elif target_type == 'origin_pool':
            # Binary: familiar_pool = 1, unfamiliar_pool = 0
            labels = (df_valid['origin_pool'] == 'familiar_pool').astype(np.int64).values
            keep_indices = df_valid['trial_num'].tolist()
            return labels, keep_indices

        elif target_type == 'familiarity_multiclass':
            # 5-class: ratings 1-5 (or 0-5 mapped to classes)
            labels = df_valid['familiarity_rating'].astype(np.int64).values
            keep_indices = df_valid['trial_num'].tolist()
            return labels, keep_indices

        else:
            raise ValueError(f"Target type '{target_type}' not implemented")

    def _interpolate_channels_per_trial(
        self,
        epochs: mne.Epochs
    ) -> mne.Epochs:
        """
        Interpolate bad channels in specific trials only.

        This matches MATLAB/EEGLAB's eeg_interp functionality where you can
        interpolate a channel only in certain trials where it's bad, while
        keeping the original data in trials where it's good.

        Config format for 'interpolate_channels':
            {
                trial_idx: [channel_names or indices],  # 0-indexed trials
                ...
            }

        Example:
            {
                4: ['Fp1', 'Fp2'],   # Interpolate Fp1, Fp2 in trial 5 (0-indexed: 4)
                48: ['FC1'],         # Interpolate FC1 in trial 49
                56: ['Cz'],          # Interpolate Cz in trial 57
            }

        Args:
            epochs: MNE Epochs object

        Returns:
            Epochs with interpolated channels in specified trials
        """
        interp_config = self.config['interpolate_channels']
        if not interp_config:
            return epochs

        # Get epochs data (n_trials, n_channels, n_times)
        data = epochs.get_data().copy()
        ch_names = epochs.ch_names
        info = epochs.info.copy()

        # Set up montage for interpolation (needed for spherical spline)
        if epochs.get_montage() is None:
            # Try to set standard 10-20 montage
            try:
                montage = mne.channels.make_standard_montage('standard_1020')
                epochs_temp = epochs.copy()
                epochs_temp.set_montage(montage, on_missing='ignore')
                info = epochs_temp.info.copy()
            except Exception as e:
                print(f"Warning: Could not set montage for interpolation: {e}")
                print("Interpolation may be less accurate without channel positions.")

        n_interpolated = 0
        for trial_idx, bad_channels in interp_config.items():
            trial_idx = int(trial_idx)
            if trial_idx >= len(data):
                print(f"Warning: Trial index {trial_idx} out of range, skipping.")
                continue

            # Convert channel names to indices if needed
            if isinstance(bad_channels, str):
                bad_channels = [bad_channels]

            bad_ch_indices = []
            bad_ch_names = []
            for ch in bad_channels:
                if isinstance(ch, int):
                    if ch < len(ch_names):
                        bad_ch_indices.append(ch)
                        bad_ch_names.append(ch_names[ch])
                elif isinstance(ch, str):
                    if ch in ch_names:
                        bad_ch_indices.append(ch_names.index(ch))
                        bad_ch_names.append(ch)
                    else:
                        print(f"Warning: Channel '{ch}' not found, skipping.")

            if not bad_ch_indices:
                continue

            # Create a temporary Raw object for this single trial for interpolation
            # This is a workaround since MNE's interpolate_bads works on Raw/Epochs level
            trial_data = data[trial_idx:trial_idx+1, :, :]  # Keep 3D shape

            # Create temporary epochs with just this trial
            temp_epochs = mne.EpochsArray(
                trial_data,
                info,
                verbose=False
            )

            # Mark channels as bad
            temp_epochs.info['bads'] = bad_ch_names

            # Interpolate bad channels
            try:
                temp_epochs.interpolate_bads(reset_bads=True, verbose=False)
                # Replace the trial data with interpolated data
                data[trial_idx] = temp_epochs.get_data()[0]
                n_interpolated += 1
            except Exception as e:
                print(f"Warning: Could not interpolate trial {trial_idx}: {e}")

        print(f"Interpolated channels in {n_interpolated} trials")

        # Create new epochs with interpolated data
        new_epochs = mne.EpochsArray(
            data,
            epochs.info,
            events=epochs.events,
            tmin=epochs.tmin,
            event_id=epochs.event_id,
            verbose=False
        )

        return new_epochs

    def load_behavioral(
        self,
        behavioral_path: Union[str, Path],
        participant_id: str
    ) -> pd.DataFrame:
        """
        Load behavioral data.

        Args:
            behavioral_path: Path to behavioral data file
            participant_id: Participant identifier

        Returns:
            DataFrame with behavioral data
        """
        behavioral_path = Path(behavioral_path)
        self.validate_paths(behavioral_path)

        if behavioral_path.is_file():
            return pd.read_csv(behavioral_path)

        # Try to find file in directory
        possible_files = [
            behavioral_path / f"{participant_id}_data.csv",
            behavioral_path / f"{participant_id}.csv",
        ]

        for f in possible_files:
            if f.exists():
                return pd.read_csv(f)

        raise FileNotFoundError(
            f"Behavioral data not found for {participant_id} in {behavioral_path}"
        )

    def _load_raw(self, filepath: Path) -> mne.io.BaseRaw:
        """
        Load raw EEG data based on file extension.

        Args:
            filepath: Path to raw EEG file

        Returns:
            MNE Raw object
        """
        suffix = filepath.suffix.lower()

        if suffix == '.edf':
            raw = mne.io.read_raw_edf(filepath, preload=True, verbose=False)
        elif suffix == '.bdf':
            raw = mne.io.read_raw_bdf(filepath, preload=True, verbose=False)
        elif suffix in ['.fif', '.fif.gz']:
            raw = mne.io.read_raw_fif(filepath, preload=True, verbose=False)
        elif suffix == '.vhdr':
            raw = mne.io.read_raw_brainvision(filepath, preload=True, verbose=False)
        elif suffix == '.set':
            raw = mne.io.read_raw_eeglab(filepath, preload=True, verbose=False)
        else:
            raise ValueError(f"Unsupported file format: {suffix}")

        return raw

    def _preprocess_raw(self, raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
        """
        Apply preprocessing to raw data.

        Args:
            raw: MNE Raw object

        Returns:
            Preprocessed Raw object
        """
        # Pick only EEG channels
        raw = raw.pick_types(eeg=True, exclude='bads')

        # Apply bandpass filter
        if self.config['l_freq'] or self.config['h_freq']:
            raw = raw.filter(
                l_freq=self.config['l_freq'],
                h_freq=self.config['h_freq'],
                verbose=False
            )

        # Apply notch filter
        if self.config['notch_freq']:
            raw = raw.notch_filter(
                freqs=self.config['notch_freq'],
                verbose=False
            )

        # Resample if needed
        if self.config['resample_freq'] and raw.info['sfreq'] != self.config['resample_freq']:
            raw = raw.resample(self.config['resample_freq'], verbose=False)

        # Apply ICA if enabled
        if self.config['apply_ica']:
            raw, ica_info = self._apply_ica(raw)
            self._ica_info = ica_info  # Store for metadata

        return raw

    def _apply_ica(
        self,
        raw: mne.io.BaseRaw,
        eog_raw: Optional[mne.io.BaseRaw] = None
    ) -> Tuple[mne.io.BaseRaw, Dict[str, Any]]:
        """
        Apply ICA for artifact removal (eye blinks, eye movements, muscle artifacts).

        The method:
        1. Fits ICA on the raw data
        2. Automatically detects EOG-related components using correlation or EOG channels
        3. Removes the identified artifact components

        Args:
            raw: Preprocessed Raw object (filtered, resampled)
            eog_raw: Optional separate Raw with EOG channels for detection

        Returns:
            Tuple of (cleaned Raw object, dict with ICA info)
        """
        ica_info = {
            'n_components': self.config['ica_n_components'],
            'method': self.config['ica_method'],
            'excluded_components': [],
            'detection_method': None,
        }

        # Create ICA object
        ica = ICA(
            n_components=self.config['ica_n_components'],
            method=self.config['ica_method'],
            random_state=self.config['ica_random_state'],
            max_iter='auto',
        )

        # Fit ICA on raw data
        # Use a higher filter for fitting to improve decomposition
        raw_for_ica = raw.copy().filter(l_freq=1.0, h_freq=None, verbose=False)
        ica.fit(raw_for_ica, verbose=False)

        # Determine which components to exclude
        exclude_indices = []

        if self.config['ica_exclude_components'] is not None:
            # Manual specification takes precedence
            exclude_indices = list(self.config['ica_exclude_components'])
            ica_info['detection_method'] = 'manual'
            print(f"ICA: Using manually specified components: {exclude_indices}")

        elif self.config['ica_eog_channels']:
            # Use EOG channels for automatic detection
            eog_channels = self.config['ica_eog_channels']
            if isinstance(eog_channels, str):
                eog_channels = [eog_channels]

            for eog_ch in eog_channels:
                if eog_ch in raw.ch_names:
                    eog_indices, eog_scores = ica.find_bads_eog(
                        raw,
                        ch_name=eog_ch,
                        threshold=self.config['ica_eog_threshold'],
                        verbose=False
                    )
                    exclude_indices.extend(eog_indices)

            exclude_indices = list(set(exclude_indices))  # Remove duplicates
            ica_info['detection_method'] = 'eog_channel'
            print(f"ICA: Detected {len(exclude_indices)} EOG components via channels: {exclude_indices}")

        else:
            # Use correlation-based detection with frontal channels
            # Frontal channels typically capture eye artifacts well
            frontal_channels = [ch for ch in raw.ch_names if ch.startswith(('Fp', 'AF', 'F'))]

            if frontal_channels:
                # Find components correlated with frontal activity
                exclude_indices = self._detect_eog_components_by_correlation(
                    ica, raw, frontal_channels
                )
                ica_info['detection_method'] = 'frontal_correlation'
                print(f"ICA: Detected {len(exclude_indices)} artifact components via frontal correlation: {exclude_indices}")
            else:
                # Fallback: exclude components with high variance in first few
                print("ICA: No EOG channels or frontal channels found. Skipping auto-detection.")
                ica_info['detection_method'] = 'none'

        # Apply ICA to remove artifact components
        if exclude_indices:
            ica.exclude = exclude_indices
            raw_clean = ica.apply(raw.copy(), verbose=False)
            ica_info['excluded_components'] = exclude_indices
        else:
            raw_clean = raw
            print("ICA: No components excluded.")

        return raw_clean, ica_info

    def _detect_eog_components_by_correlation(
        self,
        ica: ICA,
        raw: mne.io.BaseRaw,
        frontal_channels: List[str],
        threshold: Optional[float] = None
    ) -> List[int]:
        """
        Detect EOG-related ICA components by correlation with frontal channels.

        Eye artifacts typically have high correlation with frontal EEG channels
        (Fp1, Fp2, AF3, AF4, etc.) and show characteristic spatial patterns.

        Args:
            ica: Fitted ICA object
            raw: Raw data
            frontal_channels: List of frontal channel names
            threshold: Correlation threshold (default: use config)

        Returns:
            List of component indices to exclude
        """
        if threshold is None:
            threshold = self.config['ica_eog_threshold']

        exclude = []

        # Get ICA sources
        sources = ica.get_sources(raw).get_data()

        # Get frontal channel data
        frontal_idx = [raw.ch_names.index(ch) for ch in frontal_channels if ch in raw.ch_names]
        if not frontal_idx:
            return exclude

        frontal_data = raw.get_data(picks=frontal_idx)

        # Compute correlation between each component and frontal channels
        for comp_idx in range(sources.shape[0]):
            comp_data = sources[comp_idx]

            # Correlate with each frontal channel
            for frontal_ch_data in frontal_data:
                corr = np.abs(np.corrcoef(comp_data, frontal_ch_data)[0, 1])
                if corr > threshold:
                    exclude.append(comp_idx)
                    break  # Only add once per component

        return list(set(exclude))

    def _extract_events(self, raw: mne.io.BaseRaw) -> np.ndarray:
        """
        Extract events from raw data.

        Args:
            raw: MNE Raw object

        Returns:
            Events array (n_events, 3)
        """
        try:
            events = mne.find_events(raw, verbose=False)
        except ValueError:
            # Try annotations if no stim channel
            events, _ = mne.events_from_annotations(raw, verbose=False)

        return events

    def _create_epochs(
        self,
        raw: mne.io.BaseRaw,
        events: np.ndarray,
        event_id: Optional[Dict[str, int]] = None
    ) -> mne.Epochs:
        """
        Create epochs from raw data and events.

        Args:
            raw: Preprocessed Raw object
            events: Events array
            event_id: Event ID mapping

        Returns:
            MNE Epochs object
        """
        # Use provided event_id or config or all events
        if event_id is None:
            event_id = self.config.get('event_id')

        # Set rejection criteria
        reject = None
        if self.config['reject_bad_epochs']:
            reject = self.config['reject_criteria']

        epochs = mne.Epochs(
            raw,
            events,
            event_id=event_id,
            tmin=self.config['epoch_tmin'],
            tmax=self.config['epoch_tmax'],
            baseline=self.config['baseline'],
            reject=reject,
            preload=True,
            verbose=False
        )

        return epochs


# Convenience function for quick loading
def load_raw_eeg(
    filepath: Union[str, Path],
    participant_id: str,
    config: Optional[Dict] = None,
    load_labels: bool = True,
    target_type: Optional[str] = None,
    **kwargs
) -> EEGDataContainer:
    """
    Convenience function to load raw EEG data with labels.

    Args:
        filepath: Path to raw EEG file
        participant_id: Participant identifier
        config: Preprocessing configuration
        load_labels: Whether to load familiarity/liking labels from log files.
            Default True.
        target_type: Type of target variable for labels. Options:
            - 'familiarity_binary': Familiar (4-5) vs unfamiliar (1-2)
            - 'liking_binary': Liked (4-5) vs disliked (1-2)
            - 'origin_pool': From familiar vs unfamiliar genre pool
            - 'familiarity_multiclass': 5-class ratings
            If None, uses DEFAULT_TARGET from config.
        **kwargs: Additional arguments passed to RawEEGLoader.load()

    Returns:
        EEGDataContainer with preprocessed data and labels
    """
    loader = RawEEGLoader(preprocessing_config=config)
    return loader.load(
        filepath,
        participant_id,
        load_labels=load_labels,
        target_type=target_type,
        **kwargs
    )
