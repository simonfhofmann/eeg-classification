# EEG Music Familiarity Classification

Classification pipeline for EEG responses to familiar vs unfamiliar music.

## Directory Structure

```
Lab/
├── recording_logs/              # Behavioral data (familiarity/liking ratings)
│   ├── {name}_data.csv          # Trial-level ratings per participant
│   └── {name}_stimulus_order.csv
├── raw_eeg/                     # Raw EEG files (.vhdr, .eeg, .vmrk)
├── preprocessed_data/           # MATLAB-preprocessed .mat files (optional)
└── Recording/
    └── classification/          # This repo
        ├── config.py            # All paths & parameters
        ├── data/
        │   ├── containers.py    # EEGDataContainer class
        │   └── loaders/
        │       ├── raw_loader.py    # Load raw EEG + labels (recommended)
        │       └── matlab_loader.py # Load MATLAB-preprocessed data
        ├── models/
        │   └── deep_learning/   # EEGNet, trainer
        ├── features/            # Feature extraction (time/freq/connectivity)
        └── notebooks/
```

## Quick Start

```python
from data.loaders.raw_loader import load_raw_eeg

# Load EEG with familiarity labels
data = load_raw_eeg(
    filepath="path/to/participant.vhdr",
    participant_id="Sub01",  # Must match key in PARTICIPANT_INFO
    target_type="familiarity_binary"
)

print(data)          # EEGDataContainer(X=(n_trials, 30, 17500), sfreq=500Hz, ...)
print(data.X.shape)  # (n_trials, n_channels, n_timepoints)
print(data.y)        # Labels: 0=unfamiliar, 1=familiar
```

## Configuration (`config.py`)

### Paths
| Variable | Description |
|----------|-------------|
| `DATA_DIR` | Raw EEG files (.vhdr) |
| `LOGS_DIR` | Behavioral logs with ratings |
| `MAT_DATA_DIR` | MATLAB-preprocessed .mat files |

### EEG Parameters
| Parameter | Value | Description |
|-----------|-------|-------------|
| `SAMPLING_RATE` | 500 Hz | Target sampling rate |
| `STIMULUS_DURATION` | 32.0 s | Music clip length |
| `RAW_EPOCH_TMIN` | -3.0 s | Epoch start (includes baseline) |
| `RAW_EPOCH_TMAX` | 32.0 s | Epoch end |
| `N_CHANNELS` | 30 | EEG channels (excludes EOG/physio) |

### Participants (`PARTICIPANT_INFO`)
| ID | Name | Notes |
|----|------|-------|
| Sub01 | yannick | 60 trials, uses S6 end marker |
| Sub02 | daniel | Has duplicate stimuli in log |
| Sub03 | simon | Crash at trial 31 (auto-excluded) |
| Sub04 | karsten | - |
| Sub05 | philipp | - |

### EEG Markers
| Marker | Code | Description |
|--------|------|-------------|
| BASELINE_START | S1 | Baseline period starts |
| STIMULUS_START | S2 | Music onset (epoching reference) |
| STIMULUS_END | S5/S6 | Music ends |
| EXPERIMENT_RESUME | S14 | After crash recovery |

## Target Types

| Type | Description | Labels |
|------|-------------|--------|
| `familiarity_binary` | Rating 4-5 vs 1-2 (excludes 3) | 0/1 |
| `liking_binary` | Rating 4-5 vs 1-2 (excludes 3) | 0/1 |
| `origin_pool` | From familiar vs unfamiliar pool | 0/1 |
| `familiarity_multiclass` | Raw ratings | 0-5 |

## EEGDataContainer

Unified data container returned by all loaders.

### Attributes
```python
data.X              # np.ndarray (n_trials, n_channels, n_timepoints)
data.y              # np.ndarray (n_trials,) - labels
data.sfreq          # int - sampling frequency (500 Hz)
data.ch_names       # List[str] - 30 channel names
data.participant_id # str
data.metadata       # dict - preprocessing info, excluded trials, etc.
```

### Properties
```python
data.shape          # (n_trials, n_channels, n_timepoints)
data.n_trials       # number of trials
data.n_channels     # 30
data.n_timepoints   # 17500 (35s at 500Hz)
data.duration       # 35.0 seconds
```

### Methods
```python
data.select_trials(indices)       # Subset trials by index
data.select_channels(['Fz','Cz']) # Subset channels by name
data.to_microvolts()              # Convert V -> µV
data.copy()                       # Deep copy
```

## Log File Format

CSV files in `recording_logs/`:
```csv
participant_id,trial_num,stimulus_name,origin_pool,familiarity_rating,liking_rating
yannick,1,Song.wav,unfamiliar_pool,2,4
yannick,2,Song2.wav,familiar_pool,5,5
```

## Preprocessing Pipeline

The `RawEEGLoader` applies:

1. **Bandpass filter**: 0.5-60 Hz
2. **Notch filter**: 50 Hz (line noise)
3. **Resampling**: to 500 Hz
4. **Epoching**: -3.0 to 32.0 s around stimulus onset (S2 marker)
5. **Baseline correction**: -3.0 to -0.1 s
6. **Trial exclusion**: Crash trials auto-excluded per `PARTICIPANT_INFO`

### Custom Preprocessing
```python
from data.loaders.raw_loader import RawEEGLoader

config = {
    'l_freq': 1.0,           # High-pass cutoff
    'h_freq': 40.0,          # Low-pass cutoff
    'apply_ica': True,       # Enable ICA artifact removal
    'ica_n_components': 15,
}
loader = RawEEGLoader(preprocessing_config=config)
data = loader.load("file.vhdr", "Sub01")
```

## For Deep Learning

```python
import torch
from torch.utils.data import TensorDataset, DataLoader
from data.loaders.raw_loader import load_raw_eeg

# Load data
data = load_raw_eeg("file.vhdr", "Sub01", target_type="familiarity_binary")

# Convert to PyTorch
X = torch.tensor(data.X, dtype=torch.float32)
y = torch.tensor(data.y, dtype=torch.long)

# Create DataLoader
dataset = TensorDataset(X, y)
train_loader = DataLoader(dataset, batch_size=8, shuffle=True)

# Input shape for models: (batch, channels, timepoints) = (8, 30, 17500)
for X_batch, y_batch in train_loader:
    print(X_batch.shape)  # torch.Size([8, 30, 17500])
    break
```

## Loading Multiple Participants

```python
from data.loaders.raw_loader import RawEEGLoader
from config import PARTICIPANT_INFO, DATA_DIR
import numpy as np

loader = RawEEGLoader()
all_X, all_y, all_subjects = [], [], []

for subj_id, info in PARTICIPANT_INFO.items():
    eeg_path = DATA_DIR / info['eeg_file']
    data = loader.load(eeg_path, subj_id, target_type="origin_pool")

    all_X.append(data.X)
    all_y.append(data.y)
    all_subjects.extend([subj_id] * data.n_trials)

X = np.concatenate(all_X, axis=0)
y = np.concatenate(all_y, axis=0)
print(f"Total: {len(y)} trials from {len(PARTICIPANT_INFO)} participants")
```

## Metadata

After loading, `data.metadata` contains:
```python
{
    'source': 'raw_braindecode',
    'target_type': 'familiarity_binary',
    'n_epochs_original': 60,
    'n_epochs_final': 45,           # After exclusions
    'excluded_trials_crash': [32],  # Crash-related
    'excluded_trials_labels': [4, 5, ...],  # Ambiguous ratings
    'kept_trial_nums': [1, 2, 3, 6, ...],
    'preprocessing_config': {...},
}
```
