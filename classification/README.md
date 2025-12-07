# EEG Familiarity Classification

Classification module for EEG-based familiarity detection from the Music Familiarity Experiment.

## Overview

This module provides a structured framework for:
- Loading preprocessed EEG data from MATLAB (.mat files)
- Merging EEG data with behavioral responses (familiarity/liking ratings)
- Extracting features (time-domain, frequency-domain, connectivity)
- Training classification models (SVM, Deep Learning, Statistical methods)
- Evaluating model performance with proper cross-validation

## Project Structure

```
classification/
├── config.py                 # Central configuration (paths, parameters, markers)
├── data/                     # Data loading and preprocessing
│   ├── loader.py            # Load .mat files and behavioral CSV
│   ├── preprocessor.py      # Epoch extraction, normalization
│   └── dataset.py           # Dataset classes (sklearn/PyTorch compatible)
├── features/                 # Feature extraction
│   ├── time_domain.py       # Variance, Hjorth parameters, zero-crossings
│   ├── frequency_domain.py  # Band power, spectral entropy, PSD
│   └── connectivity.py      # Correlation, coherence, PLV
├── models/                   # Classification models
│   ├── base.py              # Abstract base class (all models inherit from this)
│   ├── svm/                 # Support Vector Machine
│   ├── deep_learning/       # EEGNet, CNN-LSTM
│   └── statistical/         # LDA, Logistic Regression
├── evaluation/               # Model evaluation
│   ├── metrics.py           # Accuracy, F1, AUC, confusion matrix
│   └── cross_validation.py  # Stratified K-Fold, Leave-One-Subject-Out
├── utils/                    # Utilities
│   └── visualization.py     # Plotting functions
├── scripts/                  # Executable scripts
│   ├── train.py             # Main training script
│   └── evaluate.py          # Model comparison script
├── notebooks/                # Jupyter notebooks for exploration
└── results/                  # Output directory for models and figures
```

## Quick Start

### 1. Install Dependencies

```bash
# Core dependencies
pip install numpy scipy pandas scikit-learn matplotlib

# For .mat v7.3 files (HDF5 format)
pip install h5py

# For deep learning (optional)
pip install torch

# For MNE-based preprocessing (optional)
pip install mne
```

### 2. Load Data

```python
from data.loader import load_subject_data, create_labels

# Load preprocessed EEG and behavioral data
subject_data = load_subject_data(
    participant_id="sub-001",
    mat_filepath="path/to/sub-001_preprocessed.mat"
)

eeg_epochs = subject_data['eeg_data']  # Shape: (n_trials, n_channels, n_samples)
behavioral = subject_data['behavioral']  # DataFrame with ratings

# Create classification labels
labels = create_labels(behavioral, target_type="familiarity_binary")
```

### 3. Extract Features

```python
from features.frequency_domain import extract_frequency_features
from features.time_domain import extract_time_features

# Frequency-domain features (band power, spectral entropy)
freq_features = extract_frequency_features(eeg_epochs)

# Time-domain features (variance, Hjorth parameters)
time_features = extract_time_features(eeg_epochs)

# Combine features
import numpy as np
X = np.concatenate([freq_features, time_features], axis=1)
```

### 4. Train a Model

```python
from models.svm import SVMClassifier
from evaluation.cross_validation import cross_validate

# Create classifier
model = SVMClassifier(kernel="rbf", C=1.0)

# Cross-validation
results = cross_validate(
    model=model,
    X=X,
    y=labels,
    cv_strategy="stratified_kfold",
    n_folds=5
)

print(f"Accuracy: {results['mean_score']:.3f} ± {results['std_score']:.3f}")
```

### 5. Using the Training Script

```bash
python scripts/train.py \
    --subject sub-001 \
    --mat-file ../preprocessed_data/sub-001.mat \
    --model svm \
    --target familiarity_binary \
    --features frequency \
    --cv-folds 5
```

## Available Models

### SVM (Support Vector Machine)
```python
from models.svm import SVMClassifier, LinearSVM, RBFSVM

model = SVMClassifier(kernel="rbf", C=1.0, gamma="scale")
```

### LDA (Linear Discriminant Analysis)
```python
from models.statistical import LDAClassifier

model = LDAClassifier(solver="svd")
```

### Logistic Regression
```python
from models.statistical import LogisticClassifier

model = LogisticClassifier(C=1.0, penalty="l2")
```

### EEGNet (Deep Learning)
```python
from models.deep_learning import EEGNetClassifier

model = EEGNetClassifier(
    n_channels=59,
    n_samples=16000,  # 32s at 500Hz
    n_classes=2,
    dropout_rate=0.5
)
```

## Classification Targets

Available in `config.py`:

| Target | Description |
|--------|-------------|
| `familiarity_binary` | Familiar (4-5) vs Unfamiliar (1-2) |
| `liking_binary` | Liked (4-5) vs Disliked (1-2) |
| `origin_pool` | From familiar vs unfamiliar genre pool |
| `familiarity_multiclass` | 5-class (ratings 1-5) |

## Cross-Validation Strategies

### Stratified K-Fold
```python
from evaluation.cross_validation import cross_validate

results = cross_validate(model, X, y, cv_strategy="stratified_kfold", n_folds=5)
```

### Leave-One-Subject-Out (LOSO)
```python
results = cross_validate(
    model, X, y,
    cv_strategy="leave_one_subject_out",
    subject_ids=subject_ids
)
```

## Adding a New Model

1. Create a new file in the appropriate `models/` subdirectory
2. Inherit from `BaseClassifier`:

```python
from models.base import BaseClassifier

class MyNewClassifier(BaseClassifier):
    def __init__(self, param1=1.0, **kwargs):
        super().__init__(name="MyNewClassifier", **kwargs)
        self.param1 = param1
        # Initialize your model here

    def fit(self, X, y, **kwargs):
        # Training logic
        self.is_fitted = True
        return self

    def predict(self, X):
        # Prediction logic
        return predictions

    def predict_proba(self, X):
        # Optional: probability predictions
        return probabilities
```

3. Add to the module's `__init__.py`

## EEG Markers Reference

From the experiment:

| Marker | Event |
|--------|-------|
| 1 | EXPERIMENT_START |
| 2 | EXPERIMENT_END |
| 3 | BREAK_START |
| 4 | BREAK_END |
| 10 | BASELINE_START |
| 20 | STIMULUS_START |
| 30 | STIMULUS_END |

## Data Flow

```
┌─────────────────┐     ┌─────────────────┐
│  MATLAB/EEGLAB  │     │  Behavioral CSV │
│  Preprocessing  │     │  (Ratings)      │
└────────┬────────┘     └────────┬────────┘
         │                       │
         ▼                       ▼
    ┌────────────────────────────────┐
    │      data/loader.py            │
    │   load_subject_data()          │
    └───────────────┬────────────────┘
                    │
                    ▼
    ┌────────────────────────────────┐
    │    features/ extraction        │
    │  time_domain / frequency_domain│
    └───────────────┬────────────────┘
                    │
                    ▼
    ┌────────────────────────────────┐
    │      models/ classifiers       │
    │   SVM / DL / Statistical       │
    └───────────────┬────────────────┘
                    │
                    ▼
    ┌────────────────────────────────┐
    │   evaluation/ metrics & CV     │
    └────────────────────────────────┘
```

## Tips for Team Collaboration

1. **Each team member** can work in their own model subdirectory
2. **All models** should inherit from `BaseClassifier` for consistency
3. **Use the same** evaluation pipeline (`cross_validate`) for fair comparison
4. **Save results** to `results/` with descriptive filenames
5. **Document** hyperparameters and findings in notebooks

## Notebooks

Use the `notebooks/` directory for:
- Data exploration
- Feature visualization
- Model prototyping
- Results analysis

## References

- EEGNet: Lawhern et al. (2018). "EEGNet: A Compact Convolutional Neural Network for EEG-based Brain-Computer Interfaces"
- Band power features: Standard frequency bands (delta, theta, alpha, beta, gamma)
- Hjorth parameters: Hjorth (1970). "EEG analysis based on time domain properties"
