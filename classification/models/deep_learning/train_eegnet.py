import numpy as np
import pandas as pd
import torch
import mne
import sys
from pathlib import Path
from scipy.io import loadmat
import h5py
from typing import Dict, Tuple, Optional, Union, List
import os

# Braindecode/Skorch imports
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from braindecode.models import EEGNet
from braindecode.classifier import EEGClassifier
from braindecode.util import set_random_seeds
from skorch.callbacks import LRScheduler, EarlyStopping
from torch.optim import AdamW
import torch.nn as nn


# =======================================================
# These values are assumed to be imported from your config.py
# You MUST verify these are correct for your data.
# =======================================================
LOGS_DIR = Path(r"C:\Users\simon\Documents\master\25WS\praktikum_eeg\code\eeg-classification\code_current\logs\yannick_data.csv") 
MAT_DATA_DIR = Path(r"C:\Users\simon\Documents\master\25WS\praktikum_eeg\code\eeg-classification\recordings\Sub01\data_eeg.mat") 
MAT_VARIABLE_NAME = 'cellArray'
DATA_SCALE_FACTOR = 1e-6 
SAMPLING_RATE = 500  
RAW_EPOCH_TMIN = -3.0 
BASELINE_CORRECTION_TMIN = -3.0 
BASELINE_CORRECTION_TMAX = -0.1
EPOCH_TMIN = 0.1 
EPOCH_TMAX = 32.0


def load_eeg_from_mat(filepath: Union[str, Path]) -> Dict:
    filepath = Path(filepath)
    mat = loadmat(str(filepath)) 
    
    main_struct = mat[MAT_VARIABLE_NAME][0, 0] 

    # Extract raw data - shape: (channels, timepoints, trials)
    raw_data = main_struct['data'][0, 0]
    sfreq = main_struct['srate'][0, 0].item()
    tmin = main_struct['xmin'][0, 0].item()

    # Extract and clean channel names
    chan_structs = main_struct['chanlocs'][0, 0]
    raw_labels = [ch['labels'][0] for ch in chan_structs[0]]

    ch_names = []
    for label in raw_labels:
        if isinstance(label, np.ndarray):
            label = label.item()
        if isinstance(label, np.ndarray):
            label = label.item()
        ch_names.append(str(label))

    # Transpose to (trials, channels, timepoints) for MNE compatibility
    data = np.transpose(raw_data, (2, 0, 1))

    return {
        'data': data,
        'sfreq': sfreq,
        'tmin': tmin,
        'ch_names': ch_names,
        'n_trials': data.shape[0],
        'n_channels': data.shape[1],
        'n_samples': data.shape[2]
    }


def create_mne_epochs(
    eeg_data: Dict,
    apply_baseline: bool = True,
    crop_epochs: bool = True,
    baseline: Optional[Tuple[float, float]] = None,
    tmin: Optional[float] = None,
    tmax: Optional[float] = None
) -> mne.EpochsArray:
    # Convert to volts
    data_volts = eeg_data['data'] * DATA_SCALE_FACTOR

    # Create MNE info object
    info = mne.create_info(
        ch_names=eeg_data['ch_names'],
        sfreq=eeg_data['sfreq'],
        ch_types='eeg'
    )

    # Create epochs with original tmin from the data
    epochs = mne.EpochsArray(data_volts, info, tmin=eeg_data['tmin'], verbose=False)

    # Apply baseline correction
    if apply_baseline:
        bl = baseline or (BASELINE_CORRECTION_TMIN, BASELINE_CORRECTION_TMAX)
        epochs.apply_baseline(baseline=bl, verbose=False)

    # Crop to stimulus period
    if crop_epochs:
        crop_tmin = tmin if tmin is not None else EPOCH_TMIN
        crop_tmax = tmax if tmax is not None else EPOCH_TMAX
        epochs = epochs.crop(tmin=crop_tmin, tmax=crop_tmax)

    return epochs


def load_behavioral_data(participant_id: str, logs_dir: Optional[Path] = None) -> pd.DataFrame:
    logs_dir = Path(logs_dir) if logs_dir else LOGS_DIR
    data_file = logs_dir / f"{participant_id}_data.csv"

    if not data_file.exists():
        raise FileNotFoundError(f"Behavioral data file not found: {data_file}")

    df = pd.read_csv(data_file)
    return df[['participant_id', 'trial_num', 'stimulus_name', 'origin_pool', 'familiarity_rating', 'liking_rating']]


def create_labels_and_filter(
    behavioral_df: pd.DataFrame,
    target_type: str = "familiarity_binary"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Creates binary labels from familiarity ratings and returns valid trial indices.
    
    Mapping:
    - Ratings 1-2: Unfamiliar (label = 0)
    - Rating 3: Dropped (excluded)
    - Ratings 4-5: Familiar (label = 1)
    
    Returns:
        labels: Binary labels (0 or 1) for valid trials only
        valid_indices: Indices of trials that should be kept (excludes rating=3)
    """
    if target_type != "familiarity_binary":
        raise ValueError(f"Unknown target type: {target_type}")
    
    ratings = behavioral_df["familiarity_rating"].values
    
    # Find valid trials (exclude rating = 3)
    valid_indices = np.where((ratings >= 1) & (ratings <= 2) | (ratings >= 4) & (ratings <= 5))[0]
    
    # Create binary labels for valid trials only
    # Ratings 1-2 -> 0 (unfamiliar), Ratings 4-5 -> 1 (familiar)
    valid_ratings = ratings[valid_indices]
    labels = (valid_ratings >= 4).astype(int)
    
    # Print statistics
    n_unfamiliar = np.sum(labels == 0)
    n_familiar = np.sum(labels == 1)
    n_dropped = len(ratings) - len(valid_indices)
    
    print(f"\nLabel Distribution:")
    print(f"  Unfamiliar (ratings 1-2): {n_unfamiliar} trials")
    print(f"  Familiar (ratings 4-5): {n_familiar} trials")
    print(f"  Dropped (rating 3): {n_dropped} trials")
    print(f"  Total kept: {len(valid_indices)} / {len(ratings)} trials")
    
    return labels, valid_indices


class SimpleDataset:
    """Minimal wrapper to make X, y arrays look like a braindecode dataset."""
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.description = {'n_channels': X.shape[1], 'n_times': X.shape[2]}


def load_and_prepare_data(participant_id, mat_filepath, logs_dir, target_type="familiarity_binary"):
    """
    1. Loads, processes, and labels the data.
    2. Filters out trials with rating = 3.
    3. Converts data to microVolts and appropriate types.
    4. Splits data into Train/Validation/Test sets.
    """
    print("--- 1. Data Loading and Preprocessing ---")
    
    # Load and process EEG (X)
    raw_eeg = load_eeg_from_mat(mat_filepath)
    epochs = create_mne_epochs(raw_eeg, apply_baseline=True, crop_epochs=True)

    # Load behavioral data and create labels (y)
    behavioral_df = load_behavioral_data(participant_id, logs_dir)
    y_labels, valid_indices = create_labels_and_filter(behavioral_df, target_type=target_type)
    
    # --- Filter EEG data to keep only valid trials ---
    X_data_all = epochs.get_data().astype(np.float32) * 1e6  # Convert to microVolts
    X_data = X_data_all[valid_indices]  # Keep only trials without rating=3
    y_data = y_labels.astype(np.int64)

    print(f"\nFiltered X shape: {X_data.shape}, y shape: {y_data.shape}")

    # --- 2. Split Data ---
    print("\n--- 2. Splitting Data (Train/Valid/Test) ---")
    
    # Check if we have enough samples
    if len(X_data) < 10:
        raise ValueError(f"Not enough valid trials ({len(X_data)}) for training. Need at least 10 trials.")
    
    # 70% Train, 15% Validation, 15% Test (stratified by label)
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X_data, y_data, test_size=0.15, stratify=y_data, random_state=42
    )
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train_val, y_train_val, test_size=0.176, stratify=y_train_val, random_state=42
    )
    
    # Create Braindecode-compatible Datasets
    train_set = SimpleDataset(X_train, y_train)
    valid_set = SimpleDataset(X_valid, y_valid)
    test_set = SimpleDataset(X_test, y_test)
    
    print(f"Train/Valid/Test split sizes: {len(X_train)}/{len(X_valid)}/{len(X_test)}")
    print(f"  Train - Unfamiliar: {np.sum(y_train == 0)}, Familiar: {np.sum(y_train == 1)}")
    print(f"  Valid - Unfamiliar: {np.sum(y_valid == 0)}, Familiar: {np.sum(y_valid == 1)}")
    print(f"  Test  - Unfamiliar: {np.sum(y_test == 0)}, Familiar: {np.sum(y_test == 1)}")
    
    return train_set, valid_set, test_set


def create_and_train_model(train_set, valid_set, model_name='EEGNet', n_epochs=50, batch_size=32, lr=1e-3, test_overfitting=False):
    """
    Instantiates the EEGNet model and trains the EEGClassifier using skorch.
    
    Args:
        test_overfitting: If True, removes all regularization to test pure model capacity
    """
    cuda = torch.cuda.is_available()
    device = 'cuda' if cuda else 'cpu'
    set_random_seeds(seed=87, cuda=cuda) 

    n_chans = train_set.X.shape[1]
    n_times = train_set.X.shape[2]
    n_outputs = len(np.unique(train_set.y))

    print(f"\n--- 3. Model Setup ({model_name}) ---")
    print(f"Model parameters: {n_chans} channels, {n_outputs} outputs, {n_times} time points.")
    
    if test_overfitting:
        print("⚠️  OVERFITTING TEST MODE: All regularization disabled")

    # --- Calculate class weights for imbalanced data ---
    class_counts = np.bincount(train_set.y)
    total_samples = len(train_set.y)
    class_weights = total_samples / (len(class_counts) * class_counts)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
    
    print(f"\nClass distribution in training set:")
    print(f"  Class 0 (Unfamiliar): {class_counts[0]} samples (weight: {class_weights[0]:.3f})")
    print(f"  Class 1 (Familiar): {class_counts[1]} samples (weight: {class_weights[1]:.3f})")

    # --- 1. Model Instantiation ---
    if test_overfitting:
        # NO dropout for pure overfitting test
        model = EEGNet(n_chans, n_outputs, n_times=n_times, drop_prob=0.0).to(device)
    else:
        # With dropout for regularization
        model = EEGNet(n_chans, n_outputs, n_times=n_times, drop_prob=0.5).to(device)

    # --- 2. Classifier Setup ---
    # Reshape the input data for the 4D convolutional input: (n_trials, n_channels, n_times, 1)
    X_train_skorch = train_set.X.reshape(len(train_set.X), n_chans, n_times, 1)
    X_valid_skorch = valid_set.X.reshape(len(valid_set.X), n_chans, n_times, 1)

    # Create validation dataset for early stopping
    valid_dataset = SimpleDataset(X_valid_skorch, valid_set.y)
    
    if test_overfitting:
        # Minimal regularization for overfitting test
        clf = EEGClassifier(
            model,
            criterion=nn.NLLLoss,
            criterion__weight=class_weights_tensor,
            optimizer=AdamW,
            optimizer__weight_decay=0.0,  # NO weight decay
            train_split=None,  # Don't use validation during training
            max_epochs=n_epochs,
            batch_size=batch_size,
            lr=lr,
            device=device,
            callbacks=[
                ('lr_scheduler', LRScheduler('CosineAnnealingLR', T_max=n_epochs - 1)),
            ],
            verbose=1,
        )
    else:
        # Normal training with regularization
        clf = EEGClassifier(
            model,
            criterion=nn.NLLLoss,
            criterion__weight=class_weights_tensor,
            optimizer=AdamW,
            optimizer__weight_decay=0.01,
            train_split=lambda X, y: (X_train_skorch, train_set.y, X_valid_skorch, valid_set.y),
            max_epochs=n_epochs,
            batch_size=batch_size,
            lr=lr,
            device=device,
            callbacks=[
                ('early_stopping', EarlyStopping(monitor='valid_loss', patience=15, threshold=0.001, lower_is_better=True)),
                ('lr_scheduler', LRScheduler('CosineAnnealingLR', T_max=n_epochs - 1)),
            ],
            verbose=1,
        )

    # --- 3. Training ---
    print(f"\n--- 4. Training Model ---")
    print(f"Training for {n_epochs} epochs on {device}.")
    clf.fit(X_train_skorch, train_set.y)
    
    return clf


def evaluate_model(clf, train_set, valid_set, test_set):
    """
    Evaluates the trained classifier on all datasets (train, validation, test).
    Returns predictions and targets for all sets, and saves results to CSV.
    """
    n_chans = train_set.X.shape[1]
    n_times = train_set.X.shape[2]
    
    print("\n--- 5. Evaluation Results ---\n")
    
    # Evaluate on all datasets
    results = {}
    all_results_data = []
    
    for set_name, dataset in [("Train", train_set), ("Validation", valid_set), ("Test", test_set)]:
        X_reshaped = dataset.X.reshape(len(dataset.X), n_chans, n_times, 1)
        y_pred = clf.predict(X_reshaped)
        
        # Get prediction probabilities - need to convert from log probabilities
        y_log_proba = clf.predict_proba(X_reshaped)
        y_proba = np.exp(y_log_proba)  # Convert log probabilities to probabilities
        
        accuracy = accuracy_score(dataset.y, y_pred)
        
        results[set_name] = {
            'y_true': dataset.y,
            'y_pred': y_pred,
            'y_proba': y_proba,
            'accuracy': accuracy
        }
        
        print(f"{set_name} Set:")
        print(f"  Size: {len(dataset.y)}")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Prediction distribution: {np.bincount(y_pred)}")
        print(f"  Predictions vs Targets (with confidence):")
        for idx, (true_label, pred_label, proba) in enumerate(zip(dataset.y, y_pred, y_proba)):
            match = "✓" if true_label == pred_label else "✗"
            label_name = "Familiar" if true_label == 1 else "Unfamiliar"
            pred_name = "Familiar" if pred_label == 1 else "Unfamiliar"
            confidence = proba[pred_label] * 100
            print(f"    Sample {idx+1:2d}: Target={label_name}, Predicted={pred_name} ({confidence:.1f}% conf) {match}")
            all_results_data.append({
                'set': set_name,
                'sample_id': idx + 1,
                'target': true_label,
                'predicted': pred_label,
                'confidence': confidence,
                'prob_unfamiliar': proba[0],
                'prob_familiar': proba[1],
                'correct': true_label == pred_label
            })
        print()
    
    # Save results to CSV
    results_df = pd.DataFrame(all_results_data)
    results_file = Path(__file__).parent / "evaluation_results.csv"
    results_df.to_csv(results_file, index=False)
    print(f"Results saved to: {results_file}\n")
    
    return results


# --- Main Execution Block ---
if __name__ == '__main__':
    # --- Configuration ---
    PARTICIPANT_ID = "yannick"
    
    # !!! PATHS: Adjust these paths to match your local file locations !!!
    MAT_FILEPATH = Path(r"C:\Users\simon\Documents\master\25WS\praktikum_eeg\code\eeg-classification\recordings\Sub01\data_eeg.mat") 
    LOGS_DIR = Path(r"C:\Users\simon\Documents\master\25WS\praktikum_eeg\code\eeg-classification\code_current\logs")    
    
    # Training parameters
    N_EPOCHS = 50  # Reduced from 100 - early stopping will stop earlier anyway
    BATCH_SIZE = 16
    LEARNING_RATE = 5e-4  # Reduced learning rate
    
    # Set to True to test pure overfitting capacity (no regularization)
    TEST_OVERFITTING = True  # Toggle this to test model capacity
    
    print(f"\n{'='*60}")
    if TEST_OVERFITTING:
        print("MODE: Testing Model Capacity (Overfitting Test)")
        print("All regularization disabled - model should memorize training data")
    else:
        print("MODE: Normal Training (With Regularization)")
        print("Early stopping and regularization enabled")
    print(f"{'='*60}\n")
    
    try:
        # 1. Load and Prepare Data
        train_set, valid_set, test_set = load_and_prepare_data(
            PARTICIPANT_ID, MAT_FILEPATH, LOGS_DIR, target_type="familiarity_binary"
        )
        
        # 2. Create and Train Model
        classifier = create_and_train_model(
            train_set, valid_set, n_epochs=N_EPOCHS, batch_size=BATCH_SIZE, 
            lr=LEARNING_RATE, test_overfitting=TEST_OVERFITTING
        )
        
        # 3. Evaluate Model
        results = evaluate_model(classifier, train_set, valid_set, test_set)
        
        print("Pipeline finished successfully.")
        
    except FileNotFoundError as e:
        print(f"\nFATAL ERROR: Data file not found. Check your MAT_FILEPATH and LOGS_DIR constants. {e}")
    except Exception as e:
        print(f"\nAn unexpected error occurred during the pipeline: {e}")



                