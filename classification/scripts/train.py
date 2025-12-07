#!/usr/bin/env python
# ----------------------------------------------------------------------
# train.py
#
# Main training script for EEG classification.
# ----------------------------------------------------------------------

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from config import (
    RESULTS_DIR, DEFAULT_TARGET, N_FOLDS, RANDOM_STATE
)
from data.loader import load_subject_data, create_labels
from data.preprocessor import Preprocessor
from data.dataset import EEGDataset
from features.time_domain import extract_time_features
from features.frequency_domain import extract_frequency_features
from evaluation.cross_validation import cross_validate
from evaluation.metrics import compute_metrics


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train EEG classification model"
    )

    parser.add_argument(
        "--subject", "-s",
        type=str,
        required=True,
        help="Subject ID (e.g., sub-001)"
    )

    parser.add_argument(
        "--mat-file", "-m",
        type=str,
        required=True,
        help="Path to preprocessed .mat file"
    )

    parser.add_argument(
        "--model", "-M",
        type=str,
        default="svm",
        choices=["svm", "lda", "logistic", "eegnet"],
        help="Model type to train"
    )

    parser.add_argument(
        "--target", "-t",
        type=str,
        default=DEFAULT_TARGET,
        choices=["familiarity_binary", "liking_binary", "origin_pool"],
        help="Classification target"
    )

    parser.add_argument(
        "--features", "-f",
        type=str,
        default="frequency",
        choices=["time", "frequency", "both", "raw"],
        help="Feature type to extract"
    )

    parser.add_argument(
        "--cv-folds", "-k",
        type=int,
        default=N_FOLDS,
        help="Number of cross-validation folds"
    )

    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output directory for results"
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )

    return parser.parse_args()


def get_model(model_name: str):
    """Get model instance by name."""
    if model_name == "svm":
        from models.svm import SVMClassifier
        return SVMClassifier()
    elif model_name == "lda":
        from models.statistical import LDAClassifier
        return LDAClassifier()
    elif model_name == "logistic":
        from models.statistical import LogisticClassifier
        return LogisticClassifier()
    elif model_name == "eegnet":
        from models.deep_learning import EEGNetClassifier
        return EEGNetClassifier()
    else:
        raise ValueError(f"Unknown model: {model_name}")


def extract_features(epochs: np.ndarray, feature_type: str) -> np.ndarray:
    """Extract features based on type."""
    if feature_type == "raw":
        return epochs.reshape(epochs.shape[0], -1)
    elif feature_type == "time":
        return extract_time_features(epochs)
    elif feature_type == "frequency":
        return extract_frequency_features(epochs)
    elif feature_type == "both":
        time_feat = extract_time_features(epochs)
        freq_feat = extract_frequency_features(epochs)
        return np.concatenate([time_feat, freq_feat], axis=1)
    else:
        raise ValueError(f"Unknown feature type: {feature_type}")


def main():
    args = parse_args()

    print("=" * 60)
    print("EEG Classification Training")
    print("=" * 60)
    print(f"Subject: {args.subject}")
    print(f"Model: {args.model}")
    print(f"Target: {args.target}")
    print(f"Features: {args.features}")
    print(f"CV Folds: {args.cv_folds}")
    print("=" * 60)

    # Load data
    print("\n[1/4] Loading data...")
    subject_data = load_subject_data(
        participant_id=args.subject,
        mat_filepath=args.mat_file
    )

    eeg_data = subject_data['eeg_data']
    behavioral = subject_data['behavioral']

    print(f"  Loaded {len(behavioral)} trials")

    # Create labels
    print("\n[2/4] Creating labels...")
    labels = create_labels(behavioral, target_type=args.target)
    unique, counts = np.unique(labels, return_counts=True)
    print(f"  Class distribution: {dict(zip(unique, counts))}")

    # Extract features
    print("\n[3/4] Extracting features...")

    # Note: You'll need to adapt this based on your actual .mat file structure
    # This assumes eeg_data contains epoched data
    if isinstance(eeg_data, dict):
        # If it's an EEGLAB structure, extract the data field
        if 'data' in eeg_data:
            epochs = eeg_data['data']
        else:
            print("  Warning: Could not find 'data' field in EEG structure")
            print(f"  Available keys: {list(eeg_data.keys())}")
            return
    else:
        epochs = eeg_data

    # Ensure correct shape (n_trials, n_channels, n_samples)
    if epochs.ndim == 2:
        print("  Warning: Data appears to be continuous, not epoched")
        return

    print(f"  Epochs shape: {epochs.shape}")

    if args.model == "eegnet":
        # EEGNet uses raw epochs
        X = epochs
    else:
        X = extract_features(epochs, args.features)
        print(f"  Feature matrix shape: {X.shape}")

    # Train and evaluate
    print("\n[4/4] Cross-validation...")
    model = get_model(args.model)

    results = cross_validate(
        model=model,
        X=X,
        y=labels,
        cv_strategy="stratified_kfold",
        n_folds=args.cv_folds,
        return_predictions=True,
        verbose=args.verbose
    )

    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Mean Accuracy: {results['mean_score']:.4f} ± {results['std_score']:.4f}")
    print(f"\nOverall Metrics:")
    for metric, value in results['overall_metrics'].items():
        if not np.isnan(value):
            print(f"  {metric}: {value:.4f}")

    # Save results
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = RESULTS_DIR

    output_dir.mkdir(parents=True, exist_ok=True)

    results_file = output_dir / f"{args.subject}_{args.model}_{args.target}_results.npy"
    np.save(results_file, results)
    print(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    main()
