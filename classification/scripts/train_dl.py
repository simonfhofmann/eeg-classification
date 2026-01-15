#!/usr/bin/env python
# ----------------------------------------------------------------------
# train_dl.py
#
# Main entry point for deep learning EEG classification.
# Demonstrates the refactored pipeline with clean separation of concerns.
# ----------------------------------------------------------------------

"""
EEGNet Training Script

This script demonstrates the refactored pipeline architecture:
1. Load data using MatlabPreprocessedLoader (or RawEEGLoader)
2. Preprocess: filter trials, create labels, split data
3. Train EEGNet using the trainer module
4. Evaluate using the evaluator module

Usage:
    python train_dl.py --mat_file path/to/data.mat --logs_dir path/to/logs

Example:
    python scripts/train_dl.py \
        --participant_id "yannick" \
        --mat_file "recordings/Sub01/data_eeg.mat" \
        --logs_dir "logs" \
        --output_dir "results/eegnet_run1"
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np

# Import from refactored modules
from config import RANDOM_STATE
from data.loaders import MatlabPreprocessedLoader
from data.preprocessor import (
    filter_trials_by_rating,
    create_binary_labels_exclude_neutral,
    stratified_train_val_test_split,
    scale_to_microvolts
)
from data.dataset import create_braindecode_datasets
from models.deep_learning.trainer import EEGNetTrainer, train_eegnet_pipeline
from evaluation.evaluator import ModelEvaluator


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train EEGNet for EEG classification",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required arguments
    parser.add_argument(
        "--mat_file",
        type=str,
        required=True,
        help="Path to MATLAB preprocessed .mat file"
    )
    parser.add_argument(
        "--logs_dir",
        type=str,
        required=True,
        help="Path to behavioral logs directory"
    )
    parser.add_argument(
        "--participant_id",
        type=str,
        required=True,
        help="Participant identifier"
    )

    # Optional arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Directory to save results"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Maximum training epochs"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Training batch size"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.5,
        help="Dropout probability"
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=15,
        help="Early stopping patience"
    )
    parser.add_argument(
        "--overfitting_test",
        action="store_true",
        help="Run overfitting test (disable regularization)"
    )
    parser.add_argument(
        "--exclude_ratings",
        type=int,
        nargs="+",
        default=[3],
        help="Ratings to exclude (default: neutral rating 3)"
    )

    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()

    # Convert paths
    mat_file = Path(args.mat_file)
    logs_dir = Path(args.logs_dir)
    output_dir = Path(args.output_dir)

    print("=" * 60)
    print("EEGNet Training Pipeline")
    print("=" * 60)

    # ----------------------------------------------------------------
    # Step 1: Load Data
    # ----------------------------------------------------------------
    print("\n--- Step 1: Loading Data ---")

    loader = MatlabPreprocessedLoader(
        apply_baseline=True,
        crop_epochs=True
    )

    # Load EEG data
    data = loader.load(
        eeg_path=mat_file,
        participant_id=args.participant_id
    )
    print(f"Loaded: {data}")

    # Load behavioral data
    behavioral_df = loader.load_behavioral(logs_dir, args.participant_id)
    print(f"Behavioral data: {len(behavioral_df)} trials")

    # ----------------------------------------------------------------
    # Step 2: Preprocess
    # ----------------------------------------------------------------
    print("\n--- Step 2: Preprocessing ---")

    # Create labels and filter trials (exclude neutral ratings)
    labels, valid_indices = create_binary_labels_exclude_neutral(
        behavioral_df,
        rating_column="familiarity_rating",
        low_ratings=[1, 2],
        high_ratings=[4, 5]
    )

    # Filter EEG data to keep only valid trials
    data_filtered = data.select_trials(valid_indices)
    data_filtered = data_filtered.set_labels(labels)

    print(f"After filtering: {data_filtered}")

    # Scale to microvolts for neural network
    data_scaled = scale_to_microvolts(data_filtered)

    # ----------------------------------------------------------------
    # Step 3: Split Data
    # ----------------------------------------------------------------
    print("\n--- Step 3: Splitting Data ---")

    splits = stratified_train_val_test_split(
        data_scaled,
        train_size=0.7,
        val_size=0.15,
        test_size=0.15,
        random_state=RANDOM_STATE
    )

    print(f"Splits: {splits}")

    # ----------------------------------------------------------------
    # Step 4: Train Model
    # ----------------------------------------------------------------
    print("\n--- Step 4: Training Model ---")

    class_names = ['unfamiliar', 'familiar']

    trainer = EEGNetTrainer(
        n_channels=splits.train.n_channels,
        n_classes=2,
        n_times=splits.train.n_timepoints,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        max_epochs=args.epochs,
        patience=args.patience,
        drop_prob=args.dropout,
        class_names=class_names
    )

    if args.overfitting_test:
        print("\n*** OVERFITTING TEST MODE ***")
        print("All regularization disabled - testing model capacity")
        trainer.fit_overfitting_test(splits.train, use_class_weights=True)
    else:
        trainer.fit(splits.train, splits.val, use_class_weights=True)

    # ----------------------------------------------------------------
    # Step 5: Evaluate
    # ----------------------------------------------------------------
    print("\n--- Step 5: Evaluation ---")

    eval_datasets = {
        'train': splits.train,
        'val': splits.val,
        'test': splits.test
    }

    results = trainer.evaluate(eval_datasets, verbose=True)

    # ----------------------------------------------------------------
    # Step 6: Save Results
    # ----------------------------------------------------------------
    print("\n--- Step 6: Saving Results ---")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Save model
    trainer.save(output_dir / 'model.pt')

    # Save detailed predictions
    evaluator = ModelEvaluator(
        model=trainer.classifier,
        class_names=class_names,
        output_dir=output_dir
    )

    # Get detailed predictions DataFrame
    predictions_df = evaluator.evaluate_with_predictions(
        {name: (trainer._prepare_data(data)[0], trainer._prepare_data(data)[1])
         for name, data in [('train', splits.train), ('val', splits.val), ('test', splits.test)]
         for data in [data]},
        verbose=False
    )

    # Save predictions
    predictions_df.to_csv(output_dir / 'predictions.csv', index=False)

    # Save summary metrics
    evaluator.save_results(results, 'metrics.csv')

    # Create and save report
    report = evaluator.create_report(results, model_name="EEGNet")
    with open(output_dir / 'report.txt', 'w') as f:
        f.write(report)

    print(f"\nResults saved to: {output_dir}")
    print("\nPipeline completed successfully!")


if __name__ == "__main__":
    main()
