#!/usr/bin/env python
# ----------------------------------------------------------------------
# evaluate.py
#
# Evaluation script for comparing multiple models.
# ----------------------------------------------------------------------

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from evaluation.metrics import (
    compute_metrics,
    classification_report,
    compare_to_chance
)
from utils.visualization import (
    plot_confusion_matrix,
    plot_cv_results
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate and compare classification results"
    )

    parser.add_argument(
        "--results", "-r",
        type=str,
        nargs="+",
        required=True,
        help="Path(s) to result files (.npy)"
    )

    parser.add_argument(
        "--output", "-o",
        type=str,
        default="./results",
        help="Output directory for figures"
    )

    parser.add_argument(
        "--permutation-test",
        action="store_true",
        help="Run permutation test against chance"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Model Evaluation Summary")
    print("=" * 60)

    all_results = []

    for result_path in args.results:
        result_path = Path(result_path)
        if not result_path.exists():
            print(f"Warning: {result_path} not found, skipping")
            continue

        results = np.load(result_path, allow_pickle=True).item()

        # Extract model name from filename
        model_name = result_path.stem

        print(f"\n{model_name}")
        print("-" * 40)
        print(f"Mean Accuracy: {results['mean_score']:.4f} ± {results['std_score']:.4f}")

        if 'overall_metrics' in results:
            for metric, value in results['overall_metrics'].items():
                if not np.isnan(value):
                    print(f"  {metric}: {value:.4f}")

        all_results.append({
            'name': model_name,
            'results': results
        })

        # Plot CV results
        fig, ax = plot_cv_results(
            results,
            title=f"Cross-Validation: {model_name}",
            save_path=output_dir / f"{model_name}_cv_results.png"
        )

        # Plot confusion matrix if predictions available
        if 'predictions' in results:
            # Need true labels - would need to load data again
            pass

    # Summary comparison
    if len(all_results) > 1:
        print("\n" + "=" * 60)
        print("Model Comparison")
        print("=" * 60)
        print(f"{'Model':<30} {'Accuracy':>10} {'Std':>10}")
        print("-" * 50)
        for r in sorted(all_results, key=lambda x: -x['results']['mean_score']):
            print(f"{r['name']:<30} {r['results']['mean_score']:>10.4f} {r['results']['std_score']:>10.4f}")

    print(f"\nFigures saved to: {output_dir}")


if __name__ == "__main__":
    main()
