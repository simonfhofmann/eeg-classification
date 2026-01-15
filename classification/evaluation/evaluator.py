# ----------------------------------------------------------------------
# evaluator.py
#
# Standardized model evaluation for EEG classification.
# ----------------------------------------------------------------------

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from datetime import datetime

import sys
sys.path.append(str(Path(__file__).parent.parent))

from evaluation.metrics import (
    compute_metrics,
    compute_confusion_matrix,
    classification_report,
    sensitivity_specificity,
    compare_to_chance
)


class ModelEvaluator:
    """
    Standardized evaluation for EEG classification models.

    Provides a consistent interface for evaluating any model (deep learning
    or classical ML) on train/validation/test sets. Computes multiple metrics
    and saves results to CSV.

    Works with any model that follows the sklearn interface:
    - model.predict(X) -> predictions
    - model.predict_proba(X) -> probabilities (optional)

    Args:
        model: Trained model with predict() method
        class_names: Names for each class (e.g., ['unfamiliar', 'familiar'])
        output_dir: Directory to save results (optional)

    Example:
        >>> evaluator = ModelEvaluator(model, class_names=['unfamiliar', 'familiar'])
        >>> results = evaluator.evaluate({
        ...     'train': (X_train, y_train),
        ...     'val': (X_val, y_val),
        ...     'test': (X_test, y_test)
        ... })
        >>> evaluator.save_results(results, 'evaluation_results.csv')
    """

    def __init__(
        self,
        model: Any = None,
        class_names: Optional[List[str]] = None,
        output_dir: Optional[Union[str, Path]] = None
    ):
        self.model = model
        self.class_names = class_names or ['class_0', 'class_1']
        self.output_dir = Path(output_dir) if output_dir else None

    def evaluate(
        self,
        datasets: Dict[str, tuple],
        model: Optional[Any] = None,
        verbose: bool = True
    ) -> Dict[str, Dict]:
        """
        Evaluate model on multiple datasets.

        Args:
            datasets: Dictionary mapping split names to (X, y) tuples
                     e.g., {'train': (X_train, y_train), 'test': (X_test, y_test)}
            model: Model to evaluate (uses self.model if not provided)
            verbose: Whether to print results

        Returns:
            Dictionary with results for each dataset
        """
        model = model or self.model
        if model is None:
            raise ValueError("No model provided")

        results = {}

        for split_name, (X, y) in datasets.items():
            if verbose:
                print(f"\n--- Evaluating on {split_name} set ---")

            split_results = self._evaluate_split(model, X, y, split_name, verbose)
            results[split_name] = split_results

        return results

    def _evaluate_split(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        split_name: str,
        verbose: bool = True
    ) -> Dict:
        """Evaluate model on a single dataset split."""
        # Get predictions
        y_pred = model.predict(X)

        # Get probabilities if available
        y_proba = None
        if hasattr(model, 'predict_proba'):
            try:
                y_proba = model.predict_proba(X)
                # Handle log probabilities (from some neural network models)
                if y_proba.min() < 0:
                    y_proba = np.exp(y_proba)
            except Exception:
                pass

        # Compute metrics
        metrics = compute_metrics(y, y_pred, y_proba)

        # Compute confusion matrix
        conf_matrix = compute_confusion_matrix(y, y_pred)

        # Get classification report
        report = classification_report(
            y, y_pred,
            target_names=self.class_names,
            output_dict=True
        )

        # Binary-specific metrics
        if len(np.unique(y)) == 2:
            sens_spec = sensitivity_specificity(y, y_pred)
            metrics.update(sens_spec)

        results = {
            'split': split_name,
            'n_samples': len(y),
            'metrics': metrics,
            'confusion_matrix': conf_matrix,
            'classification_report': report,
            'y_true': y,
            'y_pred': y_pred,
            'y_proba': y_proba,
        }

        if verbose:
            self._print_results(results)

        return results

    def _print_results(self, results: Dict):
        """Print evaluation results."""
        metrics = results['metrics']
        split = results['split']

        print(f"\n{split.upper()} Results:")
        print(f"  Samples: {results['n_samples']}")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
        print(f"  F1 Score: {metrics['f1']:.4f}")

        if 'sensitivity' in metrics:
            print(f"  Sensitivity: {metrics['sensitivity']:.4f}")
            print(f"  Specificity: {metrics['specificity']:.4f}")

        if 'auc' in metrics:
            print(f"  AUC-ROC: {metrics['auc']:.4f}")

        print(f"\n  Confusion Matrix:")
        print(f"    {results['confusion_matrix']}")

    def evaluate_with_predictions(
        self,
        datasets: Dict[str, tuple],
        model: Optional[Any] = None,
        verbose: bool = True
    ) -> pd.DataFrame:
        """
        Evaluate model and return detailed per-sample predictions.

        Args:
            datasets: Dictionary mapping split names to (X, y) tuples
            model: Model to evaluate
            verbose: Whether to print results

        Returns:
            DataFrame with per-sample predictions and probabilities
        """
        model = model or self.model
        if model is None:
            raise ValueError("No model provided")

        all_records = []

        for split_name, (X, y) in datasets.items():
            y_pred = model.predict(X)

            # Get probabilities
            y_proba = None
            if hasattr(model, 'predict_proba'):
                try:
                    y_proba = model.predict_proba(X)
                    if y_proba.min() < 0:
                        y_proba = np.exp(y_proba)
                except Exception:
                    pass

            for idx in range(len(y)):
                record = {
                    'split': split_name,
                    'sample_idx': idx,
                    'y_true': y[idx],
                    'y_pred': y_pred[idx],
                    'correct': y[idx] == y_pred[idx],
                    'true_label': self.class_names[y[idx]] if y[idx] < len(self.class_names) else str(y[idx]),
                    'pred_label': self.class_names[y_pred[idx]] if y_pred[idx] < len(self.class_names) else str(y_pred[idx]),
                }

                if y_proba is not None:
                    record['confidence'] = y_proba[idx, y_pred[idx]]
                    for class_idx, class_name in enumerate(self.class_names):
                        if class_idx < y_proba.shape[1]:
                            record[f'prob_{class_name}'] = y_proba[idx, class_idx]

                all_records.append(record)

        df = pd.DataFrame(all_records)

        if verbose:
            # Print summary per split
            for split in df['split'].unique():
                split_df = df[df['split'] == split]
                acc = split_df['correct'].mean()
                print(f"\n{split}: {acc:.4f} accuracy ({split_df['correct'].sum()}/{len(split_df)} correct)")

        return df

    def compare_to_chance_level(
        self,
        y_test: np.ndarray,
        accuracy: float,
        n_permutations: int = 1000
    ) -> Dict:
        """
        Compare achieved accuracy to chance level.

        Args:
            y_test: Test labels
            accuracy: Achieved accuracy
            n_permutations: Number of permutations for chance estimation

        Returns:
            Dictionary with chance level statistics and significance
        """
        return compare_to_chance(accuracy, y_test, n_permutations)

    def save_results(
        self,
        results: Union[Dict, pd.DataFrame],
        filename: str = "evaluation_results.csv",
        output_dir: Optional[Union[str, Path]] = None
    ) -> Path:
        """
        Save evaluation results to CSV.

        Args:
            results: Either evaluation results dict or predictions DataFrame
            filename: Output filename
            output_dir: Directory to save to (uses self.output_dir if not provided)

        Returns:
            Path to saved file
        """
        output_dir = Path(output_dir) if output_dir else self.output_dir
        if output_dir is None:
            output_dir = Path(".")

        output_dir.mkdir(parents=True, exist_ok=True)
        filepath = output_dir / filename

        if isinstance(results, pd.DataFrame):
            results.to_csv(filepath, index=False)
        else:
            # Convert results dict to DataFrame
            records = []
            for split_name, split_results in results.items():
                if 'metrics' in split_results:
                    record = {'split': split_name, **split_results['metrics']}
                    records.append(record)

            df = pd.DataFrame(records)
            df.to_csv(filepath, index=False)

        print(f"\nResults saved to: {filepath}")
        return filepath

    def create_report(
        self,
        results: Dict,
        model_name: str = "Model",
        include_timestamp: bool = True
    ) -> str:
        """
        Create a text report of evaluation results.

        Args:
            results: Evaluation results dictionary
            model_name: Name of the model
            include_timestamp: Whether to include timestamp

        Returns:
            Formatted report string
        """
        lines = []
        lines.append("=" * 60)
        lines.append(f"EVALUATION REPORT: {model_name}")
        if include_timestamp:
            lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("=" * 60)

        for split_name, split_results in results.items():
            lines.append(f"\n--- {split_name.upper()} SET ---")
            lines.append(f"Samples: {split_results['n_samples']}")

            metrics = split_results['metrics']
            lines.append("\nMetrics:")
            for metric_name, value in metrics.items():
                if isinstance(value, float):
                    lines.append(f"  {metric_name}: {value:.4f}")

            lines.append("\nConfusion Matrix:")
            conf_matrix = split_results['confusion_matrix']
            lines.append(f"  {conf_matrix}")

        lines.append("\n" + "=" * 60)
        return "\n".join(lines)


def quick_evaluate(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_names: Optional[List[str]] = None
) -> Dict:
    """
    Quick evaluation helper for single test set.

    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        class_names: Class names

    Returns:
        Evaluation results dictionary
    """
    evaluator = ModelEvaluator(model, class_names)
    results = evaluator.evaluate({'test': (X_test, y_test)})
    return results['test']
