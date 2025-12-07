# ----------------------------------------------------------------------
# metrics.py
#
# Evaluation metrics for EEG classification.
# ----------------------------------------------------------------------

import numpy as np
from typing import Dict, List, Optional, Union
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
    classification_report as sklearn_report,
    cohen_kappa_score,
)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    metrics: Optional[List[str]] = None
) -> Dict[str, float]:
    """
    Compute multiple classification metrics.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities (optional, needed for AUC)
        metrics: List of metrics to compute. Options:
            'accuracy', 'balanced_accuracy', 'f1', 'precision',
            'recall', 'auc', 'kappa'
            If None, computes all available metrics.

    Returns:
        Dictionary of metric names to values
    """
    if metrics is None:
        metrics = ['accuracy', 'balanced_accuracy', 'f1',
                   'precision', 'recall', 'kappa']
        if y_proba is not None:
            metrics.append('auc')

    results = {}

    if 'accuracy' in metrics:
        results['accuracy'] = accuracy_score(y_true, y_pred)

    if 'balanced_accuracy' in metrics:
        results['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)

    if 'f1' in metrics:
        # Use weighted average for multiclass
        avg = 'binary' if len(np.unique(y_true)) == 2 else 'weighted'
        results['f1'] = f1_score(y_true, y_pred, average=avg, zero_division=0)

    if 'precision' in metrics:
        avg = 'binary' if len(np.unique(y_true)) == 2 else 'weighted'
        results['precision'] = precision_score(y_true, y_pred, average=avg, zero_division=0)

    if 'recall' in metrics:
        avg = 'binary' if len(np.unique(y_true)) == 2 else 'weighted'
        results['recall'] = recall_score(y_true, y_pred, average=avg, zero_division=0)

    if 'kappa' in metrics:
        results['kappa'] = cohen_kappa_score(y_true, y_pred)

    if 'auc' in metrics and y_proba is not None:
        try:
            if y_proba.ndim == 1 or y_proba.shape[1] == 2:
                # Binary classification
                proba = y_proba[:, 1] if y_proba.ndim == 2 else y_proba
                results['auc'] = roc_auc_score(y_true, proba)
            else:
                # Multiclass
                results['auc'] = roc_auc_score(y_true, y_proba, multi_class='ovr')
        except ValueError:
            results['auc'] = np.nan

    return results


def compute_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    normalize: Optional[str] = None,
    labels: Optional[List] = None
) -> np.ndarray:
    """
    Compute confusion matrix.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        normalize: Normalization mode ('true', 'pred', 'all', or None)
        labels: List of label values to include

    Returns:
        Confusion matrix array
    """
    return confusion_matrix(y_true, y_pred, labels=labels, normalize=normalize)


def classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: Optional[List[str]] = None,
    output_dict: bool = False
) -> Union[str, Dict]:
    """
    Generate a classification report.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        target_names: Optional class names
        output_dict: If True, return as dictionary

    Returns:
        Classification report as string or dictionary
    """
    return sklearn_report(
        y_true, y_pred,
        target_names=target_names,
        output_dict=output_dict,
        zero_division=0
    )


def sensitivity_specificity(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    pos_label: int = 1
) -> Dict[str, float]:
    """
    Compute sensitivity (recall) and specificity for binary classification.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        pos_label: The positive class label

    Returns:
        Dictionary with 'sensitivity' and 'specificity'
    """
    cm = confusion_matrix(y_true, y_pred)

    if cm.shape != (2, 2):
        raise ValueError("sensitivity_specificity only works for binary classification")

    tn, fp, fn, tp = cm.ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    return {
        'sensitivity': sensitivity,
        'specificity': specificity
    }


def compute_chance_level(y: np.ndarray, n_permutations: int = 1000) -> Dict[str, float]:
    """
    Compute chance level accuracy through permutation.

    Args:
        y: True labels
        n_permutations: Number of permutations

    Returns:
        Dictionary with 'mean', 'std', and '95_ci' of chance accuracy
    """
    accuracies = []

    for _ in range(n_permutations):
        y_perm = np.random.permutation(y)
        acc = accuracy_score(y, y_perm)
        accuracies.append(acc)

    accuracies = np.array(accuracies)

    return {
        'mean': np.mean(accuracies),
        'std': np.std(accuracies),
        '95_ci': (np.percentile(accuracies, 2.5), np.percentile(accuracies, 97.5))
    }


def compare_to_chance(
    accuracy: float,
    y: np.ndarray,
    n_permutations: int = 1000,
    alpha: float = 0.05
) -> Dict[str, Union[float, bool]]:
    """
    Statistical test comparing accuracy to chance level.

    Args:
        accuracy: Achieved accuracy
        y: True labels (for computing chance distribution)
        n_permutations: Number of permutations
        alpha: Significance level

    Returns:
        Dictionary with p-value and whether result is significant
    """
    chance = compute_chance_level(y, n_permutations)

    # Compute p-value (proportion of permutations >= achieved accuracy)
    n_greater = np.sum(np.array([
        accuracy_score(y, np.random.permutation(y))
        for _ in range(n_permutations)
    ]) >= accuracy)

    p_value = (n_greater + 1) / (n_permutations + 1)

    return {
        'accuracy': accuracy,
        'chance_mean': chance['mean'],
        'chance_std': chance['std'],
        'p_value': p_value,
        'significant': p_value < alpha
    }
