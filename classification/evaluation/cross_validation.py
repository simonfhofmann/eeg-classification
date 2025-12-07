# ----------------------------------------------------------------------
# cross_validation.py
#
# Cross-validation strategies for EEG classification.
# ----------------------------------------------------------------------

import numpy as np
from typing import Dict, List, Optional, Tuple, Generator, Any
from sklearn.model_selection import (
    StratifiedKFold,
    LeaveOneGroupOut,
    cross_val_score,
    cross_val_predict,
)

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from config import N_FOLDS, RANDOM_STATE


def stratified_kfold(
    X: np.ndarray,
    y: np.ndarray,
    n_folds: int = N_FOLDS,
    random_state: int = RANDOM_STATE
) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    """
    Generate stratified k-fold cross-validation splits.

    Stratification ensures each fold has approximately the same
    class distribution as the full dataset.

    Args:
        X: Features
        y: Labels
        n_folds: Number of folds
        random_state: Random seed

    Yields:
        (train_indices, test_indices) for each fold
    """
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    for train_idx, test_idx in skf.split(X, y):
        yield train_idx, test_idx


def leave_one_subject_out(
    X: np.ndarray,
    y: np.ndarray,
    subject_ids: np.ndarray
) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    """
    Generate leave-one-subject-out cross-validation splits.

    This is the gold standard for EEG classification as it tests
    generalization to completely unseen subjects.

    Args:
        X: Features
        y: Labels
        subject_ids: Subject identifier for each sample

    Yields:
        (train_indices, test_indices) for each subject
    """
    logo = LeaveOneGroupOut()

    for train_idx, test_idx in logo.split(X, y, groups=subject_ids):
        yield train_idx, test_idx


def cross_validate(
    model,
    X: np.ndarray,
    y: np.ndarray,
    cv_strategy: str = "stratified_kfold",
    n_folds: int = N_FOLDS,
    subject_ids: Optional[np.ndarray] = None,
    return_predictions: bool = False,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Perform cross-validation with a classifier.

    Args:
        model: Classifier instance (must have fit/predict methods)
        X: Features, shape (n_samples, n_features) or (n_samples, n_channels, n_times)
        y: Labels, shape (n_samples,)
        cv_strategy: 'stratified_kfold' or 'leave_one_subject_out'
        n_folds: Number of folds (for stratified_kfold)
        subject_ids: Subject IDs (required for leave_one_subject_out)
        return_predictions: Whether to return predictions for each sample
        verbose: Whether to print progress

    Returns:
        Dictionary containing:
            - 'scores': List of scores for each fold
            - 'mean_score': Mean accuracy
            - 'std_score': Standard deviation
            - 'predictions': (optional) Predictions for each sample
            - 'fold_details': Per-fold metrics
    """
    from .metrics import compute_metrics

    # Flatten if 3D for non-DL models
    X_flat = X.reshape(X.shape[0], -1) if X.ndim == 3 else X

    # Select CV strategy
    if cv_strategy == "stratified_kfold":
        cv_splits = list(stratified_kfold(X_flat, y, n_folds))
    elif cv_strategy == "leave_one_subject_out":
        if subject_ids is None:
            raise ValueError("subject_ids required for leave_one_subject_out")
        cv_splits = list(leave_one_subject_out(X_flat, y, subject_ids))
    else:
        raise ValueError(f"Unknown cv_strategy: {cv_strategy}")

    scores = []
    fold_details = []
    all_predictions = np.zeros_like(y)
    all_probas = None

    for fold_idx, (train_idx, test_idx) in enumerate(cv_splits):
        if verbose:
            print(f"Fold {fold_idx + 1}/{len(cv_splits)}...", end=" ")

        # Split data
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Clone model for this fold (create fresh instance)
        fold_model = model.__class__(**model.get_params())

        # Train
        fold_model.fit(X_train, y_train)

        # Predict
        y_pred = fold_model.predict(X_test)
        all_predictions[test_idx] = y_pred

        # Get probabilities if available
        try:
            y_proba = fold_model.predict_proba(X_test)
            if all_probas is None:
                all_probas = np.zeros((len(y), y_proba.shape[1]))
            all_probas[test_idx] = y_proba
        except (NotImplementedError, AttributeError):
            y_proba = None

        # Compute metrics for this fold
        fold_metrics = compute_metrics(y_test, y_pred, y_proba)
        scores.append(fold_metrics['accuracy'])
        fold_details.append(fold_metrics)

        if verbose:
            print(f"Accuracy: {fold_metrics['accuracy']:.4f}")

    # Aggregate results
    results = {
        'scores': scores,
        'mean_score': np.mean(scores),
        'std_score': np.std(scores),
        'fold_details': fold_details,
        'cv_strategy': cv_strategy,
        'n_folds': len(cv_splits),
    }

    if return_predictions:
        results['predictions'] = all_predictions
        if all_probas is not None:
            results['probabilities'] = all_probas

    # Compute overall metrics from all predictions
    results['overall_metrics'] = compute_metrics(
        y, all_predictions, all_probas
    )

    if verbose:
        print(f"\nOverall: {results['mean_score']:.4f} ± {results['std_score']:.4f}")

    return results


def nested_cross_validation(
    model_class,
    param_grid: Dict,
    X: np.ndarray,
    y: np.ndarray,
    outer_cv: int = 5,
    inner_cv: int = 3,
    scoring: str = "balanced_accuracy",
    subject_ids: Optional[np.ndarray] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Perform nested cross-validation with hyperparameter tuning.

    Outer loop: Evaluate generalization performance
    Inner loop: Tune hyperparameters

    Args:
        model_class: Classifier class (not instance)
        param_grid: Dictionary of parameters to search
        X: Features
        y: Labels
        outer_cv: Number of outer folds
        inner_cv: Number of inner folds
        scoring: Scoring metric for inner CV
        subject_ids: Subject IDs for LOSO outer CV
        verbose: Print progress

    Returns:
        Dictionary with nested CV results
    """
    from sklearn.model_selection import GridSearchCV

    X_flat = X.reshape(X.shape[0], -1) if X.ndim == 3 else X

    # Outer CV
    if subject_ids is not None:
        outer_splits = list(leave_one_subject_out(X_flat, y, subject_ids))
    else:
        outer_splits = list(stratified_kfold(X_flat, y, outer_cv))

    outer_scores = []
    best_params_per_fold = []

    for fold_idx, (train_idx, test_idx) in enumerate(outer_splits):
        if verbose:
            print(f"Outer fold {fold_idx + 1}/{len(outer_splits)}")

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Inner CV for hyperparameter tuning
        base_model = model_class()

        inner_cv_obj = StratifiedKFold(n_splits=inner_cv, shuffle=True, random_state=RANDOM_STATE)

        grid_search = GridSearchCV(
            base_model._model,
            param_grid,
            cv=inner_cv_obj,
            scoring=scoring,
            n_jobs=-1
        )

        X_train_flat = X_train.reshape(X_train.shape[0], -1) if X_train.ndim == 3 else X_train
        X_test_flat = X_test.reshape(X_test.shape[0], -1) if X_test.ndim == 3 else X_test

        grid_search.fit(X_train_flat, y_train)

        # Evaluate best model on outer test set
        best_score = grid_search.score(X_test_flat, y_test)
        outer_scores.append(best_score)
        best_params_per_fold.append(grid_search.best_params_)

        if verbose:
            print(f"  Best params: {grid_search.best_params_}")
            print(f"  Test score: {best_score:.4f}")

    return {
        'outer_scores': outer_scores,
        'mean_score': np.mean(outer_scores),
        'std_score': np.std(outer_scores),
        'best_params_per_fold': best_params_per_fold,
    }
