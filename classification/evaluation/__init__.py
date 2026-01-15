# ----------------------------------------------------------------------
# Evaluation module
# ----------------------------------------------------------------------

from .metrics import (
    compute_metrics,
    compute_confusion_matrix,
    classification_report,
    sensitivity_specificity,
    compute_chance_level,
    compare_to_chance,
)
from .cross_validation import (
    cross_validate,
    leave_one_subject_out,
    stratified_kfold,
)
from .evaluator import ModelEvaluator, quick_evaluate

__all__ = [
    # Metrics
    "compute_metrics",
    "compute_confusion_matrix",
    "classification_report",
    "sensitivity_specificity",
    "compute_chance_level",
    "compare_to_chance",
    # Cross-validation
    "cross_validate",
    "leave_one_subject_out",
    "stratified_kfold",
    # Evaluator
    "ModelEvaluator",
    "quick_evaluate",
]
