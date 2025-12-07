# ----------------------------------------------------------------------
# visualization.py
#
# Plotting utilities for EEG classification results.
# ----------------------------------------------------------------------

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
    normalize: bool = True,
    title: str = "Confusion Matrix",
    cmap: str = "Blues",
    figsize: Tuple[int, int] = (8, 6),
    save_path: Optional[str] = None
):
    """
    Plot a confusion matrix.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Names for each class
        normalize: Whether to normalize by true labels
        title: Plot title
        cmap: Colormap
        figsize: Figure size
        save_path: Path to save figure (optional)

    Returns:
        matplotlib figure and axes
    """
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    if class_names is None:
        class_names = [str(i) for i in range(cm.shape[0])]

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        title=title,
        ylabel='True label',
        xlabel='Predicted label'
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add text annotations
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    fig.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig, ax


def plot_roc_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    class_names: Optional[List[str]] = None,
    title: str = "ROC Curve",
    figsize: Tuple[int, int] = (8, 6),
    save_path: Optional[str] = None
):
    """
    Plot ROC curve(s).

    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        class_names: Names for each class
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure

    Returns:
        matplotlib figure and axes
    """
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc

    fig, ax = plt.subplots(figsize=figsize)

    n_classes = y_proba.shape[1] if y_proba.ndim > 1 else 2

    if n_classes == 2:
        # Binary classification
        proba = y_proba[:, 1] if y_proba.ndim > 1 else y_proba
        fpr, tpr, _ = roc_curve(y_true, proba)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    else:
        # Multiclass - one-vs-rest
        if class_names is None:
            class_names = [f'Class {i}' for i in range(n_classes)]

        for i in range(n_classes):
            y_binary = (y_true == i).astype(int)
            fpr, tpr, _ = roc_curve(y_binary, y_proba[:, i])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, lw=2,
                    label=f'{class_names[i]} (AUC = {roc_auc:.3f})')

    ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Chance')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title)
    ax.legend(loc="lower right")

    fig.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig, ax


def plot_learning_curve(
    train_history: Dict[str, List[float]],
    title: str = "Learning Curve",
    figsize: Tuple[int, int] = (10, 4),
    save_path: Optional[str] = None
):
    """
    Plot training history (loss and accuracy curves).

    Args:
        train_history: Dictionary with 'train_loss', 'val_loss',
                      'train_acc', 'val_acc' lists
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure

    Returns:
        matplotlib figure and axes
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    epochs = range(1, len(train_history.get('train_loss', [])) + 1)

    # Loss plot
    if 'train_loss' in train_history:
        axes[0].plot(epochs, train_history['train_loss'], 'b-', label='Train')
    if 'val_loss' in train_history:
        axes[0].plot(epochs, train_history['val_loss'], 'r-', label='Validation')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy plot
    if 'train_acc' in train_history:
        axes[1].plot(epochs, train_history['train_acc'], 'b-', label='Train')
    if 'val_acc' in train_history:
        axes[1].plot(epochs, train_history['val_acc'], 'r-', label='Validation')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(title)
    fig.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig, axes


def plot_feature_importance(
    importance: np.ndarray,
    feature_names: Optional[List[str]] = None,
    top_n: int = 20,
    title: str = "Feature Importance",
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None
):
    """
    Plot feature importance scores.

    Args:
        importance: Feature importance scores
        feature_names: Names for each feature
        top_n: Number of top features to show
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure

    Returns:
        matplotlib figure and axes
    """
    import matplotlib.pyplot as plt

    if feature_names is None:
        feature_names = [f'Feature {i}' for i in range(len(importance))]

    # Sort by importance
    indices = np.argsort(importance)[::-1][:top_n]
    top_importance = importance[indices]
    top_names = [feature_names[i] for i in indices]

    fig, ax = plt.subplots(figsize=figsize)

    y_pos = np.arange(len(top_importance))
    ax.barh(y_pos, top_importance, align='center')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_names)
    ax.invert_yaxis()
    ax.set_xlabel('Importance')
    ax.set_title(title)

    fig.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig, ax


def plot_cv_results(
    cv_results: Dict,
    title: str = "Cross-Validation Results",
    figsize: Tuple[int, int] = (8, 5),
    save_path: Optional[str] = None
):
    """
    Plot cross-validation results.

    Args:
        cv_results: Dictionary from cross_validate function
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure

    Returns:
        matplotlib figure and axes
    """
    import matplotlib.pyplot as plt

    scores = cv_results['scores']
    mean_score = cv_results['mean_score']
    std_score = cv_results['std_score']

    fig, ax = plt.subplots(figsize=figsize)

    folds = range(1, len(scores) + 1)
    ax.bar(folds, scores, color='steelblue', alpha=0.7)
    ax.axhline(y=mean_score, color='red', linestyle='--',
               label=f'Mean: {mean_score:.3f} ± {std_score:.3f}')

    ax.set_xlabel('Fold')
    ax.set_ylabel('Accuracy')
    ax.set_title(title)
    ax.set_xticks(folds)
    ax.legend()
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3, axis='y')

    fig.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig, ax


def plot_band_power_comparison(
    band_power_class0: Dict[str, np.ndarray],
    band_power_class1: Dict[str, np.ndarray],
    class_names: Tuple[str, str] = ("Class 0", "Class 1"),
    title: str = "Band Power Comparison",
    figsize: Tuple[int, int] = (12, 5),
    save_path: Optional[str] = None
):
    """
    Plot comparison of frequency band power between two classes.

    Args:
        band_power_class0: Band power for class 0 (dict of band: values)
        band_power_class1: Band power for class 1
        class_names: Names for the two classes
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure

    Returns:
        matplotlib figure and axes
    """
    import matplotlib.pyplot as plt

    bands = list(band_power_class0.keys())
    n_bands = len(bands)

    # Compute mean and std for each band
    means_0 = [np.mean(band_power_class0[b]) for b in bands]
    stds_0 = [np.std(band_power_class0[b]) for b in bands]
    means_1 = [np.mean(band_power_class1[b]) for b in bands]
    stds_1 = [np.std(band_power_class1[b]) for b in bands]

    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(n_bands)
    width = 0.35

    ax.bar(x - width/2, means_0, width, yerr=stds_0, label=class_names[0],
           capsize=3, alpha=0.8)
    ax.bar(x + width/2, means_1, width, yerr=stds_1, label=class_names[1],
           capsize=3, alpha=0.8)

    ax.set_xlabel('Frequency Band')
    ax.set_ylabel('Power')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(bands)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    fig.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig, ax
