import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

def print_evaluation(y_true, y_pred, fold_index=None):
    """Prints standard classification metrics."""
    prefix = f"[Fold {fold_index}] " if fold_index is not None else "[Overall] "
    
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    
    print(f"{prefix}Accuracy: {acc:.4f} | Macro F1: {f1:.4f}")
    return acc, f1

def plot_confusion_matrix(y_true, y_pred, classes=['Excellent', 'Good', 'Bad']):
    """Plots confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()

def plot_feature_importance(importances, feature_names):
    """Plots Random Forest feature importances."""
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(10, 5))
    plt.title("Feature Importance (Random Forest)")
    plt.bar(range(len(importances)), importances[indices], align="center")
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45)
    plt.tight_layout()
    plt.show()