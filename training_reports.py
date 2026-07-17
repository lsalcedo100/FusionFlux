"""Diagnostic plot generation for a completed training run.

These are best-effort reporting artifacts: ``training.train_models`` isolates
their failures so a matplotlib/seaborn backend error degrades to "reports
skipped" instead of discarding an otherwise-successful run. matplotlib and
seaborn are imported lazily so importing the training pipeline stays cheap.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def save_residual_plots(
    y_true: pd.Series,
    predictions: np.ndarray,
    output_path: Path,
    model_name: str,
) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    residuals = y_true - predictions
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].scatter(y_true, predictions, alpha=0.7, edgecolor="none")
    min_axis = min(float(y_true.min()), float(predictions.min()))
    max_axis = max(float(y_true.max()), float(predictions.max()))
    axes[0].plot([min_axis, max_axis], [min_axis, max_axis], linestyle="--", color="black")
    axes[0].set_title(f"Actual vs Predicted ({model_name})")
    axes[0].set_xlabel("Actual Neutron Yield")
    axes[0].set_ylabel("Predicted Neutron Yield")

    axes[1].scatter(predictions, residuals, alpha=0.7, edgecolor="none")
    axes[1].axhline(0.0, linestyle="--", color="black")
    axes[1].set_title(f"Residuals ({model_name})")
    axes[1].set_xlabel("Predicted Neutron Yield")
    axes[1].set_ylabel("Residual")

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_feature_importance_plot(
    importance_df: pd.DataFrame,
    output_path: Path,
    *,
    model_name: str,
    importance_method: str,
) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    top_features = importance_df.head(12).iloc[::-1]
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(top_features["feature"], top_features["importance"], color="#2f6f9f")
    title_model_name = model_name.replace("_", " ").title()
    if "permutation_importance" in importance_method:
        ax.set_title(f"{title_model_name} Permutation Importance")
    else:
        ax.set_title(f"{title_model_name} Feature Importance")
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
