from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, brier_score_loss, roc_auc_score


@dataclass(frozen=True)
class MetricBundle:
    metrics: dict[str, float]
    group_metrics: pd.DataFrame


def _safe_rate(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return float("nan")
    return float(numerator / denominator)


def _nan_gap(values: list[float]) -> float:
    valid = np.asarray([value for value in values if not np.isnan(value)], dtype=float)
    if len(valid) < 2:
        return float("nan")
    return float(np.max(valid) - np.min(valid))


def _nan_ratio(values: list[float]) -> float:
    valid = np.asarray([value for value in values if not np.isnan(value)], dtype=float)
    valid = valid[valid >= 0]
    if len(valid) < 2 or np.max(valid) == 0:
        return float("nan")
    return float(np.min(valid) / np.max(valid))


def _nan_max(values: list[float]) -> float:
    valid = np.asarray([value for value in values if not np.isnan(value)], dtype=float)
    if len(valid) == 0:
        return float("nan")
    return float(np.max(valid))


def _group_frame(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    group_values: pd.Series,
    attr_name: str,
    min_group_size: int,
) -> pd.DataFrame:
    rows = []
    for group in sorted(pd.Series(group_values).dropna().astype(str).unique()):
        mask = pd.Series(group_values).astype(str).to_numpy() == group
        n = int(mask.sum())
        if n < min_group_size:
            continue
        yt = y_true[mask]
        yp = y_pred[mask]
        prob = y_prob[mask]
        tp = int(((yt == 1) & (yp == 1)).sum())
        tn = int(((yt == 0) & (yp == 0)).sum())
        fp = int(((yt == 0) & (yp == 1)).sum())
        fn = int(((yt == 1) & (yp == 0)).sum())
        rows.append(
            {
                "attribute": attr_name,
                "group": group,
                "n": n,
                "positive_rate": float(np.mean(yp == 1)),
                "accuracy": float(accuracy_score(yt, yp)),
                "tpr": _safe_rate(tp, tp + fn),
                "fpr": _safe_rate(fp, fp + tn),
                "fnr": _safe_rate(fn, fn + tp),
                "ppv": _safe_rate(tp, tp + fp),
                "brier": float(brier_score_loss(yt, prob)) if len(np.unique(yt)) > 1 else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def evaluate_predictions(
    y_true,
    y_pred,
    y_prob,
    protected: pd.DataFrame,
    *,
    min_group_size: int,
) -> MetricBundle:
    y_true = np.asarray(y_true).astype(int).reshape(-1)
    y_pred = np.asarray(y_pred).astype(int).reshape(-1)
    if y_prob is None:
        y_prob = y_pred.astype(float)
    y_prob = np.asarray(y_prob).astype(float).reshape(-1)

    metrics: dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "positive_rate": float(np.mean(y_pred == 1)),
        "brier": float(brier_score_loss(y_true, y_prob)),
    }
    try:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        metrics["roc_auc"] = float("nan")

    group_frames = []
    protected = protected.copy()
    if {"RAC1P", "SEX"}.issubset(protected.columns):
        protected["RAC1P_X_SEX"] = protected[["RAC1P", "SEX"]].astype(str).agg("|".join, axis=1)

    for attr in protected.columns:
        group_df = _group_frame(y_true, y_pred, y_prob, protected[attr], attr, min_group_size)
        if group_df.empty:
            continue
        group_frames.append(group_df)
        prefix = attr.lower()
        metrics[f"{prefix}_demographic_parity_diff"] = _nan_gap(group_df["positive_rate"].tolist())
        metrics[f"{prefix}_demographic_parity_ratio"] = _nan_ratio(group_df["positive_rate"].tolist())
        metrics[f"{prefix}_equal_opportunity_diff"] = _nan_gap(group_df["tpr"].tolist())
        metrics[f"{prefix}_fpr_diff"] = _nan_gap(group_df["fpr"].tolist())
        metrics[f"{prefix}_equalized_odds_diff"] = _nan_max(
            [metrics[f"{prefix}_equal_opportunity_diff"], metrics[f"{prefix}_fpr_diff"]]
        )
        metrics[f"{prefix}_calibration_brier_diff"] = _nan_gap(group_df["brier"].tolist())
        metrics[f"{prefix}_predictive_parity_diff"] = _nan_gap(group_df["ppv"].tolist())
        metrics[f"{prefix}_worst_group_accuracy"] = float(group_df["accuracy"].min())
        metrics[f"{prefix}_worst_group_fnr"] = float(group_df["fnr"].max())
        metrics[f"{prefix}_worst_group_fpr"] = float(group_df["fpr"].max())

    all_group_metrics = pd.concat(group_frames, ignore_index=True) if group_frames else pd.DataFrame()
    return MetricBundle(metrics=metrics, group_metrics=all_group_metrics)
