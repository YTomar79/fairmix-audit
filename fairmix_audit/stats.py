from __future__ import annotations

import numpy as np
import pandas as pd
import hashlib
from scipy import stats
from statsmodels.stats.multitest import multipletests


FAIRNESS_METRIC_SUFFIXES = (
    "demographic_parity_diff",
    "equal_opportunity_diff",
    "equalized_odds_diff",
    "calibration_brier_diff",
    "predictive_parity_diff",
)

GROUP_AWARE_METHODS = (
    "reweighing",
    "protected_group_mixing",
    "threshold_optimizer_eo",
    "exponentiated_gradient_dp",
    "exponentiated_gradient_eo",
)

AGNOSTIC_REFERENCE_METHODS = ("erm", "feature_mixing")

ANALYSIS_METRICS = (
    "accuracy",
    "balanced_accuracy",
    "roc_auc",
    "brier",
    "rac1p_demographic_parity_diff",
    "sex_demographic_parity_diff",
    "rac1p_x_sex_demographic_parity_diff",
    "rac1p_equalized_odds_diff",
    "sex_equalized_odds_diff",
    "rac1p_x_sex_equalized_odds_diff",
    "rac1p_worst_group_accuracy",
    "sex_worst_group_accuracy",
    "rac1p_x_sex_worst_group_accuracy",
)


def paired_bootstrap_delta(
    baseline_values: np.ndarray,
    method_values: np.ndarray,
    *,
    iterations: int,
    confidence_level: float,
    seed: int,
) -> tuple[float, float, float]:
    baseline_values = np.asarray(baseline_values, dtype=float)
    method_values = np.asarray(method_values, dtype=float)
    mask = ~(np.isnan(baseline_values) | np.isnan(method_values))
    baseline_values = baseline_values[mask]
    method_values = method_values[mask]
    if len(baseline_values) == 0:
        return float("nan"), float("nan"), float("nan")
    deltas = method_values - baseline_values
    observed = float(np.mean(deltas))
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(iterations):
        idx = rng.integers(0, len(deltas), size=len(deltas))
        samples.append(float(np.mean(deltas[idx])))
    alpha = 1.0 - confidence_level
    low, high = np.quantile(samples, [alpha / 2.0, 1.0 - alpha / 2.0])
    return observed, float(low), float(high)


def cluster_bootstrap_delta(
    baseline_values: np.ndarray,
    method_values: np.ndarray,
    clusters: np.ndarray,
    *,
    iterations: int,
    confidence_level: float,
    seed: int,
) -> tuple[float, float, float, int]:
    baseline_values = np.asarray(baseline_values, dtype=float)
    method_values = np.asarray(method_values, dtype=float)
    clusters = np.asarray(clusters).astype(str)
    mask = ~(np.isnan(baseline_values) | np.isnan(method_values))
    baseline_values = baseline_values[mask]
    method_values = method_values[mask]
    clusters = clusters[mask]
    if len(baseline_values) == 0:
        return float("nan"), float("nan"), float("nan"), 0

    deltas = method_values - baseline_values
    observed = float(np.mean(deltas))
    unique_clusters = np.unique(clusters)
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(iterations):
        sampled_clusters = rng.choice(unique_clusters, size=len(unique_clusters), replace=True)
        sampled = np.concatenate([np.flatnonzero(clusters == cluster) for cluster in sampled_clusters])
        samples.append(float(np.mean(deltas[sampled])))
    alpha = 1.0 - confidence_level
    low, high = np.quantile(samples, [alpha / 2.0, 1.0 - alpha / 2.0])
    return observed, float(low), float(high), int(len(unique_clusters))


def paired_ttest(baseline_values: np.ndarray, method_values: np.ndarray) -> tuple[float, float]:
    baseline_values = np.asarray(baseline_values, dtype=float)
    method_values = np.asarray(method_values, dtype=float)
    mask = ~(np.isnan(baseline_values) | np.isnan(method_values))
    if mask.sum() < 2:
        return float("nan"), float("nan")
    deltas = method_values[mask] - baseline_values[mask]
    if float(np.std(deltas)) == 0.0:
        mean_delta = float(np.mean(deltas))
        if mean_delta == 0.0:
            return 0.0, 1.0
        return float("inf") if mean_delta > 0 else float("-inf"), 0.0
    statistic, p_value = stats.ttest_rel(method_values[mask], baseline_values[mask])
    return float(statistic), float(p_value)


def summarize_metric_deltas(metrics: pd.DataFrame, config: dict) -> pd.DataFrame:
    id_cols = [
        "task",
        "split_name",
        "split_mode",
        "seed",
        "missingness_name",
        "missingness_mechanism",
        "missingness_scope",
    ]
    cluster_col = config.get("statistics", {}).get("cluster_column", "cluster_id")
    non_metric_cols = {
        *id_cols,
        "method",
        "fit_seconds",
        "run_id",
        "cluster_id",
        "train_sensitive_observed_fraction",
        "prediction_sensitive_observed_fraction",
    }
    metric_cols = [
        column
        for column in metrics.columns
        if column not in non_metric_cols
        and pd.api.types.is_numeric_dtype(metrics[column])
    ]
    baseline = metrics[metrics["method"] == "erm"].set_index(id_cols)
    rows = []
    for method in sorted(set(metrics["method"]) - {"erm"}):
        method_df = metrics[metrics["method"] == method].set_index(id_cols)
        shared = baseline.index.intersection(method_df.index)
        if shared.empty:
            continue
        for metric in metric_cols:
            base_values = baseline.loc[shared, metric].to_numpy()
            method_values = method_df.loc[shared, metric].to_numpy()
            delta, low, high = paired_bootstrap_delta(
                base_values,
                method_values,
                iterations=int(config["statistics"].get("bootstrap_iterations", 1000)),
                confidence_level=float(config["statistics"].get("confidence_level", 0.95)),
                seed=_stable_seed(method, metric),
            )
            t_stat, p_value = paired_ttest(base_values, method_values)
            if cluster_col in baseline.columns:
                cluster_delta, cluster_low, cluster_high, n_clusters = cluster_bootstrap_delta(
                    base_values,
                    method_values,
                    baseline.loc[shared, cluster_col].to_numpy(),
                    iterations=int(config["statistics"].get("bootstrap_iterations", 1000)),
                    confidence_level=float(config["statistics"].get("confidence_level", 0.95)),
                    seed=_stable_seed("cluster", method, metric),
                )
            else:
                cluster_delta, cluster_low, cluster_high, n_clusters = (float("nan"), float("nan"), float("nan"), 0)
            rows.append(
                {
                    "method": method,
                    "metric": metric,
                    "n_pairs": int(len(shared)),
                    "delta_vs_erm": delta,
                    "ci_low": low,
                    "ci_high": high,
                    "cluster_delta_vs_erm": cluster_delta,
                    "cluster_ci_low": cluster_low,
                    "cluster_ci_high": cluster_high,
                    "n_clusters": n_clusters,
                    "paired_t": t_stat,
                    "raw_p": p_value,
                }
            )
    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary
    valid_p = summary["raw_p"].notna()
    summary["holm_p"] = np.nan
    if valid_p.any():
        summary.loc[valid_p, "holm_p"] = multipletests(
            summary.loc[valid_p, "raw_p"], method="holm"
        )[1]
    return summary


def _stable_seed(*parts: object) -> int:
    text = "::".join(str(part) for part in parts)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:12], 16) % (2**31 - 1)


def audit_flags(metrics: pd.DataFrame, config: dict) -> pd.DataFrame:
    audit_cfg = config.get("audit", {})
    acc_tol = float(audit_cfg.get("accuracy_improvement_tolerance", 0.002))
    fair_tol = float(audit_cfg.get("fairness_worsening_tolerance", 0.01))
    worst_tol = float(audit_cfg.get("worst_group_accuracy_drop_tolerance", 0.005))
    keys = ["task", "split_name", "split_mode", "seed", "missingness_name"]
    baseline = metrics[metrics["method"] == "erm"].set_index(keys)
    rows = []
    for _, row in metrics[metrics["method"] != "erm"].iterrows():
        key = tuple(row[col] for col in keys)
        if key not in baseline.index:
            continue
        base = baseline.loc[key]
        accuracy_delta = row["accuracy"] - base["accuracy"]
        worsened = []
        for metric in metrics.columns:
            if metric.endswith(FAIRNESS_METRIC_SUFFIXES):
                delta = row[metric] - base[metric]
                if pd.notna(delta) and delta > fair_tol:
                    worsened.append(f"{metric}:{delta:.4f}")
            if metric.endswith("worst_group_accuracy"):
                delta = row[metric] - base[metric]
                if pd.notna(delta) and delta < -worst_tol:
                    worsened.append(f"{metric}:{delta:.4f}")
        rows.append(
            {
                **{col: row[col] for col in keys},
                "method": row["method"],
                "missingness_mechanism": row.get("missingness_mechanism", ""),
                "missingness_scope": row.get("missingness_scope", ""),
                "accuracy_delta": float(accuracy_delta),
                "unsafe_to_deploy": bool(accuracy_delta >= acc_tol and worsened),
                "worsened_metrics": ";".join(worsened),
            }
        )
    return pd.DataFrame(rows)


def summarize_missingness_sensitivity(metrics: pd.DataFrame, config: dict | None = None) -> pd.DataFrame:
    """Summarize metric curves as protected-attribute availability changes."""
    if metrics.empty:
        return pd.DataFrame()
    frame = _with_availability(metrics)
    metric_cols = [metric for metric in ANALYSIS_METRICS if metric in frame.columns]
    if not metric_cols:
        return pd.DataFrame()
    for column in ("train_sensitive_observed_fraction", "prediction_sensitive_observed_fraction"):
        if column not in frame.columns:
            frame[column] = np.nan
    group_cols = [
        "task",
        "split_mode",
        "method",
        "missingness_name",
        "missingness_mechanism",
        "missingness_scope",
        "availability_bucket",
    ]
    present_group_cols = [column for column in group_cols if column in frame.columns]
    summary = (
        frame.groupby(present_group_cols, dropna=False)
        .agg(
            n_runs=("method", "size"),
            train_observed_fraction=("train_sensitive_observed_fraction", "mean"),
            prediction_observed_fraction=("prediction_sensitive_observed_fraction", "mean"),
            **{metric: (metric, "mean") for metric in metric_cols},
        )
        .reset_index()
        .sort_values(["task", "split_mode", "method", "availability_bucket"])
    )
    return summary


def oracle_vs_missing_comparison(metrics: pd.DataFrame, config: dict | None = None) -> pd.DataFrame:
    """Compare group-aware methods under missing protected labels to oracle/full labels and agnostic baselines."""
    if metrics.empty or "missingness_name" not in metrics.columns:
        return pd.DataFrame()
    frame = _with_availability(metrics)
    metric_cols = [metric for metric in ANALYSIS_METRICS if metric in frame.columns]
    if not metric_cols:
        return pd.DataFrame()

    id_cols = ["task", "split_name", "split_mode", "seed"]
    full = frame[frame["missingness_name"] == "full"].set_index([*id_cols, "method"])
    if full.empty:
        return pd.DataFrame()
    all_rows = frame.set_index([*id_cols, "missingness_name", "method"])

    present_group_methods = sorted(set(frame["method"]).intersection(GROUP_AWARE_METHODS))
    rows: list[dict] = []
    missing_frame = frame[
        (frame["missingness_name"] != "full")
        & (frame["method"].isin(present_group_methods))
    ]
    for _, row in missing_frame.iterrows():
        id_key = tuple(row[column] for column in id_cols)
        oracle_key = (*id_key, row["method"])
        if oracle_key not in full.index:
            continue
        oracle = full.loc[oracle_key]
        for metric in metric_cols:
            missing_value = row.get(metric)
            oracle_value = oracle.get(metric)
            output = {
                **{column: row[column] for column in id_cols},
                "missingness_name": row["missingness_name"],
                "missingness_mechanism": row.get("missingness_mechanism", ""),
                "missingness_scope": row.get("missingness_scope", ""),
                "availability_bucket": row.get("availability_bucket", np.nan),
                "method": row["method"],
                "metric": metric,
                "missing_value": missing_value,
                "oracle_value": oracle_value,
                "delta_missing_vs_oracle": _safe_delta(missing_value, oracle_value),
            }
            for ref_method in AGNOSTIC_REFERENCE_METHODS:
                ref_key = (*id_key, row["missingness_name"], ref_method)
                ref_value = all_rows.loc[ref_key].get(metric) if ref_key in all_rows.index else np.nan
                output[f"{ref_method}_value"] = ref_value
                output[f"delta_missing_vs_{ref_method}"] = _safe_delta(missing_value, ref_value)
            rows.append(output)
    return pd.DataFrame(rows)


def fairness_conclusion_flips(metrics: pd.DataFrame, config: dict | None = None) -> pd.DataFrame:
    """Measure whether the method an auditor would choose changes when protected labels are missing."""
    if metrics.empty or "missingness_name" not in metrics.columns:
        return pd.DataFrame()
    frame = _with_availability(metrics)
    if "method" not in frame.columns:
        return pd.DataFrame()
    tolerance = float(
        (config or {}).get("audit", {}).get(
            "conclusion_accuracy_drop_tolerance",
            (config or {}).get("audit", {}).get("worst_group_accuracy_drop_tolerance", 0.01),
        )
    )
    criteria = _available_conclusion_criteria(frame)
    if not criteria:
        return pd.DataFrame()

    id_cols = ["task", "split_name", "split_mode", "seed"]
    rows: list[dict] = []
    grouped = {key: group.copy() for key, group in frame.groupby(id_cols, dropna=False)}
    for key, group in grouped.items():
        full_rows = group[group["missingness_name"] == "full"]
        if full_rows.empty:
            continue
        missing_names = sorted(name for name in group["missingness_name"].unique() if name != "full")
        for criterion_name, criterion in criteria.items():
            full_choice = _select_method_for_criterion(full_rows, criterion, tolerance=tolerance)
            if full_choice is None:
                continue
            for missingness_name in missing_names:
                missing_rows = group[group["missingness_name"] == missingness_name]
                if missing_rows.empty:
                    continue
                missing_choice = _select_method_for_criterion(missing_rows, criterion, tolerance=tolerance)
                if missing_choice is None:
                    continue
                representative = missing_rows.iloc[0]
                rows.append(
                    {
                        **dict(zip(id_cols, key, strict=True)),
                        "missingness_name": missingness_name,
                        "missingness_mechanism": representative.get("missingness_mechanism", ""),
                        "missingness_scope": representative.get("missingness_scope", ""),
                        "availability_bucket": representative.get("availability_bucket", np.nan),
                        "criterion": criterion_name,
                        "oracle_best_method": full_choice["method"],
                        "missing_best_method": missing_choice["method"],
                        "method_changed": bool(full_choice["method"] != missing_choice["method"]),
                        "oracle_score": full_choice["score"],
                        "missing_score": missing_choice["score"],
                        "oracle_accuracy": full_choice.get("accuracy", np.nan),
                        "missing_accuracy": missing_choice.get("accuracy", np.nan),
                    }
                )
    return pd.DataFrame(rows)


def hidden_intersectional_regressions(metrics: pd.DataFrame, config: dict | None = None) -> pd.DataFrame:
    """Find cases where single-axis fairness improves while intersectional fairness worsens."""
    if metrics.empty or "method" not in metrics.columns:
        return pd.DataFrame()
    frame = _with_availability(metrics)
    gap_tol = float((config or {}).get("audit", {}).get("hidden_regression_tolerance", 0.01))
    worst_tol = float((config or {}).get("audit", {}).get("hidden_worst_group_drop_tolerance", 0.005))
    id_cols = ["task", "split_name", "split_mode", "seed", "missingness_name"]
    baseline = frame[frame["method"] == "erm"].set_index(id_cols)
    if baseline.empty:
        return pd.DataFrame()

    rows: list[dict] = []
    for _, row in frame[frame["method"] != "erm"].iterrows():
        key = tuple(row[column] for column in id_cols)
        if key not in baseline.index:
            continue
        base = baseline.loc[key]
        for axis in ("rac1p", "sex"):
            for metric_family in ("demographic_parity", "equalized_odds"):
                axis_metric = f"{axis}_{metric_family}_diff"
                intersection_metric = f"rac1p_x_sex_{metric_family}_diff"
                if axis_metric not in frame.columns or intersection_metric not in frame.columns:
                    continue
                axis_delta = _safe_delta(row.get(axis_metric), base.get(axis_metric))
                intersection_delta = _safe_delta(row.get(intersection_metric), base.get(intersection_metric))
                worst_metric = "rac1p_x_sex_worst_group_accuracy"
                worst_delta = (
                    _safe_delta(row.get(worst_metric), base.get(worst_metric))
                    if worst_metric in frame.columns
                    else np.nan
                )
                single_axis_improved = pd.notna(axis_delta) and axis_delta < -gap_tol
                intersection_gap_worsened = pd.notna(intersection_delta) and intersection_delta > gap_tol
                intersection_worst_group_worsened = pd.notna(worst_delta) and worst_delta < -worst_tol
                if single_axis_improved:
                    rows.append(
                        {
                            **{column: row[column] for column in id_cols},
                            "missingness_mechanism": row.get("missingness_mechanism", ""),
                            "missingness_scope": row.get("missingness_scope", ""),
                            "availability_bucket": row.get("availability_bucket", np.nan),
                            "method": row["method"],
                            "single_axis": axis.upper(),
                            "fairness_metric": metric_family,
                            "single_axis_delta": axis_delta,
                            "intersection_delta": intersection_delta,
                            "intersection_worst_group_accuracy_delta": worst_delta,
                            "intersection_gap_worsened": bool(intersection_gap_worsened),
                            "intersection_worst_group_worsened": bool(intersection_worst_group_worsened),
                            "hidden_intersectional_regression": bool(
                                intersection_gap_worsened or intersection_worst_group_worsened
                            ),
                        }
                    )
    return pd.DataFrame(rows)


def mcar_vs_mnar_ablation(metrics: pd.DataFrame, config: dict | None = None) -> pd.DataFrame:
    """Compare matched MCAR and MNAR protected-label availability regimes."""
    if metrics.empty or "missingness_mechanism" not in metrics.columns:
        return pd.DataFrame()
    frame = _with_availability(metrics)
    metric_cols = [metric for metric in ANALYSIS_METRICS if metric in frame.columns]
    if not metric_cols:
        return pd.DataFrame()
    id_cols = ["task", "split_name", "split_mode", "seed", "method", "availability_bucket"]
    mcar = frame[frame["missingness_mechanism"] == "mcar"].set_index(id_cols)
    mnar = frame[frame["missingness_mechanism"] == "mnar"].set_index(id_cols)
    shared = mcar.index.intersection(mnar.index)
    if shared.empty:
        return pd.DataFrame()

    rows: list[dict] = []
    for index in shared:
        mcar_row = mcar.loc[index]
        mnar_row = mnar.loc[index]
        if isinstance(mcar_row, pd.DataFrame):
            mcar_row = mcar_row.iloc[0]
        if isinstance(mnar_row, pd.DataFrame):
            mnar_row = mnar_row.iloc[0]
        base = dict(zip(id_cols, index, strict=True))
        for metric in metric_cols:
            mcar_value = mcar_row.get(metric)
            mnar_value = mnar_row.get(metric)
            rows.append(
                {
                    **base,
                    "metric": metric,
                    "mcar_value": mcar_value,
                    "mnar_value": mnar_value,
                    "delta_mnar_vs_mcar": _safe_delta(mnar_value, mcar_value),
                    "mcar_missingness_name": mcar_row.get("missingness_name", ""),
                    "mnar_missingness_name": mnar_row.get("missingness_name", ""),
                }
            )
    return pd.DataFrame(rows)


def _safe_delta(value: object, baseline: object) -> float:
    value = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    baseline = pd.to_numeric(pd.Series([baseline]), errors="coerce").iloc[0]
    if pd.isna(value) or pd.isna(baseline):
        return float("nan")
    return float(value - baseline)


def _with_availability(metrics: pd.DataFrame) -> pd.DataFrame:
    frame = metrics.copy()
    if "target_availability" in frame.columns:
        availability = pd.to_numeric(frame["target_availability"], errors="coerce")
    else:
        availability = pd.Series(np.nan, index=frame.index, dtype=float)
    parsed = frame.get("missingness_name", pd.Series("", index=frame.index)).map(_availability_from_name)
    observed = pd.to_numeric(frame.get("train_sensitive_observed_fraction", np.nan), errors="coerce")
    availability = availability.fillna(parsed).fillna(observed)
    frame["availability_bucket"] = availability.round(2)
    return frame


def _availability_from_name(name: object) -> float:
    text = str(name).lower()
    if text == "full" or "100" in text:
        return 1.0
    if "none" in text or "_0" in text:
        return 0.0
    for token in ("80", "60", "50", "40", "20", "10"):
        if token in text:
            return float(token) / 100.0
    return float("nan")


def _available_conclusion_criteria(frame: pd.DataFrame) -> dict[str, dict]:
    criteria: dict[str, dict] = {}
    eo_cols = [
        column
        for column in (
            "rac1p_equalized_odds_diff",
            "sex_equalized_odds_diff",
            "rac1p_x_sex_equalized_odds_diff",
        )
        if column in frame.columns
    ]
    dp_cols = [
        column
        for column in (
            "rac1p_demographic_parity_diff",
            "sex_demographic_parity_diff",
            "rac1p_x_sex_demographic_parity_diff",
        )
        if column in frame.columns
    ]
    worst_cols = [
        column
        for column in (
            "rac1p_worst_group_accuracy",
            "sex_worst_group_accuracy",
            "rac1p_x_sex_worst_group_accuracy",
        )
        if column in frame.columns
    ]
    if eo_cols and "accuracy" in frame.columns:
        criteria["accuracy_constrained_equalized_odds"] = {
            "columns": eo_cols,
            "direction": "min",
            "aggregate": "max",
            "accuracy_constrained": True,
        }
    if dp_cols:
        criteria["demographic_parity"] = {
            "columns": dp_cols,
            "direction": "min",
            "aggregate": "max",
            "accuracy_constrained": False,
        }
    if worst_cols:
        criteria["worst_group_accuracy"] = {
            "columns": worst_cols,
            "direction": "max",
            "aggregate": "min",
            "accuracy_constrained": False,
        }
    if "rac1p_x_sex_equalized_odds_diff" in frame.columns:
        criteria["intersectional_equalized_odds"] = {
            "columns": ["rac1p_x_sex_equalized_odds_diff"],
            "direction": "min",
            "aggregate": "max",
            "accuracy_constrained": False,
        }
    return criteria


def _select_method_for_criterion(rows: pd.DataFrame, criterion: dict, *, tolerance: float) -> dict | None:
    candidates = rows.copy()
    if criterion.get("accuracy_constrained") and "accuracy" in candidates.columns:
        erm_rows = candidates[candidates["method"] == "erm"]
        if not erm_rows.empty:
            accuracy_floor = float(erm_rows["accuracy"].max()) - tolerance
            constrained = candidates[candidates["accuracy"] >= accuracy_floor]
            if not constrained.empty:
                candidates = constrained

    columns = [column for column in criterion["columns"] if column in candidates.columns]
    if not columns:
        return None
    values = candidates[columns].apply(pd.to_numeric, errors="coerce")
    if criterion["aggregate"] == "max":
        scores = values.max(axis=1)
    elif criterion["aggregate"] == "min":
        scores = values.min(axis=1)
    else:
        scores = values.mean(axis=1)
    candidates = candidates.assign(_criterion_score=scores)
    candidates = candidates[candidates["_criterion_score"].notna()]
    if candidates.empty:
        return None

    ascending = criterion["direction"] == "min"
    sort_cols = ["_criterion_score"]
    sort_ascending = [ascending]
    if "accuracy" in candidates.columns:
        sort_cols.append("accuracy")
        sort_ascending.append(False)
    sort_cols.append("method")
    sort_ascending.append(True)
    selected = candidates.sort_values(sort_cols, ascending=sort_ascending).iloc[0]
    return {
        "method": selected["method"],
        "score": float(selected["_criterion_score"]),
        "accuracy": float(selected.get("accuracy", np.nan)),
    }
