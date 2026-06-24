from __future__ import annotations

import os
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from fairmix_audit.data import combine_columns, feature_columns
from fairmix_audit.memory import rss_mb, trim_process_memory
from fairmix_audit.missingness import MISSING_GROUP
from fairmix_audit.mixing import label_conditioned_context_mix


@dataclass
class PredictionResult:
    method: str
    y_pred: np.ndarray
    y_prob: np.ndarray | None
    fit_seconds: float
    metadata: dict[str, Any] = field(default_factory=dict)


def make_estimator(config: dict, *, seed: int | None = None):
    model_cfg = config.get("model", {})
    estimator_name = model_cfg.get("estimator", "logistic_regression")
    n_jobs = _bounded_n_jobs(model_cfg)
    random_state = _model_random_state(model_cfg, seed=seed)
    if estimator_name == "logistic_regression":
        solver = model_cfg.get("solver", "saga")
        estimator_kwargs: dict[str, Any] = {
            "max_iter": int(model_cfg.get("max_iter", 1000)),
            "class_weight": model_cfg.get("class_weight"),
            "solver": solver,
            "random_state": random_state,
        }
        if solver in {"liblinear", "sag", "saga"}:
            estimator_kwargs["n_jobs"] = n_jobs
        return LogisticRegression(
            **estimator_kwargs,
        )
    if estimator_name == "random_forest":
        estimator_kwargs: dict[str, Any] = {
            "n_estimators": int(model_cfg.get("n_estimators", 300)),
            "min_samples_leaf": int(model_cfg.get("min_samples_leaf", 10)),
            "class_weight": model_cfg.get("class_weight"),
            "n_jobs": n_jobs,
            "random_state": random_state,
        }
        for optional_key in ("max_depth", "min_samples_split", "max_features", "bootstrap"):
            if optional_key in model_cfg:
                estimator_kwargs[optional_key] = model_cfg[optional_key]
        return RandomForestClassifier(
            **estimator_kwargs,
        )
    raise ValueError(f"Unsupported estimator: {estimator_name}")


def _model_random_state(model_cfg: dict, *, seed: int | None) -> int:
    if _safe_bool(model_cfg.get("random_state_from_run_seed", False)):
        offset = _safe_int(model_cfg.get("random_state_seed_offset", 0), default=0)
        if seed is not None:
            return int(seed) + offset
    return _safe_int(model_cfg.get("random_state", 2027), default=2027)


def _safe_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _bounded_n_jobs(model_cfg: dict) -> int:
    requested = _safe_int(model_cfg.get("n_jobs", 1), default=1)
    available = os.cpu_count() or 1
    if requested < 1:
        requested = available
    return max(1, min(requested, available))


def _safe_int(value: object, *, default: int) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def make_preprocessor(train_frame: pd.DataFrame, config: dict) -> tuple[ColumnTransformer, list[str]]:
    columns = feature_columns(train_frame, config)
    model_cfg = config.get("model", {})
    numeric_config = set(model_cfg.get("numeric_features", []))
    numeric = [column for column in columns if column in numeric_config]
    categorical = [column for column in columns if column not in numeric]
    transformers = []
    if numeric:
        transformers.append(("numeric", StandardScaler(), numeric))
    if categorical:
        onehot_kwargs: dict[str, Any] = {
            "sparse_output": True,
            "dtype": np.float32,
        }
        onehot_max_categories = model_cfg.get("onehot_max_categories")
        if onehot_max_categories:
            onehot_kwargs["handle_unknown"] = "infrequent_if_exist"
            onehot_kwargs["max_categories"] = int(onehot_max_categories)
        else:
            onehot_kwargs["handle_unknown"] = "ignore"
        transformers.append(
            (
                "categorical",
                OneHotEncoder(**onehot_kwargs),
                categorical,
            )
        )
    if not transformers:
        raise ValueError("No model features remain after excluding protected/context columns.")
    return ColumnTransformer(transformers=transformers, sparse_threshold=1.0), columns


def _predict_probability(estimator, X) -> np.ndarray | None:
    if hasattr(estimator, "predict_proba"):
        proba = estimator.predict_proba(X)
        if proba.ndim == 2 and proba.shape[1] > 1:
            return proba[:, 1]
        return proba.reshape(-1)
    if hasattr(estimator, "decision_function"):
        score = estimator.decision_function(X)
        return 1.0 / (1.0 + np.exp(-score))
    return None


def _optimize_design_matrix(X):
    if sparse.issparse(X):
        return X.tocsr().astype(np.float32)
    return np.asarray(X, dtype=np.float32)


def _dense_float32(X):
    if sparse.issparse(X):
        return X.toarray().astype(np.float32, copy=False)
    return np.asarray(X, dtype=np.float32)


def _dense_size_mb(X) -> float:
    rows, cols = X.shape
    return float(rows * cols * np.dtype(np.float32).itemsize / (1024 * 1024))


def _check_dense_budget(X, config: dict, *, label: str) -> None:
    max_mb = config.get("fairlearn", {}).get("max_dense_matrix_mb")
    if not max_mb:
        return
    estimate_mb = _dense_size_mb(X)
    if estimate_mb > float(max_mb):
        raise MemoryError(
            f"{label} dense Fairlearn matrix would be {estimate_mb:.1f} MB, "
            f"above fairlearn.max_dense_matrix_mb={float(max_mb):.1f}. "
            "Lower data.max_train_rows_per_split, data.max_test_rows_per_split, "
            "or model.onehot_max_categories."
        )


def _rss_mb() -> float:
    return rss_mb()


def _check_runtime_memory_budget(label: str) -> None:
    rss_mb = _rss_mb()
    target_text = os.environ.get("FAIRMIX_TARGET_RSS_MB")
    if target_text:
        try:
            target_mb = float(target_text)
        except ValueError:
            target_mb = None
        if target_mb is not None and rss_mb > target_mb:
            print(
                f"[memory] target exceeded at {label}: rss_mb={rss_mb:.1f} "
                f"target_mb={target_mb:.1f}; continuing below the hard RSS guard",
                flush=True,
            )

    hard_text = os.environ.get("FAIRMIX_HARD_RSS_MB")
    if not hard_text:
        return
    try:
        hard_mb = float(hard_text)
    except ValueError:
        return
    if rss_mb > hard_mb:
        raise MemoryError(
            f"Memory hard limit exceeded at {label}: rss_mb={rss_mb:.1f}, "
            f"hard_mb={hard_mb:.1f}. Lower row caps before rerunning this worker."
        )


def _matrix_metadata(X, prefix: str) -> dict[str, Any]:
    if sparse.issparse(X):
        rows, cols = X.shape
        density = float(X.nnz / (rows * cols)) if rows and cols else 0.0
        return {
            f"{prefix}_shape": [int(rows), int(cols)],
            f"{prefix}_nnz": int(X.nnz),
            f"{prefix}_density": density,
            f"{prefix}_dtype": str(X.dtype),
            f"{prefix}_sparse": True,
        }
    rows, cols = X.shape
    return {
        f"{prefix}_shape": [int(rows), int(cols)],
        f"{prefix}_nnz": int(np.count_nonzero(X)),
        f"{prefix}_density": float(np.count_nonzero(X) / X.size) if X.size else 0.0,
        f"{prefix}_dtype": str(X.dtype),
        f"{prefix}_sparse": False,
    }


def _predict_with_optional_sensitive(estimator, X, sensitive_features: pd.Series | None, *, seed: int) -> np.ndarray:
    if _requires_sensitive_prediction(estimator):
        return np.asarray(estimator.predict(X, sensitive_features=sensitive_features, random_state=seed))
    try:
        return np.asarray(estimator.predict(X))
    except TypeError:
        return np.asarray(estimator.predict(X, random_state=seed))


def _requires_sensitive_prediction(estimator) -> bool:
    return estimator.__class__.__name__ == "ThresholdOptimizer"


def _align_sensitive_for_threshold_prediction(train_sensitive: pd.Series, test_sensitive: pd.Series) -> tuple[pd.Series, dict]:
    train_sensitive = pd.Series(train_sensitive).astype(str).reset_index(drop=True)
    test_sensitive = pd.Series(test_sensitive).astype(str).reset_index(drop=True)
    known = set(train_sensitive.unique())
    if not known:
        return test_sensitive, {"threshold_unseen_sensitive_groups": 0}
    fallback = MISSING_GROUP if MISSING_GROUP in known else train_sensitive.value_counts().idxmax()
    unseen = ~test_sensitive.isin(known)
    aligned = test_sensitive.mask(unseen, fallback)
    return aligned, {
        "threshold_unseen_sensitive_groups": int(unseen.sum()),
        "threshold_unseen_sensitive_fallback": fallback,
    }


def _coalesce_degenerate_sensitive_groups(sensitive: pd.Series, labels: np.ndarray) -> tuple[pd.Series, dict]:
    sensitive = pd.Series(sensitive).astype(str).reset_index(drop=True)
    labels = pd.Series(labels).astype(int).reset_index(drop=True)
    counts = pd.DataFrame({"sensitive": sensitive, "label": labels}).value_counts().unstack(fill_value=0)
    for label in [0, 1]:
        if label not in counts.columns:
            counts[label] = 0
    valid_groups = counts[(counts[0] > 0) & (counts[1] > 0)]
    if valid_groups.empty:
        return pd.Series([MISSING_GROUP] * len(sensitive)), {
            "threshold_degenerate_groups_coalesced": int(sensitive.nunique()),
            "threshold_degenerate_fallback": MISSING_GROUP,
        }

    fallback = str(valid_groups.sum(axis=1).idxmax())
    degenerate_groups = set(counts.index) - set(valid_groups.index)
    coalesced = sensitive.mask(sensitive.isin(degenerate_groups), fallback)
    return coalesced, {
        "threshold_degenerate_groups_coalesced": int(len(degenerate_groups)),
        "threshold_degenerate_fallback": fallback,
    }


def reweighing_weights(groups: pd.Series, labels: np.ndarray) -> np.ndarray:
    """Kamiran-Calders style sample weights for group-label independence."""
    groups = pd.Series(groups).astype(str).reset_index(drop=True)
    labels = pd.Series(labels).astype(int).reset_index(drop=True)
    n = float(len(labels))
    weights = np.ones(len(labels), dtype=float)
    known = groups != MISSING_GROUP
    if known.sum() == 0:
        return weights

    known_groups = groups.loc[known]
    known_labels = labels.loc[known]
    group_probs = known_groups.value_counts(normalize=True).to_dict()
    label_probs = known_labels.value_counts(normalize=True).to_dict()
    joint_probs = pd.DataFrame({"group": known_groups, "label": known_labels}).value_counts(normalize=True).to_dict()

    for i, (group, label) in enumerate(zip(groups, labels, strict=True)):
        if group == MISSING_GROUP:
            continue
        joint = joint_probs.get((group, label), 0.0)
        if joint > 0:
            weights[i] = (group_probs[group] * label_probs[label]) / joint
    return weights * (n / weights.sum())


def fit_predict_method(
    method: str,
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    config: dict,
    *,
    seed: int,
    sensitive_train: pd.Series | None = None,
    sensitive_test: pd.Series | None = None,
    missingness_metadata: dict[str, Any] | None = None,
) -> PredictionResult:
    data_cfg = config["data"]
    label_col = data_cfg.get("label_column", "label")
    protected_attrs = data_cfg.get("protected_attributes", [])
    oracle_attrs = data_cfg.get("oracle_sensitive_attributes", protected_attrs)
    context_attrs = data_cfg.get("context_attributes", ["__state", "__year"])

    preprocessor, columns = make_preprocessor(train_frame, config)
    X_train_raw = train_frame[columns]
    X_test_raw = test_frame[columns]
    y_train = train_frame[label_col].astype(int).to_numpy()
    y_test = test_frame[label_col].astype(int).to_numpy()
    if sensitive_train is None:
        sensitive_train = combine_columns(train_frame, oracle_attrs)
    else:
        sensitive_train = pd.Series(sensitive_train).reset_index(drop=True)
    if sensitive_test is None:
        sensitive_test = combine_columns(test_frame, oracle_attrs)
    else:
        sensitive_test = pd.Series(sensitive_test).reset_index(drop=True)

    start = perf_counter()
    X_train = _optimize_design_matrix(preprocessor.fit_transform(X_train_raw))
    X_test = _optimize_design_matrix(preprocessor.transform(X_test_raw))
    _check_runtime_memory_budget(f"{method} preprocess")
    estimator = make_estimator(config, seed=seed)
    metadata: dict[str, Any] = {
        "features": columns,
        "estimator": config.get("model", {}).get("estimator", "logistic_regression"),
        "estimator_random_state": getattr(estimator, "random_state", None),
        **_matrix_metadata(X_train, "train_matrix"),
        **_matrix_metadata(X_test, "test_matrix"),
        **(missingness_metadata or {}),
    }

    if method == "erm":
        estimator.fit(X_train, y_train)
        _check_runtime_memory_budget(f"{method} fit")
        fitted = estimator
        X_predict = X_test
        del X_train
    elif method == "feature_mixing":
        contexts = combine_columns(train_frame, context_attrs)
        X_aug, y_aug, report = label_conditioned_context_mix(
            X_train,
            y_train,
            contexts,
            mix_ratio=float(config["feature_mixing"].get("mix_ratio", 1.0)),
            beta=float(config["feature_mixing"].get("beta", 0.4)),
            min_contexts_per_label=int(config["feature_mixing"].get("min_contexts_per_label", 2)),
            seed=seed,
        )
        metadata["mixing"] = report.__dict__
        estimator.fit(X_aug, y_aug)
        _check_runtime_memory_budget(f"{method} fit")
        fitted = estimator
        X_predict = X_test
        del X_train, X_aug, y_aug
    elif method == "protected_group_mixing":
        X_aug, y_aug, report = label_conditioned_context_mix(
            X_train,
            y_train,
            sensitive_train,
            mix_ratio=float(config["feature_mixing"].get("mix_ratio", 1.0)),
            beta=float(config["feature_mixing"].get("beta", 0.4)),
            min_contexts_per_label=int(config["feature_mixing"].get("min_contexts_per_label", 2)),
            seed=seed,
        )
        metadata["mixing"] = report.__dict__ | {"uses_protected_attributes": True}
        estimator.fit(X_aug, y_aug)
        _check_runtime_memory_budget(f"{method} fit")
        fitted = estimator
        X_predict = X_test
        del X_train, X_aug, y_aug
    elif method == "reweighing":
        weights = reweighing_weights(sensitive_train, y_train)
        metadata["uses_protected_attributes"] = True
        estimator.fit(X_train, y_train, sample_weight=weights)
        _check_runtime_memory_budget(f"{method} fit")
        fitted = estimator
        X_predict = X_test
        del X_train, weights
    elif method.startswith("exponentiated_gradient"):
        # Fairlearn reductions validate dense arrays only, so keep this conversion
        # behind split-level row caps and float32 downcasting.
        _check_dense_budget(X_train, config, label=f"{method} train")
        _check_dense_budget(X_test, config, label=f"{method} prediction")
        X_train_fairlearn = _dense_float32(X_train)
        X_predict = _dense_float32(X_test)
        _check_runtime_memory_budget(f"{method} dense conversion")
        metadata["fairlearn_dense"] = True
        metadata["fairlearn_train_dense_estimate_mb"] = _dense_size_mb(X_train)
        metadata["fairlearn_prediction_dense_estimate_mb"] = _dense_size_mb(X_test)
        metadata.update(_matrix_metadata(X_train_fairlearn, "fairlearn_train_matrix"))
        del X_train, X_test
        trim_process_memory()
        fitted = _fit_exponentiated_gradient(method, estimator, X_train_fairlearn, y_train, sensitive_train, config)
        _check_runtime_memory_budget(f"{method} fit")
        metadata["uses_protected_attributes"] = True
        del X_train_fairlearn
    elif method == "threshold_optimizer_eo":
        _check_dense_budget(X_train, config, label=f"{method} train")
        _check_dense_budget(X_test, config, label=f"{method} prediction")
        X_train_fairlearn = _dense_float32(X_train)
        X_predict = _dense_float32(X_test)
        _check_runtime_memory_budget(f"{method} dense conversion")
        metadata["fairlearn_dense"] = True
        metadata["fairlearn_train_dense_estimate_mb"] = _dense_size_mb(X_train)
        metadata["fairlearn_prediction_dense_estimate_mb"] = _dense_size_mb(X_test)
        metadata.update(_matrix_metadata(X_train_fairlearn, "fairlearn_train_matrix"))
        del X_train, X_test
        trim_process_memory()
        sensitive_train, threshold_degenerate_metadata = _coalesce_degenerate_sensitive_groups(sensitive_train, y_train)
        fitted, threshold_fit_sensitive, threshold_val_metadata = _fit_threshold_optimizer(
            estimator, X_train_fairlearn, y_train, sensitive_train, config, seed
        )
        _check_runtime_memory_budget(f"{method} fit")
        sensitive_test, threshold_metadata = _align_sensitive_for_threshold_prediction(threshold_fit_sensitive, sensitive_test)
        metadata.update(threshold_degenerate_metadata)
        metadata.update({f"validation_{key}": value for key, value in threshold_val_metadata.items()})
        metadata.update(threshold_metadata)
        metadata["uses_protected_attributes"] = True
        del X_train_fairlearn
    else:
        raise ValueError(f"Unsupported method: {method}")

    y_pred = _predict_with_optional_sensitive(fitted, X_predict, sensitive_test, seed=seed).astype(int)
    y_prob = _predict_probability(fitted, X_predict)
    _check_runtime_memory_budget(f"{method} prediction")
    fit_seconds = perf_counter() - start
    metadata["fit_rows"] = int(len(train_frame))
    metadata["test_rows"] = int(len(test_frame))
    del fitted, X_predict
    trim_process_memory()
    return PredictionResult(
        method=method,
        y_pred=y_pred,
        y_prob=y_prob,
        fit_seconds=fit_seconds,
        metadata=metadata,
    )


def _fit_exponentiated_gradient(method: str, estimator, X_train, y_train, sensitive_train, config: dict):
    try:
        from fairlearn.reductions import DemographicParity, EqualizedOdds, ExponentiatedGradient
    except ImportError as exc:
        raise RuntimeError("fairlearn is required for exponentiated_gradient baselines.") from exc

    if method == "exponentiated_gradient_eo":
        constraint = EqualizedOdds()
    else:
        constraint = DemographicParity()
    mitigator = ExponentiatedGradient(
        clone(estimator),
        constraints=constraint,
        eps=float(config.get("fairlearn", {}).get("eps", 0.02)),
    )
    mitigator.fit(X_train, y_train, sensitive_features=sensitive_train)
    return mitigator


def _fit_threshold_optimizer(estimator, X_train, y_train, sensitive_train, config: dict, seed: int):
    try:
        from fairlearn.postprocessing import ThresholdOptimizer
    except ImportError as exc:
        raise RuntimeError("fairlearn is required for threshold_optimizer_eo.") from exc

    train_idx, val_idx = train_test_split(
        np.arange(len(y_train)),
        test_size=float(config["splits"].get("validation_fraction", 0.25)),
        stratify=y_train,
        random_state=seed,
    )
    base = clone(estimator)
    base.fit(X_train[train_idx], y_train[train_idx])
    optimizer = ThresholdOptimizer(
        estimator=base,
        constraints="equalized_odds",
        objective="balanced_accuracy_score",
        prefit=True,
        predict_method="predict_proba",
    )
    sensitive_val, val_metadata = _coalesce_degenerate_sensitive_groups(
        pd.Series(sensitive_train).iloc[val_idx],
        y_train[val_idx],
    )
    optimizer.fit(
        X_train[val_idx],
        y_train[val_idx],
        sensitive_features=sensitive_val,
    )
    return optimizer, sensitive_val, val_metadata
