import numpy as np
import pandas as pd

from fairmix_audit.baselines import fit_predict_method, make_estimator
from fairmix_audit import baselines
from fairmix_audit.missingness import MissingnessRegime, apply_protected_missingness


def _tiny_config():
    return {
        "data": {
            "label_column": "label",
            "protected_attributes": ["RAC1P", "SEX"],
            "oracle_sensitive_attributes": ["RAC1P", "SEX"],
            "context_attributes": ["__state", "__year"],
        },
        "model": {
            "estimator": "logistic_regression",
            "numeric_features": ["AGEP", "WKHP"],
            "onehot_max_categories": 16,
            "max_iter": 200,
            "class_weight": None,
        },
        "methods": {
            "enabled": [
                "erm",
                "feature_mixing",
                "reweighing",
                "protected_group_mixing",
                "exponentiated_gradient_dp",
                "exponentiated_gradient_eo",
                "threshold_optimizer_eo",
            ]
        },
        "feature_mixing": {
            "mix_ratio": 0.25,
            "beta": 0.4,
            "min_contexts_per_label": 2,
        },
        "fairlearn": {"eps": 0.05, "max_dense_matrix_mb": 128},
        "splits": {"validation_fraction": 0.25},
    }


def _tiny_frame(seed=7):
    rng = np.random.default_rng(seed)
    n = 160
    frame = pd.DataFrame(
        {
            "AGEP": rng.integers(18, 70, size=n),
            "WKHP": rng.integers(1, 60, size=n),
            "COW": rng.integers(1, 5, size=n),
            "SCHL": rng.integers(1, 8, size=n),
            "MAR": rng.integers(1, 4, size=n),
            "OCCP": rng.integers(1, 12, size=n),
            "RAC1P": rng.choice([1, 2, 6], size=n, p=[0.55, 0.30, 0.15]),
            "SEX": rng.choice([1, 2], size=n),
            "__state": rng.choice(["CA", "TX"], size=n),
            "__year": rng.choice([2018, 2019], size=n),
        }
    )
    logits = 0.04 * (frame["AGEP"] - 35) + 0.03 * frame["WKHP"] - 0.25 * (frame["RAC1P"] == 6)
    probs = 1 / (1 + np.exp(-logits))
    frame["label"] = (rng.random(n) < probs).astype(int)
    return frame


def test_every_default_method_runs_with_sparse_preprocessing_and_missingness():
    config = _tiny_config()
    frame = _tiny_frame()
    train = frame.iloc[:120].reset_index(drop=True)
    test = frame.iloc[120:].reset_index(drop=True)
    regime = MissingnessRegime(
        name="mcar_50_train_prediction",
        mechanism="mcar",
        availability=0.5,
        scope="train_and_prediction",
    )
    train_missing = apply_protected_missingness(train, ["RAC1P", "SEX"], regime, seed=11)
    test_missing = apply_protected_missingness(test, ["RAC1P", "SEX"], regime, seed=12)

    for method in config["methods"]["enabled"]:
        result = fit_predict_method(
            method,
            train,
            test,
            config,
            seed=13,
            sensitive_train=train_missing.sensitive_observed,
            sensitive_test=test_missing.sensitive_observed,
            missingness_metadata=train_missing.metadata,
        )
        assert len(result.y_pred) == len(test)
        assert set(np.unique(result.y_pred)).issubset({0, 1})


def test_tree_parallelism_is_bounded_by_available_cpus(monkeypatch):
    monkeypatch.setattr("os.cpu_count", lambda: 2)

    estimator = make_estimator(
        {
            "model": {
                "estimator": "random_forest",
                "n_jobs": -1,
                "n_estimators": 5,
            }
        }
    )

    assert estimator.n_jobs == 2


def test_random_forest_can_bind_random_state_to_run_seed():
    estimator = make_estimator(
        {
            "model": {
                "estimator": "random_forest",
                "random_state": 2027,
                "random_state_from_run_seed": True,
                "random_state_seed_offset": 100,
                "n_estimators": 5,
            }
        },
        seed=2031,
    )

    assert estimator.random_state == 2131


def test_random_forest_keeps_fixed_random_state_by_default():
    estimator = make_estimator(
        {
            "model": {
                "estimator": "random_forest",
                "random_state": 2027,
                "n_estimators": 5,
            }
        },
        seed=2031,
    )

    assert estimator.random_state == 2027


def test_explicit_parallelism_is_capped_by_available_cpus(monkeypatch):
    monkeypatch.setattr("os.cpu_count", lambda: 2)

    estimator = make_estimator(
        {
            "model": {
                "estimator": "logistic_regression",
                "n_jobs": 8,
                "max_iter": 10,
            }
        }
    )

    assert estimator.n_jobs == 2


def test_baseline_runtime_memory_guard_raises_at_hard_limit(monkeypatch):
    monkeypatch.setenv("FAIRMIX_HARD_RSS_MB", "100")
    monkeypatch.setattr(baselines, "_rss_mb", lambda: 101.0)

    try:
        baselines._check_runtime_memory_budget("unit-test")
    except MemoryError as exc:
        assert "unit-test" in str(exc)
    else:
        raise AssertionError("Expected MemoryError")
