import pandas as pd

from fairmix_audit.stats import (
    fairness_conclusion_flips,
    hidden_intersectional_regressions,
    mcar_vs_mnar_ablation,
    oracle_vs_missing_comparison,
    summarize_missingness_sensitivity,
)


def _row(method, missingness, *, accuracy, race_eo, sex_eo, inter_eo, inter_wga, mechanism="mcar", availability=1.0):
    return {
        "task": "ACSIncome",
        "split_name": "split0",
        "split_mode": "temporal",
        "seed": 11,
        "method": method,
        "missingness_name": missingness,
        "missingness_mechanism": mechanism,
        "missingness_scope": "train_and_prediction",
        "target_availability": availability,
        "train_sensitive_observed_fraction": availability,
        "prediction_sensitive_observed_fraction": availability,
        "accuracy": accuracy,
        "balanced_accuracy": accuracy - 0.02,
        "roc_auc": accuracy + 0.05,
        "brier": 1.0 - accuracy,
        "rac1p_demographic_parity_diff": race_eo + 0.01,
        "sex_demographic_parity_diff": sex_eo + 0.01,
        "rac1p_x_sex_demographic_parity_diff": inter_eo + 0.01,
        "rac1p_equalized_odds_diff": race_eo,
        "sex_equalized_odds_diff": sex_eo,
        "rac1p_x_sex_equalized_odds_diff": inter_eo,
        "rac1p_worst_group_accuracy": inter_wga + 0.02,
        "sex_worst_group_accuracy": inter_wga + 0.01,
        "rac1p_x_sex_worst_group_accuracy": inter_wga,
    }


def _metrics():
    rows = [
        _row("erm", "full", accuracy=0.80, race_eo=0.10, sex_eo=0.10, inter_eo=0.12, inter_wga=0.70),
        _row("feature_mixing", "full", accuracy=0.81, race_eo=0.08, sex_eo=0.10, inter_eo=0.14, inter_wga=0.68),
        _row("reweighing", "full", accuracy=0.79, race_eo=0.05, sex_eo=0.06, inter_eo=0.06, inter_wga=0.72),
        _row("erm", "mcar_20_train_prediction", accuracy=0.80, race_eo=0.10, sex_eo=0.10, inter_eo=0.12, inter_wga=0.70, availability=0.2),
        _row("feature_mixing", "mcar_20_train_prediction", accuracy=0.81, race_eo=0.08, sex_eo=0.10, inter_eo=0.14, inter_wga=0.68, availability=0.2),
        _row("reweighing", "mcar_20_train_prediction", accuracy=0.78, race_eo=0.11, sex_eo=0.10, inter_eo=0.13, inter_wga=0.69, availability=0.2),
        _row("reweighing", "mnar_20_train_prediction", accuracy=0.77, race_eo=0.14, sex_eo=0.12, inter_eo=0.17, inter_wga=0.66, mechanism="mnar", availability=0.2),
    ]
    return pd.DataFrame(rows)


def test_oracle_vs_missing_compares_group_aware_method_to_oracle_and_agnostic_refs():
    comparison = oracle_vs_missing_comparison(_metrics(), {})

    row = comparison[
        (comparison["method"] == "reweighing")
        & (comparison["missingness_name"] == "mcar_20_train_prediction")
        & (comparison["metric"] == "rac1p_x_sex_equalized_odds_diff")
    ].iloc[0]

    assert round(row["delta_missing_vs_oracle"], 4) == 0.07
    assert round(row["delta_missing_vs_erm"], 4) == 0.01
    assert round(row["delta_missing_vs_feature_mixing"], 4) == -0.01


def test_fairness_conclusion_flips_detects_changed_best_method_under_missingness():
    flips = fairness_conclusion_flips(_metrics(), {"audit": {"conclusion_accuracy_drop_tolerance": 0.03}})

    changed = flips[
        (flips["missingness_name"] == "mcar_20_train_prediction")
        & (flips["criterion"] == "intersectional_equalized_odds")
    ].iloc[0]

    assert changed["oracle_best_method"] == "reweighing"
    assert changed["missing_best_method"] == "erm"
    assert bool(changed["method_changed"])


def test_hidden_intersectional_regressions_counts_single_axis_gains_that_hide_intersectional_harm():
    hidden = hidden_intersectional_regressions(
        _metrics(),
        {"audit": {"hidden_regression_tolerance": 0.01, "hidden_worst_group_drop_tolerance": 0.005}},
    )

    row = hidden[
        (hidden["method"] == "feature_mixing")
        & (hidden["single_axis"] == "RAC1P")
        & (hidden["fairness_metric"] == "equalized_odds")
    ].iloc[0]

    assert bool(row["hidden_intersectional_regression"])
    assert round(row["single_axis_delta"], 4) == -0.02
    assert round(row["intersection_delta"], 4) == 0.02


def test_mcar_vs_mnar_ablation_matches_availability_buckets():
    ablation = mcar_vs_mnar_ablation(_metrics(), {})

    row = ablation[
        (ablation["method"] == "reweighing")
        & (ablation["availability_bucket"] == 0.2)
        & (ablation["metric"] == "accuracy")
    ].iloc[0]

    assert round(row["delta_mnar_vs_mcar"], 4) == -0.01


def test_missingness_sensitivity_summarizes_curve_points():
    sensitivity = summarize_missingness_sensitivity(_metrics(), {})

    methods = set(sensitivity["method"])
    availability = set(sensitivity["availability_bucket"])
    assert {"erm", "feature_mixing", "reweighing"}.issubset(methods)
    assert {1.0, 0.2}.issubset(availability)
