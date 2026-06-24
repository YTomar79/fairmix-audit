import json

import pandas as pd

from fairmix_audit.reporting import write_analysis_artifacts, write_model_cards, write_tables


def test_write_model_cards_uses_disk_backed_metadata_lookup_and_cleans_up(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    metrics = pd.DataFrame(
        [
            {
                "run_id": "r0",
                "method": "erm",
                "task": "ACSIncome",
                "split_name": "split0",
                "seed": 11,
                "missingness_name": "full",
                "train_sensitive_observed_fraction": 1.0,
                "prediction_sensitive_observed_fraction": 1.0,
                "accuracy": 0.9,
                "balanced_accuracy": 0.8,
                "brier": 0.1,
                "rac1p_equalized_odds_diff": 0.2,
                "sex_equalized_odds_diff": 0.3,
                "rac1p_x_sex_worst_group_accuracy": 0.4,
            }
        ]
    )
    (run_dir / "metrics.csv").write_text(metrics.to_csv(index=False), encoding="utf-8")
    (run_dir / "model_metadata.jsonl").write_text(
        json.dumps({"run_id": "r0", "fit_rows": 123, "test_rows": 45}) + "\n",
        encoding="utf-8",
    )

    write_model_cards(run_dir)

    card = (run_dir / "model_cards" / "r0.md").read_text(encoding="utf-8")
    assert "Fit rows: `123`" in card
    assert "Test rows: `45`" in card
    assert not list(run_dir.glob(".model_metadata_lookup.*"))


def test_write_analysis_artifacts_and_tables_emit_missingness_experiment_outputs(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    metrics = pd.DataFrame(
        [
            {
                "run_id": "full_erm",
                "task": "ACSIncome",
                "split_name": "split0",
                "split_mode": "temporal",
                "seed": 11,
                "missingness_name": "full",
                "missingness_mechanism": "mcar",
                "missingness_scope": "train_and_prediction",
                "target_availability": 1.0,
                "train_sensitive_observed_fraction": 1.0,
                "prediction_sensitive_observed_fraction": 1.0,
                "method": "erm",
                "fit_seconds": 1.0,
                "accuracy": 0.8,
                "balanced_accuracy": 0.78,
                "brier": 0.2,
                "roc_auc": 0.85,
                "rac1p_equalized_odds_diff": 0.1,
                "sex_equalized_odds_diff": 0.1,
                "rac1p_x_sex_equalized_odds_diff": 0.1,
                "rac1p_demographic_parity_diff": 0.1,
                "sex_demographic_parity_diff": 0.1,
                "rac1p_x_sex_demographic_parity_diff": 0.1,
                "rac1p_worst_group_accuracy": 0.7,
                "sex_worst_group_accuracy": 0.7,
                "rac1p_x_sex_worst_group_accuracy": 0.7,
            },
            {
                "run_id": "full_reweighing",
                "task": "ACSIncome",
                "split_name": "split0",
                "split_mode": "temporal",
                "seed": 11,
                "missingness_name": "full",
                "missingness_mechanism": "mcar",
                "missingness_scope": "train_and_prediction",
                "target_availability": 1.0,
                "train_sensitive_observed_fraction": 1.0,
                "prediction_sensitive_observed_fraction": 1.0,
                "method": "reweighing",
                "fit_seconds": 1.0,
                "accuracy": 0.79,
                "balanced_accuracy": 0.77,
                "brier": 0.21,
                "roc_auc": 0.84,
                "rac1p_equalized_odds_diff": 0.05,
                "sex_equalized_odds_diff": 0.05,
                "rac1p_x_sex_equalized_odds_diff": 0.05,
                "rac1p_demographic_parity_diff": 0.05,
                "sex_demographic_parity_diff": 0.05,
                "rac1p_x_sex_demographic_parity_diff": 0.05,
                "rac1p_worst_group_accuracy": 0.72,
                "sex_worst_group_accuracy": 0.72,
                "rac1p_x_sex_worst_group_accuracy": 0.72,
            },
            {
                "run_id": "missing_erm",
                "task": "ACSIncome",
                "split_name": "split0",
                "split_mode": "temporal",
                "seed": 11,
                "missingness_name": "mcar_20_train_prediction",
                "missingness_mechanism": "mcar",
                "missingness_scope": "train_and_prediction",
                "target_availability": 0.2,
                "train_sensitive_observed_fraction": 0.2,
                "prediction_sensitive_observed_fraction": 0.2,
                "method": "erm",
                "fit_seconds": 1.0,
                "accuracy": 0.8,
                "balanced_accuracy": 0.78,
                "brier": 0.2,
                "roc_auc": 0.85,
                "rac1p_equalized_odds_diff": 0.1,
                "sex_equalized_odds_diff": 0.1,
                "rac1p_x_sex_equalized_odds_diff": 0.1,
                "rac1p_demographic_parity_diff": 0.1,
                "sex_demographic_parity_diff": 0.1,
                "rac1p_x_sex_demographic_parity_diff": 0.1,
                "rac1p_worst_group_accuracy": 0.7,
                "sex_worst_group_accuracy": 0.7,
                "rac1p_x_sex_worst_group_accuracy": 0.7,
            },
            {
                "run_id": "missing_reweighing",
                "task": "ACSIncome",
                "split_name": "split0",
                "split_mode": "temporal",
                "seed": 11,
                "missingness_name": "mcar_20_train_prediction",
                "missingness_mechanism": "mcar",
                "missingness_scope": "train_and_prediction",
                "target_availability": 0.2,
                "train_sensitive_observed_fraction": 0.2,
                "prediction_sensitive_observed_fraction": 0.2,
                "method": "reweighing",
                "fit_seconds": 1.0,
                "accuracy": 0.78,
                "balanced_accuracy": 0.76,
                "brier": 0.22,
                "roc_auc": 0.83,
                "rac1p_equalized_odds_diff": 0.11,
                "sex_equalized_odds_diff": 0.11,
                "rac1p_x_sex_equalized_odds_diff": 0.13,
                "rac1p_demographic_parity_diff": 0.11,
                "sex_demographic_parity_diff": 0.11,
                "rac1p_x_sex_demographic_parity_diff": 0.13,
                "rac1p_worst_group_accuracy": 0.69,
                "sex_worst_group_accuracy": 0.69,
                "rac1p_x_sex_worst_group_accuracy": 0.69,
            },
        ]
    )
    metrics.to_csv(run_dir / "metrics.csv", index=False)

    write_analysis_artifacts(run_dir, metrics=metrics, config={})
    write_tables(run_dir, metrics=metrics)

    assert (run_dir / "missingness_sensitivity.csv").exists()
    assert (run_dir / "oracle_vs_missing.csv").exists()
    assert (run_dir / "fairness_conclusion_flips.csv").exists()
    assert (run_dir / "tables" / "oracle_vs_missing_summary.csv").exists()
    assert (run_dir / "tables" / "fairness_conclusion_flip_summary.csv").exists()
