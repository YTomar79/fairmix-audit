from __future__ import annotations

import json
import os
import sqlite3
import tempfile
from pathlib import Path

if "MPLCONFIGDIR" not in os.environ:
    mpl_cache = Path(tempfile.gettempdir()) / "fairmix_matplotlib"
    mpl_cache.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mpl_cache)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from fairmix_audit.stats import summarize_missingness_sensitivity


def write_plots(run_dir: str | Path, *, metrics: pd.DataFrame | None = None) -> None:
    run_dir = Path(run_dir)
    if metrics is None:
        metrics = pd.read_csv(run_dir / "metrics.csv")
    plot_dir = run_dir / "plots"
    plot_dir.mkdir(exist_ok=True)

    sns.set_theme(style="whitegrid", context="paper")
    plot_metrics = _collapse_missingness(metrics)
    _barplot(
        plot_metrics,
        x="method",
        y="accuracy",
        hue="missingness_name",
        title="Accuracy by Method and Protected-Attribute Availability",
        path=plot_dir / "accuracy_by_method.png",
    )
    _barplot(
        plot_metrics,
        x="method",
        y="balanced_accuracy",
        hue="missingness_name",
        title="Balanced Accuracy by Method and Protected-Attribute Availability",
        path=plot_dir / "balanced_accuracy_by_method.png",
    )

    fairness_cols = [
        column
        for column in ["rac1p_equalized_odds_diff", "sex_equalized_odds_diff", "rac1p_x_sex_equalized_odds_diff"]
        if column in metrics.columns
    ]
    if fairness_cols:
        long = plot_metrics.melt(
            id_vars=["task", "split_mode", "method"],
            value_vars=fairness_cols,
            var_name="metric",
            value_name="value",
        )
        _barplot(
            long,
            x="method",
            y="value",
            hue="metric",
            title="Equalized Odds Gaps by Audited Group",
            path=plot_dir / "equalized_odds_gaps.png",
        )
    _write_missingness_analysis_plots(run_dir, plot_dir, metrics)


def write_tables(run_dir: str | Path, *, metrics: pd.DataFrame | None = None) -> None:
    run_dir = Path(run_dir)
    if metrics is None:
        metrics = pd.read_csv(run_dir / "metrics.csv")
    table_dir = run_dir / "tables"
    table_dir.mkdir(exist_ok=True)

    group_cols = ["task", "split_mode", "missingness_name", "missingness_mechanism", "missingness_scope", "method"]
    primary = (
        metrics.groupby(group_cols, as_index=False)
        .agg(
            accuracy=("accuracy", "mean"),
            balanced_accuracy=("balanced_accuracy", "mean"),
            brier=("brier", "mean"),
            fit_seconds=("fit_seconds", "mean"),
        )
        .sort_values(["task", "split_mode", "method"])
    )
    primary.to_csv(table_dir / "utility_summary.csv", index=False)

    fairness_cols = [column for column in metrics.columns if column.endswith("_equalized_odds_diff")]
    if fairness_cols:
        fairness = (
            metrics.groupby(group_cols, as_index=False)[fairness_cols]
            .mean()
            .sort_values(["task", "split_mode", "method"])
        )
        fairness.to_csv(table_dir / "fairness_summary.csv", index=False)
    _write_missingness_analysis_tables(run_dir, table_dir, metrics)


def write_analysis_artifacts(
    run_dir: str | Path,
    *,
    metrics: pd.DataFrame | None = None,
    config: dict | None = None,
) -> None:
    run_dir = Path(run_dir)
    if metrics is None:
        metrics = pd.read_csv(run_dir / "metrics.csv")
    if config is None:
        config_path = run_dir / "config.resolved.yml"
        if config_path.exists():
            from fairmix_audit.config import load_config

            config = load_config(config_path, merge_defaults=False)
        else:
            config = {}
    from fairmix_audit.stats import (
        fairness_conclusion_flips,
        hidden_intersectional_regressions,
        mcar_vs_mnar_ablation,
        oracle_vs_missing_comparison,
    )

    summarize_missingness_sensitivity(metrics, config).to_csv(run_dir / "missingness_sensitivity.csv", index=False)
    oracle_vs_missing_comparison(metrics, config).to_csv(run_dir / "oracle_vs_missing.csv", index=False)
    fairness_conclusion_flips(metrics, config).to_csv(run_dir / "fairness_conclusion_flips.csv", index=False)
    hidden_intersectional_regressions(metrics, config).to_csv(
        run_dir / "hidden_intersectional_regressions.csv",
        index=False,
    )
    mcar_vs_mnar_ablation(metrics, config).to_csv(run_dir / "mcar_vs_mnar_ablation.csv", index=False)


def _barplot(df: pd.DataFrame, *, x: str, y: str, hue: str, title: str, path: Path) -> None:
    plt.figure(figsize=(9, 4.8))
    ax = sns.barplot(data=df, x=x, y=y, hue=hue, errorbar="sd")
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel(y.replace("_", " ").title())
    ax.tick_params(axis="x", rotation=25)
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def _lineplot(df: pd.DataFrame, *, x: str, y: str, hue: str, title: str, path: Path) -> None:
    plt.figure(figsize=(9, 4.8))
    ax = sns.lineplot(data=df, x=x, y=y, hue=hue, marker="o", errorbar="sd")
    ax.set_title(title)
    ax.set_xlabel("Protected-Attribute Availability")
    ax.set_ylabel(y.replace("_", " ").title())
    ax.invert_xaxis()
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def _write_missingness_analysis_plots(run_dir: Path, plot_dir: Path, metrics: pd.DataFrame) -> None:
    sensitivity = _read_optional_csv(run_dir / "missingness_sensitivity.csv")
    if sensitivity.empty:
        sensitivity = summarize_missingness_sensitivity(metrics)
    if not sensitivity.empty:
        mcar = sensitivity[sensitivity["missingness_mechanism"] == "mcar"].copy()
        if not mcar.empty and "availability_bucket" in mcar.columns:
            if "rac1p_x_sex_equalized_odds_diff" in mcar.columns:
                _lineplot(
                    mcar,
                    x="availability_bucket",
                    y="rac1p_x_sex_equalized_odds_diff",
                    hue="method",
                    title="Intersectional Equalized-Odds Gap Across MCAR Availability",
                    path=plot_dir / "missingness_sensitivity_intersectional_eo.png",
                )
            if "accuracy" in mcar.columns:
                _lineplot(
                    mcar,
                    x="availability_bucket",
                    y="accuracy",
                    hue="method",
                    title="Accuracy Across MCAR Protected-Attribute Availability",
                    path=plot_dir / "missingness_sensitivity_accuracy.png",
                )

    flips = _read_optional_csv(run_dir / "fairness_conclusion_flips.csv")
    if not flips.empty and {"criterion", "method_changed"}.issubset(flips.columns):
        flip_summary = (
            flips.groupby("criterion", as_index=False)
            .agg(flip_rate=("method_changed", "mean"))
            .sort_values("flip_rate", ascending=False)
        )
        if not flip_summary.empty:
            _barplot(
                flip_summary,
                x="criterion",
                y="flip_rate",
                hue="criterion",
                title="Fairness Conclusion Flip Rate Under Missing Protected Attributes",
                path=plot_dir / "fairness_conclusion_flip_rates.png",
            )

    hidden = _read_optional_csv(run_dir / "hidden_intersectional_regressions.csv")
    if not hidden.empty and {"method", "hidden_intersectional_regression"}.issubset(hidden.columns):
        hidden_summary = (
            hidden.groupby("method", as_index=False)
            .agg(hidden_regression_rate=("hidden_intersectional_regression", "mean"))
            .sort_values("hidden_regression_rate", ascending=False)
        )
        if not hidden_summary.empty:
            _barplot(
                hidden_summary,
                x="method",
                y="hidden_regression_rate",
                hue="method",
                title="Hidden Intersectional Regression Rate After Single-Axis Gains",
                path=plot_dir / "hidden_intersectional_regression_rates.png",
            )


def _collapse_missingness(metrics: pd.DataFrame) -> pd.DataFrame:
    if "missingness_name" in metrics.columns:
        return metrics
    metrics = metrics.copy()
    metrics["missingness_name"] = "full"
    metrics["missingness_mechanism"] = "mcar"
    metrics["missingness_scope"] = "train_only"
    return metrics


def _write_missingness_analysis_tables(run_dir: Path, table_dir: Path, metrics: pd.DataFrame) -> None:
    sensitivity = _read_optional_csv(run_dir / "missingness_sensitivity.csv")
    if sensitivity.empty:
        sensitivity = summarize_missingness_sensitivity(metrics)
    if not sensitivity.empty:
        sensitivity.to_csv(table_dir / "missingness_sensitivity_summary.csv", index=False)

    oracle = _read_optional_csv(run_dir / "oracle_vs_missing.csv")
    if not oracle.empty:
        group_cols = [
            "method",
            "missingness_name",
            "missingness_mechanism",
            "missingness_scope",
            "availability_bucket",
            "metric",
        ]
        value_cols = [
            column
            for column in (
                "delta_missing_vs_oracle",
                "delta_missing_vs_erm",
                "delta_missing_vs_feature_mixing",
            )
            if column in oracle.columns
        ]
        if value_cols:
            oracle_summary = (
                oracle.groupby(group_cols, dropna=False, as_index=False)
                .agg(n_runs=("metric", "size"), **{column: (column, "mean") for column in value_cols})
                .sort_values(["method", "metric", "availability_bucket"])
            )
            oracle_summary.to_csv(table_dir / "oracle_vs_missing_summary.csv", index=False)

    flips = _read_optional_csv(run_dir / "fairness_conclusion_flips.csv")
    if not flips.empty:
        flip_summary = (
            flips.groupby(
                [
                    "criterion",
                    "missingness_name",
                    "missingness_mechanism",
                    "missingness_scope",
                    "availability_bucket",
                ],
                dropna=False,
                as_index=False,
            )
            .agg(
                n_decisions=("method_changed", "size"),
                n_flips=("method_changed", "sum"),
                flip_rate=("method_changed", "mean"),
            )
            .sort_values(["criterion", "availability_bucket"])
        )
        flip_summary.to_csv(table_dir / "fairness_conclusion_flip_summary.csv", index=False)

    hidden = _read_optional_csv(run_dir / "hidden_intersectional_regressions.csv")
    if not hidden.empty:
        hidden_summary = (
            hidden.groupby(
                [
                    "method",
                    "missingness_name",
                    "missingness_mechanism",
                    "missingness_scope",
                    "availability_bucket",
                    "single_axis",
                    "fairness_metric",
                ],
                dropna=False,
                as_index=False,
            )
            .agg(
                n_single_axis_improvements=("hidden_intersectional_regression", "size"),
                n_hidden_regressions=("hidden_intersectional_regression", "sum"),
                hidden_regression_rate=("hidden_intersectional_regression", "mean"),
            )
            .sort_values(["method", "fairness_metric", "availability_bucket"])
        )
        hidden_summary.to_csv(table_dir / "hidden_intersectional_regression_summary.csv", index=False)

    mnar = _read_optional_csv(run_dir / "mcar_vs_mnar_ablation.csv")
    if not mnar.empty:
        mnar_summary = (
            mnar.groupby(["method", "availability_bucket", "metric"], dropna=False, as_index=False)
            .agg(n_pairs=("metric", "size"), delta_mnar_vs_mcar=("delta_mnar_vs_mcar", "mean"))
            .sort_values(["method", "metric", "availability_bucket"])
        )
        mnar_summary.to_csv(table_dir / "mcar_vs_mnar_ablation_summary.csv", index=False)


def _read_optional_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def write_model_cards(run_dir: str | Path, *, metrics: pd.DataFrame | None = None) -> None:
    run_dir = Path(run_dir)
    if metrics is None:
        metrics = pd.read_csv(run_dir / "metrics.csv")
    metadata_path = run_dir / "model_metadata.jsonl"
    lookup = _MetadataLookup.from_jsonl(metadata_path, run_dir=run_dir)

    card_dir = run_dir / "model_cards"
    card_dir.mkdir(exist_ok=True)
    try:
        for _, row in metrics.iterrows():
            meta = lookup.get(row["run_id"])
            card_path = card_dir / f"{row['run_id']}.md"
            card_path.write_text(_model_card(row, meta), encoding="utf-8")
    finally:
        lookup.close()


class _MetadataLookup:
    def __init__(self, connection: sqlite3.Connection | None, path: Path | None) -> None:
        self._connection = connection
        self._path = path

    @classmethod
    def from_jsonl(cls, metadata_path: Path, *, run_dir: Path) -> "_MetadataLookup":
        if not metadata_path.exists():
            return cls(None, None)

        db_path = run_dir / f".model_metadata_lookup.{os.getpid()}.sqlite"
        _unlink_sqlite_files(db_path)
        connection = sqlite3.connect(db_path)
        try:
            connection.execute("PRAGMA journal_mode=OFF")
            connection.execute("PRAGMA synchronous=OFF")
            connection.execute("CREATE TABLE metadata (run_id TEXT PRIMARY KEY, payload TEXT NOT NULL)")
            batch: list[tuple[str, str]] = []
            with metadata_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    run_id = record.get("run_id")
                    if not run_id:
                        continue
                    batch.append((str(run_id), json.dumps(record, separators=(",", ":"))))
                    if len(batch) >= 1000:
                        connection.executemany("INSERT OR REPLACE INTO metadata VALUES (?, ?)", batch)
                        batch.clear()
            if batch:
                connection.executemany("INSERT OR REPLACE INTO metadata VALUES (?, ?)", batch)
            connection.commit()
            return cls(connection, db_path)
        except Exception:
            connection.close()
            _unlink_sqlite_files(db_path)
            raise

    def get(self, run_id: object) -> dict:
        if self._connection is None:
            return {}
        row = self._connection.execute(
            "SELECT payload FROM metadata WHERE run_id = ?",
            (str(run_id),),
        ).fetchone()
        return json.loads(row[0]) if row else {}

    def close(self) -> None:
        if self._connection is not None:
            self._connection.close()
            self._connection = None
        if self._path is not None:
            _unlink_sqlite_files(self._path)
            self._path = None


def _unlink_sqlite_files(path: Path) -> None:
    for suffix in ("", "-journal", "-wal", "-shm"):
        path.with_name(path.name + suffix).unlink(missing_ok=True)


def _model_card(row: pd.Series, metadata: dict) -> str:
    uses_protected = metadata.get("uses_protected_attributes", False)
    if not uses_protected and isinstance(metadata.get("mixing"), dict):
        uses_protected = metadata["mixing"].get("uses_protected_attributes", False)
    lines = [
        f"# Model Card: {row['run_id']}",
        "",
        "## Model Details",
        "",
        f"- Method: `{row['method']}`",
        f"- Task: `{row['task']}`",
        f"- Split: `{row['split_name']}`",
        f"- Seed: `{row['seed']}`",
        f"- Protected-attribute missingness: `{row.get('missingness_name', 'full')}`",
        f"- Training protected labels observed: `{row.get('train_sensitive_observed_fraction', 1.0):.3f}`",
        f"- Prediction protected labels observed: `{row.get('prediction_sensitive_observed_fraction', 1.0):.3f}`",
        f"- Fit rows: `{metadata.get('fit_rows', 'unknown')}`",
        f"- Test rows: `{metadata.get('test_rows', 'unknown')}`",
        f"- Uses protected attributes during training/mitigation: `{bool(uses_protected)}`",
        "",
        "## Metrics",
        "",
        f"- Accuracy: `{row.get('accuracy', float('nan')):.4f}`",
        f"- Balanced accuracy: `{row.get('balanced_accuracy', float('nan')):.4f}`",
        f"- Brier score: `{row.get('brier', float('nan')):.4f}`",
        f"- Race equalized odds difference: `{row.get('rac1p_equalized_odds_diff', float('nan')):.4f}`",
        f"- Sex equalized odds difference: `{row.get('sex_equalized_odds_diff', float('nan')):.4f}`",
        f"- Intersectional worst-group accuracy: `{row.get('rac1p_x_sex_worst_group_accuracy', float('nan')):.4f}`",
        "",
        "## Ethical Caveats",
        "",
        "- Utility improvement does not imply fairness improvement.",
        "- Attribute-agnostic mitigation still requires protected-attribute auditing where lawful and ethical.",
        "- Do not deploy methods marked unsafe in `audit_flags.csv`.",
    ]
    return "\n".join(lines) + "\n"
