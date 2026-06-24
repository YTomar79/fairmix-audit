#!/usr/bin/env python3
"""Compare finalized audit headline diagnostics across base learners.

This is a post-processing helper for the random-forest validation extension.
It reads already-finalized run directories and writes a compact CSV/Markdown
table that can be dropped into the paper as a "does the seed floor generalize?"
check. It never launches experiments or mutates run artifacts.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ID_COLS = ["task", "split_name", "split_mode", "seed"]
METHOD_LABELS = {
    "logistic": "Logistic regression",
    "random_forest": "Random forest",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="LABEL=PATH",
        help="Finalized run directory, e.g. logistic=results/... or random_forest=results/...",
    )
    parser.add_argument("--out-csv", default="paper/workshop_artifacts/tables/csv/base_learner_validation.csv")
    parser.add_argument("--out-md", default="paper/workshop_artifacts/tables/base_learner_validation.md")
    args = parser.parse_args()

    rows = []
    for item in args.run:
        label, run_dir = _parse_run_arg(item)
        rows.append(_summarize_run(label, Path(run_dir)))

    summary = pd.DataFrame(rows)
    out_csv = Path(args.out_csv)
    out_md = Path(args.out_md)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_csv, index=False)
    out_md.write_text(_to_markdown(summary), encoding="utf-8")
    print(f"Wrote {out_csv}")
    print(f"Wrote {out_md}")


def _parse_run_arg(text: str) -> tuple[str, str]:
    if "=" not in text:
        raise SystemExit(f"--run must be LABEL=PATH; got {text!r}")
    label, path = text.split("=", 1)
    label = label.strip()
    path = path.strip()
    if not label or not path:
        raise SystemExit(f"--run must be LABEL=PATH; got {text!r}")
    return label, path


def _summarize_run(label: str, run_dir: Path) -> dict[str, float | int | str]:
    metrics = _read_required(run_dir / "metrics.csv")
    flips = _read_required(run_dir / "fairness_conclusion_flips.csv")
    hidden = _read_required(run_dir / "hidden_intersectional_regressions.csv")
    seed_null_rate, seed_null_n = _full_label_seed_null(metrics)

    no_label = flips[flips["missingness_name"] == "none_observed"]
    no_label_rate = _bool_mean(no_label["method_changed"]) * 100.0
    strict_rate = _bool_mean(flips["method_changed"]) * 100.0
    margin_1pp_rate = _margin_filtered_rate(metrics, flips, margin=0.01) * 100.0

    hidden_flag = _as_bool(hidden["hidden_intersectional_regression"])
    no_threshold = hidden[hidden["method"] != "threshold_optimizer_eo"]
    return {
        "base_learner": METHOD_LABELS.get(label, label),
        "run_dir": str(run_dir),
        "strict_flip_rate_pct": strict_rate,
        "margin_1pp_flip_rate_pct": margin_1pp_rate,
        "seed_null_flip_rate_pct": seed_null_rate,
        "no_label_mcar_flip_rate_pct": no_label_rate,
        "no_label_minus_seed_null_pp": no_label_rate - seed_null_rate,
        "hidden_regression_rate_pct": _bool_mean(hidden_flag) * 100.0,
        "hidden_regression_excluding_threshold_pct": _bool_mean(no_threshold["hidden_intersectional_regression"]) * 100.0,
        "n_flip_rows": int(len(flips)),
        "n_seed_null_pairs": int(seed_null_n),
        "n_hidden_rows": int(len(hidden)),
    }


def _read_required(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"Required finalized run artifact is missing: {path}")
    return pd.read_csv(path)


def _available_criteria(frame: pd.DataFrame) -> dict[str, dict]:
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


def _criterion_scores(rows: pd.DataFrame, criterion: dict) -> pd.Series:
    values = rows[criterion["columns"]].apply(pd.to_numeric, errors="coerce")
    if criterion["aggregate"] == "max":
        return values.max(axis=1)
    if criterion["aggregate"] == "min":
        return values.min(axis=1)
    return values.mean(axis=1)


def _select_method(rows: pd.DataFrame, criterion: dict, *, tolerance: float = 0.01) -> dict | None:
    candidates = rows.copy()
    if criterion.get("accuracy_constrained") and "accuracy" in candidates.columns:
        erm = candidates[candidates["method"] == "erm"]
        if not erm.empty:
            floor = float(erm["accuracy"].max()) - tolerance
            constrained = candidates[candidates["accuracy"] >= floor]
            if not constrained.empty:
                candidates = constrained
    candidates = candidates.assign(_criterion_score=_criterion_scores(candidates, criterion))
    candidates = candidates[candidates["_criterion_score"].notna()]
    if candidates.empty:
        return None
    ascending = criterion["direction"] == "min"
    selected = candidates.sort_values(
        ["_criterion_score", "accuracy", "method"],
        ascending=[ascending, False, True],
    ).iloc[0]
    return {
        "method": selected["method"],
        "score": float(selected["_criterion_score"]),
        "accuracy": float(selected.get("accuracy", np.nan)),
    }


def _full_label_seed_null(metrics: pd.DataFrame) -> tuple[float, int]:
    criteria = _available_criteria(metrics)
    choices = []
    for key, group in metrics.groupby(ID_COLS, dropna=False):
        full = group[group["missingness_name"] == "full"]
        if full.empty:
            continue
        for criterion_name, criterion in criteria.items():
            choice = _select_method(full, criterion)
            if choice is not None:
                choices.append({**dict(zip(ID_COLS, key, strict=True)), "criterion": criterion_name, **choice})
    choices_frame = pd.DataFrame(choices)
    if choices_frame.empty:
        return float("nan"), 0

    changed = []
    for (_task, _split_name, _split_mode, _criterion), group in choices_frame.groupby(
        ["task", "split_name", "split_mode", "criterion"],
        dropna=False,
    ):
        group = group.sort_values("seed")
        for _, row_a in group.iterrows():
            for _, row_b in group[group["seed"] > row_a["seed"]].iterrows():
                changed.append(row_a["method"] != row_b["method"])
    return _bool_mean(pd.Series(changed)) * 100.0, len(changed)


def _margin_filtered_rate(metrics: pd.DataFrame, flips: pd.DataFrame, *, margin: float) -> float:
    criteria = _available_criteria(metrics)
    grouped = {key: group.copy() for key, group in metrics.groupby(ID_COLS, dropna=False)}
    qualified = []
    for _, row in flips.iterrows():
        key = tuple(row[column] for column in ID_COLS)
        group = grouped.get(key)
        criterion = criteria.get(row["criterion"])
        if group is None or criterion is None:
            continue
        full = group[group["missingness_name"] == "full"]
        missing = group[group["missingness_name"] == row["missingness_name"]]
        if full.empty or missing.empty:
            continue
        oracle_full = _method_score(full, row["oracle_best_method"], criterion)
        missing_full = _method_score(full, row["missing_best_method"], criterion)
        oracle_missing = _method_score(missing, row["oracle_best_method"], criterion)
        missing_missing = _method_score(missing, row["missing_best_method"], criterion)
        if any(np.isnan(value) for value in (oracle_full, missing_full, oracle_missing, missing_missing)):
            continue
        if criterion["direction"] == "min":
            oracle_margin = missing_full - oracle_full
            missing_margin = oracle_missing - missing_missing
        else:
            oracle_margin = oracle_full - missing_full
            missing_margin = missing_missing - oracle_missing
        qualified.append(_as_bool(pd.Series([row["method_changed"]])).iloc[0] and oracle_margin >= margin and missing_margin >= margin)
    return _bool_mean(pd.Series(qualified))


def _method_score(rows: pd.DataFrame, method: str, criterion: dict) -> float:
    subset = rows[rows["method"] == method]
    if subset.empty:
        return float("nan")
    scores = _criterion_scores(subset, criterion)
    return float(scores.iloc[0]) if not scores.empty else float("nan")


def _as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().isin({"true", "1", "yes"})


def _bool_mean(series: pd.Series) -> float:
    if series.empty:
        return float("nan")
    return float(_as_bool(series).mean())


def _to_markdown(frame: pd.DataFrame) -> str:
    display = frame.copy()
    numeric_cols = display.select_dtypes(include=[np.number]).columns
    for column in numeric_cols:
        if column.startswith("n_"):
            display[column] = display[column].astype(int)
        else:
            display[column] = display[column].map(lambda value: f"{value:.1f}" if pd.notna(value) else "")
    rows = [[str(value) for value in row] for row in display.to_numpy()]
    headers = [str(column) for column in display.columns]
    widths = [
        max(len(header), *(len(row[index]) for row in rows)) if rows else len(header)
        for index, header in enumerate(headers)
    ]

    def fmt_row(values: list[str]) -> str:
        return "| " + " | ".join(value.ljust(widths[index]) for index, value in enumerate(values)) + " |"

    lines = [
        fmt_row(headers),
        "| " + " | ".join("-" * width for width in widths) + " |",
    ]
    lines.extend(fmt_row(row) for row in rows)
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
