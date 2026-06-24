from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from fairmix_audit.baselines import fit_predict_method
from fairmix_audit.config import load_config, write_yaml
from fairmix_audit.data import (
    load_folktables_contexts,
    resolve_states,
    stable_seed,
)
from fairmix_audit.metrics import evaluate_predictions
from fairmix_audit.memory import rss_mb, trim_process_memory
from fairmix_audit.missingness import apply_protected_missingness, configured_regimes
from fairmix_audit.stats import audit_flags, summarize_metric_deltas


@dataclass(frozen=True)
class SplitPlan:
    name: str
    task: str
    mode: str
    cluster_id: str
    train_years: tuple[int, ...]
    test_year: int | None = None
    holdout_state: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "task": self.task,
            "mode": self.mode,
            "cluster_id": self.cluster_id,
            "train_years": list(self.train_years),
            "test_year": self.test_year,
            "holdout_state": self.holdout_state,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> SplitPlan:
        return cls(
            name=payload["name"],
            task=payload["task"],
            mode=payload["mode"],
            cluster_id=payload["cluster_id"],
            train_years=tuple(int(year) for year in payload.get("train_years", [])),
            test_year=int(payload["test_year"]) if payload.get("test_year") is not None else None,
            holdout_state=payload.get("holdout_state"),
        )


@dataclass(frozen=True)
class WorkItem:
    index: int
    seed: int
    task: str
    splits: tuple[SplitPlan, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "seed": self.seed,
            "task": self.task,
            "splits": [split.to_dict() for split in self.splits],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> WorkItem:
        return cls(
            index=int(payload["index"]),
            seed=int(payload["seed"]),
            task=payload["task"],
            splits=tuple(SplitPlan.from_dict(split) for split in payload["splits"]),
        )


def _log(message: str) -> None:
    timestamp = datetime.now().isoformat(timespec="seconds")
    print(f"[{timestamp}] {message}", flush=True)


def _rss_mb() -> float:
    return rss_mb()


def _release_memory() -> None:
    trim_process_memory()


def run_experiment(
    config_path: str | Path,
    *,
    chunk_size: int = 1,
    output_root: str | Path | None = None,
) -> Path:
    """Run the full audit workflow in-process and write final artifacts.

    The workflow enumerates train/test splits, evaluates every method under each
    configured missingness regime and seed, then writes metrics, statistical
    reports, tables, plots, and model cards into a timestamped run directory.
    """
    config = load_config(config_path)
    if output_root is not None:
        config = deepcopy(config)
        config.setdefault("project", {})
        config["project"]["output_dir"] = str(Path(output_root))

    run_dir = _make_run_dir(config)
    write_yaml(config, run_dir / "config.resolved.yml")

    split_plans = build_split_plans(config)
    if not split_plans:
        raise RuntimeError("No train/test splits were generated from the config.")
    work_items = build_work_items(config, split_plans, chunk_size=chunk_size)
    states = resolve_states(config["data"]["states"])
    _log(f"Prepared {len(work_items)} work items across {len(split_plans)} splits.")

    metric_rows: list[dict[str, Any]] = []
    group_frames: list[pd.DataFrame] = []
    metadata_rows: list[dict[str, Any]] = []
    for item in work_items:
        item_metrics, item_groups, item_metadata = _execute_work_item(
            item,
            config,
            states,
            download_allowed=True,
        )
        metric_rows.extend(item_metrics)
        group_frames.extend(item_groups)
        metadata_rows.extend(item_metadata)
        _release_memory()

    if not metric_rows:
        raise RuntimeError("The run produced no metric rows; check the config.")

    metrics = pd.DataFrame(metric_rows)
    sort_columns = [
        column
        for column in ("task", "split_name", "seed", "missingness_name", "method")
        if column in metrics.columns
    ]
    if sort_columns:
        metrics = metrics.sort_values(sort_columns).reset_index(drop=True)
    group_metrics = pd.concat(group_frames, ignore_index=True) if group_frames else None

    return _write_final_outputs(
        run_dir,
        config,
        config_path=str(Path(config_path)),
        metrics=metrics,
        group_metrics=group_metrics,
        metadata_rows=metadata_rows,
        item_count=len(work_items),
    )


def build_split_plans(config: dict) -> list[SplitPlan]:
    tasks = list(config["data"]["tasks"])
    years = sorted(int(year) for year in config["data"]["years"])
    states = sorted(resolve_states(config["data"]["states"]))
    split_cfg = config["splits"]
    modes = split_cfg.get("modes", ["temporal"])
    temporal_cfg = split_cfg.get("temporal", {})
    geographic_cfg = split_cfg.get("geographic", {})
    max_holdouts = int(geographic_cfg.get("max_holdout_states", len(states)))
    holdout_states = states[:max_holdouts]

    plans: list[SplitPlan] = []
    for task in tasks:
        if "temporal" in modes:
            for train_years, test_year in _temporal_specs_for_years(years, temporal_cfg):
                train_tuple = tuple(sorted(int(year) for year in train_years))
                plans.append(
                    SplitPlan(
                        name=f"{task}_temporal_{'_'.join(map(str, train_tuple))}_to_{test_year}",
                        task=task,
                        mode="temporal",
                        cluster_id=f"{task}:temporal:{test_year}",
                        train_years=train_tuple,
                        test_year=int(test_year),
                    )
                )

        if "geographic" in modes and geographic_cfg.get("leave_one_state_out", True):
            for state in holdout_states:
                plans.append(
                    SplitPlan(
                        name=f"{task}_geographic_holdout_{state}",
                        task=task,
                        mode="geographic",
                        cluster_id=f"{task}:geographic:{state}",
                        train_years=tuple(years),
                        holdout_state=state,
                    )
                )

        if "geo_temporal" in modes:
            for train_years, test_year in _temporal_specs_for_years(years, temporal_cfg):
                train_tuple = tuple(sorted(int(year) for year in train_years))
                for state in holdout_states:
                    plans.append(
                        SplitPlan(
                            name=f"{task}_geo_temporal_holdout_{state}_{'_'.join(map(str, train_tuple))}_to_{test_year}",
                            task=task,
                            mode="geo_temporal",
                            cluster_id=f"{task}:geo_temporal:{state}:{test_year}",
                            train_years=train_tuple,
                            test_year=int(test_year),
                            holdout_state=state,
                        )
                    )
    return plans


def build_work_items(config: dict, split_plans: Iterable[SplitPlan], *, chunk_size: int = 1) -> list[WorkItem]:
    if chunk_size < 1:
        raise ValueError("chunk_size must be at least 1.")

    seeds = [int(seed) for seed in config["project"].get("random_seeds", [2027])]
    tasks = list(config["data"]["tasks"])
    by_task: dict[str, list[SplitPlan]] = {task: [] for task in tasks}
    for plan in split_plans:
        by_task.setdefault(plan.task, []).append(plan)

    work_items: list[WorkItem] = []
    index = 0
    for seed in seeds:
        for task in tasks:
            task_splits = by_task.get(task, [])
            for chunk in _chunked(task_splits, chunk_size):
                work_items.append(
                    WorkItem(
                        index=index,
                        seed=seed,
                        task=task,
                        splits=tuple(chunk),
                    )
                )
                index += 1
    return work_items


def _make_run_dir(config: dict) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = Path(config["project"].get("output_dir", "results"))
    run_dir = output_root / f"{config['project'].get('name', 'run')}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _chunked(items: list[SplitPlan], chunk_size: int) -> Iterable[list[SplitPlan]]:
    for start in range(0, len(items), chunk_size):
        yield items[start : start + chunk_size]


def _temporal_specs_for_years(years: Iterable[int], temporal_cfg: dict) -> list[tuple[set[int], int]]:
    available_years = sorted(int(year) for year in years)
    if temporal_cfg.get("rolling", False):
        min_train_years = int(temporal_cfg.get("min_train_years", 1))
        return [
            (set(available_years[:i]), int(available_years[i]))
            for i in range(min_train_years, len(available_years))
        ]

    train_years = set(int(year) for year in temporal_cfg.get("train_years", []))
    test_year = int(temporal_cfg.get("test_year"))
    return [(train_years, test_year)]


def _execute_work_item(
    item: WorkItem,
    config: dict,
    states: list[str],
    *,
    download_allowed: bool,
) -> tuple[list[dict[str, Any]], list[pd.DataFrame], list[dict[str, Any]]]:
    all_metric_rows: list[dict[str, Any]] = []
    all_group_rows: list[pd.DataFrame] = []
    metadata_rows: list[dict[str, Any]] = []

    for split_number, split in enumerate(item.splits, start=1):
        _log(
            f"Loading split {split_number}/{len(item.splits)}: {split.name} "
            f"(mode={split.mode}) download_allowed={download_allowed} rss_mb={_rss_mb():.1f}"
        )
        train_frame, test_frame = _load_split_frames(
            config,
            seed=item.seed,
            split=split,
            states=states,
            download=download_allowed,
        )
        _log(
            f"Split ready: {split.name} train_rows={len(train_frame):,} "
            f"test_rows={len(test_frame):,} rss_mb={_rss_mb():.1f}"
        )
        metric_rows, group_rows, split_metadata = _run_methods_for_split(
            train_frame,
            test_frame,
            split,
            config,
            seed=item.seed,
        )
        all_metric_rows.extend(metric_rows)
        all_group_rows.extend(group_rows)
        metadata_rows.extend(split_metadata)
        del train_frame, test_frame, metric_rows, group_rows, split_metadata
        _release_memory()
        _log(f"Released split data: {split.name} rss_mb={_rss_mb():.1f}")
    return all_metric_rows, all_group_rows, metadata_rows


def _load_split_frames(
    config: dict,
    *,
    seed: int,
    split: SplitPlan,
    states: list[str],
    download: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_contexts, test_contexts = _contexts_for_split(config, split, states)
    _log(
        f"Split {split.name}: train_contexts={len(train_contexts)} "
        f"test_contexts={len(test_contexts)}"
    )
    train_frame = load_folktables_contexts(
        config,
        seed=seed,
        task=split.task,
        contexts=train_contexts,
        download=download,
        max_total_rows=config["data"].get("max_train_rows_per_split"),
    )
    train_frame = _cap_split_rows(
        train_frame,
        config,
        max_rows=config["data"].get("max_train_rows_per_split"),
        seed=stable_seed(seed, split.name, "train_cap"),
        label="train",
        split_name=split.name,
    )
    _log(
        f"Split {split.name}: loaded train rows={len(train_frame):,} "
        f"rss_mb={_rss_mb():.1f}"
    )
    test_frame = load_folktables_contexts(
        config,
        seed=seed,
        task=split.task,
        contexts=test_contexts,
        download=download,
        max_total_rows=config["data"].get("max_test_rows_per_split"),
    )
    test_frame = _cap_split_rows(
        test_frame,
        config,
        max_rows=config["data"].get("max_test_rows_per_split"),
        seed=stable_seed(seed, split.name, "test_cap"),
        label="test",
        split_name=split.name,
    )
    if train_frame.empty or test_frame.empty:
        raise RuntimeError(f"Split {split.name} produced an empty train or test frame.")
    return train_frame.reset_index(drop=True), test_frame.reset_index(drop=True)


def _contexts_for_split(
    config: dict,
    split: SplitPlan,
    states: list[str],
) -> tuple[list[tuple[int, str]], list[tuple[int, str]]]:
    all_years = [int(year) for year in config["data"]["years"]]
    if split.mode == "temporal":
        if split.test_year is None:
            raise ValueError(f"Temporal split {split.name} is missing test_year.")
        train_years = list(split.train_years)
        test_years = [int(split.test_year)]
        train_states = states
        test_states = states
    elif split.mode == "geographic":
        if split.holdout_state is None:
            raise ValueError(f"Geographic split {split.name} is missing holdout_state.")
        train_years = all_years
        test_years = all_years
        train_states = [state for state in states if state != split.holdout_state]
        test_states = [split.holdout_state]
    elif split.mode == "geo_temporal":
        if split.holdout_state is None or split.test_year is None:
            raise ValueError(f"Geo-temporal split {split.name} is missing holdout state or test year.")
        train_years = list(split.train_years)
        test_years = [int(split.test_year)]
        train_states = [state for state in states if state != split.holdout_state]
        test_states = [split.holdout_state]
    else:
        raise ValueError(f"Unknown split mode: {split.mode}")

    train_contexts = [(year, state) for year in train_years for state in train_states]
    test_contexts = [(year, state) for year in test_years for state in test_states]
    return train_contexts, test_contexts


def _cap_split_rows(
    frame: pd.DataFrame,
    config: dict,
    *,
    max_rows: int | None,
    seed: int,
    label: str,
    split_name: str,
) -> pd.DataFrame:
    if not max_rows or len(frame) <= int(max_rows):
        return frame

    label_col = config["data"].get("label_column", "label")
    strata = [column for column in ("__state", "__year", label_col) if column in frame.columns]
    if not strata:
        sampled = frame.sample(n=int(max_rows), random_state=seed).reset_index(drop=True)
        _log(f"Split {split_name}: capped {label} rows {len(frame):,}->{len(sampled):,}")
        return sampled

    rng = pd.Series(range(len(frame))).sample(frac=1.0, random_state=seed).to_numpy()
    shuffled = frame.iloc[rng]
    parts: list[pd.DataFrame] = []
    for _group_key, group in shuffled.groupby(strata, sort=False, observed=True, dropna=False):
        take = max(1, int(round(int(max_rows) * len(group) / len(frame))))
        take = min(take, len(group))
        parts.append(group.head(take))

    sampled = pd.concat(parts, ignore_index=True, copy=False)
    if len(sampled) > int(max_rows):
        sampled = sampled.sample(n=int(max_rows), random_state=seed)
    sampled = sampled.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    _log(f"Split {split_name}: capped {label} rows {len(frame):,}->{len(sampled):,}")
    return sampled


def _run_methods_for_split(
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    split: SplitPlan,
    config: dict,
    *,
    seed: int,
) -> tuple[list[dict[str, Any]], list[pd.DataFrame], list[dict[str, Any]]]:
    metric_rows: list[dict[str, Any]] = []
    group_rows: list[pd.DataFrame] = []
    metadata_rows: list[dict[str, Any]] = []

    for regime in configured_regimes(config):
        _log(
            f"Split {split.name}: applying missingness regime={regime.name} "
            f"(mechanism={regime.mechanism}, scope={regime.scope})"
        )
        train_missing = apply_protected_missingness(
            train_frame,
            config["data"]["oracle_sensitive_attributes"],
            regime,
            seed=seed,
        )
        if regime.scope == "train_and_prediction":
            test_missing = apply_protected_missingness(
                test_frame,
                config["data"]["oracle_sensitive_attributes"],
                regime,
                seed=seed + 10_000,
            )
        else:
            test_missing = apply_protected_missingness(
                test_frame,
                config["data"]["oracle_sensitive_attributes"],
                regime.__class__(
                    name=f"{regime.name}_prediction_full",
                    mechanism="mcar",
                    availability=1.0,
                    scope=regime.scope,
                ),
                seed=seed + 10_000,
            )

        missingness_metadata = {
            **{f"train_{key}": value for key, value in train_missing.metadata.items()},
            **{f"prediction_{key}": value for key, value in test_missing.metadata.items()},
        }
        for method in config["methods"]["enabled"]:
            _log(
                f"Split {split.name}: method={method} regime={regime.name} "
                f"start rss_mb={_rss_mb():.1f}"
            )
            run_id = f"{split.name}__missing_{regime.name}__seed_{seed}__{method}"
            result = fit_predict_method(
                method,
                train_frame,
                test_frame,
                config,
                seed=seed,
                sensitive_train=train_missing.sensitive_observed,
                sensitive_test=test_missing.sensitive_observed,
                missingness_metadata=missingness_metadata,
            )
            protected_test = test_frame[config["data"]["protected_attributes"]]
            bundle = evaluate_predictions(
                test_frame[config["data"].get("label_column", "label")],
                result.y_pred,
                result.y_prob,
                protected_test,
                min_group_size=int(config["metrics"].get("min_group_size", 25)),
            )
            _log(
                f"Split {split.name}: method={method} regime={regime.name} "
                f"done fit_seconds={result.fit_seconds:.2f} "
                f"train_shape={result.metadata.get('train_matrix_shape')} "
                f"train_nnz={result.metadata.get('train_matrix_nnz')} "
                f"rss_mb={_rss_mb():.1f}"
            )
            metric_rows.append(
                {
                    "run_id": run_id,
                    "task": split.task,
                    "split_name": split.name,
                    "split_mode": split.mode,
                    "cluster_id": split.cluster_id,
                    "seed": int(seed),
                    "missingness_name": regime.name,
                    "missingness_mechanism": regime.mechanism,
                    "missingness_scope": regime.scope,
                    "target_availability": train_missing.metadata["target_availability"],
                    "prediction_target_availability": test_missing.metadata["target_availability"],
                    "train_sensitive_observed_fraction": train_missing.metadata["observed_fraction"],
                    "prediction_sensitive_observed_fraction": test_missing.metadata["observed_fraction"],
                    "method": method,
                    "fit_seconds": result.fit_seconds,
                    **bundle.metrics,
                }
            )
            group_metrics = None
            if not bundle.group_metrics.empty:
                group_metrics = bundle.group_metrics.copy()
                group_metrics.insert(0, "run_id", run_id)
                group_metrics.insert(1, "task", split.task)
                group_metrics.insert(2, "split_name", split.name)
                group_metrics.insert(3, "split_mode", split.mode)
                group_metrics.insert(4, "cluster_id", split.cluster_id)
                group_metrics.insert(5, "seed", int(seed))
                group_metrics.insert(6, "missingness_name", regime.name)
                group_metrics.insert(7, "missingness_mechanism", regime.mechanism)
                group_metrics.insert(8, "missingness_scope", regime.scope)
                group_metrics.insert(9, "method", method)
                group_rows.append(group_metrics)
            metadata_rows.append({"run_id": run_id, **result.metadata})
            del result, protected_test, bundle, group_metrics
            _release_memory()
        del train_missing, test_missing, missingness_metadata
        _release_memory()
        _log(f"Split {split.name}: finished regime={regime.name} rss_mb={_rss_mb():.1f}")

    return metric_rows, group_rows, metadata_rows


def _write_final_outputs(
    run_dir: Path,
    config: dict,
    *,
    config_path: str,
    metrics: pd.DataFrame,
    group_metrics: pd.DataFrame | None,
    metadata_rows: list[dict[str, Any]],
    item_count: int,
) -> Path:
    from fairmix_audit.reporting import write_model_cards, write_plots, write_tables

    metrics.to_csv(run_dir / "metrics.csv", index=False)
    if group_metrics is not None and not group_metrics.empty:
        group_metrics.to_csv(run_dir / "group_metrics.csv", index=False)
    pd.DataFrame(metadata_rows).to_json(
        run_dir / "model_metadata.jsonl", orient="records", lines=True
    )
    metadata_count = len(metadata_rows)

    from fairmix_audit.stats import (
        fairness_conclusion_flips,
        hidden_intersectional_regressions,
        mcar_vs_mnar_ablation,
        oracle_vs_missing_comparison,
        summarize_missingness_sensitivity,
    )

    summary = summarize_metric_deltas(metrics, config)
    summary.to_csv(run_dir / "metric_deltas_vs_erm.csv", index=False)
    flags = audit_flags(metrics, config)
    flags.to_csv(run_dir / "audit_flags.csv", index=False)
    sensitivity = summarize_missingness_sensitivity(metrics, config)
    sensitivity.to_csv(run_dir / "missingness_sensitivity.csv", index=False)
    oracle_missing = oracle_vs_missing_comparison(metrics, config)
    oracle_missing.to_csv(run_dir / "oracle_vs_missing.csv", index=False)
    conclusion_flips = fairness_conclusion_flips(metrics, config)
    conclusion_flips.to_csv(run_dir / "fairness_conclusion_flips.csv", index=False)
    hidden_regressions = hidden_intersectional_regressions(metrics, config)
    hidden_regressions.to_csv(run_dir / "hidden_intersectional_regressions.csv", index=False)
    mnar_ablation = mcar_vs_mnar_ablation(metrics, config)
    mnar_ablation.to_csv(run_dir / "mcar_vs_mnar_ablation.csv", index=False)
    write_tables(run_dir, metrics=metrics)
    write_plots(run_dir, metrics=metrics)
    write_model_cards(run_dir, metrics=metrics)

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "config_path": str(config_path),
        "n_work_items": int(item_count),
        "n_metric_rows": int(len(metrics)),
        "n_model_metadata_rows": int(metadata_count),
        "n_audit_flags": int(flags["unsafe_to_deploy"].sum()) if not flags.empty else 0,
        "artifacts": [
            "metrics.csv",
            "group_metrics.csv",
            "metric_deltas_vs_erm.csv",
            "audit_flags.csv",
            "missingness_sensitivity.csv",
            "oracle_vs_missing.csv",
            "fairness_conclusion_flips.csv",
            "hidden_intersectional_regressions.csv",
            "mcar_vs_mnar_ablation.csv",
            "plots/",
            "tables/",
            "model_cards/",
        ],
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return run_dir
