from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class SplitSpec:
    name: str
    task: str
    mode: str
    train_mask: pd.Series
    test_mask: pd.Series
    cluster_id: str


def build_splits(frame: pd.DataFrame, config: dict) -> list[SplitSpec]:
    splits = []
    split_cfg = config["splits"]
    modes = split_cfg.get("modes", ["temporal"])

    for task in sorted(frame["__task"].unique()):
        task_mask = frame["__task"] == task
        if "temporal" in modes:
            temporal = split_cfg.get("temporal", {})
            for train_years, test_year in _temporal_specs(frame.loc[task_mask, "__year"], temporal):
                train_mask = task_mask & frame["__year"].isin(train_years)
                test_mask = task_mask & (frame["__year"] == test_year)
                if train_mask.any() and test_mask.any():
                    splits.append(
                        SplitSpec(
                            name=f"{task}_temporal_{'_'.join(map(str, sorted(train_years)))}_to_{test_year}",
                            task=task,
                            mode="temporal",
                            train_mask=train_mask,
                            test_mask=test_mask,
                            cluster_id=f"{task}:temporal:{test_year}",
                        )
                    )

        if "geographic" in modes and split_cfg.get("geographic", {}).get("leave_one_state_out", True):
            states = sorted(frame.loc[task_mask, "__state"].unique())
            max_holdouts = int(split_cfg.get("geographic", {}).get("max_holdout_states", len(states)))
            for state in states[:max_holdouts]:
                train_mask = task_mask & (frame["__state"] != state)
                test_mask = task_mask & (frame["__state"] == state)
                if train_mask.any() and test_mask.any():
                    splits.append(
                        SplitSpec(
                            name=f"{task}_geographic_holdout_{state}",
                            task=task,
                            mode="geographic",
                            train_mask=train_mask,
                            test_mask=test_mask,
                            cluster_id=f"{task}:geographic:{state}",
                        )
                    )

        if "geo_temporal" in modes:
            temporal = split_cfg.get("temporal", {})
            states = sorted(frame.loc[task_mask, "__state"].unique())
            max_holdouts = int(split_cfg.get("geographic", {}).get("max_holdout_states", len(states)))
            for train_years, test_year in _temporal_specs(frame.loc[task_mask, "__year"], temporal):
                for state in states[:max_holdouts]:
                    train_mask = task_mask & frame["__year"].isin(train_years) & (frame["__state"] != state)
                    test_mask = task_mask & (frame["__year"] == test_year) & (frame["__state"] == state)
                    if train_mask.any() and test_mask.any():
                        splits.append(
                            SplitSpec(
                                name=f"{task}_geo_temporal_holdout_{state}_{'_'.join(map(str, sorted(train_years)))}_to_{test_year}",
                                task=task,
                                mode="geo_temporal",
                                train_mask=train_mask,
                                test_mask=test_mask,
                                cluster_id=f"{task}:geo_temporal:{state}:{test_year}",
                            )
                        )
    return splits


def _temporal_specs(years: pd.Series, temporal_cfg: dict) -> list[tuple[set[int], int]]:
    available_years = sorted(int(year) for year in years.unique())
    if temporal_cfg.get("rolling", False):
        min_train_years = int(temporal_cfg.get("min_train_years", 1))
        specs = []
        for i in range(min_train_years, len(available_years)):
            specs.append((set(available_years[:i]), available_years[i]))
        return specs
    train_years = set(int(year) for year in temporal_cfg.get("train_years", []))
    test_year = int(temporal_cfg.get("test_year"))
    return [(train_years, test_year)]
