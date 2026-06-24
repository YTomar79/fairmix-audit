from pathlib import Path

import pandas as pd
import pytest

from fairmix_audit.data import _cap_combined_context_rows
from fairmix_audit.experiments import (
    _cap_split_rows,
    build_split_plans,
    build_work_items,
    run_experiment,
)


def _config(tmp_path: Path) -> dict:
    return {
        "project": {
            "name": "workflow_test",
            "output_dir": str(tmp_path),
            "random_seeds": [11, 12],
        },
        "data": {
            "tasks": ["ACSIncome", "ACSEmployment"],
            "years": [2018, 2019, 2021],
            "states": ["CA", "TX"],
        },
        "splits": {
            "modes": ["temporal", "geographic", "geo_temporal"],
            "temporal": {
                "rolling": True,
                "min_train_years": 1,
            },
            "geographic": {
                "leave_one_state_out": True,
                "max_holdout_states": 1,
            },
        },
    }


def test_build_split_plans_counts_temporal_geographic_and_geo_temporal(tmp_path):
    config = _config(tmp_path)
    plans = build_split_plans(config)

    assert len(plans) == 10
    assert plans[0].name == "ACSIncome_temporal_2018_to_2019"
    assert any(plan.name == "ACSIncome_geographic_holdout_CA" for plan in plans)
    assert any(plan.name == "ACSEmployment_geo_temporal_holdout_CA_2018_2019_to_2021" for plan in plans)


def test_build_work_items_chunks_by_task_and_seed(tmp_path):
    config = _config(tmp_path)
    plans = build_split_plans(config)
    work_items = build_work_items(config, plans, chunk_size=2)

    assert len(work_items) == 12
    assert work_items[0].seed == 11
    assert work_items[0].task == "ACSIncome"
    assert len(work_items[0].splits) == 2
    assert work_items[-1].seed == 12
    assert work_items[-1].task == "ACSEmployment"
    assert len(work_items[-1].splits) == 1


def test_run_experiment_raises_when_no_splits_are_generated(monkeypatch, tmp_path):
    config = _config(tmp_path)
    config["splits"]["modes"] = []
    monkeypatch.setattr("fairmix_audit.experiments.load_config", lambda *_a, **_k: config)
    monkeypatch.setattr("fairmix_audit.experiments.write_yaml", lambda *_a, **_k: None)

    with pytest.raises(RuntimeError, match="No train/test splits"):
        run_experiment("configs/does_not_matter.yml")


def test_cap_split_rows_preserves_state_year_label_strata(tmp_path):
    config = _config(tmp_path)
    config["data"]["label_column"] = "label"
    rows = []
    for state in ["CA", "TX"]:
        for year in [2018, 2019]:
            for label in [0, 1]:
                for row_id in range(10):
                    rows.append(
                        {
                            "__state": state,
                            "__year": year,
                            "label": label,
                            "row_id": f"{state}-{year}-{label}-{row_id}",
                        }
                    )
    frame = pd.DataFrame(rows)

    capped = _cap_split_rows(
        frame,
        config,
        max_rows=24,
        seed=7,
        label="train",
        split_name="test_split",
    )

    assert len(capped) == 24
    assert set(capped["__state"]) == {"CA", "TX"}
    assert set(capped["__year"]) == {2018, 2019}
    assert set(capped["label"]) == {0, 1}


def test_cap_combined_context_rows_bounds_streaming_split_memory():
    frames = []
    for state in ["CA", "TX", "NY"]:
        rows = [
            {
                "__state": state,
                "__year": 2019,
                "label": row_id % 2,
                "row_id": f"{state}-{row_id}",
            }
            for row_id in range(30)
        ]
        frames.append(pd.DataFrame(rows))

    capped = _cap_combined_context_rows(
        frames,
        max_rows=36,
        seed=13,
        label_column="label",
    )

    assert len(capped) == 36
    assert set(capped["__state"]) == {"CA", "TX", "NY"}
    assert set(capped["label"]) == {0, 1}
