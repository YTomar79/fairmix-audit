# How Reliable are Fairness Audits with Unreliable Data?

**Yash Vardhan Tomar**¹

¹ Purdue University  

[![arXiv](https://img.shields.io/badge/arXiv-2506.23033-b31b1b.svg)](https://arxiv.org/abs/2506.23033)

This repo accompanies and contains the code, raw CSVs, and results for "How Reliable are Fairness Audits with Unreliable Data?" (arXiv:2506.23033v4).

## Abstract

Fairness audits are a key component of responsible machine-learning deployment. Yet, audit-recommendation reliability under incomplete protected-label access is still poorly understood. In this work, we focused on protected-label missingness in fairness mitigation audits. We introduced a seed-calibrated stress test to separate missingness effects from seed-to-seed movement already present under complete labels. Across ACS/Folktables tasks, missingness settings that retain some protected labels usually do not move selected mitigation methods beyond a complete-label seed-to-seed baseline. At 0 protected-label access, candidates collapse to an empirical-risk-minimization baseline and deterministic tie-breaking rather than revealing a broad missingness effect. We also found that threshold optimization can turn fairness gains on a single protected axis into intersectional harm above a seed baseline, and this threshold-optimizer finding persists under random-forest validation. Overall, our results highlight that protected-label missingness should be reported with seed-null calibration, candidate-set context, and intersectional consequences before it is treated as evidence of audit fragility.

## Repository layout

```
fairmix-audit/
├── fairmix_audit/                 Core library
│   ├── cli.py                     Console entry points (fairmix-run, fairmix-tables)
│   ├── experiments.py             Workflow orchestration: split plans, work items, run loop
│   ├── config.py                  YAML loading, default merging, resolved-config writing
│   ├── data.py                    Folktables ACS loading, context slicing, state resolution
│   ├── splits.py                  Temporal / geographic / geo-temporal split construction
│   ├── missingness.py             Protected-label missingness regimes (MCAR, MNAR, none)
│   ├── mixing.py                  Label-conditioned cross-context feature mixing
│   ├── baselines.py               Methods: ERM, mixing, reweighing, group mixing, Fairlearn EO
│   ├── metrics.py                 Utility and group-fairness metric computation
│   ├── stats.py                   Bootstrap CIs, Holm-corrected tests, conclusion-flip calibration
│   ├── reporting.py               Tables, plots, and per-run model cards
│   └── memory.py                  Lightweight RSS sampling / memory-release helpers
│
├── configs/
│   ├── smoke.yml                  Fast, fully-specified example (small ACS slice)
│   └── default.yml                Full experiment configuration
│
├── scripts/
│   ├── run_experiments.py         Wrapper around the fairmix-run entry point
│   ├── make_tables.py             Wrapper around the fairmix-tables entry point
│   └── compare_base_learners.py   Post-processing: compare diagnostics across base learners
│
├── tests/                         Unit tests (planning, data, methods, metrics, stats, reporting)
│
├── docs/
│   ├── reproducibility_checklist.md
│   ├── model_card_template.md
│   └── review_to_workflow_map.md  Maps reviewer concerns to workflow components
│
├── results/                       Compact artifacts from the reference run
│   ├── tables/                    Summary CSVs (utility, fairness, flips, sensitivity, ...)
│   ├── plots/                     Figures (accuracy, EO gaps, sensitivity curves, ...)
│   ├── metric_deltas_vs_erm.csv   Paired deltas vs. the ERM baseline
│   └── missingness_sensitivity.csv
│
├── pyproject.toml                 Package metadata, dependencies, entry points
├── requirements.txt               Pinned dependency versions
├── Makefile                       setup / smoke / tables / test shortcuts
└── LICENSE                        MIT
```

## Reproducing the experiments

Requires Python 3.11+. The steps below take you from a clean checkout to the full set of
result tables and figures.

**1. Clone the repository**

```bash
git clone https://github.com/<your-username>/fairmix-audit.git
cd fairmix-audit
```

**2. Create an environment**

```bash
conda create -y -n fairmix python=3.11
conda activate fairmix
```

Any virtual environment works; conda is shown for convenience.

**3. Install the package**

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

This installs the pinned dependencies in `pyproject.toml` and the `fairmix-run` /
`fairmix-tables` console commands.

**4. Run the smoke test (≈ minutes, downloads a small ACS slice)**

```bash
fairmix-run --config configs/smoke.yml
```

This validates the full pipeline end to end on ACSIncome, CA/TX, 2018–2019. It writes a
timestamped run directory under `results/`, e.g. `results/smoke_attribute_missingness_audit_<timestamp>/`.

**5. Run the full experiment (heavy; downloads all configured ACS slices)**

```bash
fairmix-run --config configs/default.yml
```

`configs/default.yml` enumerates all four ACS tasks, all split modes, every missingness
regime, and five seeds. Tune `max_rows_per_context`, `states`, `years`, and
`random_seeds` to match your compute budget; `states: [ALL]` expands to the 50 states
supported by the Folktables downloader. Folktables data is cached under
`data/raw/folktables`, so reruns do not re-download.

**6. (Optional) Regenerate tables and figures from an existing run**

```bash
fairmix-tables results/<run-dir>
```

Re-derives every summary table, plot, and model card from that run's `metrics.csv`
without re-running the experiment.

**7. Run the test suite**

```bash
pytest
```

## Outputs

Each run directory contains:

| File | Contents |
| --- | --- |
| `metrics.csv` | One row per method / split / task / seed / missingness regime |
| `group_metrics.csv` | Subgroup-level rates and errors |
| `metric_deltas_vs_erm.csv` | Paired deltas, bootstrap CIs, Holm-corrected p-values |
| `missingness_sensitivity.csv` | Utility and fairness vs. label availability |
| `oracle_vs_missing.csv` | Method deltas vs. the full-label oracle |
| `fairness_conclusion_flips.csv` | Whether the auditor's selected method changes under missingness |
| `hidden_intersectional_regressions.csv` | Single-axis gains that hide intersectional regressions |
| `mcar_vs_mnar_ablation.csv` | Matched MCAR-vs-MNAR deltas at equal availability |
| `audit_flags.csv` | Method/regime cases that are unsafe under the configured guardrails |
| `tables/`, `plots/`, `model_cards/` | Compact summaries, figures, and per-run model cards |


## Configuration

Configs are YAML and control data selection, split modes, methods, missingness regimes,
model settings, statistics, and audit guardrails. `configs/smoke.yml` is a fast,
fully-specified example; `configs/default.yml` is the full experiment. Resolved configs
are written to each run directory as `config.resolved.yml` for provenance.


## Citation

If you use this code or find this work helpful, please cite:

```bibtex
@article{tomar2025fairness,
  title={How Reliable are Fairness Audits with Unreliable Data?},
  author={Tomar, Yash Vardhan},
  journal={arXiv preprint arXiv:2506.23033},
  year={2025}
}
```

## License

Released under the [MIT License](LICENSE).
