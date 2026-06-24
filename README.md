# Seed-Calibrated Protected-Attribute Missingness Stress Tests

A reproducible workflow for stress-testing fairness audits when protected labels are
partially or fully missing. The central question: **does protected-label missingness
change an audit's recommendation more than ordinary complete-label seed variation
already does?**

The pipeline runs a fixed set of mitigation methods across many task/split/seed
combinations, withholds protected labels under several missingness regimes, and reports
how often the conclusion an auditor would draw actually flips — calibrated against a
seed-variation null.

## What it does

- **Data** — public [Folktables](https://github.com/socialfoundations/folktables) ACS
  tasks (`ACSIncome`, `ACSEmployment`, `ACSPublicCoverage`, `ACSMobility`), with race
  (`RAC1P`), sex (`SEX`), and their intersection as protected attributes.
- **Splits** — temporal, leave-one-state-out geographic, and combined geo-temporal
  generalization.
- **Missingness regimes** — full availability, MCAR partial-availability sweeps, matched
  MNAR ablations (non-majority intersectional cells more likely missing), and zero
  observed protected labels.
- **Methods** — ERM baseline, attribute-agnostic label-conditioned feature mixing,
  protected-attribute reweighing, protected-group mixing, and a Fairlearn equalized-odds
  threshold optimizer.
- **Reports** — utility and fairness metrics, paired and cluster bootstrap confidence
  intervals, Holm-corrected paired tests, seed-pair null calibration for conclusion
  flips, and deployment flags where aggregate accuracy rises while subgroup harm worsens.

## Repository layout

```
fairmix_audit/      Core library (data, methods, metrics, missingness, statistics, reporting)
configs/            Run configurations (smoke + full default)
scripts/            Thin CLI wrappers and post-processing helpers
tests/              Unit tests
docs/               Reproducibility checklist, model-card template, review map
results/            Compact summary tables and figures from the reference run
```

## Setup

Requires Python 3.11+.

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

## Quickstart

The smoke config downloads a small Folktables slice (ACSIncome, CA/TX, 2018–2019):

```bash
fairmix-run --config configs/smoke.yml
```

Each run writes a timestamped directory under `results/`. Regenerate compact tables and
figures from an existing run:

```bash
fairmix-tables results/<run-dir>
```

## Full run

```bash
fairmix-run --config configs/default.yml
```

The full configuration is intentionally heavier. Tune `max_rows_per_context`, `states`,
`years`, and `random_seeds` in `configs/default.yml` to match your compute budget;
`states: [ALL]` expands to the 50 states supported by the Folktables downloader.

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

## Reference results

From the reference run (560 task/split/seed clusters per regime). Conclusion-flip rate
is the share of audits whose accuracy-constrained equalized-odds recommendation changes
relative to the complete-label baseline:

| Protected-label availability | Conclusion flip rate |
| --- | --- |
| 0% (none observed) | 0.66 |
| 10% (MCAR) | 0.49 |
| 20% (MCAR) | 0.44 |
| 50% (MCAR) | 0.41 |

Full tables are in `results/tables/`.

## Configuration

Configs are YAML and control data selection, split modes, methods, missingness regimes,
model settings, statistics, and audit guardrails. `configs/smoke.yml` is a fast,
fully-specified example; `configs/default.yml` is the full experiment. Resolved configs
are written to each run directory as `config.resolved.yml` for provenance.

## Tests

```bash
pytest
```

## License

Released under the MIT License. See [LICENSE](LICENSE).
