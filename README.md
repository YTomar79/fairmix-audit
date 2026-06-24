# How Reliable are Fairness Audits with Unreliable Data?

This repo accompanies "How Reliable are Fairness Audits with Unreliable Data?" (arXiv:2506.23033v4).

The pipeline shown here runs a fixed set of mitigation methods across many task/split/seed
combinations, withholds protected labels under several missingness regimes, and reports
how often the conclusion an auditor would draw actually flips, and is calibrated against a
seed-variation null.


## Repository layout

```
fairmix_audit/      Core library (data, methods, metrics, missingness, statistics, reporting)
configs/            Run configurations (smoke + full default)
scripts/            Thin CLI wrappers and post-processing helpers
tests/              Unit tests
docs/               Reproducibility checklist, model-card template, review map
results/            Compact summary tables and figures from the reference run
```



## Outputs

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

From the reference run (560 task/split/seed clusters).

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

## License

Released under the MIT License. See [LICENSE](LICENSE).
