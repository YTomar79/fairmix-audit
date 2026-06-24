# How Reliable are Fairness Audits with Unreliable Data?

**Yash Vardhan Tomar**¹

¹ Purdue University  

[![arXiv](https://img.shields.io/badge/arXiv-2506.23033-b31b1b.svg)](https://arxiv.org/abs/2506.23033)

This repo accompanies and contains the code, raw CSVs, and results for "How Reliable are Fairness Audits with Unreliable Data?" (arXiv:2506.23033v4).

## Abstract

Fairness audits are a key component of responsible machine-learning deployment. Yet, audit-recommendation reliability under incomplete protected-label access is still poorly understood. In this work, we focused on protected-label missingness in fairness mitigation audits. We introduced a seed-calibrated stress test to separate missingness effects from seed-to-seed movement already present under complete labels. Across ACS/Folktables tasks, missingness settings that retain some protected labels usually do not move selected mitigation methods beyond a complete-label seed-to-seed baseline. At 0 protected-label access, candidates collapse to an empirical-risk-minimization baseline and deterministic tie-breaking rather than revealing a broad missingness effect. We also found that threshold optimization can turn fairness gains on a single protected axis into intersectional harm above a seed baseline, and this threshold-optimizer finding persists under random-forest validation. Overall, our results highlight that protected-label missingness should be reported with seed-null calibration, candidate-set context, and intersectional consequences before it is treated as evidence of audit fragility.

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
