# Reproducibility Checklist

## Data

- [x] Public dataset source: Folktables ACS tasks.
- [x] Download scripted through the project data module.
- [x] Dataset cache path controlled by config.
- [x] Protected attributes declared in config.
- [x] Context attributes declared in config.
- [x] Protected-label missingness regimes declared in config.
- [x] Full-run dataset sizes recorded in the manuscript appendix and generated artifact manifest.

## Experiments

- [x] Headline run config: `configs/workshop_missingness_extensions.yml` (560 task/split/seed clusters, 10 non-full regimes, and 22,400 conclusion-selection decisions).
- [x] Do not use shorter smoke or tight-run configs to reproduce headline denominators.
- [x] All methods are selected through config.
- [x] Random seeds are listed in config.
- [x] Train/test split modes are listed in config.
- [x] Model feature exclusions are enforced in code.
- [x] Attribute-agnostic methods exclude protected attributes from model features.
- [x] Attribute-aware oracle methods write `uses_protected_attributes` metadata.
- [x] Attribute-aware methods receive only masked protected labels under missingness regimes.
- [x] Fairlearn reductions receive dense matrices derived from the same preprocessor.
- [x] ThresholdOptimizer receives sensitive features during prediction.

## Metrics and Statistics

- [x] Overall utility metrics are saved.
- [x] Protected-group and intersectional metrics are saved.
- [x] Paired deltas versus ERM are saved.
- [x] Bootstrap confidence intervals are saved.
- [x] Cluster bootstrap confidence intervals are saved.
- [x] Holm-corrected p-values are saved.
- [x] Unsafe deployment flags are saved.
- [x] Manuscript-ready plots are saved.
- [x] Per-run model cards are saved.

## Artifacts

- [x] `metrics.csv`
- [x] `group_metrics.csv`
- [x] `metric_deltas_vs_erm.csv`
- [x] `audit_flags.csv`
- [x] `model_metadata.jsonl`
- [x] `plots/`
- [x] `model_cards/`
- [x] `config.resolved.yml`
- [ ] Public GitHub repository URL.
- [ ] Archived release DOI or long-lived artifact URL.

## Manuscript

- [x] Use the NeurIPS-style workshop format for Reliable ML from Unreliable Data.
- [x] Remove identifying information from the paper, README, config names, and artifact manifest paths before release.
- [x] Include a completed NeurIPS-style checklist in the compiled PDF.
- [x] Review final prose, anonymization, and AI-assistance policy compliance before upload.
