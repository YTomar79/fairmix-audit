# Review-to-Workflow Map

This file maps the major peer-review concerns to concrete workflow changes.

| Review concern | Implemented response |
| --- | --- |
| Bias was operationalized as MSE on tea prices | Replaced regression task with Folktables classification tasks and protected-group fairness metrics. |
| No protected attributes | Uses `RAC1P`, `SEX`, and race-by-sex intersections for evaluation. |
| Gaussian augmentation created pseudo-daily observations | Removed all time-series augmentation. Folktables data are downloaded directly from public ACS sources. |
| SMOTE was an inappropriate regression baseline | Removed SMOTE. Baselines now match fairness classification literature. |
| Reweighting was unspecified | Implements Kamiran-Calders style group-label reweighing with documented sample weights. |
| Feature-wise mixing was overclaimed as novel | Recasts it as label-conditioned cross-context interpolation and audits it against adjacent methods. |
| No rigorous statistics | Adds paired bootstrap confidence intervals, paired tests, and Holm correction. |
| Split-level observations are not independent | Adds cluster bootstrap intervals keyed by split cluster. |
| No reproducibility artifacts | Adds configs, pinned requirements, CLI scripts, resolved configs, result manifests, and model metadata. |
| Broader impact contradicted conclusion | Adds explicit unsafe-to-deploy flags where utility gains coincide with subgroup harm. |
| Protected attributes are not actually missing | Adds MCAR/MNAR protected-label missingness regimes; group-aware methods only receive masked labels. |
| Fairlearn baselines fail on sparse one-hot matrices | Converts Fairlearn inputs to dense matrices and handles ThresholdOptimizer prediction with sensitive features. |
| ThresholdOptimizer can fail on degenerate sensitive groups | Coalesces degenerate calibration groups deterministically and records this in metadata. |

The new central claim should be conditional:

> Attribute-agnostic feature mixing can sometimes improve cross-context utility, but it is not a substitute for protected-attribute auditing and can hide intersectional harms.
