# Model Card Template

## Model Details

- Method:
- Base estimator:
- Training task:
- Train split:
- Test split:
- Random seed:

## Intended Use

- Research use only.
- Audit of fairness behavior under protected-attribute missingness.
- Not intended for deployment decisions.

## Factors

- Protected attributes used for audit:
- Protected attributes used for training:
- Context attributes used for attribute-agnostic mixing:
- Intersectional groups audited:

## Metrics

- Accuracy:
- Balanced accuracy:
- Demographic parity difference:
- Equal opportunity difference:
- Equalized odds difference:
- Calibration gap:
- Worst-group accuracy:

## Ethical Caveats

- Utility improvement does not imply fairness improvement.
- Attribute-agnostic training still requires protected-attribute auditing where lawful and ethical.
- Mixing may obscure subgroup-specific harms.
- Do not deploy if `audit_flags.csv` marks the method as unsafe.
