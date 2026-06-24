from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from fairmix_audit.data import combine_columns, stable_seed


MISSING_GROUP = "__MISSING_PROTECTED_ATTR__"


@dataclass(frozen=True)
class MissingnessRegime:
    name: str
    mechanism: str
    availability: float
    scope: str
    mnar_minority_multiplier: float = 0.25


@dataclass(frozen=True)
class MissingnessResult:
    sensitive_observed: pd.Series
    observed_mask: np.ndarray
    metadata: dict


def configured_regimes(config: dict) -> list[MissingnessRegime]:
    cfg = config.get("protected_missingness", {})
    regimes = cfg.get("regimes")
    if not regimes:
        return [MissingnessRegime(name="full", mechanism="mcar", availability=1.0, scope="train_only")]
    return [
        MissingnessRegime(
            name=str(regime["name"]),
            mechanism=str(regime.get("mechanism", "mcar")).lower(),
            availability=float(regime.get("availability", 1.0)),
            scope=str(regime.get("scope", "train_only")).lower(),
            mnar_minority_multiplier=float(regime.get("mnar_minority_multiplier", cfg.get("mnar_minority_multiplier", 0.25))),
        )
        for regime in regimes
    ]


def apply_protected_missingness(
    frame: pd.DataFrame,
    protected_attributes: list[str],
    regime: MissingnessRegime,
    *,
    seed: int,
) -> MissingnessResult:
    if not 0.0 <= regime.availability <= 1.0:
        raise ValueError(f"availability must be in [0, 1], got {regime.availability}")
    sensitive_full = combine_columns(frame, protected_attributes)
    n = len(frame)
    if n == 0:
        return MissingnessResult(pd.Series([], dtype=str), np.array([], dtype=bool), {})

    if regime.availability == 1.0:
        mask = np.ones(n, dtype=bool)
    elif regime.availability == 0.0:
        mask = np.zeros(n, dtype=bool)
    elif regime.mechanism == "mcar":
        mask = _mcar_mask(n, regime.availability, stable_seed(seed, regime.name, "mcar"))
    elif regime.mechanism == "mnar":
        mask = _mnar_mask(
            sensitive_full,
            regime.availability,
            regime.mnar_minority_multiplier,
            stable_seed(seed, regime.name, "mnar"),
        )
    else:
        raise ValueError(f"Unsupported protected missingness mechanism: {regime.mechanism}")

    observed = sensitive_full.astype(str).copy()
    observed.loc[~mask] = MISSING_GROUP
    metadata = {
        "missingness_name": regime.name,
        "missingness_mechanism": regime.mechanism,
        "missingness_scope": regime.scope,
        "target_availability": regime.availability,
        "observed_fraction": float(mask.mean()) if len(mask) else float("nan"),
        "n_observed": int(mask.sum()),
        "n_total": int(n),
    }
    return MissingnessResult(observed.reset_index(drop=True), mask, metadata)


def _mcar_mask(n: int, availability: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    target = int(round(n * availability))
    mask = np.zeros(n, dtype=bool)
    if target > 0:
        mask[rng.choice(n, size=target, replace=False)] = True
    return mask


def _mnar_mask(
    sensitive: pd.Series,
    availability: float,
    minority_multiplier: float,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    sensitive = sensitive.astype(str).reset_index(drop=True)
    majority = sensitive.value_counts().idxmax()
    weights = np.where(sensitive.to_numpy() == majority, 1.0, minority_multiplier)
    weights = weights / weights.mean()
    probs = np.clip(availability * weights, 0.0, 1.0)
    return rng.random(len(sensitive)) < probs
