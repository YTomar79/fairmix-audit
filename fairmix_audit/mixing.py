from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse


@dataclass(frozen=True)
class MixingReport:
    requested: int
    created: int
    skipped_reason: str | None = None


def label_conditioned_context_mix(
    X,
    y,
    contexts,
    *,
    mix_ratio: float,
    beta: float,
    min_contexts_per_label: int,
    seed: int,
):
    """Create label-conditioned interpolations across different contexts."""
    y = np.asarray(y).reshape(-1)
    contexts = np.asarray(contexts).astype(str).reshape(-1)
    if len(y) != X.shape[0] or len(contexts) != X.shape[0]:
        raise ValueError("X, y, and contexts must have the same number of rows.")

    requested = int(round(X.shape[0] * mix_ratio))
    if requested <= 0:
        return X, y, MixingReport(requested=0, created=0, skipped_reason="mix_ratio <= 0")

    rng = np.random.default_rng(seed)
    labels = np.unique(y)
    label_context_indices: dict[object, dict[str, np.ndarray]] = {}
    eligible_labels = []
    for label in labels:
        label_mask = y == label
        context_map: dict[str, np.ndarray] = {}
        for context in np.unique(contexts[label_mask]):
            idx = np.flatnonzero(label_mask & (contexts == context))
            if len(idx):
                context_map[str(context)] = idx
        if len(context_map) >= min_contexts_per_label:
            label_context_indices[label] = context_map
            eligible_labels.append(label)

    if not eligible_labels:
        return X, y, MixingReport(
            requested=requested,
            created=0,
            skipped_reason="no label has enough distinct contexts",
        )

    label_counts = np.array([np.sum(y == label) for label in eligible_labels], dtype=float)
    label_probs = label_counts / label_counts.sum()

    mixed_rows = []
    mixed_labels = []
    for _ in range(requested):
        label = rng.choice(eligible_labels, p=label_probs)
        context_map = label_context_indices[label]
        c1, c2 = rng.choice(list(context_map), size=2, replace=False)
        i = rng.choice(context_map[str(c1)])
        j = rng.choice(context_map[str(c2)])
        lam = float(rng.beta(beta, beta))
        mixed_rows.append((X[i] * lam) + (X[j] * (1.0 - lam)))
        mixed_labels.append(label)

    if sparse.issparse(X):
        X_mix = sparse.vstack(mixed_rows, format="csr")
        X_aug = sparse.vstack([X, X_mix], format="csr")
    else:
        X_mix = np.vstack(mixed_rows)
        X_aug = np.vstack([X, X_mix])
    y_aug = np.concatenate([y, np.asarray(mixed_labels, dtype=y.dtype)])
    return X_aug, y_aug, MixingReport(requested=requested, created=len(mixed_labels))
