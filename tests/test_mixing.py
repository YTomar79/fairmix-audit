import numpy as np
from scipy import sparse

from fairmix_audit.mixing import label_conditioned_context_mix


def test_label_conditioned_context_mix_creates_sparse_rows():
    X = sparse.csr_matrix(np.eye(6))
    y = np.array([0, 0, 0, 1, 1, 1])
    contexts = np.array(["a", "b", "c", "a", "b", "c"])
    X_aug, y_aug, report = label_conditioned_context_mix(
        X,
        y,
        contexts,
        mix_ratio=1.0,
        beta=0.4,
        min_contexts_per_label=2,
        seed=7,
    )
    assert X_aug.shape[0] == 12
    assert y_aug.shape[0] == 12
    assert report.created == 6


def test_label_conditioned_context_mix_skips_when_context_missing():
    X = np.eye(4)
    y = np.array([0, 0, 1, 1])
    contexts = np.array(["a", "a", "b", "b"])
    X_aug, y_aug, report = label_conditioned_context_mix(
        X,
        y,
        contexts,
        mix_ratio=1.0,
        beta=0.4,
        min_contexts_per_label=2,
        seed=7,
    )
    assert X_aug.shape[0] == 4
    assert y_aug.shape[0] == 4
    assert report.created == 0
