import numpy as np
import pandas as pd

from fairmix_audit.metrics import evaluate_predictions


def test_evaluate_predictions_returns_intersectional_metrics():
    y_true = np.array([1, 1, 0, 0, 1, 0, 1, 0])
    y_pred = np.array([1, 0, 0, 0, 1, 1, 1, 0])
    y_prob = np.array([0.9, 0.4, 0.2, 0.1, 0.8, 0.7, 0.9, 0.3])
    protected = pd.DataFrame(
        {
            "RAC1P": [1, 1, 1, 2, 2, 2, 2, 1],
            "SEX": [1, 2, 1, 2, 1, 2, 1, 2],
        }
    )
    bundle = evaluate_predictions(y_true, y_pred, y_prob, protected, min_group_size=1)
    assert "accuracy" in bundle.metrics
    assert "rac1p_equalized_odds_diff" in bundle.metrics
    assert "rac1p_x_sex_worst_group_accuracy" in bundle.metrics
    assert set(bundle.group_metrics["attribute"]) == {"RAC1P", "SEX", "RAC1P_X_SEX"}
