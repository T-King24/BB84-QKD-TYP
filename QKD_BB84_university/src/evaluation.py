"""Shared evaluation helpers: recall on subsets of attacks (stealthy, per attack type)."""
import numpy as np
import pandas as pd


def recall_on_subset(preds, y_true, mask):
    # isolate predictions and ground truths for the specified subset
    # prevent division by zero if there are no attacks in the subset
    if mask.sum() == 0:
        return np.nan, 0
    return (preds[mask] == 1).mean(), int(mask.sum())  # recall and number of samples


def per_attack_recall(preds, y_true, attack_type, stealthy):
    """Recall for every (attack type, stealthy/naive) combination present."""
    rows = []
    for a in sorted(set(attack_type[y_true == 1])):
        for st in (False, True):
            r, n = recall_on_subset(preds, y_true, (y_true == 1) & (attack_type == a)
                                    & (stealthy == st))
            if n:
                rows.append({"attack_type": a, "stealthy": st, "recall": r, "n": n})
    return pd.DataFrame(rows)


def false_positive_rate(preds, y_true):
    normal = y_true == 0
    return float((preds[normal] == 1).mean()) if normal.any() else np.nan
