import numpy as np
import pandas as pd


def ndcg_at_k(scores, groups, y_true, k=5):
    """One-positive NDCG@k averaged over groups, vectorized."""
    scores = np.asarray(scores, dtype="float64")
    groups = np.asarray(groups)
    y_true = np.asarray(y_true)
    order = np.lexsort((-scores, groups))
    g_sorted = groups[order]
    y_sorted = y_true[order]
    n = len(g_sorted)
    starts = np.r_[0, np.flatnonzero(g_sorted[1:] != g_sorted[:-1]) + 1]
    sizes = np.diff(np.r_[starts, n])
    pos_in_group = np.arange(n) - np.repeat(starts, sizes)
    ranks = pos_in_group[y_sorted == 1]
    gains = np.where(ranks < k, 1.0 / np.log2(ranks + 2.0), 0.0)
    return float(gains.mean()) if gains.size else 0.0


def rank_pct_within_group(scores, groups):
    return pd.Series(scores).groupby(groups, sort=False).rank(method="average", pct=True).to_numpy("float32")


def zscore_within_group(scores, groups):
    s = pd.Series(scores)
    g = s.groupby(groups, sort=False)
    std = g.transform("std").fillna(1).replace(0, 1)
    return ((s - g.transform("mean")) / std).to_numpy("float32")


def rank_blend(predictions, weights, groups):
    total = np.zeros(len(groups), dtype="float64")
    for pred, w in zip(predictions, weights):
        total += w * rank_pct_within_group(pred, groups)
    return total / sum(weights)


def group_sizes(groups):
    """Sizes of consecutive groups, as LightGBM and XGBoost expect."""
    return pd.Series(groups).groupby(groups, sort=False).size().to_numpy()
