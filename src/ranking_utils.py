import numpy as np
import pandas as pd


def ndcg_at_k_grouped(scores, groups, y_true, k=5):
    df = pd.DataFrame({"s": scores, "g": groups, "y": y_true})
    values = []
    for _, sub in df.groupby("g", sort=False):
        if sub["y"].sum() == 0:
            continue
        order = np.argsort(-sub["s"].to_numpy())
        hits = np.flatnonzero(sub["y"].to_numpy()[order] == 1)
        rank = hits[0]
        values.append(0.0 if rank >= k else 1.0 / np.log2(rank + 2))
    return float(np.mean(values))


def rank_pct_within_group(scores, groups):
    return pd.Series(scores).groupby(groups, sort=False).rank(method="average", pct=True).to_numpy()


def zscore_within_group(scores, groups):
    s = pd.Series(scores)
    g = s.groupby(groups, sort=False)
    std = g.transform("std").replace(0, 1).fillna(1)
    return ((s - g.transform("mean")) / std).to_numpy()


def rank_blend(predictions, weights, groups):
    total = np.zeros(len(groups))
    for pred, w in zip(predictions, weights):
        total += w * rank_pct_within_group(pred, groups)
    return total / sum(weights)


LISTWISE_BASE = ["rate", "eva", "eva_perc", "ncl", "limit", "term"]


def add_listwise_features(df, group_col="request_id", cols=LISTWISE_BASE):
    g = df.groupby(group_col, sort=False)
    for col in cols:
        g_min = g[col].transform("min")
        g_max = g[col].transform("max")
        df[f"{col}_rank"] = g[col].rank(method="dense", pct=True).astype("float32")
        df[f"{col}_z"] = ((df[col] - g[col].transform("mean")) / g[col].transform("std").replace(0, 1)).astype("float32")
        df[f"{col}_minus_min"] = (df[col] - g_min).astype("float32")
        df[f"{col}_minus_max"] = (df[col] - g_max).astype("float32")
        df[f"{col}_minus_median"] = (df[col] - g[col].transform("median")).astype("float32")
        df[f"{col}_range"] = (g_max - g_min).astype("float32")
        df[f"is_min_{col}"] = (df[col] == g_min).astype("int8")
        df[f"is_max_{col}"] = (df[col] == g_max).astype("int8")
    df["n_variants_in_req"] = g[group_col].transform("size").astype("int32")
    return df


def add_domain_features(df):
    rate_m = df["rate"] / 12 / 100
    n = df["term"].astype("float32")
    df["monthly_payment"] = (df["limit"] * rate_m / (1 - (1 + rate_m).pow(-n) + 1e-9)).astype("float32")
    df["total_interest"] = (df["monthly_payment"] * n - df["limit"]).astype("float32")
    df["eva_minus_ncl"] = (df["eva"] - df["ncl"]).astype("float32")
    df["eva_div_limit"] = (df["eva"] / (df["limit"] + 1e-9)).astype("float32")
    df["limit_div_req"] = (df["limit"] / df["req_loan_amount"].replace(0, np.nan)).astype("float32")
    df["term_div_req"] = (df["term"] / df["req_term"].replace(0, np.nan)).astype("float32")
    df["rate_times_term"] = (df["rate"] * df["term"]).astype("float32")
    return df
