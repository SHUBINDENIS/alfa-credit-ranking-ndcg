import numpy as np
import pandas as pd

LISTWISE_BASE = ["rate", "eva", "eva_perc", "ncl", "limit", "term"]
DOMAIN_RANKED = ["eva_minus_ncl", "monthly_payment", "eva_div_limit", "limit_div_req", "rate_times_term"]
COUNT_COLS = ["variant_no", "basket_name", "offer_type", "risk_level_map", "offer_id"]
COUNT_PAIRS = [("variant_no", "offer_type"), ("variant_no", "risk_level_map")]


def add_listwise(df, group_col="request_id"):
    """Compare each offer with the other offers of the same application."""
    g = df.groupby(group_col, sort=False)
    new = {}
    for col in LISTWISE_BASE:
        g_min = g[col].transform("min")
        g_max = g[col].transform("max")
        g_std = g[col].transform("std")
        new[f"{col}_rank"] = g[col].rank(method="dense", pct=True)
        new[f"{col}_z"] = (df[col] - g[col].transform("mean")) / g_std.replace(0, 1)
        new[f"{col}_minus_min"] = df[col] - g_min
        new[f"{col}_minus_max"] = df[col] - g_max
        new[f"{col}_minus_median"] = df[col] - g[col].transform("median")
        new[f"{col}_to_min"] = df[col] / g_min.replace(0, np.nan)
        new[f"{col}_to_max"] = df[col] / g_max.replace(0, np.nan)
        new[f"{col}_range"] = g_max - g_min
        new[f"{col}_std"] = g_std
        new[f"is_min_{col}"] = (df[col] == g_min).astype("int8")
        new[f"is_max_{col}"] = (df[col] == g_max).astype("int8")
    out = pd.DataFrame(new, index=df.index)
    float_cols = [c for c in out.columns if not c.startswith("is_")]
    out[float_cols] = out[float_cols].astype("float32")
    return pd.concat([df, out], axis=1)


def add_domain(df, group_col="request_id"):
    """Loan economics: annuity payment, total interest, fit to the requested loan."""
    rate_m = df["rate"] / 12 / 100
    n = df["term"].astype("float32")
    payment = df["limit"] * rate_m / (1 - (1 + rate_m).pow(-n) + 1e-9)
    new = {
        "eva_minus_ncl": df["eva"] - df["ncl"],
        "eva_div_ncl": df["eva"] / (df["ncl"].abs() + 1e-9),
        "monthly_payment": payment,
        "total_interest": payment * n - df["limit"],
        "eva_div_limit": df["eva"] / (df["limit"] + 1e-9),
        "limit_div_req": df["limit"] / df["req_loan_amount"].replace(0, np.nan),
        "term_div_req": df["term"] / df["req_term"].replace(0, np.nan),
        "limit_minus_req": df["limit"] - df["req_loan_amount"],
        "term_minus_req": df["term"] - df["req_term"],
        "rate_times_term": df["rate"] * df["term"],
    }
    if "profit_income_out_rur_amt_cm" in df.columns:
        new["payment_to_income"] = payment / (df["profit_income_out_rur_amt_cm"].abs() + 1e-9)
    if "age" in df.columns:
        new["term_to_age_left"] = df["term"] / (12 * (70 - df["age"]).clip(1, 70))
    out = pd.DataFrame(new, index=df.index).astype("float32")
    df = pd.concat([df, out], axis=1)
    g = df.groupby(group_col, sort=False)
    ranked = {f"{c}_rank": g[c].rank(method="dense", pct=True).astype("float32") for c in DOMAIN_RANKED}
    return pd.concat([df, pd.DataFrame(ranked, index=df.index)], axis=1)


def add_count_encoding(train, test):
    """Frequency of categorical values over train and test together."""
    combo = pd.concat([train[COUNT_COLS], test[COUNT_COLS]], axis=0)
    for col in COUNT_COLS:
        counts = combo[col].value_counts()
        train[f"{col}_cnt"] = train[col].map(counts).astype("float32")
        test[f"{col}_cnt"] = test[col].map(counts).astype("float32")
    for c1, c2 in COUNT_PAIRS:
        k_tr = train[c1].astype(str) + "_" + train[c2].astype(str)
        k_te = test[c1].astype(str) + "_" + test[c2].astype(str)
        counts = pd.concat([k_tr, k_te]).value_counts()
        train[f"{c1}_{c2}_cnt"] = k_tr.map(counts).astype("float32")
        test[f"{c1}_{c2}_cnt"] = k_te.map(counts).astype("float32")
    return train, test


def build_features(train, test):
    train = add_domain(add_listwise(train))
    test = add_domain(add_listwise(test))
    return add_count_encoding(train, test)
