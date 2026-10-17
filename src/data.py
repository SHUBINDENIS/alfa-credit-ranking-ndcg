from pathlib import Path

import numpy as np
import pandas as pd

NUMERIC_FROM_STR = ["rate", "eva", "eva_perc", "ncl"]
CAT_STR = ["verif_compl", "offer_type", "risk_level_map", "verif_need", "need_2ndfl", "channel", "basket_name"]
JOIN_KEYS = ["app_id", "date_part"]


def load_raw(data_dir):
    data_dir = Path(data_dir)
    train = pd.read_parquet(data_dir / "train_dataset_small.pq")
    test = pd.read_parquet(data_dir / "test_dataset_small.pq")
    feats = pd.read_parquet(data_dir / "features_small.pq")
    return train, test, feats


def subsample_requests(df, frac, seed):
    """Keep a random fraction of applications, for quick smoke runs."""
    ids = df["request_id"].drop_duplicates()
    keep = ids.sample(frac=frac, random_state=seed)
    return df[df["request_id"].isin(keep)].reset_index(drop=True)


def _encode_client_features(feats):
    feats = feats.drop_duplicates(subset=JOIN_KEYS, keep="last").reset_index(drop=True)
    for col in feats.select_dtypes(include=["object"]).columns:
        if col in JOIN_KEYS:
            continue
        converted = pd.to_numeric(feats[col], errors="coerce")
        if converted.notna().sum() / max(feats[col].notna().sum(), 1) > 0.9:
            feats[col] = converted.astype("float32")
        else:
            feats[col] = pd.Categorical(feats[col].fillna("NA").astype(str)).codes.astype("int32")
    return feats


def preprocess(train, test, feats):
    """Cast Hive decimals, encode categoricals and join client features.

    Row order inside each file is kept in `__original_idx`. It is used only to
    restore order and is never a model feature: in train it leaks the target,
    in test it does not.
    """
    train = train.copy()
    test = test.copy()
    for df in (train, test):
        df["__original_idx"] = np.arange(len(df), dtype="int64")
        df["n_offers_in_req"] = df.groupby("request_id", sort=False)["request_id"].transform("size").astype("int32")
        for col in NUMERIC_FROM_STR:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float32")

    for col in CAT_STR:
        values = pd.concat([train[col], test[col]]).fillna("NA").astype(str)
        mapping = {v: i for i, v in enumerate(pd.Categorical(values).categories)}
        train[col] = train[col].fillna("NA").astype(str).map(mapping).astype("int32")
        test[col] = test[col].fillna("NA").astype(str).map(mapping).astype("int32")

    feats = _encode_client_features(feats)
    feats = feats.drop(columns=[c for c in feats.columns if c in train.columns and c not in JOIN_KEYS])
    train = train.merge(feats, on=JOIN_KEYS, how="left").sort_values("__original_idx").reset_index(drop=True)
    test = test.merge(feats, on=JOIN_KEYS, how="left").sort_values("__original_idx").reset_index(drop=True)
    return train, test
