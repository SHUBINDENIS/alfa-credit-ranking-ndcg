"""Second-level LambdaMART stacker.

Inputs are base model predictions turned into within-group z-scores and rank
percentiles, plus 20 original offer features. Raw scores of different models
live on different scales and shift between train and test; their position
inside the application does not.

The stacker is trained on the same folds as the base models. Each OOF value
was produced by a base model that never saw that application, so the stacker
learns how much to trust each model on unseen data. With different folds, the
stacker would see in-fold predictions and overfit to them.
"""
import lightgbm as lgb
import numpy as np
import pandas as pd

from .ranking_utils import group_sizes, rank_pct_within_group, zscore_within_group

TOP_FEATURES = [
    "eva", "eva_perc", "rate", "ncl", "limit", "term",
    "eva_rank", "rate_rank", "limit_rank", "ncl_rank",
    "eva_z", "rate_z",
    "eva_minus_ncl", "monthly_payment", "eva_div_limit", "limit_div_req",
    "variant_no", "offer_type", "basket_name", "risk_level_map",
]

STACKER_PARAMS = {
    "objective": "lambdarank",
    "metric": "ndcg",
    "eval_at": [5],
    "lambdarank_truncation_level": 8,
    "lambdarank_norm": True,
    "label_gain": [0, 1],
    "learning_rate": 0.02,
    "num_leaves": 31,
    "min_data_in_leaf": 200,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.7,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "verbose": -1,
}


def stack_features(base, data):
    """base: {name: (oof, test_pred)}. Original features come from the full feature set."""
    tr, te = {}, {}
    for name, (oof, tp) in base.items():
        tr[f"{name}_z"] = zscore_within_group(oof, data.g_train)
        tr[f"{name}_rank"] = rank_pct_within_group(oof, data.g_train)
        te[f"{name}_z"] = zscore_within_group(tp, data.g_test)
        te[f"{name}_rank"] = rank_pct_within_group(tp, data.g_test)
    X_tr = pd.DataFrame(tr)
    X_te = pd.DataFrame(te)
    for c in [c for c in TOP_FEATURES if c in data.X_train.columns]:
        X_tr[c] = data.X_train[c].to_numpy()
        X_te[c] = data.X_test[c].to_numpy()
    return X_tr, X_te


def _fit_one(X_tr, X_te, data, seed, max_rounds):
    oof = np.zeros(len(X_tr), dtype="float32")
    test_pred = np.zeros(len(X_te), dtype="float32")
    for fold, (tr, va) in enumerate(data.splits):
        dtr = lgb.Dataset(X_tr.iloc[tr], data.y_train[tr], group=group_sizes(data.g_train[tr]))
        dva = lgb.Dataset(X_tr.iloc[va], data.y_train[va], group=group_sizes(data.g_train[va]), reference=dtr)
        params = dict(STACKER_PARAMS, seed=seed + fold)
        model = lgb.train(params, dtr, num_boost_round=max_rounds, valid_sets=[dva],
                          callbacks=[lgb.early_stopping(150, verbose=False)])
        oof[va] = model.predict(X_tr.iloc[va])
        test_pred += model.predict(X_te) / len(data.splits)
    return oof, test_pred


def train_stacker(base, data, seeds=(42,), max_rounds=3000):
    """One seed returns raw scores; several seeds are averaged by rank percentile."""
    X_tr, X_te = stack_features(base, data)
    if len(seeds) == 1:
        return _fit_one(X_tr, X_te, data, seeds[0], max_rounds)
    oof = np.zeros(len(X_tr), dtype="float32")
    test_pred = np.zeros(len(X_te), dtype="float32")
    for seed in seeds:
        oof_s, tp_s = _fit_one(X_tr, X_te, data, seed, max_rounds)
        oof += rank_pct_within_group(oof_s, data.g_train) / len(seeds)
        test_pred += rank_pct_within_group(tp_s, data.g_test) / len(seeds)
    return oof, test_pred
