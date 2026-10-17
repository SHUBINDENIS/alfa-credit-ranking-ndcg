"""Base ranking models trained on shared folds.

Every function returns (oof, test_pred): out-of-fold scores for train and the
fold-averaged score for test. Parameters match the competition runs.
"""
import lightgbm as lgb
import numpy as np
import xgboost as xgb
from catboost import CatBoostRanker, Pool

from .ranking_utils import group_sizes, rank_pct_within_group

LGB_PARAMS = {
    "objective": "lambdarank",
    "metric": "ndcg",
    "eval_at": [5],
    "lambdarank_truncation_level": 8,
    "lambdarank_norm": True,
    "label_gain": [0, 1],
    "learning_rate": 0.03,
    "num_leaves": 127,
    "min_data_in_leaf": 100,
    "feature_fraction": 0.8,
    "feature_fraction_bynode": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "lambda_l1": 0.1,
    "lambda_l2": 1.0,
    "verbose": -1,
}

XGB_PARAMS = {
    "objective": "rank:pairwise",
    "eval_metric": "ndcg@5",
    "learning_rate": 0.03,
    "max_depth": 8,
    "min_child_weight": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "tree_method": "hist",
    "verbosity": 0,
}

CB_YETI_V4 = {"iterations": 2500, "learning_rate": 0.05, "depth": 8, "l2_leaf_reg": 3.0, "max_ctr_complexity": 2}
CB_YETI_V5 = {"iterations": 4000, "learning_rate": 0.03, "depth": 6, "l2_leaf_reg": 3.0}
CB_SEEDBAG = {"iterations": 4000, "learning_rate": 0.05, "depth": 6, "l2_leaf_reg": 3.0}


def train_lgb(data, seed=42, max_rounds=5000, log_period=0):
    oof = np.zeros(len(data.X_train), dtype="float32")
    test_pred = np.zeros(len(data.X_test), dtype="float32")
    for fold, (tr, va) in enumerate(data.splits):
        dtr = lgb.Dataset(data.X_train.iloc[tr], data.y_train[tr], group=group_sizes(data.g_train[tr]),
                          categorical_feature=data.cat_features)
        dva = lgb.Dataset(data.X_train.iloc[va], data.y_train[va], group=group_sizes(data.g_train[va]),
                          categorical_feature=data.cat_features, reference=dtr)
        callbacks = [lgb.early_stopping(200, verbose=False)]
        if log_period:
            callbacks.append(lgb.log_evaluation(log_period))
        model = lgb.train(dict(LGB_PARAMS, seed=seed + fold), dtr, num_boost_round=max_rounds, valid_sets=[dva],
                          callbacks=callbacks)
        oof[va] = model.predict(data.X_train.iloc[va])
        test_pred += model.predict(data.X_test) / len(data.splits)
    return oof, test_pred


def train_xgb(data, seed=42, max_rounds=5000, device="cuda"):
    X_tr_all = data.X_train.fillna(-999.0).astype("float32")
    dtest = xgb.DMatrix(data.X_test.fillna(-999.0).astype("float32"))
    oof = np.zeros(len(X_tr_all), dtype="float32")
    test_pred = np.zeros(dtest.num_row(), dtype="float32")
    for fold, (tr, va) in enumerate(data.splits):
        dtr = xgb.DMatrix(X_tr_all.iloc[tr], label=data.y_train[tr], group=group_sizes(data.g_train[tr]))
        dva = xgb.DMatrix(X_tr_all.iloc[va], label=data.y_train[va], group=group_sizes(data.g_train[va]))
        params = dict(XGB_PARAMS, seed=seed + fold, device=device)
        model = xgb.train(params, dtr, num_boost_round=max_rounds, evals=[(dva, "va")], early_stopping_rounds=200,
                          verbose_eval=False)
        oof[va] = model.predict(dva, iteration_range=(0, model.best_iteration + 1))
        test_pred += model.predict(dtest, iteration_range=(0, model.best_iteration + 1)) / len(data.splits)
    return oof, test_pred


def _cb_folds(data, params, seed, task_type, overrides=None, fixed_seed=False):
    cat_idx = [data.feature_cols.index(c) for c in data.cat_features]
    test_pool = Pool(data.X_test, cat_features=cat_idx)
    oof = np.zeros(len(data.X_train), dtype="float32")
    test_pred = np.zeros(len(data.X_test), dtype="float32")
    params = dict(params, **(overrides or {}))
    for fold, (tr, va) in enumerate(data.splits):
        tr_pool = Pool(data.X_train.iloc[tr], data.y_train[tr], group_id=data.g_train[tr], cat_features=cat_idx)
        va_pool = Pool(data.X_train.iloc[va], data.y_train[va], group_id=data.g_train[va], cat_features=cat_idx)
        model = CatBoostRanker(
            loss_function="YetiRankPairwise",
            eval_metric="NDCG:top=5;type=Exp",
            metric_period=200,
            early_stopping_rounds=400,
            random_seed=seed if fixed_seed else seed + fold,
            task_type=task_type,
            verbose=0,
            **params,
        )
        model.fit(tr_pool, eval_set=va_pool)
        oof[va] = model.predict(va_pool)
        test_pred += model.predict(test_pool) / len(data.splits)
    return oof, test_pred


def train_catboost(data, params, seed=42, task_type="GPU", overrides=None):
    """CatBoost YetiRankPairwise with native ordered target encoding of categoricals."""
    return _cb_folds(data, params, seed, task_type, overrides)


def train_catboost_seedbag(data, seeds=(42, 101, 202, 303, 404), task_type="GPU", overrides=None):
    """Average of several CatBoost runs, combined by within-group rank percentile."""
    oof = np.zeros(len(data.X_train), dtype="float32")
    test_pred = np.zeros(len(data.X_test), dtype="float32")
    for seed in seeds:
        oof_s, tp_s = _cb_folds(data, CB_SEEDBAG, seed, task_type, overrides, fixed_seed=True)
        oof += rank_pct_within_group(oof_s, data.g_train) / len(seeds)
        test_pred += rank_pct_within_group(tp_s, data.g_test) / len(seeds)
    return oof, test_pred
