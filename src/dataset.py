from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

DROP_COLS = ["app_id", "request_received", "date_part", "request_id", "offer_id", "is_deal", "nb_date", "__original_idx"]
CAT_FEATURES = ["variant_no", "verif_compl", "offer_type", "risk_level_map", "verif_need", "need_2ndfl", "channel",
                "basket_name", "pil1mtrx_offer"]


@dataclass
class RankingData:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: np.ndarray
    g_train: np.ndarray
    g_test: np.ndarray
    feature_cols: list
    cat_features: list
    splits: list
    train_keys: pd.DataFrame
    test_keys: pd.DataFrame

    def subset(self, columns):
        """Same rows and folds, fewer features."""
        cats = [c for c in self.cat_features if c in columns]
        return RankingData(self.X_train[columns], self.X_test[columns], self.y_train, self.g_train, self.g_test,
                           list(columns), cats, self.splits, self.train_keys, self.test_keys)


def make_dataset(train, test, n_splits=5):
    """Build model matrices and one fixed set of folds shared by every model.

    Rows are sorted by application so that groups are contiguous. Keys are kept
    next to the matrices: predictions saved by row order alone are easy to
    misalign after a reload.
    """
    feature_cols = [c for c in train.columns if c not in DROP_COLS and c in test.columns]
    cat_features = [c for c in CAT_FEATURES if c in feature_cols]

    train = train.sort_values(["request_id", "__original_idx"]).reset_index(drop=True)
    test = test.sort_values(["request_id", "__original_idx"]).reset_index(drop=True)

    X_train = train[feature_cols].copy()
    X_test = test[feature_cols].copy()
    for c in cat_features:
        X_train[c] = X_train[c].fillna(-1).astype("int32")
        X_test[c] = X_test[c].fillna(-1).astype("int32")

    y_train = train["is_deal"].to_numpy("int8")
    g_train = train["request_id"].to_numpy()
    g_test = test["request_id"].to_numpy()
    splits = list(GroupKFold(n_splits=n_splits).split(X_train, y_train, groups=g_train))

    return RankingData(
        X_train=X_train, X_test=X_test, y_train=y_train, g_train=g_train, g_test=g_test,
        feature_cols=feature_cols, cat_features=cat_features, splits=splits,
        train_keys=train[["request_id", "app_id", "variant_no"]].copy(),
        test_keys=test[["request_id", "app_id", "variant_no"]].copy(),
    )
