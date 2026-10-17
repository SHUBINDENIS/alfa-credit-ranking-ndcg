import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


def adversarial_drift(X_train, X_test, seed=42):
    """Train a classifier to tell train rows from test rows.

    Returns the hold-out AUC and feature importances. AUC near 0.5 means the
    two samples look alike; AUC near 1.0 means cross-validation on train says
    little about the test set.
    """
    X = pd.concat([X_train, X_test], axis=0, ignore_index=True).fillna(-1)
    y = np.r_[np.zeros(len(X_train)), np.ones(len(X_test))].astype("int8")
    X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)
    clf = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=63, min_child_samples=200,
                             random_state=seed, verbose=-1)
    clf.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], callbacks=[lgb.early_stopping(30, verbose=False)])
    auc = roc_auc_score(y_va, clf.predict_proba(X_va)[:, 1])
    importance = pd.Series(clf.feature_importances_, index=X_train.columns).sort_values(ascending=False)
    return auc, importance


def drifting_features(importance, protected, top_k=5):
    """Top drifting features, except raw categoricals that CatBoost encodes itself."""
    return [c for c in importance.index[:top_k] if c not in set(protected)]
