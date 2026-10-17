"""End-to-end final solution: features, base models, stackers, blend.

    python -m src.pipeline --data-dir data --out artifacts
    python -m src.pipeline --data-dir data --out artifacts_smoke --sample-frac 0.01 --device cpu --fast

Each stage saves its OOF and test predictions to --out and is skipped on the
next run if the files exist, so a long GPU run can be resumed.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from . import models
from .data import load_raw, preprocess, subsample_requests
from .dataset import make_dataset
from .drift import adversarial_drift, drifting_features
from .features import build_features
from .mlp import train_mlp
from .ranking_utils import ndcg_at_k, rank_blend
from .stacker import train_stacker

DRIFT_FEATURES = ["max_grace_days", "salary_last_days", "ei_date", "eva_perc_std", "term_std"]
FINAL_BLEND = {"stacker_v5": 0.50, "stacker_v9": 0.20, "stacker_v4": 0.10, "stacker_bag_v4": 0.20}


class Cache:
    def __init__(self, out_dir, data):
        self.dir = Path(out_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.data = data
        self.scores = {}

    def get(self, name, fn):
        oof_path, tp_path = self.dir / f"oof_{name}.npy", self.dir / f"test_{name}.npy"
        if oof_path.exists() and tp_path.exists():
            oof, tp = np.load(oof_path), np.load(tp_path)
        else:
            t0 = time.time()
            oof, tp = fn()
            np.save(oof_path, oof)
            np.save(tp_path, tp)
            print(f"  {name}: trained in {(time.time() - t0) / 60:.1f} min")
        self.scores[name] = ndcg_at_k(oof, self.data.g_train, self.data.y_train)
        print(f"{name:<16} OOF NDCG@5 = {self.scores[name]:.5f}")
        return oof, tp


def run(args):
    train, test, feats = load_raw(args.data_dir)
    if args.sample_frac < 1:
        train = subsample_requests(train, args.sample_frac, args.seed)
        test = subsample_requests(test, args.sample_frac, args.seed)
    train, test = preprocess(train, test, feats)
    train, test = build_features(train, test)
    data = make_dataset(train, test, n_splits=args.folds)
    print(f"train {data.X_train.shape}, test {data.X_test.shape}, features {len(data.feature_cols)}")

    gpu = args.device == "gpu"
    cb_task = "GPU" if gpu else "CPU"
    xgb_device = "cuda" if gpu else "cpu"
    rounds = 300 if args.fast else 5000
    cb_fast = {"iterations": 100, "depth": 4} if args.fast else None
    stack_rounds = 300 if args.fast else 3000
    cache = Cache(args.out, data)

    base_v4 = {
        "lgb": cache.get("lgb_v4", lambda: models.train_lgb(data, max_rounds=rounds)),
        "cb_yeti": cache.get("cb_yeti_v4", lambda: models.train_catboost(data, models.CB_YETI_V4, task_type=cb_task,
                                                                         overrides=cb_fast)),
        "xgb": cache.get("xgb_v4", lambda: models.train_xgb(data, max_rounds=rounds, device=xgb_device)),
    }
    stackers = {
        "stacker_v4": cache.get("stacker_v4", lambda: train_stacker(base_v4, data, max_rounds=stack_rounds)),
        "stacker_bag_v4": cache.get("stacker_bag_v4", lambda: train_stacker(base_v4, data, seeds=(101, 202, 303),
                                                                            max_rounds=stack_rounds)),
    }

    auc, importance = adversarial_drift(data.X_train, data.X_test, seed=args.seed)
    detected = drifting_features(importance, data.cat_features, top_k=5)
    dropped = detected if args.recompute_drift else [c for c in DRIFT_FEATURES if c in data.feature_cols]
    print(f"adversarial AUC = {auc:.4f}, top drifting now: {detected}, dropped: {dropped}")
    data_v5 = data.subset([c for c in data.feature_cols if c not in dropped])

    base_v5 = {
        "lgb_v5": cache.get("lgb_v5", lambda: models.train_lgb(data_v5, max_rounds=rounds)),
        "cb_v5": cache.get("cb_v5", lambda: models.train_catboost(data, models.CB_YETI_V5, task_type=cb_task,
                                                                  overrides=cb_fast)),
        "xgb_v5": cache.get("xgb_v5", lambda: models.train_xgb(data_v5, max_rounds=rounds, device=xgb_device)),
    }
    stackers["stacker_v5"] = cache.get("stacker_v5", lambda: train_stacker(base_v5, data, max_rounds=stack_rounds))

    seeds = (42, 101) if args.fast else (42, 101, 202, 303, 404)
    extra = {
        "cb_seedbag5_v5": cache.get("cb_seedbag5_v5", lambda: models.train_catboost_seedbag(
            data_v5, seeds=seeds, task_type=cb_task, overrides=cb_fast)),
        "mlp_v5": cache.get("mlp_v5", lambda: train_mlp(data_v5, epochs=2 if args.fast else 8,
                                                        device="cuda" if gpu else "cpu")),
    }
    stackers["stacker_v9"] = cache.get("stacker_v9", lambda: train_stacker({**base_v5, **extra}, data,
                                                                            max_rounds=stack_rounds))

    names = list(FINAL_BLEND)
    weights = [FINAL_BLEND[n] for n in names]
    oof_blend = rank_blend([stackers[n][0] for n in names], weights, data.g_train)
    test_blend = rank_blend([stackers[n][1] for n in names], weights, data.g_test)
    cache.scores["final_blend"] = ndcg_at_k(oof_blend, data.g_train, data.y_train)
    print(f"{'final_blend':<16} OOF NDCG@5 = {cache.scores['final_blend']:.5f}")

    sub = data.test_keys[["request_id", "variant_no"]].assign(score=test_blend)
    sub.to_csv(Path(args.out) / "submission.csv", index=False, sep=";")
    summary = {"adversarial_auc": auc, "dropped_features": dropped, "oof_ndcg5": cache.scores}
    (Path(args.out) / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data")
    p.add_argument("--out", default="artifacts")
    p.add_argument("--device", choices=["gpu", "cpu"], default="gpu")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sample-frac", type=float, default=1.0, help="fraction of applications to keep")
    p.add_argument("--fast", action="store_true", help="few boosting rounds, for smoke tests")
    p.add_argument("--recompute-drift", action="store_true",
                   help="drop the top drifting features found now instead of the list from the competition run")
    run(p.parse_args())


if __name__ == "__main__":
    main()
