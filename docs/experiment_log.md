# Experiment log

Every submission scored on the public leaderboard, grouped by iteration. OOF is NDCG@5 on 5-fold `GroupKFold` by `request_id`; LB is NDCG@5 × 100 on Yandex Contest. Blend weights refer to within-group rank percentiles.

## Single models (OOF NDCG@5)

| Model | v3 | v4 | v5 | Notes |
|---|---:|---:|---:|---|
| LightGBM `lambdarank` | 0.91379 | 0.91427 | 0.91404 | v4 with Optuna params: 0.91513 |
| CatBoost `YetiRankPairwise` | 0.91390 | 0.91477 | 0.91509 | v5: native ordered TE, depth 6, 4000 iterations, 6× faster than v4 |
| XGBoost `rank:pairwise` | 0.91472 | 0.91604 | 0.91588 | v5 seed bag (3 seeds): 0.91626 |
| CatBoost `QuerySoftMax` | 0.91295 | | | dropped after v3 |
| LightGBM `rank_xendcg` | | | | 0.90605, dropped |
| MLP, listwise loss | | | 0.90232 | weak alone, useful inside a stacker |

Feature sets:

- **v3**: v2 features plus row position within the application and adversarial sample weights.
- **v4**: v3 minus row position features and external Bayesian target encoding.
- **v5**: v4 minus the five most drifting features (`max_grace_days`, `salary_last_days`, `ei_date`, `eva_perc_std`, `term_std`); adversarial AUC drops from 1.00 to 0.95.

## Stackers (OOF NDCG@5)

LambdaMART (LightGBM `lambdarank`, 31 leaves, `min_data_in_leaf = 200`, learning rate 0.02) on within-group z-scores and rank percentiles of base model OOF predictions plus the top 20 original features, trained on the same folds as the base models.

| Stacker | Base models | OOF | LB alone |
|---|---|---:|---:|
| `stacker_v4` | LGB, CB, XGB on v4 | 0.91754 | 91.9069 |
| `stacker_v5` | LGB, CB, XGB on v5 | 0.91732 | 91.9698 |
| `stacker_bag` | `stacker_v4` × 3 seeds | 0.91766 | 91.8993 |
| `stacker_v6` | v6 features (logit TE + adversarial propensity) | 0.91724 | |
| `stacker_v8` | v5 base + CatBoost seed bag (5 seeds) | 0.91765 | ≈ 91.94 |
| `stacker_v9` | v5 base + CatBoost seed bag + MLP | 0.91768 | ≈ 91.91 |

## Leaderboard submissions

### Baseline to v4: feature hygiene

| Submission | OOF | LB |
|---|---:|---:|
| baseline: LightGBM + CatBoost | | 91.6454 |
| v2: LGB + CB + XGB, grid blend | 0.91629 | 91.7855 |
| v3: + row position, QuerySoftMax, adversarial weights | 0.91598 | 91.6992 |
| v4: LGB + XGB | 0.91605 | 91.7531 |
| v4: LGB + CB + XGB, SLSQP | 0.91717 | 91.8348 |
| v4: LGB + CB + XGB, grid | 0.91724 | 91.8792 |

### Stackers vs large blends

| Submission | OOF | IW-OOF | LB |
|---|---:|---:|---:|
| `stacker_v5` | 0.91732 | 0.91768 | **91.9698** |
| mega6: 6 models, SLSQP | 0.91799 | | 91.9454 |
| mega7: 7 models, SLSQP | 0.91801 | | 91.9098 |
| mega7, differential evolution | 0.91831 | 0.91849 | 91.9599 |
| mega8: best IW-OOF | 0.91833 | 0.91855 | 91.9504 |
| mega5: 5 models, 2% grid | 0.91803 | | 91.9077 |
| v4 base + `stacker_v4` | 0.91791 | | 91.9004 |
| v6 base only, IW-OOF blend | | 0.91823 | 91.9073 |
| `stacker_v5` 0.70 + mega7 0.30 | | | 91.9674 |
| `stacker_v5` 0.50 + mega7 0.50 | | | 91.9661 |

The submission with the best OOF and the best importance-weighted OOF was among the worst on LB. Rank correlation with LB over these submissions: OOF ≈ 0.42, IW-OOF ≈ 0.45.

### Blends of stackers only

| `stacker_v5` | `stacker_v4` | `stacker_bag` | `stacker_v6` | LB |
|---:|---:|---:|---:|---:|
| 0.50 | 0.50 | | | 91.9527 |
| 0.60 | 0.40 | | | 91.9548 |
| 0.33 | 0.33 | 0.33 | | 91.9244 |
| 0.70 | 0.15 | 0.15 | | 91.9738 |
| **0.75** | **0.15** | **0.10** | | **91.9778** |
| 0.80 | 0.10 | 0.10 | | 91.9727 |
| 0.70 | 0.10 | 0.20 | | 91.9733 |
| 0.65 | 0.15 | 0.20 | | 91.9604 |
| 0.60 | 0.20 | 0.20 | | 91.9587 |
| 0.70 | | 0.30 | | 91.9733 |
| 0.80 | | 0.20 | | 91.9727 |
| 0.70 | 0.15 | 0.10 | 0.05 | 91.9772 |
| 0.65 | 0.15 | 0.15 | 0.05 | 91.9694 |
| 0.60 | 0.15 | 0.15 | 0.10 | 91.9626 |

### Adding `stacker_v8` and `stacker_v9`

| `stacker_v5` | `stacker_v8` | `stacker_v9` | `stacker_v4` | `stacker_bag` | LB |
|---:|---:|---:|---:|---:|---:|
| **0.50** | | **0.20** | **0.10** | **0.20** | **91.9988** |
| 0.50 | | 0.20 | 0.20 | 0.10 | 91.9885 |
| 0.50 | | 0.20 | 0.15 | 0.15 | 91.9861 |
| 0.55 | | 0.15 | 0.15 | 0.15 | 91.9788 |
| 0.45 | | 0.25 | 0.15 | 0.15 | 91.9772 |
| 0.40 | | 0.30 | 0.15 | 0.15 | 91.9783 |
| 0.50 | 0.20 | | 0.15 | 0.15 | 91.9842 |
| 0.50 | 0.30 | | | 0.20 | 91.9783 |
| 0.40 | 0.30 | | 0.15 | 0.15 | 91.9763 |
| 0.50 | 0.10 | 0.15 | 0.10 | 0.15 | 91.9828 |
| 0.45 | 0.10 | 0.20 | 0.10 | 0.15 | 91.9748 |
| 0.45 | 0.15 | 0.15 | 0.10 | 0.15 | 91.9722 |
| 0.40 | 0.20 | 0.20 | 0.10 | 0.10 | 91.9720 |
| 0.40 | 0.15 | 0.20 | 0.10 | 0.15 | 91.9706 |

`stacker_v8` and `stacker_v9` have almost identical OOF (0.91765 and 0.91768), yet `stacker_v9`, which includes the MLP, is consistently better inside blends. Using both at once is worse than `stacker_v9` alone: they are highly correlated.

## Leak probes

| Hypothesis | Result |
|---|---|
| Row position within the application | strong on train (9.5% vs 4.5% acceptance at position 0), −0.085 on LB |
| `commit.csv` sample submission | baseline predictions, Spearman 0.91 with `term` and 0.90 with `ncl`; no label signal |
| Exact offer matches between train positives and test | 0 matches on `(eva, ncl, rate, term, basket_name, limit)` |
| `app_id` reuse between train and test | 0.2% overlap |
| `variant_no` | `variant_no = 1` is accepted in 24% of cases (5× the base rate), already used by all models |

## Compute (Colab A100)

| Step | Time |
|---|---|
| Loading, preprocessing, features | 5 min |
| LightGBM, 5 folds | 11 min |
| XGBoost, 5 folds | 3 min |
| CatBoost v4: depth 8, 2500 iterations | 87 min |
| CatBoost v5: depth 6, 4000 iterations | 15 min |
| Stacker, 5 folds | 1.5 min |
| Optuna for LightGBM, 50 trials × 3 folds | 90 min |
