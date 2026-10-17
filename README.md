# Alfa-Bank "Cash Loans": Learning to Rank Credit Offers (NDCG@5)

My solution to the Alfa-Bank "Cash Loans" hackathon on Yandex Contest (May 2026).

**Result: NDCG@5 × 100 = 91.9988, 6th place** at the time of my final submission (leader 92.59, places 2 to 5 within 92.03 to 92.05).

This repository walks through the path from baseline to a stacked ensemble: what worked, what broke and why. The most useful part is not the code but the lessons on distribution shift and on when cross-validation cannot be trusted.

## Problem

A client submits a loan application (`request_id`). The bank shows from 1 to 50 offers (22 on average), and the client accepts **exactly one** (`is_deal = 1`). The task is to rank the offers so that the accepted one is as high as possible.

- **Metric:** NDCG@5. With one positive per group, IDCG = 1 and the metric equals `1 / log2(rank + 2)` if the accepted offer is in the top 5, otherwise 0.
- **Data:** train has 845,110 rows and 37,891 applications; test has 936,883 rows and 38,742 applications.
- **Features:** offer level (rate, term, limit, expected revenue `eva`, expected credit loss `ncl`, risk level, offer type, `variant_no`), application level (requested amount and term, channel), plus 280 client features joined by `(app_id, date_part)`.
- Synthetic data for January to March 2026. The organizers' data is not included in this repository.

Data pitfalls: `rate`, `eva`, `eva_perc` and `ncl` are stored as strings (Hive Decimal) and need `pd.to_numeric`; the client feature table has 148 duplicate `app_id` values.

## Approach

1. **Listwise features.** A client compares offers against each other, so within each application I compute rank, z-score, distance to min, max and median, and "best in application" flags for rate, term, limit, `eva` and `ncl`.
2. **Domain features.** Annuity payment, total interest, `eva - ncl`, revenue per unit of limit, match between offered and requested amount and term.
3. **Base ranking models**, 5-fold `GroupKFold` by `request_id` with identical folds for all models:
   - LightGBM `lambdarank` (`lambdarank_truncation_level = 8`, `label_gain = [0, 1]`);
   - CatBoost `YetiRankPairwise` with native ordered target encoding;
   - XGBoost `rank:pairwise`;
   - CatBoost seed bag (5 seeds) and a weak MLP with a listwise loss as sources of diversity.
4. **Second-level stacker** (the 5th place pattern from Expedia, ICDM 2013): LambdaMART on within-group z-scores and rank percentiles of base model OOF predictions, plus the top 20 original features.
5. **Blend of stackers only**, by within-group rank percentiles. Final weights: `stacker_v5 0.50 + stacker_v9 0.20 + stacker_v4 0.10 + stacker_bag 0.20`.

The full pipeline is in [`src/`](src): features, base models, stackers and the final blend.

## Leaderboard progression

| Version | Change | OOF | LB |
|---|---|---:|---:|
| baseline | LightGBM lambdarank + CatBoost, listwise ranks | | 91.65 |
| v2 | + XGBoost, domain features, count and target encoding | 0.91629 | 91.7855 |
| v3 | + row position in group, CatBoost QuerySoftMax, adversarial weights | 0.91598 | 91.6992 (drop) |
| v4 | removed row position, external target encoding and QuerySoftMax | 0.91724 | 91.8792 |
| stacker_v4 | LambdaMART stacker over v4 base models | 0.91754 | 91.9069 |
| stacker_v5 | same stacker over base models without 5 drifting features | 0.91732 | 91.9698 |
| stacker blend | v5 + v4 + bagged stackers | | 91.9778 |
| **final** | + stacker with CatBoost seed bag and MLP | | **91.9988** |

Every leaderboard submission with its OOF and LB: [`docs/experiment_log.md`](docs/experiment_log.md).

## Key lessons

**1. A row-order leak that works only on train.** In the original file order, the first row of an application was the accepted offer in 9.5% of cases versus 4.5% on average. A row position feature lost 0.085 on LB: the test file was ordered differently.

**2. Adversarial AUC = 1.0 means cross-validation is lying.** A classifier separated train from test perfectly. Removing the five most drifting features (`max_grace_days`, `salary_last_days`, `ei_date`, `eva_perc_std`, `term_std`) made OOF **worse** but improved the stacker's LB by **+0.063**. The model on all features was learning patterns that do not exist in test.

**3. OOF does not rank submissions.** Across 13+ submissions, the rank correlation between OOF and LB was about 0.42, and about 0.45 for importance-weighted OOF (Sugiyama, 2007). The blend with the best OOF (0.91833, tuned with differential evolution) scored lower on LB than a plain stacker. Density-ratio reweighting `p / (1 - p)` collapses to a constant when AUC = 1.0.

**4. Fewer base models in the blend, higher LB.** The stacker already uses the base models, and adding them to the blend again cost 0.003 to 0.012 LB each. A blend of stackers only won consistently.

**5. Diversity beats strength.** An MLP with OOF 0.902, clearly weaker than the boosters, added +0.013 LB inside the stacker. CatBoost `QuerySoftMax`, in theory ideal for one positive per group, was the weakest model (0.91295 vs 0.91390 for `YetiRankPairwise`): with one positive it reduces to ListNet top-1 and gives few gradients per group.

**6. Engineering pitfalls.** CatBoost computes NDCG on CPU even when training on GPU: with the default `metric_period` a fold took over 40 minutes, with `metric_period = 200` it was several times faster. OOF predictions must be saved with the `(app_id, variant_no)` key: reloading them by row order once produced OOF 0.19 instead of 0.915.

## What did not work

- row position features, external Bayesian target encoding, adversarial sample weights;
- `rank_xendcg` (OOF 0.906), CatBoost `QuerySoftMax`;
- FT-Transformer: -0.016 on LB at nearly the same OOF;
- leak hunting: `commit.csv` turned out to be baseline predictions, there are no exact offer matches between train and test, and `app_id` overlap is 0.2%.

The +0.59 gap to 1st place is most likely explained by a leak rather than by modeling: places 2 to 5 are within 0.03 of each other.

## Reproducing the final submission

```bash
pip install -r requirements.txt
python -m src.pipeline --data-dir data --out artifacts
```

Put the hackathon data in `data/`: `train_dataset_small.pq`, `test_dataset_small.pq`, `features_small.pq`. The full run needs a GPU (a Colab T4 is enough): CatBoost and XGBoost train on GPU, and the CatBoost seed bag alone is 25 fits. Each stage caches its OOF and test predictions in `artifacts/`, so an interrupted run resumes where it stopped. The v5 feature set drops the five drifting features found in the competition run; `--recompute-drift` reruns adversarial validation and drops whatever drifts most now. For a quick CPU check on 1% of applications (about 5 minutes):

```bash
python -m src.pipeline --data-dir data --out artifacts_smoke --sample-frac 0.01 --device cpu --fast
```

The final blend was checked against the submitted file: combining the four stacker predictions saved during the competition with `src.ranking_utils.rank_blend` and the weights above reproduces the 91.9988 submission exactly. The code in `src/` is a cleaned-up version of the competition scripts with the same features, folds and parameters. The MLP training script was not kept, so `src/mlp.py` reimplements it from the recorded setup (512, 256, 128 dense layers, softmax loss within each application); its OOF may differ slightly from the original 0.902.

## Repository structure

| Path | Contents |
|---|---|
| `src/pipeline.py` | End-to-end run: v4 and v5 base models, four stackers, final blend |
| `src/data.py`, `src/features.py`, `src/dataset.py` | Loading, listwise, domain and count features, shared folds |
| `src/models.py`, `src/mlp.py` | LightGBM, XGBoost, CatBoost, CatBoost seed bag, listwise MLP |
| `src/stacker.py` | LambdaMART stacker on within-group z-scores and rank percentiles |
| `src/drift.py` | Adversarial validation and drifting feature selection |
| `src/ranking_utils.py` | NDCG@5, rank blending |
| `notebooks/` | Early iterations, baseline to v3 (Colab) |
| `docs/experiment_log.md` | All leaderboard submissions, single-model and stacker OOF, leak probes |

## Stack

Python, pandas, LightGBM, CatBoost, XGBoost, PyTorch, scikit-learn
