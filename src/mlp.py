"""Small listwise MLP used only as an extra input to the stacker.

Alone it is clearly weaker than the boosters (OOF about 0.90 vs 0.915), but its
errors are different, and that is what the stacker needs. The loss is a
softmax cross-entropy inside each application: with exactly one accepted
offer per group it maximizes the probability of picking that offer.
"""
import numpy as np
import torch
from sklearn.preprocessing import QuantileTransformer
from torch import nn

from .ranking_utils import ndcg_at_k


def build_mlp(n_features, hidden=(512, 256, 128), dropout=0.1):
    layers = []
    width = n_features
    for h in hidden:
        layers += [nn.Linear(width, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
        width = h
    layers.append(nn.Linear(width, 1))
    return nn.Sequential(*layers)


def group_softmax_ce(scores, y, group_idx, n_groups):
    """Mean over groups of -log softmax(score of the accepted offer)."""
    scores = scores.float()
    max_per = torch.full((n_groups,), -float("inf"), device=scores.device)
    max_per = max_per.scatter_reduce(0, group_idx, scores.detach(), reduce="amax", include_self=True)
    shifted = scores - max_per[group_idx]
    denom = torch.zeros(n_groups, device=scores.device).scatter_add(0, group_idx, shifted.exp())
    log_p = shifted - denom.clamp_min(1e-20).log()[group_idx]
    pos = (y == 1).float()
    loss_per_group = torch.zeros(n_groups, device=scores.device).scatter_add(0, group_idx, -log_p * pos)
    has_pos = torch.zeros(n_groups, device=scores.device).scatter_add(0, group_idx, pos) > 0
    return loss_per_group[has_pos].mean()


def _predict(model, X, device, batch=20000):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            out.append(model(torch.from_numpy(X[i:i + batch]).to(device)).squeeze(-1).float().cpu().numpy())
    return np.concatenate(out)


def train_mlp(data, seed=42, epochs=8, lr=1e-3, groups_per_batch=256, device=None):
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    num = data.X_train.columns
    X_tr_raw = np.clip(data.X_train[num].fillna(0).to_numpy("float32"), -1e6, 1e6)
    X_te_raw = np.clip(data.X_test[num].fillna(0).to_numpy("float32"), -1e6, 1e6)
    qt = QuantileTransformer(n_quantiles=1000, output_distribution="normal", subsample=200000, random_state=seed)
    X_tr = qt.fit_transform(X_tr_raw).astype("float32")
    X_te = qt.transform(X_te_raw).astype("float32")

    _, g_codes = np.unique(data.g_train, return_inverse=True)
    oof = np.zeros(len(X_tr), dtype="float32")
    test_pred = np.zeros(len(X_te), dtype="float32")

    for tr, va in data.splits:
        model = build_mlp(X_tr.shape[1]).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

        rows_by_group = {}
        for i in tr:
            rows_by_group.setdefault(g_codes[i], []).append(i)
        group_ids = np.array(list(rows_by_group))

        best, best_va, best_state = -1.0, None, None
        for _ in range(epochs):
            model.train()
            rng.shuffle(group_ids)
            for start in range(0, len(group_ids), groups_per_batch):
                batch_groups = group_ids[start:start + groups_per_batch]
                rows = np.concatenate([rows_by_group[g] for g in batch_groups])
                local = np.repeat(np.arange(len(batch_groups)), [len(rows_by_group[g]) for g in batch_groups])
                xb = torch.from_numpy(X_tr[rows]).to(device)
                yb = torch.from_numpy(data.y_train[rows].astype("int64")).to(device)
                gb = torch.from_numpy(local).to(device)
                loss = group_softmax_ce(model(xb).squeeze(-1), yb, gb, len(batch_groups))
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
            sched.step()
            va_scores = _predict(model, X_tr[va], device)
            score = ndcg_at_k(va_scores, data.g_train[va], data.y_train[va])
            if score > best:
                best, best_va = score, va_scores
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

        model.load_state_dict(best_state)
        oof[va] = best_va
        test_pred += _predict(model, X_te, device) / len(data.splits)
    return oof, test_pred
