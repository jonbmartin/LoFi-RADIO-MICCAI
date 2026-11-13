"""
Random hyperparameter search for the probe head on frozen features (paper Sec. 2.2).

The backbone is frozen, so the MLP head and its optimizer are the only trainable parts. Each
trial is a full subject-grouped 5-fold out-of-fold run at K seeds, scored with the official
composite. Most configurations tie at the ~0.0064 composite noise floor and the maximum over N
trials is optimistically biased (selection on the same CV), so configurations are ranked by
multi-seed mean and the simplest one in the top tier is preferred.

This module also provides the probe itself (`make_head`, `train_eval_fold`), which every other
experiment reuses so that all arms train an identical head.

  python lofi_radio/head_hpo.py --cache cache/dinov2_lisa.npz --trials 48 --search_seeds 0,1,2
  # then re-run the few top configs at more seeds (the command is printed).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from lisa import assign_folds, ARTIFACTS, N_SEVERITY, SeverityCELoss, DATA_DIR, LABELS_CSV
from evaluate import official_severity_metrics, overcall_diagnostics

K = len(ARTIFACTS)
S = N_SEVERITY

# the pre-search head/optimizer -- always evaluated as trial 0 (the bar to beat)
BASELINE = dict(hidden=[512], dropout=0.3, lr=1e-3, wd=1e-2, epochs=80, batch=32)

SPACE = dict(
    hidden=[[], [256], [512], [1024], [256, 256], [512, 512], [512, 256]],
    dropout=[0.0, 0.1, 0.2, 0.3, 0.5],
    lr=[3e-4, 5e-4, 1e-3, 2e-3, 3e-3],
    wd=[1e-4, 1e-3, 1e-2, 5e-2, 1e-1],
    epochs=[40, 80, 120],
    batch=[16, 32, 64],
)


def _cache_map(path):
    z = np.load(path, allow_pickle=True)
    files = [str(f) for f in z["files"]]
    return {f: z["X"][i].astype(np.float32) for i, f in enumerate(files)}


def load_data(cache, data_dir, csv, folds, seed):
    feat = _cache_map(cache)
    df = assign_folds(data_dir, csv, folds, seed)
    keep = [r for _, r in df.iterrows() if r["filename"] in feat]
    X = np.stack([feat[r["filename"]] for r in keep])
    y = np.stack([np.array([r[a] for a in ARTIFACTS], np.int64) for r in keep])
    fold = np.array([int(r["fold"]) for r in keep], np.int64)
    return X, y, fold


def make_head(d_in, hidden, dropout):
    layers, d = [], d_in
    for h in hidden:
        layers += [nn.Linear(d, h), nn.LeakyReLU(0.1, inplace=True), nn.Dropout(dropout)]
        d = h
    layers += [nn.Linear(d, K * S)]
    return nn.Sequential(*layers)


def train_eval_fold(hp, Xtr, ytr, Xva, dev, return_logits=False):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6           # per-fold train z-score (no leak)
    Xtr, Xva = ((Xtr - mu) / sd).astype(np.float32), ((Xva - mu) / sd).astype(np.float32)
    net = make_head(Xtr.shape[1], hp["hidden"], hp["dropout"]).to(dev)
    loss_fn = SeverityCELoss()
    opt = torch.optim.AdamW(net.parameters(), lr=hp["lr"], weight_decay=hp["wd"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=hp["epochs"])
    Xt = torch.from_numpy(Xtr).to(dev); yt = torch.from_numpy(ytr).long().to(dev)
    N = Xt.shape[0]
    for _ in range(hp["epochs"]):
        net.train()
        perm = torch.randperm(N, device=dev)
        for i in range(0, N, hp["batch"]):
            b = perm[i:i + hp["batch"]]
            opt.zero_grad()
            logits = net(Xt[b]).view(-1, K, S)
            loss = loss_fn(logits, yt[b])
            loss.backward(); opt.step()
        sched.step()
    net.eval()
    with torch.no_grad():
        out = net(torch.from_numpy(Xva).to(dev)).view(-1, K, S)
        if return_logits:
            return out.cpu().numpy()
        g = out.argmax(-1)
    return g.cpu().numpy()


def run_config(hp, cache, args, dev):
    comps, diags = [], []
    for seed in args.seeds:
        np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        X, y, fold = load_data(cache, args.data_dir, args.csv, args.folds, seed)
        oof = np.zeros_like(y)
        for k in range(args.folds):
            tr, va = fold != k, fold == k
            oof[va] = train_eval_fold(hp, X[tr], y[tr], X[va], dev)
        comps.append(official_severity_metrics(oof, y)["score"])
        diags.append(overcall_diagnostics(oof, y))
    diag = {k: float(np.mean([d[k] for d in diags])) for k in ("macro", "macro_gap", "call_rate", "true_call_rate")}
    return float(np.mean(comps)), float(np.std(comps)), comps, diag


def _sample(rng):
    hp = {}
    for k, v in SPACE.items():
        choice = v[int(rng.integers(len(v)))]      # one draw per field
        hp[k] = list(choice) if k == "hidden" else choice
    return hp


def _fmt(hp):
    return (f"h={hp['hidden']} drop={hp['dropout']} lr={hp['lr']:g} "
            f"wd={hp['wd']:g} ep={hp['epochs']} bs={hp['batch']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", default="cache/dinov2_lisa.npz", help="frozen feature cache (cache_teachers.py)")
    ap.add_argument("--dim", type=int, default=384, help="feature width (recorded only)")
    ap.add_argument("--data_dir", default=DATA_DIR)
    ap.add_argument("--csv", default=LABELS_CSV)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--trials", type=int, default=48, help="random configs (baseline always added)")
    ap.add_argument("--search_seeds", default="0,1,2", help="seeds per trial during search (speed)")
    ap.add_argument("--search_seed", type=int, default=0, help="rng seed for the config sampler")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out_dir", default="runs/head_hpo")
    ap.add_argument("--hp", default=None,
                    help='JSON of a single config to evaluate (skip search), e.g. \'{"hidden":[512],...}\'')
    args = ap.parse_args()
    args.seeds = [int(s) for s in args.search_seeds.split(",")]
    dev = args.device if (not args.device.startswith("cuda") or torch.cuda.is_available()) else "cpu"
    if dev == "cpu":
        print("[device] cuda unavailable -> cpu")
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    if args.hp:                                  # single-config mode (e.g. re-run a top config at 5 seeds)
        hp = {**BASELINE, **json.loads(args.hp)}
        m, sd, comps, diag = run_config(hp, args.cache, args, dev)
        print(f"{_fmt(hp)}\n  composite {m:.4f} +/- {sd:.4f}  seeds {args.seeds} -> {[round(c,4) for c in comps]}")
        over = diag["call_rate"] - diag["true_call_rate"]
        flag = (f"OVER-CALLING RISK (+{over:.3f} above the true rate)" if over > 0.01
                else "ok -- not over-calling (conservative is right for the support-weighted metric)")
        print(f"  [guardrail] call_rate {diag['call_rate']:.3f} vs true {diag['true_call_rate']:.3f}  "
              f"| macro {diag['macro']:.4f}  -> {flag}")
        return

    rng = np.random.default_rng(args.search_seed)
    seen, configs = set(), [dict(BASELINE)]
    while len(configs) < args.trials + 1:
        hp = _sample(rng)
        key = json.dumps(hp, sort_keys=True)
        if key not in seen:
            seen.add(key); configs.append(hp)

    print(f"[hpo] {len(configs)} configs (trial 0 = baseline) x seeds {args.seeds} | "
          f"cache {args.cache} ({args.dim}-d) | device {dev}\n")
    results = []
    for i, hp in enumerate(configs):
        m, sd, comps, _ = run_config(hp, args.cache, args, dev)
        results.append((m, sd, hp, comps))
        tag = "  <- BASELINE" if i == 0 else ""
        print(f"  [{i:3d}] {m:.4f} +/- {sd:.4f}   {_fmt(hp)}{tag}")

    base_m = results[0][0]
    results.sort(key=lambda r: -r[0])
    print(f"\n=== TOP 10 (baseline {base_m:.4f}; floor +/-0.0064) ===")
    for m, sd, hp, _ in results[:10]:
        d = m - base_m
        flag = "CREDIBLE+" if d > 0.0064 else ("tie" if abs(d) <= 0.0064 else "worse")
        print(f"  {m:.4f} +/- {sd:.4f}  Δ={d:+.4f} [{flag}]   {_fmt(hp)}")

    top = results[:5]
    print("\nRe-run the top configs at 5 seeds before trusting the ranking, e.g.:")
    print(f"  python lofi_radio/head_hpo.py --cache {args.cache} --search_seeds 0,1,2,3,4 \\\n"
          f"      --hp '{json.dumps({k: top[0][2][k] for k in BASELINE})}'")
    with open(out_dir / "hpo_results.json", "w") as f:
        json.dump({"baseline": base_m, "results": [
            {"composite": m, "std": sd, "hp": hp, "per_seed": comps} for m, sd, hp, comps in results]},
            f, indent=2)
    print(f"\nsaved {out_dir/'hpo_results.json'}")


if __name__ == "__main__":
    main()
