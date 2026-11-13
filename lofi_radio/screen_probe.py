"""
Uniform foundation-model screen -> machine-readable JSON for the screening table (paper Table 1).

One protocol applied to every candidate backbone:

  * degeneracy : mean off-diagonal |cos| across volumes, isotropic scale, and effective-rank
                 fraction of the pooled features.
  * single-arm probe : the same tuned head on the same subject-grouped folds and seeds as every
                 other arm (reuses compare_probes.eval_arm), over the file subset common to the
                 candidate and the baseline -> composite mean +/- std + per-artifact.
  * paired vs baseline : matched seed-for-seed delta, paired t, win count vs DINOv2.

    python lofi_radio/screen_probe.py --baseline dinov2 --seeds 0,1,2,3,4,5,6,7,8,9 --arms \
      "dinov2=cache/dinov2_lisa.npz; biomedclip=cache/biomedclip_lisa.npz; \
       phikon=cache/phikon_lisa.npz; ..." --out runs/screen/screen_results.json
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

import head_hpo
from lisa import assign_folds, ARTIFACTS, DATA_DIR, LABELS_CSV, REPO
from compare_probes import parse_arms, eval_arm, TUNED_HEAD


def _degeneracy(X: np.ndarray) -> dict:
    """off-diag |cos|, isotropic scale, eff-rank fraction (participation ratio / D)."""
    X = X.astype(np.float32)
    n = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    c = n @ n.T
    off = float(np.abs(c[~np.eye(len(c), dtype=bool)]).mean())
    scale = float(np.sqrt(X.var(0).mean()))
    cov = np.cov(X.T)
    w = np.clip(np.linalg.eigvalsh(cov), 0, None)
    eff_rank = float((w.sum() ** 2) / (np.sum(w ** 2) + 1e-12) / len(w))
    return dict(cos=off, scale=scale, eff_rank=eff_rank, n_dim=int(X.shape[1]), n_vol=int(X.shape[0]))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arms", required=True, help="';'-sep name=cache (one cache per backbone)")
    ap.add_argument("--baseline", default="dinov2", help="arm to take the paired Δ against")
    ap.add_argument("--data_dir", default=DATA_DIR)
    ap.add_argument("--csv", default=LABELS_CSV)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=os.path.join(REPO, "runs/screen/screen_results.json"))
    args = ap.parse_args()

    dev = args.device if (not args.device.startswith("cuda") or torch.cuda.is_available()) else "cpu"
    seeds = [int(s) for s in args.seeds.split(",")]
    arms = parse_arms(args.arms)                          # name -> [cache] (screen = single cache each)
    base = args.baseline
    assert base in arms, f"baseline {base!r} not in arms {list(arms)}"

    # degeneracy per arm (on the cache's own feature matrix) + per-seed composite probe
    deg, comp = {}, {a: {c: [] for c in ["score", *ARTIFACTS]} for a in arms}
    for a, paths in arms.items():
        z = np.load(paths[0], allow_pickle=True)
        deg[a] = _degeneracy(z["X"])
    maps = {p: head_hpo._cache_map(p) for ps in arms.values() for p in ps}

    t0 = time.time()
    for seed in seeds:
        np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        for a, paths in arms.items():
            # pair each candidate with the baseline on THEIR common files (matched split for the Δ)
            common = set(maps[paths[0]]) & set(maps[arms[base][0]])
            df = assign_folds(args.data_dir, args.csv, args.folds, seed)
            df = df[df["filename"].isin(common)].reset_index(drop=True)
            m = eval_arm([maps[p] for p in paths], df, args.folds, TUNED_HEAD, dev)
            comp[a]["score"].append(m["score"])
            for art in ARTIFACTS:
                comp[a][art].append(m[art]["composite"])
        print(f"  seed {seed} done ({time.time()-t0:.0f}s)")

    # assemble: composite mean±std, per-artifact mean, paired Δ/t/wins vs baseline
    bscore = np.array(comp[base]["score"])
    results = {}
    for a in arms:
        sc = np.array(comp[a]["score"])
        row = dict(**deg[a], composite=float(sc.mean()), composite_std=float(sc.std()),
                   per_seed=sc.tolist(),
                   per_artifact={art: float(np.mean(comp[a][art])) for art in ARTIFACTS})
        if a != base:
            d = sc - bscore
            row["delta"] = float(d.mean())
            row["t"] = float(d.mean() / (d.std(ddof=1) / np.sqrt(len(d)) + 1e-12))
            row["wins"] = int((d > 0).sum())
        else:
            row["delta"] = row["t"] = row["wins"] = None
        results[a] = row

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    out = dict(baseline=base, seeds=seeds, n_seeds=len(seeds), head=TUNED_HEAD, results=results)
    json.dump(out, open(args.out, "w"), indent=2)

    # console summary (sorted by composite desc)
    print(f"\n=== uniform screen ({len(seeds)} seeds, tuned head, baseline={base}) ===")
    print(f"{'backbone':14s} {'eff_rank':>8s} {'cos':>6s} {'composite':>16s} {'Δ vs base':>10s} {'t':>6s}")
    for a, r in sorted(results.items(), key=lambda kv: -kv[1]["composite"]):
        d = "" if r["delta"] is None else f"{r['delta']:+.4f}"
        tt = "" if r["t"] is None else f"{r['t']:+.1f}"
        print(f"{a:14s} {r['eff_rank']:8.3f} {r['cos']:6.2f} "
              f"{r['composite']:.4f}±{r['composite_std']:.4f} {d:>10s} {tt:>6s}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
