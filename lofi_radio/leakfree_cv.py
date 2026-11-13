"""
Leak-free per-fold distillation CV (paper Sec. 2.5).

A student distilled on all LISA images has seen the CV validation folds (without labels) during
distillation -- an advantage the frozen baselines never get. The leak-free protocol keeps LISA
train images in distillation but excludes each validation fold: one student per fold that never
saw that fold's images. This script (a) writes the per-fold LISA id lists to feed
`distill.py --exclude_ids_file`, and (b) assembles the resulting per-fold student caches into a
leak-free out-of-fold official metric.

  # 1) write the per-fold val id lists for a seed
  python lofi_radio/leakfree_cv.py write-ids --seed 0 --out_dir cache/folds
  # 2) distill one student per fold, excluding that fold, then export (see scripts/run_leakfree.sh)
  # 3) assemble the leak-free OOF metric (fold k predicted from the student that excluded fold k):
  python lofi_radio/leakfree_cv.py eval --seed 0 \
    --fold_caches "0=cache/student_s0_f0.npz,1=...,2=...,3=...,4=..." \
    --baseline cache/dinov2_lisa.npz
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

import head_hpo
from lisa import assign_folds, ARTIFACTS, DATA_DIR, LABELS_CSV, REPO
from evaluate import official_severity_metrics

TUNED_HEAD = {**head_hpo.BASELINE, "dropout": 0.5, "lr": 5e-4}


def _oof_from_single_cache(cache_map, df, folds, dev, head=None):
    """Standard OOF for a leak-free cache (DINOv2 baseline): one feature map, 5-fold."""
    hp = head if head is not None else TUNED_HEAD
    fns = list(df["filename"])
    X = np.stack([cache_map[f] for f in fns]).astype(np.float32)
    y = np.stack([np.array([r[a] for a in ARTIFACTS], np.int64) for _, r in df.iterrows()])
    fold = df["fold"].to_numpy()
    oof = np.zeros_like(y)
    for k in range(folds):
        tr, va = fold != k, fold == k
        oof[va] = head_hpo.train_eval_fold(hp, X[tr], y[tr], X[va], dev)
    return oof, y


def _oof_leakfree(fold_maps, df, folds, dev, head=None):
    """Leak-free OOF: fold k is predicted from the student that EXCLUDED fold k (fold_maps[k]).
    Train uses that same student's features on the OTHER folds (in-sample for train is fine)."""
    hp = head if head is not None else TUNED_HEAD
    fns = list(df["filename"])
    y = np.stack([np.array([r[a] for a in ARTIFACTS], np.int64) for _, r in df.iterrows()])
    fold = df["fold"].to_numpy()
    oof = np.zeros_like(y)
    for k in range(folds):
        X = np.stack([fold_maps[k][f] for f in fns]).astype(np.float32)   # student_k features for ALL files
        tr, va = fold != k, fold == k
        oof[va] = head_hpo.train_eval_fold(hp, X[tr], y[tr], X[va], dev)   # predict held-out fold k
    return oof, y


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["write-ids", "eval"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--data_dir", default=DATA_DIR)
    ap.add_argument("--csv", default=LABELS_CSV)
    ap.add_argument("--out_dir", default=os.path.join(REPO, "cache", "folds"))
    ap.add_argument("--fold_caches", default=None, help="'k=path,...' leak-free student cache per fold")
    ap.add_argument("--baseline", default=None, help="leak-free baseline cache (e.g. dinov2_lisa.npz) for the delta")
    ap.add_argument("--hp", default=None,
                    help="JSON head-config override, e.g. '{\"hidden\":[256],\"dropout\":0.5,\"lr\":0.0005,"
                         "\"wd\":0.001,\"epochs\":40,\"batch\":16}'. Merged on top of TUNED_HEAD.")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    tuned_head = dict(TUNED_HEAD)
    if args.hp:
        tuned_head.update(json.loads(args.hp))

    df = assign_folds(args.data_dir, args.csv, args.folds, args.seed)

    if args.mode == "write-ids":
        os.makedirs(args.out_dir, exist_ok=True)
        for k in range(args.folds):
            files = df[df["fold"] == k]["filename"].tolist()
            p = os.path.join(args.out_dir, f"seed{args.seed}_fold{k}.txt")
            with open(p, "w") as f:
                f.write("\n".join(str(x) for x in files) + "\n")
            print(f"  wrote {len(files)} ids -> {p}")
        return

    # eval
    dev = args.device if (not args.device.startswith("cuda") or torch.cuda.is_available()) else "cpu"
    fc = {}
    for kv in args.fold_caches.split(","):
        k, p = kv.split("=", 1)
        fc[int(k.strip())] = "".join(p.split())
    assert set(fc) == set(range(args.folds)), f"need a cache for every fold, got {sorted(fc)}"
    fold_maps = {k: head_hpo._cache_map(p) for k, p in fc.items()}
    common = set.intersection(*[set(m) for m in fold_maps.values()])
    df = df[df["filename"].isin(common)].reset_index(drop=True)
    print(f"[leakfree] seed={args.seed} folds={args.folds} files={len(df)} device={dev}")

    oof, y = _oof_leakfree(fold_maps, df, args.folds, dev, head=tuned_head)
    m = official_severity_metrics(oof, y)
    print("\n=== leak-free student (per-fold distillation) ===")
    print(f"  head: {tuned_head}")
    print(f"  composite {m['score']:.4f}")
    for a in ARTIFACTS:
        print(f"    {a:12s} {m[a]['composite']:.4f}")

    if args.baseline:
        bm = head_hpo._cache_map(args.baseline)
        df_b = df[df["filename"].isin(set(bm))].reset_index(drop=True)
        ob, yb = _oof_from_single_cache(bm, df_b, args.folds, dev, head=tuned_head)
        mb = official_severity_metrics(ob, yb)
        print(f"\n=== baseline {os.path.basename(args.baseline)} (leak-free already) ===")
        print(f"  composite {mb['score']:.4f}   ->  leak-free student Δ = {m['score'] - mb['score']:+.4f}")
        for a in ARTIFACTS:
            print(f"    {a:12s} {mb[a]['composite']:.4f}   Δ {m[a]['composite'] - mb[a]['composite']:+.4f}")


if __name__ == "__main__":
    main()
