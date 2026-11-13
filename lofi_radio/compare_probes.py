"""
Paired, per-artifact probe comparison across frozen-feature caches.

Every arm trains the same tuned head on the same subject-grouped folds and the same seeds, over
the file subset common to all arms, so differences are matched seed-for-seed.

Reports, per arm: overall + per-artifact composite (mean +/- std over seeds). And against
--baseline: the paired delta (mean, paired t over seeds, win count) overall and per artifact.

Arms are ';'-separated  name=cache[+cache...]  (a '+' concatenates caches into one feature):
    python lofi_radio/compare_probes.py --baseline dinov2 --seeds 0,1,2,3,4 --arms \
      "student=cache/student_full_cls_lisa.npz;dinov2=cache/dinov2_lisa.npz"

(The gate is a per-artifact router, not a single feature -> use tri_probe.py for that arm.)
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

import head_hpo
from lisa import assign_folds, ARTIFACTS, DATA_DIR, LABELS_CSV
from evaluate import official_severity_metrics

TUNED_HEAD = {**head_hpo.BASELINE, "dropout": 0.5, "lr": 5e-4}


def parse_arms(spec: str) -> dict[str, list[str]]:
    """';'-sep name=cache[+cache]. ALL whitespace is stripped from names/paths so the spec
    survives terminal paste-wrapping that injects newlines mid-token (cache paths have none)."""
    nw = lambda s: "".join(s.split())
    arms = {}
    for chunk in spec.split(";"):
        if not nw(chunk):
            continue
        name, paths = chunk.split("=", 1)
        arms[nw(name)] = [nw(p) for p in paths.split("+") if nw(p)]
    return arms


def eval_arm(maps: list[dict], df, folds: int, hp, dev) -> dict:
    """One arm, one seed: 5-fold OOF over df (common files), return official metrics dict."""
    fns = list(df["filename"])
    X = np.stack([np.concatenate([m[fn] for m in maps]) for fn in fns]).astype(np.float32)
    y = np.stack([np.array([r[a] for a in ARTIFACTS], np.int64) for _, r in df.iterrows()])
    fold = df["fold"].to_numpy()
    oof = np.zeros_like(y)
    for k in range(folds):
        tr, va = fold != k, fold == k
        oof[va] = head_hpo.train_eval_fold(hp, X[tr], y[tr], X[va], dev)
    return official_severity_metrics(oof, y)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arms", required=True, help="';'-sep name=cache[+cache...]")
    ap.add_argument("--baseline", default=None, help="arm name to take paired Δ against (default: 1st)")
    ap.add_argument("--data_dir", default=DATA_DIR)
    ap.add_argument("--csv", default=LABELS_CSV)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--metric", default="composite",
                    choices=["composite", "accuracy", "f1", "f2", "precision", "recall"],
                    help="which per-artifact sub-metric to tabulate (overall 'score' stays the composite)")
    args = ap.parse_args()

    dev = args.device if (not args.device.startswith("cuda") or torch.cuda.is_available()) else "cpu"
    seeds = [int(s) for s in args.seeds.split(",")]
    arms = parse_arms(args.arms)
    baseline = args.baseline or next(iter(arms))
    assert baseline in arms, f"baseline {baseline!r} not in arms {list(arms)}"

    # load every referenced cache once; restrict to files common to ALL arms (paired, aligned)
    cache_of = {p: head_hpo._cache_map(p) for p in {p for ps in arms.values() for p in ps}}
    common = set.intersection(*[set(cache_of[p]) for ps in arms.values() for p in ps])
    print(f"[compare] arms={list(arms)} baseline={baseline} seeds={seeds} "
          f"common_files={len(common)} device={dev}\n")

    cols = ["score"] + ARTIFACTS
    per = {a: {c: [] for c in cols} for a in arms}                  # arm -> metric -> per-seed list
    for seed in seeds:
        np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        df = assign_folds(args.data_dir, args.csv, args.folds, seed)
        df = df[df["filename"].isin(common)].reset_index(drop=True)  # same split for every arm
        for a, paths in arms.items():
            m = eval_arm([cache_of[p] for p in paths], df, args.folds, TUNED_HEAD, dev)
            per[a]["score"].append(m["score"])
            for art in ARTIFACTS:
                per[a][art].append(m[art][args.metric])

    def col(arm, c):
        return np.array(per[arm][c])

    print(f"=== per-artifact {args.metric} (overall=composite; mean±std over {len(seeds)} seeds, tuned head) ===")
    print(f"{'arm':14s} {'overall':>14s} " + " ".join(f"{a[:5]:>7s}" for a in ARTIFACTS))
    for a in arms:
        row = f"{a:14s} {col(a,'score').mean():.4f}±{col(a,'score').std():.4f} "
        row += " ".join(f"{col(a,art).mean():7.3f}" for art in ARTIFACTS)
        print(row)

    print(f"\n=== paired Δ vs {baseline} (meanΔ, paired t, wins/{len(seeds)}) — credible if |t|≳2.5 ===")
    for a in arms:
        if a == baseline:
            continue
        print(f"-- {a} --")
        for c in cols:
            d = col(a, c) - col(baseline, c)                        # matched seed-for-seed
            t = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)) + 1e-12)
            tag = "overall" if c == "score" else c
            flag = " *" if abs(t) >= 2.5 and abs(d.mean()) > 0.002 else ""
            print(f"   {tag:12s} Δ={d.mean():+.4f}  t={t:+5.1f}  wins={int((d>0).sum())}/{len(d)}{flag}")


if __name__ == "__main__":
    main()
