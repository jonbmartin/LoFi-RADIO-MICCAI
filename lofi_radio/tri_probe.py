"""
Single-backbone, concatenation, and per-artifact gate probes on frozen features (paper Sec. 2.3,
Table 2).

All arms run through one matched scaffold: per-fold train z-score per feature block,
subject-grouped folds, cross-entropy head, argmax decode, official metric. Modes:

  singles : one probe head per backbone
  concat  : one head on the concatenation of all backbones
  gate    : one head per backbone, mixed per artifact by a learned softmax weight
            (gate[K, n] initialized to 0 -> uniform 1/n)

Deltas are reported against the first backbone (the reference, usually DINOv2), both marginally
and paired seed-for-seed. Reads only cached features (cache_teachers.py / export_student.py):

  python lofi_radio/tri_probe.py \
    --caches cache/dinov2_lisa.npz,cache/biomedclip_lisa.npz,cache/sam_lisa.npz,cache/phikon_lisa.npz \
    --names DINOv2,BiomedCLIP,SAM,Phikon --seeds 0,1,2,3,4,5,6,7,8,9 --dropout 0.5 --lr 5e-4
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from lisa import assign_folds, ARTIFACTS, N_SEVERITY, SeverityCELoss, DATA_DIR, LABELS_CSV
from evaluate import official_severity_metrics, present_auroc

K = len(ARTIFACTS)
S = N_SEVERITY


def _cache_map(path):
    z = np.load(path, allow_pickle=True)
    files = [str(f) for f in z["files"]]
    return {f: z["X"][i].astype(np.float32) for i, f in enumerate(files)}


def load_data(args, seed):
    """Align all backbone caches with the labels CSV + subject-grouped folds (common filename subset)."""
    maps = [_cache_map(c) for c in args.caches]
    df = assign_folds(args.data_dir, args.csv, args.folds, seed)
    keep = [r for _, r in df.iterrows() if all(r["filename"] in m for m in maps)]
    dropped = len(df) - len(keep)
    if dropped:
        print(f"[data] {dropped} row(s) absent from a cache -> dropped (using {len(keep)})")
    blocks = [np.stack([m[r["filename"]] for r in keep]) for m in maps]   # list of [N, d_i]
    y = np.stack([np.array([r[a] for a in ARTIFACTS], np.int64) for r in keep])
    fold = np.array([int(r["fold"]) for r in keep], np.int64)
    return blocks, y, fold


class _Head(nn.Module):
    """Probe head: Linear(d,512)->LeakyReLU->Dropout(p)->Linear(512,K*S)."""
    def __init__(self, d, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, 512), nn.LeakyReLU(0.1, inplace=True),
                                 nn.Dropout(dropout), nn.Linear(512, K * S))

    def forward(self, x):
        return self.net(x).view(-1, K, S)


class MultiProbe(nn.Module):
    """mode in {single:<i>, concat, gate}. 'gate' = per-artifact SOFTMAX mix over n expert heads
    (gate[K,n] init 0 -> uniform 1/n)."""
    def __init__(self, mode, dims, dropout=0.3):
        super().__init__()
        self.mode = mode
        self.n = len(dims)
        if mode == "concat":
            self.head = _Head(sum(dims), dropout)
        elif mode == "gate":
            self.heads = nn.ModuleList([_Head(d, dropout) for d in dims])
            self.gate = nn.Parameter(torch.zeros(K, self.n))
        elif mode.startswith("single:"):
            self.idx = int(mode.split(":")[1])
            self.head = _Head(dims[self.idx], dropout)
        else:
            raise ValueError(mode)

    def forward(self, xs):
        if self.mode == "concat":
            return self.head(torch.cat(xs, dim=1))
        if self.mode.startswith("single:"):
            return self.head(xs[self.idx])
        w = torch.softmax(self.gate, dim=1)                     # [K, n] per-artifact expert weights
        out = 0.0
        for e in range(self.n):
            out = out + w[:, e].view(1, K, 1) * self.heads[e](xs[e])
        return out

    @torch.no_grad()
    def decode(self, logits):
        return logits.argmax(-1), 1.0 - torch.softmax(logits, -1)[..., 0]


def _standardize(train, *blocks):
    mu, sd = train.mean(0), train.std(0) + 1e-6
    return [((b - mu) / sd).astype(np.float32) for b in blocks]


def train_eval_fold(mode, dims, tr_blocks, ytr, va_blocks, args, dev):
    tr_s, va_s = [], []
    for tb, vb in zip(tr_blocks, va_blocks):
        s = _standardize(tb, tb, vb)
        tr_s.append(s[0]); va_s.append(s[1])
    m = MultiProbe(mode, dims, args.dropout).to(dev)
    loss_fn = SeverityCELoss()
    opt = torch.optim.AdamW(m.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    Xt = [torch.from_numpy(b).to(dev) for b in tr_s]
    yt = torch.from_numpy(ytr).long().to(dev)
    N = yt.shape[0]
    for _ in range(args.epochs):
        m.train()
        perm = torch.randperm(N, device=dev)
        for i in range(0, N, args.batch):
            b = perm[i:i + args.batch]
            opt.zero_grad()
            loss = loss_fn(m([x[b] for x in Xt]), yt[b])
            if mode == "gate" and args.gate_entropy != 0.0:
                w = torch.softmax(m.gate, dim=1)                       # [K, n] per-artifact weights
                ent = -(w * (w + 1e-9).log()).sum(1).mean()           # mean over artifacts
                loss = loss + args.gate_entropy * ent                 # >0 -> minimize entropy -> sharpen
            loss.backward(); opt.step()
        sched.step()
    m.eval()
    with torch.no_grad():
        g, pr = m.decode(m([torch.from_numpy(b).to(dev) for b in va_s]))
    gate = torch.softmax(m.gate, dim=1).detach().cpu().numpy() if mode == "gate" else None
    return g.cpu().numpy(), pr.cpu().numpy(), gate


def run_seed(mode, dims, blocks, y, fold, args, dev):
    N = y.shape[0]
    oof_pred = np.zeros((N, K), np.int64); oof_present = np.zeros((N, K), np.float32)
    gates = []
    for k in range(args.folds):
        tr, va = fold != k, fold == k
        g, pr, gate = train_eval_fold(mode, dims, [b[tr] for b in blocks], y[tr],
                                      [b[va] for b in blocks], args, dev)
        oof_pred[va], oof_present[va] = g, pr
        if gate is not None:
            gates.append(gate)
    m = official_severity_metrics(oof_pred, y)
    auc = present_auroc(oof_present, y)
    for a in ARTIFACTS:
        m[a]["present_auroc"] = auc[a]
    m["present_auroc_mean"] = auc["mean"]
    return m, (np.mean(gates, 0) if gates else None)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--caches", required=True, help="comma list of frozen-feature .npz caches")
    ap.add_argument("--names", default=None, help="comma list of labels (default cache#)")
    ap.add_argument("--data_dir", default=DATA_DIR)
    ap.add_argument("--csv", default=LABELS_CSV)
    ap.add_argument("--modes", default="singles,concat,gate",
                    help="'singles' expands to one run per backbone; plus concat,gate")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-2)
    ap.add_argument("--dropout", type=float, default=0.3, help="head dropout (the paper uses 0.5)")
    ap.add_argument("--gate_entropy", type=float, default=0.0,
                    help="entropy penalty on the per-artifact gate weights: >0 SHARPENS routing "
                         "toward each artifact's argmax backbone, <0 flattens, 0 = current soft gate. "
                         "(A plain softmax temperature on a LEARNABLE gate is a no-op -- optimizer "
                         "rescales it; the entropy penalty is the real lever.)")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out_dir", default="runs/tri_probe")
    args = ap.parse_args()
    args.caches = [c.strip() for c in args.caches.split(",") if c.strip()]
    names = ([n.strip() for n in args.names.split(",")] if args.names
             else [f"cache{i}" for i in range(len(args.caches))])
    assert len(names) == len(args.caches), "names must match caches"

    dev = args.device if (not args.device.startswith("cuda") or torch.cuda.is_available()) else "cpu"
    if dev == "cpu":
        print("[device] cuda unavailable -> cpu")
    seeds = [int(s) for s in args.seeds.split(",")]
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # build the mode list: optional per-backbone singles, then concat, gate
    req = [m.strip() for m in args.modes.split(",") if m.strip()]
    modes = []
    for m in req:
        if m == "singles":
            modes += [f"single:{i}" for i in range(len(args.caches))]
        else:
            modes.append(m)
    # dims discovered from seed 0
    blocks0, _, _ = load_data(args, seeds[0])
    dims = [b.shape[1] for b in blocks0]
    print(f"[tri] backbones: {list(zip(names, dims))} | modes {modes} | seeds {seeds} | device {dev}\n")

    def label(mode):
        return names[int(mode.split(":")[1])] if mode.startswith("single:") else mode

    summary = {}
    for mode in modes:
        comps, per_art, gate_acc = [], {a: [] for a in ARTIFACTS}, []
        for seed in seeds:
            np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
            blocks, y, fold = load_data(args, seed)
            m, gate = run_seed(mode, dims, blocks, y, fold, args, dev)
            comps.append(m["score"])
            for a in ARTIFACTS:
                per_art[a].append(m[a]["composite"])
            if gate is not None:
                gate_acc.append(gate)
        cm, cs = float(np.mean(comps)), float(np.std(comps))
        summary[label(mode)] = {"composite": cm, "composite_std": cs, "per_seed": comps,
                                "per_artifact": {a: [float(np.mean(per_art[a])), float(np.std(per_art[a]))]
                                                 for a in ARTIFACTS},
                                "gate_mean": (np.mean(gate_acc, 0).tolist() if gate_acc else None)}
        print(f"=== {label(mode)}  ({len(seeds)} seeds) ===")
        print(f"  composite {cm:.4f} +/- {cs:.4f}   per-seed {[round(c,4) for c in comps]}")
        for a in ARTIFACTS:
            print(f"    {a:12s} {np.mean(per_art[a]):.4f} +/- {np.std(per_art[a]):.4f}")
        if gate_acc:
            gm = np.mean(gate_acc, 0)   # [K, n]
            print("  learned gate weights (softmax over backbones):")
            print("    " + " ".join(f"{n:>10s}" for n in names))
            for j, a in enumerate(ARTIFACTS):
                print(f"    {a:12s} " + " ".join(f"{gm[j, e]:>10.3f}" for e in range(len(names))))
        print()

    ref = label(modes[0])
    ref_ps = np.array(summary[ref]["per_seed"])
    print(f"--- composite deltas vs '{ref}'  (marginal floor +/-0.0064 | PAIRED test: arms share "
          f"folds+seeds+subset, so the per-seed diff cancels common-mode seed variance) ---")
    for mode in modes[1:]:
        lab = label(mode)
        d = summary[lab]["composite"] - summary[ref]["composite"]
        diffs = np.array(summary[lab]["per_seed"]) - ref_ps          # matched per-seed difference
        dm = float(diffs.mean())
        ds = float(diffs.std(ddof=1)) if len(diffs) > 1 else 0.0
        t = dm / (ds / np.sqrt(len(diffs)) + 1e-12)
        npos = int((diffs > 0).sum())
        mtag = "CREDIBLE(marg)" if abs(d) > 0.0064 else "within marg floor"
        # |t|>2.78 ~ p<.05 two-tailed at 4 df (5 seeds); |t|>2.26 at 9 df (10 seeds)
        ptag = "PAIRED-SIG" if abs(t) >= 2.26 else ("paired-borderline" if abs(t) >= 1.8 else "paired n.s.")
        print(f"  {lab:12s} {summary[lab]['composite']:.4f}  Δ={d:+.4f} [{mtag}]   "
              f"paired Δ={dm:+.4f}±{ds:.4f}  t={t:+.2f} [{ptag}]  ({npos}/{len(diffs)} seeds +)")

    with open(out_dir / "tri_metrics.json", "w") as f:
        json.dump({"names": names, "dims": dims, "seeds": seeds, "summary": summary}, f, indent=2)
    print(f"\nsaved {out_dir/'tri_metrics.json'}")


if __name__ == "__main__":
    main()
