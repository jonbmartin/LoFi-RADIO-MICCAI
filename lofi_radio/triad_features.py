"""
Triad (3D MRI MAE/SimMIM foundation model) per-VOLUME features -> a compare_probes-compatible cache.

Triad (wangshansong1/Triad) is a 3D MRI vision FM pretrained by MASKED AUTOENCODING (MAE / SimMIM) on a
PlainConvUNet (nnU-Net dynamic_network_architectures) encoder. We extract the multi-scale encoder feature:
`model.encoder(x)` returns per-stage skips, and `model.forward_encs(skips)` GAP-pools each of the 6 stages
(features 32/64/128/256/320/320) and concatenates -> a 1120-d vector. GAP makes the readout size-invariant.

We lean on Triad's own model code (the custom forward_encs lives there). Clone + grab the Google-Drive
weight, then:
    pip install dynamic-network-architectures
    git clone https://github.com/wangshansong1/Triad.git
    # download Triad-PlainConvUNet-MAE.pth from the repo's Google Drive (e.g. `gdown <id>`)
    python lofi_radio/triad_features.py --triad_src Triad \
        --checkpoint Triad-PlainConvUNet-MAE.pth --out cache/triad_lisa.npz
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.ndimage import zoom

from lisa import DATA_DIR, LABELS_CSV, REPO

from corpus import load_canonical_nifti


def robust_norm(vol: np.ndarray, qmin: float = 0.5, qmax: float = 99.5, eps: float = 1e-6) -> np.ndarray:
    """Triad's datasets_three_d normalization: clip to [0.5, 99.5] pct of finite voxels, min-max -> [0,1]."""
    v = vol.astype(np.float32)
    fg = v[np.isfinite(v)]
    lo, hi = np.percentile(fg, qmin), np.percentile(fg, qmax)
    return np.clip((v - lo) / (hi - lo + eps), 0.0, 1.0).astype(np.float32)


def to_input(vol_hwd: np.ndarray, size=(64, 160, 160)) -> np.ndarray:
    """Canonical [H,W,D] (thick last) -> [d0,d1,d2] normalized & resized to the Triad input grid."""
    v = np.moveaxis(vol_hwd, -1, 0)                       # thick axis first -> [D,H,W]
    v = robust_norm(v)
    factors = [size[i] / v.shape[i] for i in range(3)]
    return zoom(v, factors, order=1).astype(np.float32)   # [64,160,160]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--triad_src", required=True, help="path to cloned Triad repo (provides PlainConvUNet_load)")
    ap.add_argument("--checkpoint", required=True, help="Triad-PlainConvUNet-{MAE,SimMIM}.pth")
    ap.add_argument("--data_dir", default=DATA_DIR)
    ap.add_argument("--csv", default=LABELS_CSV)
    ap.add_argument("--out", default=os.path.join(REPO, "cache", "triad_lisa.npz"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--size", default="64,160,160", help="input grid d0,d1,d2 (GAP readout is size-invariant)")
    ap.add_argument("--readout", default="all", choices=["all", "bottleneck"],
                    help="'all' = forward_encs 6-stage GAP concat (1120-d, Triad default); 'bottleneck' = "
                         "GAP the deepest stage only (320-d) -> isolates whether the multi-scale concat is "
                         "diluting a usable bottleneck")
    ap.add_argument("--limit", type=int, default=0, help="cap #volumes (smoke; 0=all)")
    args = ap.parse_args()

    dev = args.device if (not args.device.startswith("cuda") or torch.cuda.is_available()) else "cpu"
    size = tuple(int(s) for s in args.size.split(","))
    if args.triad_src not in sys.path:
        sys.path.insert(0, args.triad_src)
    from PlainConvUNet_load import get_Plain_nnUNet  # their model w/ encoder + forward_encs

    model = get_Plain_nnUNet(num_input_channels=1).to(dev).eval()
    ckpt = torch.load(args.checkpoint, map_location=dev, weights_only=False)
    sd = ckpt.get("state_dict", ckpt.get("model", ckpt)) if isinstance(ckpt, dict) else ckpt
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[triad] {args.checkpoint}: loaded (missing={len(missing)}, unexpected={len(unexpected)})")
    if missing:  # show names: norm running-stats/projection = benign; core encoder convs = real mismatch
        print("  missing keys:", list(missing)[:25])
    if len(missing) > 50:
        print("  WARNING: many missing keys -> arch/checkpoint mismatch; features may be ~random.")
    for p in model.parameters():
        p.requires_grad_(False)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    @torch.no_grad()
    def embed(x):
        try:
            encs = model.encoder(x)                       # per-stage skips (documented feature path)
        except Exception:
            encs = model(x)                               # fallback to QuickStart's call
        if args.readout == "bottleneck":                  # deepest stage only (320-d, MAE semantic level)
            return F.adaptive_avg_pool3d(encs[-1], 1).flatten(1)
        return model.forward_encs(encs)                   # GAP all 6 stages + concat -> [B,1120]

    df = pd.read_csv(args.csv)
    files, feats, skipped = [], [], 0
    for fn in df["filename"]:
        path = os.path.join(args.data_dir, fn)
        if not os.path.exists(path):
            skipped += 1
            continue
        x = torch.from_numpy(to_input(load_canonical_nifti(path), size))[None, None].to(dev)  # [1,1,d0,d1,d2]
        f = embed(x)
        feats.append(np.asarray(f.detach().cpu(), np.float32).reshape(-1))
        files.append(str(fn))
        if args.limit and len(files) >= args.limit:
            print(f"  [limit] stopping at {len(files)} vols"); break

    X = np.stack(feats).astype(np.float32)
    np.savez(args.out, files=np.array(files), X=X,
             mean=X.mean(0), std=X.std(0) + 1e-6, model=f"triad_{args.readout}")
    print(f"[triad] {len(files)} vols (skipped {skipped} missing) -> {args.out}  X={X.shape}  "
          f"next: screen_probe.py / compare_probes.py vs dinov2")


if __name__ == "__main__":
    main()
