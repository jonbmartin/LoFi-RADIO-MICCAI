"""
Cache one frozen backbone's pooled features over a corpus (paper Sec. 2.2).

The backbones are frozen, so their features are deterministic: cache once, probe cheaply. Each 2D
backbone is applied 2.5D -- every foreground slice is encoded independently and the per-slice
embeddings are mean-pooled into one vector per volume. Output is a {files, X, mean, std} .npz
keyed by sample id (the LISA filename), which is what every probe script consumes.

Local weights via --weights / $SSL_WEIGHTS (see ssl_features for the per-backbone recipes; SAM and
SAM2 require a local checkpoint).

    python lofi_radio/cache_teachers.py --sources lisa --teacher dinov2 --out cache/dinov2_lisa.npz
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from ssl_features import load_backbone, extract_volume_features, SSL_BACKBONES
from corpus import UnifiedCorpus


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sources", default="lisa", help="comma list of corpus sources (see corpus.py)")
    ap.add_argument("--teacher", default="dinov2", choices=list(SSL_BACKBONES),
                    help="frozen teacher backbone (dinov2 | biomedclip | sam | ...)")
    ap.add_argument("--out", default=None, help="output .npz (default cache/<teacher>_<sources>.npz)")
    ap.add_argument("--weights", default=None, help="local backbone weights (.safetensors/.pth); $SSL_WEIGHTS")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=64, help="slices per forward (SAM@1024 -> ~8)")
    ap.add_argument("--limit", type=int, default=0, help="cap #volumes (smoke test; 0 = all)")
    args = ap.parse_args()

    sources = [s.strip() for s in args.sources.split(",") if s.strip()]
    out = args.out or os.path.join(_REPO, "cache",
                                   f"{args.teacher}_{'-'.join(sources)}.npz")
    os.makedirs(os.path.dirname(out), exist_ok=True)

    # fail loudly here rather than after a long backbone load
    import torch
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("[cache] CUDA requested but unavailable; pass --device cpu only for a "
                         "tiny --limit smoke test.")

    encoder = load_backbone(args.device, weights=args.weights, backbone=args.teacher)
    corpus = UnifiedCorpus(sources)
    print(f"[cache] teacher={args.teacher} (dim={encoder.dim}) sources={sources} "
          f"available={corpus.summary()} -> {out}")

    files, X, t0, n = [], [], time.time(), 0
    for s in corpus.iter_samples():
        try:
            vol = s.load()
        except Exception as e:                       # one bad file shouldn't sink the cache
            print(f"  skip {s.id}: {type(e).__name__}: {e}"); continue
        X.append(extract_volume_features(vol, encoder, batch=args.batch))
        files.append(s.id)
        n += 1
        if n % 50 == 0:
            print(f"  {n} vols  ({time.time() - t0:.0f}s)")
        if args.limit and n >= args.limit:
            print(f"  [limit] stopping at {n} vols"); break

    X = np.asarray(X, dtype=np.float32)
    assert X.ndim == 2 and X.shape[1] == encoder.dim, f"got {X.shape}, expected (*, {encoder.dim})"
    mean, std = X.mean(0), X.std(0) + 1e-6
    np.savez(out, files=np.array(files), X=X, mean=mean, std=std,
             model=np.array(args.teacher), sources=np.array(sources))
    print(f"[cache] wrote {X.shape[0]} x {X.shape[1]} pooled feats ({args.teacher}) -> {out} "
          f"in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
