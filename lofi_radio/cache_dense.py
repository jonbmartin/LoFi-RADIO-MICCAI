"""
Dense multi-teacher feature cache for distillation.

One pass over the corpus. For each kept (foreground) slice we cache:
  - the unit-range image at the student resolution (224, 1-channel; replicated to 3 channels at
    train time)
  - per teacher: a summary vector + a dense [grid,grid,Dd] grid, all teachers aligned slice-for-slice

Caching all teachers in a single pass guarantees alignment and loads each volume once. The student
later trains on `imgs` and regresses these frozen targets. Everything is fp16 to halve disk; the
full pooled corpus is still tens of GB.

Pass local weights with --weights "name=path,..." to avoid hub downloads (see ssl_features for the
per-backbone recipes). Slice normalization follows $SLICE_NORM (see ssl_features._slice_to_unit);
set it identically here and in export_student.py / submit_student.py.

    python lofi_radio/cache_dense.py --sources lisa,m4raw,lf64mt,ds006557 \
        --teachers dinov2,biomedclip,phikon \
        --weights "dinov2=dinov2_vits14.safetensors,phikon=phikon/model.safetensors" \
        --out cache/dense_full.npz
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from ssl_features import _slice_to_unit
from corpus import UnifiedCorpus
from teachers import TeacherWithDense


def _parse_weights(s: str) -> dict:
    out = {}
    for kv in (s or "").split(","):
        kv = kv.strip()
        if kv:
            k, v = kv.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def _foreground_slices(native: np.ndarray, fg_frac: float, img: int) -> np.ndarray:
    """[H,W,D] -> [S,img,img] unit images for the foreground in-plane slices (same keep rule as
    ssl_features.extract_volume_slices), resized to the student resolution."""
    D = native.shape[2]
    thr = 0.10 * float(np.percentile(native, 99.0))
    keep = [d for d in range(D) if (native[:, :, d] > thr).mean() > fg_frac] or list(range(D))
    units = np.stack([_slice_to_unit(native[:, :, d]) for d in keep])      # [S,H,W]
    t = torch.from_numpy(units)[:, None]                                   # [S,1,H,W]
    t = F.interpolate(t, size=(img, img), mode="bilinear", align_corners=False)
    return t[:, 0].numpy().astype(np.float32)                             # [S,img,img]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sources", default="lisa")
    ap.add_argument("--teachers", default="dinov2,biomedclip,phikon")
    ap.add_argument("--weights", default="", help='comma list name=path, e.g. "dinov2=...,phikon=..."')
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--grid", type=int, default=16)
    ap.add_argument("--img", type=int, default=224, help="student input resolution (cached image size)")
    ap.add_argument("--fg_frac", type=float, default=0.02)
    ap.add_argument("--batch", type=int, default=32, help="slices per forward for 224-px teachers")
    ap.add_argument("--sam_batch", type=int, default=4,
                    help="slices per forward for 1024-px teachers (sam); global attn is "
                         "batch*heads*4096*4096 -> OOMs fast, keep small")
    ap.add_argument("--limit", type=int, default=0, help="cap #volumes (smoke; 0=all)")
    args = ap.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("[dense] CUDA requested but unavailable (use --device cpu with a tiny "
                         "--limit for a smoke test).")

    sources = [s.strip() for s in args.sources.split(",") if s.strip()]
    names = [t.strip() for t in args.teachers.split(",") if t.strip()]
    wmap = _parse_weights(args.weights)
    out = args.out or os.path.join(_REPO, "cache",
                                   f"dense_{'-'.join(names)}_{'-'.join(sources)}.npz")
    os.makedirs(os.path.dirname(out), exist_ok=True)

    teachers = {n: TeacherWithDense(n, device=args.device, weights=wmap.get(n), grid=args.grid)
                for n in names}
    corpus = UnifiedCorpus(sources)
    print(f"[dense] teachers={names} sources={sources} avail={corpus.summary()} grid={args.grid} -> {out}")

    imgs, ids = [], []
    summ = {n: [] for n in names}
    dense = {n: [] for n in names}
    t0, nv = time.time(), 0
    for s in corpus.iter_samples():
        try:
            units = _foreground_slices(s.load(), args.fg_frac, args.img)      # [S,img,img]
        except Exception as e:
            print(f"  skip {s.id}: {type(e).__name__}: {e}"); continue
        x = torch.from_numpy(units)[:, None].repeat(1, 3, 1, 1)               # [S,3,img,img] unit-range
        for n, t in teachers.items():
            tb = args.sam_batch if t.enc.img >= 1024 else args.batch          # 1024-px teachers OOM at big batch
            su, de = [], []
            for i in range(0, x.shape[0], tb):
                a, b = t(x[i:i + tb])
                su.append(a); de.append(b)
            summ[n].append(np.concatenate(su, 0))
            dense[n].append(np.concatenate(de, 0))
        imgs.append(units.astype(np.float16))
        ids.extend([s.id] * units.shape[0])
        nv += 1
        if nv % 25 == 0:
            print(f"  {nv} vols  {len(ids)} slices  ({time.time() - t0:.0f}s)")
        if args.limit and nv >= args.limit:
            print(f"  [limit] stopping at {nv} vols"); break

    save = dict(imgs=np.concatenate(imgs, 0), ids=np.array(ids),
                grid=args.grid, img=args.img, teachers=np.array(names))
    for n in names:
        save[f"{n}_summary"] = np.concatenate(summ[n], 0).astype(np.float16)
        save[f"{n}_dense"] = np.concatenate(dense[n], 0).astype(np.float16)
    np.savez(out, **save)
    gb = os.path.getsize(out) / 1e9
    print(f"[dense] wrote {len(ids)} slices x {len(names)} teachers -> {out} "
          f"({gb:.2f} GB) in {time.time() - t0:.0f}s")
    for n in names:
        print(f"        {n}: summary {save[f'{n}_summary'].shape}  dense {save[f'{n}_dense'].shape}")


if __name__ == "__main__":
    main()
