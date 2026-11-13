"""
Export the distilled student's pooled features over LISA -> a probe cache.

Runs the trained student over the LISA volumes and mean-pools per-slice embeddings into one 384-d
vector per volume, keyed by filename -- the same schema cache_teachers.py writes, so the result
drops into every probe script.

Pooling default = 'cls' (per-slice summary/CLS token, mean over slices), matching how the frozen
DINOv2 cache is built. '--pool patch' uses the spatial mean of patch tokens instead.

    python lofi_radio/export_student.py --ckpt runs/student_full.pt   # -> cache/student_full_cls_lisa.npz
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch

from corpus import UnifiedCorpus
from student import Student
from cache_dense import _foreground_slices


@torch.no_grad()
def pool_volume(student: Student, native: np.ndarray, pool: str, fg_frac: float,
                batch: int, dev: str) -> np.ndarray:
    units = _foreground_slices(native, fg_frac, student.img)        # [S,img,img]
    x = torch.from_numpy(units)[:, None].repeat(1, 3, 1, 1)         # [S,3,img,img]
    feats = []
    for i in range(0, x.shape[0], batch):
        summ, dense = student(x[i:i + batch].to(dev))
        feats.append((summ if pool == "cls" else dense.mean(dim=(1, 2))).cpu())
    return torch.cat(feats, 0).mean(0).numpy().astype(np.float32)    # mean over slices -> [384]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True, help="distilled student .pt (distill.py)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--sources", default="lisa", help="usually lisa (the labeled eval domain)")
    ap.add_argument("--pool", default="cls", choices=["cls", "patch"])
    ap.add_argument("--fg_frac", type=float, default=0.02)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--limit", type=int, default=0, help="cap #volumes (smoke; 0=all)")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    dev = args.device if (not args.device.startswith("cuda") or torch.cuda.is_available()) else "cpu"
    out = args.out
    if not out:                                                     # runs/<name>.pt -> cache/<name>_<pool>_lisa.npz
        d, b = os.path.split(os.path.abspath(args.ckpt))
        if os.path.basename(d) == "runs":
            d = os.path.join(os.path.dirname(d), "cache")
        out = os.path.join(d, b.replace(".pt", f"_{args.pool}_lisa.npz"))
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    student = Student(weights=None, from_pretrained=False,          # arch only (no download); weights from ckpt
                      arch=ckpt.get("arch", "vits"))
    student.load_state_dict(ckpt["student"])
    student.eval().to(dev)
    print(f"[export] ckpt={args.ckpt} teachers={ckpt.get('teachers')} pool={args.pool} -> {out}")

    files, X, t0 = [], [], time.time()
    for s in UnifiedCorpus(args.sources.split(",")).iter_samples():
        try:
            X.append(pool_volume(student, s.load(), args.pool, args.fg_frac, args.batch, dev))
            files.append(s.id)
        except Exception as e:
            print(f"  skip {s.id}: {type(e).__name__}: {e}")
        if len(files) % 100 == 0 and files:
            print(f"  {len(files)} vols  ({time.time() - t0:.0f}s)")
        if args.limit and len(files) >= args.limit:
            break
    X = np.asarray(X, dtype=np.float32)
    np.savez(out, files=np.array(files), X=X, mean=X.mean(0), std=X.std(0) + 1e-6,
             model=np.array(f"lofi-radio-student/{args.pool}"))
    print(f"[export] wrote {X.shape[0]} x {X.shape[1]} student feats -> {out} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
