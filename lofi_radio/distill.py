"""
LoFi-RADIO multi-teacher distillation (paper Sec. 2.4) -- multi-GPU (DataParallel).

Trains the DINOv2-initialized ViT-S student to match cached frozen teacher targets, summary +
dense, through per-teacher adaptor MLPs.

Loss (AM-RADIO recipe, equal weight per teacher):
    L = sum_t [ (1 - cos(A_sum_t(s_sum),  z_sum_t)) +
                mean_cells[ (1 - cos(A_den_t(s_den), z_den_t)) + SmoothL1(...) ] ]

Teacher targets are standardized before the loss (isotropic per teacher, PHI-S style: subtract the
mean, divide by one scalar = sqrt(mean per-dim variance)). This balances teachers of very
different magnitude and width so that no teacher dominates the SmoothL1 term.

Multi-GPU: with more than one visible CUDA device, the student forward/backward is split across
them via nn.DataParallel -- one process, so the large pooled cache stays in RAM exactly once while
the compute parallelizes. The per-step batch is split across GPUs, so scale --batch up to keep
each GPU busy (e.g. 64 * num_gpus). Pick GPUs via CUDA_VISIBLE_DEVICES; --device cuda:K forces a
single GPU. Targets are held fp16 in RAM and standardized per batch on-device; the math is
identical to standardizing the whole tensor.

    CUDA_VISIBLE_DEVICES=0,1,2,3 python lofi_radio/distill.py --cache cache/dense_full.npz \
        --weights dinov2_vits14.safetensors --batch 256 --epochs 40 --out runs/student_full.pt
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from student import Student


def fit_isotropic(x: np.ndarray):
    """x [..., D] (any dtype) -> (mean [D] fp32 tensor, scale scalar). PHI-S v0 isotropic standardization.
    Stats accumulate in fp32 (dtype=) so a large fp16 cache needs no full fp32 copy."""
    flat = x.reshape(-1, x.shape[-1])
    mean = np.asarray(flat.mean(0, dtype=np.float32), dtype=np.float32)
    var = np.asarray(flat.var(0, dtype=np.float32), dtype=np.float32)
    return torch.from_numpy(mean), float(np.sqrt(var.mean()) + 1e-6)


class Adaptor(nn.Module):
    """Student feature (384) -> a teacher's space. 2-layer MLP (Linear-GELU-Linear), as in AM-RADIO."""
    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        h = max(d_out, d_in)
        self.net = nn.Sequential(nn.Linear(d_in, h), nn.GELU(), nn.Linear(h, d_out))

    def forward(self, x):
        return self.net(x)


def _cos_loss(a, b):
    return (1.0 - F.cosine_similarity(a, b, dim=-1)).mean()


class DistillNet(nn.Module):
    """Student + per-teacher adaptors. forward() standardizes the RAW fp16 targets on-device (via
    registered mean/scale buffers) and returns the summed distillation loss as [1] so DataParallel
    gathers one loss per GPU (we .mean() them in the loop = the full-batch loss when chunks are equal)."""
    def __init__(self, student, teachers, dims, stats, dense_w):
        super().__init__()
        self.student = student
        self.teachers = list(teachers)
        self.dense_w = dense_w
        self.adaptors = nn.ModuleDict()
        for t in teachers:
            ds, dd = dims[t]
            self.adaptors[f"{t}_sum"] = Adaptor(student.dim, ds)
            self.adaptors[f"{t}_den"] = Adaptor(student.dim, dd)
            self.register_buffer(f"{t}__smean", stats[t]["smean"])
            self.register_buffer(f"{t}__dmean", stats[t]["dmean"])
            self.register_buffer(f"{t}__sscale", torch.tensor(stats[t]["sscale"], dtype=torch.float32))
            self.register_buffer(f"{t}__dscale", torch.tensor(stats[t]["dscale"], dtype=torch.float32))

    @staticmethod
    def _to_grid(s_den, G):
        """[B,g,g,C] student dense -> [B,G,G,C] on the teacher's grid. Identity when g==G (the paper's
        setting: every teacher is cached on the student's 16x16 grid); otherwise average-pool."""
        if s_den.shape[1] == G and s_den.shape[2] == G:
            return s_den
        x = s_den.permute(0, 3, 1, 2)                                  # [B,C,g,g]
        x = F.adaptive_avg_pool2d(x, (G, G))
        return x.permute(0, 2, 3, 1).contiguous()

    def forward(self, x, tsum, tden):
        s_sum, s_den = self.student(x)
        loss = x.new_zeros(())
        pooled = {}                                                    # teacher grid -> pooled student dense
        for t in self.teachers:
            zs = (tsum[t].float() - getattr(self, f"{t}__smean")) / getattr(self, f"{t}__sscale")
            zd = (tden[t].float() - getattr(self, f"{t}__dmean")) / getattr(self, f"{t}__dscale")
            loss = loss + _cos_loss(self.adaptors[f"{t}_sum"](s_sum), zs)
            G = int(zd.shape[1])
            if G not in pooled:
                pooled[G] = self._to_grid(s_den, G)
            ad = self.adaptors[f"{t}_den"](pooled[G])
            loss = loss + self.dense_w * (_cos_loss(ad, zd) + F.smooth_l1_loss(ad, zd))
        return loss.unsqueeze(0)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", required=True, help="dense multi-teacher cache (cache_dense.py)")
    ap.add_argument("--weights", default=None, help="DINOv2 init for the student ($SSL_WEIGHTS)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=64, help="GLOBAL batch (split across GPUs under DataParallel)")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--dense_w", type=float, default=1.0, help="weight on the dense term vs summary")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda", help="'cuda' = all visible GPUs (DataParallel); 'cuda:K' = one")
    ap.add_argument("--exclude_ids_file", default=None,
                    help="text file of sample ids (one per line, e.g. a LISA val fold's filenames) to DROP "
                         "from distillation -> leak-free per-fold student (keep LISA-train, exclude val/test)")
    ap.add_argument("--max_steps", type=int, default=0, help="cap optimizer steps (smoke; 0=full)")
    ap.add_argument("--use_teachers", default=None,
                    help="comma subset of the cached teachers to distill from (default: all). Lets a "
                         "smaller-teacher student reuse a richer dense cache, e.g. D+BMC from a D+BMC+Phikon cache.")
    args = ap.parse_args()

    want_cuda = args.device.startswith("cuda") and torch.cuda.is_available()
    ngpu = torch.cuda.device_count() if want_cuda else 0
    use_dp = (args.device == "cuda" and ngpu > 1)
    dev = ("cuda:0" if args.device == "cuda" else args.device) if want_cuda else "cpu"
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    out = args.out or os.path.join(_REPO, "runs",
                                   "student_" + os.path.basename(args.cache).replace("dense_", "").replace(".npz", "") + ".pt")
    os.makedirs(os.path.dirname(out), exist_ok=True)

    z = np.load(args.cache, allow_pickle=True)
    teachers = [str(t) for t in z["teachers"]]
    if args.use_teachers:                                           # distill from a subset of cached teachers
        want = [t.strip() for t in args.use_teachers.split(",") if t.strip()]
        missing = [t for t in want if t not in teachers]
        assert not missing, f"--use_teachers {missing} not in cache teachers {teachers}"
        teachers = want
        print(f"[distill] teacher SUBSET: {teachers}  (cache has {[str(t) for t in z['teachers']]})")
    keep = None
    if args.exclude_ids_file:                                       # leak-free: drop val-fold slices by id
        excl = {l.strip() for l in open(args.exclude_ids_file) if l.strip()}
        ids = np.array([str(x) for x in z["ids"]])
        keep = ~np.isin(ids, list(excl))
        print(f"[distill] exclude_ids: {len(excl)} ids -> dropping {int((~keep).sum())}/{len(ids)} slices")
    imgs = torch.from_numpy(z["imgs"][keep] if keep is not None else z["imgs"])   # fp16 [N,img,img]
    N = imgs.shape[0]
    print(f"[distill] cache={args.cache}  N={N} slices  teachers={teachers}  device={dev}  gpus={ngpu if use_dp else 1}")

    # raw fp16 targets kept in RAM; per-teacher mean/scale computed once (standardize per-batch on-GPU)
    tgt, stats, dims = {}, {}, {}
    student = Student(weights=args.weights)
    for t in teachers:
        s_raw, d_raw = z[f"{t}_summary"], z[f"{t}_dense"]            # fp16 [N,Ds], [N,G,G,Dd]
        if keep is not None:
            s_raw, d_raw = s_raw[keep], d_raw[keep]
        s_mean, s_scale = fit_isotropic(s_raw)
        d_mean, d_scale = fit_isotropic(d_raw)
        tgt[t] = dict(summary=torch.from_numpy(np.ascontiguousarray(s_raw)),
                      dense=torch.from_numpy(np.ascontiguousarray(d_raw)))
        stats[t] = dict(smean=s_mean, sscale=s_scale, dmean=d_mean, dscale=d_scale)
        dims[t] = (s_raw.shape[-1], d_raw.shape[-1])
        print(f"  {t}: summary {tuple(s_raw.shape[1:])} (scale {s_scale:.2f})  dense {tuple(d_raw.shape[1:])} (scale {d_scale:.2f})")

    base = DistillNet(student, teachers, dims, stats, args.dense_w).to(dev)
    net = nn.DataParallel(base) if use_dp else base
    if use_dp:
        print(f"[distill] DataParallel over {ngpu} GPUs -> {args.batch}//{ngpu} = {args.batch // ngpu} slices/GPU")

    opt = torch.optim.AdamW(base.parameters(), lr=args.lr, weight_decay=args.wd)
    steps_per_epoch = (N + args.batch - 1) // args.batch
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * steps_per_epoch)

    t0, step, log = time.time(), 0, []
    for ep in range(args.epochs):
        perm = torch.randperm(N)
        net.train()
        run = 0.0
        for i in range(0, N, args.batch):
            b = perm[i:i + args.batch]
            x = imgs[b][:, None].repeat(1, 3, 1, 1).to(dev, non_blocking=True).float()   # [B,3,img,img]
            tsum_b = {t: tgt[t]["summary"][b].to(dev, non_blocking=True) for t in teachers}
            tden_b = {t: tgt[t]["dense"][b].to(dev, non_blocking=True) for t in teachers}
            loss = net(x, tsum_b, tden_b).mean()                     # gather per-GPU losses -> full-batch loss
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            run += float(loss); step += 1
            if args.max_steps and step >= args.max_steps:
                break
        log.append(run / max(1, steps_per_epoch))
        if ep % 5 == 0 or ep == args.epochs - 1:
            print(f"  ep {ep:3d}  loss {log[-1]:.4f}  lr {sched.get_last_lr()[0]:.2e}  ({time.time()-t0:.0f}s)")
        if args.max_steps and step >= args.max_steps:
            print("  [max_steps] stopping"); break

    torch.save({"student": base.student.state_dict(), "teachers": teachers,
                "img": base.student.img, "arch": base.student.arch,
                "cache": os.path.basename(args.cache)}, out)
    with open(out.replace(".pt", "_log.json"), "w") as f:
        json.dump({"loss": log, "args": vars(args)}, f, indent=2)
    print(f"[distill] saved student -> {out}  (final loss {log[-1]:.4f}, {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
