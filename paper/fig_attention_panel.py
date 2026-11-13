"""
Per-artifact discriminative-attention panel (paper Fig. 2).

Rows   : Original / DINOv2 attention / LoFi-RADIO attention
Columns: one volume per artifact, chosen via the PANEL / PINS config below.

Method: a training-free severe-prototype direction alpha (mean[grade-2] - mean[grade-0] in
z-scored patch-mean feature space); each patch token of the displayed slice is projected onto
alpha. Because the probe pools patch tokens, this is the spatial decomposition of the near-linear
severity readout. The displayed grayscale slice and the attention map are the same slice.

Each volume's pooled feature is artifact-independent, so every LISA volume is pooled once per
backbone and all artifact directions are derived from that matrix.

    python paper/fig_attention_panel.py --student_ckpt runs/student_full.pt \\
        --weights "dinov2=dinov2_vits14.safetensors" --device cuda

Outputs: paper/fig_attention_panel.pdf + .png
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "lofi_radio"))

from lisa import ARTIFACTS, DATA_DIR, LABELS_CSV, load_labels  # noqa: E402
from ssl_features import load_backbone, _slice_to_unit          # noqa: E402
from student import Student                                     # noqa: E402
from teachers import _dinov2_dense                              # noqa: E402


# ── backbone forwards: unit-range [B,3,img,img] -> (summary, dense [B,D,G,G]) ──
def _student_forward(model, x):
    """Adapter matching the (summ, [1,D,G,G]) contract for _extract."""
    x = (x - model.mean) / model.std
    feats = model.model.forward_features(x)
    p = getattr(model.model, "num_prefix_tokens", 1)
    patch = feats[:, p:]
    G = int(round(patch.shape[1] ** 0.5))
    dense = patch.transpose(1, 2).reshape(x.shape[0], -1, G, G)
    return feats[:, 0], dense


def _enc_forward_dinov2(enc, x):
    x = (x - enc.mean) / enc.std
    return _dinov2_dense(enc.model, x)


# ── slice selection, pooling, and the severity-direction projection ───────────
def foreground_slices(nifti_path: str, fg_frac: float = 0.02) -> tuple[np.ndarray, np.ndarray]:
    """Return (unit_slices [S,H,W], thr) for the foreground in-plane slices (thick axis last)."""
    vol = nib.load(nifti_path).get_fdata(dtype=np.float32)
    D = vol.shape[-1]
    thr = 0.10 * float(np.percentile(vol, 99.0))
    keep = [d for d in range(D) if (vol[..., d] > thr).mean() > fg_frac] or list(range(D))
    units = np.stack([_slice_to_unit(vol[..., d]) for d in keep]).astype(np.float32)
    return units, thr


@torch.no_grad()
def pool_volume(fwd, model, dev, unit_slices: np.ndarray, img: int, batch: int = 24) -> np.ndarray:
    """Patch-mean pooled feature for a volume: mean over patches per slice, then mean over slices -> [D]."""
    feats = []
    for i in range(0, len(unit_slices), batch):
        sl = unit_slices[i:i + batch]
        x = torch.from_numpy(sl)[:, None].repeat(1, 3, 1, 1)
        x = F.interpolate(x, size=(img, img), mode="bilinear", align_corners=False).to(dev)
        _, dense = fwd(model, x)                       # dense [B,D,G,G]
        feats.append(dense.mean(dim=(2, 3)).cpu().float())   # [B,D] patch-mean per slice
    return torch.cat(feats, 0).mean(0).numpy()


@torch.no_grad()
def patch_grid(fwd, model, dev, unit_slice: np.ndarray, img: int) -> tuple[np.ndarray, int]:
    """Patch tokens for one display slice -> (patches [G*G,D], G)."""
    x = torch.from_numpy(unit_slice)[None, None].repeat(1, 3, 1, 1)
    x = F.interpolate(x, size=(img, img), mode="bilinear", align_corners=False).to(dev)
    _, dense = fwd(model, x)                            # [1,D,G,G]
    G = dense.shape[-1]
    return dense[0].permute(1, 2, 0).reshape(G * G, -1).cpu().float().numpy(), G


def severe_direction(feats: np.ndarray, labels: np.ndarray):
    """feats [N,D], labels [N] in {0,1,2}. Returns (mu, sigma, alpha_hat) in z-scored space:
    alpha = mean(z | label==2) - mean(z | label==0); L2-normalised."""
    mu = feats.mean(0)
    sigma = feats.std(0) + 1e-6
    z = (feats - mu) / sigma
    sev = z[labels == 2].mean(0)
    cln = z[labels == 0].mean(0)
    alpha = sev - cln
    return mu, sigma, alpha / (np.linalg.norm(alpha) + 1e-9)


def attention_map(patches: np.ndarray, G: int, mu, sigma, alpha_hat, display: int):
    """map_ij = ((token-mu)/sigma)·alpha_hat onto the severe direction, normalised to [0,1].
    Default: ReLU (severe-like) + 99th-pct scale. Fallback: if the whole slice projects clean-ward
    (ReLU empties the map -> would render BLACK, as happens for a weak/noisy direction like the
    student's Motion class), show the RELATIVE landscape of the signed map instead of nothing."""
    z = (patches - mu) / sigma                         # [G*G,D]
    raw = (z @ alpha_hat).reshape(G, G)                # signed: + = toward severe prototype
    pos = np.maximum(raw, 0.0)
    hi = np.percentile(pos, 99.0)
    if hi < 1e-6:                                       # ReLU empty -> relative stretch of signed map
        lo, hi2 = np.percentile(raw, 5.0), np.percentile(raw, 99.0)
        m = np.clip((raw - lo) / max(hi2 - lo, 1e-6), 0, 1)
    else:                                              # unchanged behaviour when there is + signal
        m = np.clip(pos / hi, 0, 1)
    t = torch.from_numpy(m.astype(np.float32))[None, None]
    return F.interpolate(t, size=(display, display), mode="bilinear", align_corners=False)[0, 0].numpy()


def overlay(slice_gray: np.ndarray, heat: np.ndarray, alpha_max: float = 0.75) -> np.ndarray:
    """Grayscale slice + 'inferno' heat overlay (alpha ∝ heat). Returns [H,W,3] float in [0,1]."""
    base = np.repeat(slice_gray[..., None], 3, axis=2)
    hot = cm.get_cmap("inferno")(heat)[:, :, :3]
    a = (heat * alpha_max)[..., None]
    return np.clip(base * (1 - a) + hot * a, 0, 1)


_IMG = 448   # 32×32 patch grid (matches what looked best in the per-artifact runs)

# ── panel layout — edit to retune a column ───────────────────────────────────
# (artifact, example_index, orientation_filter, slice_frac)
#   example_index   : which severe example, sorted by 'fewest other severe artifacts' 
#   orientation     : restrict to 'axi'/'cor'/'sag' filenames, or None
#   slice_frac      : 0..1 position through the foreground slices (0.5 = middle)
PANEL = [
    ("Noise",       1, None,  0.50),
    ("Zipper",      1, None,  0.50),
    ("Positioning", 0, None,  0.50),   # PINNED: see PINS below (idx/orient/frac ignored here)
    ("Banding",     0, None,  0.50),
    ("Motion",      1, None,  0.50),
    ("Distortion",  1, None,  0.50),   # PINNED: see PINS below (idx/frac ignored here)
]

# Some artifact columns are PINNED to a hand-vetted CLEAN example (all six OTHER artifacts graded 0) at a
# specific foreground-slice index, so the column shows that artifact ALONE and does not conflate it with a
# co-occurring one. Maps artifact -> (filename, absolute foreground-slice index); overrides the auto picker.
#   Positioning: the auto picker (find_positioning_example) maximized off-centering regardless of
#                co-artifacts and landed on a slice with SEVERE noise. 0037_cor @ 38 is a clean mid-brain
#                slice with the brain pressed against the left edge (large empty gap on the right).
#   Distortion : 0067_axi @ 14 shows the clearest geometric warp (whole head tilted/stretched to an
#                ellipse), a cleaner read than the previous grade-2 pick (2004_cor, lumpy).
PINS = {
    "Positioning": ("LISA_0037_LF_cor.nii.gz", 38),
    "Distortion":  ("LISA_0067_LF_axi.nii.gz", 14),
}


def select_one(df: pd.DataFrame, nifti_dir: str, art: str, index: int,
               orientation: str | None) -> tuple[str, int]:
    """The `index`-th severe (grade-2, else grade-1) example for `art`, sorted by fewest other
    severe artifacts. Optional orientation filter."""
    other = [a for a in ARTIFACTS if a != art]
    for grade in (2, 1):
        sub = df[df[art] == grade].copy()
        if orientation:
            sub = sub[sub["filename"].str.contains(f"_{orientation}")]
        if sub.empty:
            continue
        sub["other_severe"] = sub[other].max(axis=1)
        sub = sub.sort_values("other_severe", ascending=True, kind="stable")
        paths = [(str(r["filename"]), grade) for _, r in sub.iterrows()
                 if (Path(nifti_dir) / str(r["filename"])).exists()]
        if paths:
            return paths[min(index, len(paths) - 1)]
    raise SystemExit(f"[panel] no example found for {art} (orientation={orientation})")


def pick_slice(nifti_path: str, slice_frac: float, display: int):
    """One foreground slice for BOTH the grayscale and the attention map. Returns (native [H,W], disp [d,d])."""
    units, _ = foreground_slices(nifti_path)              # [S,H,W] all foreground, in volume order
    idx = int(round(slice_frac * (len(units) - 1)))
    return _to_disp(units[idx], display)


def _to_disp(sl: np.ndarray, display: int):
    t = torch.from_numpy(sl)[None, None]
    disp = F.interpolate(t, size=(display, display), mode="bilinear", align_corners=False)[0, 0].numpy()
    return sl, disp


def _positioning_score(sl: np.ndarray) -> float:
    """Heuristic 'off-centering' for a unit slice = brain-centroid offset from image centre
    + 2× boundary contact (fraction of foreground in the outer 8% margin). Higher = more obvious
    positioning issue. Used to auto-pick a convincing Positioning example."""
    H, W = sl.shape
    fg = sl > 0.10
    if fg.sum() < 0.01 * fg.size:
        return -1.0
    ys, xs = np.where(fg)
    offset = float(np.hypot((ys.mean() - H / 2) / (H / 2), (xs.mean() - W / 2) / (W / 2)))
    my, mx = max(1, int(0.08 * H)), max(1, int(0.08 * W))
    border = np.zeros_like(fg)
    border[:my] = border[-my:] = True
    border[:, :mx] = border[:, -mx:] = True
    contact = float((fg & border).sum() / max(int(fg.sum()), 1))
    return offset + 2.0 * contact


def find_positioning_example(df: pd.DataFrame, nifti_dir: str, display: int):
    """Scan grade-2 (else grade-1) Positioning volumes, return the (filename, grade, native, disp) of the
    slice with the most obvious off-centering — ANY orientation. Replaces the generic picker for that column."""
    for grade in (2, 1):
        cand = [f for f in df[df["Positioning"] == grade]["filename"]
                if (Path(nifti_dir) / f).exists()]
        if not cand:
            continue
        best = (-1.0, None, None)
        for f in cand:
            units, _ = foreground_slices(str(Path(nifti_dir) / f))
            for sl in units:
                s = _positioning_score(sl)
                if s > best[0]:
                    best = (s, f, sl)
        _, fn, sl = best
        nat, disp = _to_disp(sl, display)
        return fn, grade, nat, disp
    raise SystemExit("[panel] no Positioning example found")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--student_ckpt", required=True)
    ap.add_argument("--nifti_dir", default=DATA_DIR)
    ap.add_argument("--csv", default=LABELS_CSV)
    ap.add_argument("--weights", default="", help='"dinov2=path" (adds the DINOv2 row)')
    ap.add_argument("--display", type=int, default=336)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="paper/fig_attention_panel")
    args = ap.parse_args()

    dev = args.device if torch.cuda.is_available() else "cpu"
    df = load_labels(args.csv, args.nifti_dir)
    wmap = {}
    for kv in (args.weights or "").split(","):
        if kv.strip() and "=" in kv:
            k_, v_ = kv.split("=", 1); wmap[k_.strip()] = v_.strip()

    # ── models ───────────────────────────────────────────────────────────────
    ckpt = torch.load(args.student_ckpt, map_location="cpu")
    student = Student(weights=None, from_pretrained=False)
    student.load_state_dict(ckpt["student"]); student.eval().to(dev)
    # order = render/label order top→bottom under "Original"; LoFi-RADIO last so it sits on the bottom
    models = []
    if "dinov2" in wmap:
        enc = load_backbone(dev, weights=wmap["dinov2"], backbone="dinov2")
        models.append(("DINOv2", enc, _enc_forward_dinov2))
    models.append(("LoFi-RADIO", student, _student_forward))

    # ── choose one volume + slice per artifact column ────────────────────────
    cols = []
    for art, idx, orient, frac in PANEL:
        if art in PINS:
            fn, sidx = PINS[art]
            grade = int(df.loc[df["filename"] == fn, art].iloc[0])
            units, _ = foreground_slices(str(Path(args.nifti_dir) / fn))
            sl_native, disp = _to_disp(units[sidx], args.display)
            print(f"[panel] {art:12s} -> {fn} (grade {grade}, pinned clean slice {sidx}/{len(units)-1})")
        else:
            fn, grade = select_one(df, args.nifti_dir, art, idx, orient)
            sl_native, disp = pick_slice(str(Path(args.nifti_dir) / fn), frac, args.display)
            print(f"[panel] {art:12s} -> {fn} (grade {grade}, slice_frac {frac})")
        cols.append(dict(art=art, fn=fn, grade=grade, sl=sl_native, disp=disp))

    # ── figure skeleton ──────────────────────────────────────────────────────
    row_labels = ["Original"] + [m[0] for m in models]
    n_rows, n_cols = len(row_labels), len(cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.5, n_rows * 2.6), squeeze=False)
    fig.patch.set_facecolor("#0d0d0d")
    for axr in axes:
        for a in axr:
            a.set_facecolor("#0d0d0d"); a.axis("off")
    for ri, lab in enumerate(row_labels):
        axes[ri][0].set_ylabel(lab, color="#eee", fontsize=26, rotation=90,
                               labelpad=12, va="center", fontweight="bold")
        axes[ri][0].axis("on")
        axes[ri][0].set_xticks([]); axes[ri][0].set_yticks([])
        for sp in axes[ri][0].spines.values():
            sp.set_visible(False)

    # row 0: originals + column titles
    for ci, c in enumerate(cols):
        axes[0][ci].imshow(c["disp"], cmap="gray", vmin=0, vmax=1)
        axes[0][ci].set_title(f"{c['art']}\n(grade {c['grade']})", color="#eee",
                              fontsize=22, pad=5, fontweight="bold")

    # ── per model: pool all vols once, derive each artifact direction, render ─
    for mi, (label, model, fwd) in enumerate(models, start=1):
        print(f"\n[panel] {label}: pooling {len(df)} volumes (once) ...")
        feats = []
        for fn in df["filename"]:
            units, _ = foreground_slices(str(Path(args.nifti_dir) / fn))
            feats.append(pool_volume(fwd, model, dev, units, _IMG))
        F_all = np.stack(feats)                                       # [N,384]

        for ci, c in enumerate(cols):
            mu, sigma, alpha = severe_direction(F_all, df[c["art"]].to_numpy())
            patches, G = patch_grid(fwd, model, dev, c["sl"], _IMG)
            heat = attention_map(patches, G, mu, sigma, alpha, args.display)
            axes[mi][ci].imshow(overlay(c["disp"], heat))

    # (suptitle removed — redundant with the LaTeX caption and it constrained vertical space)
    plt.tight_layout(pad=0.4)
    for ext in ("pdf", "png"):
        plt.savefig(f"{args.out}.{ext}", bbox_inches="tight", dpi=180, facecolor=fig.get_facecolor())
        print(f"saved {args.out}.{ext}")


if __name__ == "__main__":
    main()
