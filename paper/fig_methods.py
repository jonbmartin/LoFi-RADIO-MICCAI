"""
LoFi-RADIO methods diagram (Figure 1) — image-rich, distillation-explicit redesign.

Phase 1 (Distillation): the SAME unlabeled low-field slice is encoded in PARALLEL by three FROZEN
teachers (DINOv2 / BiomedCLIP / Phikon, shown by their pretraining-domain thumbnails) and by a
TRAINABLE ViT-S student with its own architecture. Per-teacher adaptor heads map the student's
features into each teacher's space; a distillation loss (PHI-S standardized, cosine+SmoothL1) pulls
each student prediction toward the corresponding teacher target. The teachers TEACH the student —
they are not part of it.
Phase 2 (Grading): the trained student is FROZEN, mean-pooled to 384-d, and a tuned MLP probe
predicts a 0/1/2 severity grade per artifact from a real ULF slice.

Embeds rasters from paper/assets/ — run paper/_methods_assets.py first.

Outputs paper/fig_methods.pdf (+ .png)
    python paper/_methods_assets.py && python paper/fig_methods.py
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import FancyBboxPatch
plt.rcParams.update({"font.family": "DejaVu Sans"})

_PAPER = os.path.dirname(os.path.abspath(__file__))
A = os.path.join(_PAPER, "assets")

# ── palette ───────────────────────────────────────────────────────────────────
C_DINO = "#4E79A7"; C_BMC = "#F28E2B"; C_PHI = "#AF7AA1"      # teachers / adaptors
C_STU  = "#59A14F"; C_PROBE = "#9C6ADE"; C_CORP = "#76B7B2"   # student / probe / corpus
C_GRADE = "#5B6770"; C_ARROW = "#5a5a5a"; C_PANEL = "#f4f5f7"; C_LOSS = "#C0504D"
INK = "#2b2b2b"

# Portrait layout: Phase 1 (distillation) on top at full width, Phase 2 (grading) as a
# full-width horizontal strip directly below it. Phase-2 coordinates are written natively.
fig, ax = plt.subplots(figsize=(8.6, 8.3))
ax.set_xlim(0, 9.25); ax.set_ylim(-2.85, 6.05); ax.axis("off")


def box(x, y, w, h, color, alpha=0.16, ec=None, lw=1.6, shadow=True, z=2, rounding=0.12):
    if shadow:
        ax.add_patch(FancyBboxPatch((x + 0.05, y - 0.06), w, h,
                     boxstyle=f"round,pad=0.02,rounding_size={rounding}",
                     facecolor="#000000", alpha=0.07, ec="none", zorder=z - 1))
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                 boxstyle=f"round,pad=0.02,rounding_size={rounding}",
                 facecolor=color, alpha=alpha, edgecolor=ec or color, lw=lw, zorder=z))
    return (x + w / 2, y + h / 2)


def panel(x, y, w, h, title):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.15",
                 facecolor=C_PANEL, edgecolor="#d7dade", lw=1.2, zorder=0))
    ax.text(x + 0.22, y + h - 0.16, title, fontsize=11, fontweight="bold",
            color="#8a929b", zorder=1, va="center", ha="left")


def txt(x, y, s, size=9, color=INK, weight="normal", style="normal", ha="center", z=6):
    ax.text(x, y, s, ha=ha, va="center", fontsize=size, color=color,
            fontweight=weight, fontstyle=style, zorder=z)


def arrow(x0, y0, x1, y1, col=C_ARROW, lw=1.7, ls="-"):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=col, lw=lw, ls=ls,
                                mutation_scale=12, shrinkA=0, shrinkB=0), zorder=4)


def line(xs, ys, col=C_ARROW, lw=1.6, ls="-", z=3):
    ax.plot(xs, ys, color=col, lw=lw, ls=ls, zorder=z, solid_capstyle="round")


def img(path, x, y, w, h, ec=None, lw=1.4, z=5):
    a = ax.inset_axes([x, y, w, h], transform=ax.transData, zorder=z)
    a.imshow(mpimg.imread(path)); a.axis("off")
    if ec:
        a.add_patch(plt.Rectangle((0, 0), 1, 1, transform=a.transAxes,
                    fill=False, ec=ec, lw=lw, zorder=z + 1))
    return a


FROZEN_BG, FROZEN_FG = "#dCE8F5", "#3a6ea5"
TRAIN_BG,  TRAIN_FG  = "#dceedd", "#3f8a44"


def badge(cx, cy, label, fg, bg, w=0.66, h=0.21):
    """Small status pill centred at (cx,cy)."""
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                 boxstyle="round,pad=0.01,rounding_size=0.10",
                 facecolor=bg, edgecolor=fg, lw=1.0, zorder=7))
    ax.text(cx, cy, label, ha="center", va="center", fontsize=6.0,
            color=fg, fontweight="bold", zorder=8)


# ═══ PHASE PANELS ═════════════════════════════════════════════════════════════
panel(0.15, 0.90, 8.95, 5.10, "1   Distillation")
panel(0.15, -2.65, 8.95, 3.20, "2   Severity grading")

ROWS = [4.48, 3.04, 1.60]   # the three teacher / adaptor / loss rows (top→bottom); lowered to clear
                            #   the header band at the top of Phase 1

# ═══ PHASE 1 ══════════════════════════════════════════════════════════════════
# ── corpus: a single "low-field corpus" block (montage + label) ──────────────
box(0.22, 2.05, 1.90, 2.60, C_CORP, alpha=0.12, ec=C_CORP, lw=2.0, shadow=False)
txt(1.17, 4.43, "Low-field MRI\ncorpus", size=8.5, weight="bold", color=C_CORP)
img(os.path.join(A, "mr_corpus_montage.png"), 0.47, 2.75, 1.40, 1.40, ec=C_CORP, lw=2.0)
txt(1.17, 2.32, "unlabeled · multi-site", size=6.4, color="#6a6a6a", style="italic")

# ── student (trainable) with its OWN architecture ────────────────────────────
sx, sy, sw, sh = 2.35, 2.45, 1.78, 1.46
box(sx, sy, sw, sh, C_STU, alpha=0.12, ec=C_STU, lw=2.4)
txt(sx + sw / 2, sy + sh - 0.20, "Student", size=8.8, weight="bold", color=C_STU)
txt(sx + sw / 2, sy + sh - 0.40, "ViT-S/14", size=6.6, color="#5a5a5a")
badge(sx + sw / 2, sy +0.175, "✎ trainable", TRAIN_FG, TRAIN_BG, w=0.72)
# mini architecture glyph: patch-embed bar + transformer blocks
gy0, gy1 = sy + 0.35, sy + 0.9
bx = sx + 0.18
ax.add_patch(plt.Rectangle((bx, gy0), 0.16, gy1 - gy0, facecolor=C_STU, alpha=0.55,
                           ec="white", lw=0.8, zorder=5))
for k in range(4):
    rx = bx + 0.30 + k * 0.27
    ax.add_patch(plt.Rectangle((rx, gy0), 0.20, gy1 - gy0, facecolor=C_STU, alpha=0.30,
                               ec=C_STU, lw=1.0, zorder=5))
# txt(sx + sw / 2, gy0 - 0.16, "patch-embed → transformer blocks", size=5.6, color="#6a6a6a")
# corpus -> student (same image)
arrow(2.12, 3.18, sx, 3.18, col=C_CORP, lw=1.7)

# ── per-teacher adaptor heads (belong to the student path) ───────────────────
TEACH = [(C_DINO, "DINOv2",     "ViT-S/14", "",    "dom_dinov2.png"),
         (C_BMC,  "BiomedCLIP",  "ViT-B/16", "", "dom_biomedclip.png"),
         (C_PHI,  "Phikon",      "ViT-B/16", "",  "dom_phikon.png")]
adx, adw, adh = 4.62, 1.02, 0.50
brx = 4.34                                                       # branch point of the 1->3 fan
line([sx + sw, brx], [3.18, 3.18], col=C_STU, lw=1.6)           # student feature stub
line([brx, brx], [ROWS[0], ROWS[2]], col=C_STU, lw=1.6)        # vertical branch bus
for (col, *_), ry in zip(TEACH, ROWS):
    box(adx, ry - adh / 2, adw, adh, col, alpha=0.22, lw=1.4, rounding=0.08)
    txt(adx + adw / 2, ry, "adaptor", size=7.4, weight="bold", color=col)
    arrow(brx, ry, adx, ry, col=C_STU, lw=1.6)                  # bus -> each adaptor (clean horizontals)

# ── distillation-loss couplers (adaptor ≈ teacher) ───────────────────────────
lx = 5.92
txt(lx - 0.15, 5, "Distillation loss", size=7.5, weight="bold", color=C_LOSS)
# txt(lx, 5.17, "cosine + SmoothL1 · PHI-S", size=6.2, color="#8a6a6a", style="italic")
for ry in ROWS:
    ax.add_patch(plt.Circle((lx, ry), 0.17, facecolor="white", ec=C_LOSS, lw=1.8, zorder=6))
    txt(lx, ry, "≈", size=12, weight="bold", color=C_LOSS, z=7)
    line([adx + adw, lx - 0.17], [ry, ry], col=C_LOSS, lw=1.4, ls=(0, (3, 2)))   # adaptor → loss

# ── teachers (frozen) with pretraining-domain thumbnails ─────────────────────
tx, tw, th = 6.45, 2.10, 1.00
txt(tx + tw / 2 + 0.15, 5.15, "Teachers", size=7.5, weight="bold", color="#8a929b")
for (col, name, arch, dom, swatch), ry in zip(TEACH, ROWS):
    box(tx, ry - th / 2, tw, th, col, alpha=0.16, lw=1.8)
    img(os.path.join(A, swatch), tx + 0.10, ry - 0.33, 0.66, 0.66, ec=col, lw=1.2)
    txt(tx + 0.45, ry - 0.46, dom, size=5.6, color="#6a6a6a")
    txt(tx + 0.88, ry + 0.22, name, size=8.8, weight="bold", color=col, ha="left")
    txt(tx + 0.88, ry + 0.02, arch, size=6.6, color="#5a5a5a", ha="left")
    badge(tx + 1.22, ry - 0.24, "❄ frozen", FROZEN_FG, FROZEN_BG, w=0.66)
    line([tx, lx + 0.17], [ry, ry], col=C_LOSS, lw=1.4, ls=(0, (3, 2)))          # teacher → loss
# input distribution to teachers: top bus from corpus, down the right edge into each card
busx = 8.82
line([1.17, 1.17], [4.65, 5.50], col=C_CORP, lw=1.6)
line([1.17, busx], [5.50, 5.50], col=C_CORP, lw=1.6)
line([busx, busx], [ROWS[2], 5.50], col=C_CORP, lw=1.6)
for ry in ROWS:
    arrow(busx, ry, tx + tw, ry, col=C_CORP, lw=1.5)
# txt(4.8, 5.62, "same image →", size=6.6, color=C_CORP, style="italic")

# ═══ PHASE 2  (full-width horizontal strip below phase 1) ══════════════════════
cy = -1.30   # pipeline centreline
gx, gw, gh, gap = 8.34, 1.08, 0.30, 0.075   # grades-column geometry (defined early for the arrow)

# ULF test slice (input)
sl_cx = 1.02
txt(sl_cx, cy + 0.86, "ULF test slice", size=8.5, weight="bold", color=INK)
img(os.path.join(A, "mr_input.png"), sl_cx - 0.63, cy - 0.62, 1.26, 1.24, ec=C_GRADE, lw=1.7)
arrow(sl_cx + 0.63, cy, 2.48, cy, col=C_ARROW)                       # slice -> student

# frozen student (weights carried over from Phase 1)
fs = box(2.48, cy - 0.52, 1.52, 1.04, C_STU, alpha=0.16, lw=1.7)
txt(fs[0], fs[1] + 0.24, "Student", size=9.2, weight="bold", color=C_STU)
txt(fs[0], fs[1] + 0.02, "ViT-S/14", size=6.6, color="#5a5a5a")
badge(fs[0], fs[1] - 0.26, "❄ frozen", FROZEN_FG, FROZEN_BG, w=0.72, h=0.23)
arrow(4.00, cy, 4.30, cy, col=C_ARROW)                              # student -> pool

mp = box(4.30, cy - 0.44, 1.14, 0.88, C_STU, alpha=0.10, lw=1.5, rounding=0.18)
txt(mp[0], mp[1], "mean-pool\n384-d", size=7.8, weight="bold", color=C_STU)
arrow(5.44, cy, 5.76, cy, col=C_ARROW)                              # pool -> probe

pc = box(5.76, cy - 0.67, 1.46, 1.34, C_PROBE, alpha=0.16, lw=1.9)
badge(pc[0], pc[1] - 0.50, "✎ trainable", TRAIN_FG, TRAIN_BG, w=0.80)
txt(pc[0], pc[1] + 0.28, "Probe head", size=9.5, weight="bold", color=C_PROBE)
txt(pc[0], pc[1] - 0.10, "MLP\n10-seed ensemble", size=7.2, color="#5a5a5a")
arrow(7.22, cy, gx - gw / 2 - 0.06, cy, col=C_ARROW)                # probe -> grades

# severity grades column
artifacts = ["Noise", "Zipper", "Positioning", "Banding", "Motion", "Contrast", "Distortion"]
gy0 = cy + (len(artifacts) * (gh + gap)) / 2 - gh                   # column centred on the pipeline
txt(gx, gy0 + gh + 0.40, "Severity  {0,1,2}", size=8.6, weight="bold", color=INK)
txt(gx, gy0 + gh + 0.18, "argmax · 7×3 logits", size=6.4, color="#8a8f94", style="italic")
for i, art in enumerate(artifacts):
    yy = gy0 - i * (gh + gap)
    box(gx - gw / 2, yy, gw, gh, C_GRADE, alpha=0.16, lw=1.0, shadow=False, rounding=0.06)
    txt(gx, yy + gh / 2, art, size=7.2, weight="bold", color="#3a4147")

# ═══ PHASE 1 -> 2 connector: trained weights carried straight DOWN into the phase-2 student ═
_cx = sx + sw / 2                                                   # aligned under the Phase-1 student
_top = fs[1] + 0.52                                                 # phase-2 student top edge
line([_cx, _cx], [sy, _top + 0.16], col=C_STU, lw=2.0, ls=(0, (4, 2)))
arrow(_cx, _top + 0.16, _cx, _top, col=C_STU, lw=2.0, ls=(0, (4, 2)))
# ax.text(_cx - 0.18, 0.75, "trained weights → frozen", rotation=90, ha="center", va="center",
#         fontsize=6.6, color=C_STU, fontstyle="italic", zorder=6)

plt.savefig("paper/fig_methods.pdf", bbox_inches="tight", dpi=200)
plt.savefig("paper/fig_methods.png", bbox_inches="tight", dpi=200)
print("saved paper/fig_methods.pdf  +  .png")
