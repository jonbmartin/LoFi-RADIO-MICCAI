"""
Asset prep for the LoFi-RADIO methods figure (Fig. 1).

Generates raster thumbnails into paper/assets/ that fig_methods.py embeds:
  - REAL MR slices from LISA (corpus montage tiles + a Phase-2 artifact input)
  - domain swatches that EVOKE each teacher's pretraining data (illustrative icons):
      dom_dinov2.png   natural images   (abstract nature-palette montage)
      dom_biomedclip.png biomedical figs (real MR + synthetic radiograph/stain montage)
      dom_phikon.png   histopathology   (procedural H&E: eosin pink + hematoxylin nuclei)

The MR tiles are real data; the natural/pathology swatches are schematic (no natural-image or
WSI data is used) — methods-figure icons, not the actual pretraining corpora.

    python paper/_methods_assets.py            # writes paper/assets/*.png
"""
from __future__ import annotations
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
from scipy.ndimage import gaussian_filter

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PAPER = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _PAPER)
from fig_attention_panel import foreground_slices, DATA_DIR, LABELS_CSV, load_labels  # noqa: E402


ASSETS = os.path.join(_PAPER, "assets")
os.makedirs(ASSETS, exist_ok=True)
NIFTI_DIR, CSV = DATA_DIR, LABELS_CSV
ARTIFACTS = ["Noise", "Zipper", "Positioning", "Banding", "Motion", "Contrast", "Distortion"]
RNG = np.random.default_rng(7)


# ── helpers ───────────────────────────────────────────────────────────────────
def _mid_slice(fn: str) -> np.ndarray:
    units, _ = foreground_slices(os.path.join(NIFTI_DIR, fn))
    return units[len(units) // 2]


def _square_crop(sl: np.ndarray) -> np.ndarray:
    h, w = sl.shape
    s = min(h, w)
    return sl[(h - s) // 2:(h - s) // 2 + s, (w - s) // 2:(w - s) // 2 + s]


def _smooth_noise(shape, sigma, seed=None):
    r = RNG if seed is None else np.random.default_rng(seed)
    n = gaussian_filter(r.standard_normal(shape), sigma)
    n -= n.min(); n /= max(n.ptp(), 1e-6)
    return n


# ── 1. real MR tiles from LISA ────────────────────────────────────────────────
def export_mr_tiles(df):
    clean = df[(df[ARTIFACTS].sum(axis=1) == 0)]
    tiles = {}
    for orient in ("axi", "cor", "sag"):
        sub = clean[clean["filename"].str.contains(f"_{orient}")]
        if sub.empty:
            sub = df[df["filename"].str.contains(f"_{orient}")]
        fn = str(sub.iloc[len(sub) // 3]["filename"])
        sl = _square_crop(_mid_slice(fn))
        mpimg.imsave(os.path.join(ASSETS, f"mr_{orient}.png"), sl, cmap="gray", vmin=0, vmax=1)
        tiles[orient] = sl
        print(f"  mr_{orient}.png  <- {fn}")

    # corpus hero: 4 clean AXIAL slices from different subjects -> one 2x2 montage tile
    from scipy.ndimage import zoom
    axi_clean = clean[clean["filename"].str.contains("_axi")].reset_index(drop=True)
    h = 128
    mont = np.zeros((2 * h, 2 * h))
    for i, frac in enumerate((0.15, 0.40, 0.62, 0.85)):
        fn = str(axi_clean.iloc[int(frac * (len(axi_clean) - 1))]["filename"])
        sl = _square_crop(_mid_slice(fn))
        sl = zoom(sl, h / sl.shape[0], order=1)[:h, :h]
        ry, rx = (i // 2) * h, (i % 2) * h
        mont[ry:ry + h, rx:rx + h] = np.clip(sl, 0, 1)
        print(f"  corpus tile <- {fn}")
    mont[h - 1:h + 1] = 1.0; mont[:, h - 1:h + 1] = 1.0
    mpimg.imsave(os.path.join(ASSETS, "mr_corpus_montage.png"), mont, cmap="gray", vmin=0, vmax=1)
    print("  mr_corpus_montage.png")

    # Phase-2 input: a clear artifact example (Motion grade-2, else Noise grade-2)
    for art in ("Motion", "Noise", "Banding"):
        sub = df[df[art] == 2]
        if not sub.empty:
            fn = str(sub.iloc[0]["filename"])
            sl = _square_crop(_mid_slice(fn))
            mpimg.imsave(os.path.join(ASSETS, "mr_input.png"), sl, cmap="gray", vmin=0, vmax=1)
            print(f"  mr_input.png <- {fn} ({art}=2)")
            break
    return tiles


# ── 2. domain swatches ────────────────────────────────────────────────────────
def swatch_phikon(n=256):
    """Procedural H&E: eosin-pink stroma + hematoxylin-purple nuclei blobs."""
    eosin = np.array([0.93, 0.76, 0.85])
    purple = np.array([0.40, 0.22, 0.52])
    base = _smooth_noise((n, n), 9, seed=11)[..., None]
    img = eosin[None, None] * (0.82 + 0.30 * base)
    nuclei = np.zeros((n, n))
    for _ in range(220):
        cy, cx = RNG.integers(0, n, 2)
        rad = RNG.integers(3, 7)
        yy, xx = np.ogrid[:n, :n]
        nuclei += np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * rad ** 2)))
    nuclei = np.clip(nuclei, 0, 1)[..., None]
    img = img * (1 - nuclei) + purple[None, None] * nuclei
    img = np.clip(img, 0, 1)
    mpimg.imsave(os.path.join(ASSETS, "dom_phikon.png"), img)
    print("  dom_phikon.png  (H&E)")


def swatch_dinov2(n=256):
    """Montage (2x2) of REAL natural-world photos (skimage sample data) — DINOv2's pretraining domain."""
    import skimage.data as sd
    from scipy.ndimage import zoom
    photos = [sd.astronaut(), sd.chelsea(), sd.coffee(), sd.rocket()]
    h = n // 2
    img = np.zeros((n, n, 3))
    for k, ph in enumerate(photos):
        ph = ph.astype(np.float32) / 255.0
        H, W = ph.shape[:2]
        s = min(H, W)
        ph = ph[(H - s) // 2:(H - s) // 2 + s, (W - s) // 2:(W - s) // 2 + s]   # square crop
        ph = zoom(ph, (h / s, h / s, 1), order=1)[:h, :h]
        ry, rx = (k // 2) * h, (k % 2) * h
        img[ry:ry + h, rx:rx + h] = np.clip(ph, 0, 1)
    img[h - 1:h + 1] = 1; img[:, h - 1:h + 1] = 1   # white grid lines
    mpimg.imsave(os.path.join(ASSETS, "dom_dinov2.png"), np.clip(img, 0, 1))
    print("  dom_dinov2.png  (natural, real photos)")


def swatch_biomedclip(tiles, n=256):
    """Biomedical-figure montage (2x2): real MR + synthetic radiograph + stained/heatmap + MR."""
    h = n // 2
    img = np.zeros((n, n, 3))

    def place(k, rgb):
        ry, rx = (k // 2) * h, (k % 2) * h
        img[ry:ry + h, rx:rx + h] = np.clip(rgb, 0, 1)

    def _resz(sl):
        from scipy.ndimage import zoom
        sl = _square_crop(sl)
        return np.clip(zoom(sl, h / sl.shape[0], order=1)[:h, :h], 0, 1)

    mr = _resz(tiles["axi"])
    place(0, np.stack([mr] * 3, -1))                                  # MR (gray)
    xray = _resz(tiles["cor"]); xray = 1 - xray                       # radiograph (inverted)
    place(1, np.stack([xray] * 3, -1))
    stain = _resz(tiles["sag"])                                       # pseudo-stained micrograph
    sc = np.stack([stain, 0.35 * stain, 0.8 * stain], -1)
    place(2, sc)
    heat = _smooth_noise((h, h), 5, seed=31)                          # heatmap-style figure
    place(3, matplotlib.cm.get_cmap("magma")(heat)[..., :3])
    img[h - 1:h + 1] = 1; img[:, h - 1:h + 1] = 1
    mpimg.imsave(os.path.join(ASSETS, "dom_biomedclip.png"), np.clip(img, 0, 1))
    print("  dom_biomedclip.png  (biomedical)")


def main():
    df = load_labels(CSV, NIFTI_DIR)
    print(f"[assets] -> {ASSETS}")
    tiles = export_mr_tiles(df)
    swatch_phikon()
    swatch_dinov2()
    swatch_biomedclip(tiles)
    print("[assets] done.")


if __name__ == "__main__":
    main()
