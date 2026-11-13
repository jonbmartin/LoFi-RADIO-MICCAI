"""
Unified distillation corpus (paper Sec. 2.4).

Pools several low-field brain-MRI sources into one stream of canonicalized volumes
(thick-acquisition-axis last -> [H, W, D]) so teacher features can be cached and a student
distilled on the union. Distillation is label-free, so an adapter only has to enumerate images.

Sources (adapters)
  - lisa     : LISA 2026 Task 1a, 0.064 T neonatal T2 (the target domain)
  - m4raw    : M4Raw, 0.3 T adult multi-contrast (Lyu et al., Sci. Data 2023)
  - lf64mt   : paired 64 mT / 3 T adult brain MRI (Zenodo 20281403); 64 mT T2w is used
  - ds006557 : OpenNeuro ds006557, 64 mT adult T1w/T2w; 64 mT T2w is used

Design notes
  * Canonicalization moves the largest-voxel-size axis to last. The frozen teachers in
    ssl_features expect exactly this [H, W, D] layout.
  * Samples are lazy: each carries a `loader` callable, so enumeration is cheap and the NIfTI /
    k-space read only happens when a teacher consumes it.
  * Each LISA sample id is its filename, so caches are keyed the way the probe looks them up.

Runnable as a CLI check:
    python lofi_radio/corpus.py --list
    python lofi_radio/corpus.py --list --probe_load --sources lisa,m4raw,lf64mt,ds006557
"""
from __future__ import annotations

import argparse
import glob
import os
from dataclasses import dataclass, field
from typing import Callable, Iterator

import numpy as np
import nibabel as nib

from lisa import DATA_DIR, LABELS_CSV, REPO

# default on-disk location for the downloadable corpora (override via adapter kwargs)
DATA_ROOT = os.path.join(REPO, "data")


# ----------------------------------------------------------------------------- helpers
def load_canonical_nifti(path: str) -> np.ndarray:
    """NIfTI -> [H, W, D] float32 with the thick (largest-zoom) axis last."""
    img = nib.load(path)
    v = img.get_fdata(dtype=np.float32)
    z = img.header.get_zooms()[:3]
    return np.moveaxis(v, int(np.argmax(z)), -1)


@dataclass
class Sample:
    """A lazily loadable canonical volume from some source."""
    id: str
    source: str
    loader: Callable[[], np.ndarray]
    meta: dict = field(default_factory=dict)

    def load(self) -> np.ndarray:
        return self.loader()


class SourceAdapter:
    """Base class. Subclasses set `name` and implement available()/iter_samples()."""
    name = "base"

    def available(self) -> bool:
        raise NotImplementedError

    def download_hint(self) -> str:
        return ""

    def iter_samples(self) -> Iterator[Sample]:
        raise NotImplementedError

    def count(self) -> int:
        return sum(1 for _ in self.iter_samples()) if self.available() else 0


# ----------------------------------------------------------------------------- LISA (live)
class LisaAdapter(SourceAdapter):
    """LISA Task 1a 0.064 T neonatal T2 -- the target domain. Enumerated from the labels CSV;
    rows whose file is not on disk are skipped."""
    name = "lisa"

    def __init__(self, data_dir: str | None = None, csv: str | None = None):
        self.data_dir = data_dir or DATA_DIR
        self.csv = csv or LABELS_CSV

    def available(self) -> bool:
        return os.path.isdir(self.data_dir) and os.path.exists(self.csv)

    def download_hint(self) -> str:
        return f"put the LISA Task 1a training volumes + labels CSV in {self.data_dir}/."

    def iter_samples(self) -> Iterator[Sample]:
        import pandas as pd
        df = pd.read_csv(self.csv)
        for fn in df["filename"]:
            path = os.path.join(self.data_dir, fn)
            if os.path.exists(path):
                yield Sample(id=str(fn), source=self.name,
                             loader=lambda p=path: load_canonical_nifti(p),
                             meta={"field_T": 0.064, "contrast": "T2"})


# ----------------------------------------------------------------------------- M4Raw
class M4RawAdapter(SourceAdapter):
    """M4Raw 0.3T adult brain (github.com/mylyu/M4Raw, Zenodo 8056074). 183+25 subjects x 18 axial
    slices x {T1,T2,FLAIR} x 2-3 reps, 4-coil k-space (fastMRI format .h5). We default to T2w for
    contrast match with LISA, and ALL reps (near-duplicate anatomy, different noise -> useful).

    Format (verified against M4Raw_tutorial.ipynb): each .h5 holds 'reconstruction_rss' [18,H,W]
    (the provided ground-truth magnitude -- we use this directly, no recon) and 'kspace' [18,4,256,256]
    = (slices,coils,H,W) complex (FFT fallback only). Filenames are <id>_<contrast><rep>.h5, e.g.
    2022061003_T201.h5, so T2 globs as *_T2*.h5. Slice axis (18) is the thick axis -> moveaxis to last.
    Unpack the multicoil_train + multicoil_val zips under data/m4raw/ (needs `pip install h5py`)."""
    name = "m4raw"

    def __init__(self, data_dir: str | None = None, contrast: str = "T2"):
        self.data_dir = data_dir or os.path.join(DATA_ROOT, "m4raw")
        self.contrast = contrast

    def _files(self) -> list[str]:
        if not os.path.isdir(self.data_dir):
            return []
        nii = sorted(glob.glob(os.path.join(self.data_dir, f"**/*_{self.contrast}*.nii*"), recursive=True))
        if nii:
            return nii
        return sorted(glob.glob(os.path.join(self.data_dir, f"**/*_{self.contrast}*.h5"), recursive=True))

    def available(self) -> bool:
        return len(self._files()) > 0

    def download_hint(self) -> str:
        return ("M4Raw (0.3T adult): from Zenodo record 8056074 grab M4RawV1.5_multicoil_train.zip "
                "(+_val.zip), unpack under "
                f"{self.data_dir}/ (`pip install h5py`). Adapter uses 'reconstruction_rss', "
                f"contrast={self.contrast!r} (all reps).")

    @staticmethod
    def _rss_from_h5(path: str) -> np.ndarray:
        import h5py
        with h5py.File(path, "r") as f:
            if "reconstruction_rss" in f:
                rss = f["reconstruction_rss"][()]                  # [S, H, W]
            else:
                ks = f["kspace"][()]                               # [S, C, H, W] complex
                img = np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(ks, axes=(-2, -1)),
                                                    axes=(-2, -1)), axes=(-2, -1))
                rss = np.sqrt((np.abs(img) ** 2).sum(axis=1))      # [S, H, W]
        return np.moveaxis(np.asarray(rss, np.float32), 0, -1)     # [H, W, S] thick(slice) last

    def iter_samples(self) -> Iterator[Sample]:
        for path in self._files():
            sid = os.path.splitext(os.path.basename(path).replace(".nii", ""))[0]
            if path.endswith(".h5"):
                loader = lambda p=path: self._rss_from_h5(p)
            else:
                loader = lambda p=path: load_canonical_nifti(p)
            yield Sample(id=f"m4raw/{sid}", source=self.name, loader=loader,
                         meta={"field_T": 0.3, "contrast": self.contrast})


# ----------------------------------------------------------------------------- 64mT/3T paired (Zenodo 20281403)
class LF64mtAdapter(SourceAdapter):
    """Paired 64mT (0.064T) + 3T healthy-ADULT brain MRI (Zenodo 20281403, MIT). The EXACT field match to
    LISA (0.064T), tighter than M4Raw's 0.3T. BIDS layout:
    <root>/Data/'{64mT,3T} data'/sub-XXXX/[ses-*/]anat/*T2w.nii.gz (note the SPACE in the field dir name).
    Defaults to the 64mT T2w scans (on-domain ULF); field='3T' pulls the high-field set (extra clean
    unlabeled data / cross-quality targets). Shared sub-IDs across fields = the HF/ULF pairing (~11). Adults
    -> anatomy gap vs neonatal LISA, but exact ULF field/contrast/noise, which is what distillation needs.
    Unpack the Zenodo zip under data/lf64mt/."""
    name = "lf64mt"

    def __init__(self, data_dir: str | None = None, field: str = "64mT", contrasts=None):
        self.data_dir = data_dir or os.path.join(DATA_ROOT, "lf64mt")
        self.field = field                                   # "64mT" (default, on-domain ULF) or "3T"
        # Which sequences to pool. Distillation is unlabeled + ULF artifact physics is ~sequence-agnostic,
        # so T1w/FLAIR at 0.064T add on-field signal; DWI/localizer are riskier (very different contrast /
        # low SNR). Default T2w (matches LISA's contrast; used in the paper); override with env
        # LF64MT_CONTRASTS="T1w,T2w,FLAIR" (anatomical) or "all" (every nifti).
        env = os.environ.get("LF64MT_CONTRASTS")
        if contrasts is None:
            contrasts = env.split(",") if env else ["T2w"]
        self.contrasts = [c.strip() for c in contrasts if c.strip()]

    def _field_dir(self) -> str | None:
        """Locate the '<field> data' directory (the Zenodo zip nests it under Data/, name has a space)."""
        if not os.path.isdir(self.data_dir):
            return None
        for d in glob.glob(os.path.join(self.data_dir, "**", f"{self.field} data"), recursive=True):
            if os.path.isdir(d):
                return d
        cand = os.path.join(self.data_dir, "Data", f"{self.field} data")
        return cand if os.path.isdir(cand) else None

    def _files(self) -> list[str]:
        base = self._field_dir()
        if not base:
            return []
        if any(c.lower() == "all" for c in self.contrasts):  # every nifti under the field dir
            return sorted(glob.glob(os.path.join(base, "**", "*.nii*"), recursive=True))
        hits: list[str] = []
        for suf in self.contrasts:                           # BIDS suffix glob per requested sequence
            hits += glob.glob(os.path.join(base, "**", f"*{suf}.nii*"), recursive=True)
        return sorted(set(hits))

    def available(self) -> bool:
        return len(self._files()) > 0

    def download_hint(self) -> str:
        return ("64mT/3T paired set (Zenodo 20281403): unpack the zip under "
                f"{self.data_dir}/ (expects Data/'{self.field} data'/sub-*/[ses-*/]anat/*T2w.nii.gz).")

    def iter_samples(self) -> Iterator[Sample]:
        ft = 0.064 if self.field.startswith("64") else 3.0
        for path in self._files():
            sid = os.path.splitext(os.path.basename(path).replace(".nii", ""))[0]
            bn = os.path.basename(path)
            con = next((c for c in ("T2w", "T1w", "FLAIR", "dwi") if c in bn), "other")
            yield Sample(id=f"lf64mt/{self.field}/{sid}", source=self.name,
                         loader=lambda p=path: load_canonical_nifti(p),
                         meta={"field_T": ft, "contrast": con})


# ----------------------------------------------------------------------------- ds006557 (OpenNeuro 64mT adult)
class Ds006557Adapter(SourceAdapter):
    """OpenNeuro ds006557: 64mT (0.064T) ULF + 3T healthy-ADULT brain MRI (test-retest reliability /
    HF correspondence; 23 adults, defaced T1w+T2w, MULTIPLE orientations & voxel resolutions). EXACT
    field match to LISA like lf64mt, but a second/larger adult cohort whose value is the extra
    orientation/resolution diversity at 0.064T. Standard BIDS NIfTI.

    IMPORTANT: the dataset also contains 3T scans -- keep ONLY the 64mT (ULF) ones or we pool high-field
    data as if it were ULF. Field is encoded in the BIDS SESSION: ses-GE = 3T (GE Premier);
    ses-HFC + ses-HFE = 64mT (the two ULF sites). We WHITELIST the 64mT sessions: a path is kept iff it
    contains one of `include` and none of `exclude`. (The 3T marker is 'ses-GE', NOT '3T' -- a naive
    '3T' filter would let it through.) T2w default (LISA contrast); env DS006557_CONTRASTS to pool
    T1w too. Unpack under data/ds006557/:
        aws s3 sync --no-sign-request s3://openneuro.org/ds006557 data/ds006557/"""
    name = "ds006557"

    def __init__(self, data_dir=None, contrasts=None, include=("ses-HFC", "ses-HFE"), exclude=("ses-GE",)):
        self.data_dir = data_dir or os.path.join(DATA_ROOT, "ds006557")
        env = os.environ.get("DS006557_CONTRASTS")
        if contrasts is None:
            contrasts = env.split(",") if env else ["T2w"]
        self.contrasts = [c.strip() for c in contrasts if c.strip()]
        env_inc = os.environ.get("DS006557_INCLUDE")
        self.include = tuple(env_inc.split(",")) if env_inc else tuple(include)   # 64mT sessions (whitelist)
        self.exclude = tuple(exclude)                                             # 3T session (ses-GE)

    def _files(self):
        if not os.path.isdir(self.data_dir):
            return []
        hits = []
        for suf in self.contrasts:
            hits += glob.glob(os.path.join(self.data_dir, "**", f"*{suf}.nii*"), recursive=True)
        out = []
        for p in sorted(set(hits)):
            if self.include and not any(tok in p for tok in self.include):
                continue
            if any(tok in p for tok in self.exclude):
                continue
            out.append(p)
        return out

    def available(self):
        return len(self._files()) > 0

    def download_hint(self):
        return ("OpenNeuro ds006557 (64mT adult T1/T2): aws s3 sync --no-sign-request "
                f"s3://openneuro.org/ds006557 {self.data_dir}/ (only the 64mT sessions "
                "ses-HFC/ses-HFE are used; the 3T session ses-GE is excluded).")

    def iter_samples(self):
        for path in self._files():
            sid = os.path.splitext(os.path.basename(path).replace(".nii", ""))[0]
            con = next((c for c in ("T2w", "T1w") if c in os.path.basename(path)), "other")
            yield Sample(id=f"ds006557/{sid}", source=self.name,
                         loader=lambda p=path: load_canonical_nifti(p),
                         meta={"field_T": 0.064, "contrast": con})


# ----------------------------------------------------------------------------- registry
_ADAPTERS = {"lisa": LisaAdapter, "m4raw": M4RawAdapter,
             "lf64mt": LF64mtAdapter, "ds006557": Ds006557Adapter}


class UnifiedCorpus:
    """Concatenate the requested source adapters into one Sample stream."""
    def __init__(self, sources: list[str], **adapter_kwargs):
        unknown = [s for s in sources if s not in _ADAPTERS]
        if unknown:
            raise ValueError(f"unknown source(s) {unknown}; have {list(_ADAPTERS)}")
        self.adapters = [_ADAPTERS[s](**adapter_kwargs.get(s, {})) for s in sources]

    def iter_samples(self, require_available: bool = True) -> Iterator[Sample]:
        for ad in self.adapters:
            if not ad.available():
                if require_available:
                    print(f"[corpus] source {ad.name!r} unavailable -- {ad.download_hint()}")
                continue
            yield from ad.iter_samples()

    def summary(self) -> dict:
        return {ad.name: (ad.count() if ad.available() else 0) for ad in self.adapters}


# ----------------------------------------------------------------------------- CLI smoke test
def main():
    ap = argparse.ArgumentParser(description="enumerate / sanity-check the distillation corpus")
    ap.add_argument("--sources", default="lisa,m4raw,lf64mt,ds006557")
    ap.add_argument("--list", action="store_true", help="print per-source counts")
    ap.add_argument("--probe_load", action="store_true",
                    help="load one volume per available source and print its canonical shape")
    args = ap.parse_args()

    sources = [s.strip() for s in args.sources.split(",") if s.strip()]
    corpus = UnifiedCorpus(sources)

    print("=== distillation corpus ===")
    total = 0
    for ad in corpus.adapters:
        if ad.available():
            n = ad.count(); total += n
            print(f"  [{ad.name:10s}] {n:5d} volumes   ({ad.download_hint()})")
        else:
            print(f"  [{ad.name:10s}]   --- NOT PRESENT: {ad.download_hint()}")
    print(f"  total available volumes: {total}")

    if args.probe_load:
        print("\n=== probe-load one volume per available source ===")
        for ad in corpus.adapters:
            if not ad.available():
                continue
            s = next(ad.iter_samples())
            v = s.load()
            print(f"  [{ad.name}] id={s.id}  canonical shape={v.shape}  "
                  f"dtype={v.dtype}  range=[{v.min():.1f},{v.max():.1f}]  meta={s.meta}")


if __name__ == "__main__":
    main()
