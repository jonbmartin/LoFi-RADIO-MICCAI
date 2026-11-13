"""
Teacher wrappers that expose both a summary vector and a dense feature grid.

ssl_features.FrozenEncoder returns only the pooled vector. Distillation also needs the spatial
token grid. `TeacherWithDense` reuses ssl_features.load_backbone for the model and per-backbone
preprocessing, adds a backbone-specific dense forward, and resamples every teacher's grid to one
common resolution (default 16x16) so the student distills onto a single shared grid.

Summary target per teacher:
  - dinov2     : CLS token (384)
  - biomedclip : encode_image projection (512, CLIP-aligned)
  - phikon     : CLS token (768)
  - sam        : spatial mean of the 256-d neck
Dense target per teacher = patch tokens (dinov2/biomedclip/phikon) or the neck map (sam).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from ssl_features import load_backbone


def _tokens_to_grid(patch: torch.Tensor, B: int) -> torch.Tensor:
    """[B, G*G, C] patch tokens -> [B, C, G, G]."""
    G = int(round(patch.shape[1] ** 0.5))
    return patch.transpose(1, 2).reshape(B, -1, G, G)


def _dinov2_dense(model, x):
    feats = model.forward_features(x)                     # [B, prefix+G*G, C]
    p = getattr(model, "num_prefix_tokens", 1)
    return feats[:, 0], _tokens_to_grid(feats[:, p:], x.shape[0])   # CLS, [B,C,G,G]


def _biomedclip_dense(model, x):
    trunk = model.visual.trunk                            # timm vit_base_patch16_224
    feats = trunk.forward_features(x)                     # [B, 1+G*G, 768]
    p = getattr(trunk, "num_prefix_tokens", 1)
    dense = _tokens_to_grid(feats[:, p:], x.shape[0])     # [B,768,G,G]
    summ = model.encode_image(x)                          # [B,512] projected (separate forward)
    return summ, dense


def _sam_dense(model, x):
    dense = model(x)                                      # [B,256,64,64] neck map
    return dense.mean(dim=(2, 3)), dense                  # mean-pool, [B,256,64,64]


def _phikon_dense(model, x):
    """Phikon (HF transformers ViTModel) -> CLS summary + patch-token grid. last_hidden_state is
    [B, 1+G*G, 768] (CLS prefix); 224/16 -> G=14. CLS is the readout the frozen probe uses, so it is
    the summary distillation target here."""
    feats = model(pixel_values=x).last_hidden_state       # [B, 1+G*G, 768]
    return feats[:, 0], _tokens_to_grid(feats[:, 1:], x.shape[0])   # CLS, [B,768,G,G]


_DENSE_FN = {"dinov2": _dinov2_dense, "biomedclip": _biomedclip_dense, "sam": _sam_dense,
             "phikon": _phikon_dense}


# Backbones whose input resolution must NOT be overridden: SAM-family run at a fixed 1024 (neck hard-wired
# to 64x64), and BiomedCLIP is an open_clip patch16-224 tower whose pos-embeds don't interpolate cleanly
# to a larger grid. For these, `img` is ignored and they stay at their native resolution (then resample
# to the shared grid if grid>0). The dinov2-family + phikon (timm, dynamic_img_size) do honour `img`.
_FIXED_IMG = {"sam", "sam2", "biomedclip"}


class TeacherWithDense:
    """Frozen teacher -> (summary [B, Ds], dense [B, grid, grid, Dd]) from unit-range slices.

    `img` raises the backbone's INPUT resolution (finer NATIVE patch grid) for dynamic-size ViTs
    (dinov2 family); SAM-family stay at their fixed 1024. `grid<=0` returns the backbone's native grid
    with NO resampling (the honest high-res path -- resampling a 16x16 map up to 32x32 adds no signal);
    `grid>0` resamples every teacher onto one shared grid (the old matched-grid behaviour)."""
    def __init__(self, name: str, device="cuda", weights=None, grid: int = 16, img: int | None = None):
        assert name in _DENSE_FN, f"no dense forward for {name!r}"
        self.name, self.grid, self.device = name, grid, device
        self.enc = load_backbone(device, weights=weights, backbone=name)   # FrozenEncoder (eval, frozen)
        if img and name not in _FIXED_IMG:
            self.enc.img = int(img)              # run the backbone at higher res -> finer native grid
        self._dense = _DENSE_FN[name]
        self.summary_dim = self.enc.dim          # pooled width (biomedclip: 512, != dense 768)

    @torch.no_grad()
    def __call__(self, x_unit: torch.Tensor):
        """x_unit [B,3,H,W] in 0..1 -> (summary np[B,Ds], dense np[B,grid,grid,Dd])."""
        x = F.interpolate(x_unit, size=(self.enc.img, self.enc.img), mode="bilinear", align_corners=False)
        x = (x.to(self.device) - self.enc.mean) / self.enc.std
        summ, dense = self._dense(self.enc.model, x)               # [B,Ds], [B,Dd,Gi,Gi]
        if self.grid and self.grid > 0:
            dense = F.interpolate(dense.float(), size=(self.grid, self.grid),
                                  mode="bilinear", align_corners=False)  # shared matched grid
        else:
            dense = dense.float()                                  # native grid (no upsampling)
        return (summ.float().cpu().numpy(),
                dense.permute(0, 2, 3, 1).cpu().numpy())            # [B,Ds], [B,grid,grid,Dd]


if __name__ == "__main__":   # tiny shape smoke test
    import argparse
    from corpus import UnifiedCorpus
    from ssl_features import _slice_to_unit

    ap = argparse.ArgumentParser()
    ap.add_argument("--teachers", default="dinov2")
    ap.add_argument("--weights", default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--grid", type=int, default=16)
    args = ap.parse_args()

    s = next(UnifiedCorpus(["lisa"]).iter_samples())
    vol = s.load()                                   # [H,W,D]
    sl = _slice_to_unit(vol[:, :, vol.shape[2] // 2])
    x = torch.from_numpy(sl)[None, None].repeat(1, 3, 1, 1)   # [1,3,H,W]
    for name in args.teachers.split(","):
        t = TeacherWithDense(name, device=args.device, weights=args.weights, grid=args.grid)
        summ, dense = t(x)
        print(f"[{name:10s}] summary {summ.shape} (Ds={t.summary_dim})  dense {dense.shape}")
