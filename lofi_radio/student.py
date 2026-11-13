"""
The LoFi-RADIO student: a trainable ViT-S/14 initialized from DINOv2 (paper Sec. 2.4).

One teacher is DINOv2 itself, so that term of the distillation loss starts near zero and the
student only has to absorb the other teachers without forgetting.

forward() returns (summary [B,384], dense [B,G,G,384]) mirroring teachers.TeacherWithDense, so the
distiller matches them through per-teacher adaptors.
"""
from __future__ import annotations

import os

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from ssl_features import _IMAGENET_MEAN, _IMAGENET_STD, _load_state_dict_file

# Student backbone by arch tag (recorded in the checkpoint). The paper uses ViT-S/14 (384-d).
_STUDENT_TAGS = {
    "vits": "vit_small_patch14_dinov2.lvd142m",
}


class Student(nn.Module):
    """Trainable DINOv2 ViT-S/14; init from a local DINOv2 checkpoint (offline-safe) or the HF hub."""
    def __init__(self, weights: str | None = None, img: int = 224, from_pretrained: bool = True,
                 arch: str | None = None):
        super().__init__()
        self.img = img
        self.arch = (arch or "vits").lower()
        assert self.arch in _STUDENT_TAGS, f"arch {self.arch!r} not in {list(_STUDENT_TAGS)}"
        weights = weights or os.environ.get("SSL_WEIGHTS")
        # download DINOv2 only when asked AND no local weights -- export loads the trained student
        # from a ckpt, so it builds with from_pretrained=False (random init, no hub download).
        self.model = timm.create_model(_STUDENT_TAGS[self.arch],
                                       pretrained=(weights is None and from_pretrained),
                                       num_classes=0, dynamic_img_size=True)
        self.dim = int(self.model.num_features)                    # 384
        if weights:
            missing, unexpected = self.model.load_state_dict(_load_state_dict_file(weights), strict=False)
            print(f"[student] init from {weights} (missing={len(missing)}, unexpected={len(unexpected)})")
        self.register_buffer("mean", _IMAGENET_MEAN)
        self.register_buffer("std", _IMAGENET_STD)

    def _prep(self, x_unit: torch.Tensor) -> torch.Tensor:
        """unit-range [B,3,H,W] (0..1) -> resized to self.img + ImageNet-normalized."""
        x = F.interpolate(x_unit, size=(self.img, self.img), mode="bilinear", align_corners=False)
        return (x - self.mean) / self.std

    def forward(self, x_unit: torch.Tensor):
        x = self._prep(x_unit)
        feats = self.model.forward_features(x)                    # [B, prefix+G*G, 384]
        p = getattr(self.model, "num_prefix_tokens", 1)
        summ = feats[:, 0]                                        # CLS
        patch = feats[:, p:]                                      # [B, G*G, 384]
        G = int(round(patch.shape[1] ** 0.5))
        dense = patch.transpose(1, 2).reshape(x.shape[0], -1, G, G)  # [B,384,G,G]
        return summ, dense.permute(0, 2, 3, 1)                    # [B,384], [B,G,G,384]

    @torch.no_grad()
    def pooled(self, x_unit: torch.Tensor) -> torch.Tensor:
        """Mean over patch tokens -> [B,384] (matches ssl_features mean-pool semantics for the probe)."""
        _, dense = self(x_unit)                                   # [B,G,G,384]
        return dense.mean(dim=(1, 2))


if __name__ == "__main__":   # forward-shape smoke test
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=None)
    args = ap.parse_args()
    net = Student(weights=args.weights)
    x = torch.rand(2, 3, 130, 150)                               # arbitrary unit-range slice batch
    summ, dense = net(x)
    n_train = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"[student] summary {tuple(summ.shape)}  dense {tuple(dense.shape)}  "
          f"pooled {tuple(net.pooled(x).shape)}  trainable_params(M)={n_train/1e6:.1f}")
