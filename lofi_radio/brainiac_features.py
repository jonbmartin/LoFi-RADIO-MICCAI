"""
BrainIAC (3D brain-MRI foundation model) per-VOLUME features -> a compare_probes-compatible cache.

BrainIAC (AIM-KannLab / DivyanshuTak, Nat. Neurosci. 2026: s41593-026-02202-6) is a MONAI 3D ViT-B/16
(img 96^3, hidden 768) pretrained with SimCLR on ~49k structural brain MRIs. Unlike DINOv2/BiomedCLIP/
SAM it is FULLY 3D: it consumes a whole volume and emits ONE 768-d CLS embedding -- so it cannot ride
the 2.5D per-slice FrozenEncoder path (ssl_features). This script gives it its own extractor and writes
a per-volume {files, X, mean, std, model} cache keyed by LISA filename, i.e. a drop-in arm for
compare_probes.py / screen_probe.py. It is not a slice-aligned distillation teacher.

We build the MONAI ViT OURSELVES with BrainIAC's exact constructor (model.py::ViTBackboneNet) so this
needs only `pip install monai` + the checkpoint -- no repo clone -- and we load with strict=False (their
load_brainiac uses strict=True, which hard-crashes on MONAI-version key drift). The loader accepts BOTH
checkpoint layouts: the official Lightning .ckpt (a 'state_dict' with 'backbone.'-prefixed keys, which
we strip) and an already-extracted backbone file (e.g. HF eugenehp/brainiac `backbone.safetensors`).
Preprocessing replicates get_validation_transform verbatim (resize 96^3 trilinear + nonzero z-score).

Get the weights (either source), then:
    pip install monai
    # official: download the BrainIAC .ckpt from the repo's Dropbox link, OR
    # HF mirror: curl -kL -o backbone.safetensors \
    #   https://huggingface.co/eugenehp/brainiac/resolve/main/backbone.safetensors
    python lofi_radio/brainiac_features.py --checkpoint <brainiac.ckpt | backbone.safetensors> \
        --out cache/brainiac_lisa.npz
    python lofi_radio/compare_probes.py --baseline dinov2 --seeds 0,1,2,3,4 \
      --arms "dinov2=cache/dinov2_lisa.npz;brainiac=cache/brainiac_lisa.npz"

(--brainiac_src <repo>/src is an optional override that defers to their own load_brainiac instead.)
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch

from lisa import DATA_DIR, LABELS_CSV, REPO


def build_validation_transform(image_size=(96, 96, 96)):
    """Exact replica of BrainIAC src/dataset.py::get_validation_transform. Maps {'image': <nifti
    path>} -> {'image': [1,96,96,96]} (channel-first, resized, nonzero per-channel z-score)."""
    from monai.transforms import (Compose, LoadImaged, EnsureChannelFirstd, Resized,
                                  NormalizeIntensityd, ToTensord)
    return Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Resized(keys=["image"], spatial_size=image_size, mode="trilinear"),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        ToTensord(keys=["image"]),
    ])


def _state_dict_from_ckpt(path: str) -> dict:
    """Flat name->tensor dict from a .safetensors or a torch .ckpt/.pth (unwrapping 'state_dict')."""
    if str(path).endswith(".safetensors"):
        from safetensors.torch import load_file
        return load_file(path)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    return ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt


def build_brainiac(checkpoint: str, device: str):
    """MONAI 3D ViT-B built with BrainIAC's exact constructor, weights loaded strict=False.
    Returns a callable x[B,1,96,96,96] -> [B,768] (the token-0 / CLS embedding, as their forward)."""
    from monai.networks.nets import ViT
    vit = ViT(in_channels=1, img_size=(96, 96, 96), patch_size=(16, 16, 16),
              hidden_size=768, mlp_dim=3072, num_layers=12, num_heads=12, save_attn=True)
    sd = _state_dict_from_ckpt(checkpoint)
    bb = {k[len("backbone."):]: v for k, v in sd.items() if k.startswith("backbone.")}
    if not bb:                       # already-extracted backbone (HF eugenehp/brainiac) -> use as-is
        bb = dict(sd)
    missing, unexpected = vit.load_state_dict(bb, strict=False)
    matched = len(bb) - len(unexpected)
    print(f"[brainiac] {checkpoint}: matched {matched}/{len(bb)} ckpt keys "
          f"(missing={len(missing)}, unexpected={len(unexpected)})")
    if matched < 100:
        print("  WARNING: few keys matched -> arch/prefix mismatch; features would be ~random. "
              "Check the checkpoint layout before trusting the cache.")
    vit.eval().to(device)
    for p in vit.parameters():
        p.requires_grad_(False)
    return lambda x: vit(x)[0][:, 0]                                  # MONAI ViT -> (x, hidden); token 0


def make_embedder(args, dev):
    """Default: build the ViT here (no clone needed). --brainiac_src defers to their load_brainiac."""
    if args.brainiac_src:
        if args.brainiac_src not in sys.path:
            sys.path.insert(0, args.brainiac_src)
        from load_brainiac import load_brainiac                       # their MONAI ViT + strict=True
        m = load_brainiac(args.checkpoint, dev)
        m.eval()
        return lambda x: m(x)                                         # their forward already -> [B,768]
    return build_brainiac(args.checkpoint, dev)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True, help="BrainIAC .ckpt (Dropbox) or backbone.safetensors (HF)")
    ap.add_argument("--brainiac_src", default=None,
                    help="optional: path to cloned BrainIAC/src to use their load_brainiac instead")
    ap.add_argument("--data_dir", default=DATA_DIR)
    ap.add_argument("--csv", default=LABELS_CSV)
    ap.add_argument("--out", default=os.path.join(REPO, "cache", "brainiac_lisa.npz"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0, help="cap #volumes (smoke; 0=all)")
    args = ap.parse_args()

    dev = args.device if (not args.device.startswith("cuda") or torch.cuda.is_available()) else "cpu"
    embed = make_embedder(args, dev)
    tf = build_validation_transform()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    df = pd.read_csv(args.csv)
    files, feats, skipped = [], [], 0
    for fn in df["filename"]:
        path = os.path.join(args.data_dir, fn)
        if not os.path.exists(path):
            skipped += 1
            continue
        d = tf({"image": path})                                          # {'image': [1,96,96,96]}
        x = torch.as_tensor(np.asarray(d["image"], dtype=np.float32))[None].to(dev)  # [1,1,96,96,96]
        with torch.no_grad():
            f = embed(x)                                                 # [1,768]
        feats.append(np.asarray(f.detach().cpu(), np.float32).reshape(-1))
        files.append(str(fn))
        if args.limit and len(files) >= args.limit:
            print(f"  [limit] stopping at {len(files)} vols"); break

    X = np.stack(feats).astype(np.float32)
    np.savez(args.out, files=np.array(files), X=X,
             mean=X.mean(0), std=X.std(0) + 1e-6, model="brainiac")
    print(f"[brainiac] {len(files)} vols (skipped {skipped} missing) -> {args.out}  X={X.shape}  "
          f"next: screen_probe.py / compare_probes.py vs dinov2")


if __name__ == "__main__":
    main()
