"""
Frozen foundation-model backbones as 2.5D feature extractors (paper Sec. 2.2).

Artifact grading is a mid-level perceptual-texture task (noise grain, blur, ghost replicas,
banding periodicity). Each 2D backbone is frozen and applied per slice: every foreground in-plane
slice of a canonicalized volume is normalized to unit range, encoded independently, and the
per-slice embeddings are mean-pooled into one descriptor per volume. Pooling is deterministic, so
features are cached once and reused by every probe.

`SSL_BACKBONES` lists the backbones and their feature widths; `load_backbone` returns a
`FrozenEncoder` that handles each backbone's own resize and normalization.
"""
from __future__ import annotations
import os
import numpy as np
import torch
import torch.nn.functional as F

SSL_MODEL = "vit_small_patch14_dinov2.lvd142m"   # DINOv2 ViT-S/14, 384-dim CLS token (via timm)
_IMG = 224                    # 224/14 = 16 patches per side
_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
# OpenAI-CLIP normalization (BiomedCLIP inherits it; values match its preprocess config)
_CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
_CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)

# Frozen backbones we can extract 2.5D per-slice features from; dim is each cache's feature width.
SSL_BACKBONES = {
    "dinov2":     dict(dim=384, img=224, timm="vit_small_patch14_dinov2.lvd142m"),  # ViT-S/14 (default)
    "radiodino":  dict(dim=384, img=224, timm="hf_hub:Snarcy/RadioDino-s16",
                       timm_offline="vit_small_patch16_224"),  # DINO ViT-S/16 on RadImageNet (CT/MRI/US)
    "mae_vit_b":  dict(dim=768, img=224, timm="vit_base_patch16_224.mae",
                       timm_offline="vit_base_patch16_224", global_pool="avg"),  # natural-image MAE ViT-B/16
                       # (reconstruction objective). GAP readout: MAE's CLS token is unsupervised, the
                       # signal lives in the patch tokens.
    "biomedclip": dict(dim=512, img=224),   # BiomedCLIP ViT-B/16 image tower, PMC biomedical CLIP
    "arniqa":     dict(dim=2048, img=224),  # ARNIQA ResNet-50 distortion-manifold SSL encoder (IQA)
    "sam":        dict(dim=256, img=1024, sam_type="vit_b"),  # SAM ViT-B image encoder, neck embed mean-pooled
    "sam2":       dict(dim=896, img=1024, sam2_cfg="configs/sam2.1/sam2.1_hiera_b+.yaml"),  # SAM2 Hiera-B+ TRUNK
                  # deepest-stage spatial GAP (896-d, pre-neck) -- the richer readout vs SAM's 256-d seg neck
    "phikon":     dict(dim=768,  img=224),
                  # Phikon ViT-B/16, iBOT on 40k TCGA WSIs (Owkin), loaded via transformers (_load_phikon).
                  # Offline: huggingface-cli download owkin/phikon --local-dir phikon/ -> --weights phikon/model.safetensors
}


def _load_state_dict_file(path):
    """Load a local weights file (.safetensors or torch .pth/.bin) to a CPU state dict."""
    if str(path).endswith(".safetensors"):
        from safetensors.torch import load_file
        return load_file(path)
    sd = torch.load(path, map_location="cpu")
    return sd.get("model", sd) if isinstance(sd, dict) else sd


class FrozenEncoder:
    """A frozen vision backbone wrapped with its own preprocessing: maps unit-range slices
    [B,3,H,W] (0..1) -> [B, dim] embeddings, handling resize + per-backbone normalization.
    `dim` is the feature width; `embed(model, x)` is the backbone-specific forward."""
    def __init__(self, model, dim, img, mean, std, embed, device):
        self.model, self.dim, self.img, self.device = model, dim, img, device
        self.mean, self.std, self._embed = mean.to(device), std.to(device), embed

    @torch.no_grad()
    def __call__(self, x_unit):                    # x_unit [B,3,H,W] in 0..1 (any device)
        x = F.interpolate(x_unit, size=(self.img, self.img), mode="bilinear", align_corners=False)
        x = (x.to(self.device) - self.mean) / self.std
        return self._embed(self.model, x).float().cpu()       # [B, dim]


def _load_dinov2(device, weights, backbone="dinov2"):
    """DINOv2 through timm (avoids torch.hub's PEP-604 import break on py<3.10). The timm tag
    comes from SSL_BACKBONES[backbone] ('dinov2' = ViT-S/14, 384-d; also used for the other
    timm-format ViTs). num_classes=0 -> pooled CLS embedding; dynamic_img_size lets us feed
    224 (native grid 518). Offline: pass weights=<file> (or $SSL_WEIGHTS) to skip the HF fetch:
        curl -kL -o dinov2_vits14.safetensors \\
          https://huggingface.co/timm/vit_small_patch14_dinov2.lvd142m/resolve/main/model.safetensors
    """
    import os, timm
    cfg = SSL_BACKBONES[backbone]
    weights = weights or os.environ.get("SSL_WEIGHTS")
    # Offline: build the architecture from a NATIVE timm tag (no hub
    # config fetch) and load the local state dict. `timm_offline` lets a hub-config backbone
    # (e.g. radiodino's 'hf_hub:Snarcy/RadioDino-s16') resolve to its equivalent native arch
    # ('vit_small_patch16_224') when weights are supplied, so nothing touches HF.
    tag = (cfg.get("timm_offline") or cfg.get("timm") or SSL_MODEL) if weights \
        else (cfg.get("timm") or SSL_MODEL)
    # global_pool='avg' (GAP over patch tokens) is the MAE-faithful readout: MAE leaves the CLS token
    # unsupervised, so the default 'token' pooling yields a near-constant/degenerate vector. DINOv2
    # supervises CLS -> keeps the timm default. (config opt; absent => timm default pooling.)
    extra = {"global_pool": cfg["global_pool"]} if cfg.get("global_pool") else {}
    model = timm.create_model(tag, pretrained=(weights is None),
                              num_classes=0, dynamic_img_size=True, **extra)
    if weights:
        missing, unexpected = model.load_state_dict(_load_state_dict_file(weights), strict=False)
        print(f"[ssl] loaded local {backbone} weights {weights} -> arch {tag!r} "
              f"(missing={len(missing)}, unexpected={len(unexpected)})")
    return model, _IMAGENET_MEAN, _IMAGENET_STD, (lambda m, x: m(x))


def _load_biomedclip(device, weights, backbone="biomedclip"):
    """BiomedCLIP (microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224) image tower via
    open_clip. encode_image -> 512-d projected embedding (unnormalized; we standardize later).
    Needs `pip install open_clip_torch` and HF access on first run."""
    import open_clip
    tag = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    model, _, _ = open_clip.create_model_and_transforms(tag)
    return model, _CLIP_MEAN, _CLIP_STD, (lambda m, x: m.encode_image(x))


def _arniqa_embed(m, x):
    """ARNIQA encoder forward -> [B, 2048] feature tensor, tolerant of (feat, proj) tuple returns."""
    f = m(x)
    return f[0] if isinstance(f, (tuple, list)) else f


def _load_arniqa(device, weights, backbone="arniqa"):
    """ARNIQA (miccunifi/ARNIQA) distortion-manifold SSL encoder. We take the ResNet-50 ENCODER
    representation (~2048-d quality-aware features), NOT the MOS regressor scalar -- it was SSL-
    trained to discriminate synthetic degradations, so it is explicitly distortion-sensitive, the
    closest off-the-shelf match to artifact-severity grading. ImageNet normalization. Pulls weights
    from torch.hub on first run (needs network).
    NOTE: encoder width is assumed 2048 (ResNet-50 global pool)."""
    model = torch.hub.load("miccunifi/ARNIQA", "ARNIQA",
                           regressor_dataset="kadid10k", trust_repo=True)
    enc = getattr(model, "encoder", model)   # the SSL encoder; fall back to the model itself
    return enc, _IMAGENET_MEAN, _IMAGENET_STD, _arniqa_embed


def _sam_embed(m, x):
    """SAM image-encoder forward -> [B,256,64,64] embedding -> spatial mean-pool -> [B,256].
    (The neck output the mask decoder consumes; the cited 'SAM image embedding'.)"""
    return m(x).mean(dim=(2, 3))


def _load_sam(device, weights, backbone="sam"):
    """SAM (Segment Anything, facebookresearch) image encoder as a frozen feature extractor. We
    take the image-encoder embedding [B,256,64,64] and
    spatial-mean-pool -> 256-d. Pixel norm is ImageNet mean/std (SAM's [123.7,116.3,103.5]/255 etc.),
    and it needs a FIXED 1024x1024 input (FrozenEncoder.img handles the resize).

    Needs `pip install segment-anything`. Offline: pass weights=<sam_vit_b_01ec64.pth> (or $SSL_WEIGHTS);
    single file, like the DINOv2 recipe:
        curl -kL -o sam_vit_b_01ec64.pth \\
          https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
    The checkpoint's ViT type must match SSL_BACKBONES['sam']['sam_type'] (default vit_b; the neck
    always projects to 256, so dim is 256 for vit_b/l/h)."""
    import os
    from segment_anything import sam_model_registry
    weights = weights or os.environ.get("SSL_WEIGHTS")
    assert weights, ("SAM needs a local checkpoint -- pass --weights sam_vit_b_01ec64.pth (or "
                     "$SSL_WEIGHTS); building it without weights gives random features.")
    sam = sam_model_registry[SSL_BACKBONES[backbone].get("sam_type", "vit_b")](checkpoint=weights)
    print(f"[ssl] loaded {backbone} image encoder from {weights}")
    return sam.image_encoder, _IMAGENET_MEAN, _IMAGENET_STD, _sam_embed


def _sam2_embed(m, x):
    """SAM2 Hiera TRUNK -> deepest multi-scale feature map -> spatial mean-pool -> [B, C]. We pool the
    high-dim trunk stage (pre-neck), NOT the 256-d FPN neck that made the original SAM low-rank. Square
    input (1024) -> square feature map, so the two equal dims are spatial and the odd one is the channel."""
    feats = m.image_encoder.trunk(x)               # Hiera -> list of multi-scale maps
    deep = feats[-1] if isinstance(feats, (list, tuple)) else feats
    if deep.dim() != 4:
        return deep.flatten(1)
    _, a, b, _ = deep.shape
    return deep.mean(dim=(1, 2)) if a == b else deep.mean(dim=(2, 3))   # channels-last vs channels-first


def _load_sam2(device, weights, backbone="sam2"):
    """SAM2 (facebookresearch/sam2) Hiera image-encoder TRUNK as a frozen extractor. 1024x1024 input +
    ImageNet norm (as SAM).
    Needs `pip install -e .` from the cloned sam2 repo. Offline: pass weights=<sam2.1_hiera_*.pt> (or
    $SSL_WEIGHTS); the config (SSL_BACKBONES[backbone]['sam2_cfg']) MUST match the checkpoint variant:
        curl -kL -o sam2.1_hiera_base_plus.pt \\
          https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt
    NOTE: emitted width = the deepest Hiera stage (b+ -> 896); cache_teachers records the ACTUAL width, so
    if it differs from SSL_BACKBONES['sam2']['dim'] just trust the printed cache shape."""
    import os
    from sam2.build_sam import build_sam2
    weights = weights or os.environ.get("SSL_WEIGHTS")
    assert weights, ("SAM2 needs a local checkpoint -- pass --weights sam2.1_hiera_base_plus.pt (or "
                     "$SSL_WEIGHTS); building without weights gives random features.")
    cfg = SSL_BACKBONES[backbone].get("sam2_cfg", "configs/sam2.1/sam2.1_hiera_b+.yaml")
    model = build_sam2(cfg, weights, device=str(device))
    print(f"[ssl] loaded {backbone} ({cfg}) from {weights}")
    return model, _IMAGENET_MEAN, _IMAGENET_STD, _sam2_embed


def _load_phikon(device, weights, backbone="phikon"):
    """Phikon ViT-B/16 (owkin/phikon) — iBOT pre-trained on 40k TCGA WSIs (Owkin). Fully open.

    Phikon weights are in HuggingFace transformers.ViTModel format (separate Q/K/V per layer,
    `embeddings.*` / `encoder.layer.N.*`), not timm's fused-QKV format — loading via timm's
    vit_base_patch16_224 architecture leaves 150 keys missing and 200 unexpected (random weights).
    We bypass timm entirely and load via transformers.ViTModel.from_pretrained().

    `weights` can be:
      - path to model.safetensors inside the downloaded dir -> dirname is inferred
      - path to the directory containing config.json + model.safetensors
      - None -> online 'owkin/phikon' (needs HF network access)

    Offline recipe:
        huggingface-cli download owkin/phikon --local-dir phikon/
        python lofi_radio/cache_teachers.py --teacher phikon --weights phikon/model.safetensors ...
    """
    import os
    from transformers import ViTModel

    weights = weights or os.environ.get("SSL_WEIGHTS")
    if weights:
        path = os.path.abspath(weights)
        model_dir = os.path.dirname(path) if os.path.isfile(path) else path
        model = ViTModel.from_pretrained(model_dir)
        print(f"[ssl] loaded phikon from {model_dir}")
    else:
        model = ViTModel.from_pretrained("owkin/phikon")
        print("[ssl] loaded phikon from HF hub")

    def _phikon_embed(m, x):
        return m(pixel_values=x).last_hidden_state[:, 0]   # CLS token [B, 768]

    return model, _IMAGENET_MEAN, _IMAGENET_STD, _phikon_embed


def load_backbone(device="cuda", weights=None, backbone="dinov2"):
    """Load a frozen feature backbone as a FrozenEncoder (eval, requires_grad off)."""
    assert backbone in SSL_BACKBONES, f"unknown backbone {backbone!r} (have {list(SSL_BACKBONES)})"
    loader = {"dinov2": _load_dinov2, "radiodino": _load_dinov2, "mae_vit_b": _load_dinov2,
              "phikon": _load_phikon, "biomedclip": _load_biomedclip, "arniqa": _load_arniqa,
              "sam": _load_sam, "sam2": _load_sam2}[backbone]
    model, mean, std, embed = loader(device, weights, backbone)
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    return FrozenEncoder(model, SSL_BACKBONES[backbone]["dim"], SSL_BACKBONES[backbone]["img"],
                         mean, std, embed, device)


def _slice_to_unit_pct(sl: np.ndarray) -> np.ndarray:
    """[H,W] magnitude -> robust 0..1 (per-slice 1/99 pct). Intensity-robust across sites."""
    lo, hi = np.percentile(sl, 1.0), np.percentile(sl, 99.0)
    if hi <= lo:
        hi = lo + 1.0
    return np.clip((sl - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def _slice_to_unit_medmad(sl: np.ndarray, k: float = 3.0) -> np.ndarray:
    """[H,W] magnitude -> robust 0..1 via a MEDIAN/MAD window over the FOREGROUND (tissue). Unlike the
    1/99 percentile window, the med+/-k*MAD window is set by the BULK, so a corruption tail (ghost, EMI
    spike) does NOT widen it -> corrupted and clean slices are normalized CONSISTENTLY, which is the
    QA-specific argument. Background is kept (maps to the low end)."""
    fg = sl > np.percentile(sl, 60.0)                              # loose tissue mask (drop the noisy bg)
    vals = sl[fg] if int(fg.sum()) > 32 else sl.reshape(-1)
    med = float(np.median(vals))
    mad = float(np.median(np.abs(vals - med))) * 1.4826           # ~std for a Gaussian bulk
    if mad <= 1e-6:
        mad = float(vals.std()) + 1e-6
    lo, hi = med - k * mad, med + k * mad
    return np.clip((sl - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def _slice_to_unit(sl: np.ndarray) -> np.ndarray:
    """Dispatch on env SLICE_NORM: 'pct' (default, 1/99 percentile) or 'medmad' (robust median/MAD window).
    Set SLICE_NORM=medmad identically upstream of BOTH cache_dense (teacher targets) and export_student
    (student LISA features) so inputs+targets stay on one distribution."""
    if os.environ.get("SLICE_NORM", "pct").lower() == "medmad":
        return _slice_to_unit_medmad(sl)
    return _slice_to_unit_pct(sl)


@torch.no_grad()
def extract_volume_slices(native: np.ndarray, encoder: "FrozenEncoder",
                          fg_frac: float = 0.02, batch: int = 64) -> np.ndarray:
    """Canonicalized native volume [H,W,D] (thick axis last) -> per-slice embeddings [S, encoder.dim].

    Iterates the high-res in-plane slices along D, drops near-empty (mostly-air) slices, and runs
    each kept slice through the frozen backbone.
    """
    D = native.shape[2]
    thr = 0.10 * float(np.percentile(native, 99.0))   # foreground intensity threshold
    keep = [d for d in range(D) if (native[:, :, d] > thr).mean() > fg_frac]
    if not keep:
        keep = list(range(D))

    imgs = np.stack([_slice_to_unit(native[:, :, d]) for d in keep])   # [S,H,W]
    x = torch.from_numpy(imgs)[:, None].repeat(1, 3, 1, 1)             # [S,3,H,W] unit-range

    feats = []
    for i in range(0, x.shape[0], batch):
        feats.append(encoder(x[i:i + batch]))                         # [b, dim]
    return torch.cat(feats, 0).numpy().astype(np.float32)             # [S, dim]


@torch.no_grad()
def extract_volume_features(native: np.ndarray, encoder: "FrozenEncoder",
                            fg_frac: float = 0.02, batch: int = 64) -> np.ndarray:
    """[H,W,D] -> mean-pooled feature vector [encoder.dim] (mean over kept in-plane slices)."""
    return extract_volume_slices(native, encoder, fg_frac, batch).mean(0)
