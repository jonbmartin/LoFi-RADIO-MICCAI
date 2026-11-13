#!/bin/bash
# Paper Table 1: uniform foundation-model screen.
#
# Every candidate backbone is cached through the same extraction path into a {files, X} LISA cache,
# then screen_probe.py measures, at one identical protocol for all of them:
#   - degeneracy (mean off-diagonal |cos|, scale, effective rank)
#   - single-arm tuned-head probe on matched subject-grouped folds + seeds
#   - paired delta / t / wins vs DINOv2
# -> runs/screen/screen_results.json -> paper/tab_screening.tex
#
# A backbone whose weights are not staged is skipped with a warning; re-run after staging more
# weights to fill rows. A single GPU is plenty (frozen extraction + tiny probe).
#
# Weights (set the env vars or drop files at these defaults):
#   DINO_W=dinov2_vits14.safetensors      SAM_W=sam_vit_b_01ec64.pth
#   PHIKON_W=phikon/model.safetensors     RADIODINO_W=radiodino_s16.safetensors
#   SAM2_W=sam2.1_hiera_base_plus.pt      ARNIQA_W (optional; torch.hub otherwise)
#   BRAINIAC_CKPT=brainiac_backbone.safetensors
#   TRIAD_SRC=Triad (cloned repo)         TRIAD_CKPT=Triad-PlainConvUNet-MAE.pth
# BiomedCLIP and MAE download from the hub.
#
# Usage (from the repository root):  bash scripts/run_screen.sh
source scripts/common.sh
set +e -u
mkdir -p runs/screen

SAM_W=${SAM_W:-sam_vit_b_01ec64.pth}
RADIODINO_W=${RADIODINO_W:-radiodino_s16.safetensors}
ARNIQA_W=${ARNIQA_W:-}
SAM2_W=${SAM2_W:-sam2.1_hiera_base_plus.pt}
BRAINIAC_CKPT=${BRAINIAC_CKPT:-brainiac_backbone.safetensors}
TRIAD_SRC=${TRIAD_SRC:-Triad}
TRIAD_CKPT=${TRIAD_CKPT:-Triad-PlainConvUNet-MAE.pth}

ARMS=""              # accumulates name=cache for every backbone that cached successfully
add_arm () { ARMS="${ARMS:+$ARMS; }$1=cache/$1_lisa.npz"; }
local_w () { [ -e "$1" ] && echo "--weights $1"; }     # pass --weights only if the file exists

# cache a backbone if its cache is missing; on failure warn + skip (don't abort the screen)
teacher () {   # name  [extra cache_teachers args...]
  local name="$1"
  if frozen_cache "$@"; then add_arm "$name"
  else echo "  !! [$name] cache FAILED (missing weights?) -- skipping this row."; rm -f "cache/${name}_lisa.npz"; fi
}
extractor () {   # name  cmd...   (3D extractors with their own scripts, same {files, X} output)
  local name="$1"; shift
  local out="cache/${name}_lisa.npz"
  if [ -f "$out" ]; then echo "  [$name] cache exists -- skip."; add_arm "$name"; return; fi
  echo "---- cache $name ----"
  if "$@"; then add_arm "$name"; else echo "  !! [$name] cache FAILED -- skipping."; rm -f "$out"; fi
}

echo "############## cache every candidate through the same path ##############"
# DINOv2 is the baseline -> must be present; everything is paired against it.
teacher dinov2     $(local_w "$DINO_W")
teacher biomedclip
teacher sam        --weights "$SAM_W" --batch 8   # ViT-B @1024: global attention OOMs at the default batch
teacher phikon     $(local_w "$PHIKON_W")
teacher radiodino  $(local_w "$RADIODINO_W")
teacher arniqa     ${ARNIQA_W:+--weights "$ARNIQA_W"}
teacher sam2       --weights "$SAM2_W"
teacher mae_vit_b
extractor brainiac python lofi_radio/brainiac_features.py --checkpoint "$BRAINIAC_CKPT" \
  --out cache/brainiac_lisa.npz
extractor triad    python lofi_radio/triad_features.py --triad_src "$TRIAD_SRC" \
  --checkpoint "$TRIAD_CKPT" --out cache/triad_lisa.npz

case "$ARMS" in
  dinov2=*) ;;   # dinov2 must be first/present as the baseline
  *) echo "ERROR: the dinov2 baseline did not cache. Aborting."; exit 1 ;;
esac

echo ""
echo "############## uniform screen probe (seeds ${SEEDS//,/ }, tuned head) ##############"
set -e
python lofi_radio/screen_probe.py --baseline dinov2 --seeds "$SEEDS" \
  --arms "$ARMS" --out runs/screen/screen_results.json

python paper/tab_screening.py
echo "Done: runs/screen/screen_results.json + paper/tab_screening.tex"
