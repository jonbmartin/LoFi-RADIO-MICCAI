#!/bin/bash
# Paper Table 2: single FM / FM concat / per-artifact gate / distilled student.
#
# All arms go through tri_probe.py at a matched 10-seed tuned head (dropout 0.5, lr 5e-4).
# DINOv2-single is the shared anchor across both runs and must come out identical in each.
#
# Distilled rows are cumulative:
#   Student <- D+BMC     DINOv2 + BiomedCLIP teachers,          corpus LISA + M4Raw + lf64mt
#   + Phikon teacher     DINOv2 + BiomedCLIP + Phikon teachers,  same corpus
#   + OpenNeuro corpus   same three teachers,                    corpus + ds006557
# The D+BMC student is distilled from a teacher subset of the three-teacher dense cache
# (distill.py --use_teachers), which avoids building a separate ~20 GB cache.
#
# Needs the four frozen caches cache/{dinov2,biomedclip,sam,phikon}_lisa.npz -- built here if
# missing (SAM needs SAM_W=sam_vit_b_01ec64.pth), or by scripts/run_screen.sh.
#
# Usage (from the repository root):  bash scripts/run_method_ablation.sh
source scripts/common.sh
HEAD="--dropout 0.5 --lr 5e-4"
SAM_W=${SAM_W:-sam_vit_b_01ec64.pth}

echo "############## frozen single-backbone caches ##############"
frozen_cache dinov2     $([ -f "$DINO_W" ] && echo --weights "$DINO_W")
frozen_cache biomedclip
frozen_cache sam        --weights "$SAM_W" --batch 8
frozen_cache phikon     $([ -e "$PHIKON_W" ] && echo --weights "$PHIKON_W")

echo "############## RUN A -- frozen FMs: singles / concat / gate ##############"
python lofi_radio/tri_probe.py \
  --caches cache/dinov2_lisa.npz,cache/biomedclip_lisa.npz,cache/sam_lisa.npz,cache/phikon_lisa.npz \
  --names DINOv2,BiomedCLIP,SAM,Phikon \
  --modes singles,concat,gate --seeds "$SEEDS" $HEAD --out_dir runs/tri_probe_fm

echo "############## distilled students ##############"
dense_cache lisa_m4raw_lf64mt  lisa,m4raw,lf64mt
student dbmc_lisa_m4raw_lf64mt lisa_m4raw_lf64mt --use_teachers dinov2,biomedclip
student phi_lisa_m4raw_lf64mt  lisa_m4raw_lf64mt
dense_cache full               lisa,m4raw,lf64mt,ds006557
student phi_full               full

echo "############## RUN B -- distilled students (paired, matched harness) ##############"
python lofi_radio/tri_probe.py \
  --caches cache/dinov2_lisa.npz,cache/student_dbmc_lisa_m4raw_lf64mt_cls_lisa.npz,cache/student_phi_lisa_m4raw_lf64mt_cls_lisa.npz,cache/student_phi_full_cls_lisa.npz \
  --names DINOv2,Student-DBMC,Student-Phi,Student-Phi-ds \
  --modes singles --seeds "$SEEDS" $HEAD --out_dir runs/tri_probe_students

python paper/tab_method_ablation.py
echo "Done: runs/tri_probe_{fm,students}/tri_metrics.json + paper/tab_method_ablation.tex"
