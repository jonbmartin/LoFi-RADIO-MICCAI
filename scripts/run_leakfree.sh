#!/bin/bash
# Paper Sec. 2.5: leak-free per-fold evaluation of the distilled student.
#
# Re-distills one student per subject-grouped fold (DINOv2 + BiomedCLIP + Phikon teachers, full
# corpus), each excluding that fold's LISA validation images, then assembles the out-of-fold
# predictions (fold k is predicted from the student that never saw fold k) and compares against the
# frozen DINOv2 baseline, which is leak-free by construction.
#
# Cost: 5 full distillations.
# Usage (from the repository root):  bash scripts/run_leakfree.sh
source scripts/common.sh
SEED=${SEED:-0}
HEAD=${HEAD:-'{"hidden": [256], "dropout": 0.5, "lr": 0.0005, "wd": 0.001, "epochs": 40, "batch": 16}'}

frozen_cache dinov2 $([ -f "$DINO_W" ] && echo --weights "$DINO_W")
dense_cache full lisa,m4raw,lf64mt,ds006557

echo "############## per-fold validation id lists ##############"
python lofi_radio/leakfree_cv.py write-ids --seed "$SEED" --out_dir cache/folds

echo "############## per-fold distillation + export ##############"
FOLD_CACHES=""
for k in 0 1 2 3 4; do
  student "phi_full_s${SEED}_f${k}" full --exclude_ids_file "cache/folds/seed${SEED}_fold${k}.txt"
  FOLD_CACHES="${FOLD_CACHES:+$FOLD_CACHES,}${k}=cache/student_phi_full_s${SEED}_f${k}_cls_lisa.npz"
done

echo "############## leak-free out-of-fold evaluation ##############"
python lofi_radio/leakfree_cv.py eval --seed "$SEED" --hp "$HEAD" \
  --fold_caches "$FOLD_CACHES" --baseline cache/dinov2_lisa.npz
