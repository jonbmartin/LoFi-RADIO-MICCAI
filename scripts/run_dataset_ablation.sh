#!/bin/bash
# Paper Sec. 3.3: contribution of the distillation corpus.
#
# Cumulative corpus build-up with fixed teachers (DINOv2 + BiomedCLIP + Phikon):
#   L     LISA
#   LM    LISA + M4Raw
#   LMF   LISA + M4Raw + lf64mt
#   LMFD  LISA + M4Raw + lf64mt + ds006557
# followed by a 4-arm paired probe comparison (10 seeds, baseline = LISA only). The LMF and LMFD
# students are the same ones scripts/run_method_ablation.sh builds, and are reused if present.
#
# Each dense cache is tens of GB; delete cache/dense_*.npz once its student is exported if disk is
# tight (the leak-free script needs cache/dense_full.npz).
#
# Usage (from the repository root):  bash scripts/run_dataset_ablation.sh
source scripts/common.sh

dense_cache lisa               lisa
student phi_lisa               lisa
dense_cache lisa_m4raw         lisa,m4raw
student phi_lisa_m4raw         lisa_m4raw
dense_cache lisa_m4raw_lf64mt  lisa,m4raw,lf64mt
student phi_lisa_m4raw_lf64mt  lisa_m4raw_lf64mt
dense_cache full               lisa,m4raw,lf64mt,ds006557
student phi_full               full

echo ""
echo "=== 4-arm paired comparison (baseline = LISA only) ==="
python lofi_radio/compare_probes.py --baseline L --seeds "$SEEDS" --arms \
  "L=cache/student_phi_lisa_cls_lisa.npz; \
   LM=cache/student_phi_lisa_m4raw_cls_lisa.npz; \
   LMF=cache/student_phi_lisa_m4raw_lf64mt_cls_lisa.npz; \
   LMFD=cache/student_phi_full_cls_lisa.npz"
