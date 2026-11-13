#!/bin/bash
# Paper Sec. 2.5 / Table 3: the challenge submission.
#
# Distills one student on all available data (no fold exclusion: the challenge test set is
# disjoint and withheld), exports its pooled LISA features, trains a 10-head probe ensemble on all
# labeled LISA volumes, and grades the test volumes.
#
# Submitted configuration: DINOv2 + BiomedCLIP teachers, corpus LISA + M4Raw + lf64mt, median/MAD
# slice normalization. SLICE_NORM must be identical for the dense cache, the export, and the test
# encode, so it is exported once here.
#
# Usage (from the repository root):  bash scripts/run_submission.sh <TEST_DIR>
source scripts/common.sh
TEST_DIR=${1:?usage: bash scripts/run_submission.sh <dir of test .nii.gz>}
export SLICE_NORM=medmad

dense_cache dbmc_medmad lisa,m4raw,lf64mt dinov2,biomedclip
student dbmc_medmad dbmc_medmad

python lofi_radio/submit_student.py --ckpt runs/student_dbmc_medmad.pt \
  --lisa_cache cache/student_dbmc_medmad_cls_lisa.npz --test_dir "$TEST_DIR"
python lofi_radio/validate_submission.py LISA_LF_QC_predictions.csv --val_dir "$TEST_DIR"
