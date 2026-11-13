# LoFi RADIO

Code for **"LoFi RADIO: A Distilled In-Domain Backbone Applied for Artifact-Severity Grading of
Ultra-Low-Field Neonatal Brain MRI"** (MICCAI 2026 LISA challenge, Task 1a).
Paper: [arXiv:2609.02676](https://arxiv.org/abs/2609.02676)

LoFi RADIO is a ViT-S/14 backbone distilled from several frozen foundation models (DINOv2,
BiomedCLIP, Phikon) on unlabeled low-field brain MRI, following AM-RADIO. A small MLP probe on its
frozen features grades seven artifacts (Noise, Zipper, Positioning, Banding, Motion, Contrast,
Distortion) on a 0/1/2 severity scale for each 0.064 T neonatal T2 volume.

This repository contains everything needed to reproduce the paper's experiments: the frozen
foundation-model screen, the concatenation and per-artifact gate baselines, the multi-teacher
distillation, the leak-free evaluation, and the challenge submission.

## How it works

**Stage 1, distillation.** Each foreground slice of every corpus volume is encoded once by the
frozen teachers, and the summary token and dense 16×16 feature grid of each teacher are cached. A
ViT-S/14 student initialized from DINOv2 is trained to reproduce those targets through per-teacher
adaptor MLPs (cosine loss on the summary; cosine + smooth-L1 on the dense grid), with each
teacher's targets standardized so no teacher dominates.

**Stage 2, grading.** The student is frozen. For each volume, the per-slice CLS embeddings are
mean-pooled into one 384-d descriptor, and an MLP head trained with cross-entropy predicts a
grade per artifact. The same probe is used for every frozen backbone, so all comparisons differ
only in the features.

## Repository layout

```
lofi_radio/                 all Python modules; run as `python lofi_radio/<script>.py` from the repo root
  lisa.py                   task definition: artifacts, labels, subject-grouped folds, loss
  evaluate.py               official metric (mean of 5 support-weighted metrics per artifact)
  ssl_features.py           frozen backbones + 2.5D slice encoding and normalization
  cache_teachers.py         pooled per-volume features of one frozen backbone
  brainiac_features.py      same, for the 3D BrainIAC model
  triad_features.py         same, for the 3D Triad model
  head_hpo.py               the probe head and its random hyperparameter search
  compare_probes.py         paired multi-seed comparison of feature caches
  screen_probe.py           uniform backbone screen (Table 1)
  tri_probe.py              single / concat / per-artifact gate probes (Table 2)
  corpus.py                 unified distillation corpus (LISA, M4Raw, lf64mt, ds006557)
  teachers.py               teacher wrappers returning summary + dense features
  cache_dense.py            dense multi-teacher target cache
  student.py                the ViT-S/14 student
  distill.py                multi-teacher distillation
  export_student.py         student features over LISA, as a probe cache
  leakfree_cv.py            leak-free per-fold evaluation
  submit_student.py         challenge submission CSV
  validate_submission.py    submission format check
scripts/                    one driver per paper experiment
paper/                      table and figure generators
```

## Setup

```bash
pip install -r requirements.txt
```

Python 3.10+ and PyTorch 2.x with CUDA. Some rows of the backbone screen need extra packages and
checkpoints (SAM, SAM2, BrainIAC, Triad); these are listed in `requirements.txt` and at the top of
`scripts/run_screen.sh`, and any backbone that is not set up is skipped.

Teacher weights are downloaded from the Hugging Face hub on first use. To work offline, pass local
files instead (`DINO_W`, `PHIKON_W`, ... in the scripts, or `--weights` on the command line);
`lofi_radio/ssl_features.py` documents where to get each one.

## Data

**LISA 2026 Task 1a.** The challenge data cannot be redistributed; obtain it from the
[LISA challenge](https://www.lisa-challenge.org). Place the training volumes and the label CSV in
`Lisa1A/`:

```
Lisa1A/
  LISA_Task1a_2026.csv          filename, Noise, Zipper, Positioning, Banding, Motion, Contrast, Distortion
  LISA_0001_LF_axi.nii.gz
  ...
```

Rows whose volume is not on disk are ignored everywhere, and folds are assigned by subject after
that filtering, so all three orientations of a subject always share a fold.

**External distillation corpora** go under `data/`:

| Source | Field | Location | Reference |
|---|---|---|---|
| M4Raw | 0.3 T adult | `data/m4raw/` | Lyu et al., *Sci. Data* 2023, doi:10.1038/s41597-023-02181-4 |
| lf64mt | 0.064 T adult | `data/lf64mt/` | van den Broek et al., Zenodo, doi:10.5281/zenodo.20281403 |
| ds006557 | 0.064 T adult | `data/ds006557/` | Váša et al., OpenNeuro ds006557, doi:10.1162/IMAG.a.930 |

Only T2-weighted scans are used, and only the 64 mT sessions of the two paired datasets. Check
what the code finds, with download instructions for anything missing:

```bash
python lofi_radio/corpus.py --list --probe_load
```

## Reproducing the paper

Run every script from the repository root. Each one skips steps whose outputs already exist.

| Paper | Command | Output |
|---|---|---|
| Table 1, backbone screen | `bash scripts/run_screen.sh` | `runs/screen/screen_results.json`, `paper/tab_screening.tex` |
| Table 2, single / concat / gate / distilled | `bash scripts/run_method_ablation.sh` | `runs/tri_probe_{fm,students}/tri_metrics.json`, `paper/tab_method_ablation.tex` |
| Sec. 3.3, corpus contribution | `bash scripts/run_dataset_ablation.sh` | paired comparison printed to the terminal |
| Sec. 2.5, leak-free evaluation | `bash scripts/run_leakfree.sh` | leak-free composite vs DINOv2, printed |
| Table 3, challenge submission | `bash scripts/run_submission.sh <TEST_DIR>` | `LISA_LF_QC_predictions.csv` |
| Fig. 1, pipeline overview | `python paper/_methods_assets.py && python paper/fig_methods.py` | `paper/fig_methods.{pdf,png}` |
| Fig. 2, attention panel | `python paper/fig_attention_panel.py --student_ckpt runs/student_phi_lisa_m4raw_lf64mt.pt --weights "dinov2=<dinov2 weights>"` | `paper/fig_attention_panel.{pdf,png}` |

The probe head search of Sec. 2.2:

```bash
python lofi_radio/head_hpo.py --cache cache/dinov2_lisa.npz --trials 48 --search_seeds 0,1,2
```

### Running the steps by hand

```bash
# 1. dense teacher targets over the distillation corpus
python lofi_radio/cache_dense.py --sources lisa,m4raw,lf64mt,ds006557 \
    --teachers dinov2,biomedclip,phikon --out cache/dense_full.npz

# 2. distill the student (uses every visible GPU)
python lofi_radio/distill.py --cache cache/dense_full.npz --batch 256 --epochs 40 \
    --out runs/student_phi_full.pt

# 3. export the student's pooled LISA features
python lofi_radio/export_student.py --ckpt runs/student_phi_full.pt \
    --out cache/student_phi_full_cls_lisa.npz

# 4. probe: paired comparison against frozen DINOv2
python lofi_radio/cache_teachers.py --teacher dinov2 --out cache/dinov2_lisa.npz
python lofi_radio/compare_probes.py --baseline dinov2 --seeds 0,1,2,3,4,5,6,7,8,9 \
    --arms "dinov2=cache/dinov2_lisa.npz; student=cache/student_phi_full_cls_lisa.npz"
```

### Cost

A dense teacher cache over the full corpus is tens of GB on disk and is held in RAM during
distillation. One distillation (batch 256, 40 epochs) took roughly 40 minutes on four GPUs in our
runs; the leak-free evaluation needs five. Feature caching and every probe experiment run on a
single GPU in minutes.

## Implementation details

- **Slice selection.** Volumes are reoriented so the thick-slice axis is last. A slice is kept if
  more than 2% of its pixels exceed 10% of the volume's 99th-percentile intensity.
- **Slice normalization.** Set by the `SLICE_NORM` environment variable: `pct` (default; 1st to
  99th percentile window) or `medmad` (median ± 3·MAD window over the slice foreground). It must
  be the same for the dense cache, the feature export, and inference. The cross-validation
  experiments use `pct`; the challenge submission uses `medmad`.
- **Probe head.** One hidden layer, LeakyReLU, dropout 0.5, AdamW with learning rate 5e-4 and
  cosine annealing, features z-scored on the training fold. The cross-validation experiments use
  512 hidden units, weight decay 1e-2, 80 epochs, batch 32. The submission uses 256 hidden units,
  weight decay 1e-3, 40 epochs, batch 16, averaged over a 10-head ensemble.
- **Evaluation.** Subject-grouped 5-fold cross-validation repeated over 10 seeds; arms are
  compared seed-for-seed on identical folds.
- **Reproducibility.** Probe experiments are seeded and repeat exactly on the same hardware.
  Distillation on GPU is not bit-reproducible, so a re-distilled student differs slightly from
  run to run.

## Data and code availability

The LISA data were provided by the challenge organizers and cannot be shared by the authors.
Trained student weights are not included in this repository.

## Citation

```bibtex
@article{martin2026lofiradio,
  title   = {LoFi RADIO: A Distilled In-Domain Backbone Applied for Artifact-Severity Grading of
             Ultra-Low-Field Neonatal Brain MRI},
  author  = {Martin, Jonathan B. and Kurmi, Yashwant and Sappo, Charlotte R.},
  journal = {arXiv preprint arXiv:2609.02676},
  year    = {2026}
}
```

This work builds on AM-RADIO (Ranzinger et al., CVPR 2024) and PHI-S (Ranzinger et al., 2024),
and on the DINOv2, BiomedCLIP, and Phikon foundation models.
