"""
LISA 2026 Task 1a task definition: artifact names, the severity scale, default data locations,
subject-grouped cross-validation folds, and the training loss.

Everything that touches labels or folds imports from here, so every experiment in the paper uses
the same split and the same objective.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F

ARTIFACTS = ["Noise", "Zipper", "Positioning", "Banding", "Motion", "Contrast", "Distortion"]
N_SEVERITY = 3   # graded severity scale: 0 none / 1 moderate / 2 severe

# Default data locations, relative to the repository root (see README "Data").
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO, "Lisa1A")                       # LISA training volumes (*.nii.gz)
LABELS_CSV = os.path.join(DATA_DIR, "LISA_Task1a_2026.csv")   # filename + one 0/1/2 column per artifact

_SUBJ_RE = re.compile(r"(LISA_\d+)")


def subject_id(filename: str) -> str:
    m = _SUBJ_RE.search(filename)
    if not m:
        raise ValueError(f"no subject id in {filename!r}")
    return m.group(1)


def load_labels(csv=LABELS_CSV, data_dir=DATA_DIR) -> pd.DataFrame:
    """Labels CSV restricted to the volumes that exist in `data_dir` (original row order kept)."""
    df = pd.read_csv(csv)
    on_disk = df["filename"].map(lambda f: (Path(data_dir) / str(f)).exists())
    if not on_disk.any():
        raise FileNotFoundError(f"none of the {len(df)} volumes listed in {csv} were found in {data_dir}")
    return df[on_disk].reset_index(drop=True)


def assign_folds(data_dir, csv_name=LABELS_CSV, n_folds=5, seed=0) -> pd.DataFrame:
    """Return the full df with a per-row 'fold' column (0..n_folds-1) assigned by SUBJECT.

    Subjects are shuffled (seeded) then round-robin'd across folds -> balanced fold sizes
    and no orientation leakage (all 3 orientations of a baby share a fold).

    `csv_name` is resolved against `data_dir`; an absolute path is used as given. Rows whose volume
    is not on disk are dropped BEFORE folds are assigned: the official CSV lists one volume that is
    absent from the data, and leaving its row in adds a subject, which changes the shuffle and
    therefore every fold.
    """
    data_dir = Path(data_dir)
    df = load_labels(data_dir / csv_name, data_dir)
    df["subject"] = df["filename"].map(subject_id)
    subjects = np.array(sorted(df["subject"].unique()))
    rng = np.random.RandomState(seed)
    rng.shuffle(subjects)
    fold_of = {s: i % n_folds for i, s in enumerate(subjects)}
    df["fold"] = df["subject"].map(fold_of)
    return df.reset_index(drop=True)


class SeverityCELoss(nn.Module):
    """Plain (unweighted) cross-entropy over the severity grades, averaged over volumes and artifacts.

    logits [B, K, 3], targets [B, K] integer grades in {0,1,2}.

    The official metric is support-weighted and grade 0 dominates, so up-weighting the rare grades
    over-calls artifacts on clean scans and loses the accuracy/precision components. No class
    weights are used.
    """
    def forward(self, logits, targets):
        return F.cross_entropy(logits.permute(0, 2, 1), targets.long())
