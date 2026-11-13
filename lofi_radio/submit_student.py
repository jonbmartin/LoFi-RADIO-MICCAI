"""
Build the LISA Task 1a submission CSV from the distilled student (paper Sec. 2.5).

The challenge test set is disjoint and withheld, so the probe is trained on all labeled LISA
volumes (student features) and applied to the test volumes' student features:

  1. load the trained student (distill.py .pt), distilled on all available data.
  2. train features = the student's LISA export (export_student.py cache) + LISA labels.
  3. test features  = the same student over each test .nii.gz (CLS mean-pool, as in the export).
  4. ensemble N tuned heads, average softmax probabilities, argmax -> grades 0/1/2.
  5. write LISA_LF_QC_predictions.csv (filename + 7 artifacts).

Set $SLICE_NORM to the value the student was distilled and exported with.

    SLICE_NORM=medmad python lofi_radio/submit_student.py --ckpt runs/student_dbmc_medmad.pt \
        --lisa_cache cache/student_dbmc_medmad_cls_lisa.npz --test_dir <TEST_DIR>
    python lofi_radio/validate_submission.py LISA_LF_QC_predictions.csv --val_dir <TEST_DIR>
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import head_hpo
from corpus import load_canonical_nifti
from student import Student
from export_student import pool_volume
from lisa import ARTIFACTS, N_SEVERITY, SeverityCELoss, LABELS_CSV

# Head search winner on the student features (head_hpo.py): [256] drop0.5 lr5e-4 wd1e-3 ep40 bs16.
# Override with --hp.
TUNED_HEAD = dict(hidden=[256], dropout=0.5, lr=5e-4, wd=1e-3, epochs=40, batch=16)
K, S = len(ARTIFACTS), N_SEVERITY


def train_predict_probs(hp, Xtr, ytr, Xte, dev):
    """Train one tuned head on (Xtr,ytr) [train z-score], return softmax probs on Xte [Nte,K,S]."""
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr_ = ((Xtr - mu) / sd).astype(np.float32)
    Xte_ = ((Xte - mu) / sd).astype(np.float32)
    net = head_hpo.make_head(Xtr_.shape[1], hp["hidden"], hp["dropout"]).to(dev)
    loss_fn = SeverityCELoss()
    opt = torch.optim.AdamW(net.parameters(), lr=hp["lr"], weight_decay=hp["wd"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=hp["epochs"])
    Xt = torch.from_numpy(Xtr_).to(dev); yt = torch.from_numpy(ytr).long().to(dev)
    Nn = Xt.shape[0]
    for _ in range(hp["epochs"]):
        net.train()
        perm = torch.randperm(Nn, device=dev)
        for i in range(0, Nn, hp["batch"]):
            b = perm[i:i + hp["batch"]]
            opt.zero_grad()
            loss = loss_fn(net(Xt[b]).view(-1, K, S), yt[b])
            loss.backward(); opt.step()
        sched.step()
    net.eval()
    with torch.no_grad():
        probs = torch.softmax(net(torch.from_numpy(Xte_).to(dev)).view(-1, K, S), -1)
    return probs.cpu().numpy()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True, help="trained student .pt (distill.py)")
    ap.add_argument("--lisa_cache", required=True, help="student LISA export (export_student.py) = train feats")
    ap.add_argument("--test_dir", required=True, help="dir of test/validation .nii.gz")
    ap.add_argument("--csv", default=LABELS_CSV, help="LISA labels")
    ap.add_argument("--out", default="LISA_LF_QC_predictions.csv")
    ap.add_argument("--n_ensemble", type=int, default=10, help="tuned heads to average (probability space)")
    ap.add_argument("--hp", default=None, help="JSON head-config override, merged onto the HPO winner "
                    "(default [256] drop0.5 lr5e-4 wd1e-3 ep40 bs16)")
    ap.add_argument("--fg_frac", type=float, default=0.02)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    head = {**TUNED_HEAD, **(json.loads(args.hp) if args.hp else {})}
    print(f"[submit] probe head: {head}")
    dev = args.device if (not args.device.startswith("cuda") or torch.cuda.is_available()) else "cpu"

    # 1) student backbone (arch only; weights from ckpt -> no HF fetch)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    student = Student(weights=None, from_pretrained=False, arch=ckpt.get("arch", "vits"))
    student.load_state_dict(ckpt["student"]); student.eval().to(dev)
    print(f"[submit] student={args.ckpt} teachers={ckpt.get('teachers')}")

    # 2) train features (student LISA export) + labels, aligned
    fmap = head_hpo._cache_map(args.lisa_cache)
    df = pd.read_csv(args.csv)
    lab = {str(r["filename"]): np.array([r[a] for a in ARTIFACTS], np.int64) for _, r in df.iterrows()}
    files_tr = [f for f in df["filename"].astype(str) if f in fmap]
    Xtr = np.stack([fmap[f] for f in files_tr]).astype(np.float32)
    ytr = np.stack([lab[f] for f in files_tr])
    print(f"[submit] train: {Xtr.shape[0]} LISA volumes x {Xtr.shape[1]}-d student feats")

    # 3) test features -- SAME student, same cls mean-pool as the export
    test_files = sorted(p.name for p in Path(args.test_dir).glob("*.nii.gz"))
    if not test_files:
        raise FileNotFoundError(f"no .nii.gz in {args.test_dir}")
    Xte, fails, t0 = [], [], time.time()
    for i, fn in enumerate(test_files):
        try:
            native = load_canonical_nifti(str(Path(args.test_dir) / fn))
            Xte.append(pool_volume(student, native, "cls", args.fg_frac, args.batch, dev))
        except Exception as e:
            print(f"  WARN extract {fn}: {type(e).__name__}: {e} -> grade-0 fallback")
            Xte.append(np.zeros(Xtr.shape[1], np.float32)); fails.append(fn)
        if (i + 1) % 25 == 0:
            print(f"  [feat] {i + 1}/{len(test_files)} ({time.time()-t0:.0f}s)")
    Xte = np.stack(Xte).astype(np.float32)

    # 4) probability-ensemble of tuned heads trained on ALL LISA
    P = np.zeros((len(test_files), K, S), np.float32)
    for s in range(args.n_ensemble):
        np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
        P += train_predict_probs(head, Xtr, ytr, Xte, dev)
    P /= args.n_ensemble
    grades = P.argmax(-1)                                    # [Nte, K] in {0,1,2}
    for fn in fails:
        grades[test_files.index(fn)] = 0                    # safe default for any failed extraction

    # 5) write submission CSV
    sub = pd.DataFrame({"filename": test_files})
    for c, a in enumerate(ARTIFACTS):
        sub[a] = grades[:, c].astype(int)
    sub.to_csv(args.out, index=False)
    print(f"\n[submit] wrote {len(sub)} predictions -> {args.out}  "
          f"({args.n_ensemble}-head ensemble; {len(fails)} extract-fallbacks)")
    print("grade counts per artifact (0/1/2):")
    for a in ARTIFACTS:
        vc = sub[a].value_counts().reindex([0, 1, 2], fill_value=0)
        print(f"  {a:12s} {int(vc[0]):4d} {int(vc[1]):4d} {int(vc[2]):4d}")

    # over-calling guardrail on the ACTUAL submission: compare its grade>=1 rate to LISA truth.
    # The official metric is support-weighted -> calling materially more than the train rate trades
    # the dominant grade-0 class for the rare grades.
    sub_call = float((grades >= 1).mean())
    train_call = float((ytr >= 1).mean())
    over = sub_call - train_call
    verdict = (f"OVER-CALLING RISK (+{over:.3f} above LISA rate)" if over > 0.02
               else "ok -- conservative, in line with LISA (good for the support-weighted metric)")
    print(f"\n[guardrail] submission call_rate {sub_call:.3f} vs LISA {train_call:.3f}  -> {verdict}")
    print(f"\nnext: python lofi_radio/validate_submission.py {args.out} --val_dir {args.test_dir}")


if __name__ == "__main__":
    main()
