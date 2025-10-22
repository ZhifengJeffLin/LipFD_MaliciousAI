#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
viz_metrics_from_csv.py

Functions:
- Read per-sample prediction results (CSV with columns: sample_id, mean_score)
- Compute ROC-AUC, EER, and corresponding threshold (in score space)
- Generate confusion matrix at the EER threshold and export per-sample decision table
- Plot two figures: ROC curve (marking the EER point) and score distribution (marking the EER threshold)
- If a calibrated CSV (with p_cal) is provided, also plot a Reliability Diagram.

Example usage:
  python viz_metrics_from_csv.py \
    --per_sample ./work/preds_per_sample.csv \
    --calibrated ./work/preds_per_sample_calibrated.csv \
    --out_dir   ./work

Dependencies:
  pip install numpy pandas scikit-learn matplotlib
"""

import argparse
import json
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix


def infer_label(sample_id: str) -> int:
    """Infer label from sample_id (0 for real, 1 for fake)."""
    s = str(sample_id)
    if re.search(r'(^|/|_)1_fake', s): return 1
    if re.search(r'(^|/|_)0_real', s): return 0
    return -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per_sample", required=True,
                    help="Path to preds_per_sample.csv (must include sample_id, mean_score)")
    ap.add_argument("--calibrated", default=None, help="Optional: preds_per_sample_calibrated.csv (must include p_cal)")
    ap.add_argument("--out_dir", default="./work", help="Output directory (plots/tables/metrics will be saved here)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # 1) Load per-sample predictions
    df = pd.read_csv(args.per_sample)
    need = {"sample_id", "mean_score"}
    if not need.issubset(df.columns):
        raise ValueError(f"CSV must contain columns {need}, but found: {list(df.columns)}")

    df = df[["sample_id", "mean_score"]].copy()
    df["sample_id"] = df["sample_id"].astype(str)
    df["mean_score"] = pd.to_numeric(df["mean_score"], errors="coerce")
    df = df.dropna(subset=["mean_score"])
    df["label"] = df["sample_id"].apply(infer_label)
    df = df[df["label"] >= 0].reset_index(drop=True)

    if df["label"].nunique() < 2:
        raise ValueError("Both 0_real and 1_fake samples are required for metric computation.")

    y = df["label"].to_numpy()
    scores = np.clip(df["mean_score"].to_numpy(float), 1e-9, 1 - 1e-9)

    # 2) ROC / AUC / EER
    auc = float(roc_auc_score(y, scores))
    fpr, tpr, thr = roc_curve(y, scores)
    eer_idx = int(np.nanargmin(np.abs(fpr - (1 - tpr))))
    eer = float((fpr[eer_idx] + (1 - tpr[eer_idx])) / 2.0)
    thr_eer = float(thr[eer_idx])

    # 3) Confusion matrix at EER threshold
    pred_eer = (scores >= thr_eer).astype(int)
    cm = confusion_matrix(y, pred_eer, labels=[0, 1])
    TN, FP, FN, TP = cm.ravel()
    fpr_eer = FP / (FP + TN + 1e-12)
    fnr_eer = FN / (FN + TP + 1e-12)

    # Export per-sample decision table at EER
    eer_table = df.copy()
    eer_table["pred_at_eer"] = pred_eer
    eer_table["correct_at_eer"] = (eer_table["pred_at_eer"] == eer_table["label"]).astype(int)
    eer_csv = os.path.join(args.out_dir, "eer_predictions.csv")
    eer_table.to_csv(eer_csv, index=False)

    # 4) Plot ROC (mark EER point)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC={auc:.3f}")
    plt.scatter([fpr[eer_idx]], [tpr[eer_idx]], marker="o", label=f"EER≈{eer:.3f} @ thr≈{thr_eer:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    roc_png = os.path.join(args.out_dir, "roc_curve.png")
    plt.savefig(roc_png, dpi=160, bbox_inches="tight")
    plt.close()

    # 5) Plot score distributions (mark EER threshold)
    real_scores = df.loc[df["label"] == 0, "mean_score"].to_numpy()
    fake_scores = df.loc[df["label"] == 1, "mean_score"].to_numpy()
    plt.figure(figsize=(6, 4.5))
    plt.hist(real_scores, bins=30, alpha=0.5, label="real (0)", density=True)
    plt.hist(fake_scores, bins=30, alpha=0.5, label="fake (1)", density=True)
    plt.axvline(thr_eer, linestyle="--", linewidth=1, label=f"thr@EER≈{thr_eer:.3f}")
    plt.xlabel("mean_score (higher = more fake-like)")
    plt.ylabel("Density")
    plt.title("Score Distributions (per-sample)")
    plt.legend()
    dist_png = os.path.join(args.out_dir, "score_distributions.png")
    plt.savefig(dist_png, dpi=160, bbox_inches="tight")
    plt.close()

    # 6) Optional: reliability diagram (if calibrated CSV provided)
    rel_png = None
    if args.calibrated and os.path.exists(args.calibrated):
        cal = pd.read_csv(args.calibrated)
        if "p_cal" in cal.columns:
            merged = cal.merge(df[["sample_id", "label"]], on="sample_id", how="inner")
            p = np.clip(merged["p_cal"].to_numpy(float), 1e-9, 1 - 1e-9)
            y_m = merged["label"].to_numpy()
            # 10-bin reliability curve
            bins = np.linspace(0.0, 1.0, 11)
            inds = np.digitize(p, bins) - 1
            centers = 0.5 * (bins[:-1] + bins[1:])
            xs, ys = [], []
            for k in range(10):
                m = inds == k
                if m.any():
                    xs.append(float(centers[k]))
                    ys.append(float(y_m[m].mean()))
            if xs:
                plt.figure(figsize=(6, 5))
                plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1, label="perfect")
                plt.plot(xs, ys, marker="o", label="empirical")
                plt.xlabel("Predicted probability (p_cal)")
                plt.ylabel("Empirical frequency of class=1 (fake)")
                plt.title("Reliability Diagram (calibrated)")
                plt.legend()
                rel_png = os.path.join(args.out_dir, "reliability_diagram.png")
                plt.savefig(rel_png, dpi=160, bbox_inches="tight")
                plt.close()

    # 7) Save summary
    summary = {
        "AUC": round(auc, 4),
        "EER": round(eer, 4),
        "thr_at_EER(score)": round(thr_eer, 3),
        "FPR@EER": round(fpr_eer, 4),
        "FNR@EER": round(fnr_eer, 4),
        "N_real": int((y == 0).sum()),
        "N_fake": int((y == 1).sum()),
        "roc_png": os.path.abspath(roc_png),
        "score_dist_png": os.path.abspath(dist_png),
        "reliability_png": os.path.abspath(rel_png) if rel_png else None,
        "eer_table_csv": os.path.abspath(eer_csv)
    }
    with open(os.path.join(args.out_dir, "metrics_from_csv.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
