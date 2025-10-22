#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Metrics & Calibration (Revised Version)

Functions:
1) Read "video-level scores" (mean_score) from prediction outputs and compute:
   - ROC-AUC (threshold-independent discriminative ability)
   - EER and corresponding threshold (balance point where false positive rate = false negative rate)
2) Probability Calibration (Isotonic/Platt):
   - Map raw scores to calibrated probabilities (p_cal)
   - Use p_cal and configurable intervals to produce the first version of real/uncertain/fake classification

Inputs (at least one required):
- --preds      : work/preds_per_sample.csv  (must include sample_id, mean_score)
- --per_image  : work/preds_per_image.csv   (used for aggregation if per-sample file lacks mean_score)

Outputs:
- work/metrics.json
- work/preds_per_sample_calibrated.csv
- work/metrics_report.txt  (short, human-readable version)
"""

import argparse
import json
import re
import sys

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve


def _infer_label(sample_id):
    s = str(sample_id)  # Prevent errors caused by list/numeric types
    if re.search(r'(^|/|_)0_real', s): return 0
    if re.search(r'(^|/|_)1_fake', s): return 1
    return -1


def _load_per_sample(preds_path, per_image_path=None):
    df = pd.read_csv(preds_path)
    if "sample_id" not in df.columns:
        raise ValueError("preds CSV must include a 'sample_id' column")
    df["sample_id"] = df["sample_id"].astype(str)  # Prevent downstream errors from non-string types

    # Prefer mean_score if available
    if "mean_score" in df.columns:
        out = df[["sample_id", "mean_score"]].copy()
        out["mean_score"] = pd.to_numeric(out["mean_score"], errors="coerce")
        return out.dropna(subset=["mean_score"])

    # If mean_score is missing, aggregate from per-image scores
    if per_image_path is None:
        raise ValueError("preds file has no mean_score; please provide --per_image for aggregation")
    pi = pd.read_csv(per_image_path)
    if "sample_id" not in pi.columns:
        raise ValueError("per-image CSV must include a 'sample_id' column")
    pi["sample_id"] = pi["sample_id"].astype(str)

    score_col = None
    for c in ["score_fake", "score"]:
        if c in pi.columns:
            score_col = c
            break
    if score_col is None:
        raise ValueError("per-image CSV must include either 'score_fake' or 'score' column")

    grp = (pi[["sample_id", score_col]]
           .dropna()
           .groupby("sample_id")[score_col]
           .mean()
           .reset_index()
           .rename(columns={score_col: "mean_score"}))
    return grp


def _compute_auc_eer(scores, labels):
    auc = float(roc_auc_score(labels, scores))
    fpr, tpr, thr = roc_curve(labels, scores)
    idx = int(np.nanargmin(np.abs(fpr - (1 - tpr))))  # Find the point where FPR ≈ 1 - TPR
    eer = float((fpr[idx] + (1 - tpr[idx])) / 2.0)
    thr_eer = float(thr[idx])
    return auc, eer, thr_eer, fpr, tpr, thr


def _brier(probs, labels):
    return float(np.mean((probs - labels) ** 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True, help="work/preds_per_sample.csv")
    ap.add_argument("--per_image", default=None, help="Optional: work/preds_per_image.csv")
    ap.add_argument("--metrics_out", default="./work/metrics.json")
    ap.add_argument("--cal_out", default="./work/preds_per_sample_calibrated.csv")
    ap.add_argument("--report_out", default="./work/metrics_report.txt")
    ap.add_argument("--cal_method", choices=["isotonic", "platt"], default="isotonic")
    ap.add_argument("--uncertain_low", type=float, default=None,
                    help="Lower bound of uncertain interval (optional; inferred automatically)")
    ap.add_argument("--uncertain_high", type=float, default=None,
                    help="Upper bound of uncertain interval (optional; inferred automatically)")
    args = ap.parse_args()

    # 1) Load per-sample predictions
    per_smp = _load_per_sample(args.preds, args.per_image)
    per_smp["label"] = per_smp["sample_id"].map(_infer_label)
    per_smp = per_smp[per_smp["label"] >= 0].reset_index(drop=True)

    if per_smp["label"].nunique() < 2:
        raise ValueError("Both 0_real and 1_fake samples are required to compute AUC/EER.")

    y = per_smp["label"].to_numpy()
    p_raw = np.clip(per_smp["mean_score"].to_numpy(dtype=float), 1e-8, 1 - 1e-8)

    # 2) AUC / EER
    auc, eer, thr_eer, fpr, tpr, thr = _compute_auc_eer(p_raw, y)

    # 3) Probability Calibration
    if args.cal_method == "isotonic":
        cal = IsotonicRegression(out_of_bounds="clip").fit(p_raw, y)
        p_cal = cal.predict(p_raw)
        p_at_thr_eer = float(cal.predict(np.array([thr_eer]))[0])
    else:
        lr = LogisticRegression(max_iter=2000).fit(p_raw.reshape(-1, 1), y)
        p_cal = lr.predict_proba(p_raw.reshape(-1, 1))[:, 1]
        p_at_thr_eer = float(lr.predict_proba(np.array([[thr_eer]]))[:, 1][0])

    brier = _brier(p_cal, y)

    # 4) Uncertain interval: use user input if provided; otherwise infer automatically
    if args.uncertain_low is not None and args.uncertain_high is not None:
        lo, hi = float(args.uncertain_low), float(args.uncertain_high)
    else:
        # If "perfect separation" occurs in small samples, p_at_thr_eer may ≈ 0 or ≈ 1; use conservative default 0.5±0.1
        if p_at_thr_eer <= 0.1 or p_at_thr_eer >= 0.9:
            lo, hi = 0.40, 0.60
        else:
            # Otherwise determine half-width from the 80th percentile of misclassified samples
            pred_tmp = (p_cal >= p_at_thr_eer).astype(int)
            err = np.where(pred_tmp != y)[0]
            if err.size > 0:
                deltas = np.abs(p_cal[err] - p_at_thr_eer)
                half = float(np.quantile(deltas, 0.80))
                half = float(np.clip(half, 0.05, 0.15))
                lo, hi = float(max(0.0, p_at_thr_eer - half)), float(min(1.0, p_at_thr_eer + half))
            else:
                lo, hi = 0.40, 0.60

    verdict = np.where(p_cal < lo, "real", np.where(p_cal > hi, "fake", "uncertain"))
    cal_df = per_smp[["sample_id"]].copy()
    cal_df["p_raw"] = p_raw
    cal_df["p_cal"] = p_cal
    cal_df["verdict_cal"] = verdict
    cal_df.to_csv(args.cal_out, index=False)

    # 5) Save metrics and report
    metrics = {
        "AUC": auc,
        "EER": eer,
        "thr_at_EER": thr_eer,
        "N": int(len(y)),
        "calibration": args.cal_method,
        "brier_score": brier,
        "p_at_thr_eer": p_at_thr_eer,
        "uncertain_band": {"low": lo, "high": hi}
    }
    with open(args.metrics_out, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    lines = [
        f"AUC = {auc:.4f}",
        f"EER = {eer:.4f}  (thr_at_EER ≈ {thr_eer:.3f})",
        f"Calibration = {args.cal_method}, Brier = {brier:.4f}",
        f"Uncertain band = [{lo:.2f}, {hi:.2f}]  (p_at_EER ≈ {p_at_thr_eer:.3f})",
        f"Saved metrics -> {args.metrics_out}",
        f"Saved calibrated -> {args.cal_out}",
    ]
    with open(args.report_out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("[ERROR]", e, file=sys.stderr)
        sys.exit(1)
