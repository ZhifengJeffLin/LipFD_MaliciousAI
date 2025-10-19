#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
compact_report.py
Combine key visualizations into ONE figure:
- ROC with AUC & EER
- Reliability diagram (needs preds_per_sample_calibrated.csv; will skip if missing)
- Confusion Matrix @ EER
- Stability Gate: verdict distribution Before vs After (needs preds_per_image.csv; will skip if missing)

Inputs (all optional but 'preds' is strongly recommended):
  --preds       path/to/preds_per_sample.csv
  --metrics     path/to/metrics.json  (if absent, compute AUC/EER from preds)
  --calibrated  path/to/preds_per_sample_calibrated.csv (for reliability & verdict_cal)
  --per_image   path/to/preds_per_image.csv (for stability-gate stats)
  --out_dir     output folder (default: ./work/large_eval)
  --tau_var     variance threshold for instability (default 0.05)
  --tau_mad     mean-abs-diff threshold for instability (default 0.15)

Usage (example):
  python compact_report.py \
    --preds      ./LipFD_MaliciousAI/work/large/preds_per_sample.csv \
    --metrics    ./LipFD_MaliciousAI/work/large/metrics.json \
    --calibrated ./LipFD_MaliciousAI/work/large/preds_per_sample_calibrated.csv \
    --per_image  ./LipFD_MaliciousAI/work/large/preds_per_image.csv \
    --out_dir    ./work/large_eval
"""

import argparse
import json
import os
import re

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix


def infer_label(sample_id: str) -> int:
    s = str(sample_id)
    if re.search(r'(^|/|_)1_fake', s): return 1
    if re.search(r'(^|/|_)0_real', s): return 0
    return -1


def load_preds(preds_csv: str):
    df = pd.read_csv(preds_csv)
    need = {"sample_id", "mean_score"}
    assert need.issubset(df.columns), f"{preds_csv} must contain {need}"
    df = df[["sample_id", "mean_score"]].dropna().copy()
    df["label"] = df["sample_id"].apply(infer_label)
    df = df[df["label"] >= 0].reset_index(drop=True)
    y = df["label"].to_numpy(int)
    s = np.clip(df["mean_score"].astype(float).to_numpy(), 1e-9, 1 - 1e-9)
    return df, y, s


def compute_auc_eer(y, s):
    auc = float(roc_auc_score(y, s))
    fpr, tpr, thr = roc_curve(y, s)
    idx = int(np.nanargmin(np.abs(fpr - (1 - tpr))))
    eer = float((fpr[idx] + (1 - tpr[idx])) / 2.0)
    thr_eer = float(thr[idx])
    return auc, (fpr, tpr, thr, idx), eer, thr_eer


def load_thr_from_metrics(metrics_json: str):
    try:
        m = json.load(open(metrics_json, "r", encoding="utf-8"))
        if "thr_at_EER" in m and m["thr_at_EER"] is not None:
            return float(m["thr_at_EER"])
    except Exception:
        pass
    return None


def plot_confusion(ax, y, s, thr_eer, title="Confusion Matrix @ EER"):
    pred = (s >= thr_eer).astype(int)
    TN, FP, FN, TP = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    cm = np.array([[TN, FP], [FN, TP]])
    im = ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center")
    ax.set_xticks([0, 1]);
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred Real", "Pred Fake"]);
    ax.set_yticklabels(["True Real", "True Fake"])
    ax.set_title(title, fontsize=11)
    return {"TN": int(TN), "FP": int(FP), "FN": int(FN), "TP": int(TP),
            "FPR@EER": float(FP / (FP + TN + 1e-12)), "FNR@EER": float(FN / (FN + TP + 1e-12))}


def plot_reliability(ax, calibrated_csv: str, preds_df: pd.DataFrame):
    if calibrated_csv is None or not os.path.exists(calibrated_csv):
        ax.text(0.5, 0.5, "(calibrated csv missing)", ha="center", va="center")
        ax.axis("off");
        return False
    cal = pd.read_csv(calibrated_csv)
    need = {"sample_id", "p_cal"}
    if not need.issubset(cal.columns):
        ax.text(0.5, 0.5, "(p_cal not found)", ha="center", va="center")
        ax.axis("off");
        return False
    lab = preds_df[["sample_id"]].copy()
    lab["label"] = lab["sample_id"].apply(infer_label)
    m = cal.merge(lab, on="sample_id", how="left").dropna(subset=["label", "p_cal"])
    p = np.clip(m["p_cal"].to_numpy(float), 1e-9, 1 - 1e-9)
    y = m["label"].to_numpy(int)

    bins = np.linspace(0, 1, 11);
    idx = np.digitize(p, bins) - 1
    centers = 0.5 * (bins[:-1] + bins[1:]);
    xs, ys = [], []
    for k in range(10):
        sel = (idx == k)
        if sel.any():
            xs.append(centers[k]);
            ys.append(y[sel].mean())

    ax.plot([0, 1], [0, 1], "--", linewidth=1, label="perfect")
    ax.plot(xs, ys, marker="o", label="calibrated")
    ax.set_xlabel("Predicted probability (p_cal)");
    ax.set_ylabel("Empirical freq (fake)")
    ax.set_title("Reliability Diagram", fontsize=11);
    ax.legend(fontsize=9)
    return True


def stringify(x):
    if isinstance(x, (list, tuple)):
        return "/".join(map(str, x))
    return str(x)


def natural_key_tuple(s: str):
    parts = re.split(r'(\d+)', s)
    return tuple(int(p) if p.isdigit() else p for p in parts)


def stability_gate(per_image_csv: str, calibrated_csv: str, preds_df: pd.DataFrame, tau_var=0.05, tau_mad=0.15):
    if per_image_csv is None or not os.path.exists(per_image_csv):
        return None, None, None
    pi = pd.read_csv(per_image_csv)
    if "score_fake" not in pi.columns and "score" in pi.columns:
        pi = pi.rename(columns={"score": "score_fake"})
    need = {"sample_id", "image", "score_fake"}
    if not need.issubset(pi.columns):
        return None, None, None
    # clean
    pi["sample_id"] = pi["sample_id"].apply(stringify)
    pi["image"] = pi["image"].apply(stringify)
    pi["score_fake"] = pd.to_numeric(pi["score_fake"], errors="coerce")
    pi = pi.dropna(subset=["score_fake"])
    pi = pi.sort_values(["sample_id", "image"], key=lambda c: c.map(natural_key_tuple))

    def summarize(g: pd.DataFrame) -> pd.Series:
        s = g["score_fake"].to_numpy(float)
        var = float(np.var(s)) if s.size else 0.0
        mad = float(np.mean(np.abs(np.diff(s)))) if s.size > 1 else 0.0
        return pd.Series({"var": var, "mean_abs_diff": mad, "n": int(s.size)})

    stab = pi.groupby("sample_id", sort=False).apply(summarize).reset_index()
    stab["unstable"] = (stab["var"] > tau_var) | (stab["mean_abs_diff"] > tau_mad)

    if calibrated_csv and os.path.exists(calibrated_csv):
        cal = pd.read_csv(calibrated_csv)[["sample_id", "verdict_cal"]]
    else:
        # fallback: derive verdict_cal from mean_score@EER later by caller if needed
        cal = preds_df[["sample_id"]].copy();
        cal["verdict_cal"] = np.nan

    final = cal.merge(stab[["sample_id", "unstable"]], on="sample_id", how="left")
    final["unstable"] = final["unstable"].fillna(False)
    final["verdict_final"] = np.where(final["unstable"], "uncertain", final["verdict_cal"])
    return final, stab, cal


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True, help="preds_per_sample.csv")
    ap.add_argument("--metrics", default=None, help="metrics.json (for thr_at_EER)")
    ap.add_argument("--calibrated", default=None, help="preds_per_sample_calibrated.csv")
    ap.add_argument("--per_image", default=None, help="preds_per_image.csv")
    ap.add_argument("--out_dir", default="./work/large_eval")
    ap.add_argument("--tau_var", type=float, default=0.05)
    ap.add_argument("--tau_mad", type=float, default=0.15)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # --- Load preds & labels ---
    df, y, s = load_preds(args.preds)

    # --- AUC / EER / ROC ---
    auc, (fpr, tpr, thr, idx), eer, thr_eer = compute_auc_eer(y, s)
    thr_from_file = load_thr_from_metrics(args.metrics) if args.metrics else None
    if thr_from_file is not None:
        thr_eer = float(thr_from_file)  # prefer metrics.json value to stay consistent with earlier stage

    # --- Build 2x2 figure ---
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.2))
    ax_roc, ax_rel, ax_cm, ax_gate = axes.ravel()

    # (1) ROC
    ax_roc.plot(fpr, tpr, label=f"AUC={auc:.3f}")
    # mark EER point (closest to FPR==1-TPR using computed arrays)
    i_mark = int(np.nanargmin(np.abs(fpr - (1 - tpr))))
    ax_roc.scatter([fpr[i_mark]], [tpr[i_mark]], label=f"EER≈{((fpr[i_mark] + (1 - tpr[i_mark])) / 2):.3f}")
    ax_roc.plot([0, 1], [0, 1], "--", linewidth=1)
    ax_roc.set_xlabel("False Positive Rate");
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title("ROC Curve", fontsize=11);
    ax_roc.legend(fontsize=9)

    # (2) Reliability diagram (if calibrated available)
    rel_ok = plot_reliability(ax_rel, args.calibrated, df)

    # (3) Confusion @ EER
    cm_stats = plot_confusion(ax_cm, y, s, thr_eer, title=f"Confusion @ EER (thr≈{thr_eer:.3f})")

    # (4) Stability Gate: verdict distribution Before vs After
    final, stab, cal = stability_gate(args.per_image, args.calibrated, df, args.tau_var, args.tau_mad)
    if final is None:
        ax_gate.text(0.5, 0.5, "(per-image csv missing)", ha="center", va="center")
        ax_gate.axis("off")
        verdict_before = {"real": np.nan, "fake": np.nan, "uncertain": np.nan}
        verdict_after = verdict_before
    else:
        # If verdict_cal absent, create from mean_score @ EER
        if final["verdict_final"].isna().any():
            # derive verdict_cal using thr@EER
            temp = df.copy()
            temp["verdict_cal"] = np.where(temp["mean_score"] >= thr_eer, "fake", "real")
            final = final.drop(columns=["verdict_final"]).merge(
                temp[["sample_id", "verdict_cal"]], on="sample_id", how="left"
            )
            final["verdict_final"] = np.where(final["unstable"], "uncertain", final["verdict_cal"])

        order = ["real", "fake", "uncertain"]
        cnt_b = final["verdict_cal"].value_counts().reindex(order, fill_value=0)
        cnt_a = final["verdict_final"].value_counts().reindex(order, fill_value=0)
        X = np.arange(3)
        ax_gate.bar(X - 0.18, cnt_b.values, width=0.36, label="Before")
        ax_gate.bar(X + 0.18, cnt_a.values, width=0.36, label="After")
        ax_gate.set_xticks(X, order);
        ax_gate.set_ylabel("# clips")
        ax_gate.set_title(f"Stability Gate (τ_var={args.tau_var}, τ_mad={args.tau_mad})", fontsize=11)
        ax_gate.legend(fontsize=9)
        verdict_before = cnt_b.to_dict();
        verdict_after = cnt_a.to_dict()

        # 另外保存两条直方图（可选小图）
        try:
            fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.2, 3.2))
            ax1.hist(stab["var"], bins=30, alpha=0.8)
            ax1.axvline(args.tau_var, linestyle="--", linewidth=1)
            ax1.set_title("Frame-level variance");
            ax1.set_xlabel("var");
            ax1.set_ylabel("count")
            ax2.hist(stab["mean_abs_diff"], bins=30, alpha=0.8)
            ax2.axvline(args.tau_mad, linestyle="--", linewidth=1)
            ax2.set_title("Frame-level jitter");
            ax2.set_xlabel("mean_abs_diff");
            ax2.set_ylabel("count")
            fig2.tight_layout()
            plt.savefig(os.path.join(args.out_dir, "stability_hist.png"), dpi=160, bbox_inches="tight")
            plt.close(fig2)
        except Exception:
            pass

    fig.tight_layout()
    out_png = os.path.join(args.out_dir, "report_compact.png")
    plt.savefig(out_png, dpi=170, bbox_inches="tight")
    plt.close(fig)

    # --- Save a tiny summary table for README ---
    summary = {
        "AUC": round(auc, 4),
        "EER": round(float((fpr[i_mark] + (1 - tpr[i_mark])) / 2), 4),
        "thr_at_EER": round(thr_eer, 3),
        "FPR@EER": round(cm_stats["FPR@EER"], 4),
        "FNR@EER": round(cm_stats["FNR@EER"], 4),
        "TN": cm_stats["TN"], "FP": cm_stats["FP"], "FN": cm_stats["FN"], "TP": cm_stats["TP"],
        "verdict_before": verdict_before,
        "verdict_after": verdict_after
    }
    pd.DataFrame([summary]).to_csv(os.path.join(args.out_dir, "summary_table.csv"), index=False)

    print(f"[saved] {out_png}")
    print(f"[saved] {os.path.join(args.out_dir, 'summary_table.csv')}")
    if os.path.exists(os.path.join(args.out_dir, "stability_hist.png")):
        print(f"[saved] {os.path.join(args.out_dir, 'stability_hist.png')}")


if __name__ == "__main__":
    main()
