#!/usr/bin/env python
# -*- coding: utf-8 -*-
# robust_viz_compact.py
# 从 ./work/robust_eval/*/preds_per_sample.csv 读取，合并出少量总图

import argparse
import math
import os
import re

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix


def infer_label(sample_id: str) -> int:
    s = str(sample_id)
    if re.search(r'(^|/|_)1_fake', s): return 1
    if re.search(r'(^|/|_)0_real', s): return 0
    return -1


def load_metrics(csv_path: str):
    df = pd.read_csv(csv_path)[["sample_id", "mean_score"]].dropna().copy()
    df["label"] = df["sample_id"].apply(infer_label)
    df = df[df["label"] >= 0].reset_index(drop=True)

    y = df["label"].to_numpy(int)
    s = np.clip(df["mean_score"].astype(float).to_numpy(), 1e-9, 1 - 1e-9)

    auc = float(roc_auc_score(y, s))
    fpr, tpr, thr = roc_curve(y, s)
    idx = int(np.nanargmin(np.abs(fpr - (1 - tpr))))  # EER 点
    eer = float((fpr[idx] + (1 - tpr[idx])) / 2.0)
    thr_eer = float(thr[idx])

    pred = (s >= thr_eer).astype(int)
    TN, FP, FN, TP = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    fpr_eer = FP / (FP + TN + 1e-12)
    fnr_eer = FN / (FN + TP + 1e-12)
    return {
        "df": df, "y": y, "scores": s,
        "fpr": fpr, "tpr": tpr, "thr": thr, "eer_idx": idx,
        "AUC": auc, "EER": eer, "thr_at_EER": thr_eer,
        "FPR@EER": fpr_eer, "FNR@EER": fnr_eer,
        "TN": TN, "FP": FP, "FN": FN, "TP": TP
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="./work/robust_eval", help="根目录，包含 clean/q80/... 子目录")
    ap.add_argument("--splits", nargs="*", default=None,
                    help="要处理的子目录名；默认自动发现，按 clean,q80,q60,q40 排序")
    args = ap.parse_args()

    BASE = args.base
    if args.splits:
        splits = args.splits
    else:
        splits = []
        for d in sorted(os.listdir(BASE)):
            p = os.path.join(BASE, d, "preds_per_sample.csv")
            if os.path.isfile(p):
                splits.append(d)
        order = [x for x in ["clean", "q80", "q60", "q40"] if x in splits]
        others = [x for x in splits if x not in order]
        splits = order + others
    assert splits, f"No splits with preds_per_sample.csv under {BASE}"
    print("[splits]", splits)

    # 读取所有 split
    results = {}
    for name in splits:
        csv_path = os.path.join(BASE, name, "preds_per_sample.csv")
        results[name] = load_metrics(csv_path)

    # 图 1：叠加 ROC
    plt.figure(figsize=(6.2, 4.6))
    for name, res in results.items():
        plt.plot(res["fpr"], res["tpr"], label=f"{name} (AUC={res['AUC']:.3f})")
    plt.plot([0, 1], [0, 1], "--", linewidth=1)
    plt.xlabel("False Positive Rate");
    plt.ylabel("True Positive Rate")
    plt.title("ROC (clean vs qXX)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(BASE, "roc_overlay.png"), dpi=170, bbox_inches="tight");
    plt.close()

    # 图 2：AUC/EER 折线（合一张）
    names = list(results.keys())
    aucs = [results[n]["AUC"] for n in names]
    eers = [results[n]["EER"] for n in names]
    fig = plt.figure(figsize=(6.8, 4.2))
    ax1 = fig.add_subplot(111)
    ax1.plot(names, aucs, marker="o", label="AUC")
    ax1.set_ylabel("AUC")
    ax2 = ax1.twinx()
    ax2.plot(names, eers, marker="s", linestyle="--", label="EER", alpha=0.9)
    ax2.set_ylabel("EER")
    ax1.set_title("AUC & EER across splits")
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="best")
    plt.tight_layout()
    plt.savefig(os.path.join(BASE, "auc_eer_lines.png"), dpi=170, bbox_inches="tight");
    plt.close()

    # 图 3：分数分布网格（每格一个 split）
    n = len(names);
    rows = math.ceil(n / 2);
    cols = 2 if n > 1 else 1
    fig, axes = plt.subplots(rows, cols, figsize=(8.5, 3.6 * rows))
    if n == 1: axes = np.array([[axes]])
    axes = axes.flatten()
    for i, name in enumerate(names):
        ax = axes[i]
        res = results[name]
        r = res["df"]
        real_scores = r.loc[r["label"] == 0, "mean_score"].astype(float).to_numpy()
        fake_scores = r.loc[r["label"] == 1, "mean_score"].astype(float).to_numpy()
        ax.hist(real_scores, bins=30, alpha=0.5, label="real", density=True)
        ax.hist(fake_scores, bins=30, alpha=0.5, label="fake", density=True)
        ax.axvline(res["thr_at_EER"], linestyle="--", linewidth=1, label=f"thr@EER≈{res['thr_at_EER']:.3f}")
        ax.set_xlim(0, 1);
        ax.set_title(f"{name}: score distributions")
        if i == 0: ax.legend()
    # 清空多余子图
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    plt.tight_layout()
    plt.savefig(os.path.join(BASE, "dist_grid.png"), dpi=170, bbox_inches="tight");
    plt.close()

    # --- 图 4：混淆矩阵网格（统一色标 + 底部横向 colorbar） ---
    vmax = 0
    cms = {}
    for name, res in results.items():
        cm = np.array([[res["TN"], res["FP"]], [res["FN"], res["TP"]]], dtype=float)
        cms[name] = cm
        vmax = max(vmax, cm.max())

    rows = math.ceil(len(names) / 2);
    cols = 2 if len(names) > 1 else 1
    fig, axes = plt.subplots(rows, cols, figsize=(8.5, 3.6 * rows))
    axes = np.atleast_1d(axes).ravel()

    im = None
    for i, name in enumerate(names):
        ax = axes[i]
        cm = cms[name]
        im = ax.imshow(cm, vmin=0, vmax=vmax, cmap="viridis")  # 统一色标范围
        for r in range(2):
            for c in range(2):
                ax.text(c, r, int(cm[r, c]), ha="center", va="center",
                        color="white" if cm[r, c] > vmax * 0.6 else "black")
        ax.set_xticks([0, 1]);
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Pred Real", "Pred Fake"])
        ax.set_yticklabels(["True Real", "True Fake"])
        ax.set_title(f"{name}: Confusion @EER")

    # 删除多余子图
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    # 留出底部空间再放一个横向 colorbar（不遮挡任何子图）
    fig.subplots_adjust(left=0.08, right=0.98, top=0.92, bottom=0.16, wspace=0.25, hspace=0.35)
    cbar_ax = fig.add_axes([0.18, 0.08, 0.64, 0.04])  # [left, bottom, width, height] in figure coords
    fig.colorbar(im, cax=cbar_ax, orientation="horizontal", label="# clips")

    plt.savefig(os.path.join(BASE, "cm_grid.png"), dpi=170, bbox_inches="tight")
    plt.close()

    # 汇总 CSV（表格）
    rows_out = []
    for name in names:
        res = results[name]
        rows_out.append({
            "split": name,
            "AUC": round(res["AUC"], 4),
            "EER": round(res["EER"], 4),
            "thr_at_EER": round(res["thr_at_EER"], 3),
            "FPR@EER": round(res["FPR@EER"], 4),
            "FNR@EER": round(res["FNR@EER"], 4),
            "TN": int(res["TN"]), "FP": int(res["FP"]),
            "FN": int(res["FN"]), "TP": int(res["TP"]),
            "N_real": int((res["df"]["label"] == 0).sum()),
            "N_fake": int((res["df"]["label"] == 1).sum())
        })
    pd.DataFrame(rows_out).to_csv(os.path.join(BASE, "metrics_summary.csv"), index=False)
    print("[saved] ->", os.path.join(BASE, "roc_overlay.png"))
    print("[saved] ->", os.path.join(BASE, "auc_eer_lines.png"))
    print("[saved] ->", os.path.join(BASE, "dist_grid.png"))
    print("[saved] ->", os.path.join(BASE, "cm_grid.png"))
    print("[saved] ->", os.path.join(BASE, "metrics_summary.csv"))


if __name__ == "__main__":
    main()
