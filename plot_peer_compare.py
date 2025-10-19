import argparse
import os
import re
from math import ceil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import gridspec


def _basename_lower(s: str) -> str:
    return os.path.basename(str(s)).lower()


def _natural_key(s: str):
    parts = re.split(r'(\d+)', str(s))
    return [int(p) if p.isdigit() else p for p in parts]


def _read_sample_csv(p, suffix):
    df = pd.read_csv(p)
    if "sample_id" not in df.columns or "mean_score" not in df.columns:
        raise ValueError(f"{p} must contain columns: sample_id, mean_score")
    if "verdict" not in df.columns: df["verdict"] = ""
    out = df.copy()
    out["mean_score"] = pd.to_numeric(out["mean_score"], errors="coerce")
    out = out[["sample_id", "mean_score", "verdict"]].rename(
        columns={"mean_score": f"mean_{suffix}", "verdict": f"verdict_{suffix}"}
    )
    return out


def _read_image_csv(p):
    df = pd.read_csv(p)
    need = {"sample_id", "image"}
    if "score_fake" not in df.columns:
        # 兼容有些导出列名叫 score
        if "score" in df.columns:
            df = df.rename(columns={"score": "score_fake"})
    need |= {"score_fake"}
    if not need.issubset(df.columns):
        raise ValueError(f"{p} must contain columns: {sorted(list(need))}")
    out = df[["sample_id", "image", "score_fake"]].copy()
    out["image_key"] = out["image"].astype(str).map(_basename_lower)
    return out


def _build_image_pairs(img_b, img_a, sample_keep):
    """
    按 sample_id 取交集，再按图像文件名（basename，不区分大小写）对齐。
    返回 dict[sample_id] = list of (img_name, score_b, score_a)
    """
    pairs = {}
    for sid in sample_keep:
        xb = img_b[img_b["sample_id"].astype(str) == sid]
        xa = img_a[img_a["sample_id"].astype(str) == sid]
        if xb.empty or xa.empty:
            continue
        mb = xb.set_index("image_key")["score_fake"]
        ma = xa.set_index("image_key")["score_fake"]
        inter = sorted(list(set(mb.index) & set(ma.index)), key=_natural_key)
        if not inter:
            continue
        pairs[sid] = [(k, float(mb[k]), float(ma[k])) for k in inter]
    return pairs


# ================== 新增：dodge + 斜率图工具函数 ==================
def _dodge_by_value(vals, base_x=0.0, width=0.20, round_decimals=3):
    """
    针对相同(四舍五入后)的 y 值分组，在 x 方向做轻微错位，避免重叠。
    返回与 vals 等长的一组 x 坐标。
    """
    v = np.round(np.asarray(vals, float), round_decimals)
    uniq, inv, counts = np.unique(v, return_inverse=True, return_counts=True)
    offs = np.zeros_like(v, dtype=float)
    for g, c in enumerate(counts):
        if c <= 1:
            continue
        # 对称分布：-0.5 ~ +0.5
        k = np.arange(c) - (c - 1) / 2.0
        offs[inv == g] = (k / max(c - 1, 1)) * 0.9  # 稍微缩一点，避免越界
    return base_x + offs * width


def _plot_slope_with_dodge(ax, y0, y1, thr=None, title=None,
                           lw=0.8, ms=3, alpha=0.65, width=0.22, round_decimals=3):
    """
    画 before/after 斜率图，带 dodge，减少重叠遮挡。
    """
    y0 = np.asarray(y0, float)
    y1 = np.asarray(y1, float)

    # 按变化量排序，减少交叉遮挡
    order = np.argsort(y1 - y0)
    y0, y1 = y0[order], y1[order]

    x0 = _dodge_by_value(y0, base_x=0.0, width=width, round_decimals=round_decimals)
    x1 = _dodge_by_value(y1, base_x=1.0, width=width, round_decimals=round_decimals)

    for i in range(len(y0)):
        ax.plot([x0[i], x1[i]], [y0[i], y1[i]],
                lw=lw, alpha=alpha, zorder=2)
        ax.scatter([x0[i], x1[i]], [y0[i], y1[i]],
                   s=ms * ms, alpha=min(1.0, alpha + 0.2), zorder=3)
    if thr is not None:
        ax.axhline(thr, ls=":", lw=0.9, zorder=1)

    ax.set_xlim(-0.35, 1.35)
    ax.set_ylim(0, 1.0)
    ax.set_xticks([0, 1], ["before", "after"])
    if title:
        ax.set_title(title)


# ============================================================

def main():
    ap = argparse.ArgumentParser()
    # per-sample 对比（必填）
    ap.add_argument("--before", required=True, help="基线 preds_per_sample.csv")
    ap.add_argument("--after", required=True, help="SA0 版 preds_per_sample.csv")
    # per-image 对比（可选；提供就会画在同一张图下半部分）
    ap.add_argument("--before_img", default=None, help="基线 preds_per_image.csv")
    ap.add_argument("--after_img", default=None, help="SA0 版 preds_per_image.csv")
    ap.add_argument("--out_dir", required=True, help="输出目录")
    ap.add_argument("--thresh", type=float, default=0.45)
    ap.add_argument("--grid_cols", type=int, default=4, help="下排每行最多小面板数")
    args = ap.parse_args()

    out_dir = Path(args.out_dir);
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------- per-sample ----------
    b = _read_sample_csv(args.before, "before")
    a = _read_sample_csv(args.after, "after")
    m = b.merge(a, on="sample_id", how="inner")
    if len(m) == 0:
        raise ValueError("两个 per-sample CSV 没有共同的 sample_id。")
    m["delta"] = m["mean_after"] - m["mean_before"]
    m["verdict_change"] = (m["verdict_before"].astype(str) != m["verdict_after"].astype(str))
    m = m.sort_values("sample_id")
    # 保存汇总
    (out_dir / "peer_video_compare_summary.csv").write_text(m.to_csv(index=False), encoding="utf-8")

    def vcnt(s):
        return s.value_counts().reindex(["real", "fake", "uncertain"], fill_value=0)
    cnt_b = vcnt(m["verdict_before"])
    cnt_a = vcnt(m["verdict_after"])

    # ---------- per-image（可选） ----------
    have_image_panels = (args.before_img and args.after_img)
    pairs = {}
    if have_image_panels:
        ib = _read_image_csv(args.before_img)
        ia = _read_image_csv(args.after_img)
        pairs = _build_image_pairs(ib, ia, sample_keep=m["sample_id"].astype(str).tolist())
        # 只保留确实有对齐帧的样本顺序
        sample_order = [sid for sid in m["sample_id"].astype(str).tolist() if sid in pairs]
    else:
        sample_order = []

    # ---------- 画图 ----------
    # 布局：上排 1x3（散点、分布、slope）；下排（每个样本一个小面板）
    n_panels = len(sample_order)
    n_cols = min(args.grid_cols, max(1, n_panels))
    n_rows = int(ceil(n_panels / n_cols)) if n_panels > 0 else 0

    top_h = 3.8
    cell_h = 2.4
    fig_h = top_h + (n_rows * cell_h if n_rows > 0 else 0) + 0.6
    fig_w = 12.0
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = gridspec.GridSpec(nrows=(1 + (n_rows if n_rows > 0 else 0)),
                           ncols=1, height_ratios=[top_h] + ([cell_h] * n_rows if n_rows > 0 else []),
                           hspace=0.35)

    # 顶部 row：再切 1x3
    gs_top = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[0], wspace=0.30)

    # (A) scatter: mean_before vs mean_after
    ax1 = fig.add_subplot(gs_top[0, 0])
    ax1.scatter(m["mean_before"], m["mean_after"], s=34, alpha=0.85)
    ax1.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    ax1.axvline(args.thresh, linestyle=":", linewidth=1)
    ax1.axhline(args.thresh, linestyle=":", linewidth=1)
    ax1.set_xlim(0, 1);
    ax1.set_ylim(0, 1)
    ax1.set_xlabel("mean_score (before)")
    ax1.set_ylabel("mean_score (after)")
    ax1.set_title("Scatter: before vs after")
    ax1.text(0.02, 0.95, f"N={len(m)}", transform=ax1.transAxes)

    # (B) verdict 分布前后
    ax2 = fig.add_subplot(gs_top[0, 1])
    X = np.arange(3)
    ax2.bar(X - 0.18, cnt_b.values, width=0.36, label="before")
    ax2.bar(X + 0.18, cnt_a.values, width=0.36, label="after")
    ax2.set_xticks(X, ["real", "fake", "uncertain"])
    ax2.set_title("Verdict distribution")
    ax2.legend()

    # (C) 每个样本前后 slope —— 改为带 dodge 的版本
    ax3 = fig.add_subplot(gs_top[0, 2])
    _plot_slope_with_dodge(
        ax3, m["mean_before"].values, m["mean_after"].values,
        thr=args.thresh, title="Per-sample change (slope)",
        lw=0.8, ms=3, alpha=0.65, width=0.5, round_decimals=3
    )
    ax3.set_ylabel("mean_score")

    # 下排：每样本一小面板（per-image）—— 同样用 dodge 的斜率图
    if n_rows > 0:
        for r in range(n_rows):
            gs_row = gridspec.GridSpecFromSubplotSpec(1, n_cols, subplot_spec=gs[1 + r], wspace=0.25)
            for c in range(n_cols):
                idx = r * n_cols + c
                ax = fig.add_subplot(gs_row[0, c])
                if idx >= n_panels:
                    ax.axis("off")
                    continue
                sid = sample_order[idx]
                lst = pairs[sid]  # list of (img_name, sb, sa)
                # 排序：按图名自然序，保证一致
                lst = sorted(lst, key=lambda t: _natural_key(t[0]))
                sb = [t[1] for t in lst]
                sa = [t[2] for t in lst]
                _plot_slope_with_dodge(
                    ax, sb, sa, thr=args.thresh, title=f"{sid} (N={len(lst)})",
                    lw=0.7, ms=2, alpha=0.70, width=0.24, round_decimals=3
                )
                ax.set_yticks([0, 0.5, 1.0])

        # 整体下排大标题
        # fig.text(0.5, (top_h + n_rows*cell_h)/fig_h - 0.02,
        #          "Per-image change per sample (image matched by filename)",
        #          ha="center", va="top", fontsize=11)

    plt.tight_layout()
    fig_path = out_dir / "peer_video_compare.png"
    plt.savefig(fig_path, dpi=160, bbox_inches="tight")
    print(f"Saved figure -> {fig_path.as_posix()}")

if __name__ == "__main__":
    main()
