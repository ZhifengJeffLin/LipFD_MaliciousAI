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
    if "verdict" not in df.columns:
        df["verdict"] = ""
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
        # Compatibility: some exports name this column 'score'
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
    Take intersection by sample_id, then align by image filename (basename, case-insensitive).
    Returns dict[sample_id] = list of (img_name, score_b, score_a)
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


# ================== New: dodge + slope plot helper functions ==================
def _dodge_by_value(vals, base_x=0.0, width=0.20, round_decimals=3):
    """
    For values with identical (rounded) y, apply small horizontal jitter to avoid overlapping.
    Returns an array of x positions with the same length as vals.
    """
    v = np.round(np.asarray(vals, float), round_decimals)
    uniq, inv, counts = np.unique(v, return_inverse=True, return_counts=True)
    offs = np.zeros_like(v, dtype=float)
    for g, c in enumerate(counts):
        if c <= 1:
            continue
        # Symmetric distribution: -0.5 ~ +0.5
        k = np.arange(c) - (c - 1) / 2.0
        offs[inv == g] = (k / max(c - 1, 1)) * 0.9  # Slightly shrink to stay within bounds
    return base_x + offs * width


def _plot_slope_with_dodge(ax, y0, y1, thr=None, title=None,
                           lw=0.8, ms=3, alpha=0.65, width=0.22, round_decimals=3):
    """
    Draw before/after slope plot with dodge to reduce overlap.
    """
    y0 = np.asarray(y0, float)
    y1 = np.asarray(y1, float)

    # Sort by change amount to reduce crossing
    order = np.argsort(y1 - y0)
    y0, y1 = y0[order], y1[order]

    x0 = _dodge_by_value(y0, base_x=0.0, width=width, round_decimals=round_decimals)
    x1 = _dodge_by_value(y1, base_x=1.0, width=width, round_decimals=round_decimals)

    for i in range(len(y0)):
        ax.plot([x0[i], x1[i]], [y0[i], y1[i]], lw=lw, alpha=alpha, zorder=2)
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
    # Per-sample comparison (required)
    ap.add_argument("--before", required=True, help="Baseline preds_per_sample.csv")
    ap.add_argument("--after", required=True, help="SA0 version preds_per_sample.csv")
    # Per-image comparison (optional; plotted in lower part if provided)
    ap.add_argument("--before_img", default=None, help="Baseline preds_per_image.csv")
    ap.add_argument("--after_img", default=None, help="SA0 version preds_per_image.csv")
    ap.add_argument("--out_dir", required=True, help="Output directory")
    ap.add_argument("--thresh", type=float, default=0.45)
    ap.add_argument("--grid_cols", type=int, default=4, help="Maximum number of panels per row (bottom)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------- Per-sample ----------
    b = _read_sample_csv(args.before, "before")
    a = _read_sample_csv(args.after, "after")
    m = b.merge(a, on="sample_id", how="inner")
    if len(m) == 0:
        raise ValueError("The two per-sample CSV files have no common sample_id.")
    m["delta"] = m["mean_after"] - m["mean_before"]
    m["verdict_change"] = (m["verdict_before"].astype(str) != m["verdict_after"].astype(str))
    m = m.sort_values("sample_id")
    # Save summary
    (out_dir / "peer_video_compare_summary.csv").write_text(m.to_csv(index=False), encoding="utf-8")

    def vcnt(s):
        return s.value_counts().reindex(["real", "fake", "uncertain"], fill_value=0)
    cnt_b = vcnt(m["verdict_before"])
    cnt_a = vcnt(m["verdict_after"])

    # ---------- Per-image (optional) ----------
    have_image_panels = (args.before_img and args.after_img)
    pairs = {}
    if have_image_panels:
        ib = _read_image_csv(args.before_img)
        ia = _read_image_csv(args.after_img)
        pairs = _build_image_pairs(ib, ia, sample_keep=m["sample_id"].astype(str).tolist())
        # Keep only samples that have aligned frames
        sample_order = [sid for sid in m["sample_id"].astype(str).tolist() if sid in pairs]
    else:
        sample_order = []

    # ---------- Plotting ----------
    # Layout: top row 1x3 (scatter, distribution, slope); bottom row (one panel per sample)
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

    # Top row: split into 1x3
    gs_top = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[0], wspace=0.30)

    # (A) scatter: mean_before vs mean_after
    ax1 = fig.add_subplot(gs_top[0, 0])
    ax1.scatter(m["mean_before"], m["mean_after"], s=34, alpha=0.85)
    ax1.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    ax1.axvline(args.thresh, linestyle=":", linewidth=1)
    ax1.axhline(args.thresh, linestyle=":", linewidth=1)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.set_xlabel("mean_score (before)")
    ax1.set_ylabel("mean_score (after)")
    ax1.set_title("Scatter: before vs after")
    ax1.text(0.02, 0.95, f"N={len(m)}", transform=ax1.transAxes)

    # (B) Verdict distribution before/after
    ax2 = fig.add_subplot(gs_top[0, 1])
    X = np.arange(3)
    ax2.bar(X - 0.18, cnt_b.values, width=0.36, label="before")
    ax2.bar(X + 0.18, cnt_a.values, width=0.36, label="after")
    ax2.set_xticks(X, ["real", "fake", "uncertain"])
    ax2.set_title("Verdict distribution")
    ax2.legend()

    # (C) Per-sample slope (dodge version)
    ax3 = fig.add_subplot(gs_top[0, 2])
    _plot_slope_with_dodge(
        ax3, m["mean_before"].values, m["mean_after"].values,
        thr=args.thresh, title="Per-sample change (slope)",
        lw=0.8, ms=3, alpha=0.65, width=0.5, round_decimals=3
    )
    ax3.set_ylabel("mean_score")

    # Bottom row: per-sample panels (per-image) — also slope plot with dodge
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
                # Sort by filename naturally to ensure consistency
                lst = sorted(lst, key=lambda t: _natural_key(t[0]))
                sb = [t[1] for t in lst]
                sa = [t[2] for t in lst]
                _plot_slope_with_dodge(
                    ax, sb, sa, thr=args.thresh, title=f"{sid} (N={len(lst)})",
                    lw=0.7, ms=2, alpha=0.70, width=0.24, round_decimals=3
                )
                ax.set_yticks([0, 0.5, 1.0])

        # Optional overall bottom title
        # fig.text(0.5, (top_h + n_rows*cell_h)/fig_h - 0.02,
        #          "Per-image change per sample (image matched by filename)",
        #          ha="center", va="top", fontsize=11)

    plt.tight_layout()
    fig_path = out_dir / "peer_video_compare.png"
    plt.savefig(fig_path, dpi=160, bbox_inches="tight")
    print(f"Saved figure -> {fig_path.as_posix()}")


if __name__ == "__main__":
    main()
