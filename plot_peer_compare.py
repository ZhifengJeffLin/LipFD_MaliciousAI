import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--before", required=True, help="基线 preds_per_sample.csv")
    ap.add_argument("--after", required=True, help="SA0 版 preds_per_sample.csv")
    ap.add_argument("--out_dir", required=True, help="输出目录")
    ap.add_argument("--thresh", type=float, default=0.45)
    args = ap.parse_args()

    out_dir = Path(args.out_dir);
    out_dir.mkdir(parents=True, exist_ok=True)

    # 读取
    b = pd.read_csv(args.before)
    a = pd.read_csv(args.after)

    # 兼容列名并转浮点
    def norm(df, suffix):
        df = df.copy()
        if "sample_id" not in df.columns: raise ValueError("csv 缺少 sample_id")
        if "mean_score" not in df.columns: raise ValueError("csv 缺少 mean_score")
        if "verdict" not in df.columns: df["verdict"] = ""
        df["mean_score"] = pd.to_numeric(df["mean_score"], errors="coerce")
        df = df[["sample_id", "mean_score", "verdict"]].rename(
            columns={"mean_score": f"mean_{suffix}", "verdict": f"verdict_{suffix}"}
        )
        return df

    b = norm(b, "before")
    a = norm(a, "after")

    m = b.merge(a, on="sample_id", how="inner")
    if len(m) == 0:
        raise ValueError("两个 csv 的 sample_id 没有交集，请确认同一批样本。")

    m["delta"] = m["mean_after"] - m["mean_before"]
    m["verdict_change"] = (m["verdict_before"].astype(str) != m["verdict_after"].astype(str))

    # 保存汇总
    m.sort_values("sample_id").to_csv(out_dir / "peer_video_compare_summary.csv", index=False)

    # 统计
    def vcnt(s):
        return s.value_counts().reindex(["real", "fake", "uncertain"], fill_value=0)

    cnt_b = vcnt(m["verdict_before"])
    cnt_a = vcnt(m["verdict_after"])

    # ---- 画图：三联图 ----
    fig = plt.figure(figsize=(12, 4.2))

    # (A) scatter: mean_before vs mean_after
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.scatter(m["mean_before"], m["mean_after"], s=30, alpha=0.8)
    ax1.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    ax1.axvline(args.thresh, linestyle=":", linewidth=1)
    ax1.axhline(args.thresh, linestyle=":", linewidth=1)
    ax1.set_xlim(0, 1);
    ax1.set_ylim(0, 1)
    ax1.set_xlabel("mean_score (before)")
    ax1.set_ylabel("mean_score (after)")
    ax1.set_title("Scatter: before vs after")
    ax1.text(0.02, 0.95, f"N={len(m)}", transform=ax1.transAxes)

    # (B) verdict 分布前后对比
    ax2 = fig.add_subplot(1, 3, 2)
    X = np.arange(3)
    ax2.bar(X - 0.18, cnt_b.values, width=0.36, label="before")
    ax2.bar(X + 0.18, cnt_a.values, width=0.36, label="after")
    ax2.set_xticks(X, ["real", "fake", "uncertain"])
    ax2.set_title("Verdict distribution")
    ax2.legend()

    # (C) slopegraph: 每个样本前后变化
    ax3 = fig.add_subplot(1, 3, 3)
    x0, x1 = 0.0, 1.0
    for _, r in m.iterrows():
        y0, y1 = r["mean_before"], r["mean_after"]
        ls = "-" if r["verdict_change"] else "-"
        ax3.plot([x0, x1], [y0, y1], linestyle=ls, linewidth=1, alpha=0.7)
        ax3.scatter([x0, x1], [y0, y1], s=12)
    ax3.axhline(args.thresh, linestyle=":", linewidth=1)
    ax3.set_xlim(-0.1, 1.1);
    ax3.set_ylim(0, 1)
    ax3.set_xticks([x0, x1], ["before", "after"])
    ax3.set_ylabel("mean_score")
    ax3.set_title("Per-sample change (slope)")

    plt.tight_layout()
    fig_path = out_dir / "peer_video_compare.png"
    plt.savefig(fig_path, dpi=160, bbox_inches="tight")
    print(f"Saved figure -> {fig_path.as_posix()}")


if __name__ == "__main__":
    main()
