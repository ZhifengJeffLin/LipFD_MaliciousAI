#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
post_rule_aggressive.py
对 predict 的输出做规则“加成”（更容易判 fake），不改动模型本体。

输入：
  --preds_per_image   work/xxx/preds_per_image.csv（必需，含 sample_id,image,score_fake）
  --preds_per_sample  work/xxx/preds_per_sample.csv（可选，用来对齐已有 mean_score / verdict）
  --slices_root       复合图所在根目录（image 列通常是相对路径，如 1_fake/0_3.png）
  --out_dir           输出目录（默认：同级 out_dir_ruleboost）
  --num_tiles         唇条带水平帧数（默认 10；与你的预处理一致即可）
  --tau_low_motion    低运动阈值（默认 0.03；越小越严格）
  --tau_av_corr       音画相关性阈值（默认 0.20；低于此认为不同步）
  --boost_low_motion  命中低运动时加分（默认 0.07）
  --boost_av_mismatch 命中不同步时加分（默认 0.07）
  --boost_cap         每个 sample 的最大总加分（默认 0.15）
  --thresh            重新出 verdict 时的阈值（默认 0.50；你也可用前面算到的 thr_at_EER）
说明：
  - 脚本会读取每个 sample 的若干复合图（默认全读；你也可以在代码里把 k_images 调小）。
  - 对每张复合图：自动找“Mel/唇条带”的分界行；把唇条带均分成 num_tiles 份估计相邻差分；
    Mel 取列均值作为时间能量，和唇相邻差分做简单相关性。
  - 规则命中会对该 sample 的 mean_score 加小幅度；最后写出新的 per-sample CSV。

用法示例：
  python post_rule_aggressive.py \
    --preds_per_image ./work/other_team/preds_per_image.csv \
    --preds_per_sample ./work/other_team/preds_per_sample.csv \
    --slices_root ./datasets/AVLips/other_team_slices \
    --out_dir ./work/other_team_ruleboost \
    --thresh 0.45
"""

import argparse
import numpy as np
import pandas as pd
import re
from pathlib import Path

from PIL import Image


def infer_label(sample_id: str) -> int:
    s = str(sample_id)
    if re.search(r'(^|/|_)1_fake', s): return 1
    if re.search(r'(^|/|_)0_real', s): return 0
    return -1


def as_np01(png_path: Path):
    with Image.open(png_path) as im:
        arr = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0
    return arr


def find_boundary_row(img: np.ndarray):
    # 在高度维做行间梯度，找到 Mel/唇分界；失败则取 60% 处
    g = np.abs(np.diff(img, axis=0)).mean(axis=(1, 2))
    if g.size < 3: return int(img.shape[0] * 0.6)
    r = int(np.argmax(g))
    # 限制在 [30%, 80%] 区间，避免异常
    r = int(np.clip(r, int(0.3 * img.shape[0]), int(0.8 * img.shape[0])))
    return r


def split_tiles(lip_strip: np.ndarray, n: int):
    H, W, C = lip_strip.shape
    w = W // n
    tiles = [lip_strip[:, i * w:(i + 1) * w, :] for i in range(n)]
    return tiles


def lip_motion_seq(lip_strip: np.ndarray, n_tiles: int):
    tiles = split_tiles(lip_strip, n_tiles)
    # 相邻帧的平均绝对差
    diffs = []
    for i in range(1, len(tiles)):
        a = tiles[i - 1];
        b = tiles[i]
        m = np.mean(np.abs(a - b))
        diffs.append(m)
    return np.array(diffs, dtype=np.float32)  # 长度 n_tiles-1


def mel_energy_seq(mel: np.ndarray, out_len: int):
    # 每列取均值作为时间能量，再线性重采样到 out_len
    v = mel.mean(axis=0)  # (W,)
    if v.size <= 1:
        return np.zeros(out_len, dtype=np.float32)
    x_old = np.linspace(0, 1, num=v.size, dtype=np.float32)
    x_new = np.linspace(0, 1, num=out_len, dtype=np.float32)
    return np.interp(x_new, x_old, v).astype(np.float32)


def corr01(a: np.ndarray, b: np.ndarray):
    if a.size != b.size or a.size < 3: return 0.0
    A = (a - a.mean()) / (a.std() + 1e-6)
    B = (b - b.mean()) / (b.std() + 1e-6)
    return float(np.clip(np.mean(A * B), -1.0, 1.0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds_per_image", required=True)
    ap.add_argument("--preds_per_sample", default=None)
    ap.add_argument("--slices_root", required=True)
    ap.add_argument("--out_dir", default=None)

    ap.add_argument("--num_tiles", type=int, default=10)
    ap.add_argument("--tau_low_motion", type=float, default=0.03)
    ap.add_argument("--tau_av_corr", type=float, default=0.20)
    ap.add_argument("--boost_low_motion", type=float, default=0.07)
    ap.add_argument("--boost_av_mismatch", type=float, default=0.07)
    ap.add_argument("--boost_cap", type=float, default=0.15)

    ap.add_argument("--thresh", type=float, default=0.50)
    args = ap.parse_args()

    slices_root = Path(args.slices_root)
    out_dir = Path(args.out_dir or (Path(args.preds_per_image).parent / "out_ruleboost"))
    out_dir.mkdir(parents=True, exist_ok=True)

    pi = pd.read_csv(args.preds_per_image)
    need = {"sample_id", "image", "score_fake"}
    assert need.issubset(pi.columns), f"{args.preds_per_image} must contain {need}"

    # 读已有 per-sample（可选）
    if args.preds_per_sample and Path(args.preds_per_sample).exists():
        ps = pd.read_csv(args.preds_per_sample)
        base_cols = ["sample_id", "num_images", "mean_score", "median_score", "vote_fake(%)", "verdict"]
        base = ps[base_cols].copy() if set(base_cols).issubset(ps.columns) else ps[["sample_id", "mean_score"]].copy()
    else:
        # 只按 per-image 聚合出 mean_score
        g = pi.groupby("sample_id")["score_fake"].mean().reset_index()
        g = g.rename(columns={"score_fake": "mean_score"})
        base = g

    # 逐 sample 计算启发式特征
    logs = []
    boosts = {}
    for sid, g in pi.groupby("sample_id"):
        # 为了快，最多取 4 张合成图估计（可改）
        subset = g.head(4)
        low_motions = []
        av_corrs = []

        for _, row in subset.iterrows():
            img_path = slices_root / str(row["image"])
            if not img_path.exists():
                continue
            img = as_np01(img_path)
            r = find_boundary_row(img)
            mel = img[:r, :, :]
            lip = img[r:, :, :]

            # 低运动
            lm_seq = lip_motion_seq(lip, args.num_tiles)
            low_motions.append(float(lm_seq.mean() if lm_seq.size else 0.0))

            # 音画相关（把 mel 能量重采样到 len(lm_seq)）
            me_seq = mel_energy_seq(mel, max(1, lm_seq.size))
            av_corrs.append(corr01(me_seq, lm_seq) if lm_seq.size else 0.0)

        if len(low_motions) == 0:
            boost = 0.0
            logs.append({"sample_id": sid, "low_motion": np.nan, "av_corr": np.nan, "boost": boost})
            boosts[sid] = boost
            continue

        lm = float(np.mean(low_motions))
        ac = float(np.mean(av_corrs)) if len(av_corrs) > 0 else 0.0

        boost = 0.0
        if lm < args.tau_low_motion:
            boost += args.boost_low_motion
        if abs(ac) < args.tau_av_corr:
            boost += args.boost_av_mismatch
        boost = float(min(args.boost_cap, boost))

        logs.append({"sample_id": sid, "low_motion": lm, "av_corr": ac, "boost": boost})
        boosts[sid] = boost

    log_df = pd.DataFrame(logs).sort_values("sample_id")
    log_df.to_csv(out_dir / "ruleboost_log.csv", index=False)

    # 应用加分
    base["boost"] = base["sample_id"].map(lambda s: boosts.get(s, 0.0))
    base["mean_score_rule"] = np.clip(base["mean_score"].astype(float) + base["boost"].astype(float), 0.0, 1.0)
    # 重新出 verdict
    base["verdict_rule"] = np.where(base["mean_score_rule"] >= args.thresh, "fake", "real")
    base.to_csv(out_dir / "preds_per_sample_ruleboost.csv", index=False)

    # 小统计
    n = len(base)
    n_boost = int((base["boost"] > 0).sum())
    print(f"[ruleboost] samples={n}, boosted={n_boost} ({n_boost / max(1, n):.1%}), "
          f"avg_boost={base['boost'].mean():.4f}, thresh={args.thresh:.3f}")
    print(f"[saved] {out_dir / 'ruleboost_log.csv'}")
    print(f"[saved] {out_dir / 'preds_per_sample_ruleboost.csv'}")

    # 生成一个小图（前后 verdict 分布）
    try:
        import matplotlib.pyplot as plt
        # 如果有旧 verdict 就对比，没有就只画 rule verdict
        if "verdict" in base.columns:
            import collections
            def cnt(s):
                return collections.Counter([str(x) for x in s])

            left = cnt(base["verdict"])
            right = cnt(base["verdict_rule"])
            keys = ["real", "fake"]
            X = np.arange(len(keys))
            plt.figure(figsize=(5.6, 3.4))
            plt.bar(X - 0.18, [left.get(k, 0) for k in keys], width=0.36, label="Before")
            plt.bar(X + 0.18, [right.get(k, 0) for k in keys], width=0.36, label="After (rule)")
            plt.xticks(X, keys);
            plt.ylabel("# clips");
            plt.title("Verdict change by rule boost")
            plt.legend();
            plt.tight_layout()
            plt.savefig(out_dir / "verdict_change.png", dpi=160, bbox_inches="tight");
            plt.close()
            print(f"[saved] {out_dir / 'verdict_change.png'}")
    except Exception as e:
        print("plot skipped:", e)


if __name__ == "__main__":
    main()
