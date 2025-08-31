# tools/predict_grouped_images.py
# 对同一目录下的“合成图”（上半音频条纹 + 下半口型拼图）逐图判别，并按 sample_id（文件名下划线前缀）聚合。
import argparse
import csv
import json
import re
import shutil
from collections import defaultdict, Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

# ---------- 默认参数（不传命令行时就用这些） ----------
DEFAULTS = dict(
    input_dir="./datasets/AVLips/0_real",  # 要判别的合成图目录；命名如 0_0.png, 0_1.png, 100_4.png ...
    ckpt="./checkpoints/ckpt.pth",  # 预训练权重
    arch="CLIP:ViT-L/14",  # 与 validate.py 一致
    gpu=0,
    tmp_root="./datasets/_group_tmp",  # 临时目录（自动清理重建）
    batch_size=8,
    thresh=0.50,  # >=thresh 判 fake
    uncert_band=0.10,  # 0.5±band 判 uncertain（样本聚合时）
    out_dir="./work",
    per_image_csv=None,  # 若为 None，则用 {out_dir}/preds_per_image.csv
    per_sample_csv=None,  # 若为 None，则用 {out_dir}/preds_per_sample.csv
    gt_json=None,  # 可选：一个 JSON 文件，形如 {"0":1,"100":1,"42":0}
    preview_n=20  # 逐图结果预览条数
)
# -----------------------------------------------------

# 复用仓库的数据集与模型
from data import AVLip
from models import build_model


def list_images(folder: Path):
    exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp")
    files = []
    for e in exts:
        files.extend(folder.glob(e))
    return sorted(files)


def sample_key_from_name(name: str) -> str:
    """取下划线前缀作为 sample_id，如 '0_9.png' -> '0'，'100_4.png' -> '100'。"""
    m = re.match(r'^([^_]+)_', name)
    return m.group(1) if m else name  # 若无下划线则用整名


def prepare_tmp_group(imgs, tmp_root: Path):
    """把一组图复制到 real/gen，交给 AVLip 解析（不改仓库数据管线）。"""
    if tmp_root.exists():
        shutil.rmtree(tmp_root, ignore_errors=True)
    (tmp_root / "real").mkdir(parents=True, exist_ok=True)
    (tmp_root / "gen").mkdir(parents=True, exist_ok=True)
    for p in imgs:
        shutil.copyfile(p, tmp_root / "real" / p.name)
        shutil.copyfile(p, tmp_root / "gen" / p.name)
    return str(tmp_root / "real"), str(tmp_root / "gen")


@torch.no_grad()
def infer_group(model, device, imgs, tmp_root: Path, batch_size=8):
    """对一组图做一次批量推理，返回与 imgs 顺序对齐的 fake 概率数组。"""
    real_dir, fake_dir = prepare_tmp_group(imgs, tmp_root)

    class Opt:
        pass

    opt = Opt()
    opt.real_list_path = real_dir
    opt.fake_list_path = fake_dir
    opt.max_sample = len(imgs)
    opt.batch_size = 1
    opt.data_label = "val"

    dataset = AVLip(opt)
    if len(dataset) == 0:
        raise RuntimeError("AVLip 数据集为空：请确认这些图是官方 preprocess 生成的‘合成图’。")
    loader = DataLoader(dataset, batch_size=min(batch_size, len(imgs)), shuffle=False, num_workers=0)

    scores = []
    for img, crops, _ in loader:
        img = img.to(device)
        crops = [[t.to(device) for t in sub] for sub in crops]
        feats = model.get_features(img).to(device)
        prob = torch.sigmoid(model(crops, feats)[0]).flatten()  # [B]
        scores.extend(prob.cpu().numpy().tolist())
    return scores


def print_table(headers, rows, col_w=None):
    """简单表格打印（不引第三方包）"""
    if col_w is None:
        col_w = [max(len(str(h)), *(len(str(r[i])) for r in rows)) + 2 for i, h in enumerate(headers)]
    line = '+' + '+'.join('-' * w for w in col_w) + '+'

    def fmt_row(r):
        return '|' + '|'.join(str(x).ljust(w) for x, w in zip(r, col_w)) + '|'

    print(line)
    print(fmt_row(headers))
    print(line)
    for r in rows:
        print(fmt_row(r))
    print(line)


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--input_dir", type=str, default=DEFAULTS["input_dir"])
    p.add_argument("--ckpt", type=str, default=DEFAULTS["ckpt"])
    p.add_argument("--arch", type=str, default=DEFAULTS["arch"])
    p.add_argument("--gpu", type=int, default=DEFAULTS["gpu"])
    p.add_argument("--tmp_root", type=str, default=DEFAULTS["tmp_root"])
    p.add_argument("--batch_size", type=int, default=DEFAULTS["batch_size"])
    p.add_argument("--thresh", type=float, default=DEFAULTS["thresh"])
    p.add_argument("--uncert_band", type=float, default=DEFAULTS["uncert_band"])
    p.add_argument("--out_dir", type=str, default=DEFAULTS["out_dir"])
    p.add_argument("--per_image_csv", type=str, default=DEFAULTS["per_image_csv"])
    p.add_argument("--per_sample_csv", type=str, default=DEFAULTS["per_sample_csv"])
    p.add_argument("--gt_json", type=str, default=DEFAULTS["gt_json"],
                   help="可选：JSON 文件路径，内容为 {'sample_id': 0/1, ...}")
    p.add_argument("--preview_n", type=int, default=DEFAULTS["preview_n"])
    return p.parse_args()


def main():
    args = parse_args()

    INPUT_DIR = Path(args.input_dir)
    CKPT_PATH = Path(args.ckpt)
    ARCH = args.arch
    GPU_ID = args.gpu
    TMP_ROOT = Path(args.tmp_root)
    BATCH_SIZE = args.batch_size
    THRESH = args.thresh
    UNCERT_BAND = args.uncert_band
    OUT_DIR = Path(args.out_dir)
    OUT_PER_IMAGE_CSV = Path(args.per_image_csv) if args.per_image_csv else OUT_DIR / "preds_per_image.csv"
    OUT_PER_SAMPLE_CSV = Path(args.per_sample_csv) if args.per_sample_csv else OUT_DIR / "preds_per_sample.csv"

    # 读取可选的 GT JSON
    GT_LABELS = {}
    if args.gt_json:
        p = Path(args.gt_json)
        if p.exists():
            GT_LABELS = json.loads(p.read_text(encoding="utf-8"))
        else:
            print(f"[warn] gt_json 文件不存在：{p}")

    # 设备 & 模型
    device = torch.device(f"cuda:{GPU_ID}" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device} | arch={ARCH}")
    model = build_model(ARCH)
    state = torch.load(CKPT_PATH, map_location="cpu")
    model.load_state_dict(state["model"] if "model" in state else state)
    model.to(device).eval()
    print("[ok] model loaded")

    # 收集全部图片并分组
    all_imgs = list_images(INPUT_DIR)
    if not all_imgs:
        raise RuntimeError(f"在 {INPUT_DIR} 下没有找到合成图（png/jpg）。")
    groups = defaultdict(list)
    for p in all_imgs:
        groups[sample_key_from_name(p.name)].append(p)
    for sid in groups:
        groups[sid] = sorted(groups[sid], key=lambda x: x.name)

    # 准备输出
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    per_img_rows = [("sample_id", "image", "score_fake", "pred(1=fake)", "pred_name", "gt", "correct")]
    per_smp_rows = [("sample_id", "num_images", "mean_score", "median_score", "vote_fake(%)", "verdict", "gt",
                     "sample_acc(%)", "sample_correct")]

    # 统计
    total_images = 0
    total_samples = len(groups)
    sample_pred_counter = Counter()
    image_pred_counter = Counter()

    # 逐组推理
    for sid, imgs in sorted(groups.items(), key=lambda kv: kv[0]):
        scores = infer_group(model, device, imgs, TMP_ROOT, BATCH_SIZE)
        preds = [1 if s >= THRESH else 0 for s in scores]
        gt = GT_LABELS.get(sid, None)

        # —— 逐图结果
        for pth, s, y in zip(imgs, scores, preds):
            pred_name = "fake" if y == 1 else "real"
            total_images += 1
            image_pred_counter[pred_name] += 1
            if gt is None:
                per_img_rows.append((sid, pth.name, f"{s:.6f}", y, pred_name, "", ""))
            else:
                ok = int(y == int(gt))
                per_img_rows.append((sid, pth.name, f"{s:.6f}", y, pred_name, int(gt), ok))

        # —— 聚合为 sample 结果
        scores_np = np.array(scores, dtype=float)
        mean_s = float(scores_np.mean())
        median_s = float(np.median(scores_np))
        vote_fake = 100.0 * (sum(preds) / len(preds))
        if abs(mean_s - 0.5) <= UNCERT_BAND:
            verdict = "uncertain"
        elif mean_s >= THRESH:
            verdict = "fake"
        else:
            verdict = "real"
        sample_pred_counter[verdict] += 1

        if gt is None:
            per_smp_rows.append(
                (sid, len(imgs), f"{mean_s:.6f}", f"{median_s:.6f}", f"{vote_fake:.1f}", verdict, "", "", ""))
        else:
            per_img_correct = [int(y == int(gt)) for y in preds]
            smp_acc = 100.0 * (sum(per_img_correct) / len(per_img_correct))
            sample_correct = int((1 if verdict == "fake" else 0) == int(gt)) if verdict in ("fake", "real") else ""
            per_smp_rows.append((sid, len(imgs), f"{mean_s:.6f}", f"{median_s:.6f}", f"{vote_fake:.1f}", verdict,
                                 int(gt), f"{smp_acc:.1f}", sample_correct))

    # —— 导出 CSV
    with open(OUT_PER_IMAGE_CSV, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(per_img_rows)
    with open(OUT_PER_SAMPLE_CSV, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(per_smp_rows)

    # —— 终端打印（表格 + 总结）
    print("\n【逐样本聚合结果】")
    headers_s = ["sample_id", "num_images", "mean_score", "median_score", "vote_fake(%)", "verdict"]
    rows_s = [[r[0], r[1], r[2], r[3], r[4], r[5]] for r in per_smp_rows[1:]]
    print_table(headers_s, rows_s)

    print("\n【逐图结果（前 %d 条预览）】" % args.preview_n)
    headers_i = ["sample_id", "image", "score_fake", "pred"]
    preview = [[r[0], r[1], r[2], r[4]] for r in per_img_rows[1:1 + args.preview_n]]
    print_table(headers_i, preview)

    print("\n【全部预测汇总】")
    total_fake_samples = sample_pred_counter["fake"]
    total_real_samples = sample_pred_counter["real"]
    total_uncertain_samples = sample_pred_counter["uncertain"]
    total = total_samples
    rows_summary = [
        ["total_samples", total],
        ["fake_samples", f"{total_fake_samples} ({(total_fake_samples / total):.1%})" if total else "0"],
        ["real_samples", f"{total_real_samples} ({(total_real_samples / total):.1%})" if total else "0"],
        ["uncertain_samples", f"{total_uncertain_samples} ({(total_uncertain_samples / total):.1%})" if total else "0"],
        ["total_images", total_images],
        ["image_preds", f"fake={image_pred_counter['fake']}, real={image_pred_counter['real']}"],
    ]
    print_table(["metric", "value"], rows_summary)

    print("saved ->", OUT_PER_IMAGE_CSV.as_posix())
    print("saved ->", OUT_PER_SAMPLE_CSV.as_posix())


if __name__ == "__main__":
    main()
