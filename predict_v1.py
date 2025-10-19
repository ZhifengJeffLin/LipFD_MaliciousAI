# tools/predict_grouped_images.py
# 多目录输入的“合成图”逐图判别，并按 sample_id（文件名下划线前缀）+ 目录前缀聚合。
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
    input_dir="./datasets/AVLips/0_real",  # 兼容：单目录
    ckpt="./checkpoints/ckpt.pth",
    arch="CLIP:ViT-L/14",
    gpu=0,
    tmp_root="./datasets/_group_tmp",
    batch_size=8,
    thresh=0.50,
    uncert_band=0.10,
    out_dir="./work",
    per_image_csv=None,
    per_sample_csv=None,
    gt_json=None,
    preview_n=20
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


def parse_input_dirs(args) -> list[Path]:
    """支持 --input_dirs 多个目录，或逗号分隔；没传则退回 --input_dir。"""
    dirs = []
    if getattr(args, "input_dirs", None):
        for token in args.input_dirs:
            dirs.extend([s for s in token.split(",") if s])
    else:
        dirs = [args.input_dir]
    paths = [Path(d).resolve() for d in dirs]
    for p in paths:
        if not p.exists():
            raise FileNotFoundError(f"输入目录不存在: {p}")
    return paths


def gather_groups(input_dirs: list[Path]):
    """
    收集图片并按 {prefix}_{sample_id} 分组。
    返回: dict[group_key] = [item,...]，
    其中 item = {path, prefix, filename, dst_name, sample_key}
    """
    groups = defaultdict(list)
    for d in input_dirs:
        prefix = d.name  # 最后一级目录名作为前缀
        for p in list_images(d):
            filename = p.name
            sid = sample_key_from_name(filename)
            sample_key = f"{prefix}_{sid}"  # 显示与聚合用
            dst_name = f"{prefix}__{filename}"  # 临时目录里避免同名冲突
            groups[sample_key].append({
                "path": p, "prefix": prefix, "filename": filename,
                "dst_name": dst_name, "sample_key": sample_key
            })
    # 组内按文件名排序
    for k in list(groups.keys()):
        groups[k] = sorted(groups[k], key=lambda it: it["filename"])
    return groups


def prepare_tmp_group(items, tmp_root: Path):
    """把一组图复制到 real/gen，交给 AVLip 解析（使用带前缀的 dst_name 防止重名）。"""
    if tmp_root.exists():
        shutil.rmtree(tmp_root, ignore_errors=True)
    (tmp_root / "real").mkdir(parents=True, exist_ok=True)
    (tmp_root / "gen").mkdir(parents=True, exist_ok=True)
    # 为了和数据集读取顺序对齐，按 dst_name 排序后复制
    items_sorted = sorted(items, key=lambda it: it["dst_name"])
    for it in items_sorted:
        shutil.copyfile(it["path"], tmp_root / "real" / it["dst_name"])
        shutil.copyfile(it["path"], tmp_root / "gen" / it["dst_name"])
    return items_sorted, str(tmp_root / "real"), str(tmp_root / "gen")


@torch.no_grad()
def infer_group(model, device, items, tmp_root: Path, batch_size=8):
    """
    对一组图做一次批量推理。
    返回: (items_sorted, scores) 保证 scores 与 items_sorted 一一对应。
    """
    items_sorted, real_dir, fake_dir = prepare_tmp_group(items, tmp_root)

    class Opt:
        ...

    opt = Opt()
    opt.real_list_path = real_dir
    opt.fake_list_path = fake_dir
    opt.max_sample = len(items_sorted)
    opt.batch_size = 1
    opt.data_label = "val"

    dataset = AVLip(opt)
    if len(dataset) == 0:
        raise RuntimeError("AVLip 数据集为空：请确认这些图是官方 preprocess 生成的‘合成图’。")
    loader = DataLoader(dataset, batch_size=min(batch_size, len(items_sorted)),
                        shuffle=False, num_workers=0)

    scores = []
    for img, crops, _ in loader:
        img = img.to(device)
        crops = [[t.to(device) for t in sub] for sub in crops]
        feats = model.get_features(img).to(device)
        prob = torch.sigmoid(model(crops, feats)[0]).flatten()  # [B]
        scores.extend(prob.cpu().numpy().tolist())
    return items_sorted, scores


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
    # 新增：多目录
    p.add_argument("--input_dirs", type=str, nargs="*", default=None,
                   help="可传多个目录或逗号分隔。若未提供，则使用 --input_dir。")
    # 兼容：单目录
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
                   help="可选：JSON 文件路径，内容为 {'sample_id': 0/1, ...}；sample_id 需使用前缀格式（如 mix_0）")
    p.add_argument("--preview_n", type=int, default=DEFAULTS["preview_n"])
    return p.parse_args()


def main():
    args = parse_args()

    input_dirs = parse_input_dirs(args)
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

    # 读取可选的 GT JSON（注意：键应为前缀后的 sample_id，如 'mix_0'）
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

    # 收集全部图片并分组（带目录前缀）
    groups = gather_groups(input_dirs)
    if not groups:
        raise RuntimeError(f"未在以下目录中找到合成图：{', '.join(str(d) for d in input_dirs)}")

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
    for sid, items in sorted(groups.items(), key=lambda kv: kv[0]):
        items_sorted, scores = infer_group(model, device, items, TMP_ROOT, BATCH_SIZE)
        preds = [1 if s >= THRESH else 0 for s in scores]
        gt = GT_LABELS.get(sid, None)
        print(f"[inferred] sample_id={sid} | num_images={len(items_sorted)} | gt={gt if gt is not None else 'N/A'}")
        # —— 逐图结果
        for it, s, y in zip(items_sorted, scores, preds):
            pred_name = "fake" if y == 1 else "real"
            total_images += 1
            image_pred_counter[pred_name] += 1
            img_disp = f"{it['prefix']}/{it['filename']}"  # 显示来源目录
            if gt is None:
                per_img_rows.append((sid, img_disp, f"{s:.6f}", y, pred_name, "", ""))
            else:
                ok = int(y == int(gt))
                per_img_rows.append((sid, img_disp, f"{s:.6f}", y, pred_name, int(gt), ok))

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
            per_smp_rows.append((sid, len(items_sorted), f"{mean_s:.6f}", f"{median_s:.6f}",
                                 f"{vote_fake:.1f}", verdict, "", "", ""))
        else:
            per_img_correct = [int(y == int(gt)) for y in preds]
            smp_acc = 100.0 * (sum(per_img_correct) / len(per_img_correct))
            sample_correct = int((1 if verdict == "fake" else 0) == int(gt)) if verdict in ("fake", "real") else ""
            per_smp_rows.append((sid, len(items_sorted), f"{mean_s:.6f}", f"{median_s:.6f}",
                                 f"{vote_fake:.1f}", verdict, int(gt), f"{smp_acc:.1f}", sample_correct))

    # —— 导出 CSV
    with open(OUT_PER_IMAGE_CSV, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(per_img_rows)
    with open(OUT_PER_SAMPLE_CSV, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(per_smp_rows)

    # —— 终端打印（简表 + 总结）
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
