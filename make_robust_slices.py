#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 用法（本地跑）：python make_robust_slices.py --src_slices ./datasets/val --dst_root ./work/robust_slices --qualities 80 60 40
# 说明：对 ./datasets/val/{0_real,1_fake} 下的 PNG/JPG 切片做 JPEG round-trip（Q=80/60/40），
#       生成 ./work/robust_slices/q80|q60|q40/{0_real,1_fake}/*.png

import argparse
import glob
import os
from pathlib import Path

from PIL import Image


def list_images(d):
    exts = ("*.png", "*.jpg", "*.jpeg", "*.webp")
    out = []
    for e in exts:
        out.extend(glob.glob(os.path.join(d, e)))
    return sorted(out)


def roundtrip_dir(src_dir, dst_dir, quality):
    Path(dst_dir).mkdir(parents=True, exist_ok=True)
    for cls in ["0_real", "1_fake"]:
        in_dir = os.path.join(src_dir, cls)
        out_dir = os.path.join(dst_dir, cls)
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        paths = list_images(in_dir)
        for p in paths:
            fn = os.path.basename(p)
            tmp_jpg = os.path.join(dst_dir, f"__tmp_{fn}.jpg")
            with Image.open(p) as im:
                im.convert("RGB").save(tmp_jpg, "JPEG", quality=quality, optimize=True)
            with Image.open(tmp_jpg) as im2:
                im2.save(os.path.join(out_dir, os.path.splitext(fn)[0] + ".png"), "PNG")
            os.remove(tmp_jpg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_slices", required=True, help="切片根目录，如 ./datasets/val（含 0_real/ 1_fake/）")
    ap.add_argument("--dst_root", required=True, help="输出根目录，如 ./work/robust_slices")
    ap.add_argument("--qualities", nargs="+", type=int, default=[80, 60, 40], help="JPEG 质量")
    args = ap.parse_args()

    assert os.path.isdir(os.path.join(args.src_slices, "0_real")), "missing 0_real"
    assert os.path.isdir(os.path.join(args.src_slices, "1_fake")), "missing 1_fake"

    for q in args.qualities:
        dst = os.path.join(args.dst_root, f"q{q}")
        print(f"[roundtrip] Q={q} -> {dst}")
        roundtrip_dir(args.src_slices, dst, q)


if __name__ == "__main__":
    main()
