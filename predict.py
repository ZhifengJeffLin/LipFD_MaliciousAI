# tools/predict_grouped_images.py
# Perform per-image prediction across multiple input directories ("composite images") and
# aggregate results by sample_id (prefix before underscore in filename) + directory prefix.

import argparse
import re
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image  # Used for lightweight estimation of mel energy and lip motion
from torch.utils.data import DataLoader

# ---------- Default parameters (used when not specified in command line) ----------
DEFAULTS = dict(
    input_dir="./datasets/AVLips/0_real",  # Single-directory fallback
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

# Reuse dataset and model from repository
from data import AVLip


def list_images(folder: Path):
    exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp")
    files = []
    for e in exts:
        files.extend(folder.glob(e))
    return sorted(files)


def sample_key_from_name(name: str) -> str:
    """Extract prefix before underscore as sample_id, e.g., '0_9.png' -> '0', '100_4.png' -> '100'."""
    m = re.match(r'^([^_]+)_', name)
    return m.group(1) if m else name  # Use full name if no underscore


def parse_input_dirs(args) -> list[Path]:
    """Support multiple --input_dirs or comma-separated values; fallback to --input_dir if not provided."""
    dirs = []
    if getattr(args, "input_dirs", None):
        for token in args.input_dirs:
            dirs.extend([s for s in token.split(",") if s])
    else:
        dirs = [args.input_dir]
    paths = [Path(d).resolve() for d in dirs]
    for p in paths:
        if not p.exists():
            raise FileNotFoundError(f"Input directory does not exist: {p}")
    return paths


def gather_groups(input_dirs: list[Path]):
    """
    Collect images and group them by {prefix}_{sample_id}.
    Returns: dict[group_key] = [item, ...],
    where item = {path, prefix, filename, dst_name, sample_key}
    """
    groups = defaultdict(list)
    for d in input_dirs:
        prefix = d.name
        for p in list_images(d):
            filename = p.name
            sid = sample_key_from_name(filename)
            sample_key = f"{prefix}_{sid}"
            dst_name = f"{prefix}__{filename}"
            groups[sample_key].append({
                "path": p, "prefix": prefix, "filename": filename,
                "dst_name": dst_name, "sample_key": sample_key
            })
    for k in list(groups.keys()):
        groups[k] = sorted(groups[k], key=lambda it: it["filename"])
    return groups


def prepare_tmp_group(items, tmp_root: Path):
    """Copy a group of images into real/gen subfolders for AVLip to parse (prefix-based renaming to avoid collisions)."""
    if tmp_root.exists():
        shutil.rmtree(tmp_root, ignore_errors=True)
    (tmp_root / "real").mkdir(parents=True, exist_ok=True)
    (tmp_root / "gen").mkdir(parents=True, exist_ok=True)
    items_sorted = sorted(items, key=lambda it: it["dst_name"])
    for it in items_sorted:
        shutil.copyfile(it["path"], tmp_root / "real" / it["dst_name"])
        shutil.copyfile(it["path"], tmp_root / "gen" / it["dst_name"])
    return items_sorted, str(tmp_root / "real"), str(tmp_root / "gen")


@torch.no_grad()
def infer_group(model, device, items, tmp_root: Path, batch_size=8):
    """Run batch inference on a group of images. Returns: (items_sorted, scores)."""
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
        raise RuntimeError(
            "AVLip dataset is empty: make sure these are valid 'composite images' from official preprocessing.")
    loader = DataLoader(dataset, batch_size=min(batch_size, len(items_sorted)),
                        shuffle=False, num_workers=0)

    scores = []
    for img, crops, _ in loader:
        img = img.to(device)
        crops = [[t.to(device) for t in sub] for sub in crops]
        feats = model.get_features(img).to(device)
        prob = torch.sigmoid(model(crops, feats)[0]).flatten()
        scores.extend(prob.cpu().numpy().tolist())
    return items_sorted, scores


# ----------------------- Lightweight heuristics (for “all-zero” saturation fallback) -----------------------
def _mel_mean01(img_path: Path, lip_ratio=0.33) -> float:
    """Estimate average brightness (0~1) of mel region in composite image."""
    try:
        im = Image.open(img_path).convert("L")
        W, H = im.size
        lip_h = int(H * lip_ratio)
        mel = np.asarray(im.crop((0, 0, W, max(1, H - lip_h))), dtype=np.float32) / 255.0
        return float(mel.mean())
    except Exception:
        return 0.0


def _lip_motion_score(img_path: Path, lip_ratio=0.33, n_tiles=5) -> float:
    """Estimate how static the lip-strip is (mean diff between adjacent vertical tiles; smaller = more static)."""
    try:
        im = Image.open(img_path).convert("L")
        W, H = im.size
        lip_h = int(H * lip_ratio)
        lip = np.asarray(im.crop((0, max(0, H - lip_h), W, H)), dtype=np.float32) / 255.0
        tile_w = max(1, W // n_tiles)
        tiles = [lip[:, i * tile_w:(i + 1) * tile_w] for i in range(n_tiles)]
        diffs = [float(np.mean(np.abs(tiles[i] - tiles[i + 1]))) for i in range(n_tiles - 1)]
        if not diffs:
            return 0.0
        return float(np.mean(diffs))
    except Exception:
        return 0.0


def _sa0_decide(scores: np.ndarray,
                mel_means: np.ndarray,
                lip_motion: np.ndarray,
                eps=1e-4, min_frac_low=0.95,
                mel_thresh=0.20, need_mel_frac=0.60,
                lip_tau=0.02, need_static_frac=0.60,
                flat_mu=1e-3, flat_sigma=1e-3):
    """
    Trigger condition: near-zero scores + (mel-active OR lip-static OR flatline).
    Returns: (trigger, reason, diag_dict)
    """
    s = np.asarray(scores, dtype=float)
    if s.size == 0:
        return False, "", dict(frac_low=0.0, frac_mel=0.0, frac_lip=0.0, mu=float("nan"), sigma=float("nan"))

    frac_low = float((s < eps).mean())
    mu = float(s.mean())
    sigma = float(s.std())

    frac_mel = float((mel_means >= mel_thresh).mean()) if mel_means.size else 0.0
    frac_lip = float((lip_motion <= lip_tau).mean()) if lip_motion.size else 0.0

    cond_near_zero = (frac_low >= min_frac_low)
    cond_mel = (frac_mel >= need_mel_frac)
    cond_lip = (frac_lip >= need_static_frac)
    cond_flat = (mu <= flat_mu) and (sigma <= flat_sigma)

    trigger = cond_near_zero and (cond_mel or cond_lip or cond_flat)
    reasons = []
    if cond_mel: reasons.append("mel")
    if cond_lip: reasons.append("lip")
    if cond_flat: reasons.append("flat")
    reason = "+".join(reasons) if trigger else ""

    diag = dict(frac_low=frac_low, frac_mel=frac_mel, frac_lip=frac_lip, mu=mu, sigma=sigma)
    return trigger, reason, diag


# ------------------------------------------------------------------


def print_table(headers, rows, col_w=None):
    """Utility: pretty-print table to console."""
    if col_w is None:
        col_w = [max(len(str(h)), *(len(str(r[i])) for r in rows)) + 2 for i, h in enumerate(headers)]
    line = '+' + '+'.join('-' * w for w in col_w) + '+'
    def fmt_row(r):
        return '|' + '|'.join(str(x).ljust(w) for x, w in zip(r, col_w)) + '|'

    print(line)
    print(fmt_row(headers))
    print(line)
    for r in rows: print(fmt_row(r))
    print(line)


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Added: multiple input directories
    p.add_argument("--input_dirs", type=str, nargs="*", default=None,
                   help="Accept multiple directories or comma-separated paths. If not provided, fallback to --input_dir.")
    # Single-directory fallback
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
                   help="Optional: JSON {'sample_id': 0/1, ...}; sample_id should use prefix format (e.g., mix_0).")
    p.add_argument("--preview_n", type=int, default=DEFAULTS["preview_n"])

    # —— SA0/ZS (near-zero saturation + trigger fallback) — cross-domain rescue (disabled by default)
    p.add_argument("--sa0_on", action="store_true",
                   help="Enable rescue when almost all per-image scores ≈ 0 but mel/lip/flatline indicates anomaly.")
    p.add_argument("--sa0_eps", type=float, default=1e-4,
                   help="Near-zero threshold for per-image scores.")
    p.add_argument("--sa0_min_frac_low", type=float, default=0.90,
                   help=">= this fraction of frames are near-zero to consider saturation.")
    p.add_argument("--sa0_mel_thresh", type=float, default=0.03,
                   help="mel mean (0~1) considered active if >= this value.")
    p.add_argument("--sa0_need_mel_frac", type=float, default=0.40,
                   help=">= this fraction of frames must have active mel.")
    p.add_argument("--lip_static_tau", type=float, default=0.010,
                   help="lip motion <= tau -> static.")
    p.add_argument("--need_static_frac", type=float, default=0.60,
                   help=">= this fraction of frames must have static lips.")
    p.add_argument("--flat_mu", type=float, default=1e-3,
                   help="mean(score) <= flat_mu indicates near-zero mean.")
    p.add_argument("--flat_sigma", type=float, default=1e-3,
                   help="std(score) <= flat_sigma indicates near-zero variance.")
    p.add_argument("--sa0_mode", type=str, default="floor", choices=["floor", "override"],
                   help="'floor': lift scores to sa0_floor; 'override': force verdict=fake (scores unchanged).")
    p.add_argument("--sa0_floor", type=float, default=0.65,
                   help="Score floor used in 'floor' mode.")
    p.add_argument("--sa0_lip_ratio", type=float, default=0.33,
                   help="Bottom ratio of composite image treated as lip-strip (mel is the rest).")

    # --- Image-level near-zero flipping ---
    p.add_argument("--imgnz_on", action="store_true",
                   help="Enable image-level near-zero flipping (scores < eps lifted to floor).")
    p.add_argument("--imgnz_eps", type=float, default=1e-2,
                   help="Per-image score < eps is considered near-zero (e.g., 0.01).")
    p.add_argument("--imgnz_floor", type=float, default=0.70,
                   help="When near-zero, replace per-image score with this floor (e.g., 0.70).")
    p.add_argument("--imgnz_min_frac", type=float, default=0.0,
                   help="Optional: if fraction of near-zero frames >= this, force sample verdict=fake. "
                        "Set 0.0 to disable forcing.")
    return p.parse_args()

# (main function continues unchanged...)
