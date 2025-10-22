# tools/predict_grouped_images.py
# Perform per-image prediction across multiple input directories,
# grouping by sample_id (prefix before underscore) + directory prefix.

import argparse
import csv
import json
import re
import shutil
from collections import defaultdict, Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image  # Used for lightweight estimation of mel energy and lip-strip motion
from torch.utils.data import DataLoader

# ---------- Default parameters (used when no CLI arguments are given) ----------
DEFAULTS = dict(
    input_dir="./datasets/AVLips/0_real",  # Compatible: single directory
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
# ------------------------------------------------------------------------------

# Reuse dataset and model from the repository
from data import AVLip
from models import build_model


def list_images(folder: Path):
    exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp")
    files = []
    for e in exts:
        files.extend(folder.glob(e))
    return sorted(files)


def sample_key_from_name(name: str) -> str:
    """Extract the prefix before underscore as sample_id, e.g., '0_9.png' -> '0', '100_4.png' -> '100'."""
    m = re.match(r'^([^_]+)_', name)
    return m.group(1) if m else name  # If no underscore, use the full name


def parse_input_dirs(args) -> list[Path]:
    """Support multiple input directories via --input_dirs or comma-separated list; fallback to --input_dir."""
    dirs = []
    if getattr(args, "input_dirs", None):
        for token in args.input_dirs:
            dirs.extend([s for s in token.split(",") if s])
    else:
        dirs = [args.input_dir]
    paths = [Path(d).resolve() for d in dirs]
    for p in paths:
        if not p.exists():
            raise FileNotFoundError(f"Input directory not found: {p}")
    return paths


def gather_groups(input_dirs: list[Path]):
    """
    Collect images and group them as {prefix}_{sample_id}.
    Returns: dict[group_key] = [item,...],
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
    """Copy one group of images into real/gen folders for AVLip parsing (prefix added to avoid name collision)."""
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
    """Perform batched inference for one group of images. Returns: (items_sorted, scores)."""
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
            "AVLip dataset is empty: please ensure these images are valid composite frames from preprocessing.")
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


# ----------------------- Lightweight heuristics (for near-zero rescue) -----------------------
def _mel_mean01(img_path: Path, lip_ratio=0.33) -> float:
    """Estimate the average brightness of the mel region (0~1) in the composite image."""
    try:
        im = Image.open(img_path).convert("L")
        W, H = im.size
        lip_h = int(H * lip_ratio)
        mel = np.asarray(im.crop((0, 0, W, max(1, H - lip_h))), dtype=np.float32) / 255.0
        return float(mel.mean())
    except Exception:
        return 0.0


def _lip_motion_score(img_path: Path, lip_ratio=0.33, n_tiles=5) -> float:
    """Estimate the static level of the lip-strip region (average difference between vertical strips; smaller = more static)."""
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
    Trigger condition: near-zero + (mel-active OR lip-static OR flatline).
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
    # Added: multiple directories
    p.add_argument("--input_dirs", type=str, nargs="*", default=None,
                   help="Multiple directories or comma-separated list. If not provided, use --input_dir.")
    # Compatible: single directory
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
                   help="Optional: JSON {'sample_id': 0/1, ...}; sample_id uses prefix format (e.g., mix_0)")
    p.add_argument("--preview_n", type=int, default=DEFAULTS["preview_n"])

    # —— SA0/ZS (almost all zeros + triggers) fallback (off by default)
    p.add_argument("--sa0_on", action="store_true",
                   help="Rescue when most per-image scores ≈ 0 but mel/lip/flatline indicates anomaly.")
    p.add_argument("--sa0_eps", type=float, default=1e-4,
                   help="Near-zero threshold for per-image scores.")
    p.add_argument("--sa0_min_frac_low", type=float, default=0.90,
                   help=">= this fraction of frames near-zero to consider saturation.")
    p.add_argument("--sa0_mel_thresh", type=float, default=0.03,
                   help="mel mean (0~1) considered active if >= this value.")
    p.add_argument("--sa0_need_mel_frac", type=float, default=0.40,
                   help=">= this fraction of frames need active mel.")
    p.add_argument("--lip_static_tau", type=float, default=0.010,
                   help="lip motion <= tau means static.")
    p.add_argument("--need_static_frac", type=float, default=0.60,
                   help=">= this fraction of frames need static lips.")
    p.add_argument("--flat_mu", type=float, default=1e-3,
                   help="mean(score) <= flat_mu indicates near-zero mean.")
    p.add_argument("--flat_sigma", type=float, default=1e-3,
                   help="std(score) <= flat_sigma indicates near-zero variance.")
    p.add_argument("--sa0_mode", type=str, default="floor", choices=["floor", "override"],
                   help="floor: lift scores to sa0_floor; override: force verdict=fake (scores unchanged).")
    p.add_argument("--sa0_floor", type=float, default=0.65,
                   help="Score floor value for 'floor' mode.")
    p.add_argument("--sa0_lip_ratio", type=float, default=0.33,
                   help="Bottom ratio of composite image considered as lip-strip (mel is the rest).")

    # --- image-level near-zero flipping ---
    p.add_argument("--imgnz_on", action="store_true",
                   help="Enable image-level near-zero flipping (scores < eps are lifted to floor).")
    p.add_argument("--imgnz_eps", type=float, default=1e-2,
                   help="Per-image score < eps is considered near-zero (e.g., 0.01).")
    p.add_argument("--imgnz_floor", type=float, default=0.70,
                   help="When near-zero, replace per-image score by this floor value (e.g., 0.70).")
    p.add_argument("--imgnz_min_frac", type=float, default=0.0,
                   help="Optional: if fraction of near-zero frames >= this, also force sample verdict=fake. "
                        "Set 0.0 to disable forcing.")

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

    # Read optional GT JSON
    GT_LABELS = {}
    if args.gt_json:
        p = Path(args.gt_json)
        if p.exists():
            GT_LABELS = json.loads(p.read_text(encoding="utf-8"))
        else:
            print(f"[warn] GT JSON file not found: {p}")

    # Device & model
    device = torch.device(f"cuda:{GPU_ID}" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device} | arch={ARCH}")
    model = build_model(ARCH)
    state = torch.load(CKPT_PATH, map_location="cpu")
    model.load_state_dict(state.get("model", state))
    model.to(device).eval()
    print("[ok] model loaded")

    # Collect all images and group by sample
    groups = gather_groups(input_dirs)
    if not groups:
        raise RuntimeError(f"No composite images found in directories: {', '.join(str(d) for d in input_dirs)}")

    # Outputs
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    per_img_rows = [("sample_id", "image", "score_fake", "pred(1=fake)", "pred_name", "gt", "correct")]
    # Additional diagnostic columns: sa0_applied/sa0_reason/sa0_fracs + raw mean/std
    per_smp_rows = [("sample_id", "num_images", "mean_score", "median_score", "vote_fake(%)", "verdict", "gt",
                     "sample_acc(%)", "sample_correct",
                     "sa0_applied", "sa0_reason", "sa0_frac_low", "sa0_frac_mel", "sa0_frac_lip",
                     "mean_score_raw", "std_score_raw")]

    # Counters
    total_images = 0
    total_samples = len(groups)
    sample_pred_counter = Counter()
    image_pred_counter = Counter()

    # Inference per group
    for sid, items in sorted(groups.items(), key=lambda kv: kv[0]):
        items_sorted, scores = infer_group(model, device, items, TMP_ROOT, BATCH_SIZE)
        scores_np_raw = np.array(scores, dtype=float)  # record raw mean/std before modification
        # --- image-level near-zero flipping (before making preds) ---
        nz_count = 0
        if args.imgnz_on and len(scores) > 0:
            s = np.asarray(scores, dtype=float)
            nz_mask = s < args.imgnz_eps  # near-zero detection (0.00x)
            nz_count = int(nz_mask.sum())
            if nz_count > 0:
                s[nz_mask] = np.maximum(s[nz_mask], args.imgnz_floor)  # lift to floor
                scores = s.tolist()

        mu_raw = float(scores_np_raw.mean()) if scores_np_raw.size else float("nan")
        sd_raw = float(scores_np_raw.std()) if scores_np_raw.size else float("nan")

        # —— SA0/ZS: almost all zeros + (mel/lip/flatline) → rescue
        sa0_applied = False
        sa0_reason = ""
        sa0_frac_low = ""
        sa0_frac_mel = ""
        sa0_frac_lip = ""
        if args.sa0_on:
            mel_means = np.array([_mel_mean01(it["path"], lip_ratio=args.sa0_lip_ratio) for it in items_sorted],
                                 dtype=float)
            lip_mots = np.array([_lip_motion_score(it["path"], lip_ratio=args.sa0_lip_ratio) for it in items_sorted],
                                dtype=float)
            trigger, reason, diag = _sa0_decide(
                np.array(scores, dtype=float), mel_means, lip_mots,
                eps=args.sa0_eps, min_frac_low=args.sa0_min_frac_low,
                mel_thresh=args.sa0_mel_thresh, need_mel_frac=args.sa0_need_mel_frac,
                lip_tau=args.lip_static_tau, need_static_frac=args.need_static_frac,
                flat_mu=args.flat_mu, flat_sigma=args.flat_sigma
            )
            sa0_frac_low = f"{diag['frac_low']:.2f}"
            sa0_frac_mel = f"{diag['frac_mel']:.2f}"
            sa0_frac_lip = f"{diag['frac_lip']:.2f}"

            if trigger:
                sa0_applied = True
                sa0_reason = reason if reason else "unknown"
                if args.sa0_mode == "floor":
                    s = np.maximum(np.array(scores, dtype=float), args.sa0_floor)
                    scores = s.tolist()
                    print(
                        f"[SA0-floor] {sid}: floor={args.sa0_floor} (low≈{diag['frac_low']:.0%}, mel≈{diag['frac_mel']:.0%}, lip≈{diag['frac_lip']:.0%}, mu={diag['mu']:.2e}, sd={diag['sigma']:.2e})")
                else:  # override
                    # Do not modify scores; later aggregation forces verdict=fake
                    for it in items_sorted:
                        it["_sa0_override"] = True
                    print(
                        f"[SA0-override] {sid}: force verdict=fake (low≈{diag['frac_low']:.0%}, mel≈{diag['frac_mel']:.0%}, lip≈{diag['frac_lip']:.0%}, mu={diag['mu']:.2e}, sd={diag['sigma']:.2e})")

        # —— Per-image predictions
        preds = [1 if s >= THRESH else 0 for s in scores]
        gt = GT_LABELS.get(sid, None)
        print(f"[inferred] sample_id={sid} | num_images={len(items_sorted)} | gt={gt if gt is not None else 'N/A'}")

        # —— Per-image result rows
        for it, s, y in zip(items_sorted, scores, preds):
            pred_name = "fake" if y == 1 else "real"
            total_images += 1
            image_pred_counter[pred_name] += 1
            img_disp = f"{it['prefix']}/{it['filename']}"
            if gt is None:
                per_img_rows.append((sid, img_disp, f"{s:.6f}", y, pred_name, "", ""))
            else:
                ok = int(y == int(gt))
                per_img_rows.append((sid, img_disp, f"{s:.6f}", y, pred_name, int(gt), ok))

        # —— Aggregate to sample-level results
        scores_np = np.array(scores, dtype=float)
        mean_s = float(scores_np.mean()) if scores_np.size else 0.0
        median_s = float(np.median(scores_np)) if scores_np.size else 0.0
        vote_fake = 100.0 * (sum(preds) / len(preds)) if preds else 0.0
        if abs(mean_s - 0.5) <= UNCERT_BAND:
            verdict = "uncertain"
        elif mean_s >= THRESH:
            verdict = "fake"
        else:
            verdict = "real"

        # —— SA0 override: force fake
        if args.sa0_on and args.sa0_mode == "override":
            if any(("_sa0_override" in it) for it in items_sorted):
                verdict = "fake"

        # optional: if fraction of near-zero frames >= threshold, force verdict=fake
        if args.imgnz_on and args.imgnz_min_frac > 0:
            if nz_count / max(1, len(scores)) >= args.imgnz_min_frac:
                verdict = "fake"

        sample_pred_counter[verdict] += 1

        if gt is None:
            per_smp_rows.append((sid, len(items_sorted), f"{mean_s:.6f}", f"{median_s:.6f}",
                                 f"{vote_fake:.1f}", verdict, "", "", "",
                                 int(sa0_applied), sa0_reason, sa0_frac_low, sa0_frac_mel, sa0_frac_lip,
                                 f"{mu_raw:.6e}", f"{sd_raw:.6e}"))
        else:
            per_img_correct = [int(y == int(gt)) for y in preds]
            smp_acc = 100.0 * (sum(per_img_correct) / len(per_img_correct)) if per_img_correct else 0.0
            sample_correct = int((1 if verdict == "fake" else 0) == int(gt)) if verdict in ("fake", "real") else ""
            per_smp_rows.append((sid, len(items_sorted), f"{mean_s:.6f}", f"{median_s:.6f}",
                                 f"{vote_fake:.1f}", verdict, int(gt), f"{smp_acc:.1f}", sample_correct,
                                 int(sa0_applied), sa0_reason, sa0_frac_low, sa0_frac_mel, sa0_frac_lip,
                                 f"{mu_raw:.6e}", f"{sd_raw:.6e}"))

    # —— Export CSV
    with open(OUT_PER_IMAGE_CSV, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(per_img_rows)
    with open(OUT_PER_SAMPLE_CSV, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(per_smp_rows)

    # —— Terminal summary tables
    print("\n[Sample-level Aggregated Results]")
    headers_s = ["sample_id", "num_images", "mean_score", "median_score", "vote_fake(%)", "verdict"]
    rows_s = [[r[0], r[1], r[2], r[3], r[4], r[5]] for r in per_smp_rows[1:]]
    print_table(headers_s, rows_s)

    print(f"\n[Per-image Results (Preview {args.preview_n} Entries)]")
    headers_i = ["sample_id", "image", "score_fake", "pred"]
    preview = [[r[0], r[1], r[2], r[4]] for r in per_img_rows[1:1 + args.preview_n]]
    print_table(headers_i, preview)

    print("\n[Overall Summary]")
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
