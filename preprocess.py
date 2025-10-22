import cv2
import librosa
import matplotlib.pyplot as plt
import numpy as np
from librosa import feature as audio
from tqdm import tqdm

"""
Flattened mode:
- video_root: directory that directly contains videos (no longer requires 0_real / 1_fake subfolders)
- audio_root: defaults to <video_root>/../wav/<basename(video_root)>
- output_root: sliced images will be directly saved here
"""

# =================== Config: paths + custom params ===================
from pathlib import Path
import argparse, os, subprocess, shutil, re


def _resolve_config():
    p = argparse.ArgumentParser(add_help=False)

    # Basic paths
    p.add_argument('--root', type=str, default=None,
                   help='Project root directory; defaults to the script directory')
    p.add_argument('--video_root', type=str, default=None,
                   help='Directory containing videos (no need for 0_real/1_fake subfolders), default: <root>/AVLips')
    p.add_argument('--audio_root', type=str, default=None,
                   help='Audio directory; defaults to <video_root>/../wav/<basename(video_root)>')
    p.add_argument('--output_root', type=str, default=None,
                   help='Output directory (sliced images will be saved directly here), default: <root>/datasets/AVLips')

    # Custom parameters
    p.add_argument('--n_extract', type=int, default=10, help='number of extracted images from each video')
    p.add_argument('--window_len', type=int, default=5, help='number of frames in each window')
    p.add_argument('--max_sample', type=int, default=100, help='maximum number of videos to process')

    # Auto-extraction of wav files when missing
    p.add_argument('--auto_extract_wav', action='store_true',
                   help='If a matching wav file is missing, automatically extract audio from video via ffmpeg (mono, 16kHz)')

    args, _ = p.parse_known_args()

    # Base root directory (default = script directory)
    script_dir = Path(__file__).resolve().parent
    root = Path(args.root).resolve() if args.root else script_dir

    # Paths (use defaults if not provided)
    video = Path(args.video_root).resolve() if args.video_root else (root / 'AVLips')
    # audio_root follows video_root by default
    if args.audio_root:
        audio = Path(args.audio_root).resolve()
    else:
        audio = (video.parent / 'wav' / video.name).resolve()

    output = Path(args.output_root).resolve() if args.output_root else (root / 'datasets' / 'AVLips')

    # Ensure directories exist
    output.mkdir(parents=True, exist_ok=True)
    audio.mkdir(parents=True, exist_ok=True)

    # Validation
    if args.n_extract <= 0:  raise ValueError('--n_extract must be > 0')
    if args.window_len <= 0: raise ValueError('--window_len must be > 0')
    if args.max_sample <= 0: raise ValueError('--max_sample must be > 0')

    return {
        'AUDIO_ROOT': audio,
        'VIDEO_ROOT': video,
        'OUTPUT_ROOT': output,
        'N_EXTRACT': args.n_extract,
        'WINDOW_LEN': args.window_len,
        'MAX_SAMPLE': args.max_sample,
        'SCRIPT_DIR': script_dir,
        'AUTO_EXTRACT_WAV': args.auto_extract_wav,
    }


_cfg = _resolve_config()

# Recommended uppercase constants
AUDIO_ROOT = _cfg['AUDIO_ROOT']
VIDEO_ROOT = _cfg['VIDEO_ROOT']
OUTPUT_ROOT = _cfg['OUTPUT_ROOT']
N_EXTRACT = _cfg['N_EXTRACT']
WINDOW_LEN = _cfg['WINDOW_LEN']
MAX_SAMPLE = _cfg['MAX_SAMPLE']
SCRIPT_DIR = _cfg['SCRIPT_DIR']
AUTO_EXTRACT_WAV = _cfg['AUTO_EXTRACT_WAV']

# Backward compatibility with lowercase variables
audio_root = AUDIO_ROOT
video_root = VIDEO_ROOT
output_root = OUTPUT_ROOT

print(f"[paths] script_dir = {SCRIPT_DIR}")
print(f"[paths] video_root = {VIDEO_ROOT}")
print(f"[paths] audio_root = {AUDIO_ROOT}")
print(f"[paths] output_root= {OUTPUT_ROOT}")
print(
    f"[params] N_EXTRACT={N_EXTRACT}  WINDOW_LEN={WINDOW_LEN}  MAX_SAMPLE={MAX_SAMPLE}  AUTO_EXTRACT_WAV={AUTO_EXTRACT_WAV}")
# ====================================================================


def get_spectrogram(audio_file):
    data, sr = librosa.load(audio_file, sr=None, mono=True)
    mel = librosa.power_to_db(audio.melspectrogram(y=data, sr=sr), ref=np.min)
    plt.imsave("./temp/mel.png", mel)


def _ffmpeg_bin() -> str:
    return os.environ.get("FFMPEG_BIN") or shutil.which("ffmpeg") or "ffmpeg"


def ffmpeg_extract_wav(video_path: Path, wav_path: Path, sr: int = 16000) -> bool:
    """Extract audio from video into wav (mono, 16kHz). Print stderr on failure."""
    wav_path = Path(wav_path)
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [_ffmpeg_bin(), '-y', '-i', str(video_path), '-vn', '-ac', '1', '-ar', str(sr), str(wav_path)]
    try:
        ret = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if ret.returncode != 0 or not wav_path.exists() or wav_path.stat().st_size <= 44:
            print("[ffmpeg stderr]")
            print(ret.stderr.strip()[:2000])  # Truncated error message
            return False
        return True
    except FileNotFoundError:
        print(f"[error] ffmpeg not found. Set FFMPEG_BIN or add ffmpeg to PATH. Tried: {_ffmpeg_bin()}")
        return False
    except Exception as e:
        print(f"[error] ffmpeg exception: {e}")
        return False


def _is_numeric_stem(stem: str) -> bool:
    # Only pure numeric names are treated as “already normalized”
    return bool(re.fullmatch(r'\d+', stem))


def normalize_video_filenames(video_dir: Path, audio_dir: Path) -> None:
    """
    Rename all video files under video_dir to numeric sequences (starting from 0, skipping used numbers).
    - Pure numeric names are kept unchanged ([keep])
    - Non-numeric names are renamed sequentially ([rename] old -> new)
    - If audio_dir contains wav files matching old stems, they will be renamed accordingly (with conflict checks)
    """
    exts = {'.mp4', '.mov', '.mkv', '.avi', '.mpg', '.mpeg', '.m4v', '.webm'}
    files = [p for p in sorted(video_dir.iterdir()) if p.is_file() and p.suffix.lower() in exts]
    if not files:
        print(f"[normalize] no video files in: {video_dir}")
        return

    used = set()
    for p in files:
        if _is_numeric_stem(p.stem):
            try:
                used.add(int(p.stem))
            except Exception:
                pass

    renamed = 0
    kept = 0
    next_id = 0

    print(f"[normalize] start renaming to numeric filenames under: {video_dir}")
    for p in files:
        stem = p.stem
        if _is_numeric_stem(stem):
            print(f"[keep]  {p.name}")
            kept += 1
            continue

        while next_id in used:
            next_id += 1
        new_name = f"{next_id}{p.suffix.lower()}"
        dst = p.with_name(new_name)

        while dst.exists():
            next_id += 1
            while next_id in used:
                next_id += 1
            new_name = f"{next_id}{p.suffix.lower()}"
            dst = p.with_name(new_name)

        p.rename(dst)
        print(f"[rename] {p.name} -> {dst.name}")
        used.add(next_id)
        renamed += 1

        old_wav = audio_dir / f"{stem}.wav"
        new_wav = audio_dir / f"{next_id}.wav"
        if old_wav.exists():
            try:
                if new_wav.exists():
                    print(f"[wav-skip] {old_wav.name} -> {new_wav.name} (target exists)")
                else:
                    old_wav.rename(new_wav)
                    print(f"[wav-rename] {old_wav.name} -> {new_wav.name}")
            except Exception as e:
                print(f"[wav-warn] rename failed: {old_wav.name} -> {new_wav.name} ({e})")

        next_id += 1

    print(f"[normalize] done. kept={kept}, renamed={renamed}, total={len(files)}")


def run():
    # If auto extraction is enabled, clear audio_root before starting to prevent misalignment
    if AUTO_EXTRACT_WAV:
        try:
            if Path(audio_root).resolve() == Path(video_root).resolve():
                raise RuntimeError(f"[abort] audio_root == video_root, refuse to wipe: {audio_root}")
            if str(Path(audio_root).resolve()) in ("\\", "/", ""):
                raise RuntimeError(f"[abort] suspicious audio_root: {audio_root}")
            if Path(audio_root).exists():
                print(f"[clean] clearing audio_root: {Path(audio_root).resolve()}")
                shutil.rmtree(Path(audio_root))
            Path(audio_root).mkdir(parents=True, exist_ok=True)
        except Exception as e:
            print(f"[warn] failed to clean audio_root: {e}")

    # 0) Normalize filenames to numeric IDs
    normalize_video_filenames(Path(video_root), Path(audio_root))

    # 1) Enumerate videos (no subdirectories)
    exts = {'.mp4', '.mov', '.mkv', '.avi', '.mpg', '.mpeg', '.m4v', '.webm'}
    all_videos = [p for p in sorted(Path(video_root).iterdir())
                  if p.is_file() and p.suffix.lower() in exts]

    if not all_videos:
        print(f"[warn] no video files found under: {video_root}")
        return

    all_videos = all_videos[:MAX_SAMPLE]
    print(f"Handling videos in: {video_root} (count={len(all_videos)})")

    for vpath in tqdm(all_videos):
        # === Read video frames ===
        video_capture = cv2.VideoCapture(str(vpath))
        frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

        frame_idx = np.linspace(
            0,
            frame_count - WINDOW_LEN - 1 if frame_count - WINDOW_LEN - 1 > 0 else 0,
            N_EXTRACT,
            endpoint=True,
            dtype=np.uint32,
        ).tolist()
        frame_idx.sort()
        frame_sequence = [ii for num in frame_idx for ii in range(num, num + WINDOW_LEN)]

        frame_list = []
        current_frame = 0
        while frame_sequence and current_frame <= frame_sequence[-1]:
            ret, frame = video_capture.read()
            if not ret:
                print(f"[warn] failed to read frame: {vpath} @ {current_frame}")
                break
            if current_frame in frame_sequence:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
                frame_list.append(cv2.resize(frame, (500, 500)))
            current_frame += 1
        video_capture.release()

        if not frame_list:
            print(f"[warn] no frames extracted: {vpath}")
            continue

        # 2) Audio: default path <video_root>/../wav/<basename(video_root)>/<name>.wav
        name = vpath.stem  # already numeric
        wav_file = Path(audio_root) / f"{name}.wav"

        if not wav_file.exists():
            if AUTO_EXTRACT_WAV:
                ok = ffmpeg_extract_wav(Path(vpath), wav_file, sr=16000)
                if not ok:
                    print(f"[skip] ffmpeg extract failed: {vpath.name} -> {wav_file}")
                    continue
                else:
                    print(f"[ok] extracted wav: {wav_file}")
            else:
                print(f"[skip] missing wav: {wav_file} (use --auto_extract_wav to enable auto-extraction)")
                continue

        # 3) Generate spectrogram + concatenate + save to output_root
        get_spectrogram(str(wav_file))
        mel = plt.imread("./temp/mel.png") * 255
        mel = mel.astype(np.uint8)
        mapping = mel.shape[1] / (frame_count if frame_count > 0 else 1)

        group = 0
        for ii in range(len(frame_list)):
            if ii % WINDOW_LEN == 0:
                try:
                    begin = int(np.round(frame_sequence[ii] * mapping))
                    end = int(np.round((frame_sequence[ii] + WINDOW_LEN) * mapping))
                    sub_mel = cv2.resize(mel[:, begin:end], (500 * WINDOW_LEN, 500))
                    x = np.concatenate(frame_list[ii: ii + WINDOW_LEN], axis=1)
                    x = np.concatenate((sub_mel[:, :, :3], x[:, :, :3]), axis=0)

                    out_png = Path(output_root) / f"{name}_{group}.png"
                    out_png.parent.mkdir(parents=True, exist_ok=True)
                    plt.imsave(out_png.as_posix(), x)
                    group += 1
                except Exception as e:
                    print(f"[warn] failed to save slice: {name}_{group}.png ({e})")
                    continue

    print("[done] preprocess finished.")


if __name__ == "__main__":
    os.makedirs("./temp", exist_ok=True)
    run()
