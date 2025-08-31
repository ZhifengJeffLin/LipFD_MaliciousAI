import cv2
import librosa
import matplotlib.pyplot as plt
import numpy as np
from librosa import feature as audio
from tqdm import tqdm

"""
Structure of the AVLips dataset:
AVLips
├── 0_real
├── 1_fake
└── wav
    ├── 0_real
    └── 1_fake
"""

############ Custom parameter ##############
# =================== Config: paths + custom params ===================
from pathlib import Path
import argparse, os


def _resolve_config():
    # 只解析我们关心的参数，不干扰你后面自己的 argparse
    p = argparse.ArgumentParser(add_help=False)

    # 路径相关
    p.add_argument('--root', type=str, default=None,
                   help='工程根目录；不传则使用脚本所在目录')
    p.add_argument('--audio_root', type=str, default=None,
                   help='音频目录，默认 <root>/AVLips/wav')
    p.add_argument('--video_root', type=str, default=None,
                   help='视频/图像目录，默认 <root>/AVLips')
    p.add_argument('--output_root', type=str, default=None,
                   help='输出目录，默认 <root>/datasets/AVLips')

    # 你的自定义参数（提供默认值，并允许覆盖）
    p.add_argument('--n_extract', type=int, default=10, help='number of extracted images from video')
    p.add_argument('--window_len', type=int, default=5, help='frames of each window')
    p.add_argument('--max_sample', type=int, default=100, help='max samples to process')

    args, _ = p.parse_known_args()

    # 基准根目录（默认=脚本目录）
    script_dir = Path(__file__).resolve().parent
    root = Path(args.root).resolve() if args.root else script_dir

    # 具体路径（若未传参则使用默认）
    audio = Path(args.audio_root).resolve() if args.audio_root else (root / 'AVLips' / 'wav')
    video = Path(args.video_root).resolve() if args.video_root else (root / 'AVLips')
    output = Path(args.output_root).resolve() if args.output_root else (root / 'datasets' / 'AVLips')

    # 创建输出目录
    output.mkdir(parents=True, exist_ok=True)

    # 一点健壮性检查
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
        'SCRIPT_DIR': script_dir
    }


_cfg = _resolve_config()

# 推荐使用的大写常量
AUDIO_ROOT = _cfg['AUDIO_ROOT']
VIDEO_ROOT = _cfg['VIDEO_ROOT']
OUTPUT_ROOT = _cfg['OUTPUT_ROOT']
N_EXTRACT = _cfg['N_EXTRACT']
WINDOW_LEN = _cfg['WINDOW_LEN']
MAX_SAMPLE = _cfg['MAX_SAMPLE']
SCRIPT_DIR = _cfg['SCRIPT_DIR']

# 兼容你原来的小写变量名（如果后面代码还在用它们）
audio_root = AUDIO_ROOT
video_root = VIDEO_ROOT
output_root = OUTPUT_ROOT

print(f"[paths] script_dir = {SCRIPT_DIR}")
print(f"[paths] audio_root = {AUDIO_ROOT}")
print(f"[paths] video_root = {VIDEO_ROOT}")
print(f"[paths] output_root= {OUTPUT_ROOT}")
print(f"[params] N_EXTRACT={N_EXTRACT}  WINDOW_LEN={WINDOW_LEN}  MAX_SAMPLE={MAX_SAMPLE}")
# ====================================================================

############################################

labels = [(0, "0_real"), (1, "1_fake")]

def get_spectrogram(audio_file):
    data, sr = librosa.load(audio_file)
    mel = librosa.power_to_db(audio.melspectrogram(y=data, sr=sr), ref=np.min)
    plt.imsave("./temp/mel.png", mel)


def run():
    i = 0
    for label, dataset_name in labels:
        if not os.path.exists(dataset_name):
            os.makedirs(f"{output_root}/{dataset_name}", exist_ok=True)

        if i == MAX_SAMPLE:
            break
        root = f"{video_root}/{dataset_name}"
        video_list = os.listdir(root)
        print(f"Handling {dataset_name}...")
        for j in tqdm(range(len(video_list))):
            v = video_list[j]
            # load video
            video_capture = cv2.VideoCapture(f"{root}/{v}")
            fps = video_capture.get(cv2.CAP_PROP_FPS)
            frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

            # select 10 starting point from frames
            frame_idx = np.linspace(
                0,
                frame_count - WINDOW_LEN - 1,
                N_EXTRACT,
                endpoint=True,
                dtype=np.uint8,
            ).tolist()
            frame_idx.sort()
            # selected frames
            frame_sequence = [
                i for num in frame_idx for i in range(num, num + WINDOW_LEN)
            ]
            frame_list = []
            current_frame = 0
            while current_frame <= frame_sequence[-1]:
                ret, frame = video_capture.read()
                if not ret:
                    print(f"Error in reading frame {v}: {current_frame}")
                    break
                if current_frame in frame_sequence:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
                    frame_list.append(cv2.resize(frame, (500, 500)))  # to floating num
                current_frame += 1
            video_capture.release()

            # load audio
            name = v.split(".")[0]
            a = f"{audio_root}/{dataset_name}/{name}.wav"

            group = 0
            get_spectrogram(a)
            mel = plt.imread("./temp/mel.png") * 255  # load spectrogram (int)
            mel = mel.astype(np.uint8)
            mapping = mel.shape[1] / frame_count
            for i in range(len(frame_list)):
                idx = i % WINDOW_LEN
                if idx == 0:
                    try:
                        begin = np.round(frame_sequence[i] * mapping)
                        end = np.round((frame_sequence[i] + WINDOW_LEN) * mapping)
                        sub_mel = cv2.resize(
                            (mel[:, int(begin) : int(end)]), (500 * WINDOW_LEN, 500)
                        )
                        x = np.concatenate(frame_list[i : i + WINDOW_LEN], axis=1)
                        # print(x.shape)
                        # print(sub_mel.shape)
                        x = np.concatenate((sub_mel[:, :, :3], x[:, :, :3]), axis=0)
                        # print(x.shape)
                        plt.imsave(
                            f"{output_root}/{dataset_name}/{name}_{group}.png", x
                        )
                        # 新增：同时写入 mix/，用类名前缀避免重名
                        cls_prefix = "real" if label == 0 else "fake"
                        mix_path = Path(output_root) / "mix"
                        mix_path.mkdir(parents=True, exist_ok=True)
                        dst = mix_path / f"{cls_prefix}_{name}_{group}.png"
                        k = 1
                        while dst.exists():
                            dst = mix_path / f"{cls_prefix}_{name}_{group}_{k}.png"
                            k += 1
                        plt.imsave(dst.as_posix(), x)

                        group = group + 1
                    except ValueError:
                        print(f"ValueError: {name}")
                        continue
            # print(frame_sequence)
            # print(frame_count)
            # print(mel.shape[1])
            # print(mapping)
            # exit(0)
        i += 1


if __name__ == "__main__":
    if not os.path.exists(output_root):
        os.makedirs(output_root, exist_ok=True)
    if not os.path.exists("./temp"):
        os.makedirs("./temp", exist_ok=True)
    run()
