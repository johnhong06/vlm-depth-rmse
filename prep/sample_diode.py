"""DIODE Outdoor(val 446 장) 공통 점을 DepthVLM-Bench 와 같은 규칙으로 뽑는다 → bench/diode_outdoor_pixel_depth_val.jsonl + intrinsics 사이드카.
규칙: 공식 sample_points.py 의 sample_and_compute 그대로 (이미지 idx 마다 RandomState(seed + idx), valid 픽셀에서 비복원 균일 추출, 소수 넷째 자리),
      --total_points 방식(전체 10,000 점을 이미지에 고르게: 446 장이면 22–23 점), seed 42, 이미지 정렬 순서 고정, 오버샘플링 없음.
valid = depth_mask.npy == 1 그리고 0.05 ≤ depth ≤ 80 m (NOTES D-14). DIODE 깊이는 z-depth (바닥 평면 검증, NOTES F-5).
intrinsics 는 공식 devkit intrinsics.txt: [fx, fy, cx, cy] = [886.81, 927.06, 512, 384] (모든 이미지 공통).
사용: python prep/sample_diode.py ~/data/diode ~/data/depthvlm_bench"""
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "../third_party/DepthVLM/data_process/sample_test_points"))
from sample_points import sample_and_compute  # noqa: E402

SEED, TOTAL, MIN_D, MAX_D = 42, 10000, 0.05, 80.0
K = [886.81, 927.06, 512.0, 384.0]
src, root = map(os.path.expanduser, sys.argv[1:3])
os.makedirs(os.path.join(root, "diode"), exist_ok=True)
link = os.path.join(root, "diode", "val")
if not os.path.exists(link):
    os.symlink(os.path.join(src, "val"), link)
imgs = sorted(glob.glob(os.path.join(src, "val/outdoor/*/*/*.png")))
base, rem = divmod(TOTAL, len(imgs))
out, n = [], 0
for i, p in enumerate(imgs):
    rel = os.path.relpath(p, src)  # val/outdoor/scene_x/scan_y/name.png
    d = np.load(p.replace(".png", "_depth.npy"))[..., 0].astype(np.float32)
    m = np.load(p.replace(".png", "_depth_mask.npy")) > 0
    d[~m | (d < MIN_D) | (d > MAX_D)] = 0.0
    coords, depths = sample_and_compute(d, base + (1 if i < rem else 0), np.random.RandomState(SEED + i))
    h, w = d.shape
    out.append(dict(image=f"diode/{rel}", depth_path=f"diode/{rel.replace('.png', '_depth.npy')}",
                    mask_valid_path=f"diode/{rel.replace('.png', '_depth_mask.npy')}", depth_scale=1.0, depth_format="npy",
                    original_rgb_size=[w, h], original_depth_size=[w, h], original_fx=K[0], min_depth=MIN_D, max_depth=MAX_D,
                    pixel_coords=coords, depth=depths, depth_type="z_depth"))
    n += len(coords)
dst = os.path.join(HERE, "../bench/diode_outdoor_pixel_depth_val.jsonl")
open(dst, "w").write("".join(json.dumps(r) + "\n" for r in out))
json.dump({r["image"]: K for r in out}, open(os.path.join(HERE, "../bench/intrinsics_diode_outdoor.json"), "w"))
print(f"DIODE Outdoor: 이미지 {len(out)}, 점 {n} (이미지당 {base}–{base + 1}), seed {SEED}, cap [{MIN_D}, {MAX_D}] m → {dst}")
