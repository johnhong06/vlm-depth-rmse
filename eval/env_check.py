"""H200 환경 점검 — 데이터 팩 없이 돈다. 모델 하나를 Track A 와 같은 코드(공식 경로)로 띄워, 공식 Metric3D 저장소에 들어 있는
KITTI 데모 이미지(hubconf.py 데모의 rgb·LiDAR 깊이·intrinsics)에 돌린다. 숫자는 '모델이 제대로 떴는가' 확인용이고 결과 표에 쓰지 않는다.
  dense 모델: 전체 맵 → LiDAR 가 있는 픽셀의 AbsRel·δ1 / DepthLM: LiDAR 점 8 개 질의 → 원답·z 변환값·GT
사용: python eval/env_check.py --model UniDepthV2-L   |   python eval/env_check.py --model DepthLM-12B --weights <DepthLM 경로>
"""
import argparse
import os

import cv2
import numpy as np
import torch

from common import ray
from dense_sparse import EXT

IMG = os.path.join(EXT, "Metric3D/data/kitti_demo/rgb/0000000050.png")
GT = os.path.join(EXT, "Metric3D/data/kitti_demo/depth/0000000050.png")  # uint16, /256 = m (hubconf.py 데모)
K = [707.0493, 707.0493, 604.0814, 180.5066]                              # hubconf.py 데모 intrinsics

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--weights", default="")
ap.add_argument("--gpu_mem", default="", help="로컬 점검용 CPU 오프로드 (DepthLM)")
a = ap.parse_args()
gt = cv2.imread(GT, -1).astype(np.float32) / 256.0
v, u = np.nonzero(gt)
if a.model == "DepthLM-12B":
    from depthlm_sparse import FOCAL, answer, ask, load, marked, marker_xy, prepare
    proc, model, attn = load(a.weights, a.gpu_mem)
    img, kn = prepare(IMG, K)
    idx = np.random.default_rng(0).choice(len(u), 8, replace=False)
    pts = [(int(u[i]), int(v[i])) for i in idx if marked(img, *marker_xy(u[i], v[i], K, kn)) is not None]
    ims = [marked(img, *marker_xy(x, y, K, kn)) for x, y in pts]
    texts = [t for i in range(0, len(ims), 2) for t in ask(proc, model, ims[i:i + 2])]  # 2 장씩 — 비전 eager attention 이 패치 합의 제곱으로 커진다 (D-8)
    for (x, y), t in zip(pts, texts):
        d = answer(t)
        print(f"  ({x},{y}) 답 {d:.2f} m → z {d / ray(x, y, K):.2f} m | GT z {gt[y, x]:.2f} m | {t.strip()[:60]!r}")
    print(f"[env_check] DepthLM-12B: attention {attn}, 초점 {FOCAL}, 정규화 이미지 {img.size}, 점 {len(pts)} 개 파싱 {sum(np.isfinite(answer(t)) for t in texts)}")
else:
    from dense_sparse import MODELS
    with torch.no_grad():
        p = np.asarray(MODELS[a.model]("Outdoor")(IMG, K), dtype=np.float32)
    p = p if p.shape == gt.shape else cv2.resize(p, gt.shape[::-1], interpolation=cv2.INTER_LINEAR)
    g, q = gt[v, u], p[v, u]
    print(f"[env_check] {a.model}: 예측 {p.shape} 중앙 {np.median(p):.1f} m | LiDAR {g.size} 점 AbsRel {np.mean(np.abs(q - g) / g):.3f} δ1 {np.mean(np.maximum(q / g, g / q) < 1.25):.3f}")
