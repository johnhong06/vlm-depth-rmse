"""공통 점마다 경계/내부 표시 → bench/boundary_<ds>.parquet (image_id, u, v, boundary). 보조 집계(eval/breakdown.py)용, NOTES D-18.
경계 = 질의 점이 가림 경계에서 R 픽셀(원본 해상도) 이내.
  iBims-1  : 공식 경계 정답 지도(edges/*.png, Koch et al. 2018). 지도는 100 장 중 86 장에만 있어 나머지 14 장은 경계 분석에서 뺀다(boundary = NA).
             비교용으로 GT 비율 방식 표시(boundary_ratio)도 같이 남긴다.
  NYUv2·DIODE Outdoor : Depth Pro (arXiv:2410.02073) 의 가림 경계 정의 — 이웃(상하좌우) 두 valid GT 픽셀의 깊이 비가 1 + T 를 넘으면 두 픽셀 모두 경계.
  nuScenes·DDAD : GT 가 LiDAR 점이라 경계를 정할 수 없어 만들지 않는다 (Mind the Edge, arXiv:2212.05315 도 같은 이유로 수작업 주석).
사용: python prep/boundary_labels.py ~/data/depthvlm_bench"""
import os
import sys

import cv2
import numpy as np
import pandas as pd
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "../eval"))
from common import ROOT, load_bench  # noqa: E402
from dense_sparse import gt_map  # noqa: E402

R, T = 3, 0.10   # 경계까지 거리(px), 깊이 비 문턱 (Depth Pro 의 5–25 % 범위 안)


def ratio_contour(d):
    """이웃 valid 픽셀의 깊이 비 > 1 + T 인 쌍의 두 픽셀."""
    c = np.zeros(d.shape, bool)
    for a, b, sa, sb in [(d[:, :-1], d[:, 1:], (slice(None), slice(None, -1)), (slice(None), slice(1, None))),
                         (d[:-1], d[1:], (slice(None, -1), slice(None)), (slice(1, None), slice(None)))]:
        ok = (a > 0) & (b > 0)
        jump = ok & (np.maximum(a, b) > (1 + T) * np.minimum(np.where(ok, a, 1), np.where(ok, b, 1)))
        c[sa] |= jump
        c[sb] |= jump
    return c


def near(contour, u, v):
    dist = cv2.distanceTransform((~contour).astype(np.uint8), cv2.DIST_L2, 5)
    return dist[v, u] <= R


root = os.path.expanduser(sys.argv[1])
for ds in ["ibims1", "nyuv2", "diode_outdoor"]:
    recs, _ = load_bench(ds)
    rows = []
    for r in recs:
        u, v = np.array(r["pixel_coords"]).T
        ratio = near(ratio_contour(gt_map(r, root, ds)), u, v)
        if ds == "ibims1":
            ep = os.path.join(root, r["image"].replace("/rgb/", "/edges/"))
            b = near(np.array(Image.open(ep)) > 0, u, v) if os.path.exists(ep) else np.full(len(u), None)
        else:
            b = ratio
        rows += [dict(image_id=r["image"], u=int(a), v=int(c), boundary=None if x is None else bool(x), boundary_ratio=bool(y)) for a, c, x, y in zip(u, v, b, ratio)]
    t = pd.DataFrame(rows).astype({"boundary": "boolean"})
    t.to_parquet(os.path.join(ROOT, "bench", f"boundary_{ds}.parquet"), index=False)
    msg = f"{ds}: 점 {len(t)}, 경계 {t.boundary.mean():.1%}"
    if ds == "ibims1":
        k = t[t.boundary.notna()]; agree = (k.boundary == k.boundary_ratio).mean()
        msg += f" (공식 지도 있는 점 {len(k)}) | 공식 경계 대비 비율 방식: 경계 {t.boundary_ratio.mean():.1%}, 일치 {agree:.1%}, 공식 경계 점 중 비율 방식도 경계 {k[k.boundary.astype(bool)].boundary_ratio.mean():.1%}"
    print(msg, flush=True)
