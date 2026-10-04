"""검증 표 (본 실험 전에 통과해야 함). 논문 수치는 인용만 하고 재계산하지 않는다.
  ① δ1 재현 — DepthLM-12B 는 변환 전(pred_raw = 유클리드 원답) → 변환 후(pred = z) 순서로 DepthLM 표 1(Pixtral 12B, 자체 8,192 점)·DepthVLM 표 1 과,
     baseline 은 DepthVLM 표 2 sparse(nuScenes, iBims-1)와 비교. 점·GT 정의가 논문마다 달라 범위 안 근접 여부로 본다. 각 모델은 자기가 답한 점 전부.
     DepthVLM 공식 eval 은 이미지별 평균, DepthLM 공식 eval 은 점 전체 평균이라 pooled·per-image 둘 다.
  ② 샘플링 대표성 — 같은 dense 예측으로 공통 점(sparse) 지표 vs 이미지 전체 valid GT(dense) 지표 (densestat_*.parquet).
     dense pooled 는 이미지를 유효 픽셀 수로 가중하고(LiDAR 세트는 이미지마다 크게 다름) 공통 점은 이미지를 거의 같게 가중하므로,
     이미지마다 MSE·AbsRel·δ1 을 낸 뒤 평균한 '이미지 균등' dense 값도 같이 낸다.
  ③ z 변환 방향 — DepthLM 의 광선 계수 구간별 중앙값: z/d (= 1/광선 계수, 중심 ≈ 1, 가장자리 < 1) 와 원답·변환값의 GT 대비 비율.
     원답이 유클리드라면 원답/GT 는 가장자리로 갈수록 커지고 변환값/GT 는 평평해야 한다.
사용: python eval/checks.py results/track_a/*.parquet   (densestat_* 도 같이 넘기면 ② 를 낸다)
"""
import sys

import numpy as np
import pandas as pd

from common import ray

REF = {  # (모델, 데이터셋) → {출처: δ1}
    ("DepthLM-12B", "ibims1"): {"DepthLM T1": 0.870, "DepthVLM T1": 0.754},
    ("DepthLM-12B", "nyuv2"): {"DepthLM T1": 0.799, "DepthVLM T1": 0.866},
    ("DepthLM-12B", "ddad"): {"DepthLM T1": 0.670, "DepthVLM T1": 0.654},
    ("DepthLM-12B", "nuscenes"): {"DepthLM T1": 0.819, "DepthVLM T1": 0.736},
    ("DAv2-metric-L", "nuscenes"): {"DepthVLM T2": 0.168}, ("DAv2-metric-L", "ibims1"): {"DepthVLM T2": 0.887},
    ("UniDepthV2-L", "nuscenes"): {"DepthVLM T2": 0.872}, ("UniDepthV2-L", "ibims1"): {"DepthVLM T2": 0.941},
    ("Metric3Dv2-L", "nuscenes"): {"DepthVLM T2": 0.747}, ("Metric3Dv2-L", "ibims1"): {"DepthVLM T2": 0.726},
    ("DepthPro", "nuscenes"): {"DepthVLM T2": 0.389}, ("DepthPro", "ibims1"): {"DepthVLM T2": 0.880},
}
files = sys.argv[1:]
px = pd.concat([pd.read_parquet(f) for f in files if "/densestat_" not in f])
d1 = lambda p, g: (p > 0) & (np.maximum(p / g, g / p) < 1.25)  # 0·음수 답 = 오답

print("### ① δ1 reproduction\n\n| Model | Dataset | pred | δ1 pooled | δ1 per-image | n | paper δ1 |\n|:--|:--|:--|--:|--:|--:|:--|")
for (m, ds), g in px.groupby(["model", "dataset"], sort=False):
    g = g[np.isfinite(g.pred)]
    ref = ", ".join(f"{k} {v:.3f}" for k, v in REF.get((m, ds), {}).items())
    for col in (["pred_raw", "pred"] if m == "DepthLM-12B" else ["pred"]):
        hit = d1(g[col], g.gt_z)
        name = {"pred_raw": "raw (answer as is)", "pred": "z-converted" if m == "DepthLM-12B" else "z"}[col]
        print(f"| {m} | {ds} | {name} | {hit.mean():.3f} | {hit.groupby(g.image_id).mean().mean():.3f} | {len(g)} | {ref} |")

ds_files = [f for f in files if "/densestat_" in f]
if ds_files:
    print("\n### ② sparse (common pixels) vs dense (all valid GT), same predictions — sparse / dense pooled / dense image-balanced\n\n"
          "| Model | Dataset | RMSE | AbsRel | δ1 | pixels sparse / dense |\n|:--|:--|:--|:--|:--|--:|")
    for f in ds_files:
        s = pd.read_parquet(f)
        m, ds = s.model.iloc[0], s.dataset.iloc[0]
        g = px[(px.model == m) & (px.dataset == ds) & np.isfinite(px.pred)]
        print(f"| {m} | {ds} | {np.sqrt(((g.pred - g.gt_z) ** 2).mean()):.3f} / {np.sqrt(s.se.sum() / s.n.sum()):.3f} / {np.sqrt((s.se / s.n).mean()):.3f} | "
              f"{(abs(g.pred - g.gt_z) / g.gt_z).mean():.3f} / {s.ar.sum() / s.n.sum():.3f} / {(s.ar / s.n).mean():.3f} | "
              f"{d1(g.pred, g.gt_z).mean():.3f} / {s.d1.sum() / s.n.sum():.3f} / {(s.d1 / s.n).mean():.3f} | {len(g)} / {s.n.sum()} |")

lm = px[(px.model == "DepthLM-12B") & np.isfinite(px.pred)].copy()
if len(lm):
    lm["r"] = ray(lm.u, lm.v, (lm.fx, lm.fy, lm.cx, lm.cy))
    lm["bin"] = pd.cut(lm.r, [1, 1.02, 1.05, 1.10, 1.20, 1.40, 3], include_lowest=True)
    print("\n### ③ z conversion (DepthLM-12B), medians per ray-factor bin\n\n| Dataset | ray factor | n | z / d | raw / GT | z-converted / GT |\n|:--|:--|--:|--:|--:|--:|")
    for (ds, b), g in lm.groupby(["dataset", "bin"], observed=True):
        print(f"| {ds} | {b} | {len(g)} | {np.median(g.pred / g.pred_raw):.3f} | {np.median(g.pred_raw / g.gt_z):.3f} | {np.median(g.pred / g.gt_z):.3f} |")
