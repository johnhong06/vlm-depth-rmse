"""NYUv2 를 Kinect 원측정(labeled.mat rawDepths) GT 로 다시 채점 (NOTES F-17) — 벤치 GT(SUN RGB-D depth_bfx /8000)는
8 m 이상을 표현하지 못하고(최댓값 7.995 m) 원측정 8–10 m 픽셀을 중앙 4.86 m 로 적는다. 같은 예측 맵·같은 공통 점으로
GT 만 원측정으로 바꾼다(0.005–10 m, 원측정이 없는 픽셀·점은 뺀다). 이미지 위치는 weak_common.nyu_offset (원측정에 맞춘 SUN RGB-D 자르기).
출력: results_vlm_weakness/data/dense_nyuraw.parquet, points_nyuraw.parquet (dataset = 'nyuv2_raw'), nyu_raw_summary.csv
"""
import os

import numpy as np
import pandas as pd

import weak_common as W
import weak_stats as S

OUT = os.path.join(W.OUT, "data")


def raw_gt(r):
    g = W.gt_map(r, W.DATA, "nyuv2")
    k, oy, ox, err = W.nyu_offset(r, g)
    raw = W.nyu_file()["f"]["rawDepths"][k].T[oy:oy + g.shape[0], ox:ox + g.shape[1]].astype(np.float32)
    raw[(raw < 0.005) | (raw >= 10)] = 0   # 정확히 10.0 = 센서 상한 포화값(실측 아님, NOTES F-18) — BTS 표준처럼 GT < 10 m
    return g, raw, err


def main():
    LM = W.depthlm_points()
    dense, pts, far = [], [], []
    for r in W.records("nyuv2"):
        g, gr, err = raw_gt(r)
        if err > 0.02:
            continue
        P = {m: W.load_map(m, "nyuv2", r) for m in W.DENSE + W.SUPP}
        valid = gr > 0
        a = W.attributes("nyuv2", r, gr)            # 거리 구간·경계도 원측정 GT 로
        rows = S.dense_rows("nyuv2", r, gr, valid, P, W.groups("nyuv2", a))
        for x in rows:
            x["dataset"] = "nyuv2_raw"
        dense += rows
        t = S.point_rows("nyuv2", r, gr, a, {m: P[m] for m in W.DENSE}, LM)
        t["gt_bfx"] = t.gt_z
        t["gt_z"] = t.gt_map                         # 공통 점의 GT 를 원측정 값으로
        t = t[t.gt_z > 0].assign(dataset="nyuv2_raw")
        pts.append(t)
        sel = (gr >= 8) & (g > 0)
        if sel.any():
            far.append(dict(image_id=r["image"], n=int(sel.sum()), raw=float(np.median(gr[sel])), bfx=float(np.median(g[sel])),
                            **{W.SHORT[m]: float(np.median(p[sel])) for m, p in P.items()}))
    pd.DataFrame(dense).to_parquet(os.path.join(OUT, "dense_nyuraw.parquet"), index=False)
    pd.concat(pts, ignore_index=True).to_parquet(os.path.join(OUT, "points_nyuraw.parquet"), index=False)
    f = pd.DataFrame(far)
    f.to_csv(os.path.join(W.OUT, "nyu_raw_far_pixels.csv"), index=False)
    print("원측정 8–10 m 픽셀이 있는 이미지", len(f), "픽셀", int(f.n.sum()))
    print("그 픽셀의 중앙값 (이미지별 중앙값의 픽셀 수 가중 평균):",
          {c: round(float(np.average(f[c], weights=f.n)), 2) for c in ["raw", "bfx"] + [W.SHORT[m] for m in W.DENSE + W.SUPP]})


if __name__ == "__main__":
    main()
