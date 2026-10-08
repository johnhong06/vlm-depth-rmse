"""NYUv2 를 Kinect 원측정 GT(labeled.mat rawDepths, 0.005 m ≤ GT < 10 m)로 채점하기 위한 입력 — 예측은 그대로, GT 만 바꾼다 (NOTES D-22, F-18).
벤치 GT(SUN RGB-D depth_bfx ÷ 8000)는 8.19 m 이상 결함·채운 값이 있어 주 표에 쓰지 않는다. 원측정 위치는 weak_nyu_raw.raw_gt (SUN 자르기 창, 정렬 오차 > 2 % 인 2 장은 뺀다).
  Track A  : 874·875 의 nyuv2 parquet 에서 gt_z 를 그 점의 원측정 값으로, 원측정 없는 점은 뺀다 → <out>/track_a/
  경계     : 원측정 GT 의 이웃 깊이 비 (D-18 과 같은 정의) → <out>/boundary_nyuv2_raw.parquet (breakdown.py --nyu_boundary)
  Track B  : 로컬 예측 맵(~/data/vdr_maps — H200 Track B 와 반올림까지 같음, F-16) + dense_full.sums 그대로 → <out>/track_b/stats_<모델>_nyuv2.parquet
             d1_canon 은 878 의 값 (DepthVLM 공식 방식 재현 확인용 — 벤치 GT 기준 그대로)
사용: python eval/nyu_raw_inputs.py --out ~/data/vdr_raw/nyuraw
"""
import argparse
import glob
import os

import cv2
import numpy as np
import pandas as pd

import weak_common as W  # noqa: I001 — prep/ 를 sys.path 에 넣는다 (boundary_labels)
from boundary_labels import near, ratio_contour
from common import load_bench, ray
from dense_full import BIN_NAMES, BINS, R, sums
from weak_nyu_raw import raw_gt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = os.path.expanduser(a.out)
    os.makedirs(os.path.join(out, "track_a"), exist_ok=True)
    os.makedirs(os.path.join(out, "track_b"), exist_ok=True)
    recs, K = load_bench("nyuv2")
    gts, bnd, skipped = {}, [], []
    for r in recs:
        _, g, err = raw_gt(r)
        if err > 0.02:
            skipped.append(r["image"])
            continue
        gts[r["image"]] = g
        u, v = np.array(r["pixel_coords"]).T
        b = near(ratio_contour(g), u, v)
        bnd += [dict(image_id=r["image"], u=int(x), v=int(y), boundary=bool(z)) for x, y, z in zip(u, v, b)]
    pd.DataFrame(bnd).to_parquet(os.path.join(out, "boundary_nyuv2_raw.parquet"), index=False)
    print(f"[nyu_raw] 이미지 {len(gts)} (정렬 오차로 뺀 것 {len(skipped)}: {skipped})")

    for f in sorted(glob.glob(os.path.join(W.RAW, "87[45]", "vdr", "track_a", "*nyuv2*.parquet"))):
        if "densestat" in f:
            continue
        d = pd.read_parquet(f)
        d = d[d.image_id.isin(gts)].copy()
        d["gt_z"] = [gts[i][v, u] for i, u, v in zip(d.image_id, d.u, d.v)]
        n0 = len(d)
        d = d[d.gt_z > 0]
        d.to_parquet(os.path.join(out, "track_a", os.path.basename(f)), index=False)
        print(f"[nyu_raw] Track A {os.path.basename(f)}: 점 {n0} → 원측정 있는 점 {len(d)}")

    for m in W.DENSE:
        canon = pd.read_parquet(os.path.join(W.RAW, "878", "vdr", "track_b", f"stats_{m}_nyuv2.parquet"))
        canon = canon[canon.group == "all"].set_index("image_id").d1_canon
        rows = []
        for r in recs:
            if r["image"] not in gts:
                continue
            gt, pred, k = gts[r["image"]], W.load_map(m, "nyuv2", r).astype(np.float32), K[r["image"]]
            valid, fin = gt > 0, np.isfinite(pred)
            ys, xs = np.nonzero(valid & fin)
            g, p, r_ = gt[ys, xs].astype(np.float64), pred[ys, xs].astype(np.float64), ray(xs, ys, k)
            groups = {"all": np.ones(g.size, bool)}
            b = np.digitize(g, BINS["Indoor"][1:-1])
            groups.update({f"dist:{n}": b == j for j, n in enumerate(BIN_NAMES)})
            bd = cv2.distanceTransform((~ratio_contour(gt)).astype(np.uint8), cv2.DIST_L2, 5)[ys, xs] <= R
            groups.update({"region:boundary": bd, "region:interior": ~bd})
            for name, sel in groups.items():
                if sel.any():
                    rows.append(dict(dataset="nyuv2", image_id=r["image"], model=m, group=name, **sums(p[sel], g[sel], r_[sel], "nyuv2"),
                                     **({"n_nonfinite": int((valid & ~fin).sum()), "d1_canon": float(canon[r["image"]])} if name == "all" else {})))
        s = pd.DataFrame(rows)
        s.to_parquet(os.path.join(out, "track_b", f"stats_{m}_nyuv2.parquet"), index=False)
        t = s[s.group == "all"]
        print(f"[nyu_raw] Track B {m}: 이미지 {len(t)}, 픽셀 {t.n.sum()}, RMSE {np.sqrt(t.se.sum() / t.n.sum()):.3f} AbsRel {t.ar.sum() / t.n.sum():.4f}")


if __name__ == "__main__":
    main()
