"""어느 영역을 틀리는가 (NOTES F-13 후속): 이미지별 배율·압축(ln 예측 = a·ln GT + b)을 맞춘 뒤 남는 잔차를 영역·질감별로 본다.
  영역: iBims-1 = 공식 평면 마스크(mask_floor·mask_wall·mask_table, 평면마다 번호), 나머지 = '기타'(물체·천장 등).
        NYUv2 = GT 깊이로 계산한 표면 방향(바닥·천장 = 법선이 세로, 벽 = 법선이 가로, 기타).
  질감: 회색조 기울기 크기의 15×15 평균 (이미지 안 순위로 3 등분).
  평면 분해(iBims-1): 같은 평면 인스턴스 안 잔차의 흩어짐 vs 평면 평균끼리의 흩어짐 — 면 전체를 통째로 다른 깊이에 놓는지 본다.
  평면 안 기울기(iBims-1): 한 평면 안에서 ln 예측 ~ ln GT 기울기 — 멀어지는 벽·바닥의 깊이 변화를 따라가는지 본다.
  평면별 맞춤(iBims-1): 평면마다 따로 맞춘 뒤 남는 잔차 — 매끈한 면으로 그리는지, 튀는 점이 평면 가장자리(다른 물체 옆)에 몰리는지.
사용: python eval/region_errors.py <parquet...> --data_root ~/data/depthvlm_bench
"""
import argparse
import os

import cv2
import numpy as np
import pandas as pd

from common import load_bench
from dense_sparse import gt_map
from score import depthlm_answers

PLANES = ["floor", "wall", "table"]


def texture(path):
    g = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY).astype(np.float32)
    m = np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))
    return cv2.blur(m, (15, 15))


def normals(z, k, s=4):
    """GT z 맵 → 카메라 좌표 법선 (s 픽셀 간격 중앙 차분, cross(du, dv)). y 는 아래 방향 — 바닥은 n_y > 0 (아래로 갈수록 가까워짐), 천장은 n_y < 0."""
    H, W = z.shape
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    P = np.dstack([(u - k[2]) * z / k[0], (v - k[3]) * z / k[1], z]).astype(np.float32)
    du, dv = np.zeros_like(P), np.zeros_like(P)
    du[:, s:-s], dv[s:-s] = P[:, 2 * s:] - P[:, :-2 * s], P[2 * s:] - P[:-2 * s]
    n = np.cross(du, dv)
    n /= np.linalg.norm(n, axis=2, keepdims=True) + 1e-9
    bad = (z <= 0) | ~np.isfinite(n).all(2)
    for d in (du, dv):
        bad |= (d[..., 2] == 0) & (d[..., 0] == 0)
    n[bad] = np.nan
    return n


def labels(ds, r, root, k, gt):
    """점마다 (영역, 평면 인스턴스 id) — 영역 지도와 인스턴스 지도."""
    name = os.path.splitext(os.path.basename(r["image"]))[0]
    if ds == "ibims1":
        reg, inst = np.full(gt.shape, "other", object), np.full(gt.shape, "", object)
        for p in PLANES:
            f = os.path.join(root, os.path.dirname(os.path.dirname(r["image"])), f"mask_{p}", name + ".png")
            if os.path.exists(f):
                m = cv2.imread(f, cv2.IMREAD_UNCHANGED)
                reg[m > 0] = p
                inst[m > 0] = np.char.add(f"{p}", m[m > 0].astype(str))
        return reg, inst
    n = normals(gt, k)
    ny = np.abs(n[..., 1])
    reg = np.where(ny > 0.85, np.where(n[..., 1] > 0, "floor", "ceiling"), np.where(ny < 0.25, "wall", "other")).astype(object)
    reg[np.isnan(ny)] = "other"
    return reg, np.full(gt.shape, "", object)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--datasets", nargs="+", default=["ibims1", "nyuv2"])
    a = ap.parse_args()
    d = pd.concat([pd.read_parquet(f, columns=["dataset", "image_id", "u", "v", "gt_z", "pred", "pred_raw", "model", "fx", "fy", "cx", "cy"]) for f in a.files])
    d = depthlm_answers(d[d.dataset.isin(a.datasets)], "official")
    for ds in a.datasets:
        x = d[d.dataset == ds]
        ok = x[np.isfinite(x.pred)].groupby(["image_id", "u", "v"]).model.nunique() == x.model.nunique()
        x = x.set_index(["image_id", "u", "v"]).loc[ok[ok].index].reset_index()
        recs, K = load_bench(ds)
        lab = []
        for r in recs:
            pts = x[(x.image_id == r["image"]) & (x.model == x.model.iloc[0])][["image_id", "u", "v"]]
            if pts.empty:
                continue
            k = K[r["image"]]
            gt = gt_map(r, a.data_root, ds)
            reg, inst = labels(ds, r, a.data_root, k, gt)
            tx = texture(os.path.join(a.data_root, r["image"]))
            u, v = pts.u.values.astype(int), pts.v.values.astype(int)
            t = tx[v, u]
            ins = inst[v, u]
            edge = np.array([cv2.distanceTransform((inst == q).astype(np.uint8), cv2.DIST_L2, 5)[v[i], u[i]] if q else np.nan for i, q in enumerate(ins)])
            lab.append(pts.assign(region=reg[v, u], inst=ins, tex=pd.Series(t).rank(pct=True).values, edge_px=edge))
        x = x.merge(pd.concat(lab), on=["image_id", "u", "v"])
        res = []
        for (m, im), h in x.groupby(["model", "image_id"]):
            lp, lg = np.log(h.pred.clip(lower=1e-3)), np.log(h.gt_z)
            A = np.vstack([lg, np.ones_like(lg)]).T
            c, *_ = np.linalg.lstsq(A, lp, rcond=None)
            res.append(h.assign(res=(lp - A @ c) * 100, lp=lp, lg=lg))
        x = pd.concat(res)
        x["tex_bin"] = pd.cut(x.tex, [0, 1 / 3, 2 / 3, 1], labels=["매끈", "중간", "질감 많음"], include_lowest=True)
        print(f"\n===== {ds} (점 {len(x) // x.model.nunique()}개). 배율·압축 보정 뒤 잔차 ×100 (+ = 상대적으로 길게)")
        print("점 비율:", x[x.model == "DepthLM-12B"].region.value_counts(normalize=True).round(3).to_dict())
        for col in ["region", "tex_bin"]:
            g = x.groupby(["model", col], observed=True).res
            print(f"\n[{col}] 부호 있는 중앙값 / |잔차| 중앙값")
            print(pd.concat([g.median().unstack().round(1).add_suffix(" 부호"), g.apply(lambda s: s.abs().median()).unstack().round(1).add_suffix(" 크기")], axis=1).to_string())
        if ds == "ibims1":
            print("\n[평면 분해] 평면 인스턴스(점 5 개 이상) 안 흩어짐 vs 인스턴스 평균끼리 흩어짐 (같은 이미지 안, 표준편차 ×100)")
            rows = []
            for m, g in x[x.inst != ""].groupby("model"):
                gi = g.groupby(["image_id", "inst"]).res
                keep = gi.transform("size") >= 5
                g = g[keep]
                within = np.sqrt((g.res - g.groupby(["image_id", "inst"]).res.transform("mean")).pow(2).mean())
                means = g.groupby(["image_id", "inst"]).res.mean().reset_index()
                between = np.sqrt((means.res - means.groupby("image_id").res.transform("mean")).pow(2)[means.groupby("image_id").res.transform("size") >= 2].mean())
                rows.append(dict(model=m, within_plane=within, between_planes=between, n_inst=len(means)))
            print(pd.DataFrame(rows).round(2).to_string(index=False))
            print("\n[평면 안 깊이 기울기] 평면 인스턴스(점 8 개 이상, GT ln 깊이 폭 ≥ 0.2) 안에서 ln 예측을 ln GT 에 회귀한 기울기 (1 = 평면이 멀어지는 정도를 그대로 따라감)")
            rows = []
            for m, g in x[x.inst != ""].groupby("model"):
                sl = [np.polyfit(h.lg, h.lp, 1)[0] for _, h in g.groupby(["image_id", "inst"]) if len(h) >= 8 and np.ptp(h.lg) >= 0.2]
                rows.append(dict(model=m, median_slope=np.median(sl), iqr=f"{np.percentile(sl, 25):.2f}–{np.percentile(sl, 75):.2f}", n_inst=len(sl)))
            print(pd.DataFrame(rows).round(2).to_string(index=False))
            print("\n[평면별 맞춤 뒤] 평면마다 ln 예측 ~ ln GT 를 따로 맞춘 뒤 남는 잔차 (×100) — 평면을 매끈한 면으로 그리는지")
            rows, out = [], []
            for (m, im, ins), h in x[x.inst != ""].groupby(["model", "image_id", "inst"]):
                if len(h) >= 8:
                    out.append(h.assign(pres=(h.lp - np.polyval(np.polyfit(h.lg, h.lp, 1), h.lg)) * 100))
            y = pd.concat(out)
            for m, g in y.groupby("model"):
                rows.append(dict(model=m, rms=np.sqrt((g.pres ** 2).mean()), median_abs=g.pres.abs().median(), over10pct=(g.pres.abs() > 10).mean(), n=len(g)))
            print(pd.DataFrame(rows).round(3).to_string(index=False))
            L = y[y.model == "DepthLM-12B"].copy()
            L["edge"] = pd.cut(L.edge_px, [0, 10, 25, 50, np.inf], labels=["≤10px", "10–25", "25–50", ">50"], include_lowest=True)
            print("DepthLM 평면 가장자리까지 거리별 |잔차| > 10 % 비율:", L.groupby("edge", observed=True).pres.apply(lambda s: round((s.abs() > 10).mean(), 3)).to_dict(),
                  "| 길게", int((L.pres > 10).sum()), "/ 짧게", int((L.pres < -10).sum()))


if __name__ == "__main__":
    main()
