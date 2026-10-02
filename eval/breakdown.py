"""보조 집계 (NOTES D-18): 주 결과(score.py)와 같은 공통 점·같은 DepthLM 처리로
  ① 거리 구간별 (실내 근 0–2 / 중 2–4 / 원 4 m–, 실외 근 0–10 / 중 10–30 / 원 30 m–, GT 기준)
  ② 경계 / 내부별 (iBims-1 = 공식 경계 지도, NYUv2 = 이웃 GT 깊이 비 > 1.1; bench/boundary_<ds>.parquet. DIODE·nuScenes·DDAD 는 제외)
  ③ log 지표 — log-RMSE (pooled·이미지별), SILog (KITTI 정의: 100·sqrt(mean d² − (mean d)²), d = ln 예측 − ln GT, 이미지별 → 평균).
     log 지표에서만 예측을 데이터셋 cap 범위로 자른다 (BTS·AdaBins·ZoeDepth 계열 평가 관행). 잘린 점 수를 적는다. 주 지표는 자르지 않는다.
  CI = 이미지 단위 bootstrap 95 % (B 회, 시드 0).
사용: python eval/breakdown.py <parquet...> --models ... --out tables/breakdown [--depthlm official]"""
import argparse
import os

import numpy as np
import pandas as pd

from common import BENCH, ROOT
from score import depthlm_answers

BINS = {"Indoor": [0, 2, 4, np.inf], "Outdoor": [0, 10, 30, np.inf]}
BIN_NAMES = ["near", "mid", "far"]
CAP = {"ibims1": (0.005, 25), "nyuv2": (0.005, 10), "ddad": (0.05, 120), "nuscenes": (0.05, 80), "diode_outdoor": (0.05, 80)}  # 벤치 DATASET_CONFIGS + DIODE (D-14)
BOUNDARY_SETS = ["ibims1", "nyuv2"]


def per_image(g):
    """이미지별 충분통계 (예측은 log 용으로 cap 범위로 자른 값도 계산)."""
    e = g.pred - g.gt_z
    lp = np.log(g.pred.clip(*CAP[g.dataset.iloc[0]]))
    dl = lp - np.log(g.gt_z)
    t = pd.DataFrame(dict(image_id=g.image_id, n=1, se=e ** 2, ar=e.abs() / g.gt_z,
                          d1=(g.pred > 0) & (np.maximum(g.pred / g.gt_z, g.gt_z / g.pred) < 1.25),
                          dl=dl, dl2=dl ** 2, n_clipped=(g.pred < CAP[g.dataset.iloc[0]][0]) | (g.pred > CAP[g.dataset.iloc[0]][1])))
    return t.groupby("image_id").sum()


def boot(s, B, rng):
    """원 표본 + B 회 이미지 복원추출 → [rmse, absrel, d1, logrmse, silog] 행렬 (B+1, 5)."""
    w = np.vstack([np.ones(len(s)), rng.multinomial(len(s), np.full(len(s), 1 / len(s)), B)])
    n = w @ s.n.values
    sil = 100 * np.sqrt(np.maximum(s.dl2 / s.n - (s.dl / s.n) ** 2, 0))   # 이미지별 SILog (점 2 개 이상)
    m2 = (s.n >= 2).values.astype(float)
    return np.stack([np.sqrt(w @ s.se.values / n), w @ s.ar.values / n, w @ s.d1.values / n, np.sqrt(w @ s.dl2.values / n),
                     (w * m2) @ sil.values / np.maximum((w * m2).sum(1), 1)], 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--B", type=int, default=1000)
    ap.add_argument("--depthlm", choices=["official", "converted", "raw"], default="official")
    a = ap.parse_args()
    d = pd.concat([pd.read_parquet(f, columns=["dataset", "image_id", "u", "v", "gt_z", "pred", "pred_raw", "model"]) for f in a.files])
    d = depthlm_answers(d[d.model.isin(a.models)], a.depthlm)
    rows = []
    for ds in [k for k in BENCH if k in set(d.dataset)]:
        x = d[d.dataset == ds]
        if x.model.nunique() < len(a.models):
            continue
        ok = x[np.isfinite(x.pred)].groupby(["image_id", "u", "v"]).model.nunique() == len(a.models)
        x = x.set_index(["image_id", "u", "v"]).loc[ok[ok].index].reset_index()   # score.py 와 같은 공통 집합
        dom = BENCH[ds][1]
        x["bin"] = pd.cut(x.gt_z, BINS[dom], labels=BIN_NAMES, right=False)
        groups = [("all", x)] + [(f"dist:{b}", x[x.bin == b]) for b in BIN_NAMES]
        if ds in BOUNDARY_SETS:
            bd = pd.read_parquet(os.path.join(ROOT, "bench", f"boundary_{ds}.parquet")).dropna(subset=["boundary"])
            y = x.merge(bd[["image_id", "u", "v", "boundary"]], on=["image_id", "u", "v"])
            groups += [("region:boundary", y[y.boundary.astype(bool)]), ("region:interior", y[~y.boundary.astype(bool)])]
        for name, g in groups:
            for m in a.models:
                s = per_image(g[g.model == m])
                r = boot(s, a.B, np.random.default_rng(0))   # 모델마다 같은 재표본 (같은 시드·같은 이미지 순서)
                lo, hi = np.percentile(r[1:], [2.5, 97.5], 0)
                rows.append(dict(dataset=ds, domain=dom, group=name, model=m, n_px=int(s.n.sum()), n_img=len(s), n_clip=int(s["n_clipped"].sum()),
                                 **{k: r[0, i] for i, k in enumerate(["rmse", "absrel", "d1", "logrmse", "silog"])},
                                 **{k + "_lo": lo[i] for i, k in enumerate(["rmse", "absrel", "d1", "logrmse", "silog"])},
                                 **{k + "_hi": hi[i] for i, k in enumerate(["rmse", "absrel", "d1", "logrmse", "silog"])}))
    t = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    t.to_csv(a.out + ".csv", index=False)
    f = lambda r, k, p: f"{r[k]:.{p}f} [{r[k + '_lo']:.{p}f}, {r[k + '_hi']:.{p}f}]"
    out = [f"DepthLM answers: {a.depthlm}. Distance bins (GT): indoor near 0–2 / mid 2–4 / far ≥4 m, outdoor near 0–10 / mid 10–30 / far ≥30 m. "
           "Boundary: iBims-1 official edge maps (86 of 100 images), NYUv2 neighbour GT depth ratio > 1.1; within 3 px. "
           f"Brackets = 95 % CI, image bootstrap B = {a.B}."]
    for title, sel, cols in [("① distance bins", t.group.str.startswith("dist") | (t.group == "all"), ["rmse", "absrel", "d1"]),
                             ("② boundary vs interior", t.group.str.startswith("region"), ["rmse", "absrel", "d1"]),
                             ("③ log metrics (all common pixels)", t.group == "all", ["logrmse", "silog"])]:
        out += [f"\n### {title}\n", "| Dataset | Group | Model | " + " | ".join(cols) + " | px | img |" + (" clipped |" if "silog" in cols else ""),
                "|:--|:--|:--|" + "--:|" * (len(cols) + 2 + ("silog" in cols))]
        for _, r in t[sel].iterrows():
            out.append(f"| {r.dataset} | {r.group} | {r.model} | " + " | ".join(f(r, k, 3) for k in cols) + f" | {r.n_px} | {r.n_img} |"
                       + (f" {r.n_clip} |" if "silog" in cols else ""))
    open(a.out + ".md", "w").write("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
