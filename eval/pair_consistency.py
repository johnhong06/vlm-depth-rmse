"""점끼리 일관성 분석 (NOTES F-13): DepthLM 이 점마다 따로 답해 이웃 점끼리 일관되지 않은지 본다.
  같은 이미지·GT 깊이가 비슷한(|ln g1 − ln g2| < 0.1) 두 점의 log 오차 차이 |e1 − e2| (e = ln 예측 − ln GT) 를 화면 거리 구간별로 잰다.
  맵을 한 번에 내는 모델은 가까울수록 오차가 비슷해 값이 작아지고, 점마다 독립으로 답하면 거리와 상관없이 평평하다.
  경계 점은 뺀다(bench/boundary_<ds>.parquet). 공통 점·DepthLM 공식 정의(score.py) 그대로. 답 자릿수(반올림 습관)도 찍는다.
사용: python eval/pair_consistency.py <parquet...> --datasets ibims1 nyuv2
"""
import argparse
import os

import numpy as np
import pandas as pd

from common import ROOT
from score import depthlm_answers

BINS = [0, 20, 40, 80, 160, np.inf]   # 화면 거리 (원본 픽셀)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--datasets", nargs="+", default=["ibims1", "nyuv2"])
    a = ap.parse_args()
    d = pd.concat([pd.read_parquet(f, columns=["dataset", "image_id", "u", "v", "gt_z", "pred", "pred_raw", "model"]) for f in a.files])
    d = depthlm_answers(d[d.dataset.isin(a.datasets)], "official")
    for ds in a.datasets:
        x = d[d.dataset == ds]
        n = x.model.nunique()
        ok = x[np.isfinite(x.pred)].groupby(["image_id", "u", "v"]).model.nunique() == n
        x = x.set_index(["image_id", "u", "v"]).loc[ok[ok].index].reset_index()
        bd = pd.read_parquet(os.path.join(ROOT, "bench", f"boundary_{ds}.parquet")).dropna(subset=["boundary"])
        x = x.merge(bd[["image_id", "u", "v", "boundary"]], on=["image_id", "u", "v"])
        x = x[~x.boundary.astype(bool)]
        x["e"] = np.log(x.pred.clip(lower=1e-3)) - np.log(x.gt_z)
        rows = []
        for (m, im), g in x.groupby(["model", "image_id"]):
            u, v, lg, e = g.u.values, g.v.values, np.log(g.gt_z.values), g.e.values
            i, j = np.triu_indices(len(g), 1)
            s = np.abs(lg[i] - lg[j]) < 0.1
            rows.append(pd.DataFrame(dict(model=m, px=np.hypot(u[i] - u[j], v[i] - v[j])[s], de=np.abs(e[i] - e[j])[s])))
        p = pd.concat(rows)
        p["bin"] = pd.cut(p.px, BINS, right=False)
        t = p.groupby(["model", "bin"], observed=True).de.median().unstack()
        t["near/far"] = t.iloc[:, 0] / t.iloc[:, -1]
        print(f"\n=== {ds}: 비슷한 GT 깊이 점 쌍의 |log 오차 차이| 중앙값 (×100), 화면 거리 구간별 (경계 제외)")
        print((t.iloc[:, :-1] * 100).round(1).assign(**{"가까움/멂": t["near/far"].round(2)}).to_string())
        print("쌍 수:", p[p.model == p.model.iloc[0]].bin.value_counts(sort=False).tolist())
        r = x[x.model == "DepthLM-12B"].pred_raw
        print(f"DepthLM 답 자릿수: 정수 {np.mean(np.isclose(r, r.round(0))):.1%}, 0.5 배수 {np.mean(np.isclose(r * 2, (r * 2).round())):.1%}, "
              f"0.1 배수 {np.mean(np.isclose(r * 10, (r * 10).round())):.1%}, 서로 다른 값 {r.nunique()} / {len(r)}")


if __name__ == "__main__":
    main()
