"""DepthLM 오차 분해 (NOTES F-13·F-14): 공통 점·공식 정의(score.py) 그대로.
  ① RMSE 를 단계별로 보정: 그대로 → 데이터셋 배율 → 이미지별 배율 → 이미지별 배율·압축(GT ≈ exp(a·ln 예측 + b)). 남는 값 = 배율·압축으로 못 고치는 오차.
  ② 답 재사용(계단): 이미지 안 서로 다른 답의 비율, 같은 답을 받은 점들의 GT 폭, 이미지별 log 선형 보정 잔차 중 '같은 답 묶음 안' 몫.
     dense 모델은 예측을 소수 둘째 자리로 반올림해 같은 계산 (DepthLM 답도 둘째 자리).
사용: python eval/error_anatomy.py <parquet...> --datasets ibims1 nyuv2
"""
import argparse

import numpy as np
import pandas as pd

from score import depthlm_answers


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--datasets", nargs="+", default=["ibims1", "nyuv2"])
    a = ap.parse_args()
    d = pd.concat([pd.read_parquet(f, columns=["dataset", "image_id", "u", "v", "gt_z", "pred", "pred_raw", "model"]) for f in a.files])
    d = depthlm_answers(d[d.dataset.isin(a.datasets)], "official")
    for ds in a.datasets:
        x = d[d.dataset == ds]
        ok = x[np.isfinite(x.pred)].groupby(["image_id", "u", "v"]).model.nunique() == x.model.nunique()
        x = x.set_index(["image_id", "u", "v"]).loc[ok[ok].index].reset_index()
        rows = []
        for m, g in x.groupby("model"):
            sm = np.exp(np.median(np.log(g.gt_z) - np.log(g.pred.clip(lower=1e-3))))
            se, n, tot, wit, share, span = np.zeros(4), 0, 0.0, 0.0, [], []
            for _, h in g.groupby("image_id"):
                p, t = h.pred.clip(lower=1e-3).values, h.gt_z.values
                lp, lg = np.log(p), np.log(t)
                c = np.polyfit(lp, lg, 1) if np.ptp(lp) > 0 else (0, lg.mean())
                se += [((p - t) ** 2).sum(), ((p * sm - t) ** 2).sum(), ((p * np.exp(np.median(lg - lp)) - t) ** 2).sum(),
                       ((np.exp(np.polyval(c, lp)) - t) ** 2).sum()]
                n += len(h)
                if len(h) >= 5:
                    ans = (h.pred_raw if m == "DepthLM-12B" else h.pred).round(2).values
                    e = lp - np.polyval(np.polyfit(lg, lp, 1), lg)
                    s = pd.DataFrame(dict(a=ans, e=e, lg=lg))
                    tot += (e ** 2).sum()
                    wit += ((s.e - s.groupby("a").e.transform("mean")) ** 2).sum()
                    share.append(s.a.nunique() / len(s))
                    gs = s.groupby("a").lg
                    span += list((gs.max() - gs.min())[gs.size() >= 2].values)
            r = np.sqrt(se / n)
            rows.append({"model": m, "RMSE": r[0], "+데이터셋 배율": r[1], "+이미지별 배율": r[2], "+이미지별 배율·압축": r[3],
                         "서로 다른 답 비율": np.mean(share), "같은 답의 GT 폭": np.expm1(np.median(span)) if span else np.nan, "잔차 중 같은 답 안 몫": wit / tot})
        print(f"\n=== {ds}\n" + pd.DataFrame(rows).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
