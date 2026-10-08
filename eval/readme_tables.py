"""README 결과 표 — score.py / breakdown.py / score_dense.py 가 만든 csv 만 읽어 마크다운 표를 찍는다 (값을 손으로 옮기지 않는다).
  ① 한눈에 보기: 실내·실외 평균 RMSE·AbsRel·δ1 + 평균 순위(그 도메인의 데이터셋 × 3 지표 순위의 평균)
  ② 지표별 상세: 지표마다 데이터셋 5 개 + 도메인 평균 열
  ③ 보조 지표: 도메인 평균 log-RMSE·SILog·원거리 RMSE·경계 δ1·경계 하락(내부 δ1 − 경계 δ1)
  굵게 = 1 위, 밑줄 = 2 위 (표시 자릿수로 반올림한 값 기준, 같은 값은 같은 순위). 도메인 평균 = 데이터셋 값의 평균.
사용: python eval/readme_tables.py --main tables/all_track_a.csv --aux tables/breakdown.csv     (Track A)
      python eval/readme_tables.py --main tables/trackb_track_b.csv --aux tables/trackb_track_b.csv  (Track B: 한 csv 에 group 열)
"""
import argparse

import numpy as np
import pandas as pd

from score import CEILING, TRAINED

NAME = {"DepthLM-12B": "DepthLM-12B", "DepthVLM-4B": "DepthVLM-4B", "DAv2-metric-L": "Depth Anything V2", "UniDepthV2-L": "UniDepthV2",
        "Metric3Dv2-L": "Metric3Dv2", "DepthPro": "Depth Pro"}
DS = {"Indoor": ["ibims1", "nyuv2"], "Outdoor": ["ddad", "nuscenes", "diode_outdoor"]}
DSN = {"ibims1": "iBims-1", "nyuv2": "NYUv2", "ddad": "DDAD", "nuscenes": "NuScenes", "diode_outdoor": "DIODE Outdoor"}
DOM = {"Indoor": "실내", "Outdoor": "실외"}
MET = [("rmse", "RMSE↓ (m)", 2, False), ("absrel", "AbsRel↓", 3, False), ("d1", "δ1↑", 3, True)]


def mark(m, ds):
    return ("†" if (m, ds) in TRAINED else "") + ("‡" if (m, ds) in CEILING else "")


def styled(vals, p, high):
    """한 열의 값 → 굵게(1 위)·밑줄(2 위) 표시 문자열. None 은 '대기'."""
    r = [None if v is None else round(v, p) for v in vals]
    u = sorted({x for x in r if x is not None}, reverse=high)
    out = []
    for x in r:
        s = "대기" if x is None else f"{x:.{p}f}"
        out.append(f"**{s}**" if u and x == u[0] else f"<u>{s}</u>" if len(u) > 1 and x == u[1] else s)
    return out


def table(head, rows, cols):
    """rows: 모델 이름 목록, cols: [(값 목록, 자릿수, 높을수록 좋음, 꼬리표 목록 또는 None)]"""
    body = [styled(v, p, h) for v, p, h, _ in cols]
    body = [[s + (t[i] if t and s != "대기" else "") for i, s in enumerate(c)] for c, (_, _, _, t) in zip(body, cols)]
    lines = ["| " + " | ".join(head) + " |", "|---|" + "--:|" * len(cols)]
    return "\n".join(lines + ["| " + " | ".join([NAME[m]] + [c[i] for c in body]) + " |" for i, m in enumerate(rows)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--main", required=True)
    ap.add_argument("--aux", required=True)
    ap.add_argument("--models", nargs="+", required=True)
    a = ap.parse_args()
    t = pd.read_csv(a.main)
    t = t[t.group == "all"] if "group" in t else t
    for d in DS:   # 표에 있는 데이터셋만 (NOTES D-24: 결과 표는 zero-shot 세트만 — 실외는 DIODE 하나)
        DS[d] = [x for x in DS[d] if x in set(t.dataset)]
    dn = lambda d: DOM[d] if len(DS[d]) > 1 else f"{DOM[d]}({DSN[DS[d][0]]})"
    val = lambda m, ds, k: (lambda x: None if x.empty or not np.isfinite(x.iloc[0]) else float(x.iloc[0]))(t[(t.model == m) & (t.dataset == ds)][k])
    mean = lambda m, dom, k, f=val: (lambda v: None if None in v else float(np.mean(v)))([f(m, ds, k) for ds in DS[dom]])

    def rank(dom):  # 모델별 평균 순위 (데이터셋 × 지표, 같은 값은 같은 순위). 값이 하나라도 없으면 None
        r = {m: [] for m in a.models}
        for ds in DS[dom]:
            for k, _, p, high in MET:
                v = {m: val(m, ds, k) for m in a.models}
                if None in v.values():
                    return {m: None for m in a.models}
                v = {m: round(x, p) for m, x in v.items()}   # 표시 자릿수에서 같으면 같은 순위 (표에서 확인할 수 있게)
                for m in a.models:
                    r[m].append(1 + sum((v[o] > v[m]) if high else (v[o] < v[m]) for o in a.models))
        return {m: float(np.mean(x)) for m, x in r.items()}

    cols = []
    for dom in DS:
        rk = rank(dom)
        cols += [([mean(m, dom, k) for m in a.models], p, h, None) for k, _, p, h in MET] + [([rk[m] for m in a.models], 1, False, None)]
    head = ["모델"] + [f"{dn(d)} {x}" for d in DS for x in ["RMSE↓", "AbsRel↓", "δ1↑", "평균 순위↓"]]
    print("#### 한눈에 보기 (도메인 평균)\n\n" + table(head, a.models, cols) + "\n\n#### 지표별 상세\n")
    for k, title, p, h in MET:
        cols, head = [], [title]
        for dom in DS:
            for ds in DS[dom]:
                cols.append(([val(m, ds, k) for m in a.models], p, h, [mark(m, ds) for m in a.models]))
                head.append(DSN[ds])
            if len(DS[dom]) > 1:   # 데이터셋이 하나면 평균 열 = 그 열이라 뺀다
                cols.append(([mean(m, dom, k) for m in a.models], p, h, None))
                head.append(f"**{DOM[dom]} 평균**")
        print(table(head, a.models, cols) + "\n")

    x = pd.read_csv(a.aux)
    g = lambda grp: (lambda m, ds, k: (lambda y: None if y.empty else float(y.iloc[0]))(x[(x.model == m) & (x.dataset == ds) & (x.group == grp)][k]))
    bnd = lambda m, dom, k: float(np.mean([g("region:boundary")(m, ds, k) for ds in DS["Indoor"]]))
    drop = [float(np.mean([g("region:interior")(m, ds, "d1") - g("region:boundary")(m, ds, "d1") for ds in DS["Indoor"]])) for m in a.models]
    cols = [([mean(m, d, "logrmse", g("all")) for m in a.models], 3, False, None) for d in DS] \
        + [([mean(m, d, "silog", g("all")) for m in a.models], 1, False, None) for d in DS] \
        + [([mean(m, d, "rmse", g("dist:far")) for m in a.models], 2, False, None) for d in DS] \
        + [([bnd(m, "Indoor", "d1") for m in a.models], 3, True, None), (drop, 3, False, None)]
    head = ["모델", f"log-RMSE↓ {dn('Indoor')}", f"log-RMSE↓ {dn('Outdoor')}", f"SILog↓ {dn('Indoor')}", f"SILog↓ {dn('Outdoor')}",
            f"원거리 RMSE↓ {dn('Indoor')} (≥4 m)", f"원거리 RMSE↓ {dn('Outdoor')} (≥30 m)",
            "경계 δ1↑ (실내)", "경계에서 δ1 하락↓"]
    print("#### 보조 지표 (도메인 평균)\n\n" + table(head, a.models, cols))


if __name__ == "__main__":
    main()
