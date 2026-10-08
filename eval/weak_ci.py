"""공통 약점 주장의 이미지 단위 bootstrap 95 % CI (B = 1,000, seed 0, 모든 모델 같은 재표본) — NOTES F-16.
 ① 원거리 모양 오차 비 (VLM ÷ pure vision 4 종 중앙값), 공통 점
 ② 평면 평탄도 비 (iBims-1 공식 평면, 공통 점, 평면 ε_plan 중앙값 ÷ pure vision 중앙값)
 ③ 깊이 차 10–25 % 쌍 앞뒤 오답률 차 (VLM − pure vision 중앙값, %p)
 ④ 압축 기울기 차 (VLM − pure vision 중앙값, 이미지별 a 의 중앙값)
출력: results_vlm_weakness/ci_common.csv
--nyuraw: ①③④ 를 NYUv2 원측정 GT 공통 점(weak_nyu_raw.py 출력)으로 → ci_common_nyuraw.csv (NOTES F-18)"""
import sys
import numpy as np, pandas as pd
import weak_common as W

NYURAW = "--nyuraw" in sys.argv

rng = np.random.default_rng(0)
B = 1000
PTS = pd.read_parquet(W.OUT + "/data/points.parquet")
if NYURAW:   # cap 은 nyuv2 것 그대로, 표의 dataset 이름만 nyuv2_raw
    PTS = pd.read_parquet(W.OUT + "/data/points_nyuraw.parquet").assign(dataset="nyuv2")
M = ["DepthLM", "DepthVLM", "UniDepthV2", "Metric3Dv2", "DepthPro", "DAv2"]
PV = M[2:]


def boot(keys, fn):
    keys = np.asarray(keys)
    est = fn(keys)
    bs = np.array([fn(keys[rng.integers(0, len(keys), len(keys))]) for _ in range(B)])
    return est, np.percentile(bs, 2.5, 0), np.percentile(bs, 97.5, 0)


rows = []
for ds in (["nyuv2"] if NYURAW else W.SETS):
    x = PTS[PTS.dataset == ds].copy()
    for m in M:
        d = np.log(np.clip(x["p_" + m], *W.CAP[ds])) - np.log(x.gt_z)
        x["r_" + m] = d - d.groupby(x.image_id).transform("mean")
    far = x[x.dist == 2]
    # 이미지별 원거리 제곱 잔차 합·개수
    S = far.groupby("image_id").agg(**{f"s_{m}": (f"r_{m}", lambda v: (v ** 2).sum()) for m in M}, n=("u", "size"))
    imgs = S.index.values
    def f1(ix):
        s = S.loc[ix].sum()
        sh = {m: np.sqrt(s[f"s_{m}"] / s.n) for m in M}
        pv = np.median([sh[m] for m in PV])
        return np.array([sh["DepthLM"] / pv, sh["DepthVLM"] / pv])
    e, lo, hi = boot(imgs, f1)
    for k, m in enumerate(["DepthLM", "DepthVLM"]):
        rows.append(dict(claim="① 원거리 모양 오차 비", dataset=ds, model=m, est=e[k], lo=lo[k], hi=hi[k]))
    # ③ 쌍
    pr = []
    for img, q in x.groupby("image_id"):
        i, j = np.triu_indices(len(q), 1)
        g = q.gt_z.values
        lr = np.log(g[i] / g[j])
        ok = (np.abs(lr) > np.log(1.10)) & (np.abs(lr) < np.log(1.25))
        if ok.sum() == 0:
            continue
        rr = dict(image_id=img, n=ok.sum())
        for m in M:
            p = np.clip(q["p_" + m].values, *W.CAP[ds])
            rr[m] = (np.sign(np.log(p[i] / p[j]))[ok] != np.sign(lr[ok])).sum()
        pr.append(rr)
    P = pd.DataFrame(pr).set_index("image_id")
    def f3(ix):
        s = P.loc[ix].sum()
        w = {m: 100 * s[m] / s.n for m in M}
        pv = np.median([w[m] for m in PV])
        return np.array([w["DepthLM"] - pv, w["DepthVLM"] - pv])
    e, lo, hi = boot(P.index.values, f3)
    for k, m in enumerate(["DepthLM", "DepthVLM"]):
        rows.append(dict(claim="③ 깊이 차 10–25 % 쌍 앞뒤 오답률 차 (%p)", dataset=ds, model=m, est=e[k], lo=lo[k], hi=hi[k]))
    # ④ 압축 기울기
    A = pd.DataFrame({img: {m: np.polyfit(np.log(q.gt_z), np.log(np.clip(q["p_" + m], *W.CAP[ds])), 1)[0] for m in M}
                      for img, q in x.groupby("image_id") if len(q) >= 8 and np.ptp(np.log(q.gt_z)) > 0.2}).T
    def f4(ix):
        med = A.loc[ix].median()
        pv = np.median([med[m] for m in PV])
        return np.array([med["DepthLM"] - pv, med["DepthVLM"] - pv])
    e, lo, hi = boot(A.index.values, f4)
    for k, m in enumerate(["DepthLM", "DepthVLM"]):
        rows.append(dict(claim="④ 압축 기울기 a 차 (VLM − PV 중앙값)", dataset=ds, model=m, est=e[k], lo=lo[k], hi=hi[k]))
if NYURAW:
    t = pd.DataFrame(rows).assign(dataset="nyuv2_raw")
    t.to_csv(W.OUT + "/ci_common_nyuraw.csv", index=False)
    print(t.round(3).to_string(index=False))
    sys.exit()
# ② 평면 (iBims-1)
sp = pd.read_csv(W.OUT + "/sparse_planes.csv")
T = sp.pivot_table(index=["image_id", "plane"], columns="model", values="plan_cm").reset_index()
imgs = T.image_id.unique()
def f2(ix):
    q = pd.concat([T[T.image_id == i] for i in ix])
    med = q[M].median()
    pv = np.median([med[m] for m in PV])
    return np.array([med["DepthLM"] / pv, med["DepthVLM"] / pv])
e, lo, hi = boot(imgs, f2)
for k, m in enumerate(["DepthLM", "DepthVLM"]):
    rows.append(dict(claim="② 평면 평탄도 비 (공통 점)", dataset="ibims1", model=m, est=e[k], lo=lo[k], hi=hi[k]))
t = pd.DataFrame(rows)
t.to_csv(W.OUT + "/ci_common.csv", index=False)
print(t.round(3).to_string(index=False))
