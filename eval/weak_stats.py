"""VLM 약점 분석 통계 (NOTES F-16) → results_vlm_weakness/data/*.parquet
 ① dense.parquet : 이미지 × 모델(DepthVLM + pure vision 4 + 부록 2) × 집단(weak_common.groups) 충분통계
    n, se, ar, d1 (Track B 와 같은 정의) + dl, dl2 (d = ln 예측 − ln GT) + lg, lg2, dlg (lg = ln GT)
    → 이미지 배율 보정(μ) 뒤 모양 오차, 배율·압축 보정(ln 예측 = a·ln GT + b, 'all' 로 맞춤) 뒤 잔차를 집단마다 다시 계산할 수 있다.
 ② points.parquet : 공통 점(Track A, DepthLM 이 답한 점) × 6 모델 예측 + 점 속성 (dense 모델은 맵의 같은 픽셀 = 규칙 2)
 ③ ibims_struct.parquet : iBims-1 공식 구조 지표 재구현 (Koch et al. 2018 식 1–4)
    DBE ε_acc·ε_comp (예측 깊이를 0–1 정규화 → Canny σ = √2, 문턱 0.1/0.2 주·0.15/0.3·log 정규화 변형, θ = 10 px, GT 자기 점검 행 포함),
    평면성 ε_plan (cm, 이미지 배율을 GT 중앙값 비로 맞춘 뒤 평면 맞춤 잔차의 표준편차)·ε_orie (°, GT 점으로 같은 방식으로 맞춘 평면과의 법선 각)
 ④ edge_profile.parquet : iBims-1 공식 경계에서 경계 수직 방향 ±12 px log 깊이 단면 (흐림 = 계단을 몇 px 에 걸쳐 오르나)
사용: ~/venv/main/bin/python eval/weak_stats.py [--workers 12]
"""
import argparse
import os
from multiprocessing import Pool

import cv2
import numpy as np
import pandas as pd
from scipy import ndimage
from skimage import feature

import weak_common as W

MODELS = W.DENSE + W.SUPP
OFFS = np.arange(-12, 13)


def dense_rows(ds, r, g, valid, P, G):
    lg = np.log(np.where(valid, g, 1.0))
    rows = []
    for m, p in P.items():
        d = W.logerr(np.where(valid, p, 1.0), np.where(valid, g, 1.0), ds)
        for name, sel in G.items():
            s = valid if sel is None else valid & sel
            n = int(s.sum())
            if n == 0:
                continue
            pp, gg, dd, ll = p[s], g[s], d[s], lg[s]
            e = pp - gg
            rows.append(dict(dataset=ds, image_id=r["image"], model=m, group=name, n=n, se=float((e ** 2).sum()),
                             ar=float((np.abs(e) / gg).sum()), d1=int(((pp > 0) & (np.maximum(pp / gg, gg / np.maximum(pp, 1e-12)) < 1.25)).sum()),
                             dl=float(dd.sum()), dl2=float((dd ** 2).sum()), lg=float(ll.sum()), lg2=float((ll ** 2).sum()), dlg=float((dd * ll).sum())))
    return rows


def point_rows(ds, r, g, a, P, lm):
    x = lm[lm.image_id == r["image"]]
    x = x[np.isfinite(x.pred)]
    u, v = x.u.values.astype(int), x.v.values.astype(int)
    t = pd.DataFrame(dict(dataset=ds, image_id=r["image"], u=u, v=v, gt_z=x.gt_z.values, gt_map=g[v, u], p_DepthLM=x.pred.values))
    for m, p in P.items():
        t["p_" + W.SHORT[m]] = p[v, u]
    t["dist"] = a["dist"][v, u]
    t["boundary"] = pd.array(a["boundary"][v, u], dtype="boolean") if a["boundary"] is not None else pd.array([pd.NA] * len(u), dtype="boolean")
    for k in ["cat", "size", "tex", "border", "vpos"]:
        t[k] = a[k][v, u]
    if a.get("nyu_raw") is not None:
        t["nyu_raw"] = a["nyu_raw"][v, u]
    if ds == "ibims1":
        t["plane"], t["plane_id"] = a["plane"][v, u], a["plane_id"][v, u]
    t["inst"] = a["inst"][v, u]
    return t


def backproject(z, k, sel):
    vv, uu = np.nonzero(sel)
    zz = z[vv, uu]
    return np.stack([(uu - k[2]) * zz / k[0], (vv - k[3]) * zz / k[1], zz], 1)


def fit_plane(Pt):
    c = Pt.mean(0)
    n = np.linalg.svd(Pt - c, full_matrices=False)[2][-1]
    return n, (Pt - c) @ n


def dbe(eg, p, log=False, lo=0.1, hi=0.2):
    """iBims-1 DBE (식 3·4): 예측 깊이(또는 log 깊이)를 0–1 로 정규화 → Canny(σ = √2) → θ = 10 px 절단 chamfer.
    ε_acc = 예측 경계에서 GT 경계까지 거리(θ 넘는 예측 경계는 무시), ε_comp = GT 경계에서 예측 경계까지 거리(θ 로 자름)."""
    x = np.log(p.astype(np.float64)) if log else p.astype(np.float64)
    x = (x - x.min()) / max(x.max() - x.min(), 1e-9)
    est = feature.canny(x, sigma=np.sqrt(2), low_threshold=lo, high_threshold=hi)
    Dg, De = ndimage.distance_transform_edt(~eg), ndimage.distance_transform_edt(~est)
    near = est & (Dg < 10)
    return (float(Dg[near].mean()) if near.any() else 10.0), float(np.minimum(De[eg], 10).mean()), int(est.sum())


def ibims_struct(r, g, valid, P, a):
    rows = []
    ep = os.path.join(W.DATA, r["image"].replace("/rgb/", "/edges/"))
    eg = cv2.imread(ep, cv2.IMREAD_UNCHANGED) > 0 if os.path.exists(ep) else None
    k = r["_k"]
    gfill = np.where(valid, g, np.median(g[valid]))          # GT 자기 점검용 (invalid 는 중앙값으로 채움)
    for m, p in {"GT": gfill, **P}.items():
        if eg is not None and eg.any():
            acc, comp, n_est = dbe(eg, p)
            acc_l, comp_l, _ = dbe(eg, p, log=True)
            acc_h, comp_h, _ = dbe(eg, p, lo=0.15, hi=0.3)
            rows.append(dict(image_id=r["image"], model=m, kind="dbe", acc=acc, comp=comp, acc_log=acc_l, comp_log=comp_l,
                             acc_hi=acc_h, comp_hi=comp_h, n_est=n_est, n_gt=int(eg.sum())))
        if m == "GT":
            continue
        s = np.median(g[valid] / p[valid])                       # 논문: 예측을 GT 에 배율 맞춤한 뒤 평면성
        for pid, (kind, _, _) in a["plane_par"].items():
            sel = valid & (a["plane_id"] == pid)
            if sel.sum() < 200:
                continue
            ng, dg = fit_plane(backproject(g, k, sel))           # GT 평면도 같은 방식으로 맞춘다 (txt 의 법선은 좌표계가 달라 쓰지 않음)
            npd, dist = fit_plane(backproject(p * s, k, sel))
            rows.append(dict(image_id=r["image"], model=m, kind="plane", plane=kind, plane_id=pid, n=int(sel.sum()),
                             plan_cm=float(dist.std() * 100), gt_plan_cm=float(dg.std() * 100),
                             orie_deg=float(np.degrees(np.arccos(min(1.0, abs(npd @ ng)))))))
    return rows


def edge_profile(r, g, valid, P):
    """공식 경계 픽셀마다 GT log 깊이 기울기 방향(먼 쪽 +)으로 ±12 px 단면을 뽑아, 양끝이 valid 이고 계단 > ln 1.15 인 것만 평균."""
    ep = os.path.join(W.DATA, r["image"].replace("/rgb/", "/edges/"))
    if not os.path.exists(ep):
        return []
    eg = cv2.imread(ep, cv2.IMREAD_UNCHANGED) > 0
    lgm = np.log(np.where(valid, g, np.nan)).astype(np.float32)
    fill = cv2.GaussianBlur(np.nan_to_num(lgm, nan=float(np.nanmedian(lgm))), (0, 0), 2)
    gx, gy = cv2.Sobel(fill, cv2.CV_32F, 1, 0, ksize=5), cv2.Sobel(fill, cv2.CV_32F, 0, 1, ksize=5)
    vv, uu = np.nonzero(eg)
    nrm = np.hypot(gx[vv, uu], gy[vv, uu])
    keep = nrm > 1e-6
    vv, uu = vv[keep], uu[keep]
    nx, ny = gx[vv, uu] / nrm[keep], gy[vv, uu] / nrm[keep]
    X = (uu[:, None] + OFFS[None] * nx[:, None]).astype(np.float32)
    Y = (vv[:, None] + OFFS[None] * ny[:, None]).astype(np.float32)
    vm = cv2.remap(valid.astype(np.float32), X, Y, cv2.INTER_LINEAR)          # 이미지 밖 = 0
    Lg = cv2.remap(np.log(np.where(valid, g, 1.0)).astype(np.float32), X, Y, cv2.INTER_LINEAR)
    ok = (vm[:, 0] > 0.999) & (vm[:, -1] > 0.999) & (Lg[:, -1] - Lg[:, 0] > np.log(1.15))
    if not ok.any():
        return []
    rows = []
    base = Lg[ok, -1] - Lg[ok, 0]
    prof = {"GT": Lg[ok]}
    for m, p in P.items():
        prof[m] = cv2.remap(np.log(np.maximum(p, 1e-3)).astype(np.float32), X, Y, cv2.INTER_LINEAR)[ok]
    for m, L in prof.items():
        z = (L - L[:, :1]) / base[:, None]                          # GT 계단 크기로 정규화: GT 는 0 → 1
        rows.append(dict(image_id=r["image"], model=m, n_edge=int(ok.sum()), **{f"t{t:+d}": float(np.nanmean(z[:, j])) for j, t in enumerate(OFFS)}))
    return rows


def work(job):
    ds, r = job
    g = W.gt_map(r, W.DATA, ds)
    P = {m: W.load_map(m, ds, r) for m in MODELS}
    valid = g > 0
    for p in P.values():
        valid &= np.isfinite(p)
    a = W.attributes(ds, r, g)
    out = {"dense": dense_rows(ds, r, g, valid, P, W.groups(ds, a)),
           "points": point_rows(ds, r, g, a, {m: P[m] for m in W.DENSE}, LM),
           "meta": [dict(dataset=ds, image_id=r["image"], H=g.shape[0], W=g.shape[1], n_valid=int(valid.sum()),
                         nyu_align=a.get("nyu_align", np.nan))]}
    if ds == "ibims1":
        out["ibims"] = ibims_struct(r, g, valid, P, a)
        out["edge"] = edge_profile(r, g, valid, P)
    return out


def init():
    global LM
    LM = W.depthlm_points()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    jobs = [(ds, r) for ds in W.SETS for r in (W.records(ds)[: a.limit] if a.limit else W.records(ds))]
    res = {"dense": [], "points": [], "meta": [], "ibims": [], "edge": []}
    with Pool(a.workers, initializer=init) as pool:
        for i, o in enumerate(pool.imap_unordered(work, jobs, chunksize=2)):
            for k, v in o.items():
                res[k].append(pd.DataFrame(v) if isinstance(v, list) else v)
            if i % 100 == 0:
                print(f"{i}/{len(jobs)}", flush=True)
    os.makedirs(os.path.join(W.OUT, "data"), exist_ok=True)
    for k, name in [("dense", "dense"), ("points", "points"), ("meta", "meta"), ("ibims", "ibims_struct"), ("edge", "edge_profile")]:
        t = pd.concat([x for x in res[k] if len(x)], ignore_index=True)
        t.to_parquet(os.path.join(W.OUT, "data", f"{name}.parquet"), index=False)
        print(name, t.shape, flush=True)
