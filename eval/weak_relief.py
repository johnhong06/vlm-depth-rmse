"""벽에 붙은 평평한 물체(액자·칠판·포스터·화이트보드·TV·모니터 등)를 벽에서 튀어나온 것처럼 보나 (NOTES F-16).
물체(대분류 wall_mounted 의 연결 성분, 300 px 이상)마다: 돌출 = 물체 안 log 오차 중앙값 − 둘레 벽(15 px 고리 안 wall 픽셀) log 오차 중앙값.
음수 = 물체를 벽보다 가깝게(튀어나오게) 본다. GT 는 정의상 0. DepthLM 은 공통 점: 이미지마다 물체 점·벽 점의 log 오차 중앙값 차.
출력: results_vlm_weakness/relief_objects.csv, relief_summary.csv
"""
import numpy as np, pandas as pd, cv2
import weak_common as W

WM, WALL = W.CAT_NAMES.index("wall_mounted"), W.CAT_NAMES.index("wall")
rows, prow = [], []
LM = W.depthlm_points()
for ds in ["ibims1", "nyuv2"]:
    for r in W.records(ds):
        g = W.gt_map(r, W.DATA, ds)
        valid = g > 0
        a = W.attributes(ds, r, g)
        cat = a["cat"]
        if not (cat == WM).any():
            continue
        P = {m: W.load_map(m, ds, r) for m in W.DENSE}
        E = {m: np.log(np.clip(p, *W.CAP[ds])) - np.log(np.where(valid, g, 1)) for m, p in P.items()}
        n, cc = cv2.connectedComponents((cat == WM).astype(np.uint8), connectivity=8)
        wall = (cat == WALL) & valid
        for i in range(1, n):
            obj = (cc == i) & valid
            if obj.sum() < 300:
                continue
            ring = (cv2.dilate((cc == i).astype(np.uint8), np.ones((31, 31), np.uint8)) > 0) & wall
            if ring.sum() < 200:
                continue
            rr = dict(dataset=ds, image_id=r["image"], obj=i, n_obj=int(obj.sum()), n_ring=int(ring.sum()),
                      gt_relief=float(np.median(np.log(g[obj])) - np.median(np.log(g[ring]))))
            for m, e in E.items():
                rr[W.SHORT[m]] = 100 * (np.median(e[obj]) - np.median(e[ring]))
            rows.append(rr)
        x = LM[(LM.image_id == r["image"]) & np.isfinite(LM.pred)]
        if len(x):
            u, v = x.u.values.astype(int), x.v.values.astype(int)
            d = np.log(np.clip(x.pred.values, *W.CAP[ds])) - np.log(x.gt_z.values)
            ci = cat[v, u]
            if (ci == WM).sum() >= 2 and (ci == WALL).sum() >= 2:
                pr = dict(dataset=ds, image_id=r["image"], n_obj=int((ci == WM).sum()), n_wall=int((ci == WALL).sum()),
                          DepthLM=100 * (np.median(d[ci == WM]) - np.median(d[ci == WALL])))
                for m, e in E.items():                              # 같은 점에서 dense 모델도
                    pr[W.SHORT[m]] = 100 * (np.median(e[v, u][ci == WM]) - np.median(e[v, u][ci == WALL]))
                prow.append(pr)
t, p = pd.DataFrame(rows), pd.DataFrame(prow)
t.to_csv(W.OUT + "/relief_objects.csv", index=False)
p.to_csv(W.OUT + "/relief_points.csv", index=False)
models = [W.SHORT[m] for m in W.DENSE]
s = []
for ds, g in t.groupby("dataset"):
    for m in models:
        s.append(dict(dataset=ds, kind="픽셀(물체별)", model=m, n=len(g), median=g[m].median(), share_nearer_5pct=(g[m] < -5).mean() * 100))
for ds, g in p.groupby("dataset"):
    for m in ["DepthLM"] + models:
        s.append(dict(dataset=ds, kind="공통 점(이미지별)", model=m, n=len(g), median=g[m].median(), share_nearer_5pct=(g[m] < -5).mean() * 100))
s = pd.DataFrame(s)
s.to_csv(W.OUT + "/relief_summary.csv", index=False)
print(s.round(2).to_string(index=False))
