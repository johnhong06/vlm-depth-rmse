"""NYUv2 원측정 ≥ 8 m 픽셀에서 각 모델이 원측정(raw)과 벤치 GT(bfx) 중 어느 쪽을 따라가나 (NOTES F-17).
bfx 가 원측정의 절반 미만인 픽셀(감김·잘못 채움)만: |ln(예측/bfx)| < |ln(예측/raw)| 인 비율, 이미지별 상관.
출력: results_vlm_weakness/nyu_follow.csv"""
import numpy as np, pandas as pd
import weak_common as W, weak_nyu_raw as R

rows = []
for r in W.records("nyuv2"):
    g, gr, err = R.raw_gt(r)
    if err > 0.02:
        continue
    sel = (gr >= 8) & (g > 0) & (g < gr / 2)
    if sel.sum() < 500:
        continue
    lb, lr = np.log(g[sel]), np.log(gr[sel])
    for m in W.DENSE + W.SUPP:
        lp = np.log(np.maximum(W.load_map(m, "nyuv2", r)[sel], 1e-3))
        rows.append(dict(image_id=r["image"], model=W.SHORT[m], n=int(sel.sum()), closer_to_bfx=float(np.mean(np.abs(lp - lb) < np.abs(lp - lr))),
                         corr_bfx=float(np.corrcoef(lp, lb)[0, 1]) if lb.std() > 0 else np.nan, med_pred=float(np.exp(np.median(lp))),
                         med_bfx=float(np.exp(np.median(lb))), med_raw=float(np.exp(np.median(lr)))))
t = pd.DataFrame(rows)
t.to_csv(W.OUT + "/nyu_follow.csv", index=False)
print("이미지", t.image_id.nunique(), "픽셀", int(t[t.model == "DepthVLM"].n.sum()))
s = t.groupby("model").apply(lambda q: pd.Series(dict(closer_to_bfx=np.average(q.closer_to_bfx, weights=q.n), corr_bfx_med=q.corr_bfx.median(),
                                                      med_pred=np.average(q.med_pred, weights=q.n), med_bfx=np.average(q.med_bfx, weights=q.n),
                                                      med_raw=np.average(q.med_raw, weights=q.n))), include_groups=False)
print(s.round(3).to_string())
