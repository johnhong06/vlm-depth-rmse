"""NYUv2: Kinect 원측정 픽셀만으로 배율을 정하고 채점 (NOTES F-15 확정용, F-16).
이미지마다 μ = 원측정 픽셀의 평균 log 오차, 모양 = 그 픽셀들의 rms(d − μ). 비교용으로 채운 GT 전체(valid) 기준도 같이.
출력: results_vlm_weakness/nyu_measured_only.csv"""
import numpy as np, pandas as pd
import weak_common as W

rows = []
for r in W.records("nyuv2"):
    g = W.gt_map(r, W.DATA, "nyuv2")
    lab, inst, raw, err = W.nyu_labels(r, g)
    if err > 0.02:
        continue
    meas = (g > 0) & raw
    for m in W.DENSE + W.SUPP:
        p = W.load_map(m, "nyuv2", r)
        d = np.log(np.clip(p, *W.CAP["nyuv2"])) - np.log(np.where(g > 0, g, 1))
        dm, da = d[meas], d[g > 0]
        pm, gm = p[meas], g[meas]
        rows.append(dict(image_id=r["image"], model=W.SHORT[m], fill=1 - meas.sum() / (g > 0).sum(),
                         shape_meas=100 * dm.std(), shape_all=100 * da.std(),
                         absrel_meas=float(np.mean(np.abs(pm - gm) / gm)), d1_meas=float(np.mean(np.maximum(pm / gm, gm / pm) < 1.25))))
t = pd.DataFrame(rows)
t.to_csv(W.OUT + "/nyu_measured_only.csv", index=False)
t["fq"] = t.groupby("model").fill.transform(lambda x: pd.qcut(x, 3, labels=["채움 적음", "중간", "채움 많음"]))
s = t.groupby("model")[["shape_meas", "shape_all", "absrel_meas", "d1_meas"]].mean().round(3)
print("이미지", t.image_id.nunique()); print(s.to_string())
print(t.pivot_table(index="model", columns="fq", values="shape_meas", aggfunc="mean", observed=True).round(2).to_string())
