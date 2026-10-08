"""손상 GT 를 '배운' 것인지, 원래 짧게 보는 것인지 (NOTES F-17): 실제 깊이(NYUv2 = Kinect 원측정, iBims-1 = 레이저 GT) 구간별
예측/실제 중앙값 곡선. bfx 결함은 8.19 m(16 비트 상한)에서 시작하므로, 그 자리에서 꺾이면 손상을 배운 것, 그 전부터 서서히 내려가면 압축이다.
출력: results_vlm_weakness/examples/E10_far_curve.png, far_curve.csv"""
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
import weak_common as W, weak_nyu_raw as R

fm.fontManager.addfont("/usr/share/fonts/truetype/nanum/NanumSquareR.ttf")
plt.rcParams["font.family"] = "NanumSquare"
plt.rcParams["axes.unicode_minus"] = False
BINS = np.arange(1.0, 12.5, 0.5)
acc = {}
for ds in ["nyuv2", "ibims1"]:
    for r in W.records(ds):
        if ds == "nyuv2":
            g, gt, err = R.raw_gt(r)
            if err > 0.02:
                continue
        else:
            gt = W.gt_map(r, W.DATA, ds)
        v = gt > 0
        b = np.digitize(gt[v], BINS)
        for m in W.DENSE:
            ratio = np.log(np.maximum(W.load_map(m, ds, r)[v], 1e-3) / gt[v])
            for k in np.unique(b):
                acc.setdefault((ds, m, k), []).append(ratio[b == k][::7])   # 픽셀 1/7 표본
rows = []
for (ds, m, k), xs in acc.items():
    x = np.concatenate(xs)
    if 0 < k < len(BINS) and x.size > 2000:
        rows.append(dict(dataset=ds, model=W.SHORT[m], depth=(BINS[k - 1] + BINS[k]) / 2, ratio=float(np.exp(np.median(x))), n=int(x.size)))
t = pd.DataFrame(rows).sort_values(["dataset", "model", "depth"])
t.to_csv(W.OUT + "/far_curve.csv", index=False)
fig, ax = plt.subplots(1, 2, figsize=(15, 4.6))
COL = {"DepthVLM": "#d62728", "UniDepthV2": "#1f77b4", "Metric3Dv2": "#ff7f0e", "DepthPro": "#2ca02c", "DAv2": "#8c564b"}
for a, ds, ttl in [(ax[0], "nyuv2", "NYUv2 — 실제 깊이 = Kinect 원측정"), (ax[1], "ibims1", "iBims-1 — 실제 깊이 = 레이저 GT")]:
    for m, q in t[t.dataset == ds].groupby("model"):
        a.plot(q.depth, q.ratio, "-o", ms=3, color=COL[m], lw=2.4 if m == "DepthVLM" else 1, label=m)
    a.axhline(1, color="k", lw=0.8)
    if ds == "nyuv2":
        a.axvline(8.19, color="k", ls=":", lw=1)
        a.text(8.25, 0.25, "벤치 GT 상한\n(65535/8000 = 8.19 m)", fontsize=8)
    a.set(xlabel="실제 깊이 (m)", ylabel="예측 / 실제 (중앙값)", title=ttl, ylim=(0.2, 1.6))
    a.legend(fontsize=8)
fig.suptitle("E10 · 깊이별 예측 비율 — DepthVLM 은 손상 GT 를 배운 것인가, 원래 먼 곳을 짧게 보는가", fontsize=12)
fig.tight_layout()
fig.savefig(W.OUT + "/examples/E10_far_curve.png", dpi=120)
p = t.pivot_table(index="depth", columns=["dataset", "model"], values="ratio")
print(p.round(2).loc[4.0:].to_string())
