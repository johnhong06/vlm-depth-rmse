"""DepthVLM 격자 무늬 검증 (NOTES F-16): 예측 log 깊이에서 저주파(가우시안 σ = 6 px)를 뺀 잔차의 행·열 방향 1D 파워 스펙트럼을
이미지마다 평균한다. DepthVLM 은 32 px 토큰을 kernel = stride 인 ConvTranspose 로 키우므로(third_party/DepthVLM/model/dpt_depth_head.py:65–85)
원래 입력 크기에서 주기 32 px → GT 해상도에서 32 × (GT 폭 / 입력 폭) px 에 봉우리가 예상된다. 같은 계산을 GT·pure vision 에도 한다.
입력 폭 = canonical 폭(W·1000/fx)을 28 의 배수로, 다시 32 의 배수로 반올림(qwen_vl_utils → 프로세서).
출력: results_vlm_weakness/examples/grid_spectrum.png, grid_peaks.csv
"""
import numpy as np
import pandas as pd
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
import weak_common as W

fm.fontManager.addfont("/usr/share/fonts/truetype/nanum/NanumSquareR.ttf")
plt.rcParams["font.family"] = "NanumSquare"
plt.rcParams["axes.unicode_minus"] = False
from matplotlib.ticker import FuncFormatter  # noqa: E402
PLAIN = FuncFormatter(lambda v, _: f"{v:g}")   # log 축 눈금을 0.01·0.1·1 처럼 (수식 글꼴의 마이너스 기호 깨짐 방지)


def native_w(W_, fx):
    c = round(W_ * 1000 / fx)
    c = round(c / 28) * 28
    return round(c / 32) * 32


def spec(x, valid, axis):
    """valid 가 모두 참인 행(열)만, 잔차의 1D 파워 스펙트럼 평균."""
    hp = x - cv2.GaussianBlur(x, (0, 0), 6)
    if axis == 1:
        hp, valid = hp.T, valid.T
    rows = hp[valid.all(1)]
    if len(rows) < 5:
        return None
    rows = rows - rows.mean(1, keepdims=True)
    return (np.abs(np.fft.rfft(rows * np.hanning(rows.shape[1]), axis=1)) ** 2).mean(0)


rows, curves = [], {}
for ds in W.SETS:
    recs = W.records(ds)
    acc = {}
    for r in recs[::max(1, len(recs) // 100)]:
        g = W.gt_map(r, W.DATA, ds)
        H, Wd = g.shape
        period = 32 * Wd / native_w(Wd, r["_k"][0])
        for m in ["GT"] + W.DENSE:
            x = np.log(np.maximum(g if m == "GT" else W.load_map(m, ds, r), 1e-3)).astype(np.float32)
            v = (g > 0) if m == "GT" else np.ones_like(g, bool)
            s = spec(x, v, 0)
            if s is not None:
                acc.setdefault(m, []).append(s / s.sum())
    f = np.fft.rfftfreq(Wd)
    curves[ds] = (f, {m: np.mean(v, 0) for m, v in acc.items()}, period)
    for m, s in curves[ds][1].items():
        k = np.argmin(np.abs(f - 1 / period))
        band = (f > 1 / period * 0.8) & (f < 1 / period * 1.25) & (np.abs(f - 1 / period) > 1.5 / Wd)
        rows.append(dict(dataset=ds, model=m, period_px=round(period, 1), peak_ratio=float(s[k - 1:k + 2].max() / np.median(s[band]))))
t = pd.DataFrame(rows)
t.to_csv(W.OUT + "/examples/grid_peaks.csv", index=False)
print(t.pivot_table(index="model", columns="dataset", values="peak_ratio").round(2).to_string())
print(t.groupby("dataset").period_px.first().to_string())
fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
for a, ds in zip(ax, W.SETS):
    f, S, period = curves[ds]
    for m, s in S.items():
        a.semilogy(f[1:], s[1:], lw=2.2 if m == "DepthVLM-4B" else 1, color={"GT": "k", "DepthVLM-4B": "#d62728"}.get(m), alpha=1 if m in ("GT", "DepthVLM-4B") else 0.6,
                   label=W.SHORT.get(m, m))
    for h in [1, 2, 3]:
        a.axvline(h / period, color="#d62728", ls=":", lw=1)
    a.set_title(f"{ds}: 토큰 주기 {period:.1f} px (점선 = 1·2·3 배 주파수)", fontsize=10)
    a.set_xlabel("주파수 (1/px, GT 해상도)")
    a.set_xlim(0, 0.25)
    a.yaxis.set_major_formatter(PLAIN)
ax[0].set_ylabel("정규화 파워 (행 방향, 고주파 잔차)")
ax[0].legend(fontsize=8)
fig.tight_layout()
fig.savefig(W.OUT + "/examples/grid_spectrum.png", dpi=120)
