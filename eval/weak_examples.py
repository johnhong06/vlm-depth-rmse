"""VLM 약점 분석 — 약점마다 예시 이미지 한 장 + 근거 그래프 (NOTES F-16) → results_vlm_weakness/examples/
  E1 평면을 평평하게 못 그림 (공통)        E2 거리 압축: 먼 곳 가깝게·가까운 곳 멀게 (공통)
  E3 깊이 차가 작은 두 물체의 앞뒤 (공통)  E4 경계 흐림 (DepthVLM, DepthLM 은 점 단위로만)
  E5 토큰 격자 무늬 (DepthVLM 구조)        E6 답 재사용 계단 (DepthLM 구조)
  E7 학습 밖 실외의 깊은 장면 배율 붕괴    E8 NYUv2 채운 GT 가 VLM 약점을 가림
사용: ~/venv/main/bin/python eval/weak_examples.py
"""
import os

import cv2
import matplotlib
import numpy as np
import pandas as pd
from PIL import Image

import weak_common as W
from weak_figs import depth_rgb, dots, err_rgb, label, tile

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager as fm  # noqa: E402

fm.fontManager.addfont("/usr/share/fonts/truetype/nanum/NanumSquareR.ttf")
plt.rcParams["font.family"] = "NanumSquare"
plt.rcParams["axes.unicode_minus"] = False
from matplotlib.ticker import FuncFormatter  # noqa: E402
PLAIN = FuncFormatter(lambda v, _: f"{v:g}")   # log 축 눈금을 0.01·0.1·1 처럼 (수식 글꼴의 마이너스 기호 깨짐 방지)
EX = os.path.join(W.OUT, "examples")
COL = {"DepthLM": "#9467bd", "DepthVLM": "#d62728", "UniDepthV2": "#1f77b4", "Metric3Dv2": "#ff7f0e", "DepthPro": "#2ca02c", "DAv2": "#8c564b", "GT": "k"}
PVS = [W.SHORT[m] for m in W.PV]
M6 = ["DepthLM", "DepthVLM"] + PVS
PTS = pd.read_parquet(os.path.join(W.OUT, "data", "points.parquet"))
LM = W.depthlm_points()
REC = {ds: {r["image"]: r for r in W.records(ds)} for ds in W.SETS}


def stack(tiles_rows):
    rows = [np.hstack(t) for t in tiles_rows]
    w = max(x.shape[1] for x in rows)
    return np.vstack([np.pad(x, ((0, 4), (0, w - x.shape[1]), (0, 0)), constant_values=255) for x in rows])


def fig_to_img(fig):
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return img


def save(name, img_top, fig=None):
    """위: 이미지 타일, 아래: matplotlib 그래프(같은 폭으로)."""
    out = img_top
    if fig is not None:
        g = fig_to_img(fig)
        g = cv2.resize(g, (out.shape[1], int(g.shape[0] * out.shape[1] / g.shape[1])), interpolation=cv2.INTER_AREA)
        out = np.vstack([out, np.full((6, out.shape[1], 3), 255, np.uint8), g])
    Image.fromarray(out).save(os.path.join(EX, name), quality=90)
    print("saved", name, flush=True)


def load(ds, image):
    r = REC[ds][image]
    g = W.gt_map(r, W.DATA, ds)
    P = {W.SHORT[m]: W.load_map(m, ds, r) for m in W.DENSE}
    valid = g > 0
    rgb = cv2.cvtColor(cv2.imread(os.path.join(W.DATA, r["image"])), cv2.COLOR_BGR2RGB)
    return r, g, P, valid, rgb


def lm_pts(image, ds):
    x = LM[(LM.image_id == image) & np.isfinite(LM.pred)]
    d = np.log(np.clip(x.pred.values, *W.CAP[ds])) - np.log(x.gt_z.values)
    return x.u.values.astype(int), x.v.values.astype(int), x.pred.values, x.gt_z.values, d


def shape_err(p, g, valid, ds):
    e = np.zeros_like(g)
    e[valid] = np.log(np.clip(p[valid], *W.CAP[ds])) - np.log(g[valid])
    e[valid] -= e[valid].mean()
    return e


# ---------------- E1 평면 ----------------
def e1():
    ds, image = "ibims1", "ibims1/ibims1_core_raw/rgb/kitchen_07.png"
    r, g, P, valid, rgb = load(ds, image)
    kind, pid, _ = W.ibims_planes(r, g.shape)
    ids, cnt = np.unique(pid[(kind == "wall") & valid], return_counts=True)
    plane = (pid == ids[cnt.argmax()]) & valid
    row = int(np.argmax(plane.sum(1)))
    H, Wd = g.shape
    th = int(round(H * 300 / Wd))
    lo, hi = np.log(np.percentile(g[valid], [1, 99]))
    hl = rgb.copy()
    hl[plane] = (0.5 * hl[plane] + 0.5 * np.array([255, 230, 0])).astype(np.uint8)
    cv2.line(hl, (0, row), (Wd - 1, row), (255, 0, 0), 2)
    u, v, pp, gg, d = lm_pts(image, ds)
    inp = plane[v, u]
    base = (cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)[..., None].repeat(3, 2) // 3).astype(np.uint8)
    t_lm = dots(tile(base, th), u[inp], v[inp], err_rgb(d[inp] - d.mean(), np.ones(int(inp.sum()), bool)), 300 / Wd, th / H, 5)
    tiles = [label(tile(hl, th), ["kitchen_07: 노랑 = 공식 벽 평면", "빨간 선 = 아래 그래프의 행"], 13),
             label(tile(depth_rgb(g, lo, hi, valid), th), ["GT (매끈한 평면)"], 14)]
    for m in ["DepthVLM", "UniDepthV2"]:
        tiles.append(label(tile(depth_rgb(P[m], lo, hi), th), [f"{m} 예측"], 14))
    for m in ["DepthVLM", "UniDepthV2"]:
        tiles.append(label(tile(err_rgb(shape_err(P[m], g, valid, ds), valid), th), [f"{m} 배율 뺀 오차"], 14))
    tiles.append(label(t_lm, [f"DepthLM: 벽 위 {int(inp.sum())}점 배율 뺀 오차"], 13))
    fig, ax = plt.subplots(1, 3, figsize=(21, 4.6))
    cols = np.nonzero(plane[row])[0]
    for m in ["DepthVLM"] + PVS:
        e = shape_err(P[m], g, valid, ds)
        ax[0].plot(cols, 100 * np.expm1(e[row, cols]), color=COL[m], lw=2.2 if m == "DepthVLM" else 1, label=m)
    ax[0].scatter(u[inp], 100 * np.expm1(d[inp] - d.mean()), color=COL["DepthLM"], s=30, zorder=5, label="DepthLM (벽 위 모든 점)")
    ax[0].axhline(0, color="k", lw=0.8)
    ax[0].set(xlabel="열 u (px)", ylabel="배율 뺀 오차 (%)", title="벽 한 줄을 따라: 0 이면 평면을 정확히 그린 것")
    ax[0].legend(fontsize=8, ncol=2)
    sp = pd.read_csv(os.path.join(W.OUT, "sparse_planes.csv"))
    ib = pd.read_parquet(os.path.join(W.OUT, "data", "ibims_struct.parquet"))
    dn = ib[ib.kind == "plane"].assign(model=lambda x: x.model.map(W.SHORT)).groupby("model")[["plan_cm", "orie_deg"]].median()
    sm = sp.groupby("model")[["plan_cm", "orie_deg"]].median()
    for a, key, ttl in [(ax[1], "plan_cm", "평면 평탄도 ε_plan (cm, 중앙값)"), (ax[2], "orie_deg", "평면 방향 오차 ε_orie (°, 중앙값)")]:
        xs = np.arange(len(M6))
        a.bar(xs - 0.2, [sm.loc[m, key] for m in M6], 0.4, color=[COL[m] for m in M6], alpha=0.55, label="공통 점 (평면 126 개, 평면당 ≥ 8 점)")
        a.bar(xs + 0.2, [dn.loc[m, key] if m in dn.index else np.nan for m in M6], 0.4, color=[COL[m] for m in M6], label="전체 픽셀 (평면 244 개)")
        a.set_xticks(xs, M6, fontsize=9)
        a.set_title(ttl)
        a.legend(fontsize=8)
    fig.suptitle("E1 · 평면을 평평하게 못 그린다 — iBims-1 공식 평면(바닥·벽·책상), 이미지 배율을 GT 에 맞춘 뒤", fontsize=12)
    fig.tight_layout()
    save("E1_planes.jpg", stack([tiles[:4], tiles[4:]]), fig)


# ---------------- E2 거리 압축 ----------------
def ratio_curve(ds, m, nb=9):
    x = PTS[PTS.dataset == ds].copy()
    d = np.log(np.clip(x["p_" + m], *W.CAP[ds])) - np.log(x.gt_z)
    x["e"] = d - d.groupby(x.image_id).transform("mean")
    edges = np.quantile(np.log(x.gt_z), np.linspace(0, 1, nb + 1))
    b = np.clip(np.digitize(np.log(x.gt_z), edges[1:-1]), 0, nb - 1)
    return np.exp(x.groupby(b).apply(lambda q: np.log(q.gt_z).median(), include_groups=False)), x.groupby(b).e.median()


def e2():
    fig, ax = plt.subplots(1, 4, figsize=(22, 4.6))
    for a, ds in zip(ax[:3], W.SETS):
        for m in M6:
            xm, ym = ratio_curve(ds, m)
            a.plot(xm, 100 * np.expm1(ym), "-o", ms=3, color=COL[m], lw=2.4 if m in ("DepthLM", "DepthVLM") else 1, label=m)
        a.axhline(0, color="k", lw=0.8)
        a.set_xscale("log")
        a.xaxis.set_major_formatter(PLAIN)
        a.xaxis.set_minor_formatter(PLAIN)
        a.set(xlabel="GT 깊이 (m, log)", ylabel="이미지 배율 뺀 치우침 (%)", title=f"{ds}: 공통 점, 깊이 9 등분 중앙값")
    ax[0].legend(fontsize=8)
    c = pd.read_csv(os.path.join(W.OUT, "compression_slope.csv"), index_col=0)
    xs = np.arange(3)
    for k, m in enumerate(M6):
        ax[3].bar(xs + (k - 2.5) * 0.13, [c.loc["pt_" + m, ds] for ds in W.SETS], 0.13, color=COL[m], label=m)
    ax[3].axhline(1, color="k", lw=0.8)
    ax[3].set_ylim(0.5, 1.05)
    ax[3].set_xticks(xs, W.SETS)
    ax[3].set_title("이미지마다 ln 예측 = a·ln GT + b 의 a (중앙값)\n1 = 거리 차를 그대로, 작을수록 압축")
    ax[3].legend(fontsize=7, ncol=2)
    fig.suptitle("E2 · 거리 압축 — 가까운 곳은 멀게, 먼 곳은 가깝게 (배율을 뺀 뒤에도 남는 기울기)", fontsize=12)
    fig.tight_layout()
    # 예시: iBims-1 에서 두 VLM 기울기가 pure vision 보다 가장 많이 낮은 이미지
    x = PTS[PTS.dataset == "ibims1"]
    sl = x.groupby("image_id").apply(lambda q: pd.Series({m: np.polyfit(np.log(q.gt_z), np.log(q["p_" + m]), 1)[0] for m in M6}), include_groups=False)
    sl["gap"] = sl[PVS].median(axis=1) - sl[["DepthLM", "DepthVLM"]].max(axis=1)
    image = sl.gap.idxmax()
    r, g, P, valid, rgb = load("ibims1", image)
    H, Wd = g.shape
    th = int(round(H * 300 / Wd))
    lo, hi = np.log(np.percentile(g[valid], [1, 99]))
    u, v, pp, gg, d = lm_pts(image, "ibims1")
    base = (cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)[..., None].repeat(3, 2) // 3).astype(np.uint8)
    best = min(PVS, key=lambda m: abs(1 - sl.loc[image, m]))
    tiles = [label(tile(rgb, th), [os.path.basename(image), f"기울기 a: DepthLM {sl.loc[image, 'DepthLM']:.2f} · DepthVLM {sl.loc[image, 'DepthVLM']:.2f}",
                                   f"{best} {sl.loc[image, best]:.2f}"], 13),
             label(tile(depth_rgb(g, lo, hi, valid), th), ["GT"], 14),
             label(tile(err_rgb(shape_err(P["DepthVLM"], g, valid, "ibims1"), valid), th), ["DepthVLM 배율 뺀 오차", "앞 빨강(멀게) → 뒤 파랑(가깝게)"], 13),
             label(tile(err_rgb(shape_err(P[best], g, valid, "ibims1"), valid), th), [f"{best} 배율 뺀 오차"], 13),
             label(dots(tile(base, th), u, v, err_rgb(d - d.mean(), np.ones(len(d), bool)), 300 / Wd, th / H, 5), ["DepthLM 점 배율 뺀 오차"], 13)]
    save("E2_far_compression.jpg", stack([tiles]), fig)


# ---------------- E3 앞뒤 순서 ----------------
def e3():
    o = pd.read_csv(os.path.join(W.OUT, "pairs_order_wrong.csv"), index_col=[0, 1])
    rel = pd.read_csv(os.path.join(W.OUT, "pairs_relerr.csv"), index_col=[0, 1])
    fig, ax = plt.subplots(2, 1, figsize=(13, 9))
    labels, xs = [], 0
    for ds in W.SETS:
        for gap in ["10–25%", "25–100%", ">2배"]:
            for k, m in enumerate(M6):
                ax[0].bar(xs + (k - 2.5) * 0.13, o.loc[(ds, gap), m], 0.13, color=COL[m], label=m if xs == 0 else None)
            labels.append(f"{ds}\n깊이 차 {gap}")
            xs += 1
    ax[0].set_xticks(range(xs), labels, fontsize=7.5)
    ax[0].set(ylabel="앞뒤를 거꾸로 답한 쌍 (%)", title="같은 이미지의 공통 점 쌍: 앞뒤 순서 오답률")
    ax[0].legend(fontsize=8)
    labels, xs = [], 0
    for ds in W.SETS:
        for pb in ["<50px", "50–150", "150–400", ">400"]:
            for k, m in enumerate(M6):
                ax[1].bar(xs + (k - 2.5) * 0.13, rel.loc[(ds, pb), m], 0.13, color=COL[m])
            labels.append(f"{ds}\n{pb}")
            xs += 1
    ax[1].set_xticks(range(xs), labels, fontsize=7)
    ax[1].set(ylabel="상대 깊이 오차 중앙값 (%)", title="두 점의 깊이 비율 오차 — 점 사이 화면 거리별")
    fig.suptitle("E3 · 깊이 차가 작은 두 곳의 앞뒤·비율을 틀린다 (배율과 무관한 비교)", fontsize=12)
    fig.tight_layout()
    # 예시 쌍: iBims-1, 깊이 차 10–25 %, 두 VLM 모두 거꾸로·pure vision 4 종 모두 맞음, 화면 거리 ≥ 120 px
    best = None
    for img, x in PTS[PTS.dataset == "ibims1"].groupby("image_id"):
        i, j = np.triu_indices(len(x), 1)
        g = x.gt_z.values
        lr = np.log(g[i] / g[j])
        ok = (np.abs(lr) > np.log(1.10)) & (np.abs(lr) < np.log(1.25)) & (np.hypot(x.u.values[i] - x.u.values[j], x.v.values[i] - x.v.values[j]) > 120)
        for m in M6:
            p = x["p_" + m].values
            wrong = np.sign(np.log(p[i] / p[j])) != np.sign(lr)
            ok &= wrong if m in ("DepthLM", "DepthVLM") else ~wrong
        if ok.any():
            k = np.nonzero(ok)[0][0]
            best = (img, x.iloc[i[k]], x.iloc[j[k]])
            break
    img, A, B = best
    r, g, P, valid, rgb = load("ibims1", img)
    vis = cv2.resize(rgb, (rgb.shape[1] * 2, rgb.shape[0] * 2))
    for nm, q in [("A", A), ("B", B)]:
        c = (int(q.u) * 2, int(q.v) * 2)
        cv2.circle(vis, c, 16, (0, 0, 0), 7)
        cv2.circle(vis, c, 16, (255, 255, 0), 4)
        cv2.putText(vis, nm, (c[0] + 20, c[1] - 14), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 0), 7)
        cv2.putText(vis, nm, (c[0] + 20, c[1] - 14), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (255, 255, 0), 3)
    near = "A" if A.gt_z < B.gt_z else "B"
    lines = [f"{os.path.basename(img)} — 실제: A {A.gt_z:.2f} m, B {B.gt_z:.2f} m → {near} 가 가까움 (깊이 차 {abs(np.log(A.gt_z / B.gt_z)) * 100:.0f} %)"]
    for m in M6:
        pa, pb = A["p_" + m], B["p_" + m]
        ans = "A" if pa < pb else "B"
        lines.append(f"{m:10s}  A {pa:5.2f} m · B {pb:5.2f} m → {ans} 가 가깝다고 답함  {'O' if ans == near else 'X (거꾸로)'}")
    txt = label(np.full((26 * len(lines) + 10, vis.shape[1], 3), 255, np.uint8), lines, 20)
    save("E3_fine_order.jpg", np.vstack([vis, txt]), fig)


# ---------------- E4 경계 흐림 ----------------
def e4():
    ib = pd.read_parquet(os.path.join(W.OUT, "data", "ibims_struct.parquet"))
    dbe = ib[ib.kind == "dbe"]
    piv = dbe.pivot_table(index="image_id", columns="model", values="comp")
    image = (piv["DepthVLM-4B"] - piv[W.PV].median(axis=1)).idxmax()
    r, g, P, valid, rgb = load("ibims1", image)
    eg = cv2.imread(os.path.join(W.DATA, image.replace("/rgb/", "/edges/")), cv2.IMREAD_UNCHANGED) > 0
    # 경계가 가장 많은 200×150 창
    k = cv2.boxFilter(eg.astype(np.float32), -1, (200, 150), normalize=False)
    cy, cx = np.unravel_index(np.argmax(k), k.shape)
    y0, x0 = int(np.clip(cy - 75, 0, g.shape[0] - 150)), int(np.clip(cx - 100, 0, g.shape[1] - 200))
    crop = lambda im: cv2.resize(im[y0:y0 + 150, x0:x0 + 200], (400, 300), interpolation=cv2.INTER_NEAREST)
    lo, hi = np.log(np.percentile(g[valid], [1, 99]))
    tiles = [label(crop(rgb), [os.path.basename(image) + " (확대)"], 13), label(crop(depth_rgb(g, lo, hi, valid)), ["GT"], 14)]
    for m in ["DepthVLM", "Metric3Dv2", "UniDepthV2"]:
        tiles.append(label(crop(depth_rgb(P[m], lo, hi)), [m], 14))
    e = pd.read_parquet(os.path.join(W.OUT, "data", "edge_profile.parquet"))
    cols = [c for c in e.columns if c.startswith("t")]
    fig, ax = plt.subplots(1, 3, figsize=(21, 4.6))
    t = np.arange(-12, 13)
    for m, q in e.groupby("model"):
        if m in W.SUPP:
            continue
        y = np.array([np.average(q[c], weights=q.n_edge) for c in cols])
        nm = W.SHORT.get(m, m)
        ax[0].plot(t, y, "-o", ms=3, color=COL[nm], lw=2.4 if nm in ("GT", "DepthVLM") else 1, label=nm)
    ax[0].set(xlabel="경계에서 수직 거리 (px, + = 먼 쪽)", ylabel="GT 계단 크기로 정규화한 log 깊이", title="iBims-1 공식 경계 117,494 px 평균 단면")
    ax[0].legend(fontsize=8)
    d = dbe.assign(model=dbe.model.map(lambda m: W.SHORT.get(m, m))).groupby("model")[["acc", "comp"]].mean()
    order = ["GT"] + [m for m in M6 if m in d.index]
    xs = np.arange(len(order))
    ax[1].bar(xs - 0.2, d.loc[order, "acc"], 0.4, color=[COL[m] for m in order], alpha=0.55, label="ε_acc (예측 경계 → GT 경계)")
    ax[1].bar(xs + 0.2, d.loc[order, "comp"], 0.4, color=[COL[m] for m in order], label="ε_comp (GT 경계 → 예측 경계)")
    ax[1].set_xticks(xs, order)
    ax[1].set(ylabel="px (낮을수록 좋음)", title="iBims-1 DBE (86 장, θ = 10 px; GT = 같은 절차의 자기 점검)")
    ax[1].legend(fontsize=8)
    pg = pd.read_csv(os.path.join(W.OUT, "points_groups.csv"))
    x = pg[(pg.dataset == "ibims1") & pg.group.isin(["region:boundary", "region:interior"])].pivot_table(index="model", columns="group", values="shape").loc[M6]
    xs = np.arange(len(M6))
    ax[2].bar(xs - 0.2, x["region:interior"], 0.4, color=[COL[m] for m in M6], alpha=0.55, label="경계 밖")
    ax[2].bar(xs + 0.2, x["region:boundary"], 0.4, color=[COL[m] for m in M6], label="경계 3 px 안")
    ax[2].set_xticks(xs, M6)
    ax[2].set(ylabel="배율 뺀 오차 (%)", title="공통 점: 경계 안/밖 (DepthLM 포함)")
    ax[2].legend(fontsize=8)
    fig.suptitle("E4 · 경계가 15 px 에 걸쳐 번진다 (DepthVLM) — 깊이 계단 10→90 % 폭: GT 1.8 · pure vision 4.2–5.1 · DepthVLM 15.0 px", fontsize=12)
    fig.tight_layout()
    save("E4_boundary_blur.jpg", stack([tiles]), fig)


# ---------------- E5 격자 ----------------
def e5():
    ds, image = "ibims1", "ibims1/ibims1_core_raw/rgb/kitchen_07.png"
    r, g, P, valid, rgb = load(ds, image)
    hp = lambda x: np.log(np.maximum(x, 1e-3)) - cv2.GaussianBlur(np.log(np.maximum(x, 1e-3)), (0, 0), 6)
    y0, x0, hh, ww = 60, 40, 200, 260
    def vis(x, lim=0.02):
        z = np.clip(hp(x)[y0:y0 + hh, x0:x0 + ww] / lim, -1, 1)
        return cv2.resize((np.stack([z, z, z], 2) * 127 + 128).astype(np.uint8), (520, 400), interpolation=cv2.INTER_NEAREST)
    tiles = [label(cv2.resize(rgb[y0:y0 + hh, x0:x0 + ww], (520, 400)), ["kitchen_07 벽 확대"], 14),
             label(vis(np.where(valid, g, np.median(g[valid]))), ["GT 고주파 성분 (±2 %)"], 14),
             label(vis(P["DepthVLM"]), ["DepthVLM 고주파 성분 — 격자"], 14),
             label(vis(P["UniDepthV2"]), ["UniDepthV2 고주파 성분"], 14)]
    sp = Image.open(os.path.join(EX, "grid_spectrum.png"))
    top = stack([tiles])
    sp = np.asarray(sp.convert("RGB"))
    sp = cv2.resize(sp, (top.shape[1], int(sp.shape[0] * top.shape[1] / sp.shape[1])))
    Image.fromarray(np.vstack([top, sp])).save(os.path.join(EX, "E5_token_grid.jpg"), quality=90)
    print("saved E5_token_grid.jpg")


# ---------------- E6 DepthLM 계단 ----------------
def e6():
    ds, image = "ibims1", "ibims1/ibims1_core_raw/rgb/corridor_01.png"
    x = PTS[PTS.image_id == image]
    r, g, P, valid, rgb = load(ds, image)
    fig, ax = plt.subplots(1, 3, figsize=(20, 4.8))
    for a, m in zip(ax, ["DepthLM", "DepthVLM", "UniDepthV2"]):
        a.scatter(x.gt_z, x["p_" + m], s=14, color=COL[m])
        a.plot([x.gt_z.min(), x.gt_z.max()], [x.gt_z.min(), x.gt_z.max()], "k--", lw=0.8)
        a.set(xscale="log", yscale="log", xlabel="GT (m)", ylabel="예측 (m)",
              title=f"{m}: 서로 다른 답 {x['p_' + m].round(2).nunique()} 개 / {len(x)} 점")
        for axis in (a.xaxis, a.yaxis):
            axis.set_major_formatter(PLAIN)
            axis.set_minor_formatter(PLAIN)
    fig.suptitle("E6 · DepthLM 은 같은 숫자를 여러 점에 재사용한다 (corridor_01, 공통 점) — 계단 모양", fontsize=12)
    fig.tight_layout()
    H, Wd = g.shape
    th = int(round(H * 300 / Wd))
    lo, hi = np.log(np.percentile(g[valid], [1, 99]))
    u, v, pp, gg, d = lm_pts(image, ds)
    base = (cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)[..., None].repeat(3, 2) // 3).astype(np.uint8)
    tiles = [label(tile(rgb, th), ["corridor_01"], 14), label(tile(depth_rgb(g, lo, hi, valid), th), ["GT"], 14),
             label(dots(tile(base, th), u, v, depth_rgb(pp, lo, hi), 300 / Wd, th / H, 5), ["DepthLM 답 (같은 색 = 같은 숫자)"], 13),
             label(tile(depth_rgb(P["DepthVLM"], lo, hi), th), ["DepthVLM"], 14)]
    save("E6_depthlm_steps.jpg", stack([tiles]), fig)


# ---------------- E7 학습 밖 실외 ----------------
def e7():
    gf = []
    for r in W.records("diode_outdoor"):
        g = W.gt_map(r, W.DATA, "diode_outdoor")
        gf.append(dict(image_id=r["image"], med=np.median(g[g > 0])))
    gf = pd.DataFrame(gf)
    dense = pd.read_parquet(os.path.join(W.OUT, "data", "dense.parquet"))
    a = dense[(dense.dataset == "diode_outdoor") & (dense.group == "all")]
    sc = a.assign(scale=np.exp(a.dl / a.n)).pivot_table(index="image_id", columns="model", values="scale").rename(columns=W.SHORT)
    x = PTS[PTS.dataset == "diode_outdoor"]
    sc["DepthLM"] = x.groupby("image_id").apply(lambda q: np.exp(np.mean(np.log(np.clip(q.p_DepthLM, 0.05, 80)) - np.log(q.gt_z))), include_groups=False)
    t = sc.join(gf.set_index("image_id"))
    fig, ax = plt.subplots(1, 2, figsize=(16, 4.8))
    for m in M6:
        b, a0 = np.polyfit(np.log(t.med), np.log(t[m]), 1)
        ax[0].scatter(t.med, t[m], s=6, color=COL[m], alpha=0.35)
        xx = np.linspace(np.log(t.med.min()), np.log(t.med.max()), 20)
        ax[0].plot(np.exp(xx), np.exp(a0 + b * xx), color=COL[m], lw=2.4 if m in ("DepthLM", "DepthVLM") else 1, label=f"{m}: 기울기 {b:+.2f}")
    ax[0].axhline(1, color="k", lw=0.8)
    ax[0].set(xscale="log", yscale="log", xlabel="장면 중앙 GT 깊이 (m)", ylabel="이미지 배율 (예측/GT)", title="DIODE Outdoor 446 장: 장면이 깊을수록 배율이 내려가나")
    for axis in (ax[0].xaxis, ax[0].yaxis):
        axis.set_major_formatter(PLAIN)
        axis.set_minor_formatter(PLAIN)
    ax[0].legend(fontsize=8)
    t["q"] = pd.qcut(t.med, 3, labels=["얕음", "중간", "깊음"])
    m3 = t.groupby("q", observed=True)[M6].median()
    xs = np.arange(3)
    for k, m in enumerate(M6):
        ax[1].bar(xs + (k - 2.5) * 0.13, m3[m], 0.13, color=COL[m], label=m)
    ax[1].axhline(1, color="k", lw=0.8)
    ax[1].set_xticks(xs, [f"{q} 1/3" for q in m3.index])
    ax[1].set(ylabel="이미지 배율 중앙값", title="장면 깊이 3 등분")
    ax[1].legend(fontsize=7, ncol=2)
    fig.suptitle("E7 · 학습 밖 실외(DIODE): 깊은 장면일수록 두 VLM 의 배율이 무너진다", fontsize=12)
    fig.tight_layout()
    deep = t[(t.q == "깊음")]
    image = (deep[PVS].median(axis=1) / deep[["DepthLM", "DepthVLM"]].max(axis=1)).idxmax()
    r, g, P, valid, rgb = load("diode_outdoor", image)
    H, Wd = g.shape
    th = int(round(H * 300 / Wd))
    lo, hi = np.log(np.percentile(g[valid], [1, 99]))
    u, v, pp, gg, d = lm_pts(image, "diode_outdoor")
    base = (cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)[..., None].repeat(3, 2) // 3).astype(np.uint8)
    tiles = [label(tile(rgb, th), [os.path.basename(image), f"장면 중앙 깊이 {t.loc[image, 'med']:.1f} m"], 13),
             label(tile(depth_rgb(g, lo, hi, valid), th), ["GT"], 14),
             label(dots(tile(base, th), u, v, depth_rgb(pp, lo, hi), 300 / Wd, th / H, 5), [f"DepthLM 답 · 배율 {t.loc[image, 'DepthLM']:.2f}배"], 13),
             label(tile(depth_rgb(P["DepthVLM"], lo, hi), th), [f"DepthVLM · 배율 {t.loc[image, 'DepthVLM']:.2f}배"], 13),
             label(tile(depth_rgb(P["Metric3Dv2"], lo, hi), th), [f"Metric3Dv2 · 배율 {t.loc[image, 'Metric3Dv2']:.2f}배"], 13)]
    save("E7_ood_deep_outdoor.jpg", stack([tiles]), fig)


# ---------------- E8 NYU 채운 GT ----------------
def e8():
    t = pd.read_csv(os.path.join(W.OUT, "nyu_measured_only.csv"))
    t["fq"] = t.groupby("model").fill.transform(lambda x: pd.qcut(x, 3, labels=["채움 적음", "중간", "채움 많음"]))
    ms = ["DepthVLM"] + PVS
    fig, ax = plt.subplots(1, 2, figsize=(17, 4.6))
    m = t.groupby("model")[["shape_all", "shape_meas"]].mean()
    xs = np.arange(len(ms))
    ax[0].bar(xs - 0.2, m.loc[ms, "shape_all"], 0.4, color=[COL[q] for q in ms], alpha=0.5, label="채운 GT 포함 (지금까지의 평가)")
    ax[0].bar(xs + 0.2, m.loc[ms, "shape_meas"], 0.4, color=[COL[q] for q in ms], label="Kinect 실측 픽셀만 (배율도 실측으로)")
    ax[0].set_xticks(xs, ms)
    ax[0].set(ylabel="이미지별 배율 뺀 오차 평균 (%)", title="NYUv2 652 장: 채운 GT 를 빼면 DepthVLM 의 1 위가 사라진다")
    ax[0].legend(fontsize=8)
    f = t.pivot_table(index="model", columns="fq", values="shape_meas", aggfunc="mean", observed=True)
    xs = np.arange(3)
    for k, q in enumerate(ms):
        ax[1].bar(xs + (k - (len(ms) - 1) / 2) * 0.16, f.loc[q, ["채움 적음", "중간", "채움 많음"]], 0.16, color=COL[q], label=q)
    ax[1].set_xticks(xs, ["채움 적음 1/3", "중간 1/3", "채움 많음 1/3"])
    ax[1].set(ylabel="실측 픽셀 배율 뺀 오차 (%)", title="채운 비율 3 등분: 깨끗한 장면에서는 DepthVLM 이 뒤처진다")
    ax[1].legend(fontsize=8)
    fig.suptitle("E8 · NYUv2 에서 DepthVLM 이 '모양 1 위'로 보였던 이유 — 채운 GT", fontsize=12)
    fig.tight_layout()
    Image.fromarray(fig_to_img(fig)).save(os.path.join(EX, "E8_nyu_filled_gt.jpg"), quality=90)
    print("saved E8_nyu_filled_gt.jpg")


if __name__ == "__main__":
    os.makedirs(EX, exist_ok=True)
    for f in [e1, e2, e3, e4, e5, e6, e7, e8]:
        f()
