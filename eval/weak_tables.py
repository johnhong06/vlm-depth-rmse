"""VLM 약점 분석 표 (NOTES F-16) — weak_stats.py 결과(results_vlm_weakness/data)만 읽어 results_vlm_weakness/tables.md 와 csv 를 만든다.
지표 (집단마다, 이미지별 충분통계를 모아 pooled):
  AbsRel, δ1, RMSE (Track B 와 같음)
  모양 = 이미지마다 배율 μ(= 평균 ln 예측/GT, 'all' 로 계산)를 뺀 log 오차의 rms ×100 (% 단위, SILog 와 같은 '배율 뺀 오차'를 집단별로)
  치우침 = 그 집단의 배율 뺀 평균 log 오차 → exp − 1 (%) : + 면 이미지 배율에 비해 그 영역을 멀게, − 면 가깝게 본다
  잔차 = 이미지마다 ln 예측 = a·ln GT + b 를 맞춘 뒤 남는 rms ×100 (%) : 배율·압축으로도 설명 안 되는 오차
약점 판정: VLM 값 / pure vision 4 종 중앙값 (모양·δ1 오답률). 1 보다 크면 그 경우에서 pure vision 보다 나쁨,
  'all' 의 비보다 크면 그 경우에 약점이 몰림. 공통 약점 = DepthLM·DepthVLM 모두 > 1, 그 경우가 있는 zero-shot 세트 모두에서.
점 단위(공통 점) = 6 모델, 픽셀 단위(dense) = DepthLM 을 뺀 5 모델(+ 부록 GT intrinsics 2 조건).
"""
import os

import numpy as np
import pandas as pd

import weak_common as W

D = os.path.join(W.OUT, "data")
PVS = [W.SHORT[m] for m in W.PV]
VL = ["DepthLM", "DepthVLM"]
NICE = {"dist:near": "근거리", "dist:mid": "중거리", "dist:far": "원거리", "region:boundary": "경계 3 px", "region:interior": "경계 밖",
        "size:small": "작은 물체 (<1 %)", "size:medium": "중간 물체", "size:large": "큰 물체 (≥5 %)", "tex:low": "질감 적음", "tex:mid": "질감 중간",
        "tex:high": "질감 많음", "pos:border": "가장자리 10 %", "pos:center": "가운데", "vpos:top": "위 1/3", "vpos:middle": "가운데 1/3",
        "vpos:bottom": "아래 1/3", "nyu:measured": "NYU 원측정", "nyu:filled": "NYU 채운 GT", "plane:floor": "바닥 평면", "plane:wall": "벽 평면",
        "plane:table": "책상 평면", "plane:none": "평면 아님", "all": "전체"}
NICE.update({f"obj:{c}": f"물체:{c}" for c in W.CAT_NAMES})


# ---------- 공통 계산 ----------
def image_params(t):
    """이미지·모델별 배율 μ 와 log 선형 맞춤 (c, b): d = c·ln GT + b."""
    a = t[t.group == "all"].copy()
    mu = a.dl / a.n
    var = a.lg2 / a.n - (a.lg / a.n) ** 2
    cov = a.dlg / a.n - (a.dl / a.n) * (a.lg / a.n)
    c = np.where(var > 1e-9, cov / np.maximum(var, 1e-12), 0.0)
    b = mu - c * a.lg / a.n
    return a.assign(mu=mu, c=c, b=b)[["dataset", "image_id", "model", "mu", "c", "b"]]


def metrics(g):
    n = g.n.sum()
    sh2 = (g.dl2 - 2 * g.mu * g.dl + g.n * g.mu ** 2).sum() / n
    bias = (g.dl - g.n * g.mu).sum() / n
    res2 = (g.dl2 - 2 * g.c * g.dlg - 2 * g.b * g.dl + g.c ** 2 * g.lg2 + 2 * g.c * g.b * g.lg + g.n * g.b ** 2).sum() / n
    return pd.Series(dict(n=n, img=g.image_id.nunique(), absrel=g.ar.sum() / n, d1=g.d1.sum() / n, rmse=np.sqrt(g.se.sum() / n),
                          shape=100 * np.sqrt(max(sh2, 0)), bias=100 * np.expm1(bias), resid=100 * np.sqrt(max(res2, 0)),
                          scale=100 * np.expm1((g.dl.sum() / n))))


def points_as_sums(pts):
    """점 표 → dense 와 같은 충분통계 행 (모델 × 점). 이미지 배율·맞춤은 그 이미지의 공통 점으로."""
    rows = []
    for m in ["DepthLM"] + [W.SHORT[x] for x in W.DENSE]:
        p, g = pts["p_" + m].values, pts.gt_z.values
        cap = pts.dataset.map(lambda s: W.CAP[s])
        lo, hi = np.array([c[0] for c in cap]), np.array([c[1] for c in cap])
        d = np.log(np.clip(p, lo, hi)) - np.log(g)
        lg = np.log(g)
        rows.append(pd.DataFrame(dict(dataset=pts.dataset, image_id=pts.image_id, model=m, n=1, se=(p - g) ** 2, ar=np.abs(p - g) / g,
                                      d1=((p > 0) & (np.maximum(p / g, g / np.maximum(p, 1e-12)) < 1.25)).astype(int),
                                      dl=d, dl2=d ** 2, lg=lg, lg2=lg ** 2, dlg=d * lg, idx=pts.index)))
    return pd.concat(rows, ignore_index=True)


def point_groups(pts):
    """점 → 집단 목록 (dense 집단과 같은 이름)."""
    G = {"all": np.ones(len(pts), bool)}
    for j, nme in enumerate(W.BIN_NAMES):
        G[f"dist:{nme}"] = pts.dist.values == j
    b = pts.boundary
    G["region:boundary"], G["region:interior"] = (b == True).fillna(False).values, (b == False).fillna(False).values  # noqa: E712
    for j, c in enumerate(W.CAT_NAMES):
        if c != "sky":
            G[f"obj:{c}"] = pts.cat.values == j
    for j, nme in enumerate(W.SIZE_NAMES):
        G[f"size:{nme}"] = pts["size"].values == j
    for j, nme in enumerate(["low", "mid", "high"]):
        G[f"tex:{nme}"] = pts.tex.values == j
    G["pos:border"], G["pos:center"] = pts.border.values.astype(bool), ~pts.border.values.astype(bool)
    for j, nme in enumerate(["top", "middle", "bottom"]):
        G[f"vpos:{nme}"] = pts.vpos.values == j
    if "nyu_raw" in pts:
        nr = pts.nyu_raw
        G["nyu:measured"], G["nyu:filled"] = (nr == True).fillna(False).values, (nr == False).fillna(False).values  # noqa: E712
    if "plane" in pts:
        for p in ["floor", "wall", "table"]:
            G[f"plane:{p}"] = (pts.plane == p).fillna(False).values
        G["plane:none"] = (pts.plane == "").fillna(False).values
    return G


def group_table(t, prm):
    t = t.merge(prm, on=["dataset", "image_id", "model"])
    out = t.groupby(["dataset", "group", "model"]).apply(metrics, include_groups=False).reset_index()
    return out


def ratio_view(tab, models, min_n):
    """VLM / pure vision 중앙값 비 (모양, δ1 오답률), 'all' 대비 집중도."""
    rows = []
    for (ds, grp), g in tab.groupby(["dataset", "group"]):
        g = g.set_index("model")
        if not set(PVS) <= set(g.index) or g.n.min() < min_n:
            continue
        pv_sh, pv_err = g.loc[PVS, "shape"].median(), (1 - g.loc[PVS, "d1"]).median()
        for m in models:
            if m in g.index:
                rows.append(dict(dataset=ds, group=grp, model=m, n=g.loc[m, "n"], img=g.loc[m, "img"], shape=g.loc[m, "shape"], pv_shape=pv_sh,
                                 shape_ratio=g.loc[m, "shape"] / pv_sh, d1=g.loc[m, "d1"], pv_d1=1 - pv_err,
                                 err_ratio=(1 - g.loc[m, "d1"]) / max(pv_err, 1e-9), bias=g.loc[m, "bias"],
                                 pv_bias=g.loc[PVS, "bias"].median(), resid=g.loc[m, "resid"], pv_resid=g.loc[PVS, "resid"].median(),
                                 shape_rank=int(g["shape"].rank(method="min")[m]), d1_rank=int(g["d1"].rank(ascending=False, method="min")[m])))
    r = pd.DataFrame(rows)
    base = r[r.group == "all"].set_index(["dataset", "model"])
    r["shape_conc"] = r.shape_ratio / base.loc[list(zip(r.dataset, r.model)), "shape_ratio"].values
    r["err_conc"] = r.err_ratio / base.loc[list(zip(r.dataset, r.model)), "err_ratio"].values
    return r


def md(df, cols=None, fmt=None):
    cols = cols or list(df.columns)
    fmt = fmt or {}
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join([":--"] + ["--:"] * (len(cols) - 1)) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(fmt.get(c, "{}").format(r[c]) if not (isinstance(r[c], float) and np.isnan(r[c])) else "–" for c in cols) + " |")
    return "\n".join(lines)


ROWS = (["all", "dist:near", "dist:mid", "dist:far", "region:boundary", "region:interior", "size:small", "size:medium", "size:large",
         "tex:low", "tex:mid", "tex:high", "pos:border", "pos:center", "vpos:top", "vpos:middle", "vpos:bottom",
         "plane:floor", "plane:wall", "plane:table", "plane:none", "nyu:measured", "nyu:filled"]
        + [f"obj:{c}" for c in W.CAT_NAMES if c != "sky"])


def heatmap(r, value, cols, title, dst, vmax=3.0):
    """행 = 경우, 열 = (세트, 모델). 값 = VLM / pure vision 중앙값 (1 = 같음, 빨강 = VLM 이 나쁨). log 색."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager as fm
    fm.fontManager.addfont("/usr/share/fonts/truetype/nanum/NanumSquareR.ttf")
    plt.rcParams["font.family"] = "NanumSquare"
    plt.rcParams["axes.unicode_minus"] = False
    rows = [g for g in ROWS if g in set(r.group)]
    M = np.full((len(rows), len(cols)), np.nan)
    for j, (ds, m) in enumerate(cols):
        x = r[(r.dataset == ds) & (r.model == m)].set_index("group")[value]
        for i, g in enumerate(rows):
            if g in x.index:
                M[i, j] = x[g]
    fig, ax = plt.subplots(figsize=(1.25 * len(cols) + 3, 0.34 * len(rows) + 1.6))
    im = ax.imshow(np.log(M), cmap="RdBu_r", vmin=-np.log(vmax), vmax=np.log(vmax), aspect="auto")
    for i in range(len(rows)):
        for j in range(len(cols)):
            if np.isfinite(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=8, color="black" if abs(np.log(M[i, j])) < 0.7 else "white")
    ax.set_yticks(range(len(rows)), [NICE.get(g, g) for g in rows], fontsize=9)
    ax.set_xticks(range(len(cols)), [f"{ds}\n{m}" for ds, m in cols], fontsize=8.5)
    ax.set_title(title, fontsize=11)
    cb = fig.colorbar(im, ax=ax, fraction=0.03)
    t = [1 / vmax, 1 / 2, 1, 2, vmax]
    cb.set_ticks(np.log(t), labels=[f"{x:.2g}" for x in t])
    fig.tight_layout()
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    fig.savefig(dst, dpi=130)
    plt.close(fig)


def pairs(pts):
    """같은 이미지의 공통 점 쌍: 앞뒤 순서 오답률(GT 깊이 비 > 1.1 인 쌍) 과 상대 깊이 오차 |Δ ln(p_i/p_j) − Δ ln(g_i/g_j)|.
    점 사이 화면 거리(px, 원본 해상도)·GT 깊이 비 구간으로 나눈다. 배율과 무관한 지표라 6 모델을 그대로 비교할 수 있다."""
    models = ["DepthLM"] + [W.SHORT[x] for x in W.DENSE]
    rows = []
    for (ds, img), x in pts.groupby(["dataset", "image_id"]):
        if len(x) < 2:
            continue
        i, j = np.triu_indices(len(x), 1)
        u, v, g = x.u.values, x.v.values, x.gt_z.values
        px = np.hypot(u[i] - u[j], v[i] - v[j])
        lr = np.log(g[i] / g[j])
        for m in models:
            p = np.clip(x["p_" + m].values, *W.CAP[ds])
            pr_ = np.log(p[i] / p[j])
            rows.append(pd.DataFrame(dict(dataset=ds, model=m, px=px, gap=np.abs(lr), wrong=(np.sign(pr_) != np.sign(lr)), relerr=np.abs(pr_ - lr))))
    t = pd.concat(rows, ignore_index=True)
    t["px_bin"] = pd.cut(t.px, [0, 50, 150, 400, 1e9], labels=["<50px", "50–150", "150–400", ">400"])
    t["gap_bin"] = pd.cut(t.gap, [-1, np.log(1.1), np.log(1.25), np.log(2), 99], labels=["<10%", "10–25%", "25–100%", ">2배"])
    a = t[t.gap > np.log(1.1)].groupby(["dataset", "gap_bin", "model"], observed=True).wrong.mean().unstack("model")[models] * 100
    b = t.groupby(["dataset", "px_bin", "model"], observed=True).relerr.median().unstack("model")[models] * 100
    return a, b


def per_image(pts, dtab_img):
    """이미지마다 공통 점 모양 오차(6 모델) + 전체 픽셀 모양 오차(5 모델). 두 VLM 이 pure vision 4 종보다 모두 나쁜 이미지 비율."""
    models = ["DepthLM"] + [W.SHORT[x] for x in W.DENSE]
    rows = []
    for (ds, img), x in pts.groupby(["dataset", "image_id"]):
        if len(x) < 8:
            continue
        r = dict(dataset=ds, image_id=img, n_pts=len(x))
        for m in models:
            d = np.log(np.clip(x["p_" + m].values, *W.CAP[ds])) - np.log(x.gt_z.values)
            r["pt_" + m] = 100 * d.std()
        rows.append(r)
    t = pd.DataFrame(rows).merge(dtab_img, on=["dataset", "image_id"], how="left")
    pv_pt = t[["pt_" + W.SHORT[m] for m in W.PV]]
    t["pt_both_worse_than_all_pv"] = (t.pt_DepthLM > pv_pt.max(axis=1)) & (t.pt_DepthVLM > pv_pt.max(axis=1))
    t["pt_both_worse_than_median_pv"] = (t.pt_DepthLM > pv_pt.median(axis=1)) & (t.pt_DepthVLM > pv_pt.median(axis=1))
    return t


def main():
    dense = pd.read_parquet(os.path.join(D, "dense.parquet"))
    dense["model"] = dense.model.map(W.SHORT)
    pts = pd.read_parquet(os.path.join(D, "points.parquet")).reset_index(drop=True)
    # ── 점 단위: 6 모델
    ps = points_as_sums(pts)
    G = point_groups(pts)
    prows = []
    for name, sel in G.items():
        idx = np.nonzero(sel)[0]
        if len(idx):
            x = ps[ps.idx.isin(idx)].groupby(["dataset", "image_id", "model"], as_index=False)[["n", "se", "ar", "d1", "dl", "dl2", "lg", "lg2", "dlg"]].sum()
            prows.append(x.assign(group=name))
    pt_sums = pd.concat(prows, ignore_index=True)
    ptab = group_table(pt_sums, image_params(pt_sums))
    dtab = group_table(dense, image_params(dense))
    ptab.to_csv(os.path.join(W.OUT, "points_groups.csv"), index=False)
    dtab.to_csv(os.path.join(W.OUT, "dense_groups.csv"), index=False)
    pr = ratio_view(ptab, VL, min_n=150)
    dr = ratio_view(dtab, ["DepthVLM"] + [W.SHORT[m] for m in W.SUPP], min_n=20000)
    pr.to_csv(os.path.join(W.OUT, "points_ratio.csv"), index=False)
    dr.to_csv(os.path.join(W.OUT, "dense_ratio.csv"), index=False)
    S = os.path.join(W.OUT, "00_summary")
    pc = [(ds, m) for ds in W.SETS for m in VL]
    heatmap(pr, "shape_ratio", pc, "공통 점: 배율 뺀 오차(모양) ÷ pure vision 4 종 중앙값", os.path.join(S, "heat_points_shape.png"))
    heatmap(pr, "err_ratio", pc, "공통 점: δ1 오답률 ÷ pure vision 4 종 중앙값", os.path.join(S, "heat_points_d1err.png"))
    dc = [(ds, "DepthVLM") for ds in W.SETS]
    heatmap(dr, "shape_ratio", dc, "전체 픽셀: DepthVLM 모양 오차 ÷ pure vision 중앙값", os.path.join(S, "heat_dense_shape.png"))
    heatmap(dr, "err_ratio", dc, "전체 픽셀: DepthVLM δ1 오답률 ÷ pure vision 중앙값", os.path.join(S, "heat_dense_d1err.png"))
    a = dense[dense.group == "all"]
    dimg = a.assign(dn_shape=100 * np.sqrt(np.maximum(a.dl2 / a.n - (a.dl / a.n) ** 2, 0))).pivot_table(
        index=["dataset", "image_id"], columns="model", values="dn_shape").add_prefix("dn_").reset_index()
    pim = per_image(pts, dimg)
    pim.to_csv(os.path.join(W.OUT, "per_image.csv"), index=False)
    oa, ob = pairs(pts)
    oa.to_csv(os.path.join(W.OUT, "pairs_order_wrong.csv"))
    ob.to_csv(os.path.join(W.OUT, "pairs_relerr.csv"))
    print("points", ptab.shape, "dense", dtab.shape)
    return ptab, dtab, pr, dr


if __name__ == "__main__":
    main()


def sparse_planes(pts):
    """iBims-1 공식 평면 안 공통 점(평면당 ≥ 8 점)으로 6 모델 평면성 — 이미지 배율(공통 점 중앙값 비) 맞춘 뒤 3D 평면 맞춤.
    ε_plan = 평면까지 거리의 표준편차(cm), ε_orie = GT 점으로 맞춘 평면과의 법선 각(°). 픽셀 단위 지표(ibims_struct)의 점 판."""
    import weak_common as Wc
    K = {r["image"]: r["_k"] for r in Wc.records("ibims1")}
    x = pts[(pts.dataset == "ibims1") & (pts.plane_id > 0)]
    models = ["DepthLM"] + [W.SHORT[m] for m in W.DENSE]
    allp = pts[pts.dataset == "ibims1"]
    scale = {m: allp.groupby("image_id").apply(lambda g: np.median(g.gt_z / g["p_" + m]), include_groups=False) for m in models}
    rows = []
    for (img, pid), g in x.groupby(["image_id", "plane_id"]):
        if len(g) < 8:
            continue
        k = K[img]
        def P(z):
            return np.stack([(g.u.values - k[2]) * z / k[0], (g.v.values - k[3]) * z / k[1], z], 1)
        ng, _ = fit(P(g.gt_z.values))
        for m in models:
            n, dist = fit(P(g["p_" + m].values * scale[m][img]))
            rows.append(dict(image_id=img, plane=g.plane.iloc[0], model=m, n=len(g), plan_cm=100 * dist.std(),
                             orie_deg=np.degrees(np.arccos(min(1.0, abs(n @ ng))))))
    return pd.DataFrame(rows)


def fit(Pt):
    c = Pt.mean(0)
    n = np.linalg.svd(Pt - c, full_matrices=False)[2][-1]
    return n, (Pt - c) @ n


def compression(pts, dense):
    """이미지마다 ln 예측 = a·ln GT + b 의 기울기 a 중앙값 (1 = 거리 차를 그대로, < 1 = 먼 곳을 가깝게·가까운 곳을 멀게 눌러 담음)."""
    out = {}
    for ds in W.SETS:
        x = pts[pts.dataset == ds]
        r = {}
        for m in ["DepthLM"] + [W.SHORT[q] for q in W.DENSE]:
            a = x.groupby("image_id").apply(lambda g: np.polyfit(np.log(g.gt_z), np.log(np.clip(g["p_" + m], *W.CAP[ds])), 1)[0]
                                            if len(g) >= 8 and np.ptp(np.log(g.gt_z)) > 0.2 else np.nan, include_groups=False)
            r["pt_" + m] = np.nanmedian(a)
        d = dense[(dense.dataset == ds) & (dense.group == "all")]
        var = d.lg2 / d.n - (d.lg / d.n) ** 2
        cov = d.dlg / d.n - (d.dl / d.n) * (d.lg / d.n)
        for m, g in d.assign(a=1 + cov / var).groupby("model"):
            r["dn_" + m] = g.a.median()
        out[ds] = r
    return pd.DataFrame(out)
