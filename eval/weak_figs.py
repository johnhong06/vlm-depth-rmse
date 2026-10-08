"""VLM 약점 분석 그림 (NOTES F-16) → results_vlm_weakness/
  cards/<세트>/<이미지>.jpg : 모든 이미지 한 장씩 — 1 행 RGB·GT·거리 구간+경계·물체 분류·평면(iBims)/채운 GT(NYU)/물체 크기(DIODE),
                              2 행 예측 깊이, 3 행 오차 ln(예측/GT), 4 행 배율 뺀 오차(모양). 열 = DepthLM(점)·DepthVLM·pure vision 4 종.
  cases/<세트>/<경우>.jpg  : 경우(거리·경계·물체·크기·질감·위치·평면·채운 GT)마다 DepthVLM 이 pure vision 보다 가장 크게 틀린 이미지 6 장.
색: 깊이 = turbo (log, GT 1–99 백분위), 오차 = 빨강 '멀게'·파랑 '가깝게' (log 비 ±0.4 ≈ ±50 %), 회색 = GT 없음.
사용: ~/venv/main/bin/python eval/weak_figs.py cards [--workers 10] | cases
"""
import argparse
import os
import sys
from multiprocessing import Pool

import cv2
import matplotlib
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

import weak_common as W

matplotlib.use("Agg")
from matplotlib import colormaps  # noqa: E402

FONT = "/usr/share/fonts/truetype/nanum/NanumSquareB.ttf"
TW = 300                                         # 타일 너비
TURBO = (colormaps["turbo"](np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)
RDBU = (colormaps["RdBu_r"](np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)
GRAY = np.array([70, 70, 70], np.uint8)
CAT_COL = np.array([[174, 199, 232], [140, 86, 75], [199, 199, 199], [255, 187, 120], [214, 39, 40], [148, 103, 189], [23, 190, 207],
                    [127, 127, 127], [44, 160, 44], [158, 218, 229], [255, 127, 14], [227, 119, 194], [188, 189, 34], [31, 119, 180], [60, 60, 60]], np.uint8)
DIST_COL = np.array([[66, 146, 198], [116, 196, 118], [253, 141, 60]], np.uint8)
ORDER = ["DepthLM-12B"] + W.DENSE
LIM = 0.4


def font(sz):
    return ImageFont.truetype(FONT, sz)


def tile(img, h):
    return cv2.resize(img, (TW, h), interpolation=cv2.INTER_AREA if img.shape[1] > TW else cv2.INTER_NEAREST)


def depth_rgb(d, lo, hi, valid=None):
    x = np.clip((np.log(np.maximum(d, 1e-3)) - lo) / (hi - lo), 0, 1)
    out = TURBO[(x * 255).astype(np.uint8)]
    if valid is not None:
        out[~valid] = GRAY
    return out


def err_rgb(e, valid):
    out = RDBU[(np.clip(e / LIM, -1, 1) * 127.5 + 127.5).astype(np.uint8)]
    out[~valid] = GRAY
    return out


def label(img, lines, size=15):
    """타일 왼쪽 위에 글자 (어두운 띠)."""
    im = Image.fromarray(img)
    dr = ImageDraw.Draw(im)
    f = font(size)
    y = 2
    for ln in lines:
        w = dr.textlength(ln, font=f)
        dr.rectangle([0, y, w + 6, y + size + 3], fill=(0, 0, 0))
        dr.text((3, y), ln, font=f, fill=(255, 255, 255))
        y += size + 4
    return np.array(im)


def dots(base, u, v, cols, sx, sy, rad=4):
    out = base.copy()
    for x, y, c in zip(u, v, cols):
        cv2.circle(out, (int(x * sx), int(y * sy)), rad + 1, (0, 0, 0), -1)
        cv2.circle(out, (int(x * sx), int(y * sy)), rad, tuple(int(t) for t in c), -1)
    return out


def img_metrics(p, g, valid, ds):
    pp, gg = p[valid], g[valid]
    d = np.log(np.clip(pp, *W.CAP[ds])) - np.log(gg)
    return dict(absrel=float(np.mean(np.abs(pp - gg) / gg)), d1=float(np.mean(np.maximum(pp / gg, gg / np.maximum(pp, 1e-12)) < 1.25)),
                shape=float(100 * np.std(d)), mu=float(np.mean(d)))


def load_all(ds, r):
    g = W.gt_map(r, W.DATA, ds)
    P = {m: W.load_map(m, ds, r) for m in W.DENSE}
    valid = g > 0
    for p in P.values():
        valid &= np.isfinite(p)
    return g, P, valid


def card(job):
    ds, r = job
    dst = os.path.join(W.OUT, "cards", ds, W.stem(r["image"]).split("__")[-1] + ".jpg")
    if ds == "diode_outdoor":
        dst = os.path.join(W.OUT, "cards", ds, "__".join(W.stem(r["image"]).split("__")[-3:]) + ".jpg")
    if os.path.exists(dst):
        return
    g, P, valid = load_all(ds, r)
    a = W.attributes(ds, r, g)
    H, Wd = g.shape
    th = int(round(H * TW / Wd))
    rgb = cv2.cvtColor(cv2.imread(os.path.join(W.DATA, r["image"])), cv2.COLOR_BGR2RGB)
    lo, hi = np.log(np.percentile(g[valid], [1, 99]))
    gray = np.repeat(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)[..., None], 3, 2)
    # 1 행: 속성
    dist = (0.45 * gray + 0.55 * DIST_COL[np.clip(a["dist"], 0, 2)]).astype(np.uint8)
    dist[~valid] = GRAY
    if a["boundary"] is not None:
        dist[a["boundary"] & valid] = 255
    cat = (0.4 * gray + 0.6 * CAT_COL[a["cat"]]).astype(np.uint8)
    present = [W.CAT_NAMES[c] for c in np.unique(a["cat"][valid])]
    if ds == "ibims1":
        third = gray.copy() // 2 + 60
        for p, c in zip(["floor", "wall", "table"], [(255, 127, 14), (31, 119, 180), (44, 160, 44)]):
            third[a["plane"] == p] = c
        third_lab = ["공식 평면: 주황 바닥", "파랑 벽 · 초록 책상"]
    elif ds == "nyuv2":
        third = gray.copy()
        third[valid & ~a["nyu_raw"]] = (255, 0, 255)
        third_lab = [f"분홍 = 채운 GT ({(valid & ~a['nyu_raw']).sum() / valid.sum():.0%})"]
    else:
        third = gray.copy() // 2 + 60
        for j, c in enumerate([(255, 0, 0), (255, 215, 0), (0, 200, 255)]):
            third[a["size"] == j] = c
        third_lab = ["물체 크기: 빨강 <1 %", "노랑 1–5 % · 하늘 ≥5 %"]
    row1 = [label(tile(rgb, th), [ds, os.path.basename(r["image"])], 13),
            label(tile(depth_rgb(g, lo, hi, valid), th), ["GT 깊이 (회색 = GT 없음)"]),
            label(tile(dist, th), ["거리: 파랑 근 · 초록 중 · 주황 원", "흰색 = 경계 3 px" if a["boundary"] is not None else "경계: 이 세트는 제외"], 13),
            label(tile(cat, th), ["물체: " + ", ".join(present[:4]), ", ".join(present[4:9])], 12),
            label(tile(third, th), third_lab, 13)]
    # 2–4 행
    lm = LM[LM.image_id == r["image"]]
    lm = lm[np.isfinite(lm.pred)]
    sx, sy = TW / Wd, th / H
    row2, row3, row4, stats = [], [], [], {}
    for m in ORDER:
        if m == "DepthLM-12B":
            u, v, pp, gg = lm.u.values, lm.v.values, lm.pred.values, lm.gt_z.values
            d = np.log(np.clip(pp, *W.CAP[ds])) - np.log(gg)
            mu = d.mean() if len(d) else 0.0
            dk = (gray // 3).astype(np.uint8)
            row2.append(dots(tile(dk, th), u, v, depth_rgb(pp, lo, hi), sx, sy))
            row3.append(dots(tile(dk, th), u, v, err_rgb(d, np.ones(len(d), bool)), sx, sy))
            row4.append(dots(tile(dk, th), u, v, err_rgb(d - mu, np.ones(len(d), bool)), sx, sy))
            st = dict(absrel=float(np.mean(np.abs(pp - gg) / gg)), d1=float(np.mean(np.maximum(pp / gg, gg / pp) < 1.25)),
                      shape=float(100 * np.std(d)), mu=float(mu), n=len(d))
        else:
            p = P[m]
            st = img_metrics(p, g, valid, ds)
            e = np.zeros_like(g)
            e[valid] = np.log(np.clip(p[valid], *W.CAP[ds])) - np.log(g[valid])
            row2.append(tile(depth_rgb(p, lo, hi), th))
            row3.append(tile(err_rgb(e, valid), th))
            row4.append(tile(err_rgb(e - st["mu"], valid), th))
        stats[m] = st
        nm = W.SHORT[m] + (f" ({st['n']}점)" if m == "DepthLM-12B" else "")
        row2[-1] = label(row2[-1], [nm, f"AbsRel {st['absrel']:.3f}  δ1 {st['d1']:.3f}"], 14)
        row3[-1] = label(row3[-1], [f"오차  배율 {np.exp(st['mu']):.2f}배"], 14)
        row4[-1] = label(row4[-1], [f"배율 뺀 오차 (모양) {st['shape']:.1f}%"], 14)
    leg = np.full((th, TW, 3), 255, np.uint8)
    bar = np.repeat(RDBU[None, ::2], 18, 0)
    leg[th // 2 - 9: th // 2 + 9, 22:150] = cv2.resize(bar, (128, 18))
    leg = label(leg, ["오차 색: 파랑 = 가깝게, 빨강 = 멀게", "±0.4 log = -33 % ~ +49 %", "깊이 색: turbo (가까움 → 멂)"], 13)
    row1.append(leg)
    rows = [np.hstack(row) for row in [row1, row2, row3, row4]]
    wmax = max(x.shape[1] for x in rows)
    grid = np.vstack([np.pad(x, ((0, 0), (0, wmax - x.shape[1]), (0, 0)), constant_values=255) for x in rows])
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    Image.fromarray(grid).save(dst, quality=86)


def case_rank(dense, ds, grp):
    """이미지별: 그 경우 픽셀의 배율 뺀 오차(모양) — DepthVLM − pure vision 중앙값이 큰 순. 경우 픽셀이 2 % 이상이고 3,000 개 이상인 이미지만."""
    x = dense[(dense.dataset == ds) & (dense.group == grp)]
    a = dense[(dense.dataset == ds) & (dense.group == "all")][["image_id", "model", "n", "dl"]].rename(columns={"n": "n_all", "dl": "dl_all"})
    x = x.merge(a, on=["image_id", "model"])
    mu = x.dl_all / x.n_all
    x = x.assign(shape=100 * np.sqrt(np.maximum((x.dl2 - 2 * mu * x.dl + x.n * mu ** 2) / x.n, 0)), share=x.n / x.n_all)
    p = x.pivot_table(index="image_id", columns="model", values="shape")
    share = x[x.model == "DepthVLM-4B"].set_index("image_id")
    ok = (share.share >= 0.02) & (share.n >= 3000)
    p = p.loc[ok[ok].index]
    if p.empty:
        return p
    p["pv_med"] = p[W.PV].median(axis=1)
    p["best_pv"] = p[W.PV].idxmin(axis=1)
    p["excess"] = p["DepthVLM-4B"] - p.pv_med
    p["share"] = share.share.reindex(p.index)
    return p.sort_values("excess", ascending=False)


def dim(img, mask):
    out = img.copy()
    out[~mask] = (0.25 * out[~mask] + 0.75 * 200).astype(np.uint8)
    return out


def case_panel(ds, grp, top, recs):
    rows = []
    for image_id, s in top.iterrows():
        r = recs[image_id]
        g, P, valid = load_all(ds, r)
        a = W.attributes(ds, r, g)
        sel = W.groups(ds, a)[grp]
        sel = valid if sel is None else sel & valid
        H, Wd = g.shape
        th = int(round(H * TW / Wd))
        rgb = cv2.cvtColor(cv2.imread(os.path.join(W.DATA, r["image"])), cv2.COLOR_BGR2RGB)
        lo, hi = np.log(np.percentile(g[valid], [1, 99]))
        hl = rgb.copy()
        hl[sel] = (0.45 * hl[sel] + 0.55 * np.array([255, 230, 0])).astype(np.uint8)
        best = s.best_pv
        tiles = [label(tile(hl, th), [os.path.basename(image_id), f"노랑 = {W_NICE(grp)} ({s.share:.0%})"], 13),
                 label(tile(depth_rgb(g, lo, hi, valid), th), ["GT"], 14)]
        for m in ["DepthVLM-4B", best]:
            tiles.append(label(tile(depth_rgb(P[m], lo, hi), th), [f"{W.SHORT[m]} 예측"], 14))
        for m in ["DepthVLM-4B", best]:
            e = np.zeros_like(g)
            e[valid] = np.log(np.clip(P[m][valid], *W.CAP[ds])) - np.log(g[valid])
            e[valid] -= e[valid].mean()
            tiles.append(label(tile(dim(err_rgb(e, valid), sel), th), [f"{W.SHORT[m]} 모양 오차", f"이 영역 {s[m]:.1f}%  (PV 중앙 {s.pv_med:.1f}%)"], 13))
        lm = LM[(LM.image_id == image_id) & np.isfinite(LM.pred)]
        u, v = lm.u.values.astype(int), lm.v.values.astype(int)
        d = np.log(np.clip(lm.pred.values, *W.CAP[ds])) - np.log(lm.gt_z.values)
        d = d - d.mean() if len(d) else d
        ins = sel[v, u] if len(u) else np.zeros(0, bool)
        base = (cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)[..., None].repeat(3, 2) // 3).astype(np.uint8)
        t = dots(tile(base, th), u[~ins], v[~ins], np.full((int((~ins).sum()), 3), 110), TW / Wd, th / H, rad=2)
        t = dots(t, u[ins], v[ins], err_rgb(d[ins], np.ones(int(ins.sum()), bool)), TW / Wd, th / H, rad=5)
        sh = f"{100 * np.sqrt(np.mean(d[ins] ** 2)):.1f}%" if ins.sum() >= 3 else "점 부족"
        tiles.append(label(t, [f"DepthLM 점 모양 오차", f"이 영역 {int(ins.sum())}점: {sh}"], 13))
        rows.append(np.hstack(tiles))
    wmax = max(x.shape[1] for x in rows)
    return np.vstack([np.pad(x, ((0, 4), (0, wmax - x.shape[1]), (0, 0)), constant_values=255) for x in rows])


def W_NICE(grp):
    from weak_tables import NICE
    return NICE.get(grp, grp)


def cases(dense):
    for ds in W.SETS:
        recs = {r["image"]: r for r in W.records(ds)}
        for grp in sorted(dense[dense.dataset == ds].group.unique()):
            if grp == "all":
                continue
            top = case_rank(dense, ds, grp).head(6)
            if top.empty:
                continue
            dst = os.path.join(W.OUT, "cases", ds, grp.replace(":", "_") + ".jpg")
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            Image.fromarray(case_panel(ds, grp, top, recs)).save(dst, quality=86)
            print(ds, grp, len(top), flush=True)


def init():
    global LM
    LM = W.depthlm_points()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["cards", "cases"])
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    if a.what == "cards":
        jobs = [(ds, r) for ds in W.SETS for r in (W.records(ds)[: a.limit] if a.limit else W.records(ds))]
        with Pool(a.workers, initializer=init) as pool:
            for i, _ in enumerate(pool.imap_unordered(card, jobs, chunksize=2)):
                if i % 200 == 0:
                    print(f"cards {i}/{len(jobs)}", flush=True)
        print("cards done", flush=True)
    else:
        init()
        cases(pd.read_parquet(os.path.join(W.OUT, "data", "dense.parquet")))
