"""Track B 표 — eval/dense_full.py 의 이미지별 통계(stats_*.parquet)만으로 계산한다 (NOTES D-19). 규칙은 Track A 와 같다:
  주 지표 = 데이터셋 안 valid GT 픽셀 전체 pooled, 보조 = 이미지별 평균, 이미지 단위 bootstrap 95 % CI(모델끼리 같은 재표본), 도메인 행 = 데이터셋 값 평균(실내·실외 따로),
  부록 = 유클리드 RMSE, 보조 집계 = 거리 구간·경계/내부·log-RMSE·SILog(KITTI 정의, D-18), 검증 = DepthVLM 공식 dense 방식 δ1(canonical, 이미지별).
사용: python eval/score_dense.py results/track_b/stats_*.parquet --models DepthVLM-4B DAv2-metric-L UniDepthV2-L Metric3Dv2-L DepthPro --out tables/track_b
"""
import argparse
import os

import numpy as np
import pandas as pd

from common import BENCH
from score import CEILING, META, TRAINED

K = ["rmse", "absrel", "d1", "logrmse", "silog", "rmse_euc", "rmse_img", "absrel_img", "d1_img"]
REF = {("DepthVLM-4B", "nuscenes"): 0.838, ("DepthVLM-4B", "ibims1"): 0.910, ("UniDepthV2-L", "nuscenes"): 0.868, ("UniDepthV2-L", "ibims1"): 0.941,
       ("Metric3Dv2-L", "nuscenes"): 0.843, ("Metric3Dv2-L", "ibims1"): 0.724, ("DepthPro", "nuscenes"): 0.379, ("DepthPro", "ibims1"): 0.879}  # DepthVLM README dense 표 (δ1)


def metrics(s, w):
    """s: 이미지 순서로 맞춘 통계 (없는 이미지는 n=0), w: (B+1, 이미지) 가중치 → (B+1, len(K))."""
    n, has, two = w @ s.n.values, (s.n > 0).values.astype(float), (s.n >= 2).values.astype(float)
    nn = np.maximum(s.n.values, 1)
    per = lambda x, m: (w * m) @ x / np.maximum((w * m).sum(1), 1)
    sil = 100 * np.sqrt(np.maximum(s.dl2.values / nn - (s.dl.values / nn) ** 2, 0))
    return np.stack([np.sqrt(w @ s.se.values / n), w @ s.ar.values / n, w @ s.d1.values / n, np.sqrt(w @ s.dl2.values / n), per(sil, two),
                     np.sqrt(w @ s.se_euc.values / n), per(np.sqrt(s.se.values / nn), has), per(s.ar.values / nn, has), per(s.d1.values / nn, has)], 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--B", type=int, default=2000)
    a = ap.parse_args()
    d = pd.concat([pd.read_parquet(f) for f in a.files])
    d = d[d.model.isin(a.models)]
    rows, boot, val = [], {}, []
    for ds in [k for k in BENCH if k in set(d.dataset)]:
        x = d[d.dataset == ds]
        if x.model.nunique() < len(a.models):
            print(f"[score_dense] {ds}: 아직 돌지 않은 모델 {sorted(set(a.models) - set(x.model))} — 건너뜀")
            continue
        imgs = sorted(set.intersection(*[set(x[(x.model == m) & (x.group == "all")].image_id) for m in a.models]))
        rng = np.random.default_rng(0)
        w = np.vstack([np.ones(len(imgs)), rng.multinomial(len(imgs), np.full(len(imgs), 1 / len(imgs)), a.B)])
        for m in a.models:
            v = x[(x.model == m) & (x.group == "all")]
            val.append(dict(dataset=ds, model=m, d1_canon=v.set_index("image_id").loc[imgs].d1_canon.mean(), ref=REF.get((m, ds))))
            for grp in sorted(set(x.group)):
                s = x[(x.model == m) & (x.group == grp)].set_index("image_id").reindex(imgs).fillna(0)
                r = metrics(s, w)
                if grp == "all":
                    boot[ds, m] = r[1:]
                rows.append(dict(dataset=ds, domain=BENCH[ds][1], group=grp, model=m, n_px=int(s.n.sum()), n_img=int((s.n > 0).sum()), n_clip=int(s.n_clip.sum()),
                                 **dict(zip(K, r[0])), **{k + "_lo": v for k, v in zip(K, np.percentile(r[1:], 2.5, 0))},
                                 **{k + "_hi": v for k, v in zip(K, np.percentile(r[1:], 97.5, 0))}))
    if not rows:
        raise SystemExit("[score_dense] 표에 넣을 데이터셋이 없다")
    t = pd.DataFrame(rows)
    for (dom, m), g in t[t.group == "all"].groupby(["domain", "model"], sort=False):
        b = np.mean([boot[ds, m] for ds in g.dataset], 0)
        rows.append(dict(dataset="mean", domain=dom, group="all", model=m, n_px=g.n_px.sum(), n_img=g.n_img.sum(), n_clip=g.n_clip.sum(),
                         **g[K].mean().to_dict(), **{k + "_lo": v for k, v in zip(K, np.percentile(b, 2.5, 0))},
                         **{k + "_hi": v for k, v in zip(K, np.percentile(b, 97.5, 0))}))
    t = pd.DataFrame(rows)
    order = {ds: i for i, ds in enumerate(list(BENCH) + ["mean"])}
    gorder = {g: i for i, g in enumerate(["all", "dist:near", "dist:mid", "dist:far", "region:boundary", "region:interior"])}
    t = t.iloc[np.lexsort((t.model.map(a.models.index), t.group.map(gorder), t.dataset.map(order), t.domain.map({"Indoor": 0, "Outdoor": 1})))]
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    t.to_csv(a.out + ".csv", index=False)
    f = lambda r, k: f"{r[k]:.3f} [{r[k + '_lo']:.3f}, {r[k + '_hi']:.3f}]"
    main_ = t[t.group == "all"]
    members = main_[main_.dataset != "mean"].groupby("domain").dataset.unique().to_dict()
    tags = lambda r: [s for ds in (members[r.domain] if r.dataset == "mean" else [r.dataset]) for s in (TRAINED.get((r.model, ds)), CEILING.get((r.model, ds))) if s]
    notes = {v: i + 1 for i, v in enumerate(dict.fromkeys(s for _, r in main_.iterrows() for s in tags(r)))}
    mark = lambda r: "".join(f"<sup>{notes[s]}</sup>" for s in dict.fromkeys(tags(r)))
    out = ["| Domain | Dataset | Model | GT intrinsics | Domain info | RMSE↓ (m) | AbsRel↓ | δ1↑ | per-image RMSE / AbsRel / δ1 | px | img |",
           "|:--|:--|:--|:--|:--|:--|:--|:--|:--|--:|--:|"]
    for _, r in main_.iterrows():
        gk, dm = META.get(r.model, ("?", "?"))
        out.append(f"| {r.domain} | {r.dataset} | {r.model}{mark(r)} | {gk} | {dm} | {f(r, 'rmse')} | {f(r, 'absrel')} | {f(r, 'd1')} | "
                   f"{r.rmse_img:.3f} / {r.absrel_img:.3f} / {r.d1_img:.3f} | {r.n_px} | {r.n_img} |")
    out += ["", "\n".join(f"{i}. {s}" for s, i in notes.items()),
            f"\nTrack B (dense): all valid GT pixels at the GT's original resolution, predictions resized bilinearly. Main = pooled over pixels of a dataset; "
            f"brackets = 95 % CI, image bootstrap (B = {a.B}, seed 0). Domain rows average their datasets; indoor and outdoor are never averaged together."]
    sec = lambda title, sel, cols: ([f"\n### {title}\n", "| Dataset | Group | Model | " + " | ".join(cols) + " | px | img |", "|:--|:--|:--|" + "--:|" * (len(cols) + 2)]
                                    + [f"| {r.dataset} | {r.group} | {r.model} | " + " | ".join(f(r, c) for c in cols) + f" | {r.n_px} | {r.n_img} |" for _, r in t[sel].iterrows()])
    out += sec("Appendix — RMSE in Euclidean distance", t.group == "all", ["rmse", "rmse_euc"])
    out += sec("Breakdown ① distance bins (indoor 0–2 / 2–4 / ≥4 m, outdoor 0–10 / 10–30 / ≥30 m)", t.group.str.startswith("dist"), ["rmse", "absrel", "d1"])
    out += sec("Breakdown ② boundary vs interior (iBims-1 official edges, NYUv2 GT depth ratio > 1.1; within 3 px)", t.group.str.startswith("region"), ["rmse", "absrel", "d1"])
    out += sec("Breakdown ③ log metrics (SILog = KITTI definition, per image; predictions clipped to the cap range for log metrics only)",
               (t.group == "all") & (t.dataset != "mean"), ["logrmse", "silog"])
    v = pd.DataFrame(val)
    out += ["\n### Check — DepthVLM official dense protocol (canonical f=1000, per-image δ1) vs DepthVLM README dense table\n",
            "| Dataset | Model | δ1 (official protocol) | paper δ1 |", "|:--|:--|--:|--:|"]
    out += [f"| {r.dataset} | {r.model} | {r.d1_canon:.3f} | {'' if pd.isna(r.ref) else f'{r.ref:.3f}'} |" for _, r in v.iterrows()]
    open(a.out + ".md", "w").write("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
