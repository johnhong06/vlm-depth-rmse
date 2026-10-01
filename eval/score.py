"""픽셀 원자료(parquet, common.COLS) → RMSE·AbsRel·δ1 + 이미지 단위 bootstrap 95 % CI. 모든 수치는 이 스크립트가 parquet 에서만 계산한다.
  공통 집합(D-2): 데이터셋마다 비교하는 모든 모델이 유한한 예측을 낸 (image_id, u, v) 만 쓴다. 제외 수를 표에 적는다.
  주 결과 = z-depth 공간 (pred vs gt_z). 부록 = 유클리드 공간 RMSE (둘 다 광선 계수를 곱함 — AbsRel·δ1 은 비율이라 공간과 무관, 규칙 8).
  주 지표 = 데이터셋 안 모든 픽셀 pooled, 보조 = 이미지별 계산 후 평균 (규칙 6). 도메인 행 = 그 도메인 데이터셋 값의 평균 (실내/실외만, 규칙 5).
  bootstrap: 데이터셋마다 이미지를 복원추출(B 회, 시드 0), 도메인 값은 같은 회차의 데이터셋 값 평균 (규칙 7).
사용: python eval/score.py results/track_a/*.parquet --models DepthLM-12B UniDepthV2-L ... --out tables/track_a
"""
import argparse
import os

import numpy as np
import pandas as pd

from common import BENCH, ray

# 추론 조건 (사용자 설계표 + 공식 코드 확인, NOTES D-13). 학습 겹침은 각 논문의 학습 데이터 목록에서 확인한 것만 (NOTES D-12)
META = {
    "DepthLM-12B": ("yes", "no"),
    "DAv2-metric-L": ("no", "yes (indoor Hypersim / outdoor VKITTI model)"),
    "UniDepthV2-L": ("no", "no"),
    "Metric3Dv2-L": ("yes", "no"),
    "DepthPro": ("no", "no"),
}
TRAINED = {("Metric3Dv2-L", "ddad"): "trained on DDAD", ("DepthLM-12B", "nuscenes"): "trained on nuScenes (other scenes)"}
CEILING = {("DAv2-metric-L", "ibims1"): "model max 20 m < cap 25 m", ("DAv2-metric-L", "ddad"): "model max 80 m < cap 120 m"}
NAMES = ["rmse", "absrel", "d1", "rmse_euc"]


def stats(g):
    """이미지별 충분통계: 픽셀 수, z 제곱오차 합, 상대오차 합, δ1 적중 합, 유클리드 제곱오차 합."""
    e, r = g.pred - g.gt_z, ray(g.u, g.v, (g.fx, g.fy, g.cx, g.cy))
    t = pd.DataFrame(dict(image_id=g.image_id, n=1, se=e ** 2, ar=e.abs() / g.gt_z,
                          d1=(g.pred > 0) & (np.maximum(g.pred / g.gt_z, g.gt_z / g.pred) < 1.25), se_euc=(e * r) ** 2))  # 0·음수 = 오답
    return t.groupby("image_id").sum()


def metrics(s, w):
    """w: (B, 이미지수) 이미지 가중치. 열 = pooled 4개 + per-image 4개."""
    n = w @ s.n.values
    pooled = [np.sqrt(w @ s.se.values / n), w @ s.ar.values / n, w @ s.d1.values / n, np.sqrt(w @ s.se_euc.values / n)]
    per = [w @ v.values / w.sum(1) for v in (np.sqrt(s.se / s.n), s.ar / s.n, s.d1 / s.n, np.sqrt(s.se_euc / s.n))]
    return np.stack(pooled + per, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--B", type=int, default=2000)
    a = ap.parse_args()
    d = pd.concat([pd.read_parquet(f, columns=["dataset", "image_id", "u", "v", "fx", "fy", "cx", "cy", "gt_z", "pred", "model"]) for f in a.files])
    d = d[d.model.isin(a.models)]
    dup = d.duplicated(["dataset", "image_id", "u", "v", "model"])
    assert not dup.any(), f"같은 (dataset, image_id, u, v, model) 행이 {dup.sum()} 개 겹친다 — 출력 폴더를 섞지 말 것"
    rng, rows, boot = np.random.default_rng(0), [], {}
    for ds in [k for k in BENCH if k in set(d.dataset)]:
        x = d[d.dataset == ds]
        if x.model.nunique() < len(a.models):
            print(f"[score] {ds}: 아직 돌지 않은 모델 {sorted(set(a.models) - set(x.model))} — 공통 집합을 만들 수 없어 건너뜀")
            continue
        ok = x[np.isfinite(x.pred)].groupby(["image_id", "u", "v"]).model.nunique() == len(a.models)
        keep, n_all = ok[ok].index, x.groupby(["image_id", "u", "v"]).ngroups
        imgs = np.unique(keep.get_level_values(0))
        if not len(imgs):
            print(f"[score] {ds}: 모든 모델이 답한 공통 점이 0 개 — 건너뜀")
            continue
        w = np.vstack([np.ones(len(imgs)), rng.multinomial(len(imgs), np.full(len(imgs), 1 / len(imgs)), a.B)])  # 0행 = 원 표본
        for m in a.models:
            r = metrics(stats(x[x.model == m].set_index(["image_id", "u", "v"]).loc[keep].reset_index()).loc[imgs], w)
            boot[ds, m] = r[1:]
            rows.append(dict(dataset=ds, domain=BENCH[ds][1], model=m, n_px=len(keep), n_excl=n_all - len(keep), n_img=len(imgs),
                             **dict(zip(NAMES + [k + "_img" for k in NAMES], r[0])),
                             **{k + "_lo": v for k, v in zip(NAMES, np.percentile(r[1:, :4], 2.5, 0))},
                             **{k + "_hi": v for k, v in zip(NAMES, np.percentile(r[1:, :4], 97.5, 0))}))
    if not rows:
        raise SystemExit("[score] 표에 넣을 데이터셋이 없다 (모든 모델이 끝난 데이터셋 0 개)")
    t = pd.DataFrame(rows)
    for (dom, m), g in t.groupby(["domain", "model"], sort=False):
        b = np.mean([boot[ds, m] for ds in g.dataset], 0)
        rows.append(dict(dataset="mean", domain=dom, model=m, n_px=g.n_px.sum(), n_excl=g.n_excl.sum(), n_img=g.n_img.sum(),
                         **g[NAMES + [k + "_img" for k in NAMES]].mean().to_dict(),
                         **{k + "_lo": v for k, v in zip(NAMES, np.percentile(b[:, :4], 2.5, 0))},
                         **{k + "_hi": v for k, v in zip(NAMES, np.percentile(b[:, :4], 97.5, 0))}))
    order = {ds: i for i, ds in enumerate(list(BENCH) + ["mean"])}
    t = pd.DataFrame(rows)
    t = t.iloc[np.lexsort((t.model.map(a.models.index), t.dataset.map(order), t.domain.map({"Indoor": 0, "Outdoor": 1})))]
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    t.to_csv(a.out + ".csv", index=False)
    f = lambda r, k, p: f"{r[k]:.{p}f} [{r[k + '_lo']:.{p}f}, {r[k + '_hi']:.{p}f}]"
    members = t[t.dataset != "mean"].groupby("domain").dataset.unique().to_dict()
    tags = lambda r: [s for ds in (members[r.domain] if r.dataset == "mean" else [r.dataset])  # 도메인 평균 행은 구성 데이터셋의 각주를 물려받는다
                      for s in (TRAINED.get((r.model, ds)), CEILING.get((r.model, ds))) if s]
    notes = {v: i + 1 for i, v in enumerate(dict.fromkeys(s for _, r in t.iterrows() for s in tags(r)))}  # 표에 나오는 각주만, 나오는 순서대로
    mark = lambda r: "".join(f"<sup>{notes[s]}</sup>" for s in dict.fromkeys(tags(r)))
    main_t = ["| Domain | Dataset | Model | GT intrinsics | Domain info | RMSE↓ (m) | AbsRel↓ | δ1↑ | per-image RMSE / AbsRel / δ1 | px (excl.) | img |",
              "|:--|:--|:--|:--|:--|:--|:--|:--|:--|--:|--:|"]
    app_t = ["| Domain | Dataset | Model | RMSE z-depth (m) | RMSE Euclidean (m) | per-image z / Euclidean |", "|:--|:--|:--|:--|:--|:--|"]
    for _, r in t.iterrows():
        gk, dom = META.get(r.model, ("?", "?"))
        main_t.append(f"| {r.domain} | {r.dataset} | {r.model}{mark(r)} | {gk} | {dom} | {f(r, 'rmse', 3)} | {f(r, 'absrel', 3)} | {f(r, 'd1', 3)} | "
                      f"{r.rmse_img:.3f} / {r.absrel_img:.3f} / {r.d1_img:.3f} | {r.n_px} ({r.n_excl}) | {r.n_img} |")
        app_t.append(f"| {r.domain} | {r.dataset} | {r.model} | {f(r, 'rmse', 3)} | {f(r, 'rmse_euc', 3)} | {r.rmse_img:.3f} / {r.rmse_euc_img:.3f} |")
    foot = "\n".join(f"{i}. {s}" for s, i in notes.items())
    note = ("Main metrics: z-depth, pooled over all common pixels of a dataset; brackets = 95 % CI from an image-level bootstrap "
            f"(B = {a.B}, seed 0). Per-image = metric per image, then averaged. Domain rows ('mean') average their datasets; "
            "indoor and outdoor are never averaged together. px (excl.) = common pixels (pixels dropped because some model gave no prediction).")
    open(a.out + ".md", "w").write("\n".join(main_t) + "\n\n" + note + "\n\n" + foot + "\n\n### Appendix — RMSE in Euclidean distance\n\n" + "\n".join(app_t) + "\n")
    print("\n".join(main_t) + "\n\n" + foot)


if __name__ == "__main__":
    main()
