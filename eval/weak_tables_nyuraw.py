"""NYUv2 원측정 GT 판 약점 표 (NOTES F-17) — weak_tables 의 함수를 그대로 써서 dataset = 'nyuv2_raw' 를 만든다.
출력: results_vlm_weakness/nyuraw_points_ratio.csv, nyuraw_dense_ratio.csv, nyuraw_pairs_order_wrong.csv, nyuraw_compression.csv, 00_summary/heat_nyuraw_shape.png"""
import os
import numpy as np, pandas as pd
import weak_common as W
import weak_tables as T

D = os.path.join(W.OUT, "data")
CAP = dict(W.CAP)
W.CAP["nyuv2_raw"] = W.CAP["nyuv2"]
dense = pd.read_parquet(os.path.join(D, "dense_nyuraw.parquet"))
dense["model"] = dense.model.map(W.SHORT)
pts = pd.read_parquet(os.path.join(D, "points_nyuraw.parquet")).reset_index(drop=True)
ps = T.points_as_sums(pts)
G = T.point_groups(pts)
rows = []
for name, sel in G.items():
    idx = np.nonzero(sel)[0]
    if len(idx):
        x = ps[ps.idx.isin(idx)].groupby(["dataset", "image_id", "model"], as_index=False)[["n", "se", "ar", "d1", "dl", "dl2", "lg", "lg2", "dlg"]].sum()
        rows.append(x.assign(group=name))
pt_sums = pd.concat(rows, ignore_index=True)
ptab = T.group_table(pt_sums, T.image_params(pt_sums))
dtab = T.group_table(dense, T.image_params(dense))
pr = T.ratio_view(ptab, T.VL, min_n=100)
dr = T.ratio_view(dtab, ["DepthVLM", "UniDepthV2+K", "DepthPro+f"], min_n=20000)
pr.to_csv(os.path.join(W.OUT, "nyuraw_points_ratio.csv"), index=False)
dr.to_csv(os.path.join(W.OUT, "nyuraw_dense_ratio.csv"), index=False)
ptab.to_csv(os.path.join(W.OUT, "nyuraw_points_groups.csv"), index=False)
dtab.to_csv(os.path.join(W.OUT, "nyuraw_dense_groups.csv"), index=False)
W.SETS = ["nyuv2_raw"]
oa, ob = T.pairs(pts)
oa.to_csv(os.path.join(W.OUT, "nyuraw_pairs_order_wrong.csv"))
T.heatmap(pd.concat([pr.assign(model=pr.model), dr[dr.model == "DepthVLM"].assign(model="DepthVLM(전체 픽셀)")]), "shape_ratio",
          [("nyuv2_raw", "DepthLM"), ("nyuv2_raw", "DepthVLM"), ("nyuv2_raw", "DepthVLM(전체 픽셀)")],
          "NYUv2 원측정 GT: 모양 오차 ÷ pure vision 중앙값", os.path.join(W.OUT, "00_summary", "heat_nyuraw_shape.png"))
sel = ["all", "dist:near", "dist:mid", "dist:far", "region:boundary", "region:interior", "obj:wall", "obj:floor_ground", "obj:furniture",
       "obj:small_object", "obj:wall_mounted", "obj:window_door", "tex:low", "tex:high", "size:small"]
print("=== 공통 점 모양 비 (원측정 GT)"); print(pr.pivot_table(index="group", columns="model", values="shape_ratio").reindex(sel).round(2).to_string())
print("=== 전체 픽셀 DepthVLM 모양 비 (원측정 GT)"); print(dr[dr.model == "DepthVLM"].set_index("group").shape_ratio.reindex(sel).round(2).to_string())
print("=== 공통 점 치우침 % (원측정 GT)"); print(ptab.pivot_table(index="group", columns="model", values="bias").reindex(["dist:near", "dist:mid", "dist:far"]).round(1).to_string())
print("=== 앞뒤 오답률 % (원측정 GT)"); print(oa.round(2).to_string())
