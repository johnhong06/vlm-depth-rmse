"""준비한 원본으로 DepthVLM-Bench 의 GT 를 재현하는지 확인하고, 이미지별 원본 intrinsics 사이드카를 만든다.
GT 는 공식 sample_points.py 의 load_depth_raw(마스크·데이터셋별 min/max) 로 읽어 jsonl 의 depth 와 비교한다 (jsonl 은 소수 넷째 자리 반올림).
intrinsics(fx, fy, cx, cy) 는 공개 jsonl 에 fx 만 있어서 원본 보정 파일에서 읽는다 → bench/intrinsics_<ds>.json
사용: python prep/verify_bench.py ibims1 ~/data/depthvlm_bench"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "../third_party/DepthVLM/data_process/sample_test_points"))
sys.path.insert(0, os.path.join(HERE, "../eval"))
from common import BENCH, ROOT  # noqa: E402
from sample_points import load_depth_raw, match_dataset_config  # noqa: E402


def colmap_k(img_path):
    """ETH3D: 보정본 calib (COLMAP cameras.txt PINHOLE + images.txt) — 공식 create_data_pixel_level_eth3d.py 와 같은 파일."""
    scene_dir = img_path.split("/images/")[0]
    cams = {int(p[0]): [float(v) for v in p[4:8]] for p in (l.split() for l in open(f"{scene_dir}/dslr_calibration_undistorted/cameras.txt")) if p and p[0] != "#"}
    lines = [l.split() for l in open(f"{scene_dir}/dslr_calibration_undistorted/images.txt") if l.strip() and not l.startswith("#")]
    return cams[{os.path.basename(p[9]): int(p[8]) for p in lines[::2]}[os.path.basename(img_path)]]


def intrinsics(ds, root, r):
    p = os.path.join(root, r["image"])
    if ds == "ibims1":  # calib/<name>.txt = "fx,fy,cx,cy"
        return [float(v) for v in open(p.replace("/rgb/", "/calib/").replace(".png", ".txt")).read().split(",")[:4]]
    if ds in ("nuscenes", "ddad", "nyuv2"):  # 공식 추출 스크립트의 intrinsics/... .json = [fx, fy, cx, cy, W, H]
        return json.load(open(p.replace("/rgb/", "/intrinsics/").rsplit(".", 1)[0] + ".json"))[:4]
    if ds == "sunrgbd":  # 프레임 폴더의 intrinsics.txt = 3×3 행 우선 "fx 0 cx 0 fy cy 0 0 1"
        v = [float(x) for x in open(os.path.join(os.path.dirname(os.path.dirname(p)), "intrinsics.txt")).read().split()]
        return [v[0], v[4], v[2], v[5]]
    if ds == "eth3d":
        return colmap_k(p)


ds, root = sys.argv[1], os.path.expanduser(sys.argv[2])
path = os.path.join(ROOT, "bench", BENCH[ds][0])
cfg = match_dataset_config(path)
K, err, bad = {}, [], 0
for r in map(json.loads, open(path)):
    d = load_depth_raw(os.path.join(root, r["depth_path"]), r["depth_scale"], cfg["min_depth"], cfg["max_depth"],
                       os.path.join(root, r["mask_valid_path"]) if r.get("mask_valid_path") else None,
                       os.path.join(root, r["mask_transp_path"]) if r.get("mask_transp_path") else None, r.get("depth_format"))
    if d is None:
        bad += 1
        continue
    u, v = np.array(r["pixel_coords"]).T
    assert d.shape == tuple(r["original_depth_size"][::-1]), (r["image"], d.shape)
    err.append(np.abs(d[v, u] - np.array(r["depth"])))
    K[r["image"]] = k = intrinsics(ds, root, r)
    assert abs(k[0] - r["original_fx"]) < 0.01, (r["image"], k[0], r["original_fx"])  # 사이드카 fx = jsonl fx
e = np.concatenate(err)
print(f"{ds}: 이미지 {len(K)} (읽기 실패 {bad}), 점 {e.size}, |GT−jsonl| 최대 {e.max():.5f} m, >0.001 m 인 점 {(e > 1e-3).sum()}")
json.dump(K, open(os.path.join(ROOT, "bench", f"intrinsics_{ds}.json"), "w"))
