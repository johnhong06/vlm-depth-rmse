"""공식 extract_rgb_depth_ddad.py 를 val 중 벤치 1,000장이 속한 sample 에만 돌린다.
공식 스크립트는 val 전체(3,950 sample × 카메라 6대)를 돌고 본문이 __main__ 안에 있어서, 같은 dgp 호출
(SynchronizedSceneDataset, generate_depth_from_datum="lidar")과 공식 save_rgb_and_depth 를 그대로 쓰고 sample 목록만 줄인다.
사용: ~/venv/prep/bin/python prep/ddad_extract.py ~/data/ddad/ddad_train_val/ddad.json ~/data/depthvlm_bench/ddad_v2 <dgp 경로>"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ddad_json, out, dgp = map(os.path.expanduser, sys.argv[1:4])
sys.path[:0] = [dgp, os.path.join(HERE, "../third_party/DepthVLM/data_process/extract_rgb_depth")]
import extract_rgb_depth_ddad as X  # noqa: E402
from dgp.datasets.synchronized_dataset import SynchronizedSceneDataset  # noqa: E402

want = {(r["scene"], int(os.path.basename(r["image"]).split("_")[0])) for r in map(json.loads, open(os.path.join(HERE, "../bench/ddad_pixel_depth_val.jsonl")))}
ds = SynchronizedSceneDataset(ddad_json, split="val", datum_names=X.DATUMS, generate_depth_from_datum="lidar")
n = done = 0
for g in range(len(ds)):
    si, k, _ = ds.dataset_item_index[g]
    sc = ds.scenes[si]
    key = (os.path.basename(sc.directory), sc.get_sample(k).id.timestamp.ToMicroseconds())
    if key not in want:
        continue
    for d in ds[g][0]:
        if "CAMERA" in d.get("datum_name", ""):
            n += X.save_rgb_and_depth(d["rgb"], d["depth"].copy(), d["intrinsics"].copy(), key[0], key[1], d["datum_name"], "val", out)
    done += 1
print(f"벤치 sample {len(want)} 중 처리 {done}, 저장한 카메라 프레임 {n}")
