"""공식 extract_rgb_depth_nuscenes.py 를 test 분할 중 받아 둔 키프레임(벤치 1,000장이 속한 927 sample)에만 돌린다.
공식 스크립트는 test 전체(6,008 sample)를 돌기 때문에, 공식 함수(build_scene_tasks·_worker_init·process_one_sample)를 그대로 쓰고 작업 목록만 줄인다.
사용: ~/venv/prep/bin/python prep/nuscenes_extract.py ~/data/nuscenes_test ~/data/depthvlm_bench/nuscenes_v2"""
import os, sys
from functools import partial
from multiprocessing import Pool
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../third_party/DepthVLM/data_process/extract_rgb_depth"))
import extract_rgb_depth_nuscenes as X
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes

root, out = map(os.path.expanduser, sys.argv[1:3])
nusc = NuScenes(version="v1.0-test", dataroot=root, verbose=False)
test = set(create_splits_scenes()["test"])
tasks = X.build_scene_tasks(nusc, {s["token"]: "test" for s in nusc.scene if s["name"] in test})
have = lambda t: all(os.path.exists(os.path.join(root, nusc.get("sample_data", tok)["filename"])) for tok in nusc.get("sample", t[0])["data"].values() if "RADAR" not in nusc.get("sample_data", tok)["channel"])
tasks = [t for t in tasks if have(t)]
print(f"test sample 중 파일이 다 있는 것 {len(tasks)}")
with Pool(16, initializer=X._worker_init, initargs=(root, "v1.0-test")) as pool:
    n = sum(c for c, _ in pool.imap_unordered(partial(X.process_one_sample, out_folder=out), tasks))
print(f"추출한 카메라 프레임 {n}")
