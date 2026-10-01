"""DepthVLM-Bench nuScenes(test) 1,000장에 필요한 키프레임 파일 목록 → need_files.txt.
벤치 jsonl 의 (scene, timestamp) 로 v1.0-test 의 sample 을 찾고, 공식 추출 스크립트가 읽는 6개 카메라 + LIDAR_TOP 파일을 적는다.
사용: python prep/nuscenes_need_files.py ~/data/nuscenes_test bench/nuscenes_pixel_depth_test.jsonl"""
import json, os, sys
root, bench = os.path.expanduser(sys.argv[1]), sys.argv[2]
L = lambda n: json.load(open(f"{root}/v1.0-test/{n}.json"))
scene = {s["token"]: s["name"] for s in L("scene")}
by_key = {(scene[s["scene_token"]], s["timestamp"]): s for s in L("sample")}
want, miss = set(), 0
for r in map(json.loads, open(bench)):
    s = by_key.get((r["scene"], int(os.path.basename(r["image"]).split("_")[0])))
    if s is None: miss += 1
    else: want.add(s["token"])
need = {d["filename"] for d in L("sample_data") if d["is_key_frame"] and d["sample_token"] in want and "RADAR" not in d["filename"]}
open(f"{root}/need_files.txt", "w").write("\n".join(sorted(need)) + "\n")
print(f"bench 레코드 중 sample 못 찾음 {miss}, 필요한 파일 {len(need)} → {root}/need_files.txt")
