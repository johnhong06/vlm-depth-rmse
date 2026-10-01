"""H200 전달용 팩: 데이터셋마다 벤치 레코드가 읽는 파일(RGB·깊이·마스크)만 tar 로 묶어 2 GB 조각(vdr_<ds>.tar.part_NN) + vdr_<ds>.sha256.
tar 안 경로 = 벤치 jsonl 의 상대 경로 그대로 (H200 에서 --data_root 하나로 읽힌다). 심볼릭 링크는 실제 파일로 담는다.
사용: python prep/pack.py ibims1 ~/data/depthvlm_bench ~/data/h200_staging/vdr [키 목록, 기본 image,depth_path,mask_valid_path,mask_transp_path]
ETH3D 깊이 맵은 장당 97 MB(합 44 GB)라 Track A 용으로는 RGB 만 묶는다: ... eth3d ... image"""
import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../eval"))
from common import load_bench  # noqa: E402

ds, root, out = sys.argv[1], os.path.expanduser(sys.argv[2]), os.path.expanduser(sys.argv[3])
keys = (sys.argv[4] if len(sys.argv) > 4 else "image,depth_path,mask_valid_path,mask_transp_path").split(",")
recs, _ = load_bench(ds)
files = sorted({r[k] for r in recs for k in keys if r.get(k)})
missing = [f for f in files if not os.path.exists(os.path.join(root, f))]
assert not missing, f"없는 파일 {len(missing)}: {missing[:3]}"
os.makedirs(out, exist_ok=True)
lst = os.path.join(out, f"vdr_{ds}.files")
open(lst, "w").write("\n".join(files) + "\n")
name = f"vdr_{ds}.tar.part_"
subprocess.run(f"rm -f {out}/{name}* && tar -chf - -C {root} -T {lst} | split -b 2000M -d -a 2 - {out}/{name}", shell=True, check=True)
subprocess.run(f"cd {out} && sha256sum {name}* > vdr_{ds}.sha256", shell=True, check=True)
size = sum(os.path.getsize(os.path.join(out, f)) for f in os.listdir(out) if f.startswith(name))
print(f"{ds}: 파일 {len(files)} 개, 팩 {size / 2**20:.0f} MiB → {out}/{name}*")
print(open(os.path.join(out, f"vdr_{ds}.sha256")).read().strip())
