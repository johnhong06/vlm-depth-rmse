"""iBims-1 공식 경계 지도(edges/*.png)를 <bench>/ibims1/ibims1_core_raw/edges 에 둔다 (경계/내부 집계, D-18). 데이터 팩에는 없어서
공식 배포처(TUM dataserv)에서 ibims1_core_raw.zip 을 받아 공식 sha512 와 맞을 때만 edges 만 푼다. 실패하면 경고만 하고 끝낸다 (iBims-1 경계 집계만 빠짐).
사용: python prep/ibims1_edges.py <bench 폴더>"""
import hashlib
import os
import sys
import urllib.request
import zipfile

URL = "https://dataserv.ub.tum.de/s/m1455541/download?path=%2F&files=ibims1_core_raw.zip"
SHA512 = "2641a837c82909e58a34af43d779051c38d57a848308733b10cda747f1e8347da035a0cf591422f0448b4c5dba6eff92ae6289aa5ba73e6d79003d33315a80c8"  # Ibims_Dataset.sha512
dst = os.path.join(sys.argv[1], "ibims1")
if os.path.isdir(os.path.join(dst, "ibims1_core_raw", "edges")):
    sys.exit(print("[edges] iBims-1 경계 지도 이미 있음"))
tmp = os.path.join(sys.argv[1], "ibims1_core_raw.zip")
try:
    urllib.request.urlretrieve(URL, tmp)
    h = hashlib.sha512(open(tmp, "rb").read()).hexdigest()
    assert h == SHA512, f"sha512 불일치 {h[:16]}"
    with zipfile.ZipFile(tmp) as z:
        names = [n for n in z.namelist() if n.startswith("ibims1_core_raw/edges/")]
        z.extractall(dst, names)
    print(f"[edges] iBims-1 경계 지도 {len(names) - 1} 장 (공식 배포본, sha512 일치)")
except Exception as e:
    print(f"!!! [edges] iBims-1 경계 지도를 받지 못함 ({e}) — iBims-1 경계/내부 집계만 빠지고 나머지는 그대로")
finally:
    if os.path.exists(tmp):
        os.remove(tmp)
