"""H200: /app/data 아래에서 <name>.tar.part_* 조각을 찾아(파일로 있든, 관리자가 드라이브 폴더를 zip 으로 받아 그 안에 있든) 순서대로 tar 에 흘려 넣어 푼다.
SHA256(<name>.sha256 또는 SHA256SUMS_parts) 은 흘리면서 검증한다. 성공하면 <dest>/.done_<name> 을 남기고, 있으면 다시 풀지 않는다.
중간 복사본을 만들지 않는다. 결과 볼륨(/app/output)에는 절대 풀지 않는다 (run.sh 가 dest 를 정한다).
사용: python h200/unpack.py <name> <dest> [tar 안 경로 ...]"""
import hashlib
import os
import re
import subprocess
import sys
import zipfile

name, dest, members = sys.argv[1], sys.argv[2], sys.argv[3:]
mark = os.path.join(dest, f".done_{name}")
if os.path.exists(mark):
    sys.exit(print(f"[unpack] {name}: 이미 풀려 있음 ({dest})"))
pat, parts, sums = re.compile(re.escape(name) + r"\.tar\.part_(\w+)$"), {}, {}
for root in ["/app/data"] + [r for r in os.environ.get("DATA_SRC", "").split(":") if r]:
    for dp, dn, fn in os.walk(root, followlinks=True):
        if dp[len(root):].count(os.sep) >= 3:
            dn[:] = []
        for f in fn:
            p, srcs = os.path.join(dp, f), []
            if f.endswith(".zip") and zipfile.is_zipfile(p):
                z = zipfile.ZipFile(p)
                srcs = [(os.path.basename(n), (z, n)) for n in z.namelist()]
            else:
                srcs = [(f, (None, p))]
            for b, src in srcs:
                if m := pat.search(b):
                    parts.setdefault(m.group(1), src)
                elif b in (f"{name}.sha256", "SHA256SUMS_parts"):
                    txt = src[0].read(src[1]).decode() if src[0] else open(src[1]).read()
                    sums.update({l.split()[1].lstrip("*"): l.split()[0] for l in txt.splitlines() if name in l})
keys = sorted(parts)
if not keys or len(keys) != len(sums):
    sys.exit(f"!!! [unpack] {name}: 조각 {len(keys)} 개 / 체크섬 {len(sums)} 개 — 업로드가 덜 됐거나 관리자에게 아직 전달되지 않았다")
os.makedirs(dest, exist_ok=True)
print(f"[unpack] {name}: 조각 {len(keys)} 개 → {dest} {' '.join(members)}", flush=True)
tar, bad = subprocess.Popen(["tar", "-xf", "-", "-C", dest, *members], stdin=subprocess.PIPE), []
for k in keys:
    z, p = parts[k]
    h = hashlib.sha256()
    with (z.open(p) if z else open(p, "rb")) as f:
        while chunk := f.read(16 << 20):
            h.update(chunk)
            tar.stdin.write(chunk)
    if sums.get(f"{name}.tar.part_{k}") != h.hexdigest():
        bad.append(k)
tar.stdin.close()
if tar.wait() != 0 or bad:
    sys.exit(f"!!! [unpack] {name}: SHA256 불일치 조각 {bad} / tar 종료코드 {tar.returncode}")
open(mark, "w").close()
print(f"[unpack] {name}: SHA256 검증 통과, 풀기 완료", flush=True)
