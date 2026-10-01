"""Metric3Dv2 RMS 재현용 NYUv2 공식 테스트 654 장 (labeled.mat 640×480, splits.mat testNdxs) → RGB png + depths(보정)·rawDepths npy.
Track A 의 NYUv2 공통 점(SUN RGB-D 판 561×427)과는 별개 — 표준 프로토콜 재현 전용.
사용: python prep/nyu_official.py ~/data/nyuv2 ~/data/nyuv2_official"""
import os
import sys

import h5py
import numpy as np
import scipy.io as sio
from PIL import Image

src, out = map(os.path.expanduser, sys.argv[1:3])
test = sio.loadmat(os.path.join(src, "splits.mat"))["testNdxs"].ravel()  # 1 부터
f = h5py.File(os.path.join(src, "nyu_depth_v2_labeled.mat"), "r")
os.makedirs(out, exist_ok=True)
for i in test:
    j = int(i) - 1  # HDF5 는 열 우선 저장이라 전치
    Image.fromarray(f["images"][j].transpose(2, 1, 0)).save(os.path.join(out, f"{i:04d}_rgb.png"))
    np.save(os.path.join(out, f"{i:04d}_depth.npy"), f["depths"][j].T.astype(np.float32))
    np.save(os.path.join(out, f"{i:04d}_rawdepth.npy"), f["rawDepths"][j].T.astype(np.float32))
print(f"NYUv2 공식 테스트 {len(test)} 장 → {out}")
