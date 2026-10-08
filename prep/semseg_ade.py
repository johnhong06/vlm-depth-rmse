"""물체·영역별 오차 분석용 의미 분할 (NOTES F-16) — iBims-1·DIODE 는 GT 라벨이 없어 공개 모델로 만든다. NYUv2 는 GT 라벨과의 대조용.
모델: facebook/mask2former-swin-large-ade-semantic (ADE20K 150 종). 출력: <out>/<ds>/<이미지 경로의 / → __>.png (uint8, 값 = ADE id 0–149).
사용: ~/venv/main/bin/python prep/semseg_ade.py --datasets ibims1 diode_outdoor nyuv2 --data_root ~/data/depthvlm_bench --out ~/data/vdr_maps/seg
"""
import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "eval"))
from common import load_bench  # noqa: E402

NAME = "facebook/mask2former-swin-large-ade-semantic"

ap = argparse.ArgumentParser()
ap.add_argument("--datasets", nargs="+", required=True)
ap.add_argument("--data_root", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
proc = AutoImageProcessor.from_pretrained(NAME)
m = Mask2FormerForUniversalSegmentation.from_pretrained(NAME).to("cuda").eval()
for ds in a.datasets:
    recs, _ = load_bench(ds)
    os.makedirs(os.path.join(a.out, ds), exist_ok=True)
    for r in recs:
        img = Image.open(os.path.join(a.data_root, r["image"])).convert("RGB")
        with torch.no_grad():
            out = m(**proc(images=img, return_tensors="pt").to("cuda"))
        lab = proc.post_process_semantic_segmentation(out, target_sizes=[img.size[::-1]])[0].cpu().numpy().astype(np.uint8)
        Image.fromarray(lab).save(os.path.join(a.out, ds, r["image"].rsplit(".", 1)[0].replace("/", "__") + ".png"))
    print(ds, len(recs), flush=True)
print("labels:", m.config.id2label)
