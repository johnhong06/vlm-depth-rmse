"""VLM 약점 분석 공통 (NOTES F-16) — 두 VLM(DepthLM·DepthVLM)이 모두 학습하지 않은 3 세트(iBims-1·NYUv2·DIODE Outdoor)에서
이미지마다 GT·예측 맵·DepthLM 점·속성 지도(거리·경계·물체·질감·위치·평면·물체 크기·NYU 채움 여부)를 한 곳에서 만든다.
예측 맵 = eval/dense_full.py --save_maps (GT 원본 크기, 평가와 같은 bilinear), DepthLM = Track A 원자료(874·875)의 공식 정의 답.
"""
import glob
import json
import os
import re
import sys

import cv2
import h5py
import numpy as np
import pandas as pd
from PIL import Image

from breakdown import BIN_NAMES, BINS, CAP
from common import BENCH, ROOT, load_bench
from dense_sparse import gt_map
from score import depthlm_answers

sys.path.insert(0, os.path.join(ROOT, "prep"))
from boundary_labels import R, contour_map  # noqa: E402

DATA = os.path.expanduser("~/data/depthvlm_bench")
MAPS = os.path.expanduser("~/data/vdr_maps")
RAW = os.path.expanduser("~/data/vdr_raw")              # 874·875·878 원자료 사본 (원본 zip 은 ~/Documents)
OUT = os.path.join(ROOT, "results_vlm_weakness")        # results*/ 라 git 에서 빠짐
NYU_MAT = os.path.expanduser("~/data/nyuv2/nyu_depth_v2_labeled.mat")
SETS = ["ibims1", "nyuv2", "diode_outdoor"]             # 두 VLM·pure vision 4 종 모두 zero-shot (PROTOCOL 2 절, D-19)
PV = ["UniDepthV2-L", "Metric3Dv2-L", "DepthPro", "DAv2-metric-L"]
DENSE = ["DepthVLM-4B"] + PV
SUPP = ["UniDepthV2-L+K", "DepthPro+f"]                 # 공정성 대조: GT intrinsics 를 준 pure vision (부록)
ALLM = ["DepthLM-12B"] + DENSE
SHORT = {"DepthLM-12B": "DepthLM", "DepthVLM-4B": "DepthVLM", "UniDepthV2-L": "UniDepthV2", "Metric3Dv2-L": "Metric3Dv2",
         "DepthPro": "DepthPro", "DAv2-metric-L": "DAv2", "UniDepthV2-L+K": "UniDepthV2+K", "DepthPro+f": "DepthPro+f"}
SIZE_BINS = [0, 0.01, 0.05, 1.01]                       # 물체 크기 = 이미지 면적 비: 작음 < 1 % ≤ 중간 < 5 % ≤ 큼
SIZE_NAMES = ["small", "medium", "large"]

# 물체·영역 대분류 — ADE20K 150 종(분할 모델)과 NYUv2 894 종(GT)을 같은 이름으로 묶는다. 나머지 물건은 small_object.
CATS = {
    "wall": ["wall"],
    "floor_ground": ["floor", "flooring", "rug", "carpet", "carpeting", "floor mat", "mat", "road", "route", "sidewalk", "pavement", "earth", "ground",
                     "field", "sand", "path", "dirt track", "land", "soil", "grass", "runway"],
    "ceiling": ["ceiling"],
    "window_door": ["window", "windowpane", "door", "double door", "curtain", "drape", "drapery", "blind", "blinds", "screen door", "shower curtain"],
    "furniture": ["table", "chair", "sofa", "couch", "bed", "cabinet", "desk", "shelf", "shelves", "counter", "bookcase", "bookshelf", "wardrobe",
                  "closet", "dresser", "chest of drawers", "armchair", "seat", "bench", "stool", "night stand", "ottoman", "drawer", "piano",
                  "coffee table", "countertop", "kitchen island", "swivel chair", "buffet", "sideboard", "cushion", "bathtub", "toilet",
                  "sink", "refrigerator", "refridgerator", "stove", "oven", "fireplace", "kitchen counter", "pool table", "crib"],
    "wall_mounted": ["picture", "painting", "poster", "whiteboard", "cork board", "classroom board", "bulletin board", "board", "map",
                     "calendar", "television", "television receiver", "tv", "monitor", "screen", "crt screen", "sconce", "clock"],
    "mirror": ["mirror"],
    "building": ["building", "edifice", "house", "skyscraper", "tower", "hovel", "hut", "shed", "column", "pillar"],
    "vegetation": ["tree", "plant", "flora", "flower", "palm", "bush", "hedge", "shrub"],
    "sky": ["sky"],
    "thin": ["pole", "fence", "fencing", "railing", "rail", "bannister", "banister", "streetlight", "street lamp", "traffic light",
             "traffic signal", "signboard", "sign"],
    "person_vehicle": ["person", "individual", "car", "automobile", "truck", "bus", "van", "bicycle", "bike", "minibike", "motorbike", "boat"],
    "stairs": ["stairs", "steps", "stairway", "staircase", "step", "stair", "escalator"],
}
CAT_NAMES = list(CATS) + ["small_object", "other"]
OTHER = ["unknown", "unlabeled", "", "mountain", "hill", "rock", "stone", "water", "sea", "river", "lake", "waterfall", "swimming pool",
         "bridge", "pier", "grandstand", "stage", "awning", "canopy", "tent", "booth", "fountain"]   # 물건이 아닌 지형·구조물 = 기타
STUFF = {"wall", "floor_ground", "ceiling", "building", "vegetation", "sky", "window_door", "stairs", "other"}   # 크기를 따지지 않는 분류


def stem(image):
    return image.rsplit(".", 1)[0].replace("/", "__")


def cat_of(name):
    """라벨 이름(ADE 는 'a, b, c' 동의어 목록) → 대분류."""
    names = [s.strip().lower() for s in name.split(",")]
    if names[0] in OTHER:
        return "other"
    for c, keys in CATS.items():
        if any(n in keys for n in names):
            return c
    return "small_object"


_ADE = None


def ade_lut():
    """ADE id(0–149) → 대분류 번호. 이름은 분할 모델 설정(prep/semseg_ade.py 가 쓴 것과 같은 체크포인트)에서 읽는다."""
    global _ADE
    if _ADE is None:
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained("facebook/mask2former-swin-large-ade-semantic")
        _ADE = np.array([CAT_NAMES.index(cat_of(cfg.id2label[i])) for i in range(len(cfg.id2label))], np.uint8)
    return _ADE


_NYU = {}


def nyu_file():
    if "f" not in _NYU:
        f = h5py.File(NYU_MAT, "r")
        names = ["".join(chr(c) for c in f[f["names"][0, i]][:].flatten()) for i in range(f["names"].shape[1])]
        _NYU.update(f=f, lut=np.array([CAT_NAMES.index("other")] + [CAT_NAMES.index(cat_of(n)) for n in names], np.uint8))
    return _NYU


def nyu_offset(r, g):
    """SUN RGB-D NYU 561×427 의 원본 640×480 안 위치 — 원측정(rawDepths) 픽셀에서 GT 와 가장 잘 맞는 자리 (이미지마다 1 px 정도 다름, NOTES F-15)."""
    k = int(re.search(r"NYU(\d+)", r["image"]).group(1)) - 1
    raw = nyu_file()["f"]["rawDepths"][k].T
    H, W = g.shape
    best = None
    for y in range(43, 49):
        for x in range(41, 48):
            rc = raw[y:y + H, x:x + W]
            m = (rc > 0) & (g > 0)
            e = np.median(np.abs(np.log(rc[m] / g[m])))
            if best is None or e < best[0]:
                best = (e, y, x)
    return k, best[1], best[2], best[0]


def nyu_labels(r, g):
    """NYUv2 GT 의미 라벨(대분류)·물체 인스턴스·원측정 여부 (GT 와 같은 561×427 격자)."""
    k, oy, ox, err = nyu_offset(r, g)
    f = nyu_file()["f"]
    H, W = g.shape
    lab = f["labels"][k].T[oy:oy + H, ox:ox + W]
    inst = f["instances"][k].T[oy:oy + H, ox:ox + W]
    raw = f["rawDepths"][k].T[oy:oy + H, ox:ox + W] > 0
    return nyu_file()["lut"][lab], lab.astype(np.int32) * 1000 + inst.astype(np.int32), raw, err


def ade_labels(r, shape):
    lab = np.array(Image.open(os.path.join(MAPS, "seg", r["_ds"], stem(r["image"]) + ".png")))
    assert lab.shape == shape, (lab.shape, shape)
    cat = ade_lut()[lab]
    # 물체 인스턴스 = 같은 ADE 라벨의 연결 성분 (GT 인스턴스가 없어 근사)
    inst = np.zeros(shape, np.int32)
    for c in np.unique(lab):
        n, cc = cv2.connectedComponents((lab == c).astype(np.uint8), connectivity=8)
        inst[lab == c] = (int(c) + 1) * 1000 + cc[lab == c]
    return cat, inst


def texture(path):
    """회색조 기울기 크기의 15×15 평균 (eval/region_errors.py 와 같은 정의)."""
    gr = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY).astype(np.float32)
    return cv2.blur(np.hypot(cv2.Sobel(gr, cv2.CV_32F, 1, 0), cv2.Sobel(gr, cv2.CV_32F, 0, 1)), (15, 15))


def ibims_planes(r, shape):
    """iBims-1 공식 평면 마스크 → (종류 지도 'floor'/'wall'/'table'/'', 인스턴스 id 지도, {id: (종류, 법선, D)})."""
    name = os.path.splitext(os.path.basename(r["image"]))[0]
    base = os.path.join(DATA, os.path.dirname(os.path.dirname(r["image"])))
    kind, inst, par = np.full(shape, "", object), np.zeros(shape, np.int32), {}
    for j, p in enumerate(["floor", "wall", "table"]):
        f = os.path.join(base, f"mask_{p}", name + ".png")
        if not os.path.exists(f):
            continue
        m = cv2.imread(f, cv2.IMREAD_UNCHANGED)
        txt = f[:-4] + ".txt"                              # 평면식 파일이 없는 마스크도 있다 (평면은 GT 점으로 다시 맞추므로 없어도 됨)
        rows = [list(map(float, l.split(","))) for l in open(txt) if l.strip()] if os.path.exists(txt) else []
        for q in range(1, int(m.max()) + 1):
            sel = m == q
            if sel.any():
                pid = (j + 1) * 100 + q
                kind[sel], inst[sel] = p, pid
                par[pid] = (p, np.array(rows[q - 1][:3]) if q <= len(rows) else None, rows[q - 1][3] if q <= len(rows) else None)
    return kind, inst, par


def records(ds):
    recs, K = load_bench(ds)
    for r in recs:
        r["_ds"], r["_k"] = ds, K[r["image"]]
    return recs


def load_map(model, ds, r):
    return np.load(os.path.join(MAPS, model, ds, stem(r["image"]) + ".npy")).astype(np.float32)


def depthlm_points():
    """Track A 공식 정의 DepthLM 답 (공통 점, 3 세트). 열: dataset, image_id, u, v, gt_z, pred."""
    fs = [f for f in glob.glob(f"{RAW}/87[45]/vdr/track_a/depthlm_*.parquet") if any(f"depthlm_{s}." in f for s in SETS)]
    d = pd.concat([pd.read_parquet(f, columns=["dataset", "image_id", "u", "v", "gt_z", "pred", "pred_raw", "model"]) for f in fs])
    return depthlm_answers(d, "official").drop(columns="pred_raw")


def attributes(ds, r, g):
    """속성 지도 묶음 (모두 GT 크기). 없는 속성은 None."""
    path = os.path.join(DATA, r["image"])
    H, W = g.shape
    dom = BENCH[ds][1]
    a = {"dist": np.digitize(g, BINS[dom][1:-1]).astype(np.int8)}            # 0 근 / 1 중 / 2 원
    cm = contour_map(ds, r, DATA, g) if ds in ("ibims1", "nyuv2") else None   # DIODE 는 경계 제외 (D-18)
    a["boundary"] = None if cm is None else cv2.distanceTransform((~cm).astype(np.uint8), cv2.DIST_L2, 5) <= R
    if ds == "nyuv2":
        a["cat"], a["inst"], a["nyu_raw"], a["nyu_align"] = nyu_labels(r, g)
    else:
        a["cat"], a["inst"] = ade_labels(r, g.shape)
        a["nyu_raw"] = None
    # 물체 크기 (stuff 가 아닌 대분류만)
    size = np.full(g.shape, -1, np.int8)
    thing = ~np.isin(a["cat"], [CAT_NAMES.index(c) for c in STUFF])
    ids, inv, cnt = np.unique(a["inst"][thing], return_inverse=True, return_counts=True)
    size[thing] = np.digitize(cnt[inv] / (H * W), SIZE_BINS[1:-1])
    a["size"] = size
    t = texture(path)
    vg = g > 0
    q = np.quantile(t[vg], [1 / 3, 2 / 3]) if vg.any() else [0, 0]
    a["tex"] = np.digitize(t, q).astype(np.int8)                                # 이미지 안 3 등분: 0 적음 / 1 중간 / 2 많음
    yy, xx = np.mgrid[:H, :W]
    a["border"] = np.minimum(np.minimum(xx, W - 1 - xx) / W, np.minimum(yy, H - 1 - yy) / H) < 0.1   # 가장자리 띠 10 %
    a["vpos"] = np.minimum(yy * 3 // H, 2).astype(np.int8)                        # 0 위 / 1 가운데 / 2 아래
    if ds == "ibims1":
        a["plane"], a["plane_id"], a["plane_par"] = ibims_planes(r, g.shape)
    return a


def groups(ds, a):
    """집단 이름 → 픽셀 선택 지도 (valid 와 AND 해서 쓴다)."""
    G = {"all": None}
    for j, n in enumerate(BIN_NAMES):
        G[f"dist:{n}"] = a["dist"] == j
    if a["boundary"] is not None:
        G["region:boundary"], G["region:interior"] = a["boundary"], ~a["boundary"]
    for j, c in enumerate(CAT_NAMES):
        if c != "sky":
            G[f"obj:{c}"] = a["cat"] == j
    for j, n in enumerate(SIZE_NAMES):
        G[f"size:{n}"] = a["size"] == j
    for j, n in enumerate(["low", "mid", "high"]):
        G[f"tex:{n}"] = a["tex"] == j
    G["pos:border"], G["pos:center"] = a["border"], ~a["border"]
    for j, n in enumerate(["top", "middle", "bottom"]):
        G[f"vpos:{n}"] = a["vpos"] == j
    if a.get("nyu_raw") is not None:
        G["nyu:measured"], G["nyu:filled"] = a["nyu_raw"], ~a["nyu_raw"]
    if ds == "ibims1":
        for p in ["floor", "wall", "table"]:
            G[f"plane:{p}"] = a["plane"] == p
        G["plane:none"] = a["plane"] == ""
    return G


def logerr(p, g, ds):
    """d = ln 예측 − ln GT, 예측은 log 에서만 cap 범위로 자른다 (D-18)."""
    return np.log(np.clip(p, *CAP[ds])) - np.log(g)
