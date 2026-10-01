"""Track A — DepthLM-12B 에 공통 점을 질의한다. 공식 eval.py 의 Pixtral 경로와 dataset_inference 전처리를 그대로 쓴다.
  전처리(공식, 규칙 3): 원본 → undistort_image → normalizing_focal_length(750 = 12B 공식값) → 원본 좌표를 같은 배율로 변환(Step 3) → 5 px 빨간 화살표(Step 5)
  생성(공식): 이미지 먼저 + 공식 질문, greedy. 답은 공식 delta1_metric 과 같은 math_verify parse(text)[0]
  화살표를 못 그리는 점(테두리 5 px)·파싱 실패는 pred=pred_raw=NaN + note 로 남긴다 → 채점에서 모든 모델 공통 제외 (NOTES D-2)
  깊이 정의(규칙 1): pred_raw = 답 d (유클리드 거리), pred = d / sqrt(1 + ((u-cx)/fx)^2 + ((v-cy)/fy)^2) (원본 좌표 + 원본 intrinsics)
출력: <out>/depthlm_<ds>.part<k>.parquet — 열 = common.COLS + note, text (생성 원문)
사용: python eval/depthlm_sparse.py --dataset ibims1 --data_root ~/data/depthvlm_bench --model <DepthLM 경로> --out results/track_a
"""
import argparse
import glob
import importlib
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from math_verify import parse
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "../third_party/DepthLM_Official"))
from utils.datasets import generate_prompt_depth_sft, normalizing_focal_length, undistort_image  # noqa: E402
from common import load_bench, ray  # noqa: E402

FOCAL, CROSS = 750.0, 5  # 공식 eval.py 의 normalized_focal_length, dataset_inference 의 cross_size


def marked(img, x, y):
    """공식 dataset_inference Step 5 와 같은 화살표를 사본에 그린다. 테두리 5 px 안이면 None (공식 코드도 건너뛴다)."""
    if not (CROSS <= x < img.width - CROSS and CROSS <= y < img.height - CROSS):
        return None
    img = img.copy()
    for dx in range(1, CROSS + 1):
        img.putpixel((x - dx, y), (255, 0, 0))
    for dy in range(1, CROSS // 2 + 1):
        img.putpixel((x - dy - 1, y + dy), (255, 0, 0))
        img.putpixel((x - dy - 1, y - dy), (255, 0, 0))
    return img


def prepare(path, k):
    """공식 dataset_inference Step 1–2: 원본 → undistort_image → normalizing_focal_length(750). (정규화 이미지, 정규화 intrinsics)"""
    img, kn = undistort_image(list(k), Image.open(path).convert("RGB"))  # RGB 이미지라 convert 는 무변화
    return normalizing_focal_length(FOCAL, kn, img)


def marker_xy(u, v, k, kn):
    """공식 Step 3: 원본 좌표 → 정규화 이미지 좌표 (같은 배율, 규칙 3)."""
    return int((u - k[2]) * (kn[0] / k[0]) + kn[2]), int((v - k[3]) * (kn[1] / k[1]) + kn[3])


def load(path, gpu_mem=""):
    """공식 eval.py 의 Pixtral 경로. 텍스트 flash_attention_2(공식) — flash-attn 이 없는 환경(로컬 Blackwell)만 sdpa, 비전 eager(공식). 둘 다 exact."""
    try:  # 설치만 되고 import 가 안 되는 휠이면 첫 generate 에서 죽으므로 실제 import 로 판단
        importlib.import_module("flash_attn")
        attn = "flash_attention_2"
    except Exception:
        attn = "sdpa"
    model = LlavaForConditionalGeneration.from_pretrained(
        path, torch_dtype=torch.bfloat16, attn_implementation={"text_config": attn, "vision_config": "eager"},
        device_map="auto", max_memory={0: gpu_mem, "cpu": "100GiB"} if gpu_mem else None).eval()
    return AutoProcessor.from_pretrained(path), model, attn


@torch.no_grad()
def ask(proc, model, images, max_new_tokens=128):
    """공식 eval.py: 이미지 먼저 + 공식 질문(GT 는 문장에 안 들어감), 왼쪽 패딩 배치, greedy. 생성 텍스트 목록."""
    problem = generate_prompt_depth_sft(0.0, is_eval=True)[0]
    chat = [[{"role": "user", "content": [{"type": "image", "image": im}, {"type": "text", "content": problem}]}] for im in images]
    inp = proc.apply_chat_template(chat, add_generation_prompt=True, tokenize=True, return_dict=True, padding=True,
                                   padding_side="left", return_tensors="pt").to("cuda", dtype=torch.bfloat16)
    ids = model.generate(**inp, max_new_tokens=max_new_tokens, do_sample=False, top_p=None, top_k=None)
    return proc.batch_decode(ids[:, inp["input_ids"].shape[1]:], skip_special_tokens=True, clean_up_tokenization_spaces=False)


def answer(text):
    """공식 delta1_metric 과 같은 파싱 (math_verify parse(text)[0]). 못 읽으면 NaN."""
    try:
        return float(parse(text)[0])
    except Exception:
        return np.nan


def queries(recs, K, root, shard, nshard, done):
    """(image_id, u, v, gt, 원본 intrinsics, 화살표 이미지|None) 를 이미지별 점 순서대로. 정규화 이미지는 이미지마다 한 번만 만든다."""
    for i, r in enumerate(recs):
        todo = [(u, v, g) for (u, v), g in zip(r["pixel_coords"], r["depth"]) if (r["image"], u, v) not in done]
        if i % nshard != shard or not todo:
            continue
        k = list(K[r["image"]])  # 원본 해상도 [fx, fy, cx, cy]
        img, kn = prepare(os.path.join(root, r["image"]), k)
        for u, v, g in todo:
            yield r["image"], u, v, g, k, marked(img, *marker_xy(u, v, k, kn))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshard", type=int, default=1)
    ap.add_argument("--bsz", type=int, default=16)
    ap.add_argument("--max_patches", type=int, default=10000,
                    help="배치 안 이미지 패치 수 합의 상한. Pixtral 비전 인코더는 배치 이미지를 한 시퀀스로 이어 eager attention(공식)으로 N×N 을 만든다")
    ap.add_argument("--max_new_tokens", type=int, default=128)  # 답은 30 토큰 안팎. 잘리면 아래 요약의 '</answer> 없음' 으로 드러난다
    ap.add_argument("--gpu_mem", default="", help="로컬 점검용 CPU 오프로드 (예: 9GiB). 비우면 GPU 에 통째로")
    ap.add_argument("--limit", type=int, default=0, help="질의 수 제한 (스모크)")
    a = ap.parse_args()

    recs, K = load_bench(a.dataset)
    os.makedirs(a.out, exist_ok=True)
    out = os.path.join(a.out, f"depthlm_{a.dataset}.part{a.shard}.parquet")
    rows = pd.read_parquet(out).to_dict("records") if os.path.exists(out) else []
    # 이미 답한 점은 모든 조각 파일에서 모은다 — 프로세스 수(nshard)를 바꿔 이어 돌려도 같은 점을 두 번 묻지 않게
    done = {k for f in glob.glob(os.path.join(a.out, f"depthlm_{a.dataset}.part*.parquet"))
            for k in pd.read_parquet(f, columns=["image_id", "u", "v"]).itertuples(index=False, name=None)}
    proc, model, attn = load(a.model, a.gpu_mem)
    print(f"[{a.dataset} shard {a.shard}/{a.nshard}] 이미 {len(done)} 점, attention text={attn} vision=eager, 초점 {FOCAL}", flush=True)

    def add(img_id, u, v, gt, k, text, note=""):
        d = answer(text)
        note = note or ("parse_fail" if np.isnan(d) else "")
        rows.append(dict(dataset=a.dataset, image_id=img_id, u=int(u), v=int(v), fx=k[0], fy=k[1], cx=k[2], cy=k[3], gt_z=float(gt),
                         pred=d / ray(u, v, k), pred_raw=d, model="DepthLM-12B", note=note, text=text))

    def save():
        pd.DataFrame(rows).to_parquet(out + ".tmp", index=False)
        os.replace(out + ".tmp", out)

    batch, n, t0 = [], 0, time.time()
    patches = lambda im: -(-im.width // 16) * -(-im.height // 16)  # 프로세서가 16 배수로 올림 (긴 변 2048 이하라 축소 없음)

    def flush():
        for q, t in zip(batch, ask(proc, model, [b[-1] for b in batch], a.max_new_tokens)):
            add(*q[:5], t)
        batch.clear()

    for q in queries(recs, K, a.data_root, a.shard, a.nshard, done):
        if q[-1] is None:
            add(*q[:5], "", note="no_marker")
            continue
        if batch and (len(batch) == a.bsz or sum(map(patches, [b[-1] for b in batch])) + patches(q[-1]) > a.max_patches):
            flush()
        batch.append(q)
        n += 1
        if n % (a.bsz * 20) == 0:
            save()
            print(f"  {n} 질의, {(time.time() - t0) / n:.3f} s/점", flush=True)
        if a.limit and n >= a.limit:
            break
    if batch:
        flush()
    save()
    z = pd.DataFrame(rows)
    print(f"[{a.dataset} shard {a.shard}] 점 {len(z)}: 화살표 불가 {(z.note == 'no_marker').sum()}, 파싱 실패 {(z.note == 'parse_fail').sum()}, "
          f"'</answer>' 없음 {(~z.text.str.contains('</answer>') & (z.note == '')).sum()}, 이번 질의 {n} ({(time.time() - t0) / max(n, 1):.3f} s/점) → {out}", flush=True)
    g = z.dropna(subset=["pred"])
    for col in ("pred_raw", "pred"):  # 검증 순서: 변환 전 → 변환 후 (GT 는 둘 다 z)
        r = np.where(g[col] > 0, np.maximum(g[col] / g.gt_z, g.gt_z / g[col]), np.inf)  # 0·음수 답은 δ1 오답
        print(f"  {col:8s} vs gt_z: δ1 {(r < 1.25).mean():.3f}  AbsRel {(abs(g[col] - g.gt_z) / g.gt_z).mean():.3f}  RMSE {np.sqrt(((g[col] - g.gt_z) ** 2).mean()):.3f}  (pooled, n={len(g)})", flush=True)


if __name__ == "__main__":
    main()
