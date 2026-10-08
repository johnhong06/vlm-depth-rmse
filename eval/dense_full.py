"""Track B — dense 예측 모델을 valid GT 전체 픽셀로 평가한다 (NOTES D-19). 각 모델은 공식 추론 코드 그대로, 자기 환경에서 돈다 (규칙 10).
  예측 맵 → GT 원본 해상도로 bilinear (규칙 2) → valid GT 전체 픽셀의 이미지별 충분통계를 집단(all / 거리 구간 / 경계·내부)마다 저장 → stats_<model>_<ds>.parquet.
  픽셀 단위 원자료 대신 이 통계로 모든 지표(pooled·이미지별 RMSE·AbsRel·δ1, log-RMSE, SILog, 유클리드 RMSE)를 다시 계산한다 (eval/score_dense.py).
  검증용: DepthVLM 공식 dense 평가 방식의 이미지별 δ1 (d1_canon — GT 를 canonical(f=1000) 크기로 최근접, 예측은 그 크기로 bilinear, pred>0 & gt>0).
사용: python eval/dense_full.py --model DepthVLM-4B --datasets ibims1 nuscenes --data_root ~/data/depthvlm_bench --out results/track_b
"""
import argparse
import importlib
import os
import sys
import time

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image

from breakdown import BIN_NAMES, BINS, BOUNDARY_SETS, CAP
from common import BENCH, ROOT, load_bench, ray
from dense_sparse import EXT, MODELS, gt_map, overlay

sys.path.insert(0, os.path.join(ROOT, "prep"))
from boundary_labels import R, contour_map  # noqa: E402

DEPTHVLM = ("JonnyYu828/DepthVLM-4B", "2b2d02fcfe0c89c8aa7d541055e5a078930077c9")


def canonical(path, k):
    """공식 큐레이션(create_data_pixel_level_*.py)과 같은 식: 원본 크기 × 1000 / fx (가로·세로 모두 fx 기준)."""
    W, H = Image.open(path).size
    return [round(W * 1000.0 / k[0]), round(H * 1000.0 / k[0])]


def depthvlm(domain):
    """DepthVLM-4B — 공식 eval/eval.py 경로: canonical 크기로 bilinear → 공식 프롬프트·채팅 틀 → process_vision_info → 프로세서 → forward 한 번의 depth_pred.
    공식은 bf16 + flash_attention_2 (없으면 sdpa, 둘 다 exact). 출력은 z, Softplus 라 > 0. 예측 맵은 원본 크기로 bilinear (규칙 2), 검증용으로 원래 맵을 .raw 에 둔다."""
    sys.path.insert(0, os.path.join(ROOT, "third_party/DepthVLM"))
    from model import Qwen3VLForConditionalGeneration
    from qwen_vl_utils import process_vision_info
    from transformers import AutoProcessor
    from utils.datasets import DEFAULT_DEPTH_PROMPT, _load_and_resize_image
    try:
        importlib.import_module("flash_attn")
        attn = "flash_attention_2"
    except ImportError:
        attn = "sdpa"
    proc = AutoProcessor.from_pretrained(DEPTHVLM[0], revision=DEPTHVLM[1])
    proc.tokenizer.padding_side = "left"
    m = Qwen3VLForConditionalGeneration.from_pretrained(DEPTHVLM[0], revision=DEPTHVLM[1], torch_dtype=torch.bfloat16,
                                                        attn_implementation=attn, device_map="cuda:0").eval()
    print(f"[DepthVLM-4B] attention {attn}", flush=True)

    def f(path, k):
        img = _load_and_resize_image(path, canonical(path, k))
        msg = [{"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
               {"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": DEFAULT_DEPTH_PROMPT}]}]
        inp = proc(text=[proc.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)], images=[process_vision_info(msg)[0]],
                   padding=True, return_tensors="pt").to("cuda")
        p = m(input_ids=inp.input_ids, attention_mask=inp.attention_mask, pixel_values=inp.get("pixel_values"),
              image_grid_thw=inp.get("image_grid_thw")).depth_pred[0].float().cpu().numpy()
        while p.ndim > 2:
            p = p[0]
        f.raw = p
        W, H = Image.open(path).size
        return cv2.resize(p, (W, H), interpolation=cv2.INTER_LINEAR)
    return f


def unidepth_k(domain):
    """공정성 대조(부록, NOTES F-16): UniDepthV2 에 GT intrinsics 를 준다 — 공식 infer(rgb, camera=3×3 K). 그 밖은 dense_sparse.unidepth 와 같다."""
    sys.path.insert(0, os.path.join(EXT, "UniDepth"))
    from unidepth.models import UniDepthV2
    m = UniDepthV2.from_pretrained("lpiccinelli/unidepth-v2-vitl14", revision="52b349b514bd8b47642f67ac78cb7b5dc5c51dd9")
    m.interpolation_mode = "bilinear"
    m = m.to("cuda").eval()

    def f(path, k):
        K = torch.tensor([[k[0], 0, k[2]], [0, k[1], k[3]], [0, 0, 1]], dtype=torch.float32)
        rgb = torch.from_numpy(np.array(Image.open(path).convert("RGB"))).permute(2, 0, 1)
        return m.infer(rgb, K)["depth"][0, 0].float().cpu().numpy()
    return f


def depthpro_f(domain):
    """공정성 대조(부록, NOTES F-16): Depth Pro 에 GT 초점 fx 를 준다 — 공식 infer(x, f_px=fx) (초점 하나만 받으므로 fx, Metric3Dv2 hub 와 같은 값. 공식 CLI 의 EXIF 값처럼 numpy 실수로 넘긴다). 그 밖은 dense_sparse.depthpro 와 같다."""
    sys.path.insert(0, os.path.join(EXT, "ml-depth-pro/src"))
    import dataclasses
    import depth_pro
    from depth_pro.depth_pro import DEFAULT_MONODEPTH_CONFIG_DICT
    from huggingface_hub import hf_hub_download
    ck = hf_hub_download("apple/DepthPro", "depth_pro.pt", revision="ccd1350a774eb2248bcdfb3be430e38f1d3087ef")
    m, tf = depth_pro.create_model_and_transforms(config=dataclasses.replace(DEFAULT_MONODEPTH_CONFIG_DICT, checkpoint_uri=ck),
                                                  device=torch.device("cuda"), precision=torch.half)
    m.eval()
    return lambda path, k: m.infer(tf(depth_pro.load_rgb(path)[0]), f_px=np.float64(k[0]))["depth"].float().cpu().numpy()


ALL = dict(MODELS, **{"DepthVLM-4B": depthvlm, "UniDepthV2-L+K": unidepth_k, "DepthPro+f": depthpro_f})


def sums(p, g, r_, ds):
    """한 집단의 충분통계. log 지표에서만 예측을 cap 범위로 자른다 (D-18)."""
    e, dl = p - g, np.log(np.clip(p, *CAP[ds])) - np.log(g)
    return dict(n=g.size, se=float((e ** 2).sum()), ar=float((np.abs(e) / g).sum()),
                d1=int(((p > 0) & (np.maximum(p / g, g / np.maximum(p, 1e-12)) < 1.25)).sum()),
                dl=float(dl.sum()), dl2=float((dl ** 2).sum()), n_clip=int(((p < CAP[ds][0]) | (p > CAP[ds][1])).sum()),
                se_euc=float(((e * r_) ** 2).sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(ALL))
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--overlays", type=int, default=5)
    ap.add_argument("--limit", type=int, default=0, help="데이터셋당 이미지 수 제한 (스모크)")
    ap.add_argument("--save_maps", default="", help="평가에 쓴 예측 맵(GT 원본 크기, float16 .npy)을 <경로>/<모델>/<데이터셋>/ 에 저장 — 오차 분석용 (NOTES F-16)")
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, "overlay"), exist_ok=True)
    predict = None
    for ds in a.datasets:
        recs, K = load_bench(ds)
        recs, dom = (recs[: a.limit] if a.limit else recs), BENCH[ds][1]
        if predict is None or a.model == "DAv2-metric-L":   # DAv2 만 도메인별 체크포인트 (실내·실외)
            predict = ALL[a.model](dom)
        rows, t0, resized = [], time.time(), 0
        with torch.no_grad():
            for i, r in enumerate(recs):
                path, k = os.path.join(a.data_root, r["image"]), K[r["image"]]
                predict.raw = None
                pred = np.asarray(predict(path, k), dtype=np.float32)
                gt = gt_map(r, a.data_root, ds)
                if pred.shape != gt.shape:
                    pred, resized = cv2.resize(pred, gt.shape[::-1], interpolation=cv2.INTER_LINEAR), resized + 1
                if a.save_maps:
                    dst = os.path.join(a.save_maps, a.model, ds, r["image"].rsplit(".", 1)[0].replace("/", "__") + ".npy")
                    os.makedirs(os.path.dirname(dst), exist_ok=True)
                    np.save(dst, pred.astype(np.float16))
                valid, fin = gt > 0, np.isfinite(pred)
                ys, xs = np.nonzero(valid & fin)
                g, p, r_ = gt[ys, xs], pred[ys, xs], ray(xs, ys, k)
                cm = contour_map(ds, r, a.data_root, gt) if ds in BOUNDARY_SETS else None
                groups = {"all": np.ones(g.size, bool)}
                b = np.digitize(g, BINS[dom][1:-1])
                groups.update({f"dist:{n}": b == j for j, n in enumerate(BIN_NAMES)})
                if cm is not None:
                    bnd = cv2.distanceTransform((~cm).astype(np.uint8), cv2.DIST_L2, 5)[ys, xs] <= R
                    groups.update({"region:boundary": bnd, "region:interior": ~bnd})
                # 검증: 공식 DepthVLM dense 방식 (canonical 크기, 이미지별 δ1)
                cw, ch = canonical(path, k)
                gc = cv2.resize(gt, (cw, ch), interpolation=cv2.INTER_NEAREST)
                src = predict.raw if getattr(predict, "raw", None) is not None else pred
                pc = cv2.resize(src, (cw, ch), interpolation=cv2.INTER_LINEAR)
                vc = (gc > 0) & (pc > 0)
                d1c = float((np.maximum(pc[vc] / gc[vc], gc[vc] / pc[vc]) < 1.25).mean()) if vc.any() else np.nan
                for name, sel in groups.items():
                    if sel.any():
                        rows.append(dict(dataset=ds, image_id=r["image"], model=a.model, group=name, **sums(p[sel], g[sel], r_[sel], ds),
                                         **({"n_nonfinite": int((valid & ~fin).sum()), "d1_canon": d1c} if name == "all" else {})))
                if i < a.overlays:
                    overlay(path, pred, gt, os.path.join(a.out, "overlay", f"B_{a.model}_{ds}_{i}.png"))
        s = pd.DataFrame(rows)
        s.to_parquet(os.path.join(a.out, f"stats_{a.model}_{ds}.parquet"), index=False)
        t = s[s.group == "all"]
        print(f"[{a.model}_{ds}] 이미지 {len(recs)} ({(time.time() - t0) / max(len(recs), 1):.2f} s/장, GT 크기로 리사이즈 {resized}, 유한하지 않은 예측 {t.n_nonfinite.sum()}) | "
              f"valid GT {t.n.sum()} 픽셀: RMSE {np.sqrt(t.se.sum() / t.n.sum()):.3f} AbsRel {t.ar.sum() / t.n.sum():.3f} δ1 {t.d1.sum() / t.n.sum():.3f} (pooled) | "
              f"공식 방식 δ1 {t.d1_canon.mean():.3f} (canonical, 이미지별)", flush=True)


if __name__ == "__main__":
    main()
