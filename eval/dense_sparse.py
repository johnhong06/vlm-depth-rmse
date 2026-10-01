"""Track A — dense 모델을 공통 점에서 평가한다. 각 모델은 공식 추론 코드 그대로, 자기 환경에서 돈다 (규칙 10).
  예측 맵 → GT 원본 해상도로 bilinear 리사이즈(규칙 2; 공식 코드가 이미 원본 크기로 돌려주면 그대로) → 공통 점 (u, v) 값 → parquet (pred = pred_raw).
  검증용으로 같은 예측에서 ① 이미지 전체 valid GT(벤치와 같은 마스크·cap)의 충분통계 → densestat_*.parquet (sparse vs dense RMSE),
  ② 앞 N 장의 RGB | 예측 | 예측 위 GT 겹침 그림 → overlay/ (정렬 확인).
사용: python eval/dense_sparse.py --model DAv2-metric-L --datasets ibims1 nuscenes --data_root ~/data/depthvlm_bench --out results/track_a
"""
import argparse
import os
import sys
import time

import cv2
import numpy as np
import pandas as pd
import torch

from common import BENCH, COLS, ROOT, load_bench

sys.path.insert(0, os.path.join(ROOT, "third_party/DepthVLM/data_process/sample_test_points"))
from sample_points import load_depth_raw, match_dataset_config  # noqa: E402

EXT = os.environ.get("VDR_EXT", os.path.join(ROOT, "ext"))  # 공식 저장소 클론 위치 (prep/fetch_ext.sh)


def dav2(domain):
    """Depth Anything V2 metric ViT-L — metric_depth/run.py 와 같은 생성·로드, infer_image(BGR, 518). 실내 Hypersim(20 m) / 실외 VKITTI(80 m)."""
    sys.path.insert(0, os.path.join(EXT, "Depth-Anything-V2/metric_depth"))
    from depth_anything_v2.dpt import DepthAnythingV2
    from huggingface_hub import hf_hub_download
    name, max_depth, rev = (("Hypersim", 20, "79720800638389a78b2defc92caa885104f69974") if domain == "Indoor"
                            else ("VKITTI", 80, "070e97e4b80e317ec1d03c19927304f4091a180b"))  # max_depth 는 체크포인트에 없다 — README 값
    ck = hf_hub_download(f"depth-anything/Depth-Anything-V2-Metric-{name}-Large", f"depth_anything_v2_metric_{name.lower()}_vitl.pth", revision=rev)
    m = DepthAnythingV2(encoder="vitl", features=256, out_channels=[256, 512, 1024, 1024], max_depth=max_depth)
    m.load_state_dict(torch.load(ck, map_location="cpu"))
    m = m.to("cuda").eval()
    return lambda path, k: m.infer_image(cv2.imread(path), 518)


def unidepth(domain):
    """UniDepthV2 ViT-L — README·scripts/demo.py 경로: from_pretrained → infer(uint8 RGB CHW), camera 없음(GT intrinsics 미사용),
    resolution_level 미설정(데모와 같음 → 화소 예산 0.2–0.6 MP), interpolation_mode bilinear. 출력 depth = z (원본 해상도)."""
    sys.path.insert(0, os.path.join(EXT, "UniDepth"))
    from PIL import Image
    from unidepth.models import UniDepthV2
    m = UniDepthV2.from_pretrained("lpiccinelli/unidepth-v2-vitl14", revision="52b349b514bd8b47642f67ac78cb7b5dc5c51dd9")
    m.interpolation_mode = "bilinear"
    m = m.to("cuda").eval()
    return lambda path, k: m.infer(torch.from_numpy(np.array(Image.open(path).convert("RGB"))).permute(2, 0, 1))["depth"][0, 0].float().cpu().numpy()


def metric3d(domain):
    """Metric3Dv2 ViT-L — hubconf.py 의 metric3d_vit_large(pretrain=True) 와 같은 파일 __main__ 데모의 전처리·후처리 그대로.
    GT intrinsics 사용(fx 로 canonical→metric), 도메인 구분 없음(ViT 는 입력 616×1064·canonical focal 1000 하나뿐, NOTES D-13):
    RGB → 616×1064 안에 비율 유지 리사이즈(INTER_LINEAR) → 평균색 가운데 패딩 → 정규화 → 패딩 제거 → 원본 크기 bilinear → × fx·scale/1000 → clamp(0, 300)."""
    m = torch.hub.load(os.path.join(EXT, "Metric3D"), "metric3d_vit_large", pretrain=True, source="local").cuda().eval()
    mean = torch.tensor([123.675, 116.28, 103.53]).float()[:, None, None]
    std = torch.tensor([58.395, 57.12, 57.375]).float()[:, None, None]

    def f(path, k):
        rgb_origin = cv2.imread(path)[:, :, ::-1]
        input_size = (616, 1064)
        h, w = rgb_origin.shape[:2]
        scale = min(input_size[0] / h, input_size[1] / w)
        rgb = cv2.resize(rgb_origin, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)
        h, w = rgb.shape[:2]
        pad_h, pad_w = input_size[0] - h, input_size[1] - w
        ph, pw = pad_h // 2, pad_w // 2
        rgb = cv2.copyMakeBorder(rgb, ph, pad_h - ph, pw, pad_w - pw, cv2.BORDER_CONSTANT, value=[123.675, 116.28, 103.53])
        x = torch.div(torch.from_numpy(rgb.transpose((2, 0, 1))).float() - mean, std)[None].cuda()
        pred = m.inference({"input": x})[0].squeeze()
        pred = pred[ph: pred.shape[0] - (pad_h - ph), pw: pred.shape[1] - (pad_w - pw)]
        pred = torch.nn.functional.interpolate(pred[None, None], rgb_origin.shape[:2], mode="bilinear").squeeze()
        return torch.clamp(pred * (k[0] * scale) / 1000.0, 0, 300).cpu().numpy()
    return f


def depthpro(domain):
    """Depth Pro — 공식 CLI(cli/run.py) 경로: create_model_and_transforms(GPU, fp16) → load_rgb → infer(f_px=None).
    CLI 는 EXIF 초점을 넘기지만 GT 초점 미사용 조건이라 f_px=None 으로 고정한다(우리 이미지에는 EXIF 초점이 없음을 확인, NOTES D-13).
    내부 1536×1536 리사이즈, 역깊이를 원본 크기로 bilinear 후 depth = 1/clamp(역깊이) (z, 하늘은 10,000 m). 도메인 구분 없음."""
    sys.path.insert(0, os.path.join(EXT, "ml-depth-pro/src"))
    import dataclasses
    import depth_pro
    from depth_pro.depth_pro import DEFAULT_MONODEPTH_CONFIG_DICT
    from huggingface_hub import hf_hub_download
    ck = hf_hub_download("apple/DepthPro", "depth_pro.pt", revision="ccd1350a774eb2248bcdfb3be430e38f1d3087ef")  # 공식 CDN 파일과 같은 SHA-256
    m, tf = depth_pro.create_model_and_transforms(config=dataclasses.replace(DEFAULT_MONODEPTH_CONFIG_DICT, checkpoint_uri=ck),
                                                  device=torch.device("cuda"), precision=torch.half)
    m.eval()
    return lambda path, k: m.infer(tf(depth_pro.load_rgb(path)[0]), f_px=None)["depth"].float().cpu().numpy()


MODELS = {"DAv2-metric-L": dav2, "UniDepthV2-L": unidepth, "Metric3Dv2-L": metric3d, "DepthPro": depthpro}


def gt_map(r, root, ds):
    """평가 GT 맵 — 공통 점과 같은 마스크·cap (벤치 = 공식 load_depth_raw, DIODE = npy + 레코드의 cap)."""
    if r.get("depth_format") == "npy":
        d = np.load(os.path.join(root, r["depth_path"]))[..., 0].astype(np.float32)
        d[(np.load(os.path.join(root, r["mask_valid_path"])) == 0) | (d < r["min_depth"]) | (d > r["max_depth"])] = 0
        return d
    c = match_dataset_config(BENCH[ds][0])
    p = lambda key: os.path.join(root, r[key]) if r.get(key) else None
    return load_depth_raw(p("depth_path"), r["depth_scale"], c["min_depth"], c["max_depth"], p("mask_valid_path"), p("mask_transp_path"), r.get("depth_format"))


def overlay(path, pred, gt, dst):
    """RGB | 예측 | 예측 위에 valid GT 를 같은 색척도로 겹친 그림 (로그 깊이 색). 경계가 어긋나면 셋째 칸에서 색이 튄다.
    희소 GT(LiDAR, 유효 30 % 미만)는 점을 5×5 로 키워 원색으로 찍는다 — 한 픽셀 점은 축소 그림에서 안 보인다."""
    lo, hi = np.log(np.percentile(gt[gt > 0], [1, 99]))
    col = lambda d: cv2.applyColorMap(np.uint8(255 * np.clip((np.log(np.maximum(d, 1e-3)) - lo) / (hi - lo), 0, 1)), cv2.COLORMAP_TURBO)
    sparse = (gt > 0).mean() < 0.3
    g_ = cv2.dilate(gt, np.ones((5, 5), np.uint8)) if sparse else gt
    p, g, m = col(pred), col(g_), g_ > 0
    mix = p.copy()
    mix[m] = g[m] if sparse else (0.35 * p[m] + 0.65 * g[m]).astype(np.uint8)
    cv2.imwrite(dst, np.hstack([cv2.resize(cv2.imread(path), pred.shape[::-1]), p, mix]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--overlays", type=int, default=5)
    ap.add_argument("--limit", type=int, default=0, help="데이터셋당 이미지 수 제한 (스모크)")
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, "overlay"), exist_ok=True)
    for ds in a.datasets:
        recs, K = load_bench(ds)
        recs = recs[: a.limit] if a.limit else recs
        predict, rows, dense, t0, resized = MODELS[a.model](BENCH[ds][1]), [], [], time.time(), 0
        with torch.no_grad():
            for i, r in enumerate(recs):
                path, k = os.path.join(a.data_root, r["image"]), K[r["image"]]
                pred = np.asarray(predict(path, k), dtype=np.float32)
                gt = gt_map(r, a.data_root, ds)
                if pred.shape != gt.shape:  # 규칙 2: 기준 격자 = GT 원본 해상도 (공식 코드는 이미 원본 크기로 준다 — 생기면 아래에 수를 찍는다)
                    pred, resized = cv2.resize(pred, gt.shape[::-1], interpolation=cv2.INTER_LINEAR), resized + 1
                u, v = np.array(r["pixel_coords"]).T
                for uu, vv, gz, p in zip(u, v, r["depth"], pred[v, u]):
                    rows.append(dict(dataset=ds, image_id=r["image"], u=int(uu), v=int(vv), fx=k[0], fy=k[1], cx=k[2], cy=k[3],
                                     gt_z=float(gz), pred=float(p), pred_raw=float(p), model=a.model))
                g, p = gt[gt > 0], pred[gt > 0]
                dense.append(dict(dataset=ds, image_id=r["image"], model=a.model, n=g.size, se=float(((p - g) ** 2).sum()),
                                  ar=float((np.abs(p - g) / g).sum()), d1=int(((p > 0) & (np.maximum(p / g, g / p) < 1.25)).sum())))
                if i < a.overlays:
                    overlay(path, pred, gt, os.path.join(a.out, "overlay", f"{a.model}_{ds}_{i}.png"))
        tag = f"{a.model}_{ds}"
        pd.DataFrame(rows, columns=COLS).to_parquet(os.path.join(a.out, f"dense_{tag}.parquet"), index=False)
        pd.DataFrame(dense).to_parquet(os.path.join(a.out, f"densestat_{tag}.parquet"), index=False)
        x, s = pd.DataFrame(rows), pd.DataFrame(dense)
        r_ = np.where(x.pred > 0, np.maximum(x.pred / x.gt_z, x.gt_z / x.pred), np.inf)
        print(f"[{tag}] 이미지 {len(recs)} ({(time.time() - t0) / len(recs):.2f} s/장, GT 크기로 리사이즈 {resized}) | 공통 점: RMSE {np.sqrt(((x.pred - x.gt_z) ** 2).mean()):.3f} "
              f"AbsRel {(abs(x.pred - x.gt_z) / x.gt_z).mean():.3f} δ1 {(r_ < 1.25).mean():.3f} | 전체 valid GT: RMSE {np.sqrt(s.se.sum() / s.n.sum()):.3f} "
              f"AbsRel {s.ar.sum() / s.n.sum():.3f} δ1 {s.d1.sum() / s.n.sum():.3f} (pooled)", flush=True)


if __name__ == "__main__":
    main()
