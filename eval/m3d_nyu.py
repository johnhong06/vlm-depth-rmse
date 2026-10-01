"""검증 — RMSE 코드 확인: Metric3Dv2 ViT-L 의 NYUv2 zero-shot RMS 0.251 (논문 표 1, AbsRel 0.063, δ1 0.975) 을 공식 테스트 654 장으로 재현.
  protocol bench = 논문 벤치마크 코드(training/mono, test_configs_vit/nyu): RGB 테두리 6 px 검게, 616×1064 비율 유지 리사이즈 + 검은 패딩,
                   canonical→metric 에 (fx'+fy')/2, 예측을 GT 크기(480×640)로 bilinear, Eigen crop [45:471, 41:601], GT 0.1–10 m, 이미지별 평균.
  protocol hub   = Track A 와 같은 hubconf 경로(평균색 패딩, fx, clamp 0–300) — 두 경로 차이를 보이기 위해 함께 낸다.
  GT 가 labeled.mat 의 depths(보정)인지 rawDepths 인지 공개되지 않아 둘 다 계산한다. 지표는 이미지별 계산 후 평균(공식) + pooled.
사용: python eval/m3d_nyu.py ~/data/nyuv2_official [--limit N]
"""
import argparse
import glob
import os

import cv2
import numpy as np
import torch

from dense_sparse import EXT, metric3d

K = [518.8579, 519.4691, 325.58245, 253.73617]  # Metric3D 의 NYU 테스트 intrinsics (training/data_server_info 예시와 같은 값)


def bench_model():
    m = torch.hub.load(os.path.join(EXT, "Metric3D"), "metric3d_vit_large", pretrain=True, source="local").cuda().eval()
    mean = torch.tensor([123.675, 116.28, 103.53]).float()[:, None, None]
    std = torch.tensor([58.395, 57.12, 57.375]).float()[:, None, None]

    def f(path, k):
        rgb = cv2.imread(path)[:, :, ::-1]
        new = np.zeros_like(rgb)
        new[6:-6, 6:-6] = rgb[6:-6, 6:-6]  # nyu_dataset.py: 테두리 6 px 검게
        h, w = new.shape[:2]
        s = min(616 / h, 1064 / w)
        x = cv2.resize(new, (int(w * s), int(h * s)), interpolation=cv2.INTER_LINEAR)
        ph, pw = 616 - x.shape[0], 1064 - x.shape[1]
        x = cv2.copyMakeBorder(x, ph // 2, ph - ph // 2, pw // 2, pw - pw // 2, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        x = torch.div(torch.from_numpy(x.transpose((2, 0, 1))).float() - mean, std)[None].cuda()
        p = m.inference({"input": x})[0].squeeze()
        p = p[ph // 2: 616 - (ph - ph // 2), pw // 2: 1064 - (pw - pw // 2)]
        p = torch.nn.functional.interpolate(p[None, None], (h, w), mode="bilinear").squeeze()
        return (p * (k[0] * s + k[1] * s) / 2 / 1000.0).cpu().numpy()
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    files = sorted(glob.glob(os.path.join(a.root, "*_rgb.png")))[: a.limit or None]
    crop = np.zeros((480, 640), bool)
    crop[45:471, 41:601] = True
    for proto, make in (("bench", bench_model), ("hub", lambda: metric3d("Indoor"))):
        f, res = make(), {"depths": [], "rawDepths": []}
        with torch.no_grad():
            for p in files:
                pred = f(p, K)
                for gname, suf in (("depths", "_depth.npy"), ("rawDepths", "_rawdepth.npy")):
                    gt = np.load(p.replace("_rgb.png", suf))
                    m = crop & (gt >= 0.1) & (gt <= 10)
                    e, g = pred[m] - gt[m], gt[m]
                    res[gname].append((m.sum(), (e ** 2).sum(), (np.abs(e) / g).sum(), (np.maximum(pred[m] / g, g / pred[m]) < 1.25).sum()))
        for gname, r in res.items():
            r = np.array(r, float)
            n = r[:, 0]
            print(f"[{proto:5s} | GT {gname:9s}] {len(n)} 장 — 이미지별 평균: RMSE {np.mean(np.sqrt(r[:, 1] / n)):.3f}  AbsRel {np.mean(r[:, 2] / n):.3f}  "
                  f"δ1 {np.mean(r[:, 3] / n):.3f} | pooled: RMSE {np.sqrt(r[:, 1].sum() / n.sum()):.3f}  (논문 ZS: RMSE 0.251, AbsRel 0.063, δ1 0.975)", flush=True)


if __name__ == "__main__":
    main()
