"""Track A 평가 세트: 데이터셋 이름 → (공통 점 jsonl, 도메인). 실내 2 + 실외 3 (NOTES D-11).
iBims-1·NYUv2·DDAD·nuScenes = HF JonnyYu828/DepthVLM-Bench 공개 jsonl, DIODE Outdoor = 같은 규칙으로 직접 샘플 (prep/sample_diode.py).
원본 해상도 intrinsics 사이드카는 bench/intrinsics_<ds>.json."""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BENCH = {
    "ibims1": ("ibims1_pixel_depth_test.jsonl", "Indoor"),
    "nyuv2": ("nyuv2_pixel_depth_test.jsonl", "Indoor"),
    "ddad": ("ddad_pixel_depth_val.jsonl", "Outdoor"),
    "nuscenes": ("nuscenes_pixel_depth_test.jsonl", "Outdoor"),
    "diode_outdoor": ("diode_outdoor_pixel_depth_val.jsonl", "Outdoor"),
}
COLS = ["dataset", "image_id", "u", "v", "fx", "fy", "cx", "cy", "gt_z", "pred", "pred_raw", "model"]  # 원자료 parquet 열 (스펙)


def load_bench(ds):
    """공통 점 jsonl 레코드 목록과 {image: [fx, fy, cx, cy]} (원본 해상도)."""
    recs = [json.loads(l) for l in open(os.path.join(ROOT, "bench", BENCH[ds][0]))]
    return recs, json.load(open(os.path.join(ROOT, "bench", f"intrinsics_{ds}.json")))


def ray(u, v, k):
    """원본 해상도 픽셀 (u, v) 의 광선 길이 계수 sqrt(1 + ((u−cx)/fx)² + ((v−cy)/fy)²) — 유클리드 거리 = z × ray."""
    return ((1 + ((u - k[2]) / k[0]) ** 2 + ((v - k[3]) / k[1]) ** 2)) ** 0.5
