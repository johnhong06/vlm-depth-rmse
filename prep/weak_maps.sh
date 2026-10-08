#!/bin/bash
# VLM 약점 분석(NOTES D-20·F-16): zero-shot 3 세트(iBims-1·NYUv2·DIODE) 예측 맵 저장 + 이미지별 통계 — 로컬 GPU, 모델별 ~/venv/<모델> (규칙 10).
# 본 비교 4 모델 + 참고 조건 UniDepthV2+K (Depth Pro 는 실험에서 제외, NOTES D-23). 맵 = ~/data/vdr_maps/<모델>/<세트>/*.npy (float16, GT 원본 크기), 로그 = ~/data/vdr_maps/logs/
cd "$(dirname "$0")/../eval"
export HF_HUB_DISABLE_PROGRESS_BARS=1 TRANSFORMERS_VERBOSITY=error PYTHONUNBUFFERED=1
mkdir -p ~/data/vdr_maps/logs
for pair in "DepthVLM-4B depthvlm" "UniDepthV2-L unidepth" "Metric3Dv2-L metric3d" "DAv2-metric-L dav2" "UniDepthV2-L+K unidepth"; do
  set -- $pair
  echo "=== $(date +%H:%M:%S) $1 start"
  ~/venv/$2/bin/python dense_full.py --model "$1" --datasets ibims1 nyuv2 diode_outdoor --data_root ~/data/depthvlm_bench \
     --out ~/data/vdr_maps/stats --overlays 0 --save_maps ~/data/vdr_maps > ~/data/vdr_maps/logs/$1.log 2>&1
  echo "=== $(date +%H:%M:%S) $1 exit $?"; grep -a "^\[$1" ~/data/vdr_maps/logs/$1.log | cut -c1-260
done
