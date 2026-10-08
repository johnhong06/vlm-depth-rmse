#!/usr/bin/env bash
# H200 진입점 (gpu-request 이슈의 실행 명령 한 줄, GPU 할당량 7). Track A = DepthLM-12B vs dense 모델 4종, 공통 점에서.
#   bash run.sh env                           # 데이터 팩 없이: 모델 5종 환경·공식 저장소·가중치·로딩 + KITTI 데모 이미지 한 장 (공식 Metric3D 저장소에 포함)
#   bash run.sh smoke [데이터셋...]          # 데이터까지: DepthLM 48 점 + dense 모델마다 3 장 (데이터셋 기본: ibims1 nuscenes)
#   bash run.sh depthlm [데이터셋...]        # DepthLM-12B 를 공통 점 전부에 질의
#   bash run.sh dense <모델|all> [데이터셋...]  # 모델 = dav2 | unidepth | metric3d (depthpro 는 지난 실행 재현용으로만, all 에서 빠짐 — NOTES D-23)
#   bash run.sh all [데이터셋...]            # depthlm + dense all 을 한 작업으로
#   bash run.sh m3d_nyu                       # 검증: Metric3Dv2 ViT-L NYUv2 RMS 0.251 재현 (공식 654 장, 팩 vdr_nyu_official)
#   bash run.sh trackb [데이터셋...] [LIMIT=n] # Track B: DepthVLM-4B + dense 모델 4종을 valid GT 전체 픽셀로 (LIMIT>0 이면 데이터셋당 n 장 스모크, 결과는 smoke_b)
# 데이터셋: ibims1 nyuv2 ddad nuscenes diode_outdoor. 끝나면 지금까지의 원자료 전체로 표·δ1 검증표를 찍고 zip 으로 묶는다.
# 규칙: /app/output 은 결과 전용(용량 100 GB). 압축 해제본·환경·공식 저장소·가중치 캐시는 WORK(쓰기 가능한 /app/data → /app/scratch → /tmp)에 둔다.
set -uo pipefail
cd "$(dirname "$0")" || exit 1
REPO=$PWD
MODE=${1:-smoke}; shift || true
case $MODE in env|smoke|depthlm|all|m3d_nyu|trackb) ;; dense) DM=${1:?"dense 다음에 모델 이름"}; shift;; *) echo "!!! MODE 는 env|smoke|depthlm|dense|all|m3d_nyu|trackb"; exit 1;; esac
ARGS=(); for a in "$@"; do case $a in   # 이슈 한 줄 명령에서는 환경변수를 줄 수 없어 KEY=값 을 인자로도 받는다 (V3 run.sh 와 같은 방식)
  NPROC=*|STALL_MIN=*|BUDGET_MIN=*|PROGRESS_SEC=*|DATA_SRC=*|LIMIT=*) export "${a%%=*}=${a#*=}";; *) ARGS+=("$a");; esac; done
DATASETS=${ARGS[*]:-ibims1 nuscenes}
OUT=$([ -d /app/output ] && echo /app/output/vdr || echo "$PWD/results")
A=$OUT/track_a; [ "$MODE" = smoke ] && A=$OUT/smoke_a   # 스모크는 따로 — 본 실행이 이어받아 점이 겹치지 않게
B=$OUT/track_b; [ "${LIMIT:-0}" -gt 0 ] && B=$OUT/smoke_b   # Track B (dense) 이미지별 통계
mkdir -p "$A" "$OUT/tables"; [ "$MODE" = trackb ] && mkdir -p "$B"
w() { mkdir -p "$1" 2>/dev/null && touch "$1/.w" 2>/dev/null && rm -f "$1/.w"; }
WORK=${WORK:-$(w /app/data/vdr_work && echo /app/data/vdr_work || { w /app/scratch/vdr_work && echo /app/scratch/vdr_work || echo /tmp/vdr_work; })}
mkdir -p "$WORK"; export HF_HOME=$WORK/hf TORCH_HOME=$WORK/torch PIP_CACHE_DIR=$WORK/pip VDR_EXT=$WORK/ext PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
TAG=${MODE}_$(date +%m%d_%H%M); LOG=$OUT/run_$TAG.log; exec > >(tee -a "$LOG") 2>&1; TEE=$!
trap 'exec >&- 2>&-; wait $TEE' EXIT   # 끝날 때 tee 가 마지막 줄까지 쓰고 나가게
# 주의: H200 이미지의 bash 5.1 은 인자 없는 `wait` 가 위 tee 까지 기다려 영원히 멈춘다 (NOTES F-8). wait 에는 항상 PID 를 준다
echo "[setup] $(date '+%F %T') MODE=$MODE ${DM:-} DATASETS=$DATASETS OUT=$OUT WORK=$WORK commit=$(git rev-parse --short HEAD 2>/dev/null)"
# 작업이 강제 종료되면 콘솔 로그를 받을 수 없다 → 시간 제한·멈춤 감지로 스스로 멈추고 정상 종료해 표·로그를 남긴다 (DepthLM 은 저장된 점부터 이어서 돈다)
BUDGET_MIN=${BUDGET_MIN:-0}   # 시간 제한(분). 정상 작업도 멈출 수 있어 기본은 끔(0) — 필요할 때만 BUDGET_MIN=<분> 으로 켠다
STALL=$((${STALL_MIN:-40} * 60)); DEADLINE=$((SECONDS + BUDGET_MIN * 60))   # 멈춤 = DepthLM 로그가 STALL 초 동안 그대로 (정상이면 64 질의마다, 1 분 안팎마다 진행 줄)
left() { if [ "$BUDGET_MIN" -eq 0 ]; then echo 999999; else echo $((DEADLINE - SECONDS)); fi; }   # 남은 초
case $MODE in smoke|depthlm|all) echo "[setup] DepthLM 이 $((STALL / 60))분 동안 진행이 없으면 멈춘 것으로 보고 중단$([ "$BUDGET_MIN" -gt 0 ] && echo ", 시간 제한 ${BUDGET_MIN}분") — 중단해도 표·로그를 남기고 정상 종료";; esac

# --- 모델마다 따로 만드는 환경 (규칙 10). conda-forge python 3.12 (defaults 채널은 비대화형에서 약관 동의를 요구할 수 있다), 실패하면 uv ---
CONDA=$(command -v conda || echo /opt/conda/bin/conda)
FA=https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.7cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
mkenv() {  # 이름 → $WORK/envs/<이름>/bin/python. 설치가 끝까지 된 환경에만 .ok 를 남기고, 없으면 처음부터 다시 만든다
  local E=$WORK/envs/$1
  if [ ! -f "$E/.ok" ]; then {   # 진행 메시지는 전부 stderr — stdout 은 파이썬 경로 하나만
    echo "[env] $1 생성"; rm -rf "$E"
    "$CONDA" create -y -q -p "$E" --override-channels -c conda-forge python=3.12 >/dev/null \
      || { echo "!!! [env] conda 실패 → uv"; pip install -q uv && uv venv -q -p 3.12 "$E" && uv pip install -q -p "$E/bin/python" pip; } || return 1
    "$E/bin/pip" install -q -r "envs/$1.txt" || { echo "!!! [env] $1 패키지 설치 실패"; return 1; }
    # mmcv-lite·mmengine(metric3d)은 GUI 판 opencv-python 을 끌어와 headless 를 덮어쓴다 — libGL 없는 이미지에서 import cv2 가 깨지므로 headless 로 되돌린다
    if "$E/bin/pip" show -q opencv-python >/dev/null 2>&1; then
      "$E/bin/pip" uninstall -y -q opencv-python && "$E/bin/pip" install -q --no-deps --force-reinstall opencv-python-headless || return 1
    fi
    case $1 in depthlm|depthvlm) "$E/bin/pip" install -q "$FA" || echo "!!! [env] flash-attn 설치 실패 → sdpa (둘 다 exact attention, 공식은 flash_attention_2)";; esac
    touch "$E/.ok"
  } >&2; fi
  "$E/bin/python" -c "import torch, cv2; p = torch.cuda.get_device_properties(0); print(f'[env] $1: torch {torch.__version__} cv2 {cv2.__version__} | {p.name} {p.total_memory / 2**30:.0f} GiB')" >&2 || return 1
  echo "$E/bin/python"
}
declare -A MNAME=([dav2]=DAv2-metric-L [unidepth]=UniDepthV2-L [metric3d]=Metric3Dv2-L [depthpro]=DepthPro [depthvlm]=DepthVLM-4B)

# --- 데이터: 데이터셋마다 팩 vdr_<ds>.tar.part_* (관리자가 /app/data 아래에 둔다) → WORK/bench ---
case $MODE in env) ;; m3d_nyu) python3 h200/unpack.py vdr_nyu_official "$WORK/bench" || exit 1;;
  *) for ds in $DATASETS; do
       python3 h200/unpack.py "vdr_$ds" "$WORK/bench" && continue
       # 새 팩이 아직 없을 때: 9/23 팩(/app/data/HJ)에 iBims-1 RGB 의 바이트 동일본이 있다. DepthLM 은 RGB 만 쓰므로(GT 는 jsonl) 그것으로 먼저 돈다.
       # 벤치 원본의 SHA256 목록(bench/ibims1_rgb.sha256)과 100/100 일치할 때만 쓴다. dense 모델은 GT 맵이 필요해 새 팩을 기다린다.
       if [ "$ds" = ibims1 ] && [ "$MODE" = depthlm ] \
          && python3 h200/unpack.py depthlm_distill_h200_app_data "$WORK/old" depthlm_distill_h200/eval/ibims1/rgb depthlm_distill_h200/models/DepthLM \
          && mkdir -p "$WORK/bench/ibims1/ibims1_core_raw" && ln -sfn "$WORK/old/depthlm_distill_h200/eval/ibims1/rgb" "$WORK/bench/ibims1/ibims1_core_raw/rgb" \
          && (cd "$WORK/bench/ibims1/ibims1_core_raw/rgb" && sha256sum -c --quiet "$REPO/bench/ibims1_rgb.sha256"); then
         echo "[data] ibims1: 새 팩 대신 9/23 팩의 RGB 100 장 사용 (벤치 원본과 SHA256 100/100 일치)"; continue
       fi
       exit 1
     done;; esac
[ "$MODE" != depthlm ] && { bash prep/fetch_ext.sh || exit 1; }

depthlm_weights() {  # 이미 풀린 폴더 → 없으면 옛 팩(depthlm-distill-h200, 2026-09-23 전달, /app/data 에 있음)에서 가중치만 푼다
  local M
  M=$(find -L /app/data "$WORK" -maxdepth 5 -name model.safetensors.index.json -path "*/models/DepthLM/*" -printf "%h\n" 2>/dev/null | head -1)   # 9/23 팩 구조
  if [ -z "$M" ]; then
    python3 h200/unpack.py depthlm_distill_h200_app_data "$WORK/old" depthlm_distill_h200/models/DepthLM >&2 || return 1
    M=$WORK/old/depthlm_distill_h200/models/DepthLM
  fi
  echo "$M"
}
alive() { local p; for p in "$@"; do kill -0 "$p" 2>/dev/null && return 0; done; return 1; }   # 하나라도 살아 있으면 참
progress() {  # 데이터셋 프로세스수 → 프로세스별 마지막 진행 줄(누적 질의·s/점) + GPU 상태
  local i l
  for i in $(seq 0 $(($2 - 1))); do
    l=$(grep -E "질의, |^\[" "$A/log_depthlm_$1_$i.txt" | tail -n 1)
    echo "[진행 $(date '+%T')] $1 $i: ${l:-(첫 진행 줄 전) $(tail -c 150 "$A/log_depthlm_$1_$i.txt" | tr '\r\n' '  ')}"
  done
  nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null | sed "s/^/[진행] GPU 사용률, 메모리: /"
}
run_depthlm() {  # $1 = 점 수 제한 (0 = 전부)
  local PY M NPROC=${NPROC:-2} EVERY=${PROGRESS_SEC:-600} pids t i sig last idle
  PY=$(mkenv depthlm) || return 1
  M=$(depthlm_weights) || return 1
  [ "$1" -gt 0 ] && NPROC=1
  echo "[depthlm] $(date '+%T') 가중치 $M ($(ls "$M"/*.safetensors | wc -l) 조각), 프로세스 $NPROC, CPU $(nproc)"
  for ds in $DATASETS; do
    [ "$(left)" -le 0 ] && { echo "[시간 제한] DepthLM $ds 건너뜀 — 같은 명령을 다시 넣으면 이어서 돈다"; continue; }
    [ -n "${DLSTALL:-}" ] && { echo "[중단] DepthLM $ds 건너뜀 — 앞 데이터셋에서 멈춤이 감지됨"; continue; }
    pids=()
    for i in $(seq 0 $((NPROC - 1))); do   # 한 GPU 에 NPROC 개 (각 가중치 25 GB + 비전 eager attention 최대 약 13 GB)
      $PY eval/depthlm_sparse.py --dataset "$ds" --data_root "$WORK/bench" --model "$M" --out "$A" --shard "$i" --nshard "$NPROC" --limit "$1" > "$A/log_depthlm_${ds}_$i.txt" 2>&1 &
      pids+=($!)
    done
    t=0; last=-1; idle=0
    while alive "${pids[@]}"; do   # 추론 중에는 콘솔이 조용하므로 EVERY 초마다 진행 상황을 찍고, 시간 제한·멈춤(로그가 STALL 초 동안 그대로)이면 중단
      sleep 10; t=$((t + 10))
      sig=$(cat "$A"/log_depthlm_"${ds}"_*.txt 2>/dev/null | wc -c)
      if [ "$sig" = "$last" ]; then idle=$((idle + 10)); else idle=0; last=$sig; fi
      if [ "$(left)" -le 0 ] || [ "$idle" -ge "$STALL" ]; then
        progress "$ds" "$NPROC"
        echo "[중단] DepthLM $ds: $([ "$idle" -ge "$STALL" ] && echo "$((STALL / 60))분 동안 진행 없음" || echo "시간 제한 ${BUDGET_MIN}분") — 저장된 점까지만 표에 들어간다. 같은 명령을 다시 넣으면 이어서 돈다"
        [ "$idle" -ge "$STALL" ] && DLSTALL=1
        kill "${pids[@]}" 2>/dev/null; break
      fi
      [ $((t % EVERY)) -eq 0 ] && progress "$ds" "$NPROC"
    done
    for i in 1 2 3 4 5 6; do alive "${pids[@]}" || break; sleep 10; done   # 중단했으면 1 분 기다린 뒤 강제 종료
    alive "${pids[@]}" && { kill -9 "${pids[@]}" 2>/dev/null; sleep 5; }
    alive "${pids[@]}" && echo "!!! DepthLM 프로세스가 종료되지 않음 (GPU 호출에서 멈춘 것으로 보임) — 기다리지 않고 진행"
    alive "${pids[@]}" || wait "${pids[@]}" 2>/dev/null
    grep -hE "^\[|vs gt_z" "$A"/log_depthlm_${ds}_*.txt | tail -n $((4 * NPROC))   # 프로세스마다 시작·요약·δ1 두 줄
    grep -l "Traceback" "$A"/log_depthlm_${ds}_*.txt 2>/dev/null | while read -r f; do echo "!!! 오류: $f"; tail -5 "$f"; done
  done
}
run_dense() {  # $1 = 모델 키, $2 = 이미지 수 제한 (0 = 전부)
  local PY; PY=$(mkenv "$1") || return 1
  [ "$(left)" -le 0 ] && { echo "[시간 제한] dense $1 건너뜀 — 같은 명령을 다시 넣으면 돈다"; return 0; }
  (cd eval && timeout -k 60 "$(left)" $PY dense_sparse.py --model "${MNAME[$1]}" --datasets $DATASETS --data_root "$WORK/bench" --out "$A" --limit "$2" 2>&1 \
     | tee "$A/log_dense_$1.txt" | grep -E "^\[|Traceback|Error")
  [ "$(left)" -le 0 ] && echo "[시간 제한] dense $1 중단 — 끝난 데이터셋까지만 저장. 같은 명령을 다시 넣으면 돈다"
  return 0
}
run_full() {  # Track B: $1 = 모델 키 — valid GT 전체 픽셀의 이미지별 통계 (eval/dense_full.py). DepthVLM 가중치는 HF 에서 고정 리비전으로 받는다
  local PY; PY=$(mkenv "$1") || return 1
  [ "$(left)" -le 0 ] && { echo "[시간 제한] Track B $1 건너뜀 — 같은 명령을 다시 넣으면 돈다"; return 0; }
  (cd eval && timeout -k 60 "$(left)" $PY dense_full.py --model "${MNAME[$1]}" --datasets $DATASETS --data_root "$WORK/bench" --out "$B" --limit "${LIMIT:-0}" 2>&1 \
     | tee "$B/log_full_$1.txt" | grep -E "^\[|Traceback|Error")
  return 0
}

case $MODE in
  env)
    for m in dav2 unidepth metric3d; do PY=$(mkenv $m) && (cd eval && $PY env_check.py --model "${MNAME[$m]}" 2>&1 | grep -E "^\[|Traceback|Error"); done
    PY=$(mkenv depthlm) && M=$(depthlm_weights) && (cd eval && $PY env_check.py --model DepthLM-12B --weights "$M" 2>&1 | grep -E "^\[|^  \(|Traceback|Error");;
  smoke) run_depthlm 48; for m in dav2 unidepth metric3d; do run_dense $m 3; done;;
  depthlm) run_depthlm 0;;
  all) run_depthlm 0; for m in dav2 unidepth metric3d; do run_dense $m 0; done;;
  dense) for m in $([ "$DM" = all ] && echo dav2 unidepth metric3d || echo "$DM"); do run_dense "$m" 0; done;;
  trackb) [[ " $DATASETS " == *" ibims1 "* ]] && python3 prep/ibims1_edges.py "$WORK/bench"
    for m in depthvlm dav2 unidepth metric3d; do run_full $m; done;;
  m3d_nyu) PY=$(mkenv metric3d) && (cd eval && $PY m3d_nyu.py "$WORK/bench/nyu_official" 2>&1 | tee "$A/log_m3d_nyu.txt" | grep -E "^\[|Traceback|Error");;
esac

# --- 지금까지의 원자료 전체로 표 + δ1 검증표 (모든 모델이 있는 데이터셋만 공통 집합으로) ---
PY=$WORK/envs/depthlm/bin/python; [ -x "$PY" ] || PY=$(ls "$WORK"/envs/*/bin/python 2>/dev/null | head -1)
MODELS=$($PY -c "import glob, pandas as pd; print(' '.join(sorted({m for f in glob.glob('$A/*.parquet') if '/densestat_' not in f for m in pd.read_parquet(f, columns=['model']).model.unique()})))")
FILES=$(ls "$A"/depthlm_*.parquet "$A"/dense_*.parquet 2>/dev/null)
if [ "$MODE" = trackb ]; then   # Track B 표: 이미지별 통계만으로 (score_dense.py)
  PY=$WORK/envs/depthvlm/bin/python
  (cd eval && $PY score_dense.py "$B"/stats_*.parquet --models DepthVLM-4B DAv2-metric-L UniDepthV2-L Metric3Dv2-L --out "$OUT/tables/${MODE}_track_b" --B 1000)
  A=$B
elif [ -n "$MODELS" ]; then
  (cd eval && $PY score.py $FILES --models $MODELS --out "$OUT/tables/${MODE}_track_a" --B 1000 && $PY checks.py $FILES $(ls "$A"/densestat_*.parquet 2>/dev/null) | tee "$OUT/tables/${MODE}_checks.md")
fi
cd "$OUT" && $PY -m zipfile -c "vdr_$TAG.zip" "$(basename "$A")" tables "run_$TAG.log" && echo "[done] $(date '+%T') 결과: $OUT/vdr_$TAG.zip ($(du -h "vdr_$TAG.zip" | cut -f1)) — 관리자에게 이 zip 을 요청"
