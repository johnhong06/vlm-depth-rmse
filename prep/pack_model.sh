#!/usr/bin/env bash
# 모델 가중치 폴더 → H200 전달용 팩 vdr_<name>.tar.part_NN (2 GB 조각) + vdr_<name>.sha256. tar 안 경로 = models/<dir>/...
# 사용: bash prep/pack_model.sh depthvlm4b <HF 스냅샷 폴더> DepthVLM-4B ~/data/h200_staging/vdr
set -euo pipefail
name=$1 src=$2 dir=$3 out=$4
mkdir -p "$out" && rm -f "$out"/vdr_"$name".tar.part_*
tar -chf - --transform "s|^\.|models/$dir|" -C "$src" . | split -b 2000M -d -a 2 - "$out/vdr_$name.tar.part_"
(cd "$out" && sha256sum vdr_"$name".tar.part_* > vdr_"$name".sha256 && cat vdr_"$name".sha256)
