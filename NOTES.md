# vlm-depth-rmse — 진행 기록

기록 규칙: 기능 하나·실험 하나마다 체크리스트와 실행 로그를 갱신한다. 결정은 "결정 기록"(D-번호), 확인이 필요한 발견은 F-번호.

## 체크리스트

순서: Track A 준비 → 검증 → 본 실험 → (그다음) Track B (사용자 지시, D-11).

### Track A — 준비 (2026-10-01)
- [x] 데이터 5종 원본 재생성·GT 점 단위 재현: iBims-1·NYUv2·DDAD·nuScenes (벤치 jsonl 10,000 / 10,000, 최대 차 0.00005 m), DIODE Outdoor 공통 점 직접 샘플 (D-14)
- [x] 팩 → 드라이브 `h200_vdr`: ibims1·nyuv2·ddad·nuscenes·diode_outdoor (+ 검증용 nyu_official, Track B 용 depthvlm4b; sunrgbd·eth3d 는 재설계로 미사용), 업로드마다 SHA256 대조
- [x] 모델 4종 공식 설정 조사(에이전트 4개, 코드 줄 단위) → D-12 학습 겹침, D-13 추론 설정
- [x] 실행기: `eval/depthlm_sparse.py`(새 스키마) + `eval/dense_sparse.py`(어댑터 4종, 공통 점·전체 GT 통계·겹침 그림), 모델별 환경 `envs/*.txt`, `prep/fetch_ext.sh`
- [x] 로컬 동작 확인 (iBims-1·nuScenes 5 장씩): DAv2·UniDepthV2(공식 데모 ARel 7.45 % 재현)·Metric3Dv2·Depth Pro, DepthLM(CPU 오프로드 3 점·배치 2)
- [x] 채점 `eval/score.py`(z 주·유클리드 부록, 실내/실외, 각주) 합성 데이터로 수기 계산과 일치, 검증 `eval/checks.py`, `eval/m3d_nyu.py`
- [x] `run.sh` 모드: smoke / depthlm / dense <모델|all> / m3d_nyu
- [x] 푸시 전 최종 점검 (정적·동적·팩 대조 + 독립 검토 에이전트) — 아래 실행 로그 2026-10-01 20:10
- [x] 드라이브 `h200_trackA`: /app/data/HJ 에 새로 넣을 것만 (Track A 팩 6 개 + README_ADMIN, 6.03 GiB), 조각 8 개 SHA256 일치. DepthLM 가중치는 /app/data/HJ 의 9/23 팩을 쓴다
- [ ] (사용자) 커밋·푸시 → H200 `bash run.sh env` (데이터 불필요) / 관리자에게 `h200_trackA` 전달
- [ ] (사용자 확인) Metric3Dv2 '도메인 정보' — 공식 ViT 경로에는 도메인 설정이 없어 '미사용'으로 둠 (D-13)

### Track A — 검증 (본 실험 전에 통과)
- [ ] H200 `bash run.sh smoke`
- [ ] 파일럿 `depthlm ibims1 nuscenes` + `dense all ibims1 nuscenes`
- [ ] DepthLM δ1: 변환 전 → 후, DepthLM 표 1·DepthVLM 표 1 범위 비교 (`checks.py`)
- [ ] baseline δ1: DepthVLM 표 2 sparse (nuScenes·iBims-1) — iBims-1 로컬 사전 점검 완료(아래 로그): DAv2 0.887·UniDepthV2 0.941·Metric3Dv2 0.726 = 표 2 와 같음,
      Depth Pro 0.829 (표 2 0.880, UniDepthV2 논문 dense 0.823). nuScenes 는 H200 파일럿에서
- [x] Metric3Dv2 NYUv2 RMS 0.251 (`m3d_nyu.py`, 654 장, 로컬): **벤치마크 방식 + rawDepths GT → RMSE 0.253·AbsRel 0.063·δ1 0.975 (논문 0.251·0.063·0.975)** — 통과.
      depths(보정) GT 로는 0.345·0.073·0.958 → 논문은 raw 깊이로 평가한 것으로 보인다. Track A 와 같은 hub 경로는 0.261·0.066·0.974.
- [ ] 샘플링 대표성: 공통 점 RMSE vs 전체 valid GT RMSE (`densestat_*`)
- [ ] z 변환 방향: 중심 z ≈ d, 가장자리 z < d
- [ ] 정렬: `overlay/` 그림 검토

### Track A — 본 실험
- [ ] 5 개 데이터셋 × 5 모델 → `tables/track_a.md` (주표 + 유클리드 부록)

### Track B (Track A 다음)
- [ ] 설계 재확인 후 DepthVLM-4B (가중치 드라이브에 있음) 등

### 이전 단계 기록 (재설계 전, 참고)
- 2026-10-01 오전: DepthVLM 비교·6 개 세트(SUN RGB-D·ETH3D 포함) 설계로 준비 → D-11 에서 Track A 재설계. F-1(ETH3D 정렬)은 Track A 에 해당 없음.

## 결정 기록

### D-1 (2026-10-01) Track A 점 = DepthVLM-Bench 공개 점 (사용자)
"코드에서 공개한 100개 샘플링 지점"의 해석이 둘이었다: DepthVLM-Bench 점(데이터셋당 1만 점; iBims-1 은 100장×100점)과
DepthLM 저장소 `examples/ibims1`(같은 100장×100점이지만 다른 위치, 겹침 5/10,000, 유클리드 GT). 사용자가 **DepthVLM-Bench 점**을 골랐다.
DepthLM 공개 점은 iBims-1 에만 있고(다른 데이터셋은 시드 없는 무작위), Track A 규칙과도 맞지 않는다.

### D-2 (2026-10-01) DepthLM 이 답하지 못한 점은 모든 모델에서 제외 (사용자)
공식 `dataset_inference` 는 초점 750 이미지의 테두리 5 px 안 점을 건너뛴다. 파싱 실패도 같은 취급. 대안(고정값 0.05 m 로 오답 처리)은
RMSE 를 그 고정값이 좌우해서 버렸다. 제외 수와, dense 모델의 '전체 점 vs 공통 점' 차이를 같이 보고한다.

### D-3 (2026-10-01) 데이터셋 = 로컬 보유분 6개 (사용자) — **D-11 로 대체** (Track A 는 iBims-1·NYUv2·DDAD·nuScenes·DIODE Outdoor)
Outdoor nuScenes·DDAD / Out+In ETH3D / Indoor iBims-1·SUN RGB-D·NYUv2. DepthVLM 표 2(pure vision)의 5개 중 Waymo 만 빠진다.
Waymo·Argoverse2·ScanNet++ 는 약관·신청·대용량 다운로드가 필요해 제외.

### D-4 (2026-10-01) Dense 주 해상도: 잠정 논문 방식(f=1000), 추가 검토 (사용자: "논문을 따라가는게 좋지 않을까? 추가로 고려해봐야 함") — **Track B 용**. Track A 는 규칙 2 (GT 원본 해상도)
DepthVLM 공식 dense 평가: GT 를 f=1000 크기로 **최근접** 리사이즈(마스크·cap 은 리사이즈 전 적용), 예측이 다른 크기면 bilinear 로 GT 크기에 맞춤,
δ1 은 이미지별 평균. 대안 = 원본 GT 해상도(GT 무변형, 예측을 원본 크기로). 결정 전까지 **두 해상도 모두 다시 계산할 수 있게 예측 맵을 저장**한다.

### D-5 (2026-10-01) Track A 의 GT = 공개 jsonl 의 값 (원본 해상도)
DepthVLM 공식 sparse 평가는 jsonl 값을 그대로 쓰지 않는다: f=1000 으로 최근접 리사이즈한 GT 맵에서 반올림 좌표의 값을 다시 읽고, 0 이면 jsonl 값으로 대체.
본 표는 모든 모델에 같은 jsonl GT 를 쓰고, 논문 수치 재현 확인에만 공식 방식을 쓴다.

### D-6 (2026-10-01) DepthLM 답의 깊이 정의 — 변환값과 무변환값을 둘 다 저장 (D-11 이후: 한 행의 `pred`(z)·`pred_raw`(원답) 열로)
DepthLM 공식 큐레이션: Argoverse2·ETH3D·NYU·SUN RGB-D·ScanNet++·Matterport3D 는 유클리드 거리, **nuScenes·DDAD 는 z-depth** (curate_nuscenes_*.py 의
`points_cam[2]`, curate_ddad.py 의 dgp 깊이맵). 질문 문장은 "how far ... from the camera". 규칙 1 의 변환을 주 행(`DepthLM-12B`)으로 하고,
같은 답을 변환 없이 z 로 쓴 행(`DepthLM-12B-noconv`)도 남겨 DepthVLM 표 1 이 어느 쪽인지 파일럿에서 확인한다 (사용자 검증 항목 1).

### D-7 (2026-10-01) intrinsics 는 원본 보정 파일에서
공개 jsonl 에는 `original_fx` 만 있다(문서와 달리 fy·cx·cy 없음). iBims-1 = `calib/<name>.txt`, nuScenes = 공식 추출 스크립트의 `intrinsics/*.json`
(calibrated_sensor). `prep/verify_bench.py` 가 fx 가 jsonl 과 0.01 안에서 같은지 확인한다.

### D-8 (2026-10-01) DepthLM-12B 실행 설정
- 공식 그대로: 초점 750(`eval.py` 12B 값), undistort → 정규화 → 좌표식(Step 3) → 5 px 화살표(Step 5), 공식 질문, greedy, `math_verify.parse(text)[0]`.
- attention: 공식은 text `flash_attention_2` + vision `eager`. H200 은 flash-attn 2.8.3 휠(torch 2.7, cxx11abi TRUE, cp312), 로컬 Blackwell 은 sdpa (둘 다 exact).
- `max_new_tokens` 128 (공식 기본 4096). 답은 30 토큰 안팎이고 잘림은 '</answer> 없음' 개수로 드러난다.
- 배치: 패치 합 ≤ 10,000 & ≤ 16 장. Pixtral 비전 인코더가 배치 이미지를 한 시퀀스로 잇고 eager 로 N×N 을 만들어 메모리가 패치 합의 제곱으로 는다
  (로컬에서 2장 배치 OOM). 프로세서는 긴 변 2048 까지 축소하지 않으므로 패치 수 = ⌈W/16⌉·⌈H/16⌉ (iBims-1 2,214, nuScenes ≈2,040, ETH3D ≈4,900).
- H200: 한 GPU 에 프로세스 2개, 이미지 단위로 나눈다. 프로세스당 가중치 25 GB + 비전 eager attention 최대 약 13 GB(16 head × 10k² × 8 B) + KV 캐시라 3개는 141 GB 에 빠듯하다. `NPROC` 로 바꿀 수 있다.

### D-9 (2026-10-01) 로컬 환경 분리
공용 `~/venv/main` 은 transformers 5.16 이라 DepthLM 공식(4.51.1)과 충돌 → `~/venv/depthlm` (uv, Python 3.12, torch 2.7.1+cu128 — sm_90·sm_120 모두 지원).
원본 준비용 `~/venv/prep` (nuscenes-devkit). 공용 환경의 torch 는 건드리지 않았다. H200 은 규칙 7 대로 conda.

### D-10 (2026-10-01) H200 데이터 전달
데이터셋마다 팩 `vdr_<ds>.tar.part_NN`(2 GB 조각) + `vdr_<ds>.sha256` → 드라이브 `h200_vdr` (옛 `h200_share` 와 분리해 관리자가 새 파일만 받게).
팩에는 벤치 레코드가 읽는 RGB·깊이·마스크만. DepthLM 가중치는 9/23 에 전달한 `h200_share` 팩(이미 /app/data)에서 가중치만 푼다.
`/app/output` 은 결과 전용(옛 프로젝트 100 GB 초과 사고) — 해제본·환경은 WORK(/app/data 쓰기 가능 → /app/scratch → /tmp).

### D-11 (2026-10-01) Track A 재설계 — Track A 를 끝낸 뒤 Track B (사용자 지시)
- 목표: DepthLM 과 pure vision 모델을 **zero-shot** 으로 RMSE·AbsRel·δ1 비교. 원래 RMSE 비교는 한 데이터셋에 fine-tune 한 뒤 하지만 현실적으로 불가하므로,
  비교 모델 대부분이 보지 않은 실내·실외 데이터셋에서 잰다. 범위는 측정까지.
- 모델: DepthLM-12B / DAv2-metric (L, 실내 Hypersim 20 m·실외 VKITTI 80 m) / UniDepthV2 (L) / Metric3Dv2 (ViT-L) / Depth Pro — 공식 추론 설정,
  GT intrinsics·도메인 정보 사용 여부를 표 열로. **DepthVLM-4B 는 Track B 로** (가중치는 이미 드라이브에 있음).
- 데이터셋 5개: 실내 iBims-1·NYUv2, 실외 DDAD·nuScenes·DIODE Outdoor. **SUN RGB-D·ETH3D 제외** → F-1(ETH3D 정렬)은 Track A 에 해당 없음.
- 규칙 추가: 평가 좌표계는 GT 원본 해상도 하나 (dense 예측은 그 크기로 리사이즈 후 같은 픽셀), z 공간 RMSE 가 주 결과·유클리드 RMSE 는 부록,
  DAv2 모델 상한 < 데이터셋 cap 이면 각주, 모델마다 실행 환경 분리, 1k 장 미만 데이터셋은 오버샘플링 없이 이미지당 점 수를 늘림.
- 원자료 열: dataset, image_id, u, v, fx, fy, cx, cy, gt_z, pred, pred_raw, model (DepthLM: pred = z 변환값, pred_raw = 유클리드 원답;
  dense 모델: GT 해상도로 리사이즈한 예측 맵에서 읽은 값, pred = pred_raw).
- 검증(본 실험 전): DepthLM δ1 (변환 전 → 후, DepthLM 표 1·DepthVLM 표 1 범위 비교), baseline δ1 (DepthVLM 표 2 sparse, nuScenes·iBims-1),
  Metric3Dv2 ViT-L NYUv2 RMS 0.251 별도 재현(공식 654 장·표준 마스크), sparse vs dense RMSE, z 변환 방향 확인, dense 예측–GT 정렬 시각화.

### D-12 (2026-10-01) 학습 데이터 겹침 — 각 논문·저장소의 학습 목록으로 확인 (조사 에이전트 4개, 출처는 아래)
| 모델 | iBims-1 | NYUv2 | DDAD | nuScenes | DIODE | 근거 |
|---|---|---|---|---|---|---|
| DepthLM-12B | – | – | – | **학습 (다른 장면)** | – | DepthLM §3: Argoverse2·Waymo·nuScenes·ScanNet++·Taskonomy·HM3D·Matterport3D, 평가는 학습과 겹치지 않는 장면 |
| DAv2-metric-L | – | – | – | – | – | 교사: 합성 5종, 학생: 의사라벨 실사 62M(BDD100K 등), metric: Hypersim / VKITTI2 (arXiv:2406.09414 표 7, §7.3) |
| UniDepthV2-L | – | – | – | – | – | 학습 23종 목록에 없음 (arXiv:2502.20110 v2 §4). iBims-1·DDAD·nuScenes 는 논문 zero-shot 세트 |
| Metric3Dv2-L | – | – | **학습** | – | – | 학습 18종(표 5)에 DDAD. nuScenes·DIODE·iBims-1·NYU 는 테스트 세트 (arXiv:2404.15506) |
| Depth Pro | – | – | – | – | – | 학습 21종(표 15)에 없음. 단 공개 가중치는 "재학습한 참조 구현"이고 그 학습 데이터는 따로 적혀 있지 않음 |
사용자 설계표(DDAD ← Metric3Dv2, nuScenes ← DepthLM)와 같다. 표에는 각주로 표시 (`eval/score.py` TRAINED).

### D-13 (2026-10-01) 모델별 공식 추론 설정 (조사 에이전트가 코드 줄 단위로 확인, 로컬 GPU 로 5 장씩 동작 확인)
| 모델 | 공식 경로 | 입력 처리 | 출력 | GT K | 도메인 | 고정 버전 |
|---|---|---|---|---|---|---|
| DepthLM-12B | eval.py Pixtral + dataset_inference | undistort → f=750 → 5 px 화살표, greedy | 유클리드 → z 변환 | 사용 | 미사용 | DepthLM_Official @3e76f58, transformers 4.51.1 |
| DAv2-metric-L | metric_depth/run.py → `infer_image(BGR, 518)` | 짧은 변 518·14 배수(bicubic), ImageNet 정규화, 출력 bilinear(align_corners=True)로 원본 크기 | z (sigmoid × max_depth) | 미사용 | **사용**: 실내 Hypersim(20 m) / 실외 VKITTI(80 m) | 저장소 @a561b84, HF Hypersim @7972080·VKITTI @070e97e |
| UniDepthV2-L | README·scripts/demo.py → `infer(uint8 RGB)`, camera 없음 | resolution_level 미설정(0.2–0.6 MP, 14 배수), fp16 autocast, bilinear 로 원본 크기 | z (`depth` 키; `radius` 가 유클리드) | 미사용 | 미사용 | @8d8cfe4, HF @52b349b. 공식 데모 ARel 7.45 % 로컬 재현 |
| Metric3Dv2-L | hubconf `metric3d_vit_large` + 같은 파일 데모 전후처리 | 616×1064 비율 유지(INTER_LINEAR)·평균색 가운데 패딩, canonical f=1000 → × fx·s/1000, bilinear, clamp 0–300 | z | 사용 | **미사용** (아래) | @eb5b6fa, HF JUGGHM/Metric3D vit_large_800k |
| Depth Pro | cli/run.py: GPU·fp16, `load_rgb` → `infer` | 1536×1536 정사각 리사이즈, 역깊이 bilinear 로 원본 크기, clamp 1e-4–1e4 | z (평면 검증으로 확인 — 코드·논문에 명시 없음) | 미사용 (`f_px=None`, EXIF 초점도 안 씀) | 미사용 | @9e65e4d, HF apple/DepthPro @ccd1350 (= 공식 CDN 파일 SHA-256) |
- **설계표와 다른 점 (보고 대상)**: 사용자 표는 Metric3Dv2 를 "도메인 정보 사용 / 실내·실외 crop size 고정" 으로 적었으나, **ViT 모델의 공식 경로에는 도메인 구분이 없다** —
  입력 616×1064·canonical focal 1000·깊이 범위 하나뿐이고, 논문도 "모든 zero-shot 평가에 같은 체크포인트". 도메인별 crop size(NYU 480×1216, KITTI 512×1088 등)는 v1 ConvNeXt 설정이다.
  공식 경로대로 돌리고 표에는 "미사용"으로 적는다 — 사용자가 다른 처리를 원하면 바꾼다.
- Metric3D 공식 경로들 사이의 작은 차이: hub 데모(fx, 평균색 패딩, clamp 0–300) / 논문 벤치마크 코드((fx+fy)/2, 검은 패딩, NYU 는 테두리 6 px 검게, clamp 없음).
  Track A 는 hub 데모를 따른다. (fx+fy)/2 와 fx 의 차이는 DIODE(fx 886.81, fy 927.06)에서 깊이 2.3 %, 나머지는 0.1 % 이하.
- Depth Pro 는 README 예시(CPU·fp32)와 CLI(GPU·fp16)가 다르다 — GPU 실행 경로인 CLI 를 따른다. 체크포인트 자체가 fp16.
- 모델 출력 상한: DAv2 실내 20 m(iBims-1 cap 25 m → 각주), 실외 80 m(DDAD cap 120 m → 각주). Metric3Dv2 는 canonical 200 m (예: nuScenes 정면 카메라 168 m),
  Depth Pro 는 10,000 m(하늘). 로컬 5 장에서 Metric3Dv2 가 nuScenes 일부 점을 상한 168.4 m·107.6 m 로 예측 — 공식 동작이고 RMSE 꼬리에 그대로 들어간다.

### D-14 (2026-10-01) DIODE Outdoor 공통 점과 cap
- val outdoor 446 장 전부(오버샘플링 없음), 공식 `sample_and_compute` 로 총 10,000 점(이미지당 22–23), seed 42, 이미지 정렬 순서 고정 (`prep/sample_diode.py`).
- **cap [0.05, 80] m**: 하한은 벤치의 다른 실외 세트와 같게(센서 최소 거리 0.6 m 라 효과 없음), 상한은 ZoeDepth 의 DIODE Outdoor 평가 범위
  (`max_depth_eval` 80, isl-org/ZoeDepth config.py)·nuScenes cap·DAv2 실외 모델 상한과 같게. 이 cap 으로 빠지는 유효 픽셀 0.24 %.
- intrinsics = 공식 devkit `intrinsics.txt` [886.81, 927.06, 512, 384] (fx ≠ fy). 깊이 = z-depth (F-5). 마스크 = `*_depth_mask.npy` == 1.
- 4 개 공개 벤치 파일도 이미지 중복 0·이미지 안 좌표 중복 0 을 확인 (iBims-1 100×100, NYUv2 654×15–16, DDAD·nuScenes 1,000×10).
- NYUv2 벤치 654 장 = 공식 NYU split(`splits.mat` testNdxs) 과 같은 집합 (SUN RGB-D 판 561×427). Metric3Dv2 RMS 재현은 표준(labeled.mat 640×480)으로 따로.

## 확인이 필요한 발견

- **F-1 ETH3D 정렬 — 확인됨 (2026-10-01)**: 벤치 RGB 는 보정본(`dslr_images_undistorted`, 약 6204×4135, PINHOLE)인데
  GT 깊이 맵(6048×4032)은 **왜곡 원본 `dslr_images` 좌표**다. 검증: COLMAP 보정 파일의 3D 점(12장, 약 5,000 점)마다 카메라 자세로 z 를 계산하고
  세 가설의 위치에서 GT 를 읽어 |GT/z − 1| 비교 —
  ① 왜곡 원본 좌표: 중앙 0.0011, 1 % 이내 93 % / ② 보정본 좌표를 깊이 격자에 그대로(= DepthVLM sparse 평가와 벤치 점의 암묵 가정): 중앙 0.028, 25 % /
  ③ 보정본을 깊이 격자 크기로 늘림(= DepthVLM dense 평가의 GT 리사이즈 가정): 중앙 0.0031, 83 %. (덤: ETH3D GT 가 z-depth 임도 0.1 % 안에서 확인)
  즉 벤치 점 (u, v) 는 왜곡 격자 위치인데 모델은 보정본 이미지의 (u, v) 를 본다 — 같은 화소 번호가 다른 장면 점을 가리킨다(이미지 가장자리에서 원본 해상도 기준 최대 ~150 px).
  모든 모델에 똑같이 적용되지만 ETH3D 의 GT 자체가 질의 위치와 어긋나 RMSE·AbsRel·δ1 에 잡음이 들어간다. 선택지: (a) 벤치 그대로(논문과 비교 가능),
  (b) 점을 ETH3D 카메라 모델로 보정본 좌표로 옮겨 질의·예측 위치만 고침(GT 값 그대로), (c) 둘 다 보고. → **사용자 결정 대기**.
- **F-2 nuScenes GT 투영**: 공식 추출은 LiDAR→카메라를 calibrated_sensor 만으로 변환한다(ego pose 보정 없음, 파일 타임스탬프는 sample 시각).
  차가 움직이는 장면의 측·후방 카메라에서 GT 위치가 약간 어긋날 수 있다. 벤치 정의로 그대로 따른다.
- **F-4 VLM 별 실제 입력 초점 (사용자 요청 '모델에 맞게 focal 검토')**:
  DepthLM-12B 는 공식 eval.py 의 12B 값 750 으로 정규화한 뒤 Pixtral 프로세서(긴 변 상한 2048 → 우리 세트는 축소 없음)가 16 배수로 **올림** 리사이즈한다.
  예: iBims-1 857×643 → 864×656 이라 실효 초점이 가로 756·세로 765 (+0.8 %·+2 %, 비등방). 공식 동작이므로 그대로 둔다.
  DepthVLM-4B 는 jsonl `canonical_size`(f=1000)로 리사이즈한 뒤 Qwen3-VL 프로세서(상한 16.7 M 화소 → 축소 없음)가 32 배수로 맞춘다.
  공식 eval.py 는 그 앞에서 `qwen_vl_utils.process_vision_info` 를 거치므로(기본 patch 14 → 28 배수일 수 있음) 실제 입력 크기는 DepthVLM 단계에서 확인한다.
  또 공식 sparse 평가는 예측 맵을 canonical 크기로 되돌리지 않고 canonical 좌표로 바로 읽는다 — 맵 크기가 다르면 위치가 조금 어긋난다(확인 예정).
- **F-5 DIODE 깊이 정의 = z-depth (확인됨)**: 논문은 "점들의 depth 의 robust mean" 이라고만 써서 판단 불가 → 실험: 바닥 평면이 지배적인 val outdoor
  42 장에서 아래쪽 영역을 두 가정(z / 유클리드)으로 3D 복원해 RANSAC 평면 인라이어(0.5 %)를 비교 — **42 장 모두 z 가정이 더 평평** (인라이어 차 중앙 +0.23).
- **F-3** nuScenes `v1.0-test_meta.tgz`: 버킷의 `md5.checksum` 과 MD5 가 다르다(크기 70,803,751 B 는 일치, gzip 무결성 통과). 체크섬 목록이 옛것으로 보인다.

## 실행 로그

- 2026-10-01 13:16 iBims-1 `ibims1_core_raw.zip` (55 MB, sha512 일치) → `~/data/ibims1`. 벤치 GT 10,000 점 재현.
- 2026-10-01 13:17–13:32 nuScenes test 카메라·라이다 블롭 스트리밍(필요 6,489 파일만, `tar --occurrence` 로 조기 종료) → 1.5 GB.
  공식 추출(927 sample × 6 카메라 = 5,562 프레임) → 벤치 GT 10,000 점 재현.
- 2026-10-01 로컬 스모크 (CPU 오프로드 — GPU 는 UDPNet 학습이 17.9 GB 사용 중): iBims-1 lectureroom_06 3 점, 파싱 3/3,
  답 6.68 / 7.38 / 7.28 m (GT z 9.17 / 10.45 / 10.76 m) — 파이프라인 동작 확인용, 수치 판단 아님.
- 2026-10-01 13:47 팩 → 드라이브 `h200_vdr`: vdr_ibims1 (ca6e4f4a…), vdr_nuscenes (9c9e59d3…), README_ADMIN.txt. 드라이브 SHA256 일치.
- 2026-10-01 14:00 DepthVLM-4B 가중치 (HF rev 2b2d02f, 9.1 GB) 다운로드 → `prep/pack_model.sh` 로 vdr_depthvlm4b 팩(tar 안 models/DepthVLM-4B) → 드라이브 업로드 시작.
- 2026-10-01 14:15–14:30 SUN RGB-D·NYUv2·ETH3D 원본 준비, GT 재현 3/3, 팩. ETH3D 정렬 검증(F-1).
- 2026-10-01 14:25 DDAD: dgp 설치(CPU torch, protobuf 6 컴파일) → `prep/ddad_extract.py` 실행 (약 180 프레임/분).
- 2026-10-01 14:40 로컬 배치 경로 확인 (nuScenes, bsz 2 = left padding, CPU 오프로드 49 s/점): 파싱 2/2, 잘림 0.
  구석 점 (1433, 856) 은 광선 계수 1.146 — 답 4.27 m, 변환 3.72 m, GT z 4.50 m (무변환이 더 가까움. 2 점이라 판단 근거 아님, D-6 은 파일럿 1만 점으로).
- 2026-10-01 run.sh 기본 NPROC 3 → 2 (메모리 계산, D-8), PYTORCH_CUDA_ALLOC_CONF=expandable_segments.
- 2026-10-01 14:48 vdr_depthvlm4b 업로드 완료, 드라이브 SHA256 일치 (14:45 드라이브 API 분당 한도 403 → rclone 2차 시도에서 성공. rclone 공용 client_id 한도라 자주 날 수 있음).
- 2026-10-01 14:55 DDAD 추출 완료: 벤치 sample 909/909, 카메라 프레임 5,454.
- 2026-10-01 15:20 업로드 대기 스크립트 2개가 멈춰 있던 것을 발견: `pgrep -f` 대기 패턴이 스크립트를 띄운 셸의 명령줄(heredoc)에도 들어 있어 조건이 계속 참.
  두 스크립트를 끄고, 대기 없이 순서대로 도는 `~/data/h200_staging/vdr/finish_uploads.sh` 로 교체 (DDAD 검증 → 팩 → sunrgbd·nyuv2·eth3d·ddad 업로드 + SHA256 대조, `--tpslimit 8 --retries 5`).
- 2026-10-01 15:40–16:55 Track A 재설계 반영: DIODE Outdoor 공통 점(446 장, 10,000 점), 모델 4종 조사·어댑터·환경, 새 스키마, 문서.
- 2026-10-01 16:20 Metric3Dv2 NYU 사전 점검 (공식 test 앞 200 장): 벤치마크 방식 + rawDepths GT → 이미지별 RMSE 0.285·AbsRel 0.071·δ1 0.973,
  depths(보정) GT → 0.427·0.081·0.956, hub 방식은 조금씩 더 나쁨 (논문 0.251·0.063·0.975). 654 장 전체는 이어서.
- 2026-10-01 16:48 `run.sh dense dav2 ibims1` 로컬 종단 실행 (팩 해제 → 공식 저장소 → 환경 → 추론 → 표·δ1 검증표 → zip) 정상.
  **DAv2-metric-L iBims-1 (100 장, 10,000 점): δ1 0.887 = DepthVLM 표 2 의 0.887**, RMSE 0.586 [0.459, 0.707], AbsRel 0.126. 공통 점 RMSE 0.586 vs 전체 valid GT 0.579.
- 2026-10-01 16:55 **UniDepthV2-L iBims-1: δ1 0.941 = DepthVLM 표 2 의 0.941**, RMSE 0.497, AbsRel 0.090 (공통 점) / 전체 valid GT RMSE 0.487.
- 2026-10-01 17:15 Metric3Dv2 NYUv2 654 장 (로컬): 벤치마크 방식 rawDepths 0.253 / 0.063 / 0.975, depths 0.345 / 0.073 / 0.958, hub 방식 rawDepths 0.261 / 0.066 / 0.974, depths 0.373 / 0.076 / 0.956 (RMSE / AbsRel / δ1, 이미지별 평균).
- 2026-10-01 17:19 iBims-1 100 장·10,000 점 사전 점검 (로컬, 공통 점 / 전체 valid GT):
  Metric3Dv2-L δ1 **0.726 = 표 2 0.726**, RMSE 0.776 / 0.774, AbsRel 0.195. Depth Pro δ1 0.829 (표 2 0.880 과 0.051 차; UniDepthV2 논문의 dense 값 0.823 과는 근접), RMSE 0.865 / 0.868.
  Depth Pro 차이는 GT 초점을 주지 않은 설정(f_px=None, 사용자 설계) 때문일 가능성 — DepthVLM 이 어떤 초점으로 돌렸는지는 공개되지 않음.
- 2026-10-01 19:00–19:45 푸시 전 최종 점검 (로컬, 작은 표본):
  정적 검사 — pyflakes 경고 0, shellcheck 경고 2 개 수정(cd 실패 시 중단, fetch_ext 의 rm -rf 빈 변수 가드; SC2046 은 의도된 파일 목록 분리).
  팩 대조 — 5 개 데이터셋 팩에 코드가 읽는 파일(이미지·깊이·마스크) 빠짐 0·여분 0, 조각 수 = 체크섬 수, 모든 이미지에 intrinsics. nyu_official 654 장 × 3 파일.
  intrinsics — 주점이 이미지 중앙 근처, fy/fx ≈ 1 (DIODE 1.045 공식값), 광선 계수 중앙 1.06–1.07·최대 1.54(nuScenes 후방), 이미지 밖 좌표 0.
  동적 검사 — env_check: dense 4 종 KITTI 데모 δ1 0.961–0.997, DepthLM 8 점 파싱 8/8. DepthLM 5 개 세트 × 2 점, dense 4 종 × 5 개 세트 × 2 장,
  score(5 모델 공통 집합·제외 수), checks(① δ1 ② sparse vs dense ③ z 변환 구간) 모두 끝까지 동작.
  발견·수정한 버그: env_check 가 DepthLM 질의 8 개를 한 배치로 보내 Pixtral 비전 eager attention 이 OOM (H200 에서도 30 GB 넘는 일시 메모리) → 2 장씩.
  로컬 DDAD 첫 시도 OOM 은 GPU 를 학습과 공유해서 생긴 것(배치 1 로 통과) — H200(141 GB, 프로세스 2 개)에서는 해당 없음.
- 2026-10-01 20:10 독립 검토 에이전트 보고 반영 (코드 수정 후 로컬 재시험 통과):
  · [막힘] metric3d 환경: mmcv-lite·mmengine 이 GUI 판 opencv-python 을 끌어와 pip 가 headless 를 덮어씀 → libGL 없는 H200 이미지에서 import cv2 실패 예상.
    pip 로 재현 확인(둘 다 설치됨) → mkenv 가 opencv-python 이 있으면 지우고 headless 를 --no-deps 재설치, 환경 점검에 import cv2 추가. 수정 후 cv2·mmengine import 정상.
  · prep/fetch_ext.sh: 실패를 숨기던 문제 + 컨테이너에 git 이 없을 가능성 → git 없이 커밋 고정 압축본(codeload.github.com)을 파이썬으로 받아 풀고 실패하면 종료코드 1.
    git 사본과 소스 동일(차이는 로컬 __pycache__·데모 출력뿐), 재실행 건너뜀, 잘못된 커밋에서 실패 확인.
  · DepthLM 이어 돌리기: 프로세스 수를 바꾸면 같은 점을 다시 질의 → done 을 모든 조각 파일에서 모음 (단위 시험: 겹침 0·중복 0).
  · δ1 이 음수 예측을 적중으로 셈 → score·checks·dense·depthlm 요약 모두 pred > 0 가드. score 의 경계 상황(채점할 세트 0 개, 공통 점 0 개) 메시지로 종료.
  · mkenv 가 반쯤 만든 환경을 재사용 → 완료 표시(.ok)가 있을 때만 재사용. flash-attn 은 실제 import 로 판단. 가중치는 9/23 팩 구조(*/models/DepthLM) 우선.
  · 로그 마지막 줄 유실 → EXIT trap 에서 tee 를 기다림. 도메인 평균 행에 구성 데이터셋 각주. dense 리사이즈 발생 수를 찍음. checks ② 에 이미지 균등 dense 값.
  · 문서: 프로토콜의 z 변환 검증 기준(원답/GT 가 광선 계수와 함께 커지고 변환값/GT 는 평평), DepthLM 배치·max_new_tokens, DIODE 비등방(fy′ ≈ 784), 예상 제외 점 수.
  · 검토가 확인한 것(버그 아님): DepthLM 전처리가 공식 dataset_inference 와 632 점에서 화소 단위 동일(화살표 불가 25 점도 같음), 좌표 규약·z/유클리드 수식,
    score 의 pooled·per-image·bootstrap(이미지 단위, 모델 간 짝지음, 도메인 층화), 팩 내용, unpack.py, dense 어댑터 4 종, bash 출력 경로.
