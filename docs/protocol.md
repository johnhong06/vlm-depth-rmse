# 평가 프로토콜 — Track A (DepthLM 의 RMSE 재평가, zero-shot)

결과 표의 모든 수치가 어떤 설정에서 나왔는지 기록한다. 바뀌면 이 문서와 `NOTES.md` 결정 기록을 같이 고친다.
상태: **확정** = 실행에 쓰는 값, **검증 중** = 파일럿 검증을 통과해야 확정.

## 1. 깊이 정의 (확정)

- 모든 GT·예측은 **z-depth**(광축 방향 거리, m).
  - 벤치 4종의 GT 는 DepthVLM-Bench `depth_type = z_depth`.
  - DIODE 는 평면 검증으로 z 임을 확인했다(NOTES F-5).
- DepthLM 답 d 는 유클리드 거리로 보고 z = d / sqrt(1 + ((u−cx)/fx)² + ((v−cy)/fy)²) 로 바꾼다.
  - (u, v) 와 intrinsics 는 원본 해상도 값을 쓴다.
  - 원답은 `pred_raw` 열에 남긴다.
- dense 모델 4종의 출력은 모두 z 이므로 변환하지 않는다(NOTES D-13).
- 부록용 유클리드 RMSE 는 GT 와 예측 양쪽에 같은 광선 계수를 곱해 계산한다. AbsRel·δ1 은 비율이라 두 공간에서 값이 같다.

## 2. 데이터와 공통 점 (확정)

| 도메인 | 데이터셋 | 분할 | 이미지 | 점/이미지 | 점 | cap (m) | 마스크 | 공통 점 출처 |
|---|---|---|--:|--:|--:|---|---|---|
| 실내 | iBims-1 | 전체 | 100 | 100 | 10,000 | 0.005–25 | `mask_invalid` + `mask_transp` | DepthVLM-Bench |
| 실내 | NYUv2 | 공식 test(SUN RGB-D 판 561×427) | 654 | 15–16 | 10,000 | 0.005–10 | 범위 | DepthVLM-Bench |
| 실외 | DDAD | val | 1,000 | 10 | 10,000 | 0.05–120 | 범위 | DepthVLM-Bench |
| 실외 | nuScenes | test (v1.0-test) | 1,000 | 10 | 10,000 | 0.05–80 | 범위 | DepthVLM-Bench |
| 실외 | DIODE Outdoor | val outdoor | 446 | 22–23 | 10,000 | 0.05–80 | `depth_mask` | 직접 샘플 (같은 규칙) |

- 벤치 4종의 점·GT 는 HF `JonnyYu828/DepthVLM-Bench` jsonl 의 원본 해상도 값 그대로다. DepthVLM 평가 로더가 리사이즈한 GT 는 쓰지 않는다.
- 원본은 공식 경로와 공식 추출 코드로 다시 만들고, 모든 점에서 GT 가 jsonl 과 일치하는지 확인했다. 10,000 / 10,000 이고 최대 차는 반올림 수준인 0.00005 m 다.
- DIODE Outdoor 는 벤치와 같은 규칙으로 뽑았다.
  - 공식 `sample_points.py` 의 `sample_and_compute` 를 쓴다(이미지 i 마다 `RandomState(42 + i)`, valid 픽셀에서 비복원 균일 추출).
  - 전체 10,000 점을 446 장에 고르게 나누고, 이미지 순서는 정렬 순서로 고정한다.
  - cap 상한 80 m 는 ZoeDepth 의 DIODE Outdoor 평가 범위와 같다. 이 cap 으로 빠지는 유효 픽셀은 0.24 % 다.
- 오버샘플링은 없다. 5 개 세트 모두 이미지 중복 0, 이미지 안 좌표 중복 0 을 확인했다.
- intrinsics 는 원본 보정 파일에서 읽는다.
  - iBims-1: `calib/*.txt`
  - NYUv2: SUN RGB-D `intrinsics.txt`
  - DDAD·nuScenes: 공식 추출 스크립트
  - DIODE: 공식 devkit [886.81, 927.06, 512, 384]

## 3. 공통 집합 (확정)

데이터셋마다 **비교하는 모든 모델이 유한한 예측을 낸 점**만 쓴다(NOTES D-2).
DepthLM 이 화살표를 못 그리는 점(초점 750 이미지의 테두리 5 px 안)과 파싱에 실패한 점이 빠진다. 표의 `px (excl.)` 열에 그 수를 적는다.

## 4. 모델별 추론 설정 (공식 경로, NOTES D-13)

| 모델 | GT intrinsics | 도메인 정보 | 입력 처리 | 출력 → 평가 격자 | 고정 버전 |
|---|---|---|---|---|---|
| DepthLM-12B | 사용 | 미사용 | undistort → 초점 750 정규화 → 5 px 화살표(공식 Step 3·5). Pixtral 프로세서가 16 배수로 올림 | 숫자 답 → z 변환 | DepthLM_Official @3e76f58, `facebook/DepthLM`, transformers 4.51.1, text FA2 + vision eager, greedy |
| DAv2-metric (L) | 미사용 | 사용 (실내 Hypersim 20 m / 실외 VKITTI 80 m) | 짧은 변 518, 14 배수, bicubic | 원본 크기 bilinear(공식) | @a561b84, HF Hypersim @7972080·VKITTI @070e97e |
| UniDepthV2 (L) | 미사용 | 미사용 | resolution_level 미설정(0.2–0.6 MP, 14 배수), fp16 autocast | 원본 크기 bilinear(공식) | @8d8cfe4, HF @52b349b |
| Metric3Dv2 (ViT-L) | 사용 (fx) | 미사용 — ViT 는 입력 616×1064 하나 | 비율 유지 리사이즈 + 평균색 가운데 패딩, canonical f=1000 | 원본 크기 bilinear, × fx·s/1000, clamp 0–300 | @eb5b6fa, HF JUGGHM/Metric3D vit_large_800k |
| Depth Pro | 미사용 (`f_px=None`) | 미사용 | 1536×1536 정사각 리사이즈, GPU·fp16(공식 CLI) | 역깊이를 원본 크기로 bilinear, clamp 1e-4–1e4 | @9e65e4d, HF apple/DepthPro @ccd1350 |

- DepthLM 배치: 이미지 ≤ 16 장·패치 합 ≤ 10,000, 왼쪽 패딩(공식 eval.py 도 배치·왼쪽 패딩, eval.sh 는 bsz 3). bf16 에서 배치 구성에 따라 드물게 greedy 답이 달라질 수 있다.
  `max_new_tokens` 128 (공식 기본 4096) — 답은 30 토큰 안팎이고 잘림은 실행 요약의 '</answer> 없음' 수로 확인한다.
- DepthLM 의 실효 초점: 공식 `undistort_image` 는 왜곡 계수가 0 이라 fx ≠ fy 여도 K 를 바꾸지 않고, 정규화는 fx 기준이다.
  그래서 정규화 이미지는 fx′ = 750, fy′ = 750·fy/fx — DIODE 는 fy′ ≈ 784(+4.5 %), DDAD ≈ ±1 %. 공식 동작이라 그대로 둔다.
  Pixtral 프로세서가 16 배수로 올리는 것까지 합쳐 NOTES F-4.
- DepthLM 이 화살표를 못 그려 빠질 점의 기하학적 예상 수(파싱 실패 제외): iBims-1 250, NYUv2 287, DDAD 80, nuScenes 184, DIODE 244 (각 10,000 점 중).
- 평가 격자는 GT 원본 해상도다(규칙 2). 다섯 세트 모두 RGB 와 GT 크기가 같다.
- 모든 dense 모델은 공식 코드가 이미 원본 크기로 돌려준다. 크기가 다르면 bilinear 로 맞추도록 코드에 넣어 두었다.
- **모델 상한 각주 (규칙 9):** DAv2 실내 모델 20 m < iBims-1 cap 25 m, DAv2 실외 모델 80 m < DDAD cap 120 m.
- 학습 데이터 겹침(표에 각주): Metric3Dv2 ← DDAD, DepthLM ← nuScenes(다른 장면). 나머지는 없다(NOTES D-12).

## 5. 지표 (확정)

- RMSE = sqrt(mean((p − g)²)) [m], AbsRel = mean(|p − g| / g), δ1 = mean(max(p/g, g/p) < 1.25). 예측 0 은 그대로 둔다(δ1 오답).
- **주 지표 = pooled**: 데이터셋 안 공통 점을 모두 모아 계산.
- **보조 = per-image**: 이미지마다 계산해 평균. DepthVLM 공식 eval 과 Metric3D·UniDepth 평가 코드의 집계 방식이다.
- 도메인 행 = 그 도메인 데이터셋 값의 평균(실내 2 개, 실외 3 개). 실내·실외 전체 평균은 내지 않는다.
- 95 % CI: 데이터셋마다 이미지를 복원추출해 다시 계산한다(B = 2,000, seed 0, 백분위; H200 요약은 B = 1,000). 도메인 CI 는 같은 회차의 데이터셋 값을 평균한다.

## 6. 검증 (검증 중 — 통과 전에는 본 실험을 돌리지 않는다)

| 항목 | 방법 | 기준 |
|---|---|---|
| DepthLM δ1 재현 | 변환 전(pred_raw) → 변환 후(pred), `eval/checks.py` | DepthLM 표 1 (iBims-1 0.870, NYUv2 0.799, DDAD 0.670, nuScenes 0.819, 자체 8,192 점)·DepthVLM 표 1 (0.754, 0.866, 0.654, 0.736) 범위 근접 |
| baseline δ1 재현 | 같은 스크립트 | DepthVLM 표 2 sparse: nuScenes / iBims-1 = DAv2 0.168 / 0.887, UniDepthV2 0.872 / 0.941, Metric3Dv2 0.747 / 0.726, Depth Pro 0.389 / 0.880 |
| RMSE 코드 | Metric3Dv2 NYUv2 공식 654 장, 논문 벤치마크 프로토콜(`eval/m3d_nyu.py`) | 논문 표 1 zero-shot RMS 0.251, AbsRel 0.063, δ1 0.975. GT(depths/rawDepths)가 공개되지 않아 둘 다 |
| 샘플링 대표성 | 같은 예측의 공통 점 RMSE vs 전체 valid GT RMSE (`densestat_*`) | 차이 보고 |
| z 변환 방향 | 광선 계수 구간별 중앙값 (`checks.py` ③): 원답/GT 와 변환값/GT | 원답이 유클리드라면 원답/GT 는 가장자리로 갈수록 커지고 변환값/GT 는 평평하다 (z/d 자체는 정의상 1/광선 계수라 검증이 되지 않는다) |
| 정렬 | 예측 위 GT 겹침 그림 (`overlay/`) | 경계 어긋남 없음 |
