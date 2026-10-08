# 평가 프로토콜 — Track A (DepthLM의 RMSE 재평가)

Track A를 재현하거나 검토할 때 필요한 설정을 모은 문서다. 설계와 각 항목의 '이유'는 프로젝트 설계에서 그대로 옮겼고, 설정값은 코드와 벤치 파일에 실제로 적용한 값이다. 정하지 않은 값은 TBD로 둔다. 결정 경위는 [NOTES.md](../NOTES.md)의 결정 기록(D-번호)에 있으며, 설정을 바꾸면 이 문서와 NOTES.md를 같이 고친다.

이유: 설정 기록은 재현성과 리뷰 대응에 필요하다.

## 1. 모델별 추론 조건

모든 모델은 공식 추론 설정 그대로 쓴다. GT intrinsics와 도메인 정보를 쓰는지는 결과 표에 열로 적는다.

이유: 각 모델을 설계된 방식대로 평가하되, 입력 정보 차이를 표에 드러내 공정성 논란을 막는다.

| 모델 | 크기 | GT intrinsics | 도메인 정보 | 출력 |
|---|---|---|---|---|
| DepthLM-12B | 12B (3B·7B는 미공개) | 사용 | 미사용 | 숫자 답 → 데이터셋별 공식 정의에 따라 z 그대로 또는 유클리드 → z 변환 (4.1절) |
| Depth Anything V2 metric | ViT-L | 미사용 | 사용: 실내 Hypersim 모델, 실외 VKITTI 모델 | z-depth |
| UniDepthV2 | ViT-L | 미사용 | 미사용 | z-depth |
| Metric3Dv2 | ViT-L | 사용 (fx) | 미사용 | z-depth |
| Depth Pro | 공개 모델 하나 | 미사용 (`f_px=None`) | 미사용 | z-depth |

| 모델 | 입력 크기 (crop size) | focal length | 모델 출력 상한 |
|---|---|---|---|
| DepthLM-12B | focal 750으로 정규화한 이미지 전체(크기는 이미지마다 다름). Pixtral 프로세서가 16의 배수로 올린다 | 통일 focal 750 | 없음 (숫자로 답함) |
| Depth Anything V2 metric | 짧은 변 518, 비율 유지, 14의 배수 | 쓰지 않음 | 실내 20 m / 실외 80 m |
| UniDepthV2 | 0.2–0.6 MP, 14의 배수 (`resolution_level` 미설정, 공식 데모와 같음) | 모델이 추정 | 사실상 없음 (e^10 ≈ 22 km) |
| Metric3Dv2 | 616×1064 안에 비율 유지 리사이즈 + 평균색 패딩. 실내·실외 같은 값 | canonical focal 1000, GT fx로 환산 | canonical 200 m × fx·s/1000 (s = 리사이즈 배율, 이미지마다 다름. 예: NuScenes 정면 카메라 168 m). 그 뒤 clamp 300 m |
| Depth Pro | 1536×1536 정사각 리사이즈 (네트워크 고정 크기) | 모델이 추정 | 10,000 m (역깊이 clamp 1e-4) |

| 모델 | 공식 경로 | 코드 | 가중치 |
|---|---|---|---|
| DepthLM-12B | `eval.py`와 `dataset_inference` 전처리. bf16, text attention flash_attention_2(설치돼 있지 않으면 sdpa) + vision eager | `third_party/DepthLM_Official` @3e76f58 | `facebook/DepthLM` |
| Depth Anything V2 metric | `metric_depth/run.py` → `infer_image(BGR, 518)` | Depth-Anything-V2 @a561b84 | HF Metric-Hypersim-Large @7972080, Metric-VKITTI-Large @070e97e |
| UniDepthV2 | README 예시 → `infer(RGB)`, 카메라 입력 없음, fp16 autocast (`scripts/demo.py` 는 GT K 를 넣는 경로라 다르다 — 2026-10-07 정정) | UniDepth @8d8cfe4 | HF `lpiccinelli/unidepth-v2-vitl14` @52b349b |
| Metric3Dv2 | `hubconf.py`의 `metric3d_vit_large`와 같은 파일의 데모 전후처리 | Metric3D @eb5b6fa | HF `JUGGHM/Metric3D` `metric_depth_vit_large_800k.pth` |
| Depth Pro | `cli/run.py` (GPU, fp16) → `infer(f_px=None)` | ml-depth-pro @9e65e4d | HF `apple/DepthPro` @ccd1350 |

dense 모델의 공식 저장소는 `prep/fetch_ext.sh`가 위 커밋으로 받는다.

**통일 focal length (규칙 3).** DepthLM에는 12B 설정값 750으로 정규화한 이미지로 질의한다. 원본 좌표를 같은 배율로 옮겨 마커(5 px 화살표)를 그린다. 순서는 공식 코드와 같이 undistort → focal 750 정규화 → 좌표 변환 → 화살표다. 공식 정규화는 fx 하나로 배율을 정하므로, fx ≠ fy인 DIODE Outdoor는 세로 focal이 약 784가 된다. 공식 동작이라 그대로 둔다.

이유: 배율이 틀리면 질의 위치와 z 변환이 모두 어긋난다.

DepthLM 생성 설정은 공식 질문 문장, greedy decoding, 공식과 같은 `math_verify.parse` 답 파싱이다. `max_new_tokens`는 128로 줄였다(공식 기본 4096). 답이 30토큰 안팎이라서이며, 잘린 답의 수는 실행 요약에 찍는다. 배치는 이미지 16장과 패치 합 10,000 이하로 묶고 왼쪽 패딩을 쓴다. bf16이라 배치 구성에 따라 greedy 답이 드물게 달라질 수 있다.

**모델 상한 (규칙 9).** DAv2 metric의 모델 상한(실내 20 m, 실외 80 m)이 데이터셋 cap보다 작으면 각주로 적는다. 해당 데이터셋은 iBims-1(20 m < cap 25 m)과 DDAD(80 m < cap 120 m)다. Metric3Dv2와 Depth Pro의 상한도 위 표처럼 공식 코드 그대로 두고, 예측을 따로 자르지 않는다.

이유: 원거리 오차가 모델 성능이 아니라 구조적 상한 때문임을 구분하기 위함.

Metric3D의 공식 경로는 hub 데모(fx, 평균색 패딩, clamp 0–300)와 논문 벤치마크 코드((fx+fy)/2, 검은 패딩, clamp 없음) 두 가지다. Track A는 hub 데모를 따른다. fx와 (fx+fy)/2의 차이는 DIODE Outdoor에서 깊이 2.3%, 나머지 세트에서 0.1% 이하다. Depth Pro도 README 예시(CPU, fp32)와 CLI(GPU, fp16)가 다른데, GPU 실행 경로인 CLI를 따른다.

**실행 환경 (규칙 10).** 모델마다 실행 환경을 따로 둔다. 패키지 목록은 `envs/<모델>.txt`이고 DepthLM은 transformers 4.51.1을 쓴다. H200에서는 `run.sh`가 모델별 가상환경(Python 3.12)을 만든다. conda가 있으면 conda를, 없으면 uv를 쓰는데 지금 서버에는 conda가 없어 uv로 만든다. `bash run.sh env`로 데이터 없이 환경과 가중치를 점검할 수 있다.

이유: 라이브러리 버전 충돌로 인한 조용한 오작동을 막는다.

## 2. 학습 데이터 겹침

이유: 비교 모델 대부분에게 zero-shot인 데이터셋으로 구성하고, 겹치는 경우는 배제 대신 명시해 실내/실외 결론을 각각 낼 수 있게 한다.

분류 기준은 각 모델 논문이 밝힌 학습 데이터 목록이다. 평가 데이터셋이 그 목록에 있으면 '학습'으로 분류하고, 장면이 달라도 같은 데이터셋이면 '학습'으로 본다(DepthLM–NuScenes). '학습'인 칸은 결과 표에 †로 표시하고 결과에서 따로 언급한다.

| 모델 | iBims-1 | NYUv2 | DDAD | NuScenes | DIODE Outdoor | 근거 |
|---|---|---|---|---|---|---|
| DepthLM-12B | – | – | – | 학습 (다른 장면) | – | DepthLM 논문 3절의 학습 데이터: Argoverse2·Waymo·NuScenes·ScanNet++·Taskonomy·HM3D·Matterport3D. 평가에는 학습과 겹치지 않는 장면을 쓴다 |
| Depth Anything V2 metric | – | – | – | – | – | 교사는 합성 데이터 5종, 학생은 의사 라벨을 단 실사 이미지 6,200만 장(BDD100K 등), metric 모델은 Hypersim / VKITTI2로 파인튜닝 (arXiv:2406.09414 표 7, 7.3절) |
| UniDepthV2 | – | – | – | – | – | 학습 데이터 23종에 없음 (arXiv:2502.20110 v2 4절). iBims-1·DDAD·NuScenes는 논문의 zero-shot 평가 세트다 |
| Metric3Dv2 | – | – | 학습 | – | – | 학습 데이터 18종(표 5)에 DDAD가 있다. NuScenes·DIODE·iBims-1·NYUv2는 테스트 세트다 (arXiv:2404.15506) |
| Depth Pro | – | – | – | – | – | 학습 데이터 21종(표 15)에 없음. 다만 공개 가중치는 '재학습한 참조 구현'이고 그 학습 데이터는 따로 밝히지 않았다 |

## 3. 공통 픽셀 샘플링

평가는 공통 샘플 픽셀에서 한다. DepthLM은 그 픽셀에만 질의하고, dense 모델은 예측 맵에서 같은 위치의 값을 꺼낸다.

이유: DepthLM은 픽셀마다 VLM forward가 필요해 dense 평가가 사실상 불가능하다.

### 3.1 DepthVLM-Bench sparse points (iBims-1, NYUv2, DDAD, NuScenes)

- Hugging Face `JonnyYu828/DepthVLM-Bench`의 jsonl에서 원본 해상도 좌표(`pixel_coords`)와 원본 GT 값(`depth`)을 그대로 쓴다. DepthVLM 평가 로더가 리사이즈한 GT는 쓰지 않는다.
- 좌표는 벤치 공식 `sample_points.py`(seed 42)로 똑같이 다시 만들어진다. 4개 세트의 모든 이미지에서 확인했다.
- GT 값도 공식 원본 데이터와 공식 추출 코드로 다시 계산해 확인했다. 세트마다 10,000점이 모두 일치하며, 최대 차 0.00005 m는 jsonl의 소수 넷째 자리 반올림 수준이다.
- jsonl에는 fx만 있으므로 intrinsics는 원본 보정 파일에서 읽는다. iBims-1은 `calib/*.txt`, NYUv2·DDAD·NuScenes는 공식 추출 단계가 만든 `intrinsics/*.json`이다. 읽은 값은 `bench/intrinsics_<데이터셋>.json`에 있고, fx는 jsonl 값과 0.01 안에서 같다.
- iBims-1 RGB는 이전(9/23) 데이터 팩의 사본도 쓸 수 있다. 벤치 원본과 SHA256이 100장 모두 같을 때만 쓴다(`bench/ibims1_rgb.sha256`, NOTES D-15).

### 3.2 DIODE Outdoor

- 같은 규칙으로 직접 샘플링했다(`prep/sample_diode.py`). 공식 `sample_and_compute`로 valid GT 픽셀에서 비복원 균일 추출한다. seed는 42(이미지 i마다 `RandomState(42 + i)`)이고 이미지 순서는 정렬 순서로 고정한다.
- val outdoor 446장을 모두 쓰고, 총 10,000점을 이미지에 고르게 나눈다(이미지당 22–23점).
- valid mask는 `*_depth_mask.npy` == 1이다. cap은 0.05–80 m로 정했다. 하한은 벤치의 다른 실외 세트와 같다. 상한 80 m는 ZoeDepth의 DIODE Outdoor 평가 범위, NuScenes cap, DAv2 실외 모델 상한과 같은 값이다. 이 cap 때문에 빠지는 유효 픽셀은 0.24%다.
- GT는 z-depth다(바닥 평면 검증, NOTES F-5). intrinsics는 공식 devkit 값 [fx 886.81, fy 927.06, cx 512, cy 384]이다.

### 3.3 오버샘플링 금지

- 이미지가 1k장 미만인 데이터셋은 오버샘플링하지 않고 전체 이미지를 한 번씩 쓴다. 대신 이미지당 픽셀 수를 늘려 총 10,000점을 맞춘다(iBims-1 100장×100점, NYUv2 654장×15–16점, DIODE Outdoor 446장×22–23점).
- 5개 세트 모두 이미지 중복 0, 이미지 안 좌표 중복 0을 확인했다.

이유: 모든 모델이 정확히 같은 픽셀에서 비교되어야 하고, 중복 이미지가 있으면 이미지 단위 bootstrap의 신뢰구간이 왜곡된다.

### 3.4 공통 집합

- 데이터셋마다 비교하는 모든 모델이 유한한 예측을 낸 점만 채점한다(NOTES D-2). DepthLM이 마커를 그릴 수 없는 점(focal 750 이미지의 테두리 5 px 안, 공식 `dataset_inference`도 건너뛴다)과 답을 파싱하지 못한 점은 모든 모델에서 뺀다. 뺀 점의 수는 결과 표에 적는다.
- 마커를 그릴 수 없어 빠질 것으로 예상되는 점은 각 10,000점 중 iBims-1 250, NYUv2 287, DDAD 80, NuScenes 184, DIODE Outdoor 244다(파싱 실패 제외).

## 4. 평가 규칙

### 4.1 깊이 정의 (규칙 1)

모든 GT와 예측은 z-depth(광축 방향 거리, m)다. dense 모델 4개의 출력은 z-depth라 변환하지 않는다. UniDepthV2는 `depth` 출력이 z이고 `radius`가 유클리드 거리다. Depth Pro는 코드와 논문에 명시가 없어 평면 검증으로 z임을 확인했다.

DepthLM은 "카메라에서 얼마나 떨어져 있나"에 숫자로 답한다. 이 답을 어떤 거리로 볼지는 **DepthLM 공식 코드가 그 데이터셋에 둔 GT 정의**를 따른다(2026-10-02 개정, NOTES D-17).

| 데이터셋 | DepthLM 공식 GT 정의 | 답 처리 |
|---|---|---|
| NuScenes, DDAD | z (공식 데이터 정리 코드 `curate_nuscenes_*.py`의 `points_cam[2]`, `curate_ddad.py`의 깊이맵) | 그대로 z로 쓴다 |
| iBims-1 | z (공식 예제 `examples/ibims1/ibims1_val.jsonl`의 라벨이 GT 깊이 지도 값과 10,000점 모두 같다. 정리 코드는 공개되지 않았다) | 그대로 z로 쓴다 |
| NYUv2 | 유클리드 거리 (공식 정리 코드 `curate_NYU.py`가 z에서 유클리드 거리를 계산해 라벨로 쓴다) | 아래 식으로 z로 바꾼다 |
| DIODE Outdoor | 공식 정의 없음 → 질문의 기본 뜻(카메라에서의 거리) = 유클리드 | 아래 식으로 z로 바꾼다 |

```
z = d / sqrt(1 + ((u − cx)/fx)² + ((v − cy)/fy)²)
```

(u, v)와 intrinsics는 원본 해상도 값을 쓴다. 답 그대로는 `pred_raw`, 변환값은 `pred`에 둘 다 남기고, 표는 `eval/score.py --depthlm`으로 고른다(`official` = 위 표, 주 결과).
부록으로 `converted`(모든 데이터셋 변환 — 원래 규칙 1, DepthVLM 논문과 같은 방식)와 `raw`(모든 데이터셋 그대로)를 함께 낸다.

이유: 정의가 섞이면 가장자리·원거리에서 체계적 편향이 생겨 RMSE가 부풀려진다.

개정 이유: 원래 규칙은 DepthLM 답을 모두 유클리드로 보고 변환했다. 그런데 DepthLM은 데이터셋마다 다른 정의로 학습·평가했다(주행 데이터셋과 iBims-1은 z 라벨, NYU 등은 유클리드 라벨).
dense 모델은 자기가 학습한 정의(z)로 평가받으므로, DepthLM도 자기가 배운 정의로 평가해야 공정하다. DepthLM 논문도 이 방식으로 평가했다(우리 답 그대로의 δ1이 DepthLM 논문 표 1과 맞음: NuScenes 0.823 vs 0.819, DDAD 0.680 vs 0.670).
데이터셋마다 무엇을 쓸지는 우리 결과(GT와의 비교)가 아니라 공식 코드의 정의로 미리 정했다. GT를 보고 유리한 쪽을 고르면 정답을 보고 점수를 고르는 셈이 되기 때문이다.
파일럿 실측(검증 ③, NOTES F-10)은 이 결정의 계기일 뿐 데이터셋별 선택 기준이 아니다.

정정(2026-10-04): 처음 개정 때는 iBims-1을 유클리드로 분류해 변환했다. 공식 예제 라벨을 GT 깊이 지도와 같은 좌표에서 대조해 보니 z와 10,000점 모두 같고 유클리드와는 0.8%만 같아서, iBims-1을 z로 고쳤다. 기준은 그대로 공식 정의이고, 분류만 바로잡았다(NOTES D-17).

### 4.2 기준 좌표계 (규칙 2)

평가 좌표계는 GT 원본 해상도 하나로 고정한다. dense 예측은 이 크기로 리사이즈(bilinear)한 뒤 같은 픽셀의 값을 꺼낸다. 다섯 세트 모두 RGB와 GT 크기가 같고, 네 모델의 공식 코드는 예측을 이미 원본 크기로 돌려준다. 크기가 다르면 bilinear로 맞추고 그 횟수를 로그에 찍는다.

이유: 모델마다 출력 해상도가 달라 기준 격자가 하나여야 같은 픽셀을 비교할 수 있다.

### 4.3 valid mask와 cap (규칙 4)

valid mask와 데이터셋별 min/max depth cap은 모든 모델에 똑같이 적용한다. 값은 6절에 있다. cap은 GT에 적용되며(공통 픽셀은 cap 안의 valid GT에서만 뽑힌다) 예측은 자르지 않는다.

이유: RMSE는 어떤 픽셀을 포함하느냐에 크게 좌우된다.

### 4.4 지표

- RMSE = sqrt(mean((p − g)²)), 단위 m
- AbsRel = mean(|p − g| / g)
- δ1 = mean(max(p/g, g/p) < 1.25). 예측이 0 이하이면 오답으로 센다.

### 4.5 보고 단위와 집계 (규칙 5, 6)

결과는 데이터셋별, 그리고 도메인별(실내/실외)로 나눠 보고한다. 도메인 값은 그 도메인 데이터셋 값의 평균이다(실내 2개, 실외 3개). 실내와 실외를 섞은 평균은 내지 않는다.

이유: 실외 RMSE(수십 m)가 실내(수 m)를 압도해 단순 평균은 의미가 없다.

주 지표는 데이터셋 안 전체 픽셀을 모아 계산한 pooled 값이고, 보조 지표는 이미지별로 계산한 뒤 평균한 값이다. 표에 방식을 적는다.

이유: 이미지당 10픽셀로는 이미지별 RMSE가 불안정하고, 집계 방식에 따라 값이 달라진다.

### 4.6 bootstrap (규칙 7)

데이터셋마다 이미지를 복원추출해 지표를 다시 계산하고, 2.5·97.5 백분위로 95% 신뢰구간을 낸다. 결과 표(`tables/`)는 B = 1,000, seed 0으로 만들었다(`run.sh`가 `--B 1000`을 넘긴다. `score.py`·`score_dense.py`의 기본값은 2,000). 모든 모델이 같은 재표본을 쓰며, 도메인 신뢰구간은 같은 회차의 데이터셋 값을 평균해 낸다. (2026-10-07 정정: 예전 문구는 표가 B = 2,000이라고 적었다.)

이유: RMSE는 꼬리 오차에 민감해 모델 간 차이가 유의미한지 보여야 한다.

### 4.7 유클리드 RMSE 부록 (규칙 8)

z 공간 RMSE가 주 결과다. 유클리드 공간 RMSE는 GT와 예측에 같은 광선 계수 sqrt(1 + ((u − cx)/fx)² + ((v − cy)/fy)²)를 곱해 계산하고 부록 표로 낸다. AbsRel과 δ1은 비율이라 두 공간에서 값이 같다.

이유: 깊이 공간 선택이 결과를 좌우하지 않는다는 근거를 미리 확보한다.

### 4.8 결과 표

`eval/score.py`가 원자료 parquet만 읽어 만든다. RMSE↓, AbsRel↓, δ1↑와 95% CI에 GT intrinsics·도메인 정보 사용 여부 열, 학습 데이터 겹침 표시, DAv2 모델 상한 각주, 공통 집합에서 뺀 점의 수가 함께 들어간다.

### 4.9 보조 집계 (2026-10-02 추가, NOTES D-18)

주 결과(4.4–4.8)는 그대로 두고, 같은 공통 점과 같은 DepthLM 처리로 `eval/breakdown.py`가 보조 표를 만든다. 원자료만 다시 읽으므로 재실행은 없다.

이유: 데이터셋 전체를 모은 RMSE는 먼 거리 점이 지배한다. 거리·영역별로 나누고 log 지표를 같이 보면 해석이 안정되고, 다음 단계(원인 분석)의 재료가 된다 (외부 피드백).

- **거리 구간** (GT 기준, 도메인마다 고정 미터): 실내 근 0–2 / 중 2–4 / 원 4 m 이상, 실외 근 0–10 / 중 10–30 / 원 30 m 이상.
  - 이유: 데이터셋마다 같은 거리에서 비교할 수 있어야 한다(데이터셋별 3 등분은 '원거리'의 뜻이 데이터셋마다 달라진다). 가장 적은 구간도 약 1,000 점(NYUv2 원 13 %, DIODE 원 10 %)이라 CI 를 낼 수 있다.
- **경계 / 내부**: 질의 점이 가림 경계에서 3 px(원본 해상도) 이내면 경계. `prep/boundary_labels.py` → `bench/boundary_<ds>.parquet`.
  - iBims-1: 공식 경계 정답 지도(Koch et al. 2018). 지도가 있는 86 장(8,500 점)만 쓴다.
  - NYUv2: Depth Pro(arXiv:2410.02073)의 정의 — 이웃 두 valid GT 픽셀의 깊이 비 > 1.1. iBims-1 에서 이 정의는 공식 경계 점의 88 % 를 잡고 전체 일치 95 %.
  - DIODE Outdoor, NuScenes, DDAD: 제외. LiDAR GT(NuScenes·DDAD)는 경계를 정할 수 없고(Mind the Edge, arXiv:2212.05315 도 같은 이유로 수작업 주석),
    DIODE 실외 GT 는 나뭇잎·유리창·스캔 줄무늬에서 이웃 깊이가 원래 크게 튀어 같은 정의가 점의 41 % 를 경계로 잡는다(물체 경계가 아니라 'GT 가 고르지 않은 영역'이 된다).
  - DepthLM 은 깊이 맵이 없어 경계 F1·DBE 같은 맵 기반 지표를 쓸 수 없다. 그래서 점을 경계/내부로 나눠 각 오차를 비교한다.
- **log 지표**: log-RMSE = sqrt(mean d²), SILog = 100 · sqrt(mean d² − (mean d)²) (KITTI 벤치마크 정의, 이미지별 계산 후 평균, 점 2 개 이상인 이미지), d = ln(예측) − ln(GT).
  - log 지표에서만 예측을 데이터셋 cap 범위로 자르고(BTS·AdaBins·ZoeDepth 계열 평가 관행) 잘린 점 수를 표에 적는다. 주 지표는 자르지 않는다.
  - 해석: log-RMSE 는 비율 오차라 먼 점의 지배가 줄어든다. SILog 는 이미지마다 전체 배율을 맞춘 뒤 남는 오차라 '전체적으로 짧게/길게 답함'은 지워지고 장면 구조 오차가 남는다.
    NuScenes·DDAD 는 이미지당 10 점이라 이미지별 SILog 가 흔들린다.
- CI: 이미지 단위 bootstrap (B = 1,000, seed 0).

## 5. 검증

검증은 본 실험 전에 통과해야 한다. 파일럿(iBims-1 실내 + NuScenes 실외)에서 파이프라인과 아래 항목을 먼저 통과시킨다. 그다음 5개 데이터셋 전체(DIODE Outdoor 샘플링 포함)로 넓히고 결과 표를 만든다.

이유: 숫자가 틀려도 에러가 나지 않는 작업이라, 논문 수치 재현으로 파이프라인을 먼저 확인한다.

이유(순서): 작은 규모에서 문제를 먼저 잡아야 전체 실험의 재실행 비용을 줄일 수 있다.

| 항목 | 방법 | 참고 수치와 출처 | 판단 기준 |
|---|---|---|---|
| DepthLM-12B δ1 재현 | z 변환 전 값(`pred_raw`)으로 먼저 확인한 뒤 변환값(`pred`)으로 확인한다. pooled와 이미지별 둘 다 (`eval/checks.py` ①) | DepthLM 논문 표 1 "Ours - Pixtral (12b)" 행: iBims-1 0.870, NYUv2 0.799, DDAD 0.670, NuScenes 0.819 (데이터셋마다 무작위 8,192 샘플). DepthVLM 논문 표 1 "DepthLM-12B" 행: iBims-1 0.754, NYUv2 0.866, DDAD 0.654, NuScenes 0.736 | 두 논문은 샘플링이 달라 값이 다르다. 그 범위에 근접하는지로 본다. DIODE Outdoor는 참고 수치가 없다. NYUv2는 DepthLM 공식 정리 코드(`curate_NYU.py`)가 SUN RGB-D 깊이 png를 10000으로 나눠(공식 툴박스 기준은 8000) GT가 실제의 0.8배다. 우리 답을 GT×0.8과 비교하면 0.865로 DepthVLM 표 1(0.866)과 맞는다(NOTES F-11) |
| Baseline δ1 재현 | 같은 스크립트 | DepthVLM 논문 표 2 (VLM 평가와 같은 샘플 픽셀, sparse), NuScenes / iBims-1: Depth Anything V2 0.168 / 0.887, UniDepthV2 0.872 / 0.941, Metric3Dv2 0.747 / 0.726, Depth Pro 0.389 / 0.880 | 근접 여부 |
| RMSE 코드 검증 | Metric3Dv2 ViT-L을 NYUv2 공식 테스트 split 654장(`labeled.mat`, 640×480)과 표준 평가 마스크(Eigen crop, GT 0.1–10 m)로 따로 돌린다. 논문 벤치마크 코드의 전처리를 따르고 이미지별 평균을 낸다 (`eval/m3d_nyu.py`) | Metric3D v2 논문 표 1, NYUv2 "Ours ViT-L CSTM_label ZS" 행: RMS 0.251, AbsRel 0.063, δ1 0.975 | 근접 여부. 논문이 GT로 `depths`(채운 깊이)와 `rawDepths` 중 무엇을 썼는지 밝히지 않아 둘 다 계산한다 |
| 샘플링 대표성 | 같은 dense 예측으로 공통 픽셀(sparse) RMSE와 이미지 전체 valid GT(dense) RMSE를 비교한다. dense 값은 pooled와 이미지 균등 평균 둘 다 (`densestat_*`, `checks.py` ②) | – | 차이를 보고한다 |
| z 변환 확인 | DepthLM 점을 광선 계수 구간으로 나눠 중앙값을 비교한다 (`checks.py` ③) | – | 변환 후 이미지 중심은 d ≈ z이고 가장자리로 갈수록 z < d인지 본다. z/d는 정의상 1/광선 계수이므로, 원답이 실제로 유클리드 거리인지는 원답/GT가 가장자리로 갈수록 커지고 변환값/GT는 평평한지로 판단한다 |
| 정렬 확인 | dense 예측 맵 위에 GT를 겹친 그림 (`overlay/`) | – | 경계가 어긋나지 않는지 |

- 논문 수치는 표 번호와 행 이름까지 원문과 대조했다(DepthLM arXiv:2509.25413 v2, DepthVLM arXiv:2605.15876 v3, Metric3D v2 arXiv:2404.15506 v4).
- DepthLM 논문 표 2에도 pure vision 모델 수치가 있지만 UniDepthV2·Depth Pro 논문에서 옮겨 온 값이다. baseline 재현 기준으로는 DepthVLM 표 2만 쓴다.
- DepthLM 공식 학습 데이터 정리 코드에서 NuScenes·DDAD 라벨은 z-depth이고 나머지 데이터셋은 유클리드 거리다(NOTES D-6). iBims-1 공식 예제 라벨도 z다(4.1절 정정). 이 두 세트에서 DepthLM 원답이 어느 정의인지는 첫째와 다섯째 항목으로 파일럿에서 확인한다.

> 2026-10-08: Depth Pro 는 아래 설정대로 돌렸지만 결과 표(`tables/`, README)에서는 뺐다 (NOTES D-23). 원자료에는 남아 있다.
> 2026-10-08: DDAD·NuScenes 도 결과 표에서 뺐다 — 비교 모델 일부의 학습 데이터 (NOTES D-24). 결과 표 = iBims-1·NYUv2·DIODE Outdoor, 실외 평균 = DIODE.

## 6. 데이터셋별 설정값

| 도메인 | 데이터셋 | 분할 | min cap (m) | max cap (m) | 추가 valid mask | 이미지 수 | 이미지당 픽셀 | 총 픽셀 | seed |
|---|---|---|--:|--:|---|--:|--:|--:|--:|
| 실내 | iBims-1 | 전체 (core) | 0.005 | 25 | `mask_invalid`, `mask_transp` | 100 | 100 | 10,000 | 42 |
| 실내 | NYUv2 | 공식 test (SUN RGB-D 판, 561×427) | 0.005 | 10 | – | 654 | 15–16 | 10,000 | 42 |

NYUv2 정답 (2026-10-08 개정, NOTES D-22·F-18): 점 위치와 입력 이미지는 위 벤치 그대로 두고, 정답만 `nyu_depth_v2_labeled.mat` rawDepths(Kinect 실측 픽셀)로 바꾼다. 0.005 m ≤ 정답 < 10 m (정확히 10.0 m는 센서 포화값이라 뺀다), SUN RGB-D 자르기 위치는 이미지마다 원측정에 맞춘다(`eval/weak_common.nyu_offset`, 정렬 오차 > 2 %인 NYU0273·NYU1176 제외 → 652장). 원측정이 없는 점·픽셀은 빠진다(Track A 공통 점 8,483). 경계는 원측정 정답으로 같은 정의(이웃 깊이 비 > 1.1). 입력은 `eval/nyu_raw_inputs.py`가 만든다(Track B는 로컬 예측 맵 — H200 값과 반올림까지 같다). 벤치 정답(SUN RGB-D `depth_bfx` ÷ 8,000)은 8.19 m 너머가 더 작은 값으로 채워져 있고 센서가 못 잰 곳도 채운 값이라 쓰지 않는다.
| 실외 | DDAD | val | 0.05 | 120 | – | 1,000 | 10 | 10,000 | 42 |
| 실외 | NuScenes | test (v1.0-test) | 0.05 | 80 | – | 1,000 | 10 | 10,000 | 42 |
| 실외 | DIODE Outdoor | val outdoor | 0.05 | 80 | `depth_mask` | 446 | 22–23 | 10,000 | 42 |

- 모든 세트에서 GT가 0이거나 cap 밖인 픽셀은 invalid다. '추가 valid mask'는 그 밖에 쓰는 마스크다.
- 벤치 4종의 cap은 벤치 `sample_points.py`의 `DATASET_CONFIGS` 값이고, DIODE Outdoor의 cap은 3.2절에서 정했다.
- seed 42는 벤치 `sample_points.py`의 기본값이며 이미지 i마다 `RandomState(42 + i)`를 쓴다.

## 7. 원자료 (parquet) 스키마

픽셀 하나가 한 행이다. 모든 지표는 이 파일에서 다시 계산할 수 있어야 하며, 결과 표와 검증 표도 `eval/score.py`와 `eval/checks.py`가 이 파일만 읽어서 만든다.

이유: 원자료가 있으면 다음 단계의 오차 분석을 재추론 없이 바로 할 수 있다.

| 열 | 형식 | 내용 |
|---|---|---|
| `dataset` | str | `ibims1`, `nyuv2`, `ddad`, `nuscenes`, `diode_outdoor` |
| `image_id` | str | 벤치 레코드의 이미지 경로 (예: `ibims1/ibims1_core_raw/rgb/lectureroom_06.png`) |
| `u`, `v` | int | 원본 해상도 픽셀 좌표 (u = 열, v = 행) |
| `fx`, `fy`, `cx`, `cy` | float | 원본 해상도 intrinsics |
| `gt_z` | float | GT z-depth (m), jsonl 값 그대로 |
| `pred` | float | z-depth 예측 (m). DepthLM은 변환한 값이고, 답이 없으면 NaN |
| `pred_raw` | float | DepthLM은 변환 전 원답, dense 모델은 `pred`와 같은 값 |
| `model` | str | `DepthLM-12B`, `DAv2-metric-L`, `UniDepthV2-L`, `Metric3Dv2-L`, `DepthPro` |

DepthLM 파일에는 열이 두 개 더 있다. `note`는 점이 빠진 이유(`no_marker` = 마커를 그릴 수 없는 점, `parse_fail` = 답 파싱 실패)이고 `text`는 모델이 생성한 원문이다.

| 파일 | 내용 |
|---|---|
| `depthlm_<데이터셋>.part<k>.parquet` | DepthLM-12B (GPU 프로세스별 조각) |
| `dense_<모델>_<데이터셋>.parquet` | dense 모델 4개 |
| `densestat_<모델>_<데이터셋>.parquet` | 5절 샘플링 대표성 검증용 이미지별 통계: 전체 valid GT 픽셀 수 `n`, 제곱오차 합 `se`, AbsRel 합 `ar`, δ1 적중 수 `d1` |

## 8. 작업 원칙

규칙과 다르게 처리해야 하는 상황이 생기면 임의로 진행하지 않고 먼저 보고한다. 수치가 논문과 크게 다르면 버그 가능성부터 의심하고 원인을 확인한 뒤 진행한다.

이유: 임의 판단이 쌓이면 어떤 설정 차이가 결과를 만들었는지 추적할 수 없게 된다.

- 검증(5절)을 통과하기 전에는 본 실험을 돌리지 않는다.
- 이번 단계의 범위는 측정까지다. 원인 분석과 보완 방법은 다음 단계에서 다룬다.
- 모든 지표는 원자료 parquet에서만 계산한다(7절).

보고한 뒤 확인을 기다리는 항목은 DepthLM 원답의 깊이 정의 하나다. NuScenes·DDAD는 학습 라벨이 z-depth라서, 원답이 어느 정의인지 파일럿에서 확인한다 (5절, NOTES D-6).

## 9. Track B — dense 평가 (2026-10-03 추가, NOTES D-19)

깊이 맵 전체를 내는 모델끼리 valid GT 픽셀 전체로 비교한다. DepthLM 은 dense 예측이 불가능해 빠진다.

이유: Track A 는 DepthLM 때문에 공통 점에서만 비교했다. 맵 전체를 내는 모델끼리는 이미지 전체에서 비교해야 각 모델의 실제 성능이 드러난다.

- **모델**: DepthVLM-4B + Track A 와 같은 pure vision 4 종. 학습 데이터 겹침은 2 절에 DepthVLM 을 더한 것 — DepthVLM 은 DDAD·NuScenes 의 학습 분할로 학습했다(논문 3 절·부록 A·표 9). 평가 분할과 장면은 다르지만 † 로 표시한다. iBims-1·NYUv2·DIODE 는 zero-shot.
- **DepthVLM 추론** (`eval/dense_full.py`, 공식 `eval/eval.py` 경로): 사진을 canonical 크기(가로·세로 × 1000 / fx, 공식 큐레이션 식)로 bilinear 리사이즈 → 공식 프롬프트·채팅 틀 → `process_vision_info` → 프로세서 → forward 한 번의 `depth_pred`. bf16, text attention flash_attention_2 (없으면 sdpa). 출력은 z 이고 항상 > 0. 가중치 HF `JonnyYu828/DepthVLM-4B` @2b2d02f, 코드 `third_party/DepthVLM`, 환경 `envs/depthvlm.txt` (transformers 5.2.0 = 체크포인트에 적힌 버전).
  - GT intrinsics: 사용 (입력 크기를 GT fx 로 정한다). 도메인 정보: 미사용.
- **평가 픽셀**: 데이터셋마다 Track A 와 같은 valid mask·cap 을 통과한 GT 픽셀 전부. 예측 맵은 GT 원본 해상도로 bilinear (규칙 2). 모든 모델이 같은 픽셀을 쓴다.
- **집계와 지표**: Track A 와 같다 — 주 지표 pooled(데이터셋 안 valid 픽셀 전체), 보조 이미지별 평균, 이미지 bootstrap 95 % CI(표는 B = 1,000, seed 0, 모델끼리 같은 재표본 — 4.6 절), 도메인 행 = 데이터셋 평균, 부록 = 유클리드 RMSE, 보조 집계 = 4.9 절(거리 구간, 경계/내부, log-RMSE, SILog).
  경계/내부는 픽셀마다 같은 정의로 나눈다(iBims-1 공식 경계 지도, NYUv2 이웃 GT 깊이 비 > 1.1, 3 px).
- **원자료**: 픽셀 단위 대신 이미지별 통계 `stats_<모델>_<데이터셋>.parquet` — 집단(all / 거리 구간 / 경계·내부)마다 픽셀 수, 제곱오차 합, 상대오차 합, δ1 적중 수, log 오차 합·제곱합, cap 밖 예측 수, 유클리드 제곱오차 합. 표는 `eval/score_dense.py` 가 이것만 읽어 만든다.
  이유: dense 픽셀을 모두 저장하면 모델당 수억 행이다. 위 통계로 모든 지표를 다시 계산할 수 있다(사용자 결정).
- **검증**: DepthVLM 공식 dense 방식(GT 를 canonical 크기로 최근접 리사이즈, 예측은 그 크기로 bilinear, 이미지별 δ1)으로 DepthVLM 저장소 README 의 dense 표와 비교한다.
  기준: DepthVLM NuScenes 0.838 / iBims-1 0.910, UniDepthV2 0.868 / 0.941, Metric3Dv2 0.843 / 0.724, Depth Pro 0.379 / 0.879 (논문 본문에는 없고 README 그림에만 있다. Depth Pro 는 GT 초점 사용으로 추정 — NOTES F-9).
  로컬 확인: DepthVLM iBims-1 100 장 = **0.910** (일치). NuScenes 는 로컬 GPU 메모리 부족으로 H200 에서 확인한다.
- **실행**: `bash run.sh trackb [데이터셋...]` (스모크는 `LIMIT=n`, 결과는 `smoke_b`).

## 10. VLM 약점 분석 (2026-10-07 추가, NOTES D-20·F-16)

두 VLM(DepthLM-12B, DepthVLM-4B)의 공통 약점을 찾는 분석이다. 결과와 그림은 [VLM_WEAKNESS.md](VLM_WEAKNESS.md)에 있다. 주 결과 표(Track A·B)는 바꾸지 않는다.

- **세트**: 두 VLM과 pure vision 4 종이 모두 학습하지 않은 iBims-1·NYUv2·DIODE Outdoor (2 절, 9 절). DDAD·nuScenes 는 VLM 학습 데이터라 뺀다.
- **예측 맵**: 로컬 GPU 에서 Track B 와 같은 코드로 다시 추론해 저장한다(`eval/dense_full.py --save_maps`, GT 원본 크기, float16, `~/data/vdr_maps/<모델>/<세트>/`). 로컬은 attention 이 sdpa(H200 은 flash-attention 2)지만 전체 지표가 Track B 표와 반올림 자리까지 같고, 공통 점 값은 H200 Track A 와 ±0.3 % 안에서 같다.
- **공정성 대조 (부록)**: UniDepthV2 에 GT intrinsics(공식 `infer(rgb, camera=K)`), Depth Pro 에 GT 초점 fx(공식 `infer(x, f_px)`)를 준 두 조건을 더 돌린다(`UniDepthV2-L+K`, `DepthPro+f`). 주 비교에는 넣지 않는다.
- **두 단위**: 공통 점(Track A 점, DepthLM 공식 정의 답 + dense 모델은 같은 픽셀) 6 모델 / 전체 valid 픽셀 5 모델.
- **경우**: 거리 구간(4.9 절과 같음), 경계 3 px(4.9 절과 같음, DIODE 제외), 물체 대분류(NYUv2 = GT 라벨 894 종, iBims-1·DIODE = `facebook/mask2former-swin-large-ade-semantic` 분할 150 종을 같은 대분류로 묶음, `prep/semseg_ade.py`·`eval/weak_common.py` CATS), 물체 크기(인스턴스 면적 < 1 % / 1–5 % / ≥ 5 %; NYUv2 = GT 인스턴스, 나머지 = 같은 라벨 연결 성분), 질감(이미지 안 3 등분), 화면 위치(가장자리 10 % 띠, 세로 3 등분), iBims-1 공식 평면, NYUv2 Kinect 실측/채움(rawDepths, 이미지마다 1 px 단위로 정렬).
- **지표**: AbsRel·δ1·RMSE(4.4 절), 모양 = 이미지 배율(평균 log 오차)을 뺀 log 오차 rms(%), 치우침 = 배율 뺀 평균 log 오차, 압축 기울기 a(이미지마다 ln 예측 = a·ln GT + b), 점 쌍 앞뒤 오답률(GT 깊이 비 > 1.1)·상대 깊이 오차.
- **구조 지표 (iBims-1)**: Koch et al. 2018 식 1–4 재구현 — DBE(예측 깊이 0–1 정규화, Canny σ = √2·문턱 0.1/0.2, θ = 10 px; 변형 0.15/0.3·log 정규화, GT 자기 점검 행), 평면성(이미지 배율을 GT 중앙값 비로 맞춘 뒤 SVD 평면 맞춤, ε_plan = 거리 표준편차, ε_orie = GT 점으로 맞춘 평면과의 법선 각). 공식 평가 코드의 매개변수와 다를 수 있어 모델끼리 상대 비교로만 쓴다.
- **NYUv2 원측정 GT 보조 채점 (NOTES F-17)**: 벤치 GT(SUN RGB-D `depth_bfx` ÷ 8,000)는 16 비트 한계(65,535 ÷ 8,000 = 8.19 m)로 그 이상을 적지 못하고 더 작은 값이 채워져 있다(단순 넘침으로 맞는 픽셀 4 %)(654 장 최댓값 7.995 m, 원측정 8–10 m 픽셀의 벤치 GT 중앙값 4.86 m). 같은 예측을 `nyu_depth_v2_labeled.mat` rawDepths(Kinect 실측 픽셀만, 0.005–10 m, SUN RGB-D 자르기 위치는 이미지마다 원측정에 맞춤)로 다시 채점한다(`eval/weak_nyu_raw.py`). 2026-10-08 부터 주 결과 표(Track A·B)의 NYUv2 행도 이 원측정 정답이다(6 절, D-22). (처음 재채점은 정확히 10.0 m인 포화값을 넣었다 — F-18 에서 고침.)
- **판정**: VLM ÷ pure vision 4 종 중앙값. 1 보다 크면 그 경우에서 pure vision 보다 나쁘다. 주장마다 이미지 단위 bootstrap 95 % CI(B = 1,000, seed 0).
- **코드**: `eval/weak_common.py`(로더·속성), `weak_stats.py`(통계), `weak_tables.py`(표·히트맵), `weak_figs.py`(카드·갤러리), `weak_grid.py`(토큰 격자 스펙트럼), `weak_relief.py`(벽에 붙은 물체), `weak_nyu_measured.py`(NYUv2 실측만), `weak_ci.py`(CI), `weak_examples.py`(예시 그림). 출력은 `results_vlm_weakness/`(git 제외), 대표 그림 사본은 `docs/figs/vlm_weakness/`.
