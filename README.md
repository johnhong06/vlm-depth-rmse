# vlm-depth-rmse

δ1로만 보고된 VLM 기반 metric depth 모델(DepthLM-12B, DepthVLM-4B)의 성능이 RMSE·AbsRel로 재도 유지되는지, pure vision 모델 3개와 같은 조건에서 비교해 확인한다.

## 동기

DepthLM과 DepthVLM은 VLM으로 metric depth를 추정하는데, 두 논문 모두 성능을 δ1로만 보고했다. δ1은 예측이 정답의 25% 안에 드는지만 세기 때문에, 틀린 예측이 얼마나 크게 틀렸는지와 큰 오차가 얼마나 자주 나오는지는 보여 주지 못한다. 그래서 오차 크기를 그대로 반영하는 RMSE로 봐도 두 모델의 성능이 유지되는지 확인한다.

## 공통 설계

zero-shot으로, 파인튜닝 없이 평가한다. RMSE 비교는 보통 한 데이터셋에 파인튜닝한 모델끼리 하지만 우리 여건에서는 그렇게 할 수 없다. 두 VLM 논문도 zero-shot으로 평가했으니 같은 조건이어야 지표만 바꾼 비교가 된다. 그래서 실내·실외 데이터셋은 비교 모델 대부분이 학습에 쓰지 않은 것으로 골랐고, 겹치는 경우는 빼지 않고 결과 표에 표시한다.

결과 표의 평가 데이터셋은 모든 모델이 학습하지 않은 3개(실내 iBims-1·NYUv2, 실외 DIODE Outdoor)다. 실외는 DIODE 하나라 실외 값은 곧 DIODE 값이다. DDAD·NuScenes도 돌렸지만 비교 모델 일부의 학습 데이터라 표에서 뺐다. 깊이 정의는 z-depth로 통일하고, valid mask와 거리 범위(cap)는 모든 모델에 똑같이 적용한다. 모든 모델은 공식 추론 설정 그대로 돌리며, 모델마다 받는 입력 정보가 다르므로 그 차이를 표로 드러낸다.

NYUv2의 정답은 Kinect 원측정 픽셀만 쓴다(`nyu_depth_v2_labeled.mat`의 rawDepths, 0.005 m ≤ 정답 < 10 m). DepthVLM-Bench가 쓰는 SUN RGB-D 판 정답(`depth_bfx`)은 8.19 m 너머를 적지 못해 더 작은 값으로 채워져 있고, 센서가 재지 못한 곳도 채운 값이라서 쓰지 않는다. 원측정에서 정확히 10.0 m인 값은 센서 상한의 포화값이라 뺀다. 그래서 원측정이 없는 점·픽셀은 빠지고, 정렬이 어긋난 2장을 빼 652장으로 잰다.

평가는 두 트랙으로 나눈다. DepthLM은 픽셀 하나마다 VLM forward를 돌려야 해서 dense 평가가 사실상 불가능하다. 그래서 DepthLM은 정해 둔 점에서 비교하고(Track A), 깊이 맵 전체를 내는 모델끼리는 맵 전체로 비교한다(Track B).

Depth Pro도 같은 조건으로 돌렸지만 결과 표에서는 뺐다. 예측을 자르지 않아 일부 픽셀에 10 km 가까운 값을 내서 실외 RMSE가 수십 m로 커지고, DepthVLM 논문의 Depth Pro 수치는 GT 초점거리를 넣고 잰 것이라 우리가 쓴 공식 CLI 설정(초점거리 미입력)과 맞지 않으며, 공개 가중치의 학습 데이터도 밝혀져 있지 않다. 원자료에는 남아 있다.

이번 단계는 측정까지만 한다. 원인 분석과 보완은 다음 단계에서 다룬다.

## Track A — DepthLM (sparse)

공통 샘플 픽셀에서 모든 모델을 비교한다. DepthLM은 정해 둔 픽셀에만 질의하고 dense 모델은 예측 맵에서 같은 위치의 값을 꺼낸다. DepthLM의 답은 DepthLM 공식 코드가 그 데이터셋에 둔 거리 정의대로 읽는다. 주행 데이터셋과 iBims-1은 z 그대로, 나머지는 유클리드 거리로 보고 z로 바꾼다. 정의가 섞이면 가장자리와 원거리에서 RMSE가 부풀려지기 때문이다.

### 비교 모델

| 모델 | GT intrinsics | 도메인 정보 |
|---|---|---|
| DepthLM-12B | 사용 | 미사용 |
| Depth Anything V2 (metric, L) | 미사용 | 사용 (실내·실외 모델 따로) |
| UniDepthV2 (L) | 미사용 | 미사용 |
| Metric3Dv2 (ViT-L) | 사용 | 미사용 |

### 학습 데이터 겹침

| 도메인 | 데이터셋 | 학습에 사용한 모델 |
|---|---|---|
| 실내 | iBims-1 | 없음 |
| 실내 | NYUv2 | 없음 |
| 실외 | DIODE Outdoor | 없음 |

DDAD·NuScenes도 같은 조건으로 돌렸지만 결과 표에서는 뺐다. DepthLM은 NuScenes를, Metric3Dv2는 DDAD를 학습했다. 결과 표는 모든 모델이 학습하지 않은 데이터셋만으로 비교하고, 원자료에는 두 데이터셋도 남아 있다.

### 결과

굵게 = 1위, <u>밑줄</u> = 2위, 빨간 숫자 = 꼴찌. 평균 순위 = 그 도메인의 데이터셋별 RMSE·AbsRel·δ1 순위를 평균한 값(작을수록 좋음). 값은 데이터셋 안 픽셀을 모은(pooled) 점추정이며 95% CI는 결과 원본 표(`tables/`)에 있다.

#### 한눈에 보기 (도메인 평균)

| 모델 | 실내 RMSE↓ | 실내 AbsRel↓ | 실내 δ1↑ | 실내 평균 순위↓ | 실외(DIODE Outdoor) RMSE↓ | 실외(DIODE Outdoor) AbsRel↓ | 실외(DIODE Outdoor) δ1↑ | 실외(DIODE Outdoor) 평균 순위↓ |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| DepthLM-12B | $\color{red}{0.94}$ | $\color{red}{0.156}$ | $\color{red}{0.746}$ | $\color{red}{3.7}$ | 10.44 | <u>0.518</u> | $\color{red}{0.255}$ | 3.0 |
| Depth Anything V2 | <u>0.53</u> | <u>0.123</u> | <u>0.890</u> | <u>2.3</u> | <u>8.19</u> | 0.742 | 0.336 | <u>2.7</u> |
| UniDepthV2 | **0.47** | **0.106** | **0.922** | **1.7** | $\color{red}{21.82}$ | $\color{red}{1.246}$ | <u>0.442</u> | $\color{red}{3.3}$ |
| Metric3Dv2 | 0.54 | 0.131 | 0.848 | <u>2.3</u> | **4.12** | **0.205** | **0.849** | **1.0** |

#### 지표별 상세

| RMSE↓ (m) | iBims-1 | NYUv2 | **실내 평균** | DIODE Outdoor |
|---|--:|--:|--:|--:|
| DepthLM-12B | $\color{red}{1.26}$ | $\color{red}{0.62}$ | $\color{red}{0.94}$ | 10.44 |
| Depth Anything V2 | <u>0.59</u>‡ | 0.46 | <u>0.53</u> | <u>8.19</u> |
| UniDepthV2 | **0.50** | <u>0.44</u> | **0.47** | $\color{red}{21.82}$ |
| Metric3Dv2 | 0.78 | **0.30** | 0.54 | **4.12** |

| AbsRel↓ | iBims-1 | NYUv2 | **실내 평균** | DIODE Outdoor |
|---|--:|--:|--:|--:|
| DepthLM-12B | 0.141 | $\color{red}{0.172}$ | $\color{red}{0.156}$ | <u>0.518</u> |
| Depth Anything V2 | <u>0.125</u>‡ | <u>0.120</u> | <u>0.123</u> | 0.742 |
| UniDepthV2 | **0.090** | 0.123 | **0.106** | $\color{red}{1.246}$ |
| Metric3Dv2 | $\color{red}{0.194}$ | **0.068** | 0.131 | **0.205** |

| δ1↑ | iBims-1 | NYUv2 | **실내 평균** | DIODE Outdoor |
|---|--:|--:|--:|--:|
| DepthLM-12B | 0.810 | $\color{red}{0.682}$ | $\color{red}{0.746}$ | $\color{red}{0.255}$ |
| Depth Anything V2 | <u>0.888</u>‡ | 0.891 | <u>0.890</u> | 0.336 |
| UniDepthV2 | **0.943** | <u>0.902</u> | **0.922** | <u>0.442</u> |
| Metric3Dv2 | $\color{red}{0.727}$ | **0.970** | 0.848 | **0.849** |

‡ DAv2 실내 모델의 상한(20 m)이 평가 cap(iBims-1 25 m)보다 작다.

#### 보조 지표 (도메인 평균)

| 모델 | log-RMSE↓ 실내 | log-RMSE↓ 실외(DIODE Outdoor) | SILog↓ 실내 | SILog↓ 실외(DIODE Outdoor) | 원거리 RMSE↓ 실내 (≥4 m) | 원거리 RMSE↓ 실외(DIODE Outdoor) (≥30 m) | 경계 δ1↑ (실내) | 경계에서 δ1 하락↓ |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| DepthLM-12B | $\color{red}{0.269}$ | 0.725 | $\color{red}{13.4}$ | $\color{red}{43.5}$ | $\color{red}{1.87}$ | 25.43 | $\color{red}{0.635}$ | <u>0.123</u> |
| Depth Anything V2 | <u>0.152</u> | <u>0.597</u> | 6.8 | 35.3 | 0.93 | <u>14.38</u> | <u>0.769</u> | 0.131 |
| UniDepthV2 | **0.139** | $\color{red}{0.730}$ | **6.0** | <u>32.9</u> | **0.84** | $\color{red}{26.46}$ | **0.802** | $\color{red}{0.134}$ |
| Metric3Dv2 | 0.167 | **0.299** | <u>6.2</u> | **22.1** | <u>0.87</u> | **7.60** | 0.762 | **0.089** |

log-RMSE와 SILog는 비율 오차라 먼 픽셀이 값을 지배하지 않는다. SILog는 사진마다 전체 배율을 맞춘 뒤 남는 오차라서, 전체적으로 길게·짧게 답하는 오차는 빠지고 장면 구조 오차만 남는다. 원거리는 실내 4 m 이상, 실외 30 m 이상이다. 경계는 iBims-1(공식 경계 지도)과 NYUv2(이웃 GT 깊이 비 > 1.1)에서 경계 3 px 이내 픽셀이고, 하락 = 내부 δ1 − 경계 δ1이다. Track A는 공통 점만 쓰므로 경계 점이 적고(iBims-1 391개, NYUv2 271개) 그만큼 CI가 넓다.

### 진행 상황

- [x] 파이프라인 검증 (DepthLM δ1 재현, baseline δ1 재현, RMSE 코드 검증)
- [x] 파일럿 (iBims-1, NuScenes)
- [x] 전체 데이터셋 평가
- [x] 결과 표 작성

## Track B — DepthVLM (dense)

깊이 맵 전체를 내는 모델끼리 valid GT 픽셀 전체로 비교한다. 예측 맵은 GT 원본 해상도로 맞춘 뒤 같은 픽셀에서 비교한다. 픽셀 수가 많아 원자료는 픽셀 단위가 아니라 이미지별 통계로 저장하며, 모든 지표는 이 통계에서 다시 계산할 수 있다.

### 비교 모델

| 모델 | GT intrinsics | 도메인 정보 |
|---|---|---|
| DepthVLM-4B | 사용 | 미사용 |
| Depth Anything V2 (metric, L) | 미사용 | 사용 (실내·실외 모델 따로) |
| UniDepthV2 (L) | 미사용 | 미사용 |
| Metric3Dv2 (ViT-L) | 사용 | 미사용 |

DepthVLM은 입력 사진을 GT 초점 거리 기준(f=1000) 크기로 맞춰서 넣으므로 GT intrinsics를 쓰는 모델로 분류한다.

### 학습 데이터 겹침

| 도메인 | 데이터셋 | 학습에 사용한 모델 |
|---|---|---|
| 실내 | iBims-1 | 없음 |
| 실내 | NYUv2 | 없음 |
| 실외 | DIODE Outdoor | 없음 |

DDAD·NuScenes도 같은 조건으로 돌렸지만 결과 표에서는 뺐다. DepthVLM은 DDAD·NuScenes를, Metric3Dv2는 DDAD를 학습했다. 결과 표는 모든 모델이 학습하지 않은 데이터셋만으로 비교하고, 원자료에는 두 데이터셋도 남아 있다.

### 결과

굵게 = 1위, <u>밑줄</u> = 2위, 빨간 숫자 = 꼴찌. 평균 순위 = 그 도메인의 데이터셋별 RMSE·AbsRel·δ1 순위를 평균한 값(작을수록 좋음). 값은 데이터셋 안 픽셀을 모은(pooled) 점추정이며 95% CI는 결과 원본 표(`tables/`)에 있다.

#### 한눈에 보기 (도메인 평균)

| 모델 | 실내 RMSE↓ | 실내 AbsRel↓ | 실내 δ1↑ | 실내 평균 순위↓ | 실외(DIODE Outdoor) RMSE↓ | 실외(DIODE Outdoor) AbsRel↓ | 실외(DIODE Outdoor) δ1↑ | 실외(DIODE Outdoor) 평균 순위↓ |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| DepthVLM-4B | $\color{red}{0.56}$ | **0.090** | **0.927** | <u>2.5</u> | <u>7.58</u> | <u>0.408</u> | 0.402 | <u>2.3</u> |
| Depth Anything V2 | <u>0.53</u> | 0.124 | 0.889 | $\color{red}{3.0}$ | 7.68 | 0.718 | $\color{red}{0.334}$ | $\color{red}{3.3}$ |
| UniDepthV2 | **0.47** | <u>0.107</u> | <u>0.921</u> | **2.0** | $\color{red}{19.90}$ | $\color{red}{1.042}$ | <u>0.476</u> | $\color{red}{3.3}$ |
| Metric3Dv2 | $\color{red}{0.56}$ | $\color{red}{0.132}$ | $\color{red}{0.845}$ | <u>2.5</u> | **3.84** | **0.193** | **0.859** | **1.0** |

#### 지표별 상세

| RMSE↓ (m) | iBims-1 | NYUv2 | **실내 평균** | DIODE Outdoor |
|---|--:|--:|--:|--:|
| DepthVLM-4B | 0.64 | $\color{red}{0.48}$ | $\color{red}{0.56}$ | <u>7.58</u> |
| Depth Anything V2 | <u>0.58</u>‡ | 0.47 | <u>0.53</u> | 7.68 |
| UniDepthV2 | **0.49** | <u>0.46</u> | **0.47** | $\color{red}{19.90}$ |
| Metric3Dv2 | $\color{red}{0.77}$ | **0.34** | $\color{red}{0.56}$ | **3.84** |

| AbsRel↓ | iBims-1 | NYUv2 | **실내 평균** | DIODE Outdoor |
|---|--:|--:|--:|--:|
| DepthVLM-4B | <u>0.099</u> | <u>0.081</u> | **0.090** | <u>0.408</u> |
| Depth Anything V2 | 0.126‡ | 0.121 | 0.124 | 0.718 |
| UniDepthV2 | **0.090** | $\color{red}{0.124}$ | <u>0.107</u> | $\color{red}{1.042}$ |
| Metric3Dv2 | $\color{red}{0.195}$ | **0.069** | $\color{red}{0.132}$ | **0.193** |

| δ1↑ | iBims-1 | NYUv2 | **실내 평균** | DIODE Outdoor |
|---|--:|--:|--:|--:|
| DepthVLM-4B | <u>0.911</u> | <u>0.943</u> | **0.927** | 0.402 |
| Depth Anything V2 | 0.887‡ | $\color{red}{0.891}$ | 0.889 | $\color{red}{0.334}$ |
| UniDepthV2 | **0.941** | 0.901 | <u>0.921</u> | <u>0.476</u> |
| Metric3Dv2 | $\color{red}{0.723}$ | **0.967** | $\color{red}{0.845}$ | **0.859** |

‡ DAv2 실내 모델의 상한(20 m)이 평가 cap(iBims-1 25 m)보다 작다.

#### 보조 지표 (도메인 평균)

| 모델 | log-RMSE↓ 실내 | log-RMSE↓ 실외(DIODE Outdoor) | SILog↓ 실내 | SILog↓ 실외(DIODE Outdoor) | 원거리 RMSE↓ 실내 (≥4 m) | 원거리 RMSE↓ 실외(DIODE Outdoor) (≥30 m) | 경계 δ1↑ (실내) | 경계에서 δ1 하락↓ |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| DepthVLM-4B | **0.134** | <u>0.489</u> | $\color{red}{11.0}$ | $\color{red}{38.5}$ | $\color{red}{1.22}$ | 19.19 | <u>0.765</u> | $\color{red}{0.168}$ |
| Depth Anything V2 | 0.153 | 0.583 | <u>7.6</u> | 38.4 | 0.94 | <u>12.99</u> | 0.762 | <u>0.139</u> |
| UniDepthV2 | <u>0.140</u> | $\color{red}{0.676}$ | **6.9** | <u>37.0</u> | **0.85** | $\color{red}{24.18}$ | **0.793** | 0.142 |
| Metric3Dv2 | $\color{red}{0.181}$ | **0.297** | 9.3 | **27.3** | <u>0.89</u> | **7.50** | $\color{red}{0.757}$ | **0.093** |

log-RMSE와 SILog는 비율 오차라 먼 픽셀이 값을 지배하지 않는다. SILog는 사진마다 전체 배율을 맞춘 뒤 남는 오차라서, 전체적으로 길게·짧게 답하는 오차는 빠지고 장면 구조 오차만 남는다. 원거리는 실내 4 m 이상, 실외 30 m 이상이다. 경계는 iBims-1(공식 경계 지도)과 NYUv2(이웃 GT 깊이 비 > 1.1)에서 경계 3 px 이내 픽셀이고, 하락 = 내부 δ1 − 경계 δ1이다.

### 진행 상황

- [x] 파이프라인 검증 (DepthVLM dense δ1 재현, dense 평가 코드 검증)
- [x] 파일럿 (iBims-1, NuScenes)
- [x] 전체 데이터셋 평가
- [x] 결과 표 작성

세부 설정과 규칙별 근거, 검증 기준은 [docs/PROTOCOL.md](docs/PROTOCOL.md)에 정리했다.

이 저장소에서 작성한 코드는 MIT 라이선스다([LICENSE](LICENSE)). 외부 코드와 모델 가중치, 데이터셋은 각자의 라이선스를 따르며 출처는 [NOTICE.md](NOTICE.md)에 적었다.
