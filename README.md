# vlm-depth-rmse

δ1로만 보고된 DepthLM-12B의 metric depth 성능이 RMSE·AbsRel로 재도 유지되는지, pure vision 모델 4개와 같은 조건에서 비교해 확인한다.

## 동기

DepthLM은 Pixtral 기반 VLM으로 metric depth를 추정하는데, 논문은 성능을 δ1 하나로만 보고했다. δ1은 예측이 정답의 25% 안에 드는지만 세기 때문에, 틀린 예측이 얼마나 크게 틀렸는지와 큰 오차가 얼마나 자주 나오는지는 보여 주지 못한다. 그래서 오차 크기를 그대로 반영하는 RMSE로 봐도 DepthLM의 성능이 유지되는지 확인한다.

## 실험 설계

zero-shot으로, 파인튜닝 없이 평가한다. RMSE 비교는 보통 한 데이터셋에 파인튜닝한 모델끼리 하지만 우리 여건에서는 그렇게 할 수 없다. DepthLM 논문도 zero-shot으로 평가했으니 같은 조건이어야 지표만 바꾼 비교가 된다. 그래서 실내·실외 데이터셋은 비교 모델 대부분이 학습에 쓰지 않은 것으로 골랐다.

Track A는 공통 샘플 픽셀(sparse)에서 모든 모델을 비교한다. DepthLM은 픽셀 하나마다 VLM forward를 돌려야 해서 dense 평가가 사실상 불가능하기 때문이다. DepthLM은 정해 둔 픽셀에만 질의하고 dense 모델은 예측 맵에서 같은 위치의 값을 꺼낸다.

비교 모델은 5개, 평가 데이터셋도 5개(실내 2, 실외 3)다. 깊이 정의는 z-depth로 통일한다. 유클리드 거리로 답하는 DepthLM의 출력은 z-depth로 변환한다. 정의가 섞이면 가장자리와 원거리에서 RMSE가 부풀려지기 때문이다.

이번 단계는 측정까지만 한다. 원인 분석과 보완은 다음 단계에서 다룬다.

## 비교 모델

| 모델 | GT intrinsics | 도메인 정보 |
|---|---|---|
| DepthLM-12B | 사용 | 미사용 |
| Depth Anything V2 (metric, L) | 미사용 | 사용 (실내·실외 모델 따로) |
| UniDepthV2 (L) | 미사용 | 미사용 |
| Metric3Dv2 (ViT-L) | 사용 | 미사용 |
| Depth Pro | 미사용 | 미사용 |

모든 모델은 공식 추론 설정 그대로 돌린다. 모델마다 받는 입력 정보가 다르므로, 그 차이를 이 표처럼 결과와 함께 드러낸다.

## 평가 데이터셋

| 도메인 | 데이터셋 | 학습에 사용한 모델 |
|---|---|---|
| 실내 | iBims-1 | 없음 |
| 실내 | NYUv2 | 없음 |
| 실외 | DDAD | Metric3Dv2 |
| 실외 | NuScenes | DepthLM (같은 데이터셋의 다른 장면) |
| 실외 | DIODE Outdoor | 없음 |

학습에 쓰인 데이터셋도 빼지 않고 결과 표에 표시만 한다. 그래야 실내와 실외 결론을 따로 낼 수 있다.

## 결과

### 실내 (iBims-1, NYUv2)

| 모델 | iBims-1 RMSE↓ | iBims-1 AbsRel↓ | iBims-1 δ1↑ | NYUv2 RMSE↓ | NYUv2 AbsRel↓ | NYUv2 δ1↑ |
|---|---|---|---|---|---|---|
| DepthLM-12B | - | - | - | - | - | - |
| Depth Anything V2 | -‡ | -‡ | -‡ | - | - | - |
| UniDepthV2 | - | - | - | - | - | - |
| Metric3Dv2 | - | - | - | - | - | - |
| Depth Pro | - | - | - | - | - | - |

### 실외 (DDAD, NuScenes, DIODE Outdoor)

| 모델 | DDAD RMSE↓ | DDAD AbsRel↓ | DDAD δ1↑ | NuScenes RMSE↓ | NuScenes AbsRel↓ | NuScenes δ1↑ | DIODE Outdoor RMSE↓ | DIODE Outdoor AbsRel↓ | DIODE Outdoor δ1↑ |
|---|---|---|---|---|---|---|---|---|---|
| DepthLM-12B | - | - | - | -† | -† | -† | - | - | - |
| Depth Anything V2 | -‡ | -‡ | -‡ | - | - | - | - | - | - |
| UniDepthV2 | - | - | - | - | - | - | - | - | - |
| Metric3Dv2 | -† | -† | -† | - | - | - | - | - | - |
| Depth Pro | - | - | - | - | - | - | - | - | - |

† 학습 데이터가 겹친다. DDAD는 Metric3Dv2의 학습 데이터에 들어 있고 NuScenes는 DepthLM이 같은 데이터셋의 다른 장면으로 학습했다.

‡ DAv2 모델 상한(실내 20 m / 실외 80 m)이 평가 cap보다 작은 데이터셋이다. iBims-1은 cap이 25 m, DDAD는 120 m다.

RMSE는 z-depth 공간에서 데이터셋 안의 픽셀을 모두 모아(pooled) 계산하며 단위는 m다. 값은 "평균 (95% CI)" 형식으로 채울 예정이다.

## 진행 상황

- [ ] 파이프라인 검증 (DepthLM δ1 재현, baseline δ1 재현, RMSE 코드 검증)
- [ ] 파일럿 (iBims-1, NuScenes)
- [ ] 전체 데이터셋 평가
- [ ] 결과 표 작성

세부 설정과 규칙별 근거, 검증 기준은 [docs/PROTOCOL.md](docs/PROTOCOL.md)에 정리했다.

이 저장소에서 작성한 코드는 MIT 라이선스다([LICENSE](LICENSE)). 외부 코드와 모델 가중치, 데이터셋은 각자의 라이선스를 따르며 출처는 [NOTICE.md](NOTICE.md)에 적었다.
