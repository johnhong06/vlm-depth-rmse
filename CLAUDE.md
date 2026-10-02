# vlm-depth-rmse — 프로젝트 규칙

상위 `Jihyuck/CLAUDE.md` 의 공통 규칙이 그대로 적용된다. 진행 상황·결정 근거는 `NOTES.md`, 실제 적용한 평가 설정 전부는 `docs/PROTOCOL.md`.
**Track A 를 끝낸 뒤 Track B** (사용자 지시, NOTES D-11). 지금 규칙은 Track A 용이다.

## Track A — DepthLM 의 RMSE 재평가 (zero-shot)

DepthLM 은 δ1 만 보고했다. 같은 조건에서 **RMSE / AbsRel / δ1** 을 재서 DepthLM 과 pure vision 모델을 zero-shot(파인튜닝 없음)으로 비교한다.
원래 RMSE 비교는 한 데이터셋에 fine-tune 한 뒤 하지만 현실적으로 불가하므로, 비교 모델 대부분이 보지 않은 실내·실외 데이터셋에서 잰다. 범위는 측정까지.

- 평가: 공통 샘플 픽셀(sparse). DepthLM 은 그 픽셀에만 질의, dense 모델은 예측 맵(GT 원본 해상도로 리사이즈)에서 같은 위치 값.
- 모델: DepthLM-12B / DAv2-metric (L) / UniDepthV2 (L) / Metric3Dv2 (ViT-L) / Depth Pro — 공식 추론 설정 그대로 (NOTES D-13).
- 데이터셋: 실내 iBims-1·NYUv2, 실외 DDAD·nuScenes·DIODE Outdoor. 학습에 쓴 모델은 배제하지 않고 표에 표시 (NOTES D-12).

## 반드시 지킬 규칙 (사용자 지정 — 바꿔야 하면 진행하지 말고 먼저 보고)

1. 깊이 = z-depth. DepthLM 답은 DepthLM 공식 코드의 데이터셋별 GT 정의를 따른다 (2026-10-02 사용자 개정, NOTES D-17): nuScenes·DDAD 는 그대로 z, 나머지는 유클리드로 보고 z = d / sqrt(1 + ((u−cx)/fx)² + ((v−cy)/fy)²) (원본 좌표 + 원본 intrinsics). 부록: 전부 변환·전부 그대로 (`score.py --depthlm`).
2. 평가 좌표계는 GT 원본 해상도 하나. dense 예측은 이 크기로 리사이즈한 뒤 같은 픽셀 값을 뽑는다.
3. DepthLM 질의는 12B 의 통일 초점 750, 원본 좌표를 같은 배율로 옮겨 화살표를 그린다.
4. valid mask 와 데이터셋별 min/max cap 은 모든 모델에 동일 (벤치 `sample_points.py` 의 `DATASET_CONFIGS`, DIODE 는 0.05–80 m).
5. 결과는 데이터셋별 + 도메인별(실내/실외). 실내·실외를 섞은 평균은 내지 않는다.
6. 주 지표 = 데이터셋 안 전체 픽셀 pooled, 보조 = 이미지별 계산 후 평균. 표에 방식 명시.
7. 이미지 단위 bootstrap 95 % CI.
8. z 공간 RMSE 가 주 결과, 유클리드 공간 RMSE 는 부록.
9. DAv2-metric 모델 상한(실내 20 m, 실외 80 m) < 데이터셋 cap 이면 각주.
10. 모델마다 실행 환경 분리 (`envs/<모델>.txt`; H200 = conda, 없으면 uv — 지금 서버는 uv (NOTES F-6), 로컬 = `~/venv/<모델>`).

## 작업 원칙

- 수치가 논문과 크게 다르면 버그부터 의심하고 원인을 확인한 뒤 진행한다. 검증(NOTES 체크리스트)을 통과하기 전에는 본 실험을 돌리지 않는다.
- 공개 코드를 최대한 그대로 쓴다: DepthLM·DepthVLM 은 `third_party/`(고정 커밋), dense 모델 공식 저장소는 `prep/fetch_ext.sh` 가 `ext/` 에 고정 커밋으로 받는다. 새 코드는 짧게.
- H200 관리자에게 줄 데이터는 미리 팩으로 만들어 드라이브(`gdrive:h200_vdr`)에 올린다. 저장소 푸시는 사용자가 직접 한다.

## 원자료와 실행

원자료 parquet 열: `dataset, image_id, u, v, fx, fy, cx, cy, gt_z, pred, pred_raw, model` (DepthLM 은 pred = z 변환값, pred_raw = 유클리드 원답).
모든 지표는 이 파일에서 `eval/score.py`(표)·`eval/checks.py`(검증표: δ1 재현·sparse vs dense·z 변환 방향)로만 계산한다.

```bash
bash run.sh env                                    # H200: 데이터 없이 모델 5종 환경·공식 저장소·가중치·로딩 (KITTI 데모 한 장)
bash run.sh smoke                                  # H200: 환경·데이터·가중치 확인 (모든 모델, 몇 장씩)
bash run.sh depthlm ibims1 nuscenes                # DepthLM-12B 파일럿
bash run.sh dense all ibims1 nuscenes              # dense 모델 4종 파일럿
bash run.sh all ibims1 nuscenes                    # DepthLM + dense 4종을 한 작업으로
bash run.sh m3d_nyu                                # 검증: Metric3Dv2 NYUv2 RMS 0.251
```
