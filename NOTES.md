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
- [x] (사용자) 커밋·푸시 → H200 `bash run.sh env` 제출 / 관리자에게 `h200_trackA` 전달 (서버 반영 대기)
- [x] (사용자) 추가 커밋·푸시 → H200 `bash run.sh depthlm ibims1` — 새 팩 없이 9/23 팩의 iBims-1 RGB 로 DepthLM 먼저 (D-15) → 멈춤(F-8) 뒤 smoke·파일럿으로 대체
- [x] 문서: README 한국어(humanize-korean 윤문)·`docs/PROTOCOL.md` 8 절 (`docs/protocol.md` 대체, D-16) — 아래 실행 로그 21:12
- [x] (사용자) 문서 변경 커밋·푸시 (c8a5945)
- [x] H200 `bash run.sh env` 통과 (commit 174141d, 서버 11:14–11:30) — 모델 5 종 환경·공식 저장소·가중치·로딩, 로컬 사전 점검과 같은 값 (실행 로그, F-6)
- [x] (사용자 확인) Metric3Dv2 '도메인 정보' — 사용자 확정(10-01): 공식 추론 설정 그대로, 표기는 '미사용', 따로 설명 없음 (D-13)

### Track A — 검증 (본 실험 전에 통과)
- [x] H200 `bash run.sh smoke` (10-02, commit bda89b0)
- [x] 파일럿 `all ibims1 nuscenes` (10-02, zip 874)
- [x] DepthLM δ1: 원답 = DepthLM 표 1 (nuScenes 0.823/0.819, DDAD 0.680/0.670), 변환값 = DepthVLM 표 1 (iBims-1 0.755/0.754, nuScenes 0.735/0.736, DDAD 0.651/0.654). NYUv2 는 GT 배율 차이로 설명됨(F-11, 2026-10-05)
- [x] baseline δ1: DepthVLM 표 2 sparse — 6/8 일치, 다른 3 칸은 F-9 (nuScenes·iBims-1) — iBims-1 로컬 사전 점검 완료(아래 로그): DAv2 0.887·UniDepthV2 0.941·Metric3Dv2 0.726 = 표 2 와 같음,
      Depth Pro 0.829 (표 2 0.880, UniDepthV2 논문 dense 0.823). nuScenes 는 H200 파일럿에서
- [x] Metric3Dv2 NYUv2 RMS 0.251 (`m3d_nyu.py`, 654 장, 로컬): **벤치마크 방식 + rawDepths GT → RMSE 0.253·AbsRel 0.063·δ1 0.975 (논문 0.251·0.063·0.975)** — 통과.
      depths(보정) GT 로는 0.345·0.073·0.958 → 논문은 raw 깊이로 평가한 것으로 보인다. Track A 와 같은 hub 경로는 0.261·0.066·0.974.
- [x] 샘플링 대표성: 공통 점 RMSE vs 전체 valid GT RMSE (`densestat_*`) — tables/track_a_checks.md ②
- [x] z 변환 방향 (F-10 → D-17)
- [x] 정렬: `overlay/` 그림 검토 — 파일럿 + 875·878 (NYUv2·DDAD·DIODE, DepthVLM 5 세트) 경계 어긋남 없음

### Track A — 본 실험
- [x] 5 개 데이터셋 × 5 모델 → `tables/track_a.md` (주표 + 유클리드 부록) + 부록 converted·raw + breakdown + checks (2026-10-04, zip 874 + 875)

### Track B (Track A 다음)
- [x] D-19 설계 → H200 `trackb` 5 세트 (zip 878) → `tables/track_b.md` (2026-10-04)

### VLM 약점 분석 (2026-10-07, D-20 · F-16 · F-17)
- [x] 프로토콜·지표 재점검: 두 VLM 논문·공식 코드, pure vision 4 종 공식 저장소 대조 (조사 에이전트 2 개 + 핵심 주장 직접 확인) — F-16
- [x] 로컬 예측 맵 7 조건 × 3 세트 (`~/data/vdr_maps`, Track B 와 반올림까지 일치) + ADE20K 분할 (`prep/semseg_ade.py`)
- [x] 경우별 통계·표·히트맵·CI, 이미지 카드 1,200 장, 경우별 갤러리 75 장, 예시 E1–E10 → `results_vlm_weakness/`, 보고서 `docs/VLM_WEAKNESS.md`
- [x] NYUv2 벤치 GT 8.19 m 상한 결함 확인 · Kinect 원측정 GT 로 재채점 (F-17)
- [x] (사용자 결정 10-08) NYUv2 GT = Kinect 원측정 픽셀만 (D-22)
- [x] Track A·B 주 결과 표·README 의 NYUv2 행을 원측정 GT(< 10 m)로 다시 냄 (D-22, `eval/nyu_raw_inputs.py`, `breakdown.py --nyu_boundary`) — 실행 로그 10-08 F-19 다음
- [x] Depth Pro 제외(D-23)·DDAD·NuScenes 표에서 제외, 실외 = DIODE(D-24) → tables/·README 재생성, 독립 검수 통과 (2026-10-08)
- [ ] (보류 — 사용자: '약점 분석 말고 표 먼저') `docs/VLM_WEAKNESS.md` 수치를 pure vision 3 종 기준으로 고치기. 결과 csv·그림(`results_vlm_weakness/`)은 이미 3 종으로 다시 만듦, 문서 숫자는 아직 4 종(Depth Pro 포함) 기준

### zero-shot 재정리·NYUv2 평가 재점검 (2026-10-08, D-21 · F-18 · F-19)
- [x] zero-shot 조합 확정 (논문 학습 목록 재확인) → D-21, 프로젝트 CLAUDE.md 규칙 11
- [x] NYUv2 두 VLM 평가 재점검: 원측정 10.0 m 포화값 포함 오류 발견·수정(`weak_nyu_raw.py`), 원측정 판 표·곡선·CI 재생성(`weak_ci.py --nyuraw`), `docs/VLM_WEAKNESS.md` 수치 정정 → F-18
- [x] 입력 틀 영향(SUN RGB-D 561×427 자른 입력 vs NYU 원본 640×480) 대조, Metric3Dv2 테두리 0 근처 예측 → F-18 · F-19
- [ ] (사용자 결정) NYUv2 주 평가를 ① 원측정 GT(<10 m, 지금 입력) ② 표준 BTS(원본 640×480 입력·Eigen crop) 중 무엇으로 — ② 면 pure vision 3 종·DepthLM 재추론 필요
- [ ] (선택) 실외 zero-shot 세트 추가 (KITTI·ETH3D, F-12) — 지금은 DIODE 한 세트뿐

### 이전 단계 기록 (재설계 전, 참고)
- 2026-10-01 오전: DepthVLM 비교·6 개 세트(SUN RGB-D·ETH3D 포함) 설계로 준비 → D-11 에서 Track A 재설계. F-1(ETH3D 정렬)은 Track A 에 해당 없음.

## 결정 기록

### D-1 (2026-10-01) Track A 점 = DepthVLM-Bench 공개 점 (사용자)
"코드에서 공개한 100개 샘플링 지점"의 해석이 둘이었다: DepthVLM-Bench 점(데이터셋당 1만 점; iBims-1 은 100장×100점)과
DepthLM 저장소 `examples/ibims1`(같은 100장×100점이지만 다른 위치, 겹침 5/10,000, 유클리드 GT — 2026-10-04 정정: 예제 라벨은 z, D-17). 사용자가 **DepthVLM-Bench 점**을 골랐다.
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
  → **사용자 확정 (10-01)**: "공식 추론 설정이니까 따로 설명 없이 그냥 사용" — 공식 경로 그대로, README·PROTOCOL·결과 표 모두 '미사용', 각주·설명 없음.
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

### D-15 (2026-10-01) 새 팩이 서버에 오기 전에 DepthLM iBims-1 을 먼저 돌린다
9/23 팩(/app/data/HJ)의 `depthlm_distill_h200/eval/ibims1/rgb` 100 장이 벤치 원본(ibims1_core_raw/rgb)과 바이트 단위로 같다 (로컬 md5 100/100).
DepthLM 은 RGB 만 쓰므로(GT·intrinsics 는 저장소) `run.sh depthlm ibims1` 이 새 팩을 못 찾으면 9/23 팩에서 그 RGB 와 가중치만 풀고,
`bench/ibims1_rgb.sha256`(벤치 원본의 SHA256)과 100/100 일치할 때만 쓴다. 로컬에서 같은 구조의 작은 팩으로 데이터 단계와 2 점 질의를 확인 —
원본 이미지로 돌린 결과와 같다. 결과는 /app/output 에 남아 이후 `all` 작업이 이어받는다(이미 답한 점은 건너뜀). dense 모델은 GT 맵이 필요해 새 팩을 기다린다.
`h200/unpack.py` 는 일부 경로만 푼 경우를 경로 목록별로 기록하도록 고쳤다(가중치만 푼 뒤 다른 경로가 필요할 때 건너뛰지 않게).

### D-16 (2026-10-01) README·PROTOCOL 문서 기준 (사용자 지시: "README 와 세부 프로토콜 문서 작성", README 는 한국어 + AI 티 없는 문체)
- README = 처음 보는 사람용 한 화면 요약(사용자 지정 순서·결과 표 템플릿·체크리스트). 실행 명령·폴더 구조는 넣지 않았다(CLAUDE.md 에 있음).
- 설계와 '이유'는 Track A 설계 프롬프트에서만 가져오고, 설정값은 실제 적용한 값(코드·벤치 파일·결정 기록)으로 채운다 — 설계가 "실제 적용한 모든 설정값을 프로토콜 문서에 기록"을 요구하므로. 정하지 않은 값은 TBD (현재 6 절에 해당 없음).
- Metric3Dv2 도메인 정보: 처음엔 README 에 설계값 '사용' + 각주(확인 중)를 달았으나, 사용자 확정에 따라 README·PROTOCOL·결과 표 모두 '미사용'으로 각주·설명 없이 적는다 (D-13).
- 결과 표의 DAv2 상한 표시는 해당 칸(iBims-1·DDAD 의 DAv2)에 ‡ 로 단다 — 사용자 템플릿의 "데이터셋 표시" 각주를 칸 표시로 해석.
- 파일 이름은 사용자 지정대로 `docs/PROTOCOL.md` (git mv). 참조는 CLAUDE.md 한 곳.

### D-17 (2026-10-02) 규칙 1 개정 — DepthLM 답의 깊이 정의는 DepthLM 공식 코드의 데이터셋별 GT 정의를 따른다 (사용자 결정)
- 주 결과(`score.py --depthlm official`): nuScenes·DDAD 는 답을 그대로 z 로(공식 정리 코드가 z 라벨: `curate_nuscenes_*.py` points_cam[2], `curate_ddad.py` 깊이맵),
  iBims-1·NYUv2 는 유클리드로 보고 z 로 변환(공식 예제·정리 코드가 유클리드), DIODE Outdoor 는 공식 정의가 없어 질문의 기본 뜻(유클리드)으로 변환.
- 부록: `converted`(전부 변환 = 원래 규칙 1, DepthVLM 논문 방식) · `raw`(전부 그대로). 두 값이 원자료에 있어 재실행 없음.
- 왜: dense 모델은 자기가 학습한 정의(z)로 평가받으므로 DepthLM 도 자기가 배운 정의로 평가해야 공정하다(사용자 제기). DepthLM 논문도 그렇게 평가했다
  (답 그대로 δ1 이 DepthLM 표 1 과 일치: nuScenes 0.823 vs 0.819, DDAD 0.680 vs 0.670).
- 선택 기준은 우리 결과가 아니라 공식 코드의 정의로 미리 정했다 — GT 와 비교한 실측(F-10, 검증 ③)으로 데이터셋마다 고르면 정답을 보고 유리한 쪽을 고르는 셈.
  그래서 실측상 답이 z 에 가까운 iBims-1 도 공식 정의대로 변환한다(DepthLM 에 불리한 쪽).
- **정정 (2026-10-04, 사용자 승인)**: 위 'iBims-1 공식 예제가 유클리드' 는 틀렸다. 공식 예제 `DepthLM_Official/examples/ibims1/ibims1_val.jsonl`(100 장 × 100 점)의 라벨을
  GT 깊이 지도(png / 1310.7)와 같은 좌표에서 대조 → z 와 10,000/10,000 일치, 유클리드와는 0.8 %. iBims-1 정리 코드는 공개돼 있지 않다. NYUv2 는 `curate_NYU.py` 가 유클리드(재확인).
  → Z_SETS = {nuscenes, ddad, ibims1}. 원칙(공식 정의)은 그대로, 분류만 바로잡음. 원자료에 두 값이 있어 재실행 없음.
  iBims-1 DepthLM (공통 9,750 점): RMSE 1.352 → 1.261, AbsRel 0.167 → 0.141, δ1 0.755 → 0.810. RMSE 는 여전히 5 개 중 꼴찌. 실내 평균 순위 4.2 → 4.0.
  δ1 0.755 = DepthVLM 표 1 의 0.754 → DepthVLM 논문은 iBims-1 도 변환해 쟀다. DepthLM 표 1 의 0.870 과 남은 차이(0.06)는 원인 미확인:
  공식 예제 점의 거리 분포가 우리 점과 거의 같아(4 m 이상 23.6 % vs 22.6 %, 중앙 2.76 vs 2.70 m) 거리 구성으로는 설명되지 않고, 논문의 '8,192 무작위 샘플' 이 이 예제와 같은 세트인지 알 수 없다.
- 다른 선택지: 전부 변환(원래 규칙, DepthVLM 과 비교 쉬움, 주행 세트에서 DepthLM 에 불리) / 전부 그대로(NYUv2·DIODE 에서 정의가 틀림) / 실측 기반 선택(GT 를 보고 고름 — 기각).

### D-18 (2026-10-02) 보조 집계 추가 — 거리 구간·경계/내부·log 지표 (외부 피드백, 사용자 결정)
- 거리: 고정 미터 (실내 0–2 / 2–4 / 4 m–, 실외 0–10 / 10–30 / 30 m–). 데이터셋별 3 등분은 '원거리' 뜻이 달라져 기각. 처음 검토한 실내 3·6 m 는 NYUv2 원거리 210 점(2 %)이라 기각.
- 경계: iBims-1 공식 경계 지도(86/100 장, 나머지 14 장 제외), NYUv2 이웃 GT 깊이 비 > 1.1 (Depth Pro 정의; iBims-1 공식 대비 재현 88 %, 일치 95 %), 3 px.
  NuScenes·DDAD(LiDAR)·DIODE 제외 — DIODE 는 같은 정의가 점의 41 %(문턱 25 % 로도 29 %)를 경계로 잡음: 그림으로 보니 덤불·나무 전체, 유리창 안쪽(레이저 투과), 보도 스캔 줄무늬.
  → 'DIODE 실외 GT 가 식생·유리에서 고르지 않다'는 것은 이제 그림 근거가 있다(앞서 철회한 GT 품질 얘기의 일부는 맞았던 셈). 이 표시(`boundary_ratio`)는 원인 분석 단계에서 'GT 요철 영역'으로 쓸 수 있게 파일에 남긴다.
- SILog: KITTI 정의(λ = 1, ln, ×100, 이미지별 평균). 저장소마다 다름(DAv2 λ = 0.5, Metric3D log10, UniDepth std) — 확인한 코드는 ext/ 의 각 평가 파일. log 지표에서만 cap 범위로 자름.
- 검증: breakdown 의 'all' 행 = score.py 주 결과와 같음, SILog 직접 계산과 같음(UniDepthV2 iBims-1 5.702).

### D-19 (2026-10-03) Track B 설계 (사용자 결정)
- 목적: dense 예측이 가능한 모델끼리 valid GT 전체 픽셀로 비교. **DepthVLM 의 sparse(공통 점) 평가는 하지 않는다** — Track B 의 목적이 아님(사용자).
- 모델: DepthVLM-4B + Track A 와 같은 pure vision 4 종 (DAv2-metric-L, UniDepthV2-L, Metric3Dv2-L, Depth Pro). DA3 metric 은 넣지 않음.
- 데이터셋·mask·cap·깊이 정의(z)·평가 해상도(GT 원본, 예측 bilinear 리사이즈)·집계(pooled 주, 이미지별 보조, 이미지 bootstrap)·보조 집계 = Track A 와 같음.
- 원자료: 이미지별 통계 (거리 구간·경계/내부별 n·제곱오차 합·상대오차 합·δ1 적중·log 오차 합·log 오차 제곱합) — 픽셀 parquet 은 모델당 수억 행이라 대신함.
- zero-shot 여부 (DepthVLM 논문 §3·부록 A·표 9: 학습 = Argoverse2·Waymo·DDAD·nuScenes·ScanNet++·Taskonomy·HM3D·MP3D):
  DepthVLM 은 DDAD·nuScenes 를 학습 분할로 학습(평가 분할과 장면 다름) → †. iBims-1·NYUv2·DIODE 는 zero-shot. pure vision 4 종은 D-12 그대로.
- DepthVLM: Qwen3-VL-4B + DPT 헤드, 입력을 jsonl canonical_size(= GT fx 기준 f=1000)로 리사이즈 → GT intrinsics 사용, 출력 z, 값 > 0 (Softplus).
- 검증: DepthVLM 저장소 README 의 dense 표(δ1, 공식 방식 = canonical 해상도·이미지별 평균): DepthVLM nuScenes 0.838 / iBims-1 0.910,
  UniDepthV2 0.868 / 0.941, Metric3Dv2 0.843 / 0.724, DepthPro 0.379 / 0.879 (논문 본문에는 없음, NYUv2·DDAD·DIODE 참고 수치 없음).
- README 를 공통 설계 / Track A / Track B 로 나눔.

### D-20 (2026-10-07) VLM 약점 분석 설계 (사용자 지시: 두 VLM 이 학습하지 않은 세트 기준, 그 데이터를 보지 않은 pure vision 과 비교, 이미지 예시와 모든 경우를 그림으로 저장)
- 세트: iBims-1·NYUv2·DIODE (두 VLM·pure vision 4 종 모두 zero-shot). DDAD·nuScenes 는 VLM 학습 데이터라 뺀다. KITTI·ETH3D 추가(F-12)는 이번에 하지 않음(사용자 '우선' — 다음 후보).
- 예측 맵: Track B 는 통계만 저장했으므로 로컬 GPU 로 같은 코드를 다시 돌려 맵을 저장한다(`eval/dense_full.py --save_maps`, 실행 `prep/weak_maps.sh`). H200 은 왕복 비용이 커서 기각. 로컬 sdpa 와 H200 flash-attention 2 의 차이는 반올림 수준(검증함).
- 물체 라벨: NYUv2 = GT 894 종. iBims-1·DIODE 는 GT 가 없어 Mask2Former Swin-L ADE20K 공개 체크포인트로 분할한다. 대안(평면 마스크만 쓰거나 물체 분석 생략)은 '물체' 요청을 채우지 못한다. 두 라벨을 같은 대분류(`weak_common.CATS`)로 묶는다.
- 공정성 대조: GT intrinsics 비대칭의 크기를 재려고 UniDepthV2+K·DepthPro+f 부록 조건을 더한다. 주 비교는 바꾸지 않는다.
- 구조 지표: iBims-1 DBE·평면성은 공식 코드를 구하지 못해 논문 식 1–4 를 재구현한다(Canny 0.1/0.2 + 변형 2 개, GT 자기 점검). 모델 간 상대 비교로만 쓴다.
- 저장 위치: 그림·표 = `results_vlm_weakness/`(results*/ 규칙으로 git 제외, 약 0.5 GB), 맵 6 GB = `~/data/vdr_maps`(대용량은 ~/data 공통 규칙), 원자료 사본 = `~/data/vdr_raw/874·875·878`(예전엔 /tmp 에만 있었음). 보고서용 사본만 `docs/figs/vlm_weakness/`(약 3 MB).
- 판정: VLM ÷ pure vision 4 종 중앙값. 공통 약점 = 두 VLM 모두 > 1 이고, 그 경우가 있는 세트에서 이미지 bootstrap 95 % CI 가 1(또는 0)을 넘는 것.

### D-21 (2026-10-08) 분석은 zero-shot 조합만, 실내·실외 따로 (사용자 지시: "이제부터 분석할 때 zero-shot 인 데이터만 판단, 비교 모델도 zero-shot 인 경우만")
- 학습 목록(논문 원문 재확인): DepthLM = Argoverse2·Waymo·nuScenes·ScanNet++·Taskonomy·HM3D·Matterport3D (부록 표 4), DepthVLM = 같은 7 종 + DDAD (§3).
  pure vision 은 D-12 그대로 (Metric3Dv2 만 DDAD 학습).
- 결과 조합: 실내 iBims-1·NYUv2 = 6 모델 모두 zero-shot. 실외 DIODE = 6 모델 모두. DDAD = DepthLM + DAv2·UniDepthV2·Depth Pro (DepthVLM·Metric3Dv2 제외).
  nuScenes 는 두 VLM 모두 학습 → VLM 분석에서 뺀다. 순위·비는 이 조합 안에서만 낸다.
- 기존 표(Track A·B 주 결과)는 그대로 두고, 분석·보고에서만 이 조합을 쓴다.

### D-22 (2026-10-08) NYUv2 는 원측정만, GT 초점거리는 공식 설정 문제로만 본다 (사용자 지시)
- NYUv2 GT = labeled.mat rawDepths 의 실측 픽셀만 (0.005 m ≤ GT < 10 m, F-18). 채운 값(벤치 depth_bfx, 채움 픽셀)과는 비교하지 않는다 — 주 표·README 도 이것으로 바꾼다.
- GT 초점거리: 각 모델은 논문·공식 코드대로 (쓰도록 설계된 모델에만 준다, D-13). 이것을 공정성 문제로 다루지 않는다. UniDepthV2+K·Depth Pro+f 는 참고용 부록 조건, 주장에 쓰지 않는다.
- 'SILog 만 유독 나쁜가' 판단은 대체적인 경향이면 충분 (모든 모델보다 나쁠 필요 없음).

### D-23 (2026-10-08) Depth Pro 를 결과 표에서 뺀다 (사용자: "DepthPro 는 뭔가 이상하다. 표에서 다 빼자")
- 대상: tables/ 의 track_a·track_a_breakdown·track_b(.csv·.md)·track_a_depthlm_converted·raw.md, README 결과 표·비교 모델 표. 원자료(874·875·878)와 실행 코드는 그대로 둔다.
- 남은 모델 값은 바뀌지 않는다: 공통 점은 DepthLM 이 정하고 bootstrap 재표본은 모델 목록과 무관 → 다시 만든 csv 의 남은 행이 이전과 차이 0.0 (확인함). 순위·굵게·밑줄·평균 순위만 바뀐다.
- 그동안 본 이상한 점(README 에 한 문단): 예측을 자르지 않아 0.08 % 픽셀이 10⁴ m → 실외 RMSE 지배(DIODE 88.8, 80 m 로 자르면 9.25), DepthVLM 표 2 의 수치는 GT 초점을 넣은 값(F-9),
  공개 가중치는 재학습한 참조 구현이고 학습 데이터 미공개(D-12).
- 그대로 둔 것: track_a_checks.md(논문 재현 확인 기록), docs/VLM_WEAKNESS.md(pure vision 4 종 중앙값 기준 — 빼려면 비·CI 를 다시 계산해야 함, 사용자 확인 대기).
- **확장 (같은 날, 사용자: "뎁스프로는 걍 우리 실험에서 제외")**: 약점 분석도 pure vision 3 종(UniDepthV2·Metric3Dv2·DAv2)으로 — `weak_common.PV`·`SUPP`(UniDepthV2+K 만), 목록을 박아 둔 weak_ci·weak_tables(_nyuraw)·weak_examples 를 공용 목록으로, E4 확대 그림의 Depth Pro 칸은 Metric3Dv2 로. 앞으로의 실행에서도 뺌: run.sh 의 env·smoke·all·dense all·trackb 기본 목록, prep/weak_maps.sh. depthpro 어댑터·환경은 지난 결과 재현용으로 남김(`run.sh dense depthpro` 로만).

### D-24 (2026-10-08) 결과 표는 zero-shot 세트만 — DDAD·NuScenes 를 빼고 실외는 DIODE 하나 (사용자: "DDAD nuscene 도 표에서 제외, 실외는 DIODE 만 해서 다시 순위 집계")
- 이유: DDAD = Metric3Dv2·DepthVLM 학습, NuScenes = DepthLM·DepthVLM 학습 (D-12·D-19·D-21). 원자료와 실행 코드는 그대로.
- tables/ 재생성(NYUv2 원측정 입력, Depth Pro 제외 상태 그대로). 실외 평균 = DIODE 값. `readme_tables.py` 는 csv 에 있는 데이터셋만 쓰고, 도메인에 데이터셋이 하나면
  상세 표의 평균 열을 빼고 머리에 '실외(DIODE Outdoor)' 로 적는다 — 5 세트 csv 로는 예전 출력과 바이트 단위로 같음(회귀 확인).
- 남은 세트의 점추정은 그대로. 바뀐 것: 실외 평균·평균 순위, 그리고 Track A DIODE 의 CI (score.py 는 데이터셋 순서대로 한 난수열에서 재표본을 뽑아
  앞의 DDAD·NuScenes 가 빠지면 DIODE 재표본이 달라진다; Track B·breakdown 은 데이터셋마다 seed 0 이라 그대로).
- 검수(생성 코드와 따로 짠 코드, scratchpad audit_tables.py): 원자료 → csv (Track A 공통 점 직접 계산, Track B 통계 합) 일치, 도메인 평균 = 데이터셋 평균·실외 = DIODE,
  README 모든 숫자 = csv 반올림, 굵게·밑줄 = 표시 자릿수 기준 1·2 위, 평균 순위 재계산 일치, 보조 지표 표 일치 — 전부 통과.
- README: 공통 설계 문장, 두 트랙 '학습 데이터 겹침' 표(남은 3 세트 + 뺀 이유 한 줄), † 각주 삭제, ‡ 각주는 iBims-1 만. 진행 상황 체크리스트(파일럿 NuScenes 등)는 기록이라 그대로.

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
- **F-6 H200 이미지에 conda 없음 (2026-10-01 env 작업)**: PATH 에도 `/opt/conda/bin/conda` 에도 없어 run.sh 가 모델마다 uv 로 대체했다(`uv venv -p 3.12`).
  모델별 환경 분리(규칙 10)는 그대로이고 Python 3.12·torch 2.7.1+cu128·flash-attn 휠(cp312)도 정상. 로그의 '!!! [env] conda 실패 → uv' 와 pip root 경고는 이 대체 과정의 메시지라 무시해도 된다.
- **F-8 H200 작업이 끝나지 않던 원인 = bash 5.1 의 `wait` (확인됨, 2026-10-02)**: run.sh 는 출력을 `exec > >(tee …)`(프로세스 치환)로 보낸다.
  H200 이미지(Ubuntu 22.04, bash 5.1.16)에서는 인자 없는 `wait` 가 백그라운드 작업뿐 아니라 이 tee 까지 기다리는데, tee 는 스크립트가 끝나야 끝나므로 영원히 멈춘다.
  로컬 bash 5.3 에서는 멈추지 않아 로컬 시험에서 드러나지 않았다. V3 는 프로세스 치환을 쓰지 않아 해당 없음.
  · 증거: smoke(작업 872) 묶음 — DepthLM iBims-1 48 점은 끝났는데(`log_depthlm_ibims1_0.txt` 요약 있음) run.sh 는 요약을 찍지 않았고 NYUv2 로그도 없음 = DepthLM 프로세스가 끝난 뒤 `wait` 에서 멈춤.
    10 분 진행 줄이 없던 것도 프로세스가 10 분 전에 끝났기 때문. 첫 depthlm 작업(21:42 KST, 11 h+)도 같은 `wait` — DepthLM 은 끝났을 가능성이 크다(track_a 결과 확인 필요).
  · 재현: 로컬 pytorch/pytorch 컨테이너(bash 5.1.16)에서 최소 스크립트가 `wait` 에서 멈춤(종료 코드 124), 로컬 bash 5.3 은 통과.
    run.sh 구조(tee + 종료 trap + DepthLM 함수)를 그대로 옮긴 시험: 8b695c4 는 멈춤(124), 6a6f66b 이후(PID 를 준 `wait`)는 두 데이터셋을 돌고 정상 종료.
  · 수정: 6a6f66b 에서 `wait "${pids[@]}"` 로 바뀌어 이미 해결(멈춤 감지 수정 때 우연히). 남은 `wait` 는 종료 trap 의 `wait $TEE` 뿐(5.1 에서 정상 — env 작업이 끝남). run.sh 에 경고 주석.
  · 부수 실측: **H200 DepthLM 속도 0.447 s/점 (프로세스 1 개)**, 파싱 실패·잘린 답 0. 48 점 δ1 0.417 은 첫 이미지 한 장(lectureroom_06, 로컬에서도 크게 낮게 답한 장면)이라 대표값 아님.
- **F-7 V3 (depthlm-distill-v3, H200 에서 4 h 작업 정상) 와 H200 사용 방식 차이 (2026-10-02, DepthLM 본 실행이 끝나지 않는 원인 후보)**:
  ① 로그 — V3 는 명령마다 `| tee -a 로그`, 여기는 스크립트 전체를 `exec > >(tee …)` 로 돌리고 EXIT 에서 그 tee 를 기다림.
  ② 파이썬 출력 — V3 는 `PYTHONUNBUFFERED=1`·`python -u`, 여기는 없음(주요 줄만 flush).
  ③ DepthLM 환경 — V3/V2 교사(H200 4 프로세스 라벨링 정상)는 이미지 파이썬 + torch 2.11 + transformers 5.16.1, `device_map="cuda:0"`, sdpa.
     여기는 공식 버전(uv 가상환경, torch 2.7.1 + transformers 4.51.1 + flash-attn 2.8.3, text FA2 + vision eager, `device_map="auto"`).
     H200 에서 이 조합은 env 점검 8 점(배치 2, 같은 크기 이미지 → 패딩 없음)만 통과했고, 긴 실행(배치 최대 4, 크기가 다른 이미지 → 왼쪽 패딩)은 끝난 적이 없다.
  ④ 옵션 — V3 는 `KEY=값` 을 명령 인자로 받음(이슈 한 줄 명령에서 환경변수 불가), 여기는 환경변수만 → **수정함**(NPROC·STALL_MIN·BUDGET_MIN·PROGRESS_SEC·DATA_SRC).
  같은 점: 백그라운드 프로세스 + wait, /app/scratch 작업마다 비워짐. → 디버그 묶음(`vdr_debug.tar.gz`)으로 ①과 ③ 중 어느 쪽인지 가린 뒤 고친다.
  ③이면 V3 의 검증된 방식(sdpa, cuda:0)으로 바꾸는 것은 공식 설정(FA2)과 다르므로 사용자 확인 후(둘 다 exact attention, D-12 에서 δ1 동일).
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
- 2026-10-01 21:12 문서 정리 (D-16): README 한국어로 새로 씀 → humanize-korean light 경로(변경률 0.6 %, 게이트 수렴, 문장 5 곳). `docs/protocol.md` → `docs/PROTOCOL.md` 8 절로 다시 씀.
  문서에 넣기 전 확인한 것:
  · 벤치 4 종의 점 좌표가 공식 `sample_points.py` seed 42(`RandomState(42 + i)`)로 그대로 재생성됨 — iBims-1 100/100, NYUv2 654/654, DDAD 1,000/1,000, nuScenes 1,000/1,000 장 일치 → 6 절 seed 칸.
  · 논문 참고 수치 19 개를 원문 표와 대조, 모두 일치: DepthLM v2 표 1 "Ours - Pixtral (12b)"(데이터셋마다 무작위 8,192 샘플), DepthVLM v3 표 1 "DepthLM-12B"·표 2(sparse),
    Metric3D v2 v4 표 1 NYUv2 "Ours ViT-L CSTM_label ZS" 0.251 / 0.063 / 0.975. DIODE 는 DepthLM·DepthVLM 표에 없음. DepthLM 표 2 의 pure vision 수치는 UniDepthV2·Depth Pro 논문에서 옮긴 값이라 기준으로 쓰지 않음.
  · 코드 확인: UniDepthV2 출력 상한 exp(10) ≈ 22 km(decoder `radius` clip), Metric3D ViT-L canonical 상한 200 m(config max_value), score.py 는 예측을 자르지 않음.
- 2026-10-01 21:26 Metric3Dv2 도메인 정보 사용자 확정 반영 (D-13): README 모델 표 '미사용'·각주 삭제, PROTOCOL 1 절 표 '미사용'·설명 단락 삭제·8 절 대기 항목은 D-6 하나로,
  `eval/score.py` META 의 괄호 설명 삭제("no"). 측정 방식·코드 경로는 그대로 (py_compile 통과).
- H200 `bash run.sh env` (commit 174141d, 서버 시각 11:14:49–11:30:09, 15 분, H200 NVL 140 GiB) — **통과**. 사용자가 콘솔 로그를 붙여 줌. 결과 zip 4 KB (로그와 같은 내용).
  · 환경: 5 종 모두 uv 로 생성(F-6), torch 2.7.1+cu128, cv2 5.0.0 (로컬과 같은 버전). 공식 저장소 4 개 고정 커밋으로 받음.
  · dense 4 종 KITTI 데모(375×1242, LiDAR 96,131 점) — 로컬 사전 점검과 표시 자릿수까지 같음:
    DAv2 중앙 28.0 m·AbsRel 0.085·δ1 0.961 / UniDepthV2 28.9·0.109·0.980 / Metric3Dv2 26.4·0.042·0.997 / Depth Pro 28.8·0.079·0.993.
  · DepthLM: 9/23 팩 가중치 16 조각 SHA256 통과 → /app/scratch/vdr_work/old. attention flash_attention_2(공식, 로컬은 sdpa), 정규화 이미지 1317×397, 8 점 파싱 8/8.
    답 5 개는 로컬과 같고 3 개가 0.07–0.17 m(0.2–0.7 %) 다름 — attention 커널·GPU 가 달라 생기는 bf16 수치 차이로 본다.
    8 점에서는 변환 전 답이 GT z 에 더 가깝다(δ1 8/8 vs 변환 후 6/8, AbsRel 0.116 vs 0.151). 다만 사진 1 장·8 점이고 KITTI 는 DepthLM 학습에 없으며,
    먼 점(GT 44–47 m)을 37–39 m 로 낮게 답해 값을 줄이는 변환이 불리하게 보이는 면도 있다 → 판단 근거 아님. D-6 은 파일럿 1 만 점의 checks ③ 로 정한다.
  · 다음 작업에서 '[env] … 생성' 이 다시 나오면 /app/scratch 가 작업마다 비워지는 것 → 매 작업 환경·가중치 준비에 10 여 분이 더 든다.
- 2026-10-01 run.sh 콘솔 요약 수정: DepthLM 프로세스 로그에서 뽑는 줄이 프로세스마다 4 줄(시작·요약·δ1 두 줄)인데 `tail -n 3×NPROC` 이라
  첫 프로세스의 요약 줄(s/점·파싱 실패·잘린 답 수)이 콘솔에서 빠졌다 → 4×NPROC. 중간 작업은 콘솔 로그(채점표·δ1 재현·z 변환 구간표)만으로 확인하고,
  zip 은 파일럿(겹침 그림 = 정렬 확인)과 본 실험 끝(원자료 parquet·유클리드 부록)에만 받는다.
- 2026-10-02 09:02 H200 `depthlm ibims1` (10-01 21:43 KST 시작)이 11 h 20 m 지나도 끝나지 않음 — 비정상(로컬 GPU 속도 초당 0.8 점이어도 3.4 h).
  추론 중 콘솔 출력이 없어 멈춘 단계·원인 불명 → 사용자에게 콘솔 로그 요청, 작업 취소 권고(결과는 프로세스마다 320 질의마다 저장, 재실행 시 이어서).
  run.sh: 추론 중 10 분마다(PROGRESS_SEC) 프로세스별 마지막 진행 줄(누적 질의 수·s/점, 진행 줄 전이면 로그 끝 150 자)과 nvidia-smi 사용률·메모리를 콘솔에 찍고,
  [depthlm] 줄에 시작 시각·CPU 수 추가. 가짜 프로세스 2 개로 로컬 시험(진행 줄·로딩 중 표시·종료 감지·요약 8 줄) 통과.
- 2026-10-02 연구원이 보내 준 그 작업의 콘솔 로그 (commit c8a5945, 서버 12:42:48 UTC = 21:42 KST): 새 팩 없음 → 9/23 팩 대체(16 조각 SHA256 통과, RGB 100/100 일치),
  `[env] depthlm 생성` = /app/scratch 가 작업 사이에 비워짐(F-6 의 예상대로, 작업마다 환경 재생성), `[depthlm] … 프로세스 2` 다음 줄 없음 → 추론 단계에서 끝남.
  느렸는지·멈췄는지·강제 종료됐는지는 이 로그로 구분 불가 → 프로세스 로그 `/app/output/vdr/track_a/log_depthlm_ibims1_{0,1}.txt` 요청.
  같은 시점에 연구원이 `h200_trackA` 데이터를 서버에 올림(아마 /app/data/HJ). unpack.py 탐색 범위를 가짜 팩으로 시험: HJ 바로 아래·HJ/폴더·HJ/폴더/폴더·드라이브 zip 그대로 → 찾음,
  HJ 아래 세 단계 폴더 → 못 찾음('조각 0 개', 그때는 `DATA_SRC=<경로>` 로 지정). 드라이브 폴더는 평평한 구조(팩 조각·sha256·README_ADMIN).
  9/23 팩 우회(D-15)는 더 필요 없음.
- 2026-10-02 10:35 run.sh 스스로 멈추고 정상 종료 (사용자 지적: 콘솔 로그는 작업이 정상 종료해야 받을 수 있고 강제 종료하면 못 본다):
  · 시간 제한 `BUDGET_MIN` (smoke 120, depthlm·all·dense 360분, 그 밖 무제한) — 넘으면 DepthLM 프로세스를 멈추고 남은 데이터셋·dense 를 건너뛴 뒤 표·zip 까지 정상 종료.
  · DepthLM 멈춤 감지 `STALL_MIN` 40분 — 프로세스 로그가 그동안 그대로면 중단하고 다음 데이터셋 DepthLM 도 건너뜀(원인이 같을 가능성), dense·표는 계속.
  · 종료 신호를 무시하면 1 분 뒤 SIGKILL, 그래도 살아 있으면(GPU 호출에서 멈춤) 기다리지 않고 진행. dense 는 `timeout -k 60 <남은 시간>`.
  · `eval/depthlm_sparse.py` 저장·진행 줄을 320 → 64 질의마다(정상 속도 1 분 안팎) — 멈춤 감지가 느린 경우를 멈춤으로 오인하지 않게, 중단 시 잃는 양도 줄어듦. 결과 값과 무관.
  · 가짜 프로세스로 시험 6 종 통과: 정상 종료 / 멈춤 → 중단 / 멈춤 뒤 다음 데이터셋 건너뜀 / 시간 제한(진행 중) / dense 도중 시간 제한 / SIGTERM 무시 프로세스.
  · **수정 (사용자 지적: "임계를 두면 정상인데 멈출 수도")**: 시간 제한은 정상 작업도 끊을 수 있어 기본을 끔(`BUDGET_MIN` 0, 원할 때만 켬). 멈춤 감지만 기본으로 남김 —
    진행 줄이 64 질의(정상 1 분 안팎)마다라서 40 분 무진행은 정상보다 40 배 이상 느려야 생긴다. 걸려도 64 질의마다 저장돼 다시 넣으면 이어서 돈다. 기본값으로 재시험(정상·멈춤) 통과.
  · 지금 도는 smoke 는 이전 코드(8b695c4: 진행 줄만 있음). 서버의 실시간 로그 파일 `/app/output/vdr/run_smoke_<MMDD_HHMM>.log`, `smoke_a/log_depthlm_<ds>_0.txt` 는 강제 종료돼도 남는다. 다음: 진행 출력 수정본 푸시 → smoke 5 개 세트(새 팩 확인 + DepthLM 1 프로세스 속도) → 파일럿.
- 2026-10-02 12:12–12:34 KST H200 smoke 5 개 세트 (commit bda89b0, 21 분) — **처음으로 끝까지 정상 종료**. 새 팩 5 개 SHA256 통과, DepthLM 세트마다 48 점(파싱 실패 0·잘린 답 0,
  0.41–0.88 s/점, 프로세스 1 개; DDAD 가 가장 느림 = 큰 입력), dense 4 종 × 5 세트 × 3 장(리사이즈 0, 0.1–0.45 s/장), 채점표·검증표 ①②③ 출력.
  수치는 세트마다 첫 1–3 장이라 판단 근거 아님. 다음: 파일럿 `all ibims1 nuscenes` (예상 2.5–3 h, 이전 depthlm 작업의 iBims-1 결과가 track_a 에 남아 있으면 더 짧음).
- 2026-10-02 13:52–16:52 KST H200 파일럿 `all ibims1 nuscenes` (commit bda89b0, 3 h) — 정상 종료. 공통 점 iBims-1 9,750 (제외 250), nuScenes 9,816 (제외 184) = 기하 예상과 같음. 파싱 실패·잘린 답 0.
  · DepthLM 속도: 프로세스 2 개, iBims-1 0.80 s/점·프로세스(61 분), nuScenes 1.0 s/점·프로세스(70 분), GPU 100 %, 80 GB.
  · baseline δ1 vs DepthVLM 표 2: DAv2 0.887/0.168, UniDepthV2 0.941/0.872, Metric3Dv2 iBims-1 0.726 = **6/8 정확히 일치**.
    다름: Metric3Dv2 nuScenes 0.847 (표 0.747), Depth Pro iBims-1 0.829 (0.880)·nuScenes 0.482 (0.389) → 원인 확인 필요 (F-9).
  · DepthLM δ1: iBims-1 원답 0.810 / 변환 0.755, nuScenes 원답 0.823 / 변환 0.735.
    **DepthVLM 표 1(0.754·0.736) = 변환값과 일치, DepthLM 표 1 nuScenes(0.819) = 원답과 일치** (iBims-1 0.870 은 DepthLM 자체 8,192 점·다른 GT).
  · z 변환 확인 ③: 원답/GT 가 광선 계수와 함께 커지지 않음(iBims-1 0.94→0.98, nuScenes 0.98→0.92), 변환값/GT 는 가장자리로 갈수록 작아짐(0.93→0.81, 0.97→0.65)
    → **DepthLM 원답은 유클리드가 아니라 z 에 가깝다** = 규칙 1(유클리드 가정)과 다름 → 사용자 결정 대기 (F-10). 두 값 모두 원자료에 있어 재실행 불필요.
  · ② 공통 점 vs 전체 GT: iBims-1 은 거의 같음. nuScenes AbsRel 은 공통 점이 크게 높음(UniDepthV2 0.520 vs 0.312 등)·CI 넓음 = 소수 근거리 GT 꼬리, Depth Pro nuScenes 전체 GT RMSE 28.4 (공통 점 9.6) = 원거리 이상치. 본 결과 해석 때 다룸.
- 2026-10-02 22:10 파일럿 원자료 분석 (zip johnhong06_874, 공통 점 5 모델):
  · DepthLM 은 먼 거리를 체계적으로 짧게 답한다 — iBims-1 예측/GT 중앙값 0–2 m 0.94 → 6–10 m 0.73 → 10–25 m 0.63 (원답도 0.65), RMSE 0.33 → 6.6 m.
    다른 모델은 0.92–1.00. RMSE 큰 이미지 상위 10 % 가 DepthLM 제곱오차의 71 % (factory_04·03, lectureroom_09 = 넓은 장면). nuScenes 도 60–80 m 에서 0.77.
  · nuScenes dense 모델의 큰 RMSE 는 소수 이미지(상위 10 % 가 UniDepthV2 81 %·Metric3Dv2 89 %). 근거리 GT(≈7 m)를 모든 모델이 ≈33 m 로 보는 점이 23–38 개
    → 모델 문제가 아니라 GT 투영 어긋남 후보(F-2: ego pose 보정 없음, 측·후방 카메라). Metric3Dv2 는 그 점들을 141 m(상한 근처)로 예측해 RMSE 가 커짐.
  · 정렬 확인: 겹침 그림(iBims-1 Metric3Dv2, nuScenes UniDepthV2) 경계 어긋남 없음 → 검증 항목 통과.
- 2026-10-02 17:04–21:54 KST H200 본 실험 `all nyuv2 ddad diode_outdoor` (commit bda89b0, 4 h 50 m) — 정상 종료. 공통 점 NYUv2 9,713 / DDAD 9,920 / DIODE 9,756 (기하 예상과 같음), 파싱 실패 0.
  · DepthLM 속도(프로세스당): NYUv2 0.76, DDAD 1.51, DIODE 0.87 s/점.
  · **이 작업의 표에 iBims-1·nuScenes 가 없음** = 파일럿 결과가 /app/output/vdr/track_a 에 남아 있지 않았다(작업 사이에 비워지거나 옮겨짐). → 최종 5 개 세트 표는 두 zip(874·875)의 parquet 을 로컬에서 합쳐 score.py 로 만든다.
  · DepthLM δ1 재현: DDAD 원답 0.680 (DepthLM 표 1 0.670) / 변환 0.651 (DepthVLM 표 1 0.654) — iBims-1·nuScenes 와 같은 패턴.
    NYUv2 원답 0.816 (DepthLM 0.799) / 변환 0.673 — DepthVLM 표 1 0.866 과는 어느 쪽도 안 맞음(원인 미확인, F-11).
  · z 변환 확인 ③ 은 데이터셋마다 다름: DDAD·nuScenes·iBims-1 은 원답/GT 가 광선 계수와 거의 무관(z 에 가까움), NYUv2·DIODE 는 원답/GT 가 광선 계수와 함께 커지고 변환값/GT 가 평평(유클리드에 가까움).
    DepthLM 공식 큐레이션(D-6: 주행 세트는 z 라벨, NYU 등은 유클리드)과 맞는 방향 → F-10 결정에 반영.
- 2026-10-02 23:xx **정정 — 앞선 'GT 이상 점이 nuScenes dense RMSE 를 키운다'는 해석은 틀렸다**: 다섯 모델 모두 GT 의 2.5 배 넘게 예측한 점은 17 / 9,816 개이고,
  빼도 RMSE 변화는 0.01–0.03 m. 이미지로 보면 자기 차 범퍼 경계(GT 0.21 m), 가는 표지판(GT 7.2 m, 모델은 뒤 벽 32–41 m), 기둥 옆 잔디(GT 2.0 m, 투영 어긋남 의심) — 섞여 있다.
  nuScenes dense 모델의 큰 RMSE 는 각 모델 자신의 큰 오차(꼬리) 때문. DIODE 'GT 품질' 도 근거 없음: Metric3Dv2 가 δ1 0.85 를 내므로 GT 는 쓸 만하고,
  나머지 모델의 낮은 δ1 은 모델별 배율 편향(DepthLM 약 0.70 배, UniDepthV2 AbsRel 1.25 = 과대) 쪽이 유력 — 본 실험 zip 으로 확인.
- **F-9 baseline δ1 이 DepthVLM 표 2 와 다른 세 칸 — 원인 확인 (2026-10-02, 로컬, 스크립트·결과 scratchpad/baseline_check)**:
  · DepthVLM 저장소에 pure vision baseline 추론 코드 없음, 논문에도 설정(초점·해상도·체크포인트) 기재 없음 (README 130–133 은 표 그림뿐).
  · **Depth Pro = 논문은 GT 초점(f_px = fx)을 넣었다 (신뢰도 높음)**: iBims-1 f=None 0.8285 → GT 초점 **0.8800 = 표 2 0.880**. nuScenes 앞 300 장 0.455 → 0.371 (같은 비율이면 전체 ≈ 0.393, 표 0.389).
    우리 설정(f_px=None)은 공식 CLI 이고 설계표의 'GT intrinsics 미사용' 그대로 → 유지. 논문 수치 인용 시 '논문은 GT 초점 사용' 각주.
  · **Metric3Dv2 nuScenes (0.847 vs 0.747) — 원인 못 찾음**: canonical 입력(0.829)·DepthVLM 식 GT/좌표 읽기(변화 2/3,000 점)·정면 fx 공용(0.686)·배율 누락(0.096) 모두 아님.
    예측 ×0.9 이면 0.754 → 논문 쪽 예측이 약 10 % 짧았던 셈. iBims-1 은 정확히 맞으므로 체크포인트 차이는 아닌 것으로 추정(낮은 신뢰). 우리 설정은 hub 공식 경로 그대로 → 유지, 표에 '논문과 다름, 원인 미확인' 기록.
- 2026-10-03 보조 집계를 파일럿 원자료로 시험 (eval/breakdown.py, prep/boundary_labels.py → bench/boundary_{ibims1,nyuv2,diode_outdoor}.parquet):
  · iBims-1 원거리(4 m–, 2,224 점): DepthLM RMSE 2.69 vs 다른 모델 0.90–1.44, δ1 0.545 vs 0.82–0.94. 근거리는 DepthLM 0.33 (UniDepthV2 0.29, Metric3Dv2 0.44) 로 대등.
  · iBims-1 경계(391 점) vs 내부: 모든 모델이 경계에서 나빠짐. δ1 하락 DepthLM 0.772 → 0.606 (−0.17), DAv2 −0.15, UniDepthV2 −0.10, Depth Pro −0.10, Metric3Dv2 −0.05 (경계 점이 적어 CI 넓음).
  · SILog iBims-1: DepthLM 16.1 vs 다른 모델 5.7–6.4 → DepthLM 오차는 전체 배율만의 문제가 아니라 거리에 따라 다르게 줄이는(압축) 구조 오차. nuScenes 는 19.6–26.2 로 비슷.
- 2026-10-03 **F-12 실외 zero-shot 추가 후보 조사** (논문 학습 목록 대조): 6 모델 모두 zero-shot = **KITTI**, **ETH3D outdoor**. Cityscapes(Metric3Dv2 학습)·Argoverse2·Waymo(VLM 학습)·vKITTI2 탈락, Make3D 는 GT 사양 미확인.
  KITTI 주의: VKITTI2(KITTI 장면 합성)를 DAv2 실외·Depth Pro·Metric3Dv2 가 학습 → 각주. 로컬: ~/data/kitti/depth_selection/val_selection_cropped (GT 1,000 장, 누적 LiDAR semi-dense, 위쪽 GT 없음, 관례 cap 80 m).
  ETH3D outdoor 235 장(로컬)은 F-1 정렬 문제 미해결. 우리 체크포인트(Metric3D vit_large_800k, UniDepthV2 vitl14)는 KITTI 파인튜닝판이 아닌 일반판. Depth Pro 공개 가중치 학습 데이터는 미공개.
- 2026-10-03 00:10 **Track B 코드·환경 최종 점검 (로컬)**:
  · 환경 `envs/depthvlm.txt` (torch 2.7.1 cu128, transformers 5.2.0 = 체크포인트 config 버전, qwen-vl-utils 0.0.14): 공식 model/ 코드 import·가중치 로딩 정상. run.sh mkenv 가 depthvlm 에도 flash-attn 휠을 깐다.
  · `eval/dense_full.py` (DepthVLM 어댑터 = 공식 eval.py 경로 + dense 4 종은 Track A 어댑터 재사용, 이미지별 통계·d1_canon) → `eval/score_dense.py` (표·도메인 평균·유클리드 부록·보조 집계·공식 방식 검증표).
  · **검증: DepthVLM iBims-1 100 장 공식 방식 δ1 0.910 = README dense 표 0.910**. pooled RMSE 0.643, AbsRel 0.100, δ1 0.910. NuScenes 는 로컬 GPU 가 다른 학습(17 GB)과 공유돼 OOM → H200 에서 확인.
  · run.sh `trackb` 종단 시험 (ibims1·nyuv2, LIMIT=2): 팩 풀기 → 모델 5 종 추론 → stats → 채점표 → zip 정상. 로컬은 시스템 파이썬에 uv 를 못 깔아(PEP 668) 기존 ~/venv/<모델> 을 WORK/envs 에 연결해 시험 — H200 은 uv 대체가 이미 정상(F-6).
  · 겹침 그림(DepthVLM iBims-1) 경계 정렬 정상. pyflakes 깨끗(breakdown 의 안 쓰는 변수 제거), shellcheck 경고는 의도된 SC2046 1 건.
  · H200 순서: `bash run.sh trackb ibims1 nuscenes LIMIT=3` (스모크) → `bash run.sh trackb ibims1 nyuv2 ddad nuscenes diode_outdoor` (DepthVLM NuScenes 검증 포함).
- 2026-10-03 H200 Track B 스모크 `trackb ibims1 nuscenes LIMIT=3` (commit a73343c, 16 분) — 정상 종료. DepthVLM attention = flash_attention_2 (공식), 0.4–0.8 s/장.
  dense 4 종 값은 Track A smoke 의 전체 GT 통계와 같음(예: DAv2 nuScenes RMSE 10.582) = 두 경로 일치. 검증표 값은 3 장이라 판단 근거 아님.
  **발견: 경계/내부 표가 비었다** — 데이터 팩에는 추론에 쓰는 파일만 넣어 iBims-1 공식 경계 지도(edges/)가 없음. → `prep/ibims1_edges.py`: trackb 에서 공식 배포처(TUM dataserv)
  ibims1_core_raw.zip 을 받아 공식 sha512 일치 시 edges 만 풀기(실패하면 경고만, iBims-1 경계 집계만 빠짐). 로컬 시험 15 초, 86 장, 재실행 시 건너뜀. Track A 보조 집계는 저장소의 bench/boundary_*.parquet 를 써서 영향 없음.
- 2026-10-03 H200 Track B 본 실행 1 차 (commit e508ca9) — **실패**: 데이터 팩 5 개는 정상, 공식 저장소 받기에서 Metric3D 압축본(GitHub codeload)이 읽기 시간 초과(TimeoutError) → fetch_ext 가 의도대로 멈춤.
  일시적 네트워크 문제. 수정: fetch_ext.sh 가 30 초 쉬고 3 번까지 다시 받기, 읽기 제한 120 → 300 초. 로컬 시험: 정상 받기 4 개 통과, 없는 커밋은 3 번 시도 뒤 종료 코드 1.
- 2026-10-03 11:07–12:01 KST H200 **Track B 본 실행** `trackb` 5 개 세트 (commit 6d116a6, 54 분) — 정상 종료. iBims-1 경계 지도 86 장 공식 배포본 sha512 일치, DepthVLM flash_attention_2.
  · 검증(공식 dense 방식 vs README dense 표): DepthVLM iBims-1 0.910 / nuScenes 0.838 = **일치**, UniDepthV2 0.943 / 0.869 (0.941 / 0.868), Metric3Dv2 0.730 / 0.844 (0.724 / 0.843) = 일치,
    Depth Pro 0.829 / 0.478 (0.879 / 0.379) = F-9 와 같은 원인(논문은 GT 초점). Metric3Dv2 nuScenes 는 dense 로는 맞는데 Track A sparse 표 2 와는 0.10 차 → 표 2 sparse 쪽 설정 차이로 보임(F-9 보강).
  · dense 4 종 수치는 Track A 의 전체 GT 통계(densestat)와 같음 (예: DAv2 iBims-1 0.579 / 0.887).
  · 결과 요지: 실내 DepthVLM 평균 RMSE 0.547 · δ1 0.916 으로 1 위 (NYUv2 RMSE 0.453 vs 0.88–0.98). iBims-1 은 UniDepthV2 가 1 위(0.487 / 0.941), DepthVLM 은 원거리(1.24 vs 0.88)·경계(δ1 −0.20)·SILog(11.1 vs 5.9–7.3)에서 약함.
    실외 DDAD·nuScenes 는 DepthVLM RMSE 1 위지만 둘 다 학습 데이터(†). zero-shot DIODE 는 Metric3Dv2 압도(3.84 / 0.859), DepthVLM 7.58 / 0.402. Depth Pro 는 극단 예측으로 RMSE 폭증(DIODE 88.8, DDAD 32.9).
- 2026-10-03 README 결과 표 재구성 (사용자: 가독성): 공식 저장소 README 양식 참고(UniDepth·DepthVLM = 지표 하나에 데이터셋을 열로, Metric3D = 데이터셋별 지표 묶음, 1 위 굵게).
  트랙마다 ① 한눈에 보기(실내·실외 평균 + 평균 순위 = 데이터셋×지표 순위 평균) ② 지표별 상세(RMSE·AbsRel·δ1 표 각각, 데이터셋 + 도메인 평균 열, 1 위 굵게·2 위 밑줄) ③ 보조 지표(Track B; Track A 는 zip 후).
  값은 두 H200 로그의 점추정. Track A DepthLM DDAD 는 '답 그대로' 규칙이라 원자료(zip) 필요 → '대기'. 표는 scratchpad 생성 스크립트로 만들어 순위·굵게 표시를 손으로 하지 않음.
- 2026-10-04 iBims-1 DepthLM 정의 정정 (D-17 정정): 사용자 질문 'DepthLM 논문 iBims-1 δ1 과 왜 다른가' → 공식 예제와 비교.
  처음 답한 '샘플링 차이(먼 점 비중)' 가설은 거리 분포 비교로 기각. 대신 공식 예제 라벨이 z 임을 확인 → score.py Z_SETS 에 ibims1, CLAUDE.md 규칙 1·PROTOCOL 4.1·README 반영.
  파일럿 zip(874) 원자료로 score.py 재계산 1.261 / 0.141 / 0.810 확인. README Track A 표는 같은 생성 스크립트로 다시 만듦(iBims-1·실내 평균·순위만 바뀜).
- 2026-10-04 본 실험 결과 수신·최종 표 (zip 875 = Track A `all nyuv2 ddad diode_outdoor`, 878 = Track B `trackb`; 874 = 파일럿):
  · 원자료 완비: 875 = DepthLM 3 세트 × 2 조각 + dense 4 종 × 3 세트 + densestat, 878 = 이미지별 통계 25 개(5 모델 × 5 세트) + 겹침 그림 125 장.
  · Track A 최종 = 874 + 875 parquet 을 로컬에서 합쳐 score.py (official·converted·raw), breakdown.py, checks.py → `tables/` (저장소에 포함, CI 포함).
    DepthLM DDAD (공식 = 답 그대로 z): RMSE 13.69 / AbsRel 0.253 / δ1 0.680 (변환 시 δ1 0.651). DDAD RMSE 3 위 (Metric3Dv2† 8.86, UniDepthV2 9.63).
    실외 평균: DepthLM RMSE 10.23·AbsRel 0.351 로 2 위, δ1 0.586 은 4 위, 평균 순위 2.6 (2 위). 단 실외 강세의 큰 몫은 nuScenes(†, RMSE 1 위). zero-shot 실외(DDAD·DIODE)는 둘 다 RMSE 3 위.
    보조: DepthLM SILog 실내 15.7 (다른 모델 8.7–9.5), 원거리 RMSE 실내 1.88 (1.09–1.41), 경계 δ1 0.589 (0.726–0.781) — 모두 꼴찌. 실외 log-RMSE 0.496 은 3 위.
  · Track B: 878 통계로 로컬 score_dense.py 재계산 = 서버 표와 같음. README 값은 반올림 마지막 자리만 바뀜(예전 표는 로그의 반올림 값으로 만들었음).
  · README 표 생성 스크립트를 저장소로 옮김 (`eval/readme_tables.py`, csv 만 읽음). scratchpad 의 이전 생성기가 지워져 다시 씀. 순위는 표시 자릿수에서 같으면 같은 순위(표에서 확인 가능하게).
    Track B 로 회귀 확인: 이전 README 와 순위·굵게·밑줄 같음.
  · checks.py 의 DepthLM 원답 라벨 'raw (Euclidean answer)' → 'raw (answer as is)' (D-17 이후 세트마다 정의가 달라서).
  · 겹침 그림: 정렬 이상 없음. DepthVLM 예측 맵은 다른 모델보다 흐리고 격자 무늬가 있고, DIODE 에서 먼 건물을 가깝게 본다(δ1 0.40 과 맞는 방향).
- 2026-10-04 **F-13 DepthLM 실내 오차의 구성** (원자료 874·875, 공통 점, 공식 정의; `eval/pair_consistency.py` + 일회성 분석):
  · 배율: NYUv2 는 1 m 이상 전 구간 약 0.85 배(짧음), 압축 기울기는 다른 모델과 같음(0.90). iBims-1 은 평균 배율 0.97 로 정상, 대신 멀수록 짧음(6–10 m 0.79, 10 m– 0.65; 기울기 0.80 vs 0.95–0.99).
  · 배율·압축(이미지별 log 선형) 보정 뒤 잔차: iBims-1 중앙 4.5 % vs 1.3–1.9 %, 20 % 넘게 틀린 점 10.2 % vs 1.6–1.9 % → 일부 이상치가 아니라 분포 전체가 넓다.
    NYUv2 는 이 잔차가 다른 모델의 1.2–1.4 배로 작다 → NYUv2 약점은 주로 배율.
  · 점 쌍(경계 제외, GT 깊이 차 < 10 %): 20 px 안 |log 오차 차| 1.2 % (다른 모델 0.2–0.3), 160 px 넘으면 5.0 % (1.3–2.1). 가까운 점끼리는 일관되고 멀수록 벌어짐
    → '점마다 독립으로 답해 흩어진다' 가설 기각. 이미지의 영역마다 다른 배율로 답한다(영역 간 3D 배치 불일치). 앞뒤 순서는 대체로 맞음(Spearman 0.92 vs 0.985–0.99).
  · 화면 위치(광선 계수·세로 위치)로 설명되는 잔차 분산 < 3 % → 위치 규칙이 아니라 장면 내용에 따른 것. 답 양자화: 끝자리 8 이 57 %(1.38·1.88·2.38…), 상위 10 값이 28 %, 그러나 인접 값 간격 0.7 % 라 오차 원인은 아님.
  · 다음 후보: 어떤 영역이 틀리는지(벽·바닥 같은 무늬 없는 면 vs 물체) — 질감 지표 또는 오차 큰 이미지 겹침 그림.
- 2026-10-04 **F-14 DepthLM 은 어느 영역을 틀리나 — 특정 영역이 아니라 '면을 연속으로 못 그림'** (iBims-1 공식 평면 마스크, NYUv2 GT 법선; `eval/region_errors.py`, `eval/error_anatomy.py`):
  · 영역·질감: 바닥·벽·책상·기타, 매끈·질감 많음 모두 보정 잔차가 다른 모델의 2.5–3 배로 고르게 나쁨. 특정 영역 편향 없음 (책상이 조금 짧은 건 모든 모델 공통).
  · 평면 분해: DepthLM 은 평면 안 흩어짐(6.7) ≈ 평면끼리(6.5) — 오차가 면 경계를 따르지 않음. dense 모델은 안(1.3–1.8) < 끼리(2.3–3.0) = 면 단위로 틀림.
    → F-13 의 '영역마다 다른 배율' 해석은 부정확: 면 안에서 끊긴다. (가까운 점끼리 비슷한 건 같은 답을 재사용하기 때문)
  · 평면마다 따로 맞춘 뒤 잔차: DepthLM rms 6.0 % (중앙 1.4 %) vs 0.4–0.7 % (9–14 배). 평면 점의 4 % 가 10 % 넘게 튐 (dense 0 %). 평면 안 기울기 중앙 0.90 (IQR 0.77–1.03) vs 0.97–1.02.
  · 두 모양: (a) 계단 — 이미지당 98 점에 서로 다른 답 36 개(dense 74–78 %), 같은 답을 받은 점의 GT 폭 6.9 % vs 1.1–1.5 %, 보정 잔차의 23 %(NYUv2 10 %).
    예 corridor_01 벽 GT 1.6→6.7 m 매끈 ↔ 답 1.38(×6)·1.88·2.00·2.08·2.38·2.88. (b) 튐 — livingroom_14 바닥 GT 1.49–2.03 m 에 '2.00' 이 21 점 중 9 점.
    튐은 평면 가장자리에 몰리지 않고(≤10 px 2.9 %, 안쪽 4.0–4.4 %) 길게 49 / 짧게 48 → 옆 물체 깊이를 읽는 것(화살표 위치 모호) 아님. 일부 이미지에 몰림.
    상위 20 값(1.38·1.88·2.38…)으로 답한 점이 더 틀리지는 않음(4.6 vs 4.5) → 특정 숫자가 아니라 '한 이미지 안 재사용'이 문제.
  · RMSE 분해 (iBims-1): 1.261 → 이미지별 배율 1.016 → +압축 0.831, 다른 모델 → 0.31–0.32. MSE 로 배율·압축 약 57 %, 나머지(비일관성) 43 %; 완벽히 보정해도 꼴찌.
    NYUv2: 0.831 → 0.772 → 0.513 vs 0.45–0.47 → 보정하면 거의 비슷, NYUv2 약점은 배율.
  · 가설(미확인): 숫자를 텍스트로 내며 익숙한 값(끝자리 8 이 57 %, 2.00, 실외 10.0·23.0)을 고르는 성향 → 연속 측정이 아니라 값 선택에 가까움. 확인하려면 GPU(답 토큰 확률, 화살표를 몇 px 옮긴 재질의).
- 2026-10-05 **F-11 해결 — NYUv2 DepthLM δ1 이 논문과 다른 이유 = DepthLM 공식 NYU GT 가 실제의 0.8 배**:
  · `DepthLM_Official/utils/curate_NYU.py` 84 행: SUN RGB-D `kv1/NYUdata/*/depth/*.png` 를 `/ 10000.0` 으로 미터 변환 (`curate_sunRGBD.py` 도 같음).
    SUN RGB-D 공식 툴박스 `read3dPoints.m` 은 bitshift(−3) 뒤 /1000 = png/8000 (png 끝 3 비트 100 % 0 확인). → DepthLM 라벨 = 실제 깊이 × 0.8.
  · 우리 GT 는 실제 미터: 우리 NYUv2 depth png = SUN RGB-D `depth_bfx` 와 654/654 동일, `depth/`(원본)와도 값 있는 곳에서 비 1.0000, scale 8000.
    dense 모델 예측/GT 0.99–1.18 (GT×0.8 이면 δ1 0.12–0.49 로 무너짐) → /8000 이 맞다.
  · 우리 점에서 DepthLM 답을 GT×0.8 과 비교하면 δ1 **0.865** = DepthVLM 표 1 의 DepthLM NYUv2 **0.866** (DepthVLM 이 DepthLM 정리 코드의 NYU 라벨을 쓴 것으로 보임, 추정).
    DepthLM 논문 0.799 와는 0.07 차이 — 평가 세트가 다름(SUN RGB-D 의 NYU 1,449 장 전체·원본 깊이·이미지당 100 점) 으로 추정, 미확인.
  · 원답 그대로 vs GT z 0.816 이 0.799 에 가까웠던 것은 우연: 유클리드 답이 z 보다 광선 계수만큼 큰 것이 DepthLM 의 약 15 % 짧은 답을 상쇄했을 뿐.
  · 원본 NYUv2 와 직접 대조 (~/data/nyuv2/nyu_depth_v2_labeled.mat, 미터 실수값): SUN RGB-D NYU 이미지는 원본 640×480 을 561×427 로 자른 것(위치 y≈45–46, x≈43–45).
    NYU0001·0329·1000 에서 png / 원본 미터 = 7,970–7,983 (depth_bfx ↔ depths, depth/ ↔ rawDepths 모두) → /8000 이 원본 미터, /10000 은 0.8 배. 확정.
  · 우리 결과(0.673)는 바꾸지 않는다: 실제 미터 GT 가 맞고, DepthLM 은 NYU 를 학습하지 않았으므로 약 0.85 배로 짧게 답하는 것이 모델 자체의 행동.
- 2026-10-07 **F-15 DepthVLM-4B 의 약점 (Track B, 878 이미지별 통계 + 로컬 GT 장면 특성 + 겹침 그림; 일회성 분석, GPU 사용 없음)**:
  · 배율은 5 모델 중 가장 정확: DIODE 를 뺀 4 세트에서 exp(평균 log 오차) 1.000–1.002, 이미지 간 배율 흔들림(sd) 0.025–0.068 로 1 위. log-MSE 의 76–99 % 가 모양(배율 무관) 오차.
  · δ1 은 어디서도 1 위가 아님: 5 세트 × (전체·근·중·원·경계·내부) 칸 모두 2–4 위, DDAD·nuScenes(학습†) 카메라 6 대 모두 2–3 위. 이미지별 δ1 1:1 승률 vs UniDepthV2: iBims-1 10 %·DDAD 21 %·nuScenes 19 %.
    반대로 RMSE 는 1 위가 많고 AbsRel p99 꼬리가 가장 짧음 → 크게 망가지는 이미지는 적지만 픽셀 정밀도가 낮다.
  · 모양 오차는 GT 가 조밀하고 날카로울 때만 드러남: SILog iBims-1 11.1 (다른 모델 5.9–7.3, 이미지 78 % 에서 꼴찌, 장면 종류 10 개·깊이 3 등분 모두 꼴찌), DIODE 38.5 (5 위).
    채움 GT(NYUv2 depth_bfx)·희소 LiDAR(DDAD·nuScenes) 에서는 SILog 1 위. 겹침 그림: 예측 맵이 흐리고 격자 무늬, 의자 다리·창틀·나무 윤곽이 사라짐(DDAD 나무 위쪽은 LiDAR 점이 없어 채점 안 됨).
  · 경계: δ1 경계 − 내부 = −0.20 (iBims-1 다른 모델 −0.05 – −0.14 → 가장 큼, NYUv2 −0.17 – −0.20 → 공동 최대). iBims-1 경계 RMSE·δ1 4 위.
  · 거리: 모든 세트에서 가까운 곳 5–9 % 멀게(근거리 배율 1.05–1.09), 먼 곳 7–17 % 가깝게(0.83–0.93) — 압축. 실내 원거리 δ1 4 위(iBims-1 0.868 vs 0.939, NYUv2 0.847 vs 0.900).
    압축 정도(원/근 배율 비)는 NYUv2 0.868(다른 모델 0.896–0.955)·DIODE 0.524(UniDepthV2·Metric3Dv2 0.83–0.85)에서 가장 심함.
  · DIODE(실외 zero-shot): 장면이 깊을수록 배율이 무너짐 — 이미지 배율 ~ 장면 중앙 깊이 기울기 −0.25 (r −0.61), 중앙 깊이 16 m 이상 3 분위 배율 0.716·δ1 0.174. DDAD·nuScenes 깊은 장면은 0.995–0.999 → 학습 도메인 밖에서만.
    scene_00023 (scan 198·199) 배율 0.70, 모든 지표 4–5 위. δ1 < 0.5 인 이미지 68 %.
  · 강점(대조): nuScenes 밤(현지 19 시 이후 108 장) δ1 0.714 로 1 위(다른 모델 최고 0.687), NYUv2 장면 종류 14 개 전부 RMSE 1 위.
  · (추가) 흐린데도 점수가 좋은 이유: ① 배율 — 이미지별 배율을 GT 로 맞춰 주면 iBims-1 log-RMSE DepthVLM 0.121 vs 0.070–0.083 (꼴찌), 실제는 0.138 로 2 위 = 순위는 배율 덕.
    ② 디테일 픽셀이 적음 — 경계(3 px) 픽셀 iBims-1 4.4 %·NYUv2 6.5 %, DepthVLM log 제곱오차 중 경계 몫 13 %·21 %. ③ GT 가 디테일을 못 봄 — GT 덮는 픽셀 DDAD 1.9 %·nuScenes 0.2 %(이미지 위 1/3 은 ≈ 0 %), NYUv2 는 채움 GT.
    NYUv2·DDAD·nuScenes 에서는 배율을 맞춘 뒤에도 DepthVLM 이 1 위(0.151 / 0.232 / 0.256). 해석(미확인): 흐림 = 경계에서 중간값으로 헤지 → RMSE 엔 유리, δ1 엔 불리.
  · (추가) 이미지별 배율 b (평균 log 비, a = 1) 를 GT 로 맞춘 뒤 순위 — DepthLM (Track A 공통 점, 공식 정의, 선형 지표는 자르지 않음):
    RMSE iBims-1 5→5 · NYUv2 2→5 · DDAD 3→5 · nuScenes† 1→1 · DIODE 3→4, δ1 4→5 · 5→5 · 3→4 · 3→3 · 4→5. iBims-1 log-RMSE 0.205 vs dense 0.070–0.076.
    (F-14 는 중앙값 배율이라 NYUv2 값이 다름 0.772 vs 0.640, 결론 같음.) DepthVLM (Track B, log-RMSE 만 가능): iBims-1 2→5, DIODE 2→4, NYUv2·DDAD†·nuScenes† 1→1.
  · (추가) NYUv2 에서 SILog 1 위인 이유 = GT 의 채운 픽셀: labeled.mat rawDepths 로 이미지별 '원측정 없음 → 채움' 비율(평균 13 %, 최대 61 %; 오프셋 이미지별 y 43–48·x 41–47, 원측정 픽셀 차 중앙 0.5 %, 652/654 장).
    채움 ≤ 7 % 이미지 SILog DepthVLM 9.1 **4 위** (UniDepthV2·DepthPro 7.8), 7–15 % 3 위, 15 % 이상 18.4 **1 위** (다른 모델 22.0–26.9). 채움 비율 ~ (DepthVLM − 다른 모델 중앙) Spearman −0.41.
    iBims-1 → NYUv2 SILog: 다른 모델 ×2.2–2.5, DepthVLM ×1.18. 남은 교란: 채움 많은 장면 = Kinect 가 못 재는 곳(먼 벽·유리·검은 면) → 원측정 픽셀만으로 재채점해야 확정(예측 맵 필요).
- 2026-10-07 **F-16 VLM 약점 분석 (D-20)** — zero-shot 3 세트, DepthLM·DepthVLM vs pure vision 4 종. 보고서 `docs/VLM_WEAKNESS.md`, 그림 `results_vlm_weakness/`.
  · 실행: 로컬 맵 7 조건(5 모델 + UniDepthV2+K·DepthPro+f) × 3 세트 1,200 장, 약 30 분(DepthVLM 0.2 s/장). 전체 지표가 Track B 표와 반올림까지 같고(예: DepthVLM iBims-1 0.643/0.100/0.910 vs 0.641/0.099/0.911), 공통 점 pure vision 값은 H200 Track A 와 ±0.3 %.
    DepthPro+f 첫 실행은 f_px 를 float 로 넘겨 실패(공식 infer 는 `.squeeze()` 를 부름) → 공식 CLI 처럼 numpy 실수로 고쳐 재실행.
  · 프로토콜 재점검(조사 에이전트 2 개 + 직접 확인): 지표 식·마스크·cap·DepthVLM/DepthLM/pure vision 추론 경로가 공식과 같음. 점 샘플러는 균일(경계 점 비율 = 픽셀 비율).
    고친 문서: PROTOCOL 4.6·9 절 B = 2,000 → 실제 1,000, UniDepthV2 '카메라 없음' 경로를 scripts/demo.py → README 예시(demo 는 GT K 를 넣음), D-1 의 '유클리드 GT' 정정 표시.
    부록 대조: GT intrinsics 를 주면 UniDepthV2 iBims-1 AbsRel 0.090 → 0.077·δ1 0.943(DepthVLM 0.100·0.910 추월), DepthPro NYUv2 배율 치우침 +9.2 → +0.9 %. 모양 오차는 그대로 → VLM 배율 강점 일부는 GT 초점 입력 덕.
    Depth Pro 예측 미절단(0.08 % 픽셀 10⁴ m)이 실외 RMSE 를 지배(DIODE 80 m 로 자르면 88.8 → 9.25). DepthVLM 학습 라벨 일부 유클리드(Argoverse2·Waymo)지만 출력에 징후 없음.
    논문 쪽: DepthVLM 공식 sparse 평가는 맵 크기와 canonical 좌표가 어긋남(논문 수치에만 영향), DepthLM iBims 예제 점 254 개가 투명 영역.
  · 공통 약점 (CI = 이미지 bootstrap, B = 1,000):
    ① 깊이 차 10–25 % 쌍 앞뒤 오답률 − PV 중앙값: iBims-1 LM +13.5 %p [11.1, 16.3]·VLM +9.9 [8.2, 11.6], NYUv2 원측정 +5.7 [4.7, 6.7]·+6.6 [5.7, 7.5], DIODE +10.6 [9.1, 12.2]·+8.0 [6.8, 9.0].
    ② 평면(iBims-1 공식 126 평면, 공통 점) ε_plan LM 2.87 cm·VLM 5.05 cm vs PV 0.33–0.59 (비 6.6 [5.3, 7.9]·10.7 [9.6, 12.3]), ε_orie 6.4°·9.1° vs 1.5–2.3°. 픽셀(244 평면) DepthVLM 6.53 cm·13.1° vs 0.50–1.37·1.9–2.8.
       벽에 붙은 평평한 물체: |물체 − 둘레 벽| LM 5.1 %·VLM 6.6 % vs PV 1.1–2.1 % (방향은 일정하지 않음).
    ③ 원거리 모양 오차 비: iBims-1 LM 3.45 [2.39, 4.66]·VLM 1.59 [1.41, 1.84], DIODE 1.65 [1.51, 1.84]·1.16 [1.08, 1.25], NYUv2 원측정 1.42 [1.22, 1.69]·2.06 [1.60, 2.67]. 압축 기울기 a: iBims-1 0.80·0.90 vs 0.95–0.99, DIODE 0.60·0.64 vs 0.69–0.87.
    ④ DIODE 깊은 장면 배율(장면 깊이 기울기): LM −0.41·VLM −0.25 vs Metric3Dv2 +0.01 (같은 GT 초점 입력). Depth Pro −0.25·DAv2 −0.49 도 같은 방향 → VLM 만의 것은 아님.
    이미지 단위: 두 VLM 이 PV 4 종 전부보다 모양 오차가 큰 이미지 iBims-1 87 %, NYUv2(벤치 GT) 34 %, DIODE 29 %.
  · DepthVLM 만: 경계 계단 폭 15.0 px vs 4.2–5.1(GT 1.8), DBE ε_comp 7.63 vs 2.08–3.24(GT 자기 점검 1.08), 토큰 주기(17.3/16.5/28.4 px) 스펙트럼 봉우리 2.1–2.5 배 vs 1.0–1.3
    (원인 후보: dpt_depth_head.py:65–85 kernel = stride 전치 합성곱). PV 대비로는 경계(1.33)보다 면 안쪽(1.68)이 나쁨 → 가장 큰 약점은 면.
  · DepthLM 만: 답 재사용 계단(corridor_01 99 점에 56 개, F-14), 모양 오차 20.5 % vs 7.0–7.6.
- 2026-10-07 **F-17 NYUv2 벤치 GT 결함 — 8.19 m 이상을 표현 못 하고 더 작은 값으로 채워짐 (결론 바뀜)**: 벤치 GT = SUN RGB-D depth_bfx png ÷ 8,000 → 16 비트 한계 8.19 m. 654 장 최댓값 7.995 m(png 63,960).
  손상 방식: 원측정 ≥ 8.25 m 픽셀 59.8 만 개 중 png 값이 '원측정×8000 mod 65536'(단순 넘침)과 맞는 것은 4.2 %뿐 → 넘침이 아니라 더 작은 값으로 채워진 것(SUN RGB-D 채우기 과정 추정, 원본 처리 코드 없음). 예 NYU0335 원측정 8.67 m → png 45,416 = 5.68 m.
  Kinect 원측정(labeled.mat rawDepths, 이미지마다 자르기 위치 맞춤)이 8–10 m 인 픽셀(실측의 0.51 %, 109 장)에서 벤치 GT 중앙 4.86 m vs 원측정 9.07 m. 실측 픽셀 전체로는 벤치 − 원측정 중앙 −0.006 m 지만 RMSE 0.487 m.
  그 픽셀은 대부분 실제 먼 표면(라벨 없음 47 %·벽 19 %·문 5 %·액자 4 %, 창문 3 %·거울 0.1 %). 예측 중앙: PV 4 종 9.2–10.1 m(원측정 쪽), DepthVLM 4.1 m(벤치 쪽), DepthLM 점 원측정 대비 0.78 배(무너지지 않음).
  벤치 GT < 원측정/2 인 픽셀(55 장): DepthVLM 79 % 가 벤치 GT 에 더 가까움(PV 3–5 %). 공개 학습 설정(configs/train_datasets.conf)에 NYUv2·SUN RGB-D 없음,
  DepthVLM 은 8.19 m 보다 앞인 5–6 m 부터 서서히 꺾임(NYUv2 원측정 10.25 m 에서 0.26 배; iBims-1 레이저는 12 m 까지 0.80 이상) → 손상을 배웠다기보다 Kinect 실내 원거리 압축이 같은 방향의 GT 결함과 겹친 것으로 해석(미확정).
  원측정 GT 재채점(전체 픽셀; **F-18 정정: 10.0 m 포화값을 넣은 값이라 아래 수치 중 RMSE 는 부풀려짐 — DepthVLM 0.476, PV 3 종과 동률**): DepthVLM RMSE 0.453(1 위) → 0.547(5 위), SILog 13.1(1 위) → 11.2(4 위), AbsRel 0.082·δ1 0.942(둘 다 2 위). Metric3Dv2 0.877 → 0.348. PV SILog 14–18 → 8–11.
  공통 점(원측정 있는 8,492 점): DepthVLM 0.374 → 0.551, DepthLM 0.695 → 0.626. → Track A·B NYUv2 행(특히 RMSE·SILog)은 GT 결함의 영향을 받음. 주 표를 바꿀지는 사용자 결정 대기.
  읽는 방법 문제인지 확인: 원측정 ≥ 8.25 m 픽셀에서 bfx png 하위 3 비트가 100 % 0 — SUN RGB-D 툴박스 방식(read3dPoints.m 의 3 비트 회전)으로 읽어도 원측정 ±5 % 안 0.1 % (÷8000 도 0.1 %).
  가까운 곳(0.5–7.5 m)은 두 방식 모두 98.2 % 일치 → 파일 자체에 8.19 m 너머 정보가 없다. 다른 저장소의 NYU 출처: Metric3D = BTS 식 sync_depth png ÷1000(mm), UniDepth = nyuv2.hdf5 depth_scale 1000,
  우리 Metric3Dv2 재현(m3d_nyu.py) = labeled.mat → 모두 SUN RGB-D 사본이 아니라 NYU 공식 데이터 계열. SUN RGB-D 사본을 쓰는 쪽은 DepthVLM 벤치(depth_bfx ÷8000)와 DepthLM(curate_NYU.py, 원측정 png ÷10000).
  DepthVLM utils/datasets.py 는 sunrgbd 상한 8.0 m, nyuv2 상한 10.0 m — 같은 SUN RGB-D 파일에서 꺼낸 NYUv2 에 NYU 표준 상한을 건 것이 어긋난 지점으로 보임.
  원측정 GT 로 보면 NYUv2 도 iBims-1 과 같은 그림(두 VLM 모든 경우 > 1: 벽 1.79·2.13, 벽에 붙은 물체 2.07·2.21, 작은 물체 1.61·1.64). F-15 의 'NYUv2 1 위 = 채운 GT' 해석은 '채운 GT + 8.19 m 이상 결함'으로 보강.
- 2026-10-08 **F-18 NYUv2 원측정 재채점(F-17)의 오류 — 원측정 10.0 m 포화값을 정답으로 넣었다 (수정함)**: rawDepths(SUN 자르기 창 안)에서 정확히 10.0 m 인 픽셀 195,574 개,
  [9.999, 10) 은 35 개, [9.9, 9.999) 5,679 개 → 10.0 은 센서 상한에서 잘린 포화값(실측 아님). `weak_nyu_raw.py` 가 0.005 ≤ GT ≤ 10 (포함)으로 잘라 이 값을 넣었다. BTS 표준은 GT < 10.
  · 영향 (전체 픽셀, pooled RMSE, 포함 → 제외): DepthVLM 0.547 → **0.476** (DepthVLM 제곱오차의 약 25 % 가 이 0.14 % 픽셀), Metric3Dv2 0.348 → 0.339, UniDepthV2 0.480 → 0.459, Depth Pro 0.468 → 0.454, DAv2 0.480 → 0.472.
    DepthVLM 순위는 그대로 5 위지만 UniDepthV2·Depth Pro·DAv2 와의 차이는 +0.004–0.021 m, 짝지은 이미지 bootstrap 95 % CI 가 0 을 포함 → '꼴찌'가 아니라 3 종과 동률, Metric3Dv2 보다만 유의하게 나쁨(+0.136 [0.079, 0.186]).
    AbsRel 0.081·δ1 0.943 은 2 위(3 종보다 유의하게 좋음), 모양 오차 12.0 은 4 위(3 종보다 +2.1–2.8 유의하게 나쁨). 원거리(≥ 4 m)는 RMSE 1.199·δ1 0.849·모양 17.3 모두 4 종 전부보다 유의하게 나쁨.
    공통 점(8,483 점): 벤치 GT → 원측정 DepthVLM 0.364 → 0.489 (1 → 5 위), DepthLM 0.656 → 0.623 (6 위, RMSE·AbsRel·δ1·모양 모두 4 종 전부보다 유의하게 나쁨).
  · 같이 바뀐 것: E10 곡선의 NYUv2 10.25 m 칸(= 포화값만 모인 칸, 'DepthVLM 0.26 배·UniDepthV2 1.28') 사라짐 → 마지막 칸 9.75 m DepthVLM 0.38. 원측정 8–10 m 픽셀 108 장 501,338 개
    (벤치 GT 4.6 m · 원측정 8.8 m · DepthVLM 4.6 m · PV 8.6–9.0 m, 이미지별 중앙값의 픽셀 가중 평균). 벤치 GT 를 따르는 비율 DepthVLM 79 → 69 % (49 장). 원거리 모양 오차 비 DepthVLM 2.06 → 1.85 [1.49, 2.35],
    DepthLM 1.42 그대로. 공통 점 모양 비(LM·VLM): 벽 1.80·2.00, 벽에 붙은 물체 2.09·1.88, 작은 물체 1.51·1.25. 앞뒤 오답률 차는 그대로(+5.7·+6.6 %p).
    `weak_ci.py --nyuraw` 가 이전 ci_common_nyuraw.csv(일회성)를 재현함을 확인(DepthLM ① 1.420 vs 1.417), 기본 실행 결과는 기존 ci_common.csv 와 같음.
  · 입력 틀 대조 (같은 픽셀 = Eigen crop ∩ SUN 창, 원측정 < 10 m, 652 장): DepthVLM SUN 자른 입력(561×427) pooled RMSE 0.474 / 이미지별 0.375 / AbsRel 0.081 / δ1 0.944
    vs NYU 원본 입력(640×480, depthvlm-finetune F-6 예측) 0.405 / 0.350 / 0.088 / 0.938. 예측 비 중앙 1.000 (5–95 % 0.94–1.06). Metric3Dv2 도 0.313 vs 0.282 (전처리도 hub vs 벤치마크로 다름).
    → 입력 틀만으로 RMSE 가 10–15 % 달라진다. 버그는 아니고 프로토콜 차이 — 문헌(BTS 표준) 수치와 비교하려면 원본 640×480 입력이어야 한다.
  · 확인한 것(문제 없음): DepthLM NYU z 변환의 intrinsics 는 SUN 자르기 기준(fx 518.86, cx 284.58, cy 208.74), NYU 는 두 VLM 모두 학습에 없음(D-21), DepthLM 공식 NYU 라벨 0.8 배(F-11)는 논문 수치에만 영향.
- 2026-10-08 **F-19 Metric3Dv2 의 0 근처 예측 (원인 미확인, 공식 경로 그대로)**: 예측 < 0.1 m 이면서 GT 유효한 픽셀 NYUv2 280,588 (0.179 %, 516 장)·DIODE 0.014 %·iBims-1 0.005 %, 다른 모델은 0.
  NYUv2 에서는 아래 10 행 42 %·오른쪽 10 열 48 % — 테두리. 그 자리 RGB 밝기는 정상(검은 띠 아님), GT 중앙 1.7 m. 어댑터는 hub 데모와 줄 단위로 같다(dense_sparse.py 61–73).
  NYU 원본 입력·벤치마크 전처리(depthvlm-finetune)에서는 3,213 픽셀뿐. 영향: Metric3Dv2 의 log 지표만 — NYUv2 원측정 SILog 10.7 → 이 픽셀 빼면 7.8 (원본 입력 6.7), RMSE·AbsRel 은 미미.
  NYUv2 에서 'VLM ÷ PV 중앙값' 모양 비는 Metric3Dv2 가 부풀어 VLM 쪽에 약간 유리(보수적). 공식 설정 변경은 사용자 확인 사항이라 그대로 둔다.
- 2026-10-08 **F-20 모양 오차(SILog)의 구성 — 순서를 지키는 왜곡 ≈ 절반, 순서를 깨는 비일관성 ≈ 절반 (zero-shot 3 세트, 일회성 분석, CPU)**:
  이미지마다 log 예측을 GT 의 ① 직선(압축) ② 단조 증가 함수(isotonic)로 맞춘 잔차. ② 잔차 = 어떤 순서 보존 재매핑으로도 못 없애는 부분(같은 실제 깊이인데 다르게 예측 = 순서 뒤바뀜의 크기).
  전체 픽셀(이미지당 40k 표본), log rms ×100 — 모양 → 직선 뺀 뒤 → 단조 뺀 뒤 (모양 분산 중 순서 보존 몫):
  · iBims-1: DepthVLM 12.1 → 9.9 → 8.4 (52 %) / PV 7.0–8.3 → 6.1–7.1 → 5.0–6.2 (43–54 %). 두 부분 모두 PV 의 약 1.5–1.7 배.
  · NYUv2 원측정: DepthVLM 12.4 → 10.4 → 8.4 (55 %) / PV(Metric3Dv2 제외, F-19 테두리 탓에 11.3) 9.3–10.1 → 7.3–8.5 → 6.1–7.0 (52–57 %).
  · DIODE: DepthVLM 42.9 → 28.4 → 23.7 (69 %) / PV 31.0–43.8 → 25.9–30.1 → 21.8–25.3 (50–75 %) — 순서를 깨는 부분은 PV 와 같음 → 실외 약점은 압축·장면 깊이별 배율(순서 보존 쪽).
  · 공통 점 iBims-1: DepthLM 20.5 → 15.3 → 13.1 (59 %) vs PV 7.0–7.6 → 6.0–6.3 → 4.0–4.4 — DepthLM 은 순서를 깨는 부분이 PV 의 3 배 (답 재사용·튐, F-14).
  해석: 순서 뒤바뀜은 모양 오차의 원인이 아니라 결과의 일부 — 국소 비 오차가 실제 깊이 차보다 클 때만 생긴다 (10–25 % 쌍 오답 DepthVLM 11.8 % vs PV 1.6–2.3 %, 2 배 넘는 쌍 0.4 % = PV 와 같음).
  나머지 절반은 순서를 전혀 바꾸지 않는 거리별 왜곡(압축)이다. 스크립트: scratchpad shape_decomp.py (GT 로 맞추는 분석 전용 분해).
- 2026-10-08 **F-19 원인 실험 — Metric3Dv2 의 0 근처 예측 = 가까운 물체가 이미지 가장자리에 잘린 곳 (가까운 거리 자체를 못 보는 것은 아님)**:
  · 위치: 아래·오른쪽 가장자리에서 안쪽으로 약 8 px 에 걸쳐 줄어드는 띠 (맨 끝 행 19,078 → 7 번째 6,367 → 9 번째 1,083). 그 자리 GT 중앙 1.1–1.4 m(이미지에서 가장 가까운 면:
    NYU0469·0387 오른쪽 끝 문틀·문, NYU0334 아래쪽 책상 모서리). 같은 거리(1 m)라도 이미지 안쪽은 정상으로 예측 → 근거리 일반의 문제 아님.
  · 입력 띠 아님: 원본 NYU 의 흰 무효 테두리(행 ≥ 474, 열 ≥ 633)는 SUN 자르기 창(아래 끝 469–474, 오른쪽 끝 601–607) 밖. 창 테두리 RGB 밝기 정상.
  · 실험 (공식 hub 전처리 그대로, CPU — 깊이 기대값 bin 텐서만 같은 장치로 옮기는 스크립트 내 패치; 조건 ① 이 저장 맵과 픽셀 수까지 같음 4691 vs 4688 등), 3 장의 0.1 m 미만 픽셀 수:
    ① 그대로 4,691 / 3,905 / 3,373  ② 상하좌우 뒤집어 넣고 되돌림 3,455 / 828 / 820 — 띠가 원래 자리(내용)를 따라감, 양은 줄어듦
    ③ 가장자리 반사 16 px 늘림 186 / 0 / 1,423(다른 쪽에 새로 생김)  ④ NYU 원본 640×480 입력 → 같은 창 21 / 303 / 906.
  · 해석: 가까운 면이 화면 끝에서 잘리면 Metric3Dv2 가 그 면이 화면 밖으로 계속 다가온다고 외삽해 끝 몇 px 에서 0 으로 떨어지고, 공식 clamp(0, 300) 로 0 이 된다.
    가장자리 너머 맥락(원본 이미지·반사)을 주면 대부분 사라진다. SUN RGB-D 판은 원본을 Eigen crop 과 거의 같은 창으로 잘라 입력해서 이 띠가 평가 영역 안에 들어온다
    (표준 BTS 평가는 원본을 넣고 Eigen crop 으로 가장자리를 빼서 거의 안 보임 — depthvlm-finetune 3,213 px). 공식 출력이라 값은 그대로, log 지표 각주. 스크립트: scratchpad m3d_edge_test.py.
- 2026-10-08 **주 표·README 의 NYUv2 를 원측정 GT 로 (D-22)**: `eval/nyu_raw_inputs.py` → `~/data/vdr_raw/nyuraw/` (Track A: 875 nyuv2 parquet 의 gt_z 만 원측정으로, 9,970 → 8,684 점;
  경계 = 원측정 GT 이웃 깊이 비; Track B: 로컬 맵 + dense_full.sums, d1_canon 은 878 값). 정렬 오차로 NYU0273·NYU1176 제외 → 652 장.
  · 회귀 확인: 바꾸기 전 874·875·878 원자료로 score.py·breakdown.py·score_dense.py 를 다시 돌려 tables/ 의 csv 3 개와 차이 0.0. 바꾼 뒤 달라진 행은 nyuv2 와 실내 평균뿐.
  · `breakdown.py --nyu_boundary` 추가(기본은 예전과 같음). tables/ track_a·track_a_breakdown·track_b(.csv·.md), track_a_depthlm_converted·raw.md 재생성. track_a_checks.md 는 논문 재현 확인(벤치 GT)이라 그대로.
  · README: `readme_tables.py` 출력으로 표 10 개 교체, 공통 설계에 NYU 정답 문단, 경계 점 NYUv2 643 → 271. PROTOCOL 6 절·10 절 갱신.
  · NYUv2 (pooled RMSE / AbsRel / δ1, 벤치 → 원측정): Track A 공통 점 8,483 — DepthLM 0.831 → 0.623 / 0.213 → 0.172 / 0.673 → 0.682, DAv2 0.908 → 0.462, UniDepthV2 0.980 → 0.444,
    Metric3Dv2 0.807 → 0.304, Depth Pro 0.959 → 0.449 → DepthLM RMSE 2 위 → 5 위. Track B — DepthVLM 0.453 → 0.476 / 0.099 → 0.081 / 0.921 → 0.943 (RMSE 1 위 → 5 위, AbsRel·δ1 2 위),
    PV 4 종 RMSE 0.34–0.47. 실내 평균 Track B: DepthVLM RMSE 0.547 → 0.558 (1 위 → 3 위), AbsRel 0.090·δ1 0.927 은 1 위 유지.

