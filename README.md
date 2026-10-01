# vlm-depth-rmse

Re-measuring DepthLM with **RMSE and AbsRel**, not only δ1. DepthLM (arXiv:2509.25413) reports δ1 alone. δ1 only counts predictions within 25 % of the ground truth, so it can hide the size of errors and their tail. This repository measures RMSE / AbsRel / δ1 for DepthLM-12B and four pure vision models under one zero-shot protocol (no fine-tuning). It uses indoor and outdoor datasets that most compared models have not seen. Scope of this stage (Track A): measurement only. Error analysis comes next and reuses the per-pixel raw data.

## Track A at a glance

| | |
|---|---|
| Evaluation | shared sparse pixels: DepthLM is queried at each pixel; dense models are read at the same pixel of their prediction resized to the GT resolution |
| Models | DepthLM-12B (only the 12B checkpoint is public) · Depth Anything V2 metric (ViT-L) · UniDepthV2 (ViT-L) · Metric3Dv2 (ViT-L) · Depth Pro, each run with its official inference code |
| Datasets | indoor: iBims-1 (100 images), NYUv2 (654) · outdoor: DDAD (1,000), nuScenes (1,000), DIODE Outdoor (446). 10,000 pixels each |
| Pixels | DepthVLM-Bench points (arXiv:2605.15876) at original resolution with their original GT; DIODE Outdoor sampled with the same rule (seed 42) |
| Metrics | RMSE (z-depth, main; Euclidean in an appendix), AbsRel, δ1. Pooled over pixels (main) and per image (auxiliary), 95 % image-level bootstrap CIs, indoor and outdoor reported separately |

| Model | GT intrinsics | Domain info | Trained on (among our sets) |
|---|---|---|---|
| DepthLM-12B | yes (focal-750 normalisation, z conversion) | no | nuScenes (other scenes) |
| DAv2-metric (L) | no | yes: indoor Hypersim model (max 20 m) / outdoor VKITTI model (max 80 m) | — |
| UniDepthV2 (L) | no | no | — |
| Metric3Dv2 (ViT-L) | yes | no (the ViT model has a single setting) | DDAD |
| Depth Pro | no (focal estimated) | no | — |

## Protocol in one paragraph

Depth is z-depth everywhere. DepthLM's answer is a Euclidean distance d and is converted with z = d / sqrt(1 + ((u−cx)/fx)² + ((v−cy)/fy)²), using the original-resolution pixel and intrinsics. The raw answer is kept in `pred_raw`. Valid masks and per-dataset depth caps follow DepthVLM-Bench `sample_points.py` and are identical for all models; DIODE Outdoor uses 0.05–80 m. A pixel is scored only if every compared model returned a prediction there; DepthLM cannot mark pixels within 5 px of the border of its focal-750 image. The number of excluded pixels is reported. Every benchmark GT value was reproduced from the official raw data with the official extraction code: 10,000 / 10,000 points per dataset, max difference 0.00005 m. Full settings: [docs/protocol.md](docs/protocol.md) (Korean).

## Running

Experiments run on an H200 job runner that clones this repository and executes one command. Data packs are delivered to `/app/data` beforehand, and results go to `/app/output`.

```bash
bash run.sh env                            # no data needed: per-model environments, official repos, weights, one demo image each
bash run.sh smoke                          # environments, data, weights: a few pixels/images per model
bash run.sh depthlm ibims1 nuscenes        # DepthLM-12B on every common pixel (pilot datasets)
bash run.sh dense all ibims1 nuscenes      # the four dense models (dav2 | unidepth | metric3d | depthpro | all)
bash run.sh all ibims1 nuscenes            # DepthLM-12B and the four dense models in one job
bash run.sh m3d_nyu                        # check: Metric3Dv2 NYUv2 zero-shot RMS 0.251 on the official 654 images
```

What `run.sh` does:
- builds one conda environment per model (`envs/<model>.txt`; DepthLM uses transformers 4.51.1 as officially tested)
- fetches the official model repositories at pinned commits (`prep/fetch_ext.sh`)
- unpacks the data packs, verifying SHA-256 while streaming
- writes per-pixel parquet files
- prints the result table and the δ1 reproduction table, then zips everything

Raw data columns: `dataset, image_id, u, v, fx, fy, cx, cy, gt_z, pred, pred_raw, model`. All numbers come from these files via `eval/score.py` and `eval/checks.py`.

## Layout

```
run.sh                     H200 entry point
h200/unpack.py             finds data-pack parts under /app/data (plain or inside a Drive zip), verifies SHA-256 while untarring
eval/depthlm_sparse.py     DepthLM-12B: official preprocessing, prompt, greedy decoding, answer parsing; z conversion
eval/dense_sparse.py       dense models through their official inference code; common-pixel values, full-GT statistics, overlays
eval/score.py              metrics from the parquet only: common pixel set, pooled / per-image, bootstrap CIs, domains, footnotes
eval/checks.py             validation tables: δ1 reproduction (DepthLM Table 1, DepthVLM Tables 1–2), sparse vs dense, z-conversion direction
eval/env_check.py          environment check without data packs (each model on the KITTI demo image shipped in the Metric3D repo)
eval/m3d_nyu.py            Metric3Dv2 NYUv2 RMS reproduction (paper benchmark protocol)
eval/common.py             datasets, domains, raw-data columns
prep/                      raw data → official extraction → GT check → common pixels (DIODE) → data packs; fetch_ext.sh
bench/                     DepthVLM-Bench annotation files (Hugging Face, Apache-2.0), DIODE Outdoor points, intrinsics
envs/                      pinned requirements per model
third_party/               DepthVLM @ 5d1472d (Apache-2.0), DepthLM_Official @ 3e76f58 (CC BY-NC 4.0)
docs/protocol.md           every applied setting (Korean)
```

## Status

2026-10-01:
- The five datasets are prepared and the benchmark GT is reproduced point by point.
- The official inference settings of the four dense models were checked line by line. All five models run end to end locally on a few images; UniDepthV2's official demo reproduces its documented ARel of 7.45 %.
- Next: H200 smoke test, then the pilot (iBims-1, nuScenes) and the validation checks in docs/protocol.md §6, before the full run.

## License

Code written for this repository: MIT (see LICENSE). Third-party code, model weights and datasets keep their own licenses, several of them noncommercial; see [NOTICE.md](NOTICE.md).
