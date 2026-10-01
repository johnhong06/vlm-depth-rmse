# NOTICE

vlm-depth-rmse — Copyright (c) 2026 Jihyuck Hong.

The MIT License (see LICENSE) covers ONLY the code written for this repository: `run.sh`, `h200/`, `eval/`, `prep/`, `envs/`.
It does not cover the third-party material below.

## 1. DepthVLM (Hanxun Yu et al.) — Apache License 2.0
- `third_party/DepthVLM/` is a copy of https://github.com/hanxunyu/DepthVLM at commit `5d1472d3b983fb7bf8bec2e9adf23bd0d69ed860` (2026-07-22): `model/`, `utils/`, `eval/`, `data_process/`, `configs/`, `README.md`, `requirements.txt`, `LICENSE`. Assets, examples and training scripts were not copied. No file was modified.
- `bench/*_pixel_depth_*.jsonl` are the DepthVLM-Bench annotation files from https://huggingface.co/datasets/JonnyYu828/DepthVLM-Bench (revision `72a1f15`, Apache-2.0), unmodified.
- The DepthVLM-4B checkpoint (https://huggingface.co/JonnyYu828/DepthVLM-4B, Apache-2.0) is not stored here.
- Citation: Yu et al., *Unlocking Dense Metric Depth Estimation in VLMs*, arXiv:2605.15876.

## 2. DepthLM (Meta Platforms, Inc.) — CC BY-NC 4.0 (code), FAIR Noncommercial Research License (model)
- `third_party/DepthLM_Official/` holds `eval.py`, `requirements.txt`, `README.md`, `LICENSE`, `utils/datasets.py` and `utils/metrics.py` from https://github.com/facebookresearch/DepthLM_Official at commit `3e76f58c394cded6ea922fa2fe6967021ceb16bd` (2026-06-01), unmodified (an empty `utils/__init__.py` was added).
- `eval/depthlm_sparse.py` imports the official preprocessing (`undistort_image`, `normalizing_focal_length`, `generate_prompt_depth_sft`). It also re-implements the arrow marker and the Pixtral message format of `eval.py` / `dataset_inference`; to the extent that code follows the original, it remains under CC BY-NC 4.0.
- The checkpoint `facebook/DepthLM` (Pixtral-12B based) is not stored here; a copy of its license is in `third_party/DepthLM_Official/MODEL_LICENSE`. Its outputs (the DepthLM rows of the result files) may be used for noncommercial research only, and publications must acknowledge DepthLM.
- Citation: Cai et al., *DepthLM: Metric Depth From Vision Language Models*, arXiv:2509.25413.

## 3. Dense depth models — fetched at run time, not included
`prep/fetch_ext.sh` downloads the official repositories as pinned-commit archives from GitHub into `ext/` (git-ignored); weights are downloaded at run time.
- Depth Anything V2: https://github.com/DepthAnything/Depth-Anything-V2 @ `a561b84` (code Apache-2.0). Weights depth-anything/Depth-Anything-V2-Metric-{Hypersim,VKITTI}-Large: the GitHub README states CC-BY-NC-4.0 for the Large models (the Hugging Face cards say Apache-2.0; we follow the stricter one). Yang et al., arXiv:2406.09414.
- UniDepth: https://github.com/lpiccinelli-eth/UniDepth @ `8d8cfe4` (CC BY-NC 4.0); weights lpiccinelli/unidepth-v2-vitl14. Piccinelli et al., arXiv:2502.20110.
- Metric3D: https://github.com/YvanYin/Metric3D @ `eb5b6fa` (BSD 2-Clause); weights JUGGHM/Metric3D `metric_depth_vit_large_800k.pth`. Hu et al., arXiv:2404.15506.
- Depth Pro: https://github.com/apple/ml-depth-pro @ `9e65e4d` (Apple sample-code license); weights apple/DepthPro (Apple ML Research model license, `apple-amlr`). Bochkovskii et al., arXiv:2410.02073.
- `eval/dense_sparse.py` re-implements the documented inference calls of these repositories (and, for Metric3D, the pre/post-processing of its `hubconf.py` demo); those lines follow the respective licenses.

## 4. Datasets — not redistributed in this repository
- The annotation files and intrinsics sidecars contain only pixel coordinates, depth values at those pixels and camera parameters.
- Images and depth maps are rebuilt locally from the official sources and are delivered privately to the compute server for noncommercial research use.
  - iBims-1: TUM, CC BY 4.0
  - nuScenes: Motional, CC BY-NC-SA 4.0
  - DDAD: TRI, CC BY-NC-SA 4.0
  - ETH3D: CC BY-NC-SA 4.0
  - SUN RGB-D and NYU Depth v2: research use
  - DIODE: MIT (diode-dataset.org); `bench/diode_outdoor_pixel_depth_val.jsonl` holds our sampled pixels and their depth values
- If a license above is uncertain, check the original dataset page; this file does not override any dataset's terms.

## 5. Python packages
PyTorch (BSD-3), transformers, accelerate and huggingface_hub (Apache-2.0), math-verify (Apache-2.0), nuscenes-devkit (Apache-2.0), pandas, numpy, pyarrow, OpenCV, Pillow (BSD/MIT/Apache/HPND).
