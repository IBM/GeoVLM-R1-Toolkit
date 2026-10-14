# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

## [0.1.0] - 2026-10-14

### Added
- Initial public release of the GeoVLM-R1 Toolkit:
  - `train/GeoVLM_SFT`: supervised fine-tuning of Qwen2.5-VL on EarthDial-Instruct.
  - `train/GeoVLM_R1`: GRPO reinforcement fine-tuning with task-aware rewards for EO tasks.
  - `GeoVLM_inference`: benchmark inference scripts.
  - `GeoVLM_Eval`: benchmark evaluation scripts.
- README, CC BY-NC 4.0 LICENSE, NOTICE and copyright/SPDX headers.

### Fixed
- Evaluation scripts now fall back to a regex tokenizer when the NLTK Punkt model is unavailable.
- Added division-by-zero guards in the evaluation metrics.
- Box parsing in the xBD (referred) object detection evaluation now handles boxes with fewer than 4 coordinates.
- The change-detection caption evaluation now ignores empty references.
- `eval.sh` now reads predictions from `GeoVLM_inference/GeoVLM_VLM_R1_Results` by default.

[unreleased]: https://github.com/IBM/GeoVLM-R1-Toolkit/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/IBM/GeoVLM-R1-Toolkit/releases/tag/v0.1.0
