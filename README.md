# GeoVLM-R1 Toolkit

[![Paper](https://img.shields.io/badge/arXiv-2509.25026-b31b1b.svg)](https://arxiv.org/abs/2509.25026)
[![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://mustansarfiaz.github.io/GeoVLM-R1/)
[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](LICENSE)

Official code release for **GeoVLM-R1: Reinforcement Fine-Tuning for Improved Remote Sensing Reasoning**.

> **License notice:** this repository is released for **non-commercial research purposes only** under
> [CC BY-NC 4.0](LICENSE). Commercial use is not permitted without prior written permission.
> Contact: [mustansar.fiaz@ibm.com](mailto:mustansar.fiaz@ibm.com).

## Scope

Reinforcement learning (RL) delivers strong reasoning in natural images, but its potential for Earth
Observation (EO) is largely unexplored. EO tasks such as referred object detection, image and region
captioning, change detection, grounding and temporal analysis require task-aware reasoning.
GeoVLM-R1 is a post-training framework with **task-aware rewards** that adapts reasoning-based RL
(GRPO) to diverse EO tasks. This makes reasoning on remote-sensing images stronger, keeps
optimisation stable and improves robustness.

This toolkit provides:

| Component | Folder | Description |
|-----------|--------|-------------|
| Supervised fine-tuning | [`train/GeoVLM_SFT`](train/GeoVLM_SFT) | SFT of Qwen2.5-VL on EarthDial-Instruct (DeepSpeed, Liger kernel). |
| RL fine-tuning (GRPO) | [`train/GeoVLM_R1`](train/GeoVLM_R1) | GRPO with task-aware rewards, one run script per EO task. |
| Inference | [`GeoVLM_inference`](GeoVLM_inference) | Generates predictions on the GeoVLM benchmark (`*.jsonl`). |
| Evaluation | [`GeoVLM_Eval`](GeoVLM_Eval) | Computes the metrics reported in the paper from the predictions. |

Supported tasks: scene classification, VQA, image captioning, region captioning, visual grounding,
referred object detection, change-detection captioning/classification and xBD (referred) object
detection.

## Repository structure

```text
GeoVLM-R1-Toolkit/
├── train/
│   ├── GeoVLM_SFT/            # Stage 1: supervised fine-tuning
│   │   ├── scripts/sft_GeoVLM.sh
│   │   └── src/
│   └── GeoVLM_R1/             # Stage 2: GRPO reinforcement fine-tuning
│       ├── setup.sh
│       ├── run_finetune_geovlmr1.sh
│       └── src/geovlm_r1/run_scripts/run_grpo_geovlm_r1_<task>.sh
├── GeoVLM_inference/          # Benchmark inference
│   ├── run_inference.sh
│   └── generate_inference_<task>.py
└── GeoVLM_Eval/               # Benchmark evaluation
    ├── eval.sh
    └── rs_<task>/eval*.py
```

## Usage

All commands below are run from the **repository root** unless stated otherwise. In every
`*.sh` script, replace the placeholder `conda activate /your/...` path with your own environment.

### 1. Installation

Two environments are used: one for SFT and one for GRPO, inference and evaluation.

```bash
git clone https://github.com/IBM/GeoVLM-R1-Toolkit.git
cd GeoVLM-R1-Toolkit

# SFT environment (Python 3.11)
conda env create -f train/GeoVLM_SFT/environment.yaml
# or: pip install -r train/GeoVLM_SFT/requirements.txt

# GRPO / inference / evaluation environment
conda create -n geovlm-r1 python=3.11 -y
conda activate geovlm-r1
cd train/GeoVLM_R1 && bash setup.sh && cd ../..
pip install nltk rouge_score shapely scikit-learn colorama numpy   # evaluation metrics
```

The evaluation scripts download NLTK data (`punkt`, `wordnet`) into `./nltk_data` the first time
they run, so they need internet access then. If the Punkt tokenizer is unavailable, they fall back
to a regex tokenizer.

### 2. Data

Put the training data and the benchmark in these folders, which the default script paths expect:

```text
train/GeoVLM_SFT/GeoVLM_R1_dataset/EarthDial_Instruct/...     # SFT jsons + images
train/GeoVLM_R1/GeoVLM_R1_dataset/EarthDial_Instruct/...      # GRPO jsonl + images
GeoVLM_inference/GeoVLM_Bench/                                 # benchmark *.jsonl + images
```

See the [project page](https://mustansarfiaz.github.io/GeoVLM-R1/) for download links for the
data and models. Every path can be changed in the corresponding script.

### 3. Training

**Stage 1: SFT.** This starts from `Qwen/Qwen2.5-VL-3B-Instruct` and writes the model to
`train/GeoVLM_SFT/GeoVLM_weights/SFT/Qwen2.5-VL-3B-GeoVLM-SFT`.

```bash
cd train/GeoVLM_SFT
bash scripts/sft_GeoVLM.sh
```

**Stage 2: GRPO.** This starts from the SFT checkpoint. There is one script per task, and
`run_finetune_geovlmr1.sh` runs all of them.

```bash
bash train/GeoVLM_R1/run_finetune_geovlmr1.sh
# or a single task, from train/GeoVLM_R1:
bash src/geovlm_r1/run_scripts/run_grpo_geovlm_r1_classification.sh
```

Before running, set `CUDA_VISIBLE_DEVICES`, `NUM_GPUS`, `model_path`, `data_paths` and
`image_folders` in the run scripts. The optional LLM-based reward reads `OPENAI_API_KEY` from the
environment. Without it, the reward falls back to string matching.

### 4. Inference

```bash
bash GeoVLM_inference/run_inference.sh
```

Each `generate_inference_<task>.py` script accepts these arguments:

| Argument | Default | Meaning |
|----------|---------|---------|
| `--model_path` | `./GeoVLM_R1` | GeoVLM-R1 checkpoint |
| `--test_data_directory` | `./GeoVLM_Bench/` | benchmark folder |
| `--output_jsonls_directory` | `./GeoVLM_VLM_R1_Results/` | prediction folder |

Predictions are written to `GeoVLM_inference/GeoVLM_VLM_R1_Results/`.

### 5. Evaluation

```bash
bash GeoVLM_Eval/eval.sh
# custom prediction folder:
RESULTS_DIR=/path/to/GeoVLM_VLM_R1_Results bash GeoVLM_Eval/eval.sh
```

By default, `eval.sh` reads predictions from `../GeoVLM_inference/GeoVLM_VLM_R1_Results`. Each
evaluation script also accepts `--output_path` and `--datasets`.

| Task | Script | Metrics |
|------|--------|---------|
| Scene classification | `rs_classification/eval.py` | Accuracy |
| VQA | `rs_vqa/eval.py` | Accuracy |
| Image captioning | `rs_image_caption/eval.py` | METEOR, ROUGE-1, ROUGE-L |
| Region captioning | `rs_region_captioning/eval.py` | METEOR, ROUGE-1, ROUGE-L |
| Grounding description | `rs_grounding_description/eval.py` | Precision@0.5 / @0.25 IoU, METEOR, ROUGE |
| Referred detection | `rs_detection/eval.py` | Precision@0.5 IoU, per group (small / medium / large / single / multiple) |
| Change-detection captioning | `rs_change_detection/eval_caption.py` | METEOR, ROUGE-1, ROUGE-L |
| Change-detection classification | `rs_change_detection/eval_classification.py` | Accuracy |
| xBD (referred) object detection | `rs_change_detection/eval_xBD_*object_detection.py` | Precision@0.5 / @0.25 IoU |

## Notes

**NOTE: This repository is licensed under CC BY-NC 4.0 and may be used for non-commercial
research purposes only.** Third-party code included here keeps its original license. See
[NOTICE](NOTICE).

**NOTE: This repository is configured with the [DCO bot](https://github.com/probot/dco).
Every commit must be signed off (`git commit -s`).** See [CONTRIBUTING.md](CONTRIBUTING.md).

If you have questions or find a problem, please open a [new issue][issues] or contact
[mustansar.fiaz@ibm.com](mailto:mustansar.fiaz@ibm.com).

Pull requests are welcome. Please make sure your patches are well tested, and create a separate
topic branch for each change. For example:

1. Fork the repo
2. Create your feature branch (`git checkout -b my-new-feature`)
3. Commit your changes (`git commit -s -am 'Added some feature'`)
4. Push to the branch (`git push origin my-new-feature`)
5. Create a new Pull Request

## License

This project is licensed under the
[Creative Commons Attribution-NonCommercial 4.0 International License](LICENSE) (CC BY-NC 4.0)
and is provided **for non-commercial research purposes only**. For commercial licensing, please
contact [mustansar.fiaz@ibm.com](mailto:mustansar.fiaz@ibm.com).

All source files include a copyright and SPDX license header:

```text
#
# Copyright IBM Corp. 2025 - 2026
# SPDX-License-Identifier: CC-BY-NC-4.0
#
```

Files derived from third-party Apache-2.0 or BSD projects keep their original notices and are
marked, for example, `SPDX-License-Identifier: Apache-2.0 AND CC-BY-NC-4.0`.

## Citation

If you find this work useful, please cite:

```bibtex
@article{fiaz2025geovlmr1,
  title   = {GeoVLM-R1: Reinforcement Fine-Tuning for Improved Remote Sensing Reasoning},
  author  = {Fiaz, Mustansar and Debary, Hiyam and Fraccaro, Paolo and Paudel, Danda and
             Van Gool, Luc and Khan, Fahad and Khan, Salman},
  journal = {arXiv preprint arXiv:2509.25026},
  year    = {2025}
}
```

## Acknowledgements

This code builds on [VLM-R1](https://github.com/om-ai-lab/VLM-R1),
[open-r1](https://github.com/huggingface/open-r1),
[Qwen2-VL-Finetune](https://github.com/2U1/Qwen2-VL-Finetune),
[Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL), [EarthDial](https://github.com/hiyamdebary/EarthDial)
and [GeoChat](https://github.com/mbzuai-oryx/GeoChat). We thank the authors for releasing their code.

## Authors

- Mustansar Fiaz (maintainer) <mustansar.fiaz@ibm.com>
- Hiyam Debary, Paolo Fraccaro, Danda Paudel, Luc Van Gool, Fahad Khan, Salman Khan

[issues]: https://github.com/IBM/GeoVLM-R1-Toolkit/issues/new
