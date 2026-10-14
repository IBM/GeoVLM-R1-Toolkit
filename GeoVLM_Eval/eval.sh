#
# Copyright IBM Corp. 2025 - 2026
# SPDX-License-Identifier: CC-BY-NC-4.0
#
# Usage (from the repository root): bash GeoVLM_Eval/eval.sh
#Initial settings
cd  ./GeoVLM_Eval
eval "$(conda shell.bash hook)"
conda activate /your/conda/env/path

# Folder holding the *.jsonl predictions written by GeoVLM_inference/run_inference.sh.
# Override with: RESULTS_DIR=/path/to/results bash GeoVLM_Eval/eval.sh
RESULTS_DIR=${RESULTS_DIR:-../GeoVLM_inference/GeoVLM_VLM_R1_Results}

python rs_detection/eval.py --output_path "$RESULTS_DIR"
python rs_grounding_description/eval.py --output_path "$RESULTS_DIR"
python rs_region_captioning/eval.py --output_path "$RESULTS_DIR"
python rs_classification/eval.py --output_path "$RESULTS_DIR"

python rs_image_caption/eval.py --output_path "$RESULTS_DIR"
python rs_vqa/eval.py --output_path "$RESULTS_DIR"

python rs_change_detection/eval_caption.py --output_path "$RESULTS_DIR"
python rs_change_detection/eval_classification.py --output_path "$RESULTS_DIR"
python rs_change_detection/eval_xBD_object_detection.py --output_path "$RESULTS_DIR"
python rs_change_detection/eval_xBD_referred_object_detection.py --output_path "$RESULTS_DIR"

