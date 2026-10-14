#
# Copyright IBM Corp. 2025 - 2026
# SPDX-License-Identifier: CC-BY-NC-4.0
#
# Usage (from the repository root): bash GeoVLM_inference/run_inference.sh
# Predictions are written to ./GeoVLM_inference/GeoVLM_VLM_R1_Results/
#Initial settings
cd  ./GeoVLM_inference
eval "$(conda shell.bash hook)"
conda activate /your_environment


python generate_inference_CD.py
python generate_inference_classification.py
python generate_inference_grounding.py
python generate_inference_Image_Captions.py
python generate_inference_refer.py
python generate_inference_region_captioning.py
python generate_inference_vqa.py
python generate_inference_xBD_caption.py
python generate_inference_xBD_classification.py
python generate_inference_xBD_object_detection.py
python generate_inference_xBD_referred_object_detection.py
