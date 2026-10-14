#
# Copyright IBM Corp. 2025 - 2026
# SPDX-License-Identifier: Apache-2.0 AND CC-BY-NC-4.0
# Derived from om-ai-lab/VLM-R1 and huggingface/open-r1 (Apache-2.0); IBM modifications under CC-BY-NC-4.0.
#
#Initial settings
cd ./train/GeoVLM_R1

conda activate /your_environment/path

bash src/geovlm_r1/run_scripts/run_grpo_geovlm_r1_image_caption.sh
bash src/geovlm_r1/run_scripts/run_grpo_geovlm_r1_classification.sh
bash src/geovlm_r1/run_scripts/run_grpo_geovlm_r1_CD_Caption.sh
bash src/geovlm_r1/run_scripts/run_grpo_geovlm_r1_grounding.sh
bash src/geovlm_r1/run_scripts/run_grpo_geovlm_r1_Refer.sh
bash src/geovlm_r1/run_scripts/run_grpo_geovlm_r1_Region_Captioning.sh
bash src/geovlm_r1/run_scripts/run_grpo_geovlm_r1_VQA.sh
bash src/geovlm_r1/run_scripts/run_grpo_geovlm_r1_xBD_object_Detection.sh
bash src/geovlm_r1/run_scripts/run_grpo_geovlm_r1_xBD_refer_object_Detection.sh

