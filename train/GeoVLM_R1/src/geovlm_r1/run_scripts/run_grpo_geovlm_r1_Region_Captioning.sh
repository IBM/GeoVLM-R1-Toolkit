#
# Copyright IBM Corp. 2025 - 2026
# SPDX-License-Identifier: Apache-2.0 AND CC-BY-NC-4.0
# Derived from om-ai-lab/VLM-R1 and huggingface/open-r1 (Apache-2.0); IBM modifications under CC-BY-NC-4.0.
#


export DEBUG_MODE="true"
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NUM_GPUS=4

RUN_NAME="GeoVLM_R1_Region_Captioning"
export LOG_PATH="./debug_log_$RUN_NAME.txt"
data_paths="./GeoVLM_R1_dataset/EarthDial_Instruct/jsons/GRPO_EarthDial_Instructs/GRPO_GeoChat_region_caption_refined.jsonl"
image_folders="./GeoVLM_R1_dataset/EarthDial_Instruct/images/"

model_path="../GeoVLM_SFT/GeoVLM_weights/SFT/Qwen2.5-VL-3B-GeoVLM-SFT"
export WANDB_MODE=offline

torchrun --nproc_per_node="$NUM_GPUS" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12321" \
    src/geovlm_r1/grpo_geovlm_r1.py \
    --deepspeed local_scripts/zero3.json \
    --output_dir ./GeoVLM_R1_Models/$RUN_NAME \
    --model_name_or_path  $model_path \
    --dataset_name none \
    --image_folders $image_folders \
    --data_file_paths $data_paths \
    --freeze_vision_modules true \
    --max_prompt_length 1024 \
    --num_generations 2 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --logging_steps 1 \
    --bf16 \
    --torch_dtype bfloat16 \
    --data_seed 42 \
    --report_to wandb \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --num_train_epochs 2 \
    --run_name $RUN_NAME \
    --save_steps 100 \
    --save_only_model true \
    --save_strategy "epoch" \
    --save_total_limit 2 \
    --dataloader_num_workers 4
