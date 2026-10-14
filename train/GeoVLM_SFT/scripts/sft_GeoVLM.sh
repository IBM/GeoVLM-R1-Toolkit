#!/bin/bash
#
# Copyright IBM Corp. 2025 - 2026
# SPDX-License-Identifier: Apache-2.0 AND CC-BY-NC-4.0
# Derived from 2U1/Qwen2-VL-Finetune (Apache-2.0); IBM modifications under CC-BY-NC-4.0.
#

# MODEL_NAME="Qwen/Qwen2-VL-7B-Instruct"
# MODEL_NAME="Qwen/Qwen2-VL-2B-Instruct"
MODEL_NAME="Qwen/Qwen2.5-VL-3B-Instruct"
#MODEL_NAME="Qwen/Qwen2.5-VL-7B-Instruct"


GLOBAL_BATCH_SIZE=128
BATCH_PER_DEVICE=4
NUM_DEVICES=8
GRAD_ACCUM_STEPS=$((GLOBAL_BATCH_SIZE / (BATCH_PER_DEVICE * NUM_DEVICES)))

export PYTHONPATH=.:./src:$PYTHONPATH

data_path_json="./GeoVLM_R1_dataset/EarthDial_Instruct/jsons/EarthDial_Instruct_jsons/sft_EarthDial_Instruct_jsons.json"
data_path_image="./GeoVLM_R1_dataset/"
deepspeed src/train/train_sft.py \
    --use_liger True \
    --deepspeed scripts/zero3_offload.json \
    --model_id $MODEL_NAME \
    --data_path $data_path_json \
    --image_folder $data_path_image \
    --remove_unused_columns False \
    --freeze_vision_tower False \
    --freeze_llm False \
    --freeze_merger False \
    --bf16 True \
    --fp16 False \
    --disable_flash_attn2 True \
    --output_dir ./GeoVLM_weights/SFT/Qwen2.5-VL-3B-GeoVLM-SFT \
    --num_train_epochs 2 \
    --per_device_train_batch_size $BATCH_PER_DEVICE \
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    --image_min_pixels $((512 * 28 * 28)) \
    --image_max_pixels $((1280 * 28 * 28)) \
    --learning_rate 1e-5 \
    --merger_lr 1e-5 \
    --vision_lr 2e-6 \
    --weight_decay 0.1 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --gradient_checkpointing True \
    --report_to tensorboard \
    --lazy_preprocess True \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 2 \
    --dataloader_num_workers 4
