#!/bin/bash
#
# Copyright IBM Corp. 2025 - 2026
# SPDX-License-Identifier: Apache-2.0 AND CC-BY-NC-4.0
# Derived from 2U1/Qwen2-VL-Finetune (Apache-2.0); IBM modifications under CC-BY-NC-4.0.
#

MODEL_NAME="Qwen/Qwen2-VL-7B-Instruct"
# MODEL_NAME="Qwen/Qwen2-VL-2B-Instruct"

export PYTHONPATH=src:$PYTHONPATH

python src/merge_lora_weights.py \
    --model-path ./output/testing_lora \
    --model-base $MODEL_NAME  \
    --save-model-path ./output/merge_test \
    --safe-serialization
