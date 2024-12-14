#!/bin/bash

WORK_DIR="..."
GRADIENT_CHECKPOINTING_KWARGS='{"use_reentrant": false}'
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 运行 Python 脚本
accelerate launch --config_file=${WORK_DIR}/config/deepspeed_zero3.yaml ${WORK_DIR}/script/sft_llama.py \
    --dataset_name "..." \
    --model_name_or_path "..." \
    --output_dir "..." \
    --max_seq_length 8192 \
    --per_device_train_batch_size 2 \
    --save_steps 1000 \
    --dataloader_drop_last \
    --logging_steps 5 \
    --do_train \
    --gradient_checkpointing_kwargs "$GRADIENT_CHECKPOINTING_KWARGS" \
    --fp16 \
    --attn_implementation "flash_attention_2" \
    --torch_dtype bfloat16 \
    --gradient_checkpointing 1 \