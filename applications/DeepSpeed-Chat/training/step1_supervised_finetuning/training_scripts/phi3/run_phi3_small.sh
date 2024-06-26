#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
OUTPUT=$1
ZERO_STAGE=$2
if [ "$OUTPUT" == "" ]; then
    OUTPUT=./output_step1_phi3_7b_1
fi
if [ "$ZERO_STAGE" == "" ]; then
    ZERO_STAGE=3
fi


mkdir -p $OUTPUT

wandb login $WANDB_API_KEY

deepspeed main.py \
   --data_path html_primary_identification \
   --data_split 1,0,0 \
   --model_name_or_path microsoft/Phi-3-small-8k-instruct \
   --per_device_train_batch_size 3 \
   --per_device_eval_batch_size 4 \
   --max_seq_len 4096 \
   --learning_rate 9.65e-6 \
   --weight_decay 0. \
   --num_train_epochs 4  \
   --gradient_accumulation_steps 1 \
   --lr_scheduler_type cosine \
   --num_warmup_steps 0 \
   --seed 1234 \
   --gradient_checkpointing \
   --zero_stage $ZERO_STAGE \
   --deepspeed \
   --output_dir $OUTPUT \
   --enable_wandb \
   --enable_tensorboard \
   &> $OUTPUT/training.log

#    --model_name_or_path meta-llama/Llama-2-7b-hf \
