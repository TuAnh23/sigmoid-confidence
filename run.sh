#!/bin/bash

GPU_TYPE=$1 # H100 A100
NR_GPUS=$2
WANDB_RUN_ID=$3  # Can pass in a pre-existing ID to continue training from checkpoint

if [ -z "$WANDB_RUN_ID" ]; then
    WANDB_RUN_ID=$(date +%s%N)
fi

BATCH_CONFIG="configs/batch_size_${GPU_TYPE}.yaml"

if [[ ${NR_GPUS} == 1 ]]; then
    echo "Training starts..."
    python train_sigmoid_head.py \
        --config-file-path "configs/train_head_only.yaml" ${BATCH_CONFIG} \
        --wandb-run-id ${WANDB_RUN_ID}

    echo "Inference starts..."
    python inference_sigmoid_head.py \
        --config-file-path "configs/train_head_only.yaml" ${BATCH_CONFIG} \
        --wandb-run-id ${WANDB_RUN_ID}
else
    echo "Training starts..."
    accelerate launch \
        --num_processes=${NR_GPUS} \
    train_sigmoid_head.py \
        --config-file-path "configs/train_head_only.yaml" ${BATCH_CONFIG} \
        --wandb-run-id ${WANDB_RUN_ID}

    echo "Inference starts..."
    accelerate launch \
        --num_processes=${NR_GPUS} \
    inference_sigmoid_head.py \
        --config-file-path "configs/train_head_only.yaml" ${BATCH_CONFIG} \
        --wandb-run-id ${WANDB_RUN_ID}
fi

