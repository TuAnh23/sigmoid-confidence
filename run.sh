#!/bin/bash

GPU_TYPE=$1 # H100 A100
NR_GPUS=$2
TRAIN_CONFIG=$3  # "configs/train_head_only.yaml"
WANDB_RUN_ID=$4  # Can pass in a pre-existing ID to continue training from checkpoint
TRAIN_DATA_CONFIG="configs/paracrawl_data.yaml"  # "configs/towerblocks_data.yaml"
TEST_DATA_CONFIG="configs/paracrawl_data.yaml"

if [ -z "$WANDB_RUN_ID" ]; then
    WANDB_RUN_ID=$(date +%s%N)
fi

BATCH_CONFIG="configs/batch_size_${GPU_TYPE}.yaml"

# Copy git diff to run output
mkdir -p output/${WANDB_RUN_ID}
cp commit.txt output/${WANDB_RUN_ID}
cp diff.txt output/${WANDB_RUN_ID}

if [[ ${NR_GPUS} == 1 ]]; then
    echo "Training starts..."
    python train_sigmoid_head.py \
        --config-file-path ${TRAIN_CONFIG} ${BATCH_CONFIG} ${TRAIN_DATA_CONFIG} \
        --wandb-run-id ${WANDB_RUN_ID}

    echo "Inference starts..."
    python inference_sigmoid_head.py \
        --config-file-path "configs/inference.yaml" ${BATCH_CONFIG} ${TEST_DATA_CONFIG} \
        --wandb-run-id ${WANDB_RUN_ID}

    echo "Evaluation starts..."
    python evaluate.py \
        --config-file-path ${TEST_DATA_CONFIG} \
        --wandb-run-id ${WANDB_RUN_ID}
else
    echo "Training starts..."
    accelerate launch \
        --num_processes=${NR_GPUS} \
    train_sigmoid_head.py \
        --config-file-path ${TRAIN_CONFIG} ${BATCH_CONFIG} ${TRAIN_DATA_CONFIG} \
        --wandb-run-id ${WANDB_RUN_ID}

    echo "Inference starts..."
    accelerate launch \
        --num_processes=${NR_GPUS} \
    inference_sigmoid_head.py \
        --config-file-path "configs/inference.yaml" ${BATCH_CONFIG} ${TEST_DATA_CONFIG} \
        --wandb-run-id ${WANDB_RUN_ID}

    echo "Evaluation starts..."
    python evaluate.py \
        --config-file-path ${TEST_DATA_CONFIG} \
        --wandb-run-id ${WANDB_RUN_ID}
fi

