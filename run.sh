#!/bin/bash

NR_GPUS=$1
TRAIN_CONFIG=$2  # "configs/train_head_only.yaml"
TRAIN_DATA_CONFIG=$3  # "configs/tulu3_data.yaml" "configs/paracrawl_data.yaml" "configs/towerblocks_data.yaml"
BATCH_CONFIG=$4  # "configs/batch_size_${GPU_TYPE}.yaml"
WANDB_RUN_ID=$5  # Can pass in a pre-existing ID to continue training from checkpoint
declare -a TEST_GENERATE_DATA_CONFIGS=( "configs/paracrawl_data.yaml" "configs/wmt24_data.yaml" ) # "configs/pawsx.yaml")
declare -a TEST_SCORE_DATA_CONFIGS=( "configs/wmt22_general_MT_ende_DA_bestmt.yaml" "configs/wmt22_general_MT_ende_DA_worstmt.yaml" "configs/wmt24_enzh_FD_bestmt.yaml" "configs/wmt24_enzh_FD_worstmt.yaml" "configs/wmt24_enzh_FD_allmt.yaml" "configs/wmt24_ende_FD_bestmt.yaml" "configs/wmt24_ende_FD_worstmt.yaml" "configs/wmt24_ende_FD_allmt.yaml" ) #"configs/pawsx.yaml" )

if [ -z "$WANDB_RUN_ID" ]; then
    WANDB_RUN_ID=$(date +%s%N)
fi


# Copy git diff to run output
mkdir -p output/${WANDB_RUN_ID}
cp commit.txt output/${WANDB_RUN_ID}
cp diff.txt output/${WANDB_RUN_ID}

if [[ ${NR_GPUS} == 1 ]]; then
    echo "Training starts..."
    python train_sigmoid_head.py \
        --config-file-path ${TRAIN_CONFIG} ${BATCH_CONFIG} ${TRAIN_DATA_CONFIG} \
        --wandb-run-id ${WANDB_RUN_ID}

    for TEST_GENERATE_DATA_CONFIG in "${TEST_GENERATE_DATA_CONFIGS[@]}"; do
        echo "Inference starts..."
        python inference_sigmoid_head.py \
            --config-file-path "configs/generate_and_eval.yaml" ${BATCH_CONFIG} ${TEST_GENERATE_DATA_CONFIG} \
            --wandb-run-id ${WANDB_RUN_ID}

        echo "Evaluation starts..."
        python evaluate.py \
            --config-file-path "configs/generate_and_eval.yaml" ${TEST_GENERATE_DATA_CONFIG} \
            --wandb-run-id ${WANDB_RUN_ID}
    done

    for TEST_SCORE_DATA_CONFIG in "${TEST_SCORE_DATA_CONFIGS[@]}"; do
        echo "Inference starts..."
        python inference_sigmoid_head.py \
            --config-file-path "configs/force_decoding_and_eval.yaml" ${BATCH_CONFIG} ${TEST_SCORE_DATA_CONFIG} \
            --wandb-run-id ${WANDB_RUN_ID}

        echo "Evaluation starts..."
        python evaluate.py \
            --config-file-path "configs/force_decoding_and_eval.yaml" ${TEST_SCORE_DATA_CONFIG} \
            --wandb-run-id ${WANDB_RUN_ID}
    done

else
    echo "Training starts..."
    accelerate launch \
        --num_processes=${NR_GPUS} \
    train_sigmoid_head.py \
        --config-file-path ${TRAIN_CONFIG} ${BATCH_CONFIG} ${TRAIN_DATA_CONFIG} \
        --wandb-run-id ${WANDB_RUN_ID}

    for TEST_DATA_CONFIG in "${TEST_DATA_CONFIGS[@]}"; do
        echo "Inference starts..."
        accelerate launch \
            --num_processes=${NR_GPUS} \
        inference_sigmoid_head.py \
            --config-file-path "configs/generate_and_eval.yaml" ${BATCH_CONFIG} ${TEST_DATA_CONFIG} \
            --wandb-run-id ${WANDB_RUN_ID}

        echo "Evaluation starts..."
        python evaluate.py \
            --config-file-path "configs/generate_and_eval.yaml" ${TEST_DATA_CONFIG} \
            --wandb-run-id ${WANDB_RUN_ID}
    done
fi

