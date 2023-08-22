#!/bin/bash

# stop execution if any command fails
set -e

MODEL_NAME_OR_PATH=dmis-lab/biobert-base-cased-v1.1
OUTPUT_DIR=models/biosyn-biobert-ncbi-disease
DATA_DIR=datasets/ncbi-disease

CUDA_VISIBLE_DEVICES=1 python train.py \
    --model_name_or_path "${MODEL_NAME_OR_PATH}" \
    --train_dictionary_path "${DATA_DIR}/train_dictionary.txt" \
    --train_dir "${DATA_DIR}/processed_traindev"  \
    --output_dir "${OUTPUT_DIR}" \
    --use_cuda \
    --topk 20 \
    --epoch 10 \
    --train_batch_size 16\
    --learning_rate 1e-5 \
    --max_length 25
