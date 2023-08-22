#!/bin/bash

# stop execution if any command fails
set -e

DIRECTORY=""

if [[ -z "$DIRECTORY" ]] ; then
    echo "Error! Please edit this file: add path to data (DIRECTORY)"
    exit 1
fi

MODEL_NAME_OR_PATH="dmis-lab/biobert-base-cased-v1.1"
CORPORA="ncbi_disease bc5cdr_disease bc5cdr_chemical nlm_chem linnaeus s800 bioid_cell_line gnormplus nlm_gene medmentions"

for CORPUS in $CORPORA; do

    OUTPUT_DIR="models/$CORPUS"
    DATA_DIR="$DIRECTORY/runs/$CORPUS"

    python train.py \
        --model_name_or_path ${MODEL_NAME_OR_PATH} \
        --train_dictionary_path "${DATA_DIR}/processed_dictionary.txt" \
        --train_dir "${DATA_DIR}/processed_train" \
        --output_dir "${OUTPUT_DIR}" \
        --use_cuda \
        --topk 20 \
        --epoch 10 \
        --train_batch_size 16 \
        --initial_sparse_weight 0 \
        --learning_rate 1e-5 \
        --max_length 25 \
        --dense_ratio 0.5

done
