#!/bin/bash

# stop execution if any command fails
set -e

DIRECTORY=""

if [[ -z "$DIRECTORY" ]] ; then
    echo "Error! Please edit this file: add path to data (DIRECTORY)"
    exit 1
fi

CORPORA="ncbi_disease bc5cdr_disease bc5cdr_chemical nlm_chem linnaeus s800 bioid_cell_line gnormplus nlm_gene medmentions"

for CORPUS in $CORPORA; do

    OUTPUT_DIR="models/$CORPUS/predictions"
    mkdir -p "$OUTPUT_DIR"
    DATA_DIR="$DIRECTORY/$CORPUS"
    MODEL_NAME_OR_PATH="models/$CORPUS"

    python eval.py \
        --model_name_or_path ${MODEL_NAME_OR_PATH} \
        --dictionary_path "${DATA_DIR}/processed_dictionary.txt" \
        --data_dir "${DATA_DIR}/processed_test" \
        --output_dir "${OUTPUT_DIR}" \
        --use_cuda \
        --topk 20 \
        --max_length 25 \
        --save_predictions \
        --score_mode hybrid

done
