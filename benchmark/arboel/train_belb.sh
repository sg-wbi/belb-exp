#!/bin/bash

# stop execution if any command fails
set -e

DIRECTORY=""
PLM=""

if [[ -z "$DIRECTORY" ]] || [[ -z "$PLM" ]] ; then
    echo "Error! Please edit this file: add path to data (DIRECTORY) and local path to BioBERT weights"
    exit 1
fi


CORPORA="ncbi_disease bc5cdr_disease bc5cdr_chemical nlm_chem linnaeus s800 bioid_cell_line gnormplus nlm_gene medmentions"

# if not enough GPU memory: you get very similar results
# --train_batch_size=64
# --gradient_accumulation_steps=8

for CORPUS in $CORPORA; do

    DATA_PATH="$DIRECTORY/processed/$CORPUS/"
    OUTPUT="models/belb/run1/trained/$CORPUS/pos_neg_loss"
    PICKLE_PATH="models/belb/run1/trained/$CORPUS"

    python -m blink.biencoder.train_biencoder_mst \
        --bert_model="${PLM}" \
        --data_path="${DATA_PATH}"\
        --output_path="${OUTPUT}" \
        --pickle_src_path="${PICKLE_PATH}" \
        --num_train_epochs=5 \
        --train_batch_size=128 \
        --gradient_accumulation_steps=4 \
        --eval_interval=10000 \
        --pos_neg_loss \
        --force_exact_search \
        --embed_batch_size=3096 \
        --data_parallel
done
