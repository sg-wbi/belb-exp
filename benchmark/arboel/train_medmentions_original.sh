#!/bin/bash

# stop execution if any command fails
set -e


# if not enough GPU memory: you get very similar results
# --train_batch_size=64
# --gradient_accumulation_steps=8

PLM="/vol/home-vol3/wbi/gardasam/data/models/biobert-v1.1/"
DATA_PATH="data/medmentions/processed"
OUTPUT="models/original/trained/medmentions/pos_neg_loss"
PICKLE_PATH="models/original/trained/medmentions"

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
