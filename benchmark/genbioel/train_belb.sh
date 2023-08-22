#!/usr/bin/env bash

# stop execution if any command fails
set -e

DIRECTORY=""
PLM=""

if [[ -z "$DIRECTORY" ]] || [[ -z "$PLM" ]] ; then
    echo "Error! Please edit this file: add path to data (DIRECTORY) and local path to BART weights"
    exit 1
fi


CORPORA="ncbi_disease bc5cdr_disease bc5cdr_chemical nlm_chem linnaeus s800 bioid_cell_line gnormplus nlm_gene medmentions"
LR=1e-5
WARMUP_STEPS=500

for CORPUS in $CORPORA; do
    python ./train.py \
        "$DIRECTORY/runs/$CORPUS" \
        -model_load_path "$PLM" \
        -model_token_path "$PLM" \
        -model_save_path "./model_checkpoints/$CORPUS" \
        -save_steps 20000 \
        -logging_path "./logs/$CORPUS" \
        -logging_steps 100 \
        -init_lr "$LR" \
        -per_device_train_batch_size 8 \
        -evaluation_strategy no \
        -label_smoothing_factor 0.1 \
        -max_grad_norm 0.1 \
        -max_steps 20000 \
        -warmup_steps "$WARMUP_STEPS" \
        -weight_decay 0.01 \
        -rdrop 0.0 \
        -lr_scheduler_type polynomial \
        -attention_dropout 0.1  \
        -prompt_tokens_enc 0 \
        -prompt_tokens_dec 0 \
        -max_position_embeddings 1024 \
        -seed 0 \
        -finetune \
        -prefix_mention_is
done
