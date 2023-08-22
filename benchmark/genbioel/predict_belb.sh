#!/usr/bin/env bash

#!/usr/bin/env bash

# stop execution if any command fails
set -e

DIRECTORY=""
PLM=""

if [[ -z "$DIRECTORY" ]] || [[ -z "$PLM" ]] ; then
    echo "Error! Please edit this file: add path to data (DIRECTORY) and local path to BioBERT weights"
    exit 1
fi

CORPORA="ncbi_disease bc5cdr_disease bc5cdr_chemical nlm_chem linnaeus s800 bioid_cell_line gnormplus nlm_gene medmentions"

for CORPUS in $CORPORA; do

    python ./train.py \
        "$DIRECTORY/runs/$CORPUS" \
        -model_token_path "$PLM" \
        -evaluation \
        -dict_path "$DIRECTORY/runs/$CORPUS/target_kb.json" \
        -trie_path "$DIRECTORY/runs/$CORPUS/trie.pkl" \
        -per_device_eval_batch_size 1 \
        -model_load_path "./model_checkpoints/$CORPUS/checkpoint-20000" \
        -max_position_embeddings 1024 \
        -seed 0 \
        -prompt_tokens_enc 0 \
        -prompt_tokens_dec 0 \
        -prefix_prompt \
        -num_beams 5 \
        -max_length 2014 \
        -min_length 1 \
        -dropout 0.1 \
        -attention_dropout 0.1 \
        -prefix_mention_is \
        -testset

done
