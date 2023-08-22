#!/bin/bash
#
# stop execution if any command fails
set -e

DIRECTORY=""
PLM=""

if [[ -z "$DIRECTORY" ]] || [[ -z "$PLM" ]] ; then
    echo "Error! Please edit this file: add path to data (DIRECTORY) and local path to BioBERT weights"
    exit 1
fi

CORPORA="ncbi_disease bc5cdr_disease bc5cdr_chemical nlm_chem linnaeus s800 bioid_cell_line gnormplus nlm_gene medmentions"
BEST="epoch_4"

for CORPUS in $CORPORA; do

    if [ "$CORPUS" == "gnormplus" ] || [ "$CORPUS" == "bc5cdr_disease" ]; then
        BEST="epoch_3"
    else
        BEST="epoch_4"
    fi

    DATA_PATH="$DIRECTORY/processed/$CORPUS/"
    PICKLE_PATH="models/belb/run1/trained/$CORPUS"
    OUTPUT="models/belb/run1/candidates/arbo/$CORPUS"
    MODEL_PATH="models/trained/belb/run1/$CORPUS/pos_neg_loss/$BEST/pytorch_model.bin"

    echo "Predict with model: $MODEL_PATH"
    python -m blink.biencoder.eval_cluster_linking \
        --bert_model="${PLM}" \
        --data_path="${DATA_PATH}"\
        --output_path="${OUTPUT}" \
        --pickle_src_path="${PICKLE_PATH}"\
        --path_to_model="${MODEL_PATH}"\
        --recall_k=64 \
        --embed_batch_size=2048 \
        --force_exact_search \
        --save_topk_result \
        --force_exact_search \
        --data_parallel
done

#NOTE: FIND OUT WHICH MODEL PERFORMED BEST
# (blink37) arboEL $ grep "Best performance in epoch" models/belb/trained/**/pos_neg_loss/no_type/log.txt

# python blink/crossencoder/eval_cluster_linking.py
#   --data_path=data/medmentions/processed
#   --output_path=models/trained/medmentions/candidates/arbo
#   --pickle_src_path=models/trained/medmentions
#   --path_to_biencoder_model=models/trained/medmentions_mst/pos_neg_loss/no_type/epoch_best_5th/pytorch_model.bin
#   --bert_model=models/biobert-base-cased-v1.1
#   --data_parallel
#   --scoring_batch_size=64
#   --save_topk_result
#
