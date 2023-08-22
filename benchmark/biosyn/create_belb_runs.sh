#!/usr/bin/env bash

# stop execution if any command fails
set -e

DIRECTORY=$1

if [[ -z "$DIRECTORY" ]]; then
    echo "Please specify path where BELB data is stored!"
    echo "Usage: create_belb_runs.sh <directory w/ converted belb data>"
    exit 1
fi

cd "preprocess"

echo "PREPROCESS KBS"

KBS="ctd_diseases ctd_chemicals cellosaurus ncbi_taxonomy umls"

for KB in $KBS; do
    echo "Preprocess KB: $KB"
    python dictionary_preprocess.py \
        --input_dictionary_path "$DIRECTORY/kbs/$KB/dictionary.txt" \
        --output_dictionary_path "$DIRECTORY/kbs/$KB/processed_dictionary.txt" \
        --lowercase \
        --remove_punctuation
done

NCBI_GENE_SUBSET="gnormplus nlm_gene"

for SUBSET in $NCBI_GENE_SUBSET; do
    echo "Preprocess NCBI Gene subset: $SUBSET"
    python dictionary_preprocess.py \
        --input_dictionary_path "$DIRECTORY/kbs/ncbi_gene/$SUBSET/dictionary.txt" \
        --output_dictionary_path "$DIRECTORY/kbs/ncbi_gene/$SUBSET/processed_dictionary.txt" \
        --lowercase \
        --remove_punctuation
done

cp -r "$DIRECTORY/kbs/umls/processed_dictionary.txt"                "$DIRECTORY/runs/medmentions"
cp -r "$DIRECTORY/kbs/ctd_diseases/processed_dictionary.txt"        "$DIRECTORY/runs/ncbi_disease"
cp -r "$DIRECTORY/kbs/ctd_diseases/processed_dictionary.txt"        "$DIRECTORY/runs/bc5cdr_disease"
cp -r "$DIRECTORY/kbs/ctd_chemicals/processed_dictionary.txt"       "$DIRECTORY/runs/bc5cdr_chemical"
cp -r "$DIRECTORY/kbs/ctd_chemicals/processed_dictionary.txt"       "$DIRECTORY/runs/nlm_chem"
cp -r "$DIRECTORY/kbs/ncbi_taxonomy/processed_dictionary.txt"       "$DIRECTORY/runs/linnaeus"
cp -r "$DIRECTORY/kbs/ncbi_taxonomy/processed_dictionary.txt"       "$DIRECTORY/runs/s800"
cp -r "$DIRECTORY/kbs/cellosaurus/processed_dictionary.txt"         "$DIRECTORY/runs/bioid_cell_line"
cp -r "$DIRECTORY/kbs/ncbi_gene/gnormplus/processed_dictionary.txt" "$DIRECTORY/runs/gnormplus"
cp -r "$DIRECTORY/kbs/ncbi_gene/nlm_gene/processed_dictionary.txt"  "$DIRECTORY/runs/nlm_gene"

echo "PREPROCESS CORPORA"

AB3P_PATH="../Ab3P/identify_abbr"
CORPORA="ncbi_disease bc5cdr_disease bc5cdr_chemical nlm_chem linnaeus s800 bioid_cell_line gnormplus nlm_gene medmentions"
SPLITS="train test"

for CORPUS in $CORPORA; do
    for SPLIT in $SPLITS; do

        echo "Preprocess $CORPUS: $SPLIT split"

        INPUT_DIR="${DIRECTORY}/runs/$CORPUS/$SPLIT"
        OUTPUT_DIR="${DIRECTORY}/runs/$CORPUS/processed_$SPLIT"
        DICTIONARY_PATH="${DIRECTORY}/runs/$CORPUS/processed_dictionary.txt"
        mkdir -p "$OUTPUT_DIR"

        python query_preprocess.py \
            --input_dir "${INPUT_DIR}" \
            --output_dir "${OUTPUT_DIR}" \
            --dictionary_path "${DICTIONARY_PATH}"  \
            --ab3p_path "${AB3P_PATH}" \
            --lowercase "true" \
            --remove_punctuation "true"
    done
done
