#!/usr/bin/env bash

# stop execution if any command fails
set -e

DIRECTORY=$1

if [[ -z "$DIRECTORY" ]]; then
    echo "Please specify path where BELB data is stored!"
    echo "Usage: create_belb_runs.sh <directory w/ converted belb data>"
    exit 1
fi

cp -r "$DIRECTORY"/kbs/umls/*.json                "$DIRECTORY/runs/medmentions/documents"
cp -r "$DIRECTORY"/kbs/ctd_diseases/*.json        "$DIRECTORY/runs/ncbi_disease/documents"
cp -r "$DIRECTORY"/kbs/ctd_diseases/*.json        "$DIRECTORY/runs/bc5cdr_disease/documents"
cp -r "$DIRECTORY"/kbs/ctd_chemicals/*.json       "$DIRECTORY/runs/bc5cdr_chemical/documents"
cp -r "$DIRECTORY"/kbs/ctd_chemicals/*.json       "$DIRECTORY/runs/nlm_chem/documents"
cp -r "$DIRECTORY"/kbs/ncbi_taxonomy/*.json       "$DIRECTORY/runs/linnaeus/documents"
cp -r "$DIRECTORY"/kbs/ncbi_taxonomy/*.json       "$DIRECTORY/runs/s800/documents"
cp -r "$DIRECTORY"/kbs/cellosaurus/*.json         "$DIRECTORY/runs/bioid_cell_line/documents"
cp -r "$DIRECTORY"/kbs/ncbi_gene/gnormplus/*.json "$DIRECTORY/runs/gnormplus/documents"
cp -r "$DIRECTORY"/kbs/ncbi_gene/nlm_gene/*.json  "$DIRECTORY/runs/nlm_gene/documents"

python preprocess_belb_run.py --dir "$DIRECTORY"
