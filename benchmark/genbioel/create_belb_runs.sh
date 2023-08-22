#!/usr/bin/env bash

# stop execution if any command fails
set -e

DIRECTORY=$1

if [[ -z "$DIRECTORY" ]]; then
    echo "Please specify path where BELB data is stored!"
    echo "Usage: create_belb_runs.sh <directory w/ converted belb data>"
    exit 1
fi

cp -r "$DIRECTORY"/kbs/ctd_diseases/*  "$DIRECTORY/runs/ncbi_disease/"
cp -r "$DIRECTORY"/kbs/ctd_diseases/*  "$DIRECTORY/runs/bc5cdr_disease/"
cp -r "$DIRECTORY"/kbs/ctd_chemicals/*  "$DIRECTORY/runs/bc5cdr_chemical/"
cp -r "$DIRECTORY"/kbs/ctd_chemicals/*  "$DIRECTORY/runs/nlm_chem/"
cp -r "$DIRECTORY"/kbs/ncbi_taxonomy/*  "$DIRECTORY/runs/linnaeus/"
cp -r "$DIRECTORY"/kbs/ncbi_taxonomy/*  "$DIRECTORY/runs/s800/"
cp -r "$DIRECTORY"/kbs/cellosaurus/*  "$DIRECTORY/runs/bioid_cell_line/"
cp -r "$DIRECTORY"/kbs/umls/*  "$DIRECTORY/runs/medmentions/"
cp -r "$DIRECTORY"/kbs/ncbi_gene/gnormplus/*  "$DIRECTORY/runs/gnormplus/"
cp -r "$DIRECTORY"/kbs/ncbi_gene/nlm_gene/*  "$DIRECTORY/runs/nlm_gene/"
