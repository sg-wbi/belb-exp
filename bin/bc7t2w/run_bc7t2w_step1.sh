#!/usr/bin/env bash

# stop execution if any command fails
set -e

BELB_DIR="$1"
OUT_DIR="$2"

if [[ -z "$OUT_DIR" ]] || [[ -z "$BELB_DIR" ]] ; then
    echo "Error: please specify the BELB and output directory"
    echo "Usage: run_bc7t2w_step1.sh <belb directory> <output directory>"
    exit 1
fi

echo "1. DOWNLOAD"
wget -nc -c -P "$OUT_DIR" https://github.com/bioinformatics-ua/biocreativeVII_track2/archive/refs/heads/main.zip

if [[ ! -d "$OUT_DIR/biocreativeVII_track2" ]]; then
    unzip "$OUT_DIR/main.zip" -d "$OUT_DIR"
    mv "$OUT_DIR/biocreativeVII_track2-main"  "$OUT_DIR/biocreativeVII_track2"
fi

echo "2. SETUP"
# test python is the one of the virtual enviroment w/ BELB installed
if ! python -c "import belb"; then
    echo "Make sure to activate the python virtual enviroment where 'belb' is installed before running this script!"
    exit 1
fi

python -m benchmark.bc7t2w.bc7t2w --run input --in_dir "$OUT_DIR/biocreativeVII_track2" --belb "$BELB_DIR"

echo "Input data for baseline is ready! Please deactivate the virtual enviroment for this repository"
