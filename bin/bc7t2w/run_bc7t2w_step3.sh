#!/usr/bin/env bash

# stop execution if any command fails
set -e

BELB_DIR="$1"
OUT_DIR="$2"

if [[ -z "$OUT_DIR" ]] || [[ -z "$BELB_DIR" ]] ; then
    echo "Error: please specify the BELB and output directory"
    echo "Usage: run_bc7t2_step3.sh <belb directory> <output directory>"
    exit 1
fi

# test python is the one of the virtual enviroment w/ BELB installed
if ! python -c "import belb" &> /dev/null; then
    echo "Make sure to activate the python virtual enviroment where 'belb' is installed before running this script!"
    exit 1
fi

echo "5. COLLECT RESULTS"
python -m benchmark.bc7t2w.bc7t2w --run output --in_dir "$OUT_DIR/biocreativeVII_track2" --belb "$BELB_DIR"
