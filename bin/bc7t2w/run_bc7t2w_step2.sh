#!/usr/bin/env bash

# stop execution if any command fails
set -e


BELB_DIR="$1"
OUT_DIR="$2"
CORPORA="bc5cdr nlm_chem"

if [[ -z "$OUT_DIR" ]] || [[ -z "$BELB_DIR" ]] ; then
    echo "Error: please specify the BELB and output directory"
    echo "Usage: run_bc7t2w_step2.sh <belb directory> <output directory>"
    exit 1
fi

if ! command -v conda &> /dev/null
then
    echo "You need to install conda for this step"
    exit 1
fi

echo "WARNING: Make sure you have activated the conda enviroment required by this tool!"

cd "$OUT_DIR"

# conda create --name biocreative python=3.6.9
# conda activate biocreative
# pip install -r requirements.txt

if [[ -z "$CUDA_VISIBLE_DEVICES" ]]; then
    echo "You may want to limit the number of GPUs to use: 'CUDA_VISIBLE_DEVICES=0 run_bc7t2w_step2.sh'"
fi


cd "$OUT_DIR/biocreativeVII_track2" || exit 1

if [[ ! -f "$OUT_DIR/biocreativeVII_track2/installed" ]]; then
    chmod +x ./setup.sh
    ./setup.sh || "Setup failed! Likely due to compilation of A3bP: please compile it beferehand and place it in tools/A3bP" && exit 1
    touch "installed"
fi


echo "3. RUN"

# conda activate biocrative

for CORPUS in $CORPORA; do
    if [[ ! -d "outputs/normalizer/$CORPUS" ]]; then
        mkdir -p "outputs/normalizer/$CORPUS"
        python src/main.py "./input/$CORPUS" --normalizer  --normalizer.write_path "outputs/normalizer/$CORPUS"
    fi
done

# conda deactivate biocrative
#

cd -
