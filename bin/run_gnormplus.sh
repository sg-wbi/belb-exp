#!/usr/bin/env bash

# stop execution if any command fails
set -e

BELB_DIR="$1"
OUT_DIR="$2"
CORPORA="gnormplus nlm_gene s800 linnaeus"

if [[ -z "$OUT_DIR" ]] || [[ -z "$BELB_DIR" ]] ; then
    echo "Error: please specify the BELB and output directory"
    echo "Usage: run_gnormplus.sh <belb directory> <output directory>"
    exit 1
fi


echo "1. DOWNLOAD"
wget -nc -c -P "$OUT_DIR" https://www.ncbi.nlm.nih.gov/CBBresearch/Lu/Demo/tmTools/download/GNormPlus/GNormPlusJava.zip

if [[ ! -d "$OUT_DIR/GNormPlusJava" ]]; then
    unzip "$OUT_DIR/GNormPlusJava.zip" -d "$OUT_DIR"
fi

# test python is the one of the virtual enviroment w/ BELB installed
if ! python -c "import belb"; then
    echo "Make sure to activate the python virtual enviroment where 'belb' is installed before running this script!"
    exit 1
fi

echo "2. SETUP"
python -m benchmark.sr4gn.sr4gn --run input --in_dir "$OUT_DIR/GNormPlusJava/" --belb_dir "$BELB_DIR"
python -m benchmark.gnormplus.gnormplus --run input --in_dir "$OUT_DIR/GNormPlusJava/" --belb_dir "$BELB_DIR"

cd "$OUT_DIR/GNormPlusJava" || exit 1

if [[ ! -f "installed" ]]; then
    sh Installation.sh || echo "Installing GNormPlus failed! Please consult its README" && exit 1
    chmod +x GNormPlus.sh
    chmod +x Ab3P
    touch installed
fi

if [[ ! -f "setup_nlm_gene.txt" ]]; then
    cp setup.txt setup_nlm_gene.txt
    sed -i 's/GNR.Model/GNR.GNormPlusCorpus_NLMGeneTrain.Model/' setup_nlm_gene.txt
fi

echo "3. RUN"
for CORPUS in $CORPORA; do

    if [[ "$CORPUS" == "nlm_gene" ]]; then
        SETUP="setup_nlm_gene.txt"
    else
        SETUP="setup.txt"
    fi

    if [[ ! -d "./output/$CORPUS" ]]; then
        mkdir -p "./output/$CORPUS"
        echo "RUN: ./input/$CORPUS - $SETUP"
        java -Xmx60G -Xms30G -jar GNormPlus.jar "./input/$CORPUS" "./output/$CORPUS" $SETUP
    fi

done

echo "4. COLLECT RESULTS"
cd - || exit 1
python -m benchmark.sr4gn.sr4gn --run output --in_dir "$OUT_DIR/GNormPlusJava/" --belb "$BELB_DIR"
python -m benchmark.gnormplus.gnormplus --run output --in_dir "$OUT_DIR/GNormPlusJava/" --belb "$BELB_DIR"
