#!/usr/bin/env bash

# stop execution if any command fails
set -e

BELB_DIR="$1"
GNORMPLUS_DIR="$2"
OUT_DIR="$3"
CORPORA="gnromplus"

if [[ -z "$OUT_DIR" ]] || [[ -z "$BELB_DIR" ]] ; then
    echo "Error: please specify the BELB, GNormPlus and output directory"
    echo "Usage: run_tmvar.sh <belb directory> <gnormplus directory> <output directory>"
    exit 1
fi

echo "1. DOWNLOAD"
wget -nc -c -P "$OUT_DIR" ftp://ftp.ncbi.nlm.nih.gov/pub/lu/tmVar3/tmVar3.tar.gz

if [[ ! -d "$OUT_DIR/tmVar3" ]]; then
    tar xf "$OUT_DIR/tmVar3.tar.gz" -C "$OUT_DIR"
fi

# test python is the one of the virtual enviroment w/ BELB installed
if ! python -c "import belb"; then
    echo "Make sure to activate the python virtual enviroment where 'belb' is installed before running this script!"
    exit 1
fi

echo "2. SETUP"
python -m baselines.tmvar --run input --dir "$OUT_DIR/tmVar3" --belb "$BELB_DIR"

if [[ ! -f "$OUT_DIR/tmVar3/installed" ]]; then

    cd "$OUT_DIR/tmVar3" || exit 1
    chmod +x ./Installation.sh
    ./Installation.sh || echo "Installing tmVar3 failed! Please consult its README" && exit 1
    chmod +x ./
    touch installed
    cd - || exit 1

fi

cd "$GNORMPLUS_DIR" || exit 1

if [[ ! -f "installed" ]]; then
    echo "Please call run_gnormplus.sh first!" && exit 1
fi

for CORPUS in $CORPORA; do

    GNORMPLUS_INPUT="$OUT_DIR/tmVar3/input/$CORPUS/"
    GNORMPLUS_OUTPUT="$OUT_DIR/tmVar3/input_gene/$CORPUS/"

    if [[ ! -d "$GNORMPLUS_OUTPUT" ]]; then
        mkdir -p "$GNORMPLUS_OUTPUT"
        java -Xmx60G -Xms30G -jar GNormPlus.jar  "$GNORMPLUS_INPUT" "$GNORMPLUS_OUTPUT" setup_nlm_gene.txt
    fi

done

echo "3. RUN"

cd "$OUT_DIR/tmVar3" || exit 1
for CORPUS in $CORPORA; do

    if [[ ! -f "$OUT_DIR/tmVar3/output/$CORPUS" ]]; then
        mkdir -p "$OUT_DIR/tmVar3/output/$CORPUS"
        java -Xmx5G -Xms5G -jar tmVar.jar "$OUT_DIR/tmVar3/input_gene/$CORPUS/" "$OUT_DIR/tmVar3/output/$CORPUS"
    fi

done
cd - || exit 1

# echo "4. COLLECT RESULTS"
# python -m baselines.gnormplus --run output --dir "$OUT_DIR/GNormPlusJava/" --belb "$BELB_DIR"
