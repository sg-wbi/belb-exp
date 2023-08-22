#!/usr/bin/env bash

BELB_DIR="$1"
OUT_DIR="$2"
CORPORA="ncbi_disease bc5cdr"

if [[ -z "$OUT_DIR" ]] || [[ -z "$BELB_DIR" ]] ; then
    echo "Error: please specify the BELB and output directory"
    echo "Usage: run_taggerone.sh <belb directory> <output directory>"
    exit 1
fi

echo "1. DOWNLOAD"
wget -nc -c -P "$OUT_DIR" https://www.ncbi.nlm.nih.gov/research/bionlp/taggerone/TaggerOne-0.2.1.tgz

if [[ ! -d "$OUT_DIR/TaggerOne-0.2.1.tgz" ]]; then
    tar xf "$OUT_DIR/TaggerOne-0.2.1.tgz" -C "$OUT_DIR"
fi


if [[ ! -d "$OUT_DIR/TaggerOne-0.2.1/Ab3P-v1.5" ]]; then
    wget -nc -c -P "$OUT_DIR/TaggerOne-0.2.1" https://ftp.ncbi.nlm.nih.gov/pub/wilbur/Ab3P-v1.5.tar.gz
    tar xf "$OUT_DIR/TaggerOne-0.2.1/Ab3P-v1.5.tar.gz" -C "$OUT_DIR/TaggerOne-0.2.1" --skip-old-files
fi

if [[ ! -f "$OUT_DIR/TaggerOne-0.2.1/Ab3P-v1.5/identify_abbr" ]]; then
    echo "Compiling Ab3P..."
    (cd "$OUT_DIR/TaggerOne-0.2.1/Ab3P-v1.5" && make)
    # check if previous command ran successfully
    if [ $? -eq 0 ]; then
        echo "Complining Ab3P failed! Try again on another machine or find a precompiled version here: https://github.com/dmis-lab/BioSyn/tree/master/Ab3P"
    fi
fi

echo "2. SETUP"
# test python is the one of the virtual enviroment w/ BELB installed
if ! python -c "import belb"; then
    echo "Make sure to activate the python virtual enviroment where 'belb' is installed before running this script!"
    exit 1
fi
python -m benchmark.taggerone.taggerone --run input --in_dir "$OUT_DIR/TaggerOne-0.2.1/" --belb "$BELB_DIR"

cd "$OUT_DIR/TaggerOne-0.2.1" || exit 1

echo "3. RUN"
for CORPUS in $CORPORA; do

    if [[ "$CORPUS" == "ncbi_disease" ]]; then
        MODEL="./output/model_NCBID.bin"
    elif [[ "$CORPUS" == "bc5cdr" ]]; then
        MODEL="./output/model_BC5CDRD.bin"
    fi

    if [[ ! -d "./output/$CORPUS" ]]; then
        mkdir -p "output/$CORPUS"
        ./ProcessText.sh "Bioc" "$MODEL" "./input/$CORPUS/$CORPUS.test.bioc" "./output/$CORPUS/$CORPUS.test.bioc"
    fi

done


echo "4. COLLECT RESULTS"
cd - || exit 1
python -m benchmark.taggerone.taggerone --run output --in_dir "$OUT_DIR/TaggerOne-0.2.1/" --belb "$BELB_DIR"
