#!/usr/bin/env bash

# stop execution if any command fails
set -e

BELB_DIR="$1"
OUT_DIR="$2"
CORPORA="s800 linnaeus"

if [[ -z "$OUT_DIR" ]] || [[ -z "$BELB_DIR" ]] ; then
    echo "Error: please specify the BELB and output directory"
    echo "Usage: run_linnaeus.sh <belb directory> <output directory>"
    exit 1
fi

echo "1. DOWNLOAD"
wget -nc -c -P "$OUT_DIR" https://sourceforge.net/projects/linnaeus/files/Linnaeus/linnaeus-2.0.tar.gz
wget -nc -c -P "$OUT_DIR" https://sourceforge.net/projects/linnaeus/files/Entity_packs/species-1.2.tar.gz
wget -nc -c -P "$OUT_DIR" https://sourceforge.net/projects/linnaeus/files/Entity_packs/species-proxy-1.2.tar.gz
wget -nc -c -P "$OUT_DIR" https://sourceforge.net/projects/linnaeus/files/Entity_packs/genera-species-proxy-1.0.tar.gz
wget -nc -c -P "$OUT_DIR" https://sourceforge.net/projects/linnaeus/files/Corpora/manual-corpus-species-1.0.tar.gz

tar xf "$OUT_DIR/linnaeus-2.0.tar.gz" -C "$OUT_DIR" --skip-old-files
tar xf "$OUT_DIR/species-1.2.tar.gz" -C "$OUT_DIR" --skip-old-files
tar xf "$OUT_DIR/species-proxy-1.2.tar.gz" -C "$OUT_DIR" --skip-old-files
tar xf "$OUT_DIR/genera-species-proxy-1.0.tar.gz" -C "$OUT_DIR" --skip-old-files
tar xf "$OUT_DIR/manual-corpus-species-1.0.tar.gz" -C "$OUT_DIR" --skip-old-files

echo "2. PREPARE INPUT"
# test python is the one of the virtual enviroment w/ BELB installed
if ! python -c "import belb"; then
    echo "Make sure to activate the python virtual enviroment where 'belb' is installed before running this script!"
    exit 1
fi

python -m benchmark.linnaeus.linnaeus --run input --in_dir "$OUT_DIR" --belb_dir "$BELB_DIR"


echo "3. SETUP"
mkdir -p "$OUT_DIR/properties"
cp "$OUT_DIR/genera-species-proxy/dict-genera-proxy.tsv" "$OUT_DIR/species"
cp "$OUT_DIR/species-proxy/dict-species-proxy.tsv" "$OUT_DIR/species"
if [[ ! -f "$OUT_DIR/properties/base.conf" ]]; then

    cat > "$OUT_DIR/properties/base.conf" <<EOL
#EDIT THIS LINE, SET TO PATH OF SPECIES PACK (WITHOUT TRAILING SLASH):
\$dir = $OUT_DIR/species

variantMatcher=\$dir/dict-species.tsv;

ppStopTerms=\$dir/stoplist.tsv
ppAcrProbs=\$dir/synonyms-acronyms.tsv
ppSpeciesFreqs=\$dir/species-frequency.tsv

postProcessing
EOL

    sed "s/variantMatcher=.*$/variantMatcher=\$dir\/dict-species.tsv;\$dir\/dict-genera-proxy.tsv/g" "$OUT_DIR/properties/base.conf" > "$OUT_DIR/properties/genera_proxy.conf"
    sed "s/variantMatcher=.*$/variantMatcher=\$dir\/dict-species.tsv;\$dir\/dict-species-proxy.tsv/g" "$OUT_DIR/properties/base.conf" > "$OUT_DIR/properties/species_proxy.conf"
    sed "s/variantMatcher=.*$/variantMatcher=\$dir\/dict-species.tsv;\$dir\/dict-genera-proxy.tsv;\$dir\/dict-species-proxy.tsv/g" "$OUT_DIR/properties/base.conf" >  "$OUT_DIR/properties/genera_species_proxy.conf"

fi

echo "4. RUN"
PROPERTIES="base genera_proxy species_proxy genera_species_proxy"
for CORPUS in $CORPORA; do

    TEXT_DIR="$OUT_DIR/input/$CORPUS"

    # if [[ "$CORPUS" == "s800" ]]; then
    #     TEXT_DIR="$OUT_DIR/input/$CORPUS"
    # else
    #     TEXT_DIR="$OUT_DIR/manual-corpus-species-1.0/txt"
    # fi

    for PROPERTY in $PROPERTIES; do
        if [[ ! -d "$OUT_DIR/output/$CORPUS/$PROPERTY" ]]; then

            mkdir -p "$OUT_DIR/output/$CORPUS/$PROPERTY"


            java -jar "$OUT_DIR/linnaeus/bin/linnaeus-2.0.jar" \
                --properties "$OUT_DIR/properties/$PROPERTY.conf" \
                --textDir "$TEXT_DIR" \
                --out "$OUT_DIR/output/$CORPUS/$PROPERTY/mentions.tsv"

        fi
    done
done

echo "4. COLLECT RESULTS"
python -m benchmark.linnaeus.linnaeus --run output --in_dir "$OUT_DIR" --belb_dir "$BELB_DIR"
