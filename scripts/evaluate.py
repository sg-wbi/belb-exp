#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Get results from entity-specific rule-based systems
"""

import argparse
import os
import random

import pandas as pd
from belb import (ENTITY_TO_CORPORA_NAMES, ENTITY_TO_KB_NAME, AutoBelbCorpus,
                  AutoBelbKb, BelbKb, Entities, Splits)
from belb.resources import Corpora
from belb.utils import load_stratified, load_zeroshot

from benchmark.model import CORPORA_MULTI_ENTITY_TYPES, NIL
from benchmark.utils import load_json

EVAL_MODES = ["std", "strict", "lenient"]


CORPORA = [
    # (Corpora.GNORMPLUS.name, Entities.GENE),
    # (Corpora.NLM_GENE.name, Entities.GENE),
    (Corpora.NCBI_DISEASE.name, Entities.DISEASE),
    # (Corpora.BC5CDR.name, Entities.DISEASE),
    # (Corpora.BC5CDR.name, Entities.CHEMICAL),
    # (Corpora.NLM_CHEM.name, Entities.CHEMICAL),
    # (Corpora.LINNAEUS.name, Entities.SPECIES),
    # (Corpora.S800.name, Entities.SPECIES),
    # (Corpora.BIOID.name, Entities.CELL_LINE),
    # (Corpora.MEDMENTIONS.name, Entities.UMLS),
    # (Corpora.SNP.name, Entities.VARIANT),
    # (Corpora.OSIRIS.name, Entities.VARIANT),
    # (Corpora.TMVAR.name, Entities.VARIANT),
]

ENTITY_TYPE_STRING_IDENTIFIERS = [
    Entities.DISEASE,
    Entities.CHEMICAL,
    Entities.CELL_LINE,
    Entities.UMLS,
]

CORPUS_TO_RBES = {
    "gnormplus": "gnormplus",
    "nlm_gene": "gnormplus",
    "linnaeus": "sr4gn",
    "s800": "sr4gn",
    "ncbi_disease": "taggerone",
    "bc5cdr_disease": "taggerone",
    "bc5cdr_chemical": "bc7t2w",
    "nlm_chem": "bc7t2w",
    "medmentions": "scispacy",
    "snp": "tmvar",
    "osiris": "tmvar",
    "tmvar": "tmvar",
    "bioid_cell_line": "fuzzysearch",
}

RBES_JOINT = ["gnormplus", "taggerone", "tmvar", "sr4gn"]


def parse_args():
    parser = argparse.ArgumentParser(description="Generate results")
    parser.add_argument(
        "--belb_dir",
        type=str,
        required=True,
        help="Directory where all BELB data is stored",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=1,
        help="Ranks to consider in prediction",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=tuple(EVAL_MODES),
        default="std",
        help="If multiple predictions are return consider it wrong",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Do not include joint ner-nen models in comparison (full test set)",
    )
    return parser.parse_args()


def multi_label_recall(gold: dict, pred: dict, k: int = 1, mode: str = "std") -> float:
    hits = 0

    for h, y_true in gold.items():
        # int
        y_true = set(int(y) for y in y_true)

        # get topk predictions
        y_pred = [list(set(yp)) for yp in pred[h][:k]]

        if mode in ["std", "strict"]:
            # get single prediction
            if mode == "strict":
                # in strict mode default wrong if multiple predictions
                y_pred = [NIL if len(y) > 1 else y[0] for y in y_pred]
            elif mode == "std":
                # sample if multiple predictions
                y_pred = [random.sample(y, 1)[0] for y in y_pred]

            # go over k predicitons
            for y in y_pred:
                # int
                y = -1 if y == NIL else int(y)
                if y in y_true:
                    hits += 1
                    # if you get a hit stop
                    break
        else:
            for ys in y_pred:
                ys = [-1 if y == NIL else int(y) for y in ys]
                if any(y in y_true for y in ys):
                    hits += 1
                    # if you get a hit stop
                    break

    return round(hits / len(gold), 2)


def get_integer_identifiers(kb: BelbKb, gold: dict) -> dict:
    identifiers = set(i for ids in gold.values() for i in ids)

    with kb as handle:
        map = handle.get_identifier_mapping(identifiers)

    gold = {h: set([map[i] for i in ids]) for h, ids in gold.items()}

    return gold


def load_gold(
    corpus_name: str,
    entity_type: str,
    belb_dir: str,
    db_config,
) -> dict:
    corpus = AutoBelbCorpus.from_name(
        name=corpus_name,
        directory=belb_dir,
        entity_type=entity_type,
        sentences=False,
        mention_markers=False,
        add_foreign_annotations=False,
    )

    gold = {
        a.infons["hexdigest"]: a.identifiers
        for e in corpus[Splits.TEST]
        for p in e.passages
        for a in p.annotations
    }

    if entity_type in ENTITY_TYPE_STRING_IDENTIFIERS:
        kb = AutoBelbKb.from_name(
            directory=belb_dir,
            name=ENTITY_TO_KB_NAME[entity_type],
            db_config=db_config,
            debug=False,
        )

        gold = get_integer_identifiers(kb=kb, gold=gold)

    return gold


def filter_gold(gold: dict, directory: str, corpus_name: str) -> dict:
    ner_tp = None
    if corpus_name in CORPUS_TO_RBES:
        rbes_pred_path = os.path.join(
            directory,
            CORPUS_TO_RBES[corpus_name],
            "filtered_predictions.json",
        )
        if os.path.exists(rbes_pred_path):
            rbes_pred = load_json(rbes_pred_path)
            ner_tp = set(p["hexdigest"] for p in rbes_pred)

    if ner_tp is not None:
        gold = {h: y for h, y in gold.items() if h in ner_tp}

    return gold


def get_results_by_corpus(
    gold: dict, preds: dict, mode: str = "std", k: int = 1
) -> pd.DataFrame:
    # TODO: add document level
    data: dict = {}

    for corpus, corpus_gold in gold.items():
        for model, corpora_pred in preds.items():
            corpus_pred = corpora_pred.get(corpus, [])

            if len(corpus_pred) == 0:
                continue

            if model not in data:
                data[model] = {}

            data[model][corpus] = multi_label_recall(
                gold=corpus_gold, pred=corpus_pred, mode=mode, k=k
            )

    return pd.DataFrame(data)


def get_results_by_entity(
    gold: dict, preds: dict, mode: str = "std", k: int = 1
) -> pd.DataFrame:
    data: dict = {}
    for entity, corpora in ENTITY_TO_CORPORA_NAMES.items():
        corpora = [
            f"{c}_{entity}" if c in CORPORA_MULTI_ENTITY_TYPES else c for c in corpora
        ]

        entity_gold = {
            h: y
            for corpus, corpus_gold in gold.items()
            for h, y in corpus_gold.items()
            if corpus in corpora
        }

        for model, corpora_pred in preds.items():
            entity_pred = {
                h: y
                for corpus, corpus_pred in corpora_pred.items()
                for h, y in corpus_pred.items()
                if corpus in corpora
            }

            if len(entity_pred) == 0:
                continue

            if model not in data:
                data[model] = {}

            data[model][entity] = multi_label_recall(
                gold=entity_gold, pred=entity_pred, mode=mode, k=k
            )

    return pd.DataFrame(data)


def get_results_by_subset(
    gold: dict, preds: dict, mode: str = "std", k: int = 1
) -> dict:
    subsets = {
        "zeroshot": load_zeroshot(),
        "stratified": load_stratified(),
        # "homonyms": load_homonyms(),
    }

    out = {}

    for subset_name, subset_df in subsets.items():
        data: dict = {}
        for entity_type in Entities:
            subset = set(
                subset_df[subset_df["entity_type"] == entity_type]["hexdigest"]
            )
            if len(subset) == 0:
                continue

            subset_gold = {
                h: y
                for name, corpus_gold in gold.items()
                for h, y in corpus_gold.items()
                if h in subset
            }

            for model, corpora_pred in preds.items():
                subset_pred = {
                    h: y
                    for _, corpus_pred in corpora_pred.items()
                    for h, y in corpus_pred.items()
                    if h in subset_gold
                }

                if len(subset_gold) != len(subset_pred):
                    continue

                if entity_type not in data:
                    data[entity_type] = {}

                data[entity_type][model] = multi_label_recall(
                    gold=subset_gold, pred=subset_pred, mode=mode, k=k
                )

        out[subset_name] = pd.DataFrame(data)

    return out


def main():
    args = parse_args()

    results_dir = os.path.join(os.getcwd(), "results")

    DB_CONFIG = os.path.join(os.getcwd(), "config", "db.yaml")

    gold = {}
    preds = {}
    for corpus_name, entity_type in CORPORA:
        full_corpus_name = (
            f"{corpus_name}_{entity_type}"
            if corpus_name in CORPORA_MULTI_ENTITY_TYPES
            else corpus_name
        )

        corpus_gold = load_gold(
            corpus_name=corpus_name,
            entity_type=entity_type,
            belb_dir=args.belb_dir,
            db_config=DB_CONFIG,
        )

        corpus_dir = os.path.join(results_dir, full_corpus_name)

        if not args.full:
            corpus_gold = filter_gold(
                gold=corpus_gold,
                directory=corpus_dir,
                corpus_name=full_corpus_name,
            )

        for model in os.listdir(corpus_dir):
            model_name = "rbes" if model in set(CORPUS_TO_RBES.values()) else model

            if args.full and model_name == "rbes":
                continue

            pred = {
                p["hexdigest"]: p["y_pred"]
                for p in load_json(os.path.join(corpus_dir, model, "predictions.json"))
            }
            pred = {h: y for h, y in pred.items() if h in corpus_gold}

            if model_name not in preds:
                preds[model_name] = {}

            preds[model_name][full_corpus_name] = pred
            gold[full_corpus_name] = corpus_gold

    print("CORPORA:")
    corpora_df = get_results_by_corpus(gold=gold, preds=preds, mode=args.mode, k=args.k)
    print(corpora_df)
    print("\n")
    #
    # print("ENTITIES:")
    # entity_df = get_results_by_entity(
    #     gold=gold, preds=preds, strict=args.strict, k=args.k
    # )
    # print(entity_df)
    # print("\n")
    #
    # print("SUBSETS")
    # subsets_results = get_results_by_subset(
    #     gold=gold, preds=preds, strict=args.strict, k=args.k
    # )
    #
    # for name, subset_df in subsets_results.items():
    #     print(f"\t{name.upper()}")
    #     print(subset_df)

    # print(subsets_df)
    # print("\n")
    #
    # for result, df in [
    #     ("corpora", corpora_df),
    #     ("entity", entity_df),
    #     ("subsets", subsets_df),
    # ]:
    #     df.to_csv(
    #         os.path.join(
    #             os.getcwd(),
    #             "results",
    #             "tables",
    #             f"{result}_k{args.k}_strict{int(args.strict)}_full{int(args.full)}.tsv",
    #         ),
    #     )
    #


if __name__ == "__main__":
    main()
