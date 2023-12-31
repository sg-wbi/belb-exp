#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert output of BelBERT to format for evaluation
"""

import json
import os

from belb import Entities
from belb.resources import Corpora
from utils import load_json

from models.base import CORPORA_MULTI_ENTITY_TYPES, Model, get_argument_parser


class BelBERT(Model):
    """
    Helper to deal w/ arboEL input/output
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert (
            self.out_dir is not None
        ), "ArboEL has different input and output directories. Please pass `out_dir`"

    @property
    def corpora(self):
        """
        Corpora supported
        """

        return [
            (Corpora.GNORMPLUS.name, Entities.GENE),
            (Corpora.NLM_GENE.name, Entities.GENE),
            (Corpora.NCBI_DISEASE.name, Entities.DISEASE),
            (Corpora.BC5CDR.name, Entities.DISEASE),
            (Corpora.BC5CDR.name, Entities.CHEMICAL),
            (Corpora.NLM_CHEM.name, Entities.CHEMICAL),
            (Corpora.LINNAEUS.name, Entities.SPECIES),
            (Corpora.S800.name, Entities.SPECIES),
            (Corpora.BIOID.name, Entities.CELL_LINE),
            # (Corpora.MEDMENTIONS.name, Entities.UMLS),
        ]

    def create_input(self):
        """
        Works directly w/ BELB
        """

    def parse_output(self, corpus_name, entity_type, gold):
        """
        Parse output generated by BelBERT
        """

        path = os.path.join(
            self.out_dir,
            "results",
            f"{corpus_name}.test.{entity_type}",
            f"{corpus_name}_{entity_type}" if corpus_name == "bc5cdr" else corpus_name,
            "results.json",
        )

        predictions = load_json(path)

        pred: dict = {}
        for did, offset_to_data in gold.items():
            if did not in pred:
                pred[did] = {}
            for offset, data in offset_to_data.items():
                if offset not in pred[did]:
                    hexdigest = data["hexdigest"]
                    pred[did][offset] = [predictions[hexdigest]]

        return {"base": pred}


def main():
    """
    Script
    """

    parser = get_argument_parser()

    args = parser.parse_args()

    db_conifg = os.path.join(os.getcwd(), "data", "config", "db.yaml")

    belbert = BelBERT(
        in_dir=args.in_dir,
        out_dir=args.out_dir,
        belb_dir=args.belb_dir,
        db_config=db_conifg,
        joint_ner_nen=False,
        obsolete_kb=False,
        identifier_mapping=True,
    )

    if args.run == "input":
        belbert.create_input()

    elif args.run == "output":
        results_dir = os.path.join(os.getcwd(), "data", "baselines", "belbert")
        os.makedirs(results_dir, exist_ok=True)
        results = belbert.collect_results()
        with open(os.path.join(results_dir, "results.json"), "w") as fp:
            json.dump(results, fp)


if __name__ == "__main__":
    main()
