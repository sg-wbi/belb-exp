#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TaggerOne
"""

import json
import os
from typing import Optional

from belb.preprocessing.data import Entities
from belb.resources import Corpora
from bioc import biocxml

from benchmark.model import NIL, Model
from benchmark.utils import get_argument_parser


class TaggerOne(Model):
    """
    Helper to deal w/ GNormPlus input/output
    """

    @property
    def corpora(self):
        """
        Corpora supported by baseline
        """

        return [
            (Corpora.NCBI_DISEASE.name, Entities.DISEASE),
            (Corpora.BC5CDR.name, Entities.DISEASE),
        ]

    def create_input(self):
        """
        Create input for GNormPlus
        """

        for corpus_name, entity_type in self.corpora:
            path = os.path.join(self.in_dir, "input", corpus_name)
            if not os.path.exists(path):
                os.makedirs(path, exist_ok=True)
                self.save_converted_documents(
                    name=corpus_name,
                    path=path,
                    files=False,
                    output_format="bioc",
                    entity_type=entity_type,
                )

    def parse_output(
        self, corpus_name: str, gold: dict, entity_type: Optional[str] = None
    ) -> dict:
        """
        Load BioC file w/ GNormPlus predictions
        """

        path = os.path.join(
            self.in_dir, "output", corpus_name, f"{corpus_name}.{self.split}.bioc"
        )
        assert os.path.exists(path), (
            f"Result file `{path}` does not exists!",
            "Please make sure `run_taggerone.sh` ran successfully!",
        )

        with open(path) as fp:
            collection = biocxml.load(fp)

        annotations: dict = {}
        for d in collection.documents:
            if d.id not in annotations:
                annotations[d.id] = {}
            for p in d.passages:
                for a in p.annotations:
                    if a.infons["type"].lower() != entity_type:
                        continue

                    identifiers = a.infons.get("identifier")

                    if identifiers is not None:
                        identifiers = identifiers.split()
                    else:
                        identifiers = [NIL]

                    for loc in a.locations:
                        offset = (loc.offset, loc.end)
                        annotations[d.id][offset] = [identifiers]

        return {"taggerone": annotations}


def main():
    """
    Script
    """

    parser = get_argument_parser()
    args = parser.parse_args()

    db_conifg = os.path.join(os.getcwd(), "config", "db.yaml")

    taggerone = TaggerOne(
        in_dir=args.in_dir,
        belb_dir=args.belb_dir,
        db_config=db_conifg,
        split=args.split,
        joint_ner_nen=True,
        obsolete_kb=True,
        identifier_mapping=True,
    )

    if args.run == "input":
        taggerone.create_input()

    elif args.run == "output":
        results_dir = os.path.join(os.getcwd(), "results")
        os.makedirs(results_dir, exist_ok=True)
        taggerone.collect_results(results_dir)


if __name__ == "__main__":
    main()
