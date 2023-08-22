#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GNormPlus
"""

import os
from typing import Optional

from belb.preprocessing.data import Entities
from belb.resources import Corpora
from bioc import biocxml

from benchmark.model import NIL, Model
from benchmark.utils import get_argument_parser

GNORMPLUS_ENTITY_TYPES = [Entities.GENE, Entities.SPECIES]


class GNormPlus(Model):
    """
    Helper to deal w/ GNormPlus input/output
    """

    @property
    def corpora(self):
        """
        Corpora supported by baseline
        """

        corpora = [
            (Corpora.GNORMPLUS.name, Entities.GENE),
            (Corpora.NLM_GENE.name, Entities.GENE),
        ]
        return corpora

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
                    entity_type=entity_type,
                    files=False,
                    output_format="bioc",
                )

    def parse_output(
        self, corpus_name: str, gold: dict, entity_type: Optional[str] = None
    ) -> dict:
        """
        Load BioC file w/ GNormPlus predictions
        """

        path = os.path.join(
            self.in_dir,
            "output",
            corpus_name,
            f"{corpus_name}.{self.split}.bioc",
        )
        assert os.path.exists(path), (
            f"Result file `{path}` does not exists!",
            "Please make sure `run_gnormplus.sh` ran successfully!",
        )

        assert entity_type in GNORMPLUS_ENTITY_TYPES, (
            f"Gnormplus can only predict {GNORMPLUS_ENTITY_TYPES}.",
            f"Asked for {entity_type}",
        )

        with open(path) as fp:
            collection = biocxml.load(fp)

        annotations: dict = {}
        db_name = "NCBI Gene" if entity_type == Entities.GENE else "NCBI Taxonomy"
        for d in collection.documents:
            if d.id not in annotations:
                annotations[d.id] = {}
            for p in d.passages:
                for a in p.annotations:
                    if a.infons["type"].lower() != entity_type:
                        continue

                    identifiers = a.infons.get(db_name)

                    identifiers = (
                        identifiers.split(";") if identifiers is not None else [NIL]
                    )
                    identifiers = [NIL if i == "-" else i for i in identifiers]

                    for loc in a.locations:
                        offset = (loc.offset, loc.end)
                        annotations[d.id][offset] = [identifiers]

        return {"gnormplus": annotations}


def main():
    """
    Script
    """

    parser = get_argument_parser()

    args = parser.parse_args()

    db_conifg = os.path.join(os.getcwd(), "config", "db.yaml")

    gnormplus = GNormPlus(
        in_dir=args.in_dir,
        belb_dir=args.belb_dir,
        db_config=db_conifg,
        split=args.split,
        obsolete_kb=True,
        joint_ner_nen=True,
    )

    if args.run == "input":
        gnormplus.create_input()

    elif args.run == "output":
        results_dir = os.path.join(os.getcwd(), "results")
        os.makedirs(results_dir, exist_ok=True)
        gnormplus.collect_results(results_dir)


if __name__ == "__main__":
    main()
