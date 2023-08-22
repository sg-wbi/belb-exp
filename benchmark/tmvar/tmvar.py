#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tmvar
"""

import os
from typing import Optional

from belb.preprocessing.data import Entities
from belb.resources import Corpora
from bioc import biocxml

from benchmark.model import NIL, Model
from benchmark.utils import get_argument_parser


class TmVar(Model):
    """
    Helper to deal w/ tmVar (v3) input/output
    """

    @property
    def corpora(self):
        """
        Corpora supported by baseline
        """

        return [
            (Corpora.SNP.name, Entities.VARIANT),
            (Corpora.OSIRIS.name, Entities.VARIANT),
            (Corpora.TMVAR.name, Entities.VARIANT),
        ]

    def create_input(self):
        """
        Create input for tmVar (v3)
        """

        for corpus_name, entity_type in self.corpora:
            path = os.path.join(self.in_dir, "tmvar_input", corpus_name)
            if not os.path.exists(path):
                os.makedirs(path, exist_ok=True)
                self.save_converted_documents(
                    name=corpus_name,
                    entity_type=entity_type,
                    path=path,
                    files=False,
                    output_format="bioc",
                )

    def parse_identifiers(self, identifiers: str) -> list[str]:
        """Expand identifiers to list, apply mapping"""

        ids = [i for i in identifiers.split(";") if i.lower().startswith("rs")]
        if len(ids) > 0:
            annotation_ids = [ids[0].replace("RS#:", "").replace("rs", "")]
        else:
            annotation_ids = [NIL]

        return annotation_ids

    def parse_output(
        self, corpus_name: str, gold: dict, entity_type: Optional[str] = None
    ) -> dict:
        """
        Load BioC file w/ tmVar predictions
        """

        path = os.path.join(
            self.in_dir,
            "output",
            f"{corpus_name}.{self.split}.bioc.BioC.XML",
        )

        assert os.path.exists(path), (
            f"Result file `{path}` does not exists!",
            "Please make sure `run_tmvar.sh` ran successfully!",
        )

        with open(path) as fp:
            collection = biocxml.load(fp)

        annotations: dict = {}
        for d in collection.documents:
            if d.id not in annotations:
                annotations[d.id] = {}
            for p in d.passages:
                for a in p.annotations:
                    identifiers = a.infons.get("Identifier")
                    if identifiers is not None:
                        identifiers = [self.parse_identifiers(a.infons["Identifier"])]
                    else:
                        identifiers = [[NIL]]
                    for loc in a.locations:
                        offset = (loc.offset, loc.end)
                        annotations[d.id][offset] = identifiers

        return {"tmvar": annotations}


def main():
    """
    Script
    """

    parser = get_argument_parser()
    args = parser.parse_args()

    db_conifg = os.path.join(os.getcwd(), "config", "db.yaml")

    tmvar = TmVar(
        in_dir=args.in_dir,
        belb_dir=args.belb_dir,
        db_config=db_conifg,
        split=args.split,
        joint_ner_nen=True,
        obsolete_kb=True,
    )

    if args.run == "input":
        tmvar.create_input()

    elif args.run == "output":
        results_dir = os.path.join(os.getcwd(), "results")
        os.makedirs(results_dir, exist_ok=True)
        tmvar.collect_results(results_dir)


if __name__ == "__main__":
    main()
