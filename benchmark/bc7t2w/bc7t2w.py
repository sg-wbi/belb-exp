#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prepare data and collect results for method: BC7 NLM-CHEM track winner
"""
import json
import os
from typing import Optional

import bioc
from belb import AutoBelbCorpus, Entities
from belb.resources import Corpora
from bioc import biocjson

from benchmark.model import NIL, Model
from benchmark.utils import get_argument_parser

# pylint: disable=singleton-comparison

ENTITY_TYPE = "Chemical"
DUMMY_IDENTIFIER = "MESH:C029350"


class BioCreative7Track2Winner(Model):
    """
    Helper to deal w/ BC7T2 input/output
    """

    @property
    def corpora(self):
        """
        Corpora supported by baseline
        """

        return [
            (Corpora.BC5CDR.name, Entities.CHEMICAL),
            (Corpora.NLM_CHEM.name, Entities.CHEMICAL),
        ]

    def create_input(self):
        """
        Prepare input for method
        """

        for corpus_name, entity_type in self.corpora:
            out_dir = os.path.join(self.in_dir, "input", corpus_name)

            os.makedirs(out_dir, exist_ok=True)

            corpus = AutoBelbCorpus.from_name(
                name=corpus_name,
                directory=self.belb_dir,
                entity_type=entity_type,
                sentences=False,
                mention_markers=False,
                add_foreign_annotations=False,
            )

            collection = bioc.BioCCollection()
            for d in corpus.data["test"]:
                bd = d.to_belb()
                for p in bd.passages:
                    for a in p.annotations:
                        a.infons["type"] = ENTITY_TYPE
                        a.infons["identifier"] = DUMMY_IDENTIFIER

                collection.add_document(bd)

                with open(os.path.join(out_dir, f"{self.split}.bioc.json"), "w") as fp:
                    biocjson.dump(collection, fp, ensure_ascii=False)

    def parse_output(
        self, corpus_name: str, gold: dict, entity_type: Optional[str] = None
    ) -> dict:
        """
        Extract annotations from output
        """

        path = os.path.join(
            self.in_dir,
            "outputs",
            "normalizer",
            corpus_name,
            f"BaseCorpus_{self.split}.bioc_wEmbeddings.json",
        )

        assert os.path.exists(path), (
            f"Result file `{path}` does not exists!",
            "Please make sure `run_bc7t2.sh` executed successfully!",
        )

        with open(path) as fp:
            documents = json.load(fp)

        annotations: dict = {}
        for document in documents["documents"]:
            eid = document["id"]
            if eid not in annotations:
                annotations[eid] = {}
            for passage in document["passages"]:
                for a in passage["annotations"]:
                    identifiers = a["infons"]["identifier"].split(",")
                    identifiers = [i if i != "-" else NIL for i in identifiers]

                    for location in a["locations"]:
                        start = location["offset"]
                        end = start + location["length"]

                        offset = (int(start), int(end))

                        if offset not in annotations:
                            annotations[eid][offset] = set()

                        annotations[eid][offset].update(identifiers)

        for eid, offsets in annotations.items():
            for offset in offsets:
                annotations[eid][offset] = [list(annotations[eid][offset])]

        return {"bc7t2w": annotations}


def main():
    """
    Script
    """

    parser = get_argument_parser()
    args = parser.parse_args()

    db_conifg = os.path.join(os.getcwd(), "config", "db.yaml")

    bc7t2w = BioCreative7Track2Winner(
        in_dir=args.in_dir,
        belb_dir=args.belb_dir,
        db_config=db_conifg,
        split=args.split,
        obsolete_kb=True,
        joint_ner_nen=False,
        identifier_mapping=True,
    )

    if args.run == "input":
        bc7t2w.create_input()

    elif args.run == "output":
        results_dir = os.path.join(os.getcwd(), "results")
        os.makedirs(results_dir, exist_ok=True)
        bc7t2w.collect_results(results_dir)


if __name__ == "__main__":
    main()
