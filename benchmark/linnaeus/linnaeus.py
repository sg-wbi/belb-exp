#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Linnaues
"""
import json
import os
from typing import Optional

from belb.preprocessing.data import Entities
from belb.resources import Corpora

from benchmark.model import Model
from benchmark.utils import get_argument_parser


def extract_identifier(identifier: str):
    """
    species:ncbi:12305?0.03296277145811789
    """

    return identifier.replace("species:ncbi:", "").split("?")[0]


class Linnaeus(Model):
    """
    Helper to deal w/ Linnaeus input/output
    """

    @property
    def corpora(self):
        return [
            (Corpora.S800.name, Entities.SPECIES),
            (Corpora.LINNAEUS.name, Entities.SPECIES),
        ]

    def create_input(self):
        """
        Create input for Linnaeus
        """

        for corpus_name, _ in self.corpora:
            path = os.path.join(self.in_dir, "input", corpus_name)
            if not os.path.exists(path):
                os.makedirs(path, exist_ok=True)
                self.save_converted_documents(
                    name=corpus_name,
                    path=path,
                    files=True,
                    output_format="txt",
                )

    def parse_output(
        self, corpus_name: str, gold: dict, entity_type: Optional[str] = None
    ) -> dict:
        """
        Load linnaeus predictions
        """

        out: dict = {}
        for config in ["base", "genera_proxy", "species_proxy", "genera_species_proxy"]:
            path = os.path.join(
                self.in_dir, "output", corpus_name, config, "mentions.tsv"
            )
            assert os.path.exists(path), (
                f"Result file `{path}` does not exists!",
                "Please make sure `run_linnaeus.sh` executed successfully!",
            )
            annotations: dict = {}
            with open(path) as fp:
                for idx, line in enumerate(fp):
                    if idx == 0:
                        continue
                    items = line.strip().split("\t")

                    # species:ncbi:12305?0.03296277145811789|species:ncbi:36410?3.8779731127197516E-4
                    identifiers = [extract_identifier(i) for i in items[0].split("|")]

                    parsed_identifiers = []
                    for i in identifiers:
                        try:
                            int(i)
                        except ValueError as e:
                            raise ValueError(
                                f"Linnaeus: invalid identifier: {i}"
                            ) from e
                            # i = NIL
                        parsed_identifiers.append(i)

                    # eid = (
                    #     items[1].split("_")[0]
                    #     if corpus_name == "s800"
                    #     else items[1].replace("pmcA", "")
                    # )

                    eid = items[1].split("_")[0]

                    offset = (int(items[2]), int(items[3]))

                    if eid not in annotations:
                        annotations[eid] = {}

                    annotations[eid][offset] = [parsed_identifiers]

            out[f"linnaeus_{config}"] = annotations

        return out


def main():
    """
    Script
    """

    parser = get_argument_parser()

    args = parser.parse_args()

    db_conifg = os.path.join(os.getcwd(), "config", "db.yaml")

    linnaeus = Linnaeus(
        in_dir=args.in_dir,
        belb_dir=args.belb_dir,
        db_config=db_conifg,
        split=args.split,
        joint_ner_nen=True,
        obsolete_kb=True,
    )

    if args.run == "input":
        linnaeus.create_input()
    elif args.run == "output":
        results_dir = os.path.join(os.getcwd(), "data", "results")
        os.makedirs(results_dir, exist_ok=True)
        linnaeus.collect_results(results_dir)


if __name__ == "__main__":
    main()
