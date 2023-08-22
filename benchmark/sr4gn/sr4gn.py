#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GNormPlus
"""

import os
from typing import Optional

from belb.preprocessing.data import Entities
from belb.resources import Corpora

from benchmark.gnormplus.gnormplus import GNormPlus
from benchmark.utils import get_argument_parser


class Sr4gn(GNormPlus):
    """
    Helper to deal w/ GNormPlus input/output
    """

    @property
    def corpora(self):
        """
        Corpora supported by baseline
        """

        corpora = [
            (Corpora.S800.name, Entities.SPECIES),
            (Corpora.LINNAEUS.name, Entities.SPECIES),
        ]
        return corpora

    def parse_output(
        self, corpus_name: str, gold: dict, entity_type: Optional[str] = None
    ) -> dict:
        out = super().parse_output(
            corpus_name=corpus_name, entity_type=entity_type, gold=gold
        )

        out["sr4gn"] = out.pop("gnormplus")

        return out


def main():
    """
    Script
    """

    parser = get_argument_parser()

    args = parser.parse_args()

    db_conifg = os.path.join(os.getcwd(), "config", "db.yaml")

    sr4gn = Sr4gn(
        in_dir=args.in_dir,
        belb_dir=args.belb_dir,
        db_config=db_conifg,
        split=args.split,
        sentences=True,
        obsolete_kb=True,
        joint_ner_nen=True,
    )

    if args.run == "input":
        sr4gn.create_input()

    elif args.run == "output":
        results_dir = os.path.join(os.getcwd(), "results")
        os.makedirs(results_dir, exist_ok=True)
        sr4gn.collect_results(results_dir)


if __name__ == "__main__":
    main()
