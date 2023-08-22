#!/usr/bin/env python3
"""
Run rapidfuzz: fuzzy string matching
"""
import os
from collections import defaultdict
from typing import Optional

from belb import (ENTITY_TO_KB_NAME, AutoBelbCorpus, AutoBelbKb, Entities,
                  Tables)
from belb.resources import Corpora
from rapidfuzz import fuzz, process
from sqlalchemy import select

from benchmark.model import Model
from benchmark.utils import get_argument_parser


class TheFuzz(Model):
    """
    Helper to deal w/ BioSyn input/output
    """

    def __init__(self, topk: int = 5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.topk = topk

    @property
    def corpora(self):
        """
        Corpora supported
        """

        return [
            (Corpora.BIOID.name, Entities.CELL_LINE),
        ]

    def create_input(self):
        pass

    def parse_output(
        self, corpus_name: str, gold: dict, entity_type: Optional[str] = None
    ) -> dict:
        corpus = AutoBelbCorpus.from_name(
            name=corpus_name,
            directory=self.belb_dir,
            entity_type=entity_type,
            sentences=False,
            mention_markers=False,
            add_foreign_annotations=False,
        )

        kb = AutoBelbKb.from_name(
            directory=self.belb_dir,
            name=ENTITY_TO_KB_NAME[entity_type],
            db_config=self.db_config,
            debug=False,
        )

        table = kb.schema.get(Tables.KB)

        query = select(table.c.identifier, table.c.name)

        name_to_identifiers = defaultdict(set)
        with kb as handle:
            for row in handle.query(query):
                name = row["name"].lower()
                identifier = row["identifier"]
                name_to_identifiers[name].add(identifier)

        choices = list(set(name_to_identifiers))

        cache = {}
        pred: dict = {}
        for e in corpus["test"]:
            if e.id not in pred:
                pred[e.id] = {}
            for p in e.passages:
                for a in p.annotations:
                    query = a.text.lower()
                    if query not in cache:
                        results = process.extract(
                            query, choices, scorer=fuzz.WRatio, limit=self.topk
                        )
                        results = [list(name_to_identifiers[ns[0]]) for ns in results]
                        cache[query] = results
                    else:
                        results = cache[query]

                    pred[e.id][(a.start, a.end)] = results

        return {"fuzzysearch": pred}


def main():
    """Script"""

    parser = get_argument_parser()

    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Ranks to consider in prediction",
    )

    args = parser.parse_args()

    db_conifg = os.path.join(os.getcwd(), "config", "db.yaml")

    thefuzz_model = TheFuzz(
        topk=args.k,
        in_dir=args.in_dir,
        out_dir=args.out_dir,
        belb_dir=args.belb_dir,
        db_config=db_conifg,
        joint_ner_nen=False,
        obsolete_kb=False,
        identifier_mapping=False,
    )

    if args.run == "input":
        thefuzz_model.create_input()

    elif args.run == "output":
        results_dir = os.path.join(os.getcwd(), "results")
        os.makedirs(results_dir, exist_ok=True)
        thefuzz_model.collect_results(results_dir)


if __name__ == "__main__":
    main()
