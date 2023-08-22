#!/usr/bin/env python3
"""
SciScapy on MedMentions
"""
import os
from typing import Optional

import spacy
from belb import AutoBelbCorpus, Entities
from belb.resources import Corpora
from scispacy.abbreviation import AbbreviationDetector  # noqa: F401
from scispacy.linking import EntityLinker  # noqa: F401

from benchmark.model import NIL, Model
from benchmark.utils import get_argument_parser


class SciSpacy(Model):
    """
    Helper to deal w/ BioSyn input/output
    """

    def __init__(self, topk: int = 5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.topk = topk
        self.nlp = spacy.load("en_core_sci_sm")
        self.nlp.add_pipe("abbreviation_detector")
        # self.nlp.disable_pipe
        self.nlp.add_pipe(
            "scispacy_linker",
            config={"resolve_abbreviations": True, "linker_name": "umls"},
        )
        self.linker = self.nlp.get_pipe("scispacy_linker")

    @property
    def corpora(self):
        """Corpora this model can be run on"""

        return [(Corpora.MEDMENTIONS.name, Entities.UMLS)]

    def create_input(self):
        """Directly parse corpus"""

    def parse_output(
        self, corpus_name: str, gold: dict, entity_type: Optional[str] = None
    ) -> dict:
        """Run SciSpacy pipeline:
        - detect abbraviation
        - link entity mentions
        """

        corpus = AutoBelbCorpus.from_name(
            name=corpus_name,
            directory=self.belb_dir,
            entity_type=entity_type,
            sentences=False,
            mention_markers=False,
            add_foreign_annotations=False,
        )

        preds: dict = {}

        for d in corpus.data["test"]:
            if d.id not in preds:
                preds[d.id] = {}

            doc = self.nlp(" ".join([p.text for p in d.passages]))
            abbreviations = {}
            for abbr in doc._.abbreviations:
                abbreviations[(abbr.start_char, abbr.end_char)] = abbr._.long_form.text

            annotations = [
                ((a.start, a.end), a.text) for p in d.passages for a in p.annotations
            ]

            for i, (o, _) in enumerate(annotations):
                if o in abbreviations:
                    annotations[i] = (o, abbreviations[o])

            offsets, texts = zip(*annotations)

            candidates = self.linker.candidate_generator(texts, self.topk)

            candidates = [
                [[c.concept_id] for c in ranks] if len(ranks) > 0 else [NIL]
                for ranks in candidates
            ]

            for offset, candidate in zip(offsets, candidates):
                preds[d.id][offset] = candidate

        return {"scispacy": preds}


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

    scispacy_model = SciSpacy(
        topk=args.k,
        in_dir=args.in_dir,
        out_dir=args.out_dir,
        belb_dir=args.belb_dir,
        db_config=db_conifg,
        joint_ner_nen=False,
        obsolete_kb=False,
        identifier_mapping=True,
    )

    if args.run == "input":
        scispacy_model.create_input()

    elif args.run == "output":
        results_dir = os.path.join(os.getcwd(), "results")
        os.makedirs(results_dir, exist_ok=True)
        scispacy_model.collect_results(results_dir)


if __name__ == "__main__":
    main()
