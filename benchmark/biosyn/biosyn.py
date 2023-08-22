#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert BELB to format required by BioSyn:https://github.com/dmis-lab/BioSyn
See: https://aclanthology.org/2020.acl-main.335/
"""
import os
from typing import Optional

from belb import (ENTITY_TO_KB_NAME, AutoBelbCorpus, AutoBelbKb, Entities,
                  Queries)
from belb.kbs.ncbi_gene import NCBI_GENE_SUBSETS
from belb.resources import Corpora, Kbs
from loguru import logger

from benchmark.model import CORPORA_MULTI_ENTITY_TYPES, Model
from benchmark.utils import get_argument_parser, load_json

# pylint: disable=singleton-comparison


class BioSyn(Model):
    """
    Helper to deal w/ BioSyn input/output
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert (
            self.out_dir is not None
        ), "BioSyn has different input and output directories. Please pass `out_dir`"

    @property
    def kbs(self):
        """
        KBs supported
        """
        kbs = [
            {"name": Kbs.CTD_DISEASES.name},
            {"name": Kbs.CTD_CHEMICALS.name},
            {"name": Kbs.CELLOSAURUS.name},
            {"name": Kbs.NCBI_TAXONOMY.name},
            {"name": Kbs.UMLS.name},
            {"name": Kbs.NCBI_GENE.name, "subset": "gnormplus"},
            {"name": Kbs.NCBI_GENE.name, "subset": "nlm_gene"},
        ]

        return kbs

    @property
    def corpora(self):
        """
        Corpora supported
        """

        return [
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
        ]

    def convert_corpora(self):
        """
        Convert corpora in BELB into format for BioSyn
        """

        logger.info("Start converting BELB corpora into BioSyn format...")

        input_dir = "runs_ar" if self.ab3p is not None else "runs"
        out_dir = os.path.join(self.in_dir, input_dir)

        for corpus_name, entity_type in self.corpora:
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

            with kb as handle:
                data = self.prepare_corpus(kb=handle, corpus=corpus.data)

            corpus_name = (
                f"{corpus_name}_{entity_type}"
                if corpus_name in CORPORA_MULTI_ENTITY_TYPES
                else corpus_name
            )
            corpus_outdir = os.path.join(out_dir, corpus_name)

            abbreviations = {}
            for split, documents in data.items():
                if self.ab3p:
                    abbreviations = self.build_abbreviation_dictionary(
                        examples=documents
                    )
                split_outdir = os.path.join(corpus_outdir, f"{split}")
                os.makedirs(split_outdir, exist_ok=True)

                for document in documents:
                    with open(
                        os.path.join(split_outdir, f"{document.id}.concept"), "w"
                    ) as mfp, open(
                        os.path.join(split_outdir, f"{document.id}.txt"), "w"
                    ) as dfp:
                        passages = []
                        for p in document.passages:
                            passages.append(p.text)
                            for a in p.annotations:
                                entity_type = a.original["entity_type"]
                                identifiers = "|".join(str(i) for i in a.identifiers)

                                ann_text = a.text
                                if ann_text in abbreviations.get(document.id, {}):
                                    print(
                                        f"Expand {ann_text} to {abbreviations[document.id][ann_text]}"
                                    )
                                    ann_text = abbreviations[document.id][ann_text]

                                mfp.write(
                                    f"{document.id}||{a.start}|{a.end}||{entity_type}||{ann_text}||{identifiers}\n"
                                )

                        text = passages[0] + "\n\n" + " ".join(passages[1:]) + "\n"
                        dfp.write(text)

    def convert_kbs(self, shard_size: int = int(1e6)):
        """Convert kbs in BELB into format for BioSyn"""

        logger.info("Start converting BELB kbs into BioSyn format...")

        for spec in self.kbs:
            name = spec["name"]
            subset = spec.get("subset")

            kb_outdir = os.path.join(self.in_dir, "kbs", name)

            if subset is not None:
                kb_outdir = os.path.join(kb_outdir, subset)

            os.makedirs(kb_outdir, exist_ok=True)

            kb = AutoBelbKb.from_name(
                name=name,
                directory=self.belb_dir,
                db_config=self.db_config,
                subset=spec.get("subset"),
            )

            if spec["name"] == Kbs.NCBI_GENE.name and subset is not None:
                subset = NCBI_GENE_SUBSETS[subset]

            query = kb.queries.get(Queries.SYNSET, subset=subset)

            with open(os.path.join(kb_outdir, "dictionary.txt"), "w") as fp:
                with kb as handle:
                    for row in handle.query(query):
                        parsed_row = kb.queries.parse_result(
                            name=Queries.SYNSET, row=row
                        )
                        identifier = parsed_row["identifier"]
                        names = "|".join([n for n in parsed_row["names"] if n != ""])
                        line = f"{identifier}||{names}\n"

                        fp.write(line)

    def create_input(self):
        """
        Generate input for arboEL
        """

        self.convert_corpora()
        # self.convert_kbs()

    def parse_output(
        self, corpus_name: str, gold: dict, entity_type: Optional[str] = None
    ) -> dict:
        corpus_name = (
            f"{corpus_name}_{entity_type}"
            if corpus_name in CORPORA_MULTI_ENTITY_TYPES
            else corpus_name
        )

        test_dir = os.path.join(self.in_dir, "runs_ar", corpus_name, "processed_test")
        test_entity_mentions = []
        for f in sorted(os.listdir(test_dir)):
            with open(os.path.join(test_dir, f)) as fp:
                for line in fp:
                    pmid, span, _, text, _ = line.split("||")
                    start, end = span.split("|")
                    test_entity_mentions.append((pmid, (start, end), text))

        results = load_json(
            os.path.join(
                self.out_dir, "models_ar", corpus_name, "predictions_eval.json"
            )
        )
        results = results["queries"]

        pred = {}
        for m, r in zip(test_entity_mentions, results):
            mentions = r["mentions"]
            assert len(mentions) == 1, f"Test entity mention is composite: {mentions}"
            mention = mentions[0]
            pmt = mention["mention"]
            pmid, offset, gmt = m
            offset = tuple(int(o) for o in offset)
            assert gmt == pmt, (
                f"Mention text in gold file `{gmt}` != `{pmt}`",
                "mention text in prediction file",
            )

            if pmid not in pred:
                pred[pmid] = {}
            if offset not in pred[pmid]:
                identifiers = [
                    [int(i) for i in c["cui"].split("|")] for c in mention["candidates"]
                ]
                pred[pmid][offset] = identifiers

        return {"biosyn_ar": pred}


def main():
    """
    Script
    """

    parser = get_argument_parser()

    args = parser.parse_args()

    db_conifg = os.path.join(os.getcwd(), "config", "db.yaml")

    biosyn = BioSyn(
        in_dir=args.in_dir,
        out_dir=args.out_dir,
        belb_dir=args.belb_dir,
        ab3p=args.ab3p,
        db_config=db_conifg,
        joint_ner_nen=False,
        obsolete_kb=False,
        identifier_mapping=False,
    )

    if args.run == "input":
        biosyn.create_input()

    elif args.run == "output":
        results_dir = os.path.join(os.getcwd(), "results")
        os.makedirs(results_dir, exist_ok=True)
        biosyn.collect_results(results_dir)


if __name__ == "__main__":
    main()
