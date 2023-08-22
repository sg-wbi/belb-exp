#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert BELB to format required for BLINK: https://github.com/dhdhagar/arboEL
See: https://aclanthology.org/2022.naacl-main.343/
"""
import copy
import json
import os
import re
from typing import Optional

from belb import (ENTITY_TO_KB_NAME, SYMBOL_CODE, Annotation, AutoBelbCorpus,
                  AutoBelbKb, Entities, Example, Queries)
from belb.kbs.ncbi_gene import NCBI_GENE_SUBSETS
from belb.resources import Corpora, Kbs
from loguru import logger

from benchmark.model import CORPORA_MULTI_ENTITY_TYPES, Model
from benchmark.utils import get_argument_parser, load_json


class ArboEl(Model):
    """
    Helper to deal w/ arboEL input/output
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert (
            self.out_dir is not None
        ), "ArboEL has different input and output directories. Please pass `out_dir`"

    @property
    def kbs(self):
        """
        KBs supported
        """
        kbs = [
            {"name": Kbs.CTD_DISEASES.name},
            # {"name": Kbs.CTD_CHEMICALS.name},
            # {"name": Kbs.CELLOSAURUS.name},
            # {"name": Kbs.NCBI_TAXONOMY.name},
            {"name": Kbs.UMLS.name},
            # {"name": Kbs.NCBI_GENE.name, "subset": "gnormplus"},
            # {"name": Kbs.NCBI_GENE.name, "subset": "nlm_gene"},
        ]

        return kbs

    @property
    def corpora(self):
        """
        Corpora supported
        """

        return [
            # (Corpora.GNORMPLUS.name, Entities.GENE),
            (Corpora.NLM_GENE.name, Entities.GENE),
            # (Corpora.NCBI_DISEASE.name, Entities.DISEASE),
            # (Corpora.BC5CDR.name, Entities.DISEASE),
            # (Corpora.BC5CDR.name, Entities.CHEMICAL),
            # (Corpora.NLM_CHEM.name, Entities.CHEMICAL),
            # (Corpora.LINNAEUS.name, Entities.SPECIES),
            # (Corpora.S800.name, Entities.SPECIES),
            # (Corpora.BIOID.name, Entities.CELL_LINE),
            # (Corpora.MEDMENTIONS.name, Entities.UMLS),
        ]

    def expand_abbreviations(
        self, example: Example, abbreviations: dict
    ) -> tuple[str, list[Annotation]]:
        text = " ".join(p.text for p in example.passages)
        annotations = [
            a for p in example.passages for a in p.annotations if not a.foreign
        ]

        if len([a for a in annotations if a.text in abbreviations]) == 0:
            remapped = annotations
        else:
            annotations_text = [a.text for a in annotations]

            abbreviations = {
                sf: lf for sf, lf in abbreviations.items() if sf in annotations_text
            }

            expanded_annotations = copy.deepcopy(annotations)

            for sf, lf in abbreviations.items():
                text = text.replace(sf, lf)
                for a in expanded_annotations:
                    if a.text == sf:
                        a.text = lf
                    elif sf in a.text:
                        a.text = a.text.replace(sf, lf)

            remapped = []
            annotation_match_checks = [False] * len(expanded_annotations)
            last_match = 0
            for idx, a in enumerate(
                sorted(expanded_annotations, key=lambda x: x.start)
            ):
                # print(f"Query: {a.text}")
                pattern_str = re.escape(a.text)
                pattern_str = rf"(?<!\w){pattern_str}(?!\w)"
                pattern = re.compile(pattern_str)
                # from last match found in sentence
                # check for exact match of # of sentinel tokens
                # print(f"Search: {text[last_match:]}")
                match = re.search(pattern, text[last_match:])

                if match is not None:
                    # print(f"Found: {a.text}\n")
                    text_offset = len(text[:last_match])

                    last_match = match.end() + text_offset

                    a.start = match.start() + text_offset
                    a.end = match.end() + text_offset

                    remapped.append(a)

                    annotation_match_checks[idx] = True
                # else:
                # print(f"Not found: {a.text}\n")

            if not all(annotation_match_checks):
                unmatched = [
                    a.text
                    for idx, a in enumerate(expanded_annotations)
                    if not annotation_match_checks[idx]
                ]

                # raise RuntimeError(f"EID:{example.id}| Could not remap: `{unmatched}`")
                logger.debug(f"EID:{example.id}| Could not remap: `{unmatched}`")

                # if example.id == "27322685":
                #     breakpoint()

        return text, remapped

    def convert_corpora(self):
        """
        Convert corpora in BELB into format for arboEL (BLINK)
        """

        logger.info("Start converting BELB corpora into BLINK format...")

        folder = "runs_ar" if self.ab3p is not None else "runs"

        out_dir = os.path.join(self.in_dir, folder)

        for corpus_name, entity_type in self.corpora:
            corpus = AutoBelbCorpus.from_name(
                name=corpus_name,
                directory=self.belb_dir,
                entity_type=entity_type,
                sentences=False,
                mention_markers=False,
                add_foreign_annotations=False,
            )

            corpus_name = (
                f"{corpus_name}_{entity_type}"
                if corpus_name in CORPORA_MULTI_ENTITY_TYPES
                else corpus_name
            )

            corpus_outdir_mentions = os.path.join(out_dir, corpus_name, "mentions")
            os.makedirs(corpus_outdir_mentions, exist_ok=True)
            corpus_outdir_documents = os.path.join(out_dir, corpus_name, "documents")
            os.makedirs(corpus_outdir_documents, exist_ok=True)

            kb = AutoBelbKb.from_name(
                directory=self.belb_dir,
                name=ENTITY_TO_KB_NAME[entity_type],
                db_config=self.db_config,
                debug=False,
            )

            with kb as handle:
                data = self.prepare_corpus(kb=handle, corpus=corpus.data)

            mention_id_to_hexdigest = {}

            abbreviations = None
            for split, documents in data.items():
                if self.ab3p is not None:
                    abbreviations = self.build_abbreviation_dictionary(
                        examples=documents
                    )

                split = "val" if split == "dev" else split

                split_outfile_mentions = os.path.join(
                    corpus_outdir_mentions, f"{split}.json"
                )
                split_outfile_documents = os.path.join(
                    corpus_outdir_documents, f"{split}.json"
                )

                with open(split_outfile_mentions, "w") as out_fp_mentions, open(
                    split_outfile_documents, "w"
                ) as out_fp_documents:
                    for document in documents:
                        if self.ab3p:
                            text, annotations = self.expand_abbreviations(
                                example=document,
                                abbreviations=abbreviations.get(document.id, {}),
                            )
                        else:
                            text = " ".join(p.text for p in document.passages)
                            annotations = [
                                a
                                for p in document.passages
                                for a in p.annotations
                                if not a.foreign
                            ]

                        json_document = {
                            "document_id": document.id,
                            "title": document.id,
                            "text": text,
                        }

                        out_fp_documents.write(f"{json.dumps(json_document)}\n")

                        for a in annotations:
                            assert a.identifiers is not None

                            # same mention multiple times since BLINK
                            # cannot handle multiple identifiers
                            for i in a.identifiers:
                                mention_id = f"{document.id}.{a.id}"
                                hexdigest = a.infons["hexdigest"]

                                json_mention = {
                                    "mention_id": mention_id,
                                    "context_document_id": document.id,
                                    "start_index": a.start,
                                    "end_index": a.end,
                                    "text": a.text,
                                    "label_document_id": str(i),
                                    "category": a.entity_type,
                                    "corpus": split,
                                }

                                mention_id_to_hexdigest[mention_id] = hexdigest

                                out_fp_mentions.write(f"{json.dumps(json_mention)}\n")

            map_path = os.path.join(out_dir, corpus_name, "mention_id_to_hash.json")
            with open(map_path, "w") as fp:
                json.dump(mention_id_to_hexdigest, fp)

    def convert_kbs(self, shard_size: int = int(1e6)):
        """
        Convert kbs in BELB into format for BLINK
        """

        logger.info("Start converting BELB kbs into BLINK format...")

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

            shard: list = []
            idx = 0

            with kb as handle:
                for row in handle.query(query):
                    parsed_row = kb.queries.parse_result(name=Queries.SYNSET, row=row)

                    assert isinstance(parsed_row, dict)

                    line = {"document_id": str(parsed_row["identifier"])}

                    names_descriptions = list(
                        zip(parsed_row["names"], parsed_row["descriptions"])
                    )

                    title = [
                        p[0] for p in names_descriptions if int(p[1]) == SYMBOL_CODE
                    ]

                    names = sorted(
                        [p[0] for p in names_descriptions if int(p[1]) != SYMBOL_CODE]
                    )

                    if len(title) == 0:
                        symbol = names[0]
                        logger.debug(
                            "Sysnet {} has no symbol: removed during disambiguation...",
                            parsed_row,
                        )
                    else:
                        symbol = title[0]

                    line["title"] = symbol

                    if len(names) > 0:
                        text = " ; ".join(names)
                        text = f"{symbol} ( {text} )"
                    else:
                        text = ""

                    line["text"] = text

                    if len(shard) < shard_size:
                        shard.append(line)
                    else:
                        with open(
                            os.path.join(kb_outdir, f"shard{idx}.json"), "w"
                        ) as fp:
                            for line in shard:
                                fp.write(f"{json.dumps(line)}\n")
                        shard = []
                        idx += 1

            if len(shard) > 0:
                with open(os.path.join(kb_outdir, f"shard{idx}.json"), "w") as fp:
                    for line in shard:
                        fp.write(f"{json.dumps(line)}\n")

    def create_input(self):
        """
        Generate input for arboEL
        """

        self.convert_corpora()
        # self.convert_kbs(shard_size=int(1e6))

    def parse_output(
        self, corpus_name: str, gold: dict, entity_type: Optional[str] = None
    ) -> dict:
        dir_name = (
            f"{corpus_name}_{entity_type}"
            if corpus_name in CORPORA_MULTI_ENTITY_TYPES
            else corpus_name
        )

        assert self.out_dir is not None

        path = os.path.join(
            self.out_dir,
            "models",
            "belb",
            "biencoder_inference_ar",
            # "biencoder_inference",
            # "tbs64gas8",
            dir_name,
        )

        files = [f for f in os.listdir(path) if "undirected" in f]

        best_accuracy = 0.0
        best_file = None
        results = None
        for f in files:
            current = load_json(os.path.join(path, f))
            accuracy = float(current["accuracy"].strip(" %"))
            if results is None or accuracy >= best_accuracy:
                results = current
                best_accuracy = accuracy
                best_file = f

        assert results is not None

        hexdigest_to_mention_id = {
            h: mid
            for mid, h in load_json(
                os.path.join(self.in_dir, "runs", dir_name, "mention_id_to_hash.json")
            ).items()
        }

        predictions = {
            r["mention_id"]: r for r in results["success"] + results["failure"]
        }

        pred: dict = {}
        # missing: dict = {}
        for did, offset_to_data in gold.items():
            if did not in pred:
                pred[did] = {}
            for offset, data in offset_to_data.items():
                if offset not in pred[did]:
                    hexdigest = data["hexdigest"]
                    mention_id = hexdigest_to_mention_id[hexdigest]
                    # try:
                    y_pred = predictions[mention_id]["predicted_cui"]
                    pred[did][offset] = [[y_pred]]
                    # except KeyError:
                    #     # print(f"{dir_name}: no prediction for {mention_id}")
                    #     pmid, mid = mention_id.split(".")
                    #     if pmid not in missing:
                    #         missing[pmid] = []
                    #     missing[pmid].append(mid)

        # if len(missing) > 0:
        #     print(
        #         f"arboel:{corpus_name}: pred: missing {len(missing)} documents"
        #     )
        #
        return {"arboel_ar": pred}


def main():
    """
    Script
    """

    parser = get_argument_parser()

    args = parser.parse_args()

    db_conifg = os.path.join(os.getcwd(), "config", "db.yaml")

    arboel = ArboEl(
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
        arboel.create_input()

    elif args.run == "output":
        results_dir = os.path.join(os.getcwd(), "results")
        os.makedirs(results_dir, exist_ok=True)
        arboel.collect_results(results_dir)


if __name__ == "__main__":
    main()
