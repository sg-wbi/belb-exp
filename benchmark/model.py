#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Base functionalities for baseline
"""
import os
from abc import ABCMeta, abstractmethod
from typing import Optional, Union

import bioc
import pandas as pd
from belb import (ENTITY_TO_KB_NAME, NA, OBSOLETE_IDENTIFIER, Annotation,
                  AutoBelbCorpus, AutoBelbKb, BelbKb, Example)
from belb.resources import Corpora
from bioc import biocxml, pubtator

from benchmark.utils import run_ab3p, save_json

# pylint: disable=singleton-comparison

CORPORA_MULTI_ENTITY_TYPES = [Corpora.BC5CDR.name, Corpora.BIOID.name]
FORMATS = ["txt", "pubtator", "bioc"]
NIL = "NIL"
INVALID_IDENTIFIERS = [
    OBSOLETE_IDENTIFIER,  # obsolote according to history
    NA,  # not-in-kb and no history information
]


def valid_xml_char_ordinal(charachter: str):
    """
    Check if character can be safely written to XML.
    Parameters
    ----------
    charachter : str
        charachter
    """
    codepoint = ord(charachter)
    # conditions ordered by presumed frequency
    return (
        0x20 <= codepoint <= 0xD7FF
        or codepoint in (0x9, 0xA, 0xD)
        or 0xE000 <= codepoint <= 0xFFFD
        or 0x10000 <= codepoint <= 0x10FFFF
    )


def remove_ner_false_negatives(gold: dict, pred: dict) -> tuple[dict, int]:
    """
    Exclude all annotations that were not found by the NER component
    """

    tp = 0
    for did, offsets_true in gold.items():
        offsets_pred = pred.get(did, {})
        # loop over copy
        for offset_pred in list(offsets_pred.keys())[:]:
            if offset_pred not in offsets_true:
                offsets_pred.pop(offset_pred)
            else:
                tp += 1

    return pred, tp


def update_identifiers(pred: dict, kb: BelbKb) -> tuple[dict, int, dict]:
    """
    Update predicted identifiers to current KB version.
    E.g. GNormPlus may use an old version of NCBI Gene.
    This ensures that wrong identifiers are
    due to wrong predictions and not to obsolote kb.
    """

    identifiers = set(
        i
        for did, offsets in pred.items()
        for offset, ranks in offsets.items()
        for identifiers in ranks
        for i in [i for i in identifiers if i != NIL]
    )

    notinkb_history = kb.get_notinkb_history(identifiers=identifiers)

    obsolote = 0
    for did, offsets in pred.items():
        # loop over copy
        for offset in list(offsets.keys())[:]:
            ranks = [
                [notinkb_history.get(i, i) for i in rank] for rank in offsets[offset]
            ]
            if all(all(i in INVALID_IDENTIFIERS for i in ids) for ids in ranks):
                offsets.pop(offset)
                obsolote += 1

    return pred, obsolote, notinkb_history


def handle_pred_identifiers(pred: dict, kb: BelbKb) -> dict:
    identifiers = set(
        i
        for did, offsets in pred.items()
        for offset, ranks in offsets.items()
        for identifiers in ranks
        for i in [i for i in identifiers if i != NIL]
    )

    if kb.kb_config.string_identifier:
        im = kb.get_identifier_mapping(identifiers=identifiers)

    for did, offsets in pred.items():
        for offset, ranks in offsets.items():
            pred[did][offset] = [[im.get(i, NIL) for i in rank] for rank in ranks]

    return pred


def to_dataframe(gold: dict, pred: dict) -> list[dict]:
    """
    Create dataframe containing predicted and gold identifiers
    for each entity
    """

    data = []
    for did, offsets in pred.items():
        for offset, y_pred in offsets.items():
            try:
                y_true = gold[did][offset]
            except KeyError:
                y_true = {"hexdigest": "-", "identifiers": "-"}
            data.append(
                {
                    "eid": did,
                    "start": offset[0],
                    "end": offset[1],
                    "hexdigest": y_true["hexdigest"],
                    "y_pred": y_pred,
                }
            )

    return data


class Model(metaclass=ABCMeta):
    """
    Base functionalities for baseline
    """

    def __init__(
        self,
        in_dir: str,
        belb_dir: str,
        db_config: str,
        split: str = "test",
        ab3p: Optional[str] = None,
        out_dir: Optional[str] = None,
        sentences: bool = False,
        joint_ner_nen: bool = True,
        obsolete_kb: bool = True,
        identifier_mapping: bool = False,
    ):
        self.in_dir = in_dir
        self.out_dir = out_dir
        self.ab3p = ab3p
        self.db_config = db_config
        self.belb_dir = belb_dir
        self.belb_corpora = pd.read_json(
            os.path.join(self.belb_dir, "processed", "corpora", "corpora.jsonl"),
            lines=True,
        )
        self.split = split
        self.sentences = sentences
        self.joint_ner_nen = joint_ner_nen
        self.obsolete_kb = obsolete_kb
        self.identifier_mapping = identifier_mapping

    @property
    @abstractmethod
    def corpora(self) -> list[tuple[str, str]]:
        """
        Corpora supported by baseline with entity type
        """

    @abstractmethod
    def create_input(self):
        """
        Prepare input for baseline
        """

    def build_abbreviation_dictionary(self, examples: list[Example]) -> dict:
        assert self.ab3p is not None

        texts = []
        for e in examples:
            text = " ".join(p.text for p in e.passages)
            text = f"{e.id}|{text}"
            texts.append(text)

        abbreviations = run_ab3p(ab3p_path=self.ab3p, texts=texts)

        return abbreviations

    @abstractmethod
    def parse_output(
        self, corpus_name: str, gold: dict, entity_type: Optional[str] = None
    ):
        """
        Collect results fomr basline ouput
        """

    def load_mappings_for_identifiers(self, kb: BelbKb, identifiers: set) -> dict:
        """
        Load mappings for homogeneization of identifiers
        """

        out: dict = {}
        out["ih"] = kb.get_identifier_homonyms(identifiers=identifiers)

        if kb.kb_config.string_identifier:
            identifier_mapping = kb.get_identifier_mapping(identifiers=identifiers)
            out["im"] = identifier_mapping

        return out

    def prepare_corpus(self, kb: BelbKb, corpus: dict):
        """
        map indetifiers to integers
        """

        identifiers = set()
        for _, examples in corpus.items():
            for e in examples:
                for p in e.passages:
                    for a in p.annotations:
                        if a.foreign:
                            continue
                        identifiers.update(a.identifiers)

        maps = self.load_mappings_for_identifiers(kb=kb, identifiers=identifiers)

        for _, examples in corpus.items():
            for e in examples:
                for p in e.passages:
                    for a in p.annotations:
                        if a.foreign:
                            a.identifiers = [-1]
                        else:
                            if maps.get("im") is not None:
                                a.identifiers = [maps["im"][i] for i in a.identifiers]
                            a.identifiers = [
                                int(maps["ih"].get(i, i)) for i in a.identifiers
                            ]

        return corpus

    def convert_documents(
        self,
        documents: list[bioc.BioCDocument],
        output_format: str = "pubtator",
    ) -> list[Union[pubtator.PubTator, bioc.BioCDocument]]:
        """
        Convert BELB BioC JSON to PubTator format
        """

        if output_format in ["pubtator", "txt"]:
            converted_documents = []
            for d in documents:
                title = d.passages[0].text
                abstract = " ".join(p.text for p in d.passages[1:])
                converted_documents.append(
                    pubtator.PubTator(pmid=d.id, title=title, abstract=abstract)
                )
        else:
            for d in documents:
                for p in d.passages:
                    p.text = "".join(
                        c if valid_xml_char_ordinal(c) else "?" for c in p.text
                    )
                    p.annotations = []
            converted_documents = documents

        return converted_documents

    def save_converted_documents(
        self,
        name: str,
        entity_type: str,
        path: str,
        files: bool = False,
        output_format: str = "txt",
    ):
        """
        Save converted corpus split
        """

        assert output_format in FORMATS, f"Invalid format {output_format}"

        if output_format == "pubtator":
            extension = "PubTator.txt"
        elif output_format == "bioc":
            extension = "bioc"
        elif output_format == "txt":
            extension = "txt"

        corpus = AutoBelbCorpus.from_name(
            name=name,
            directory=self.belb_dir,
            entity_type=entity_type,
            sentences=self.sentences,
            mention_markers=False,
            add_foreign_annotations=False,
        )

        documents = self.convert_documents(
            documents=[d.to_belb() for d in corpus.data[self.split]],
            output_format=output_format,
        )

        if files:
            self.save_to_files(
                path=path,
                documents=documents,
                name=name,
                split=self.split,
                extension=extension,
                output_format=output_format,
            )
        else:
            self.save_to_file(
                path=path,
                documents=documents,
                name=name,
                split=self.split,
                extension=extension,
                output_format=output_format,
            )

    def load_gold_annotations(
        self, corpus_name: str, entity_type: str, split: str = "test"
    ) -> dict:
        """
        Load BELB corpus
        """

        corpus = AutoBelbCorpus.from_name(
            name=corpus_name,
            directory=self.belb_dir,
            entity_type=entity_type,
            sentences=self.sentences,
            mention_markers=False,
            add_foreign_annotations=False,
        )

        annotations: dict = {}
        for d in corpus.data[split]:
            annotations[d.id] = {}
            for p in d.passages:
                for a in p.annotations:
                    if a.foreign:
                        continue

                    annotations[d.id][(a.start, a.end)] = {
                        "hexdigest": a.infons["hexdigest"],
                        "identifiers": a.identifiers,
                    }

        return annotations

    def collect_results(self, directory: str):
        """
        Collect results.
        """

        for corpus_name, entity_type in self.corpora:
            subdir = (
                f"{corpus_name}_{entity_type}"
                if corpus_name in CORPORA_MULTI_ENTITY_TYPES
                else corpus_name
            )

            metadata: dict = {}

            kb = AutoBelbKb.from_name(
                directory=self.belb_dir,
                name=ENTITY_TO_KB_NAME[entity_type],
                db_config=self.db_config,
                debug=False,
            )

            gold = self.load_gold_annotations(
                corpus_name=corpus_name, entity_type=entity_type, split=self.split
            )

            preds = self.parse_output(
                corpus_name=corpus_name, entity_type=entity_type, gold=gold
            )

            with kb as handle:
                for model_name, pred in preds.items():
                    out_dir = os.path.join(directory, subdir, model_name)
                    os.makedirs(out_dir, exist_ok=True)

                    if self.obsolete_kb:
                        pred, obsolete, notinkb_history = update_identifiers(
                            pred=pred, kb=handle
                        )
                        metadata["obsolete"] = obsolete
                        metadata["notinkb_history"] = notinkb_history

                    if self.identifier_mapping:
                        pred = handle_pred_identifiers(pred=pred, kb=handle)

                    predictions = to_dataframe(gold=gold, pred=pred)

                    if self.joint_ner_nen:
                        pred, tp = remove_ner_false_negatives(gold=gold, pred=pred)
                        metadata["tp"] = tp

                    total_gold = sum(len(offsets) for did, offsets in gold.items())
                    tp = metadata.get("tp", total_gold)
                    metadata["ner_recall"] = round(tp / total_gold, 2)

                    filtered_predictions = to_dataframe(gold=gold, pred=pred)

                    save_json(
                        path=os.path.join(out_dir, "metadata.json"),
                        item=metadata,
                        indent=1,
                    )

                    save_json(
                        path=os.path.join(out_dir, "predictions.json"),
                        item=predictions,
                        indent=1,
                    )

                    save_json(
                        path=os.path.join(out_dir, "filtered_predictions.json"),
                        item=filtered_predictions,
                        indent=1,
                    )

    def save_to_files(
        self,
        path: str,
        documents: list[Union[pubtator.PubTator, bioc.BioCDocument]],
        name: str,
        split: str,
        extension: str,
        output_format: str,
    ):
        """
        Save each converted document to a file
        """
        for d in documents:
            pmid = d.pmid if output_format in ["pubtator", "txt"] else d.id
            file_path = os.path.join(path, f"{pmid}_{name}_{split}.{extension}")
            with open(file_path, "w") as fp:
                if output_format == "pubtator":
                    pubtator.dump([d], fp)
                if output_format == "bioc":
                    collection = bioc.BioCCollection()
                    collection.documents = [d]
                    biocxml.dump(collection, fp)
                else:
                    text_units = [d.title, d.abstract]
                    text = " ".join([t for t in text_units if t is not None])
                    fp.write(text)

    def save_to_file(
        self,
        path: str,
        documents: list[Union[pubtator.PubTator, bioc.BioCDocument]],
        name: str,
        split: str,
        extension: str,
        output_format: str,
    ):
        """
        Save converted documents to single file
        """
        with open(os.path.join(path, f"{name}.{split}.{extension}"), "w") as fp:
            if output_format == "pubtator":
                pubtator.dump(documents, fp)
            elif output_format == "bioc":
                collection = bioc.BioCCollection()
                collection.documents = documents
                biocxml.dump(collection, fp)
            else:
                texts = []
                for d in documents:
                    text = []
                    if d.title is not None:
                        text.append(d.title)
                    if d.abstract is not None:
                        text.append(d.abstract)
                    texts.append(" ".join(text))
                fp.write("\n\n".join(texts))
