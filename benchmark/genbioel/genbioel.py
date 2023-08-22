#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert BELB to format required for GenBioEl: https://github.com/Yuanhy1997/GenBioEL
See: https://aclanthology.org/2022.naacl-main.296/
"""
import itertools
import json
import os
import pickle
import re
import sys
from collections import defaultdict
from typing import Optional

import numpy as np
from belb import (ENTITY_TO_KB_NAME, AutoBelbCorpus, AutoBelbKb, Entities,
                  Example, Queries)
from belb.kbs.ncbi_gene import NCBI_GENE_SUBSETS
from belb.resources import Corpora, Kbs
from loguru import logger
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import BartTokenizer

from benchmark.model import CORPORA_MULTI_ENTITY_TYPES, Model
from benchmark.utils import get_argument_parser, load_json, save_json

# keep parenthesis for disambigated entries
# PUNCTUATION = re.compile(r"[,.;{}[\]+\-_*/?!`\"\'=%><]")

# as defined in https://github.com/Yuanhy1997/GenBioEL/blob/main/src/data_utils/ncbi/prepare_dataset.py
PUNCTUATION = re.compile(r"[,.;{}[\]()+\-_*/?!`\"\'=%><]")


def preprocess_name(name: str, lowercase: bool = False):
    """
    https://github.com/Yuanhy1997/GenBioEL/blob/main/src/data_utils/ncbi/prepare_dataset.py
    """
    name = name.lower() if lowercase else name
    name = PUNCTUATION.sub(" ", name)
    name = " ".join(name.split())
    return name


def vectorize_kb(entity_names: set[str]) -> TfidfVectorizer:
    """
    Fit vectorizer to all KB names
    """
    # only 3-grams
    vectorizer = TfidfVectorizer(ngram_range=(3, 3), analyzer="char")
    vectorizer.fit(entity_names)
    return vectorizer


def cal_similarity_tfidf(a: list[str], b: str, vectorizer):
    """
    Get most similar name via tfidf scores
    """
    features_a = vectorizer.transform(a)
    features_b = vectorizer.transform([b])
    sim = features_b.dot(features_a.T).todense()
    return sim[0].argmax(), np.max(np.array(sim)[0])


def chunks(iterable, size=10):
    iterator = iter(iterable)
    for first in iterator:
        yield itertools.chain([first], itertools.islice(iterator, size - 1))


def tokenize_entities(entities):
    """
    Apply transformer tokenizer
    """
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")
    for chunk in chunks(entities, 10000):
        chunk_input_ids = tokenizer.batch_encode_plus(
            [" " + e for e in chunk],
            padding=False,
            return_offsets_mapping=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )

        for input_ids in chunk_input_ids["input_ids"]:
            tokens = [16] + input_ids[1:]
            yield tokens


class Trie:
    """
    https://github.com/Yuanhy1997/GenBioEL/blob/main/src/trie/trie.py
    """

    def __init__(self, sequences: list[list[int]] = []):
        self.trie_dict = {}
        self.len = 0
        if sequences:
            for sequence in sequences:
                Trie._add_to_trie(sequence, self.trie_dict)
                self.len += 1

        self.append_trie = None
        self.bos_token_id = None

    def append(self, trie, bos_token_id):
        self.append_trie = trie
        self.bos_token_id = bos_token_id

    def add(self, sequence: list[int]):
        Trie._add_to_trie(sequence, self.trie_dict)
        self.len += 1

    def get(self, prefix_sequence: list[int]):
        return Trie._get_from_trie(
            prefix_sequence, self.trie_dict, self.append_trie, self.bos_token_id
        )

    @staticmethod
    def load_from_dict(trie_dict):
        trie = Trie()
        trie.trie_dict = trie_dict
        trie.len = sum(1 for _ in trie)
        return trie

    @staticmethod
    def _add_to_trie(sequence: list[int], trie_dict: dict):
        if sequence:
            if sequence[0] not in trie_dict:
                trie_dict[sequence[0]] = {}
            Trie._add_to_trie(sequence[1:], trie_dict[sequence[0]])

    @staticmethod
    def _get_from_trie(
        prefix_sequence: list[int],
        trie_dict: dict,
        append_trie=None,
        bos_token_id: int = None,
    ):
        if len(prefix_sequence) == 0:
            output = list(trie_dict.keys())
            if append_trie and bos_token_id in output:
                output.remove(bos_token_id)
                output += list(append_trie.trie_dict.keys())
            return output
        elif prefix_sequence[0] in trie_dict:
            return Trie._get_from_trie(
                prefix_sequence[1:],
                trie_dict[prefix_sequence[0]],
                append_trie,
                bos_token_id,
            )
        else:
            if append_trie:
                return append_trie.get(prefix_sequence)
            else:
                return []

    def __iter__(self):
        def _traverse(prefix_sequence, trie_dict):
            if trie_dict:
                for next_token in trie_dict:
                    yield from _traverse(
                        prefix_sequence + [next_token], trie_dict[next_token]
                    )
            else:
                yield prefix_sequence

        return _traverse([], self.trie_dict)

    def __len__(self):
        return self.len

    def __getitem__(self, value):
        return self.get(value)


def get_truncated_text(
    mention: str, context_left: str, context_right: str, tokenizer: BartTokenizer
) -> str:
    mention_tokens = tokenizer.tokenize(mention)

    max_len_context = tokenizer.max_len_single_sentence - len(mention_tokens) - 2
    max_len_context_side = int(np.ceil(max_len_context / 2))

    ctx_left_tokens = tokenizer.tokenize(context_left)
    ctx_left_tokens = ctx_left_tokens[::-1][:max_len_context_side][::-1]

    ctx_right_tokens = tokenizer.tokenize(context_right)
    ctx_right_tokens = ctx_right_tokens[:max_len_context_side]

    tokens = ctx_left_tokens + mention_tokens + ctx_right_tokens

    if len(tokens) > 1020:
        print(f"Long text input: {len(tokens)} tokens")

    return tokenizer.convert_tokens_to_string(tokens)


def convert_to_text_annotations(example: Example, title_abstract: bool = True):
    """
    Merge title and abstract
    """

    if title_abstract:
        texts_annotatations = []

        text = " ".join([example.passages[0].text, example.passages[1].text])

        annotations = example.passages[0].annotations + example.passages[1].annotations

        texts_annotatations.append((text, annotations))

        for p in example.passages[2:]:
            for a in p.annotations:
                a.start = a.start - p.offset
                a.end = a.end - p.offset
            texts_annotatations.append((p.text, p.annotations))

    else:
        for p in example.passages:
            for a in p.annotations:
                a.start = a.start - p.offset
                a.end = a.end - p.offset

        texts_annotatations = [(p.text, p.annotations) for p in example.passages]

    return texts_annotatations


class GenBioEl(Model):
    """
    Helper to deal w/ arboEL input/output
    """

    def __init__(self, *args, lowercase: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        assert (
            self.out_dir is not None
        ), "GenBioEl has different input and output directories. Please pass `out_dir`"
        self.lowercase = lowercase

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
            (Corpora.GNORMPLUS.name, Entities.GENE),
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

    def convert_kbs(self):
        """
        Convert KBs to format required by GenBioEl
        """

        logger.info("Converting BELB KBs into GenBioEl format")

        for spec in self.kbs:
            logger.debug("Converting KB: `{}`", spec)

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

            query = kb.queries.get(Queries.DICTIONARY_ENTRIES, subset=subset)

            id_to_names = defaultdict(list)
            with kb as handle:
                for row in handle.query(query):
                    identifier = row["identifier"]
                    name = preprocess_name(name=row["name"], lowercase=self.lowercase)
                    id_to_names[identifier].append(name)

            save_json(path=os.path.join(kb_outdir, "target_kb.json"), item=id_to_names)

    def convert_corpora(self):
        """
        BELB corpora to GenBioEl format
        """

        logger.info("Converting BELB corpora into GenBioEl format")

        tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")

        for corpus_name, entity_type in self.corpora:
            logger.debug("Converting corpus: `{}` ({})", corpus_name, entity_type)

            kb = AutoBelbKb.from_name(
                directory=self.belb_dir,
                name=ENTITY_TO_KB_NAME[entity_type],
                db_config=self.db_config,
                debug=False,
            )

            corpus = AutoBelbCorpus.from_name(
                name=corpus_name,
                directory=self.belb_dir,
                entity_type=entity_type,
                sentences=False,
                mention_markers=False,
                add_foreign_annotations=False,
            )

            with kb as handle:
                data = self.prepare_corpus(kb=handle, corpus=corpus.data)

            parts = [self.in_dir, "kbs", kb.kb_config.name]
            if corpus_name in [Corpora.GNORMPLUS.name, Corpora.NLM_GENE.name]:
                parts.append(corpus_name)

            id_to_names = load_json(os.path.join(*parts, "target_kb.json"))

            vectorizer = vectorize_kb(
                set(n for id, names in id_to_names.items() for n in names)
            )

            corpus_name = (
                f"{corpus_name}_{entity_type}"
                if corpus_name in CORPORA_MULTI_ENTITY_TYPES
                else corpus_name
            )

            input_dir = "runs_ar" if self.ab3p is not None else "runs"
            out_dir = os.path.join(self.in_dir, input_dir, corpus_name)
            os.makedirs(out_dir, exist_ok=True)

            abbreviations = {}
            for split, examples in data.items():
                if self.ab3p:
                    abbreviations = self.build_abbreviation_dictionary(
                        examples=examples
                    )

                with open(
                    os.path.join(out_dir, f"{split}.source"), "w"
                ) as source, open(
                    os.path.join(out_dir, f"{split}.target"), "w"
                ) as target, open(
                    os.path.join(out_dir, f"{split}label.txt"), "w"
                ) as labels, open(
                    os.path.join(out_dir, f"{split}.hexdigests.txt"), "w"
                ) as hexdigests:
                    for e in examples:
                        texts_annotatations = convert_to_text_annotations(
                            example=e, title_abstract=corpus.config.title_abstract
                        )

                        for text, annotations in texts_annotatations:
                            for a in annotations:
                                hexdigest = a.infons["hexdigest"]

                                lower_text = text.lower() if self.lowercase else text

                                ann_text = a.text
                                if ann_text in abbreviations.get(e.id, {}):
                                    # print(
                                    #     f"Expanding abbreviation {a.text} to {abbreviations[e.id][a.text]}"
                                    # )
                                    ann_text = abbreviations[e.id][a.text]

                                ann_text = preprocess_name(
                                    name=ann_text, lowercase=self.lowercase
                                )

                                marked_text = get_truncated_text(
                                    tokenizer=tokenizer,
                                    mention=f"START {ann_text} END",
                                    context_left=lower_text[: a.start],
                                    context_right=lower_text[a.end :],
                                )

                                identifiers = [str(i) for i in a.identifiers]
                                candidate_entities = [
                                    name
                                    for i in identifiers
                                    for name in id_to_names[str(i)]
                                ]

                                idx = cal_similarity_tfidf(
                                    candidate_entities, ann_text, vectorizer
                                )
                                entity_name = candidate_entities[idx[0]]

                                entity_link = [f"{ann_text} is", entity_name]

                                source.write(f"{json.dumps([marked_text])}\n")
                                target.write(f"{json.dumps(entity_link)}\n")
                                labels.write(f"{'|'.join(identifiers)}\n")
                                hexdigests.write(f"{hexdigest}\n")

    def convert_kbs_to_trie(self):
        """
        Tokenize KB and convert it into prefix-tree
        """

        logger.info("Converting KBs into prefix-trees")

        # https://stackoverflow.com/questions/2134706/hitting-maximum-recursion-depth-using-pickle-cpickle
        sys.setrecursionlimit(10000)

        for spec in self.kbs:
            logger.debug("Converting KB into prefix-tree: `{}`", spec)

            name = spec["name"]
            subset = spec.get("subset")

            kb_outdir = os.path.join(self.in_dir, "kbs", name)

            if subset is not None:
                kb_outdir = os.path.join(kb_outdir, subset)

            id_to_names = load_json(os.path.join(kb_outdir, "target_kb.json"))

            trie = Trie(
                tokenize_entities(
                    set(n for id, names in id_to_names.items() for n in names)
                )
            )

            trie_out = os.path.join(kb_outdir, "trie.pkl")
            with open(trie_out, "wb") as output:
                pickle.dump(trie.trie_dict, output)

    def create_input(self):
        # self.convert_kbs()
        # self.convert_kbs_to_trie()
        self.convert_corpora()

    def parse_output(
        self, corpus_name: str, gold: dict, entity_type: Optional[str] = None
    ) -> dict:
        dir_name = (
            f"{corpus_name}_{entity_type}"
            if corpus_name in CORPORA_MULTI_ENTITY_TYPES
            else corpus_name
        )

        assert self.out_dir is not None
        results_path = os.path.join(
            self.out_dir,
            "src",
            # "model_checkpoints",
            "model_checkpoints_ar",
            dir_name,
            "checkpoint-20000",
            "results_test.pkl",
        )

        with open(results_path, "rb") as fp:
            results = pickle.load(fp)

        identifiers = results[1]

        hexdigests_path = os.path.join(
            self.in_dir, "runs_ar", dir_name, "test.hexdigests.txt"
        )

        with open(hexdigests_path) as fp:
            hexdigests = [line.strip() for line in fp]

        assert len(identifiers) == len(hexdigests), (
            f"Corpus {corpus_name}:"
            f"# of pred {len(identifiers)} != # of gold {len(hexdigests)}"
        )

        hexdigest_to_pred = dict(zip(hexdigests, identifiers))

        pred: dict = {}
        for did, offset_to_data in gold.items():
            if did not in pred:
                pred[did] = {}
            for offset, data in offset_to_data.items():
                if offset not in pred[did]:
                    hexdigest = data["hexdigest"]
                    pred[did][offset] = hexdigest_to_pred[hexdigest]

        return {"genbioel_ar": pred}


def main():
    """
    Script
    """

    parser = get_argument_parser()
    args = parser.parse_args()

    db_conifg = os.path.join(os.getcwd(), "config", "db.yaml")

    genbioel = GenBioEl(
        in_dir=args.in_dir,
        out_dir=args.out_dir,
        belb_dir=args.belb_dir,
        ab3p=args.ab3p,
        db_config=db_conifg,
        joint_ner_nen=False,
        obsolete_kb=False,
        lowercase=True,
        identifier_mapping=False,
    )

    if args.run == "input":
        genbioel.create_input()

    elif args.run == "output":
        results_dir = os.path.join(os.getcwd(), "results")
        os.makedirs(results_dir, exist_ok=True)
        genbioel.collect_results(results_dir)


if __name__ == "__main__":
    main()
