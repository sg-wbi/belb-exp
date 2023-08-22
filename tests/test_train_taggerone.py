#!/usr/bin/env python3
"""
Convert corpus to PubTator format
"""
import argparse
import os

import pandas as pd
from belb.kbs import ENTITY_TO_KB_NAME, AutoBelbKb
from belb.kbs.query import Queries
from belb.kbs.schema import Tables
from belb.preprocessing.data import SYMBOL_CODE, Example
from bioc import biocjson, pubtator
from sqlalchemy.sql.expression import select


def parse_args() -> argparse.Namespace:
    """
    CLI
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dir",
        type=str,
        default=None,
        help="Directory where all BELB data is stored",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Directory where to store documents in PubTator format",
    )
    parser.add_argument(
        "--db",
        required=True,
        type=str,
        help="Database configuration",
    )
    return parser.parse_args()


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def example_to_pubtator(example: Example, k=10):
    """
    DOCUMENTS MUST BE SPLIT INTO SENTENCES.
    Aggregate sentences in full text document in to smaller documents of max K sentences.
    Use the first sentencen as title.
    """

    for chunk in chunks(example.passages, 10):

        text: dict = {"abstract": []}
        annotations = []

        chunk_offset = 0

        for idx, p in enumerate(chunk):

            if idx == 0:
                text["title"] = p.text
            else:
                text["abstract"].append(p.text)

            for a in p.annotations:

                if a.foreign:
                    continue

                annotations.append(
                    pubtator.PubTatorAnn(
                        pmid=example.id,
                        start=a.start - p.offset + chunk_offset,
                        end=a.end - p.offset + chunk_offset,
                        text=a.text,
                        id=",".join(
                            [i.replace("CVCL_", "CVCL:") for i in a.identifiers]
                        ),
                        type=a.entity_type,
                    )
                )

            chunk_offset += len(p.text) + 1

        if len(annotations) == 0:
            continue

        document = pubtator.PubTator(
            pmid=example.id,
            title=text["title"],
            abstract=" ".join(text["abstract"]),
        )

        for a in annotations:
            document.add_annotation(a)

        yield document


def belb_corpus_to_taggerone_format(
    belb_dir: str, out_dir: str, corpus_name: str, entity_type: str
):

    path = os.path.join(belb_dir, "processed", "corpora", "corpora.jsonl")

    df = pd.read_json(path, lines=True)

    corpus = df[
        (df["sentences"] == True)
        & (df["mention_markers"] == False)
        & (df["add_foreign_annotations"] == False)
        & (df["name"] == corpus_name)
        & (df["entity_type"] == entity_type)
    ]

    assert len(corpus) == 1

    corpus = corpus.to_dict("records")[0]

    corpus_path = os.path.join(
        belb_dir, "processed", "corpora", "bioid", corpus["hexdigest"]
    )

    documents: list = []
    for split in corpus["splits"]:
        split_path = os.path.join(corpus_path, f"{split}.bioc.json")
        with open(split_path) as fp:
            examples = [Example.from_bioc(d) for d in biocjson.load(fp).documents]

        with open(os.path.join(out_dir, f"{split}.txt"), "w") as fp:
            documents = []
            for example in examples:
                fp.write(f"{example.id}\n")
                for document in example_to_pubtator(example, k=10):
                    documents.append(document)

    out_path = os.path.join(out_dir, f"{corpus_name}.pubtator.txt")
    with open(out_path, "w") as fp:
        pubtator.dump(documents, fp)


def belb_kb_to_taggerone_format(
    belb_dir: str, db_config: str, out_dir: str, entity_type: str
):

    kb = AutoBelbKb.from_name(
        directory=belb_dir,
        name=ENTITY_TO_KB_NAME[entity_type],
        db_config=db_config,
        debug=False,
    )

    ifm = kb.schema.get(Tables.IDENTIFIER_MAPPING)

    with open(os.path.join(out_dir, f"{kb.kb_config.name}.tsv"), "w") as fp:
        with kb as handle:

            identifier_mapping = {}
            for row in handle.query(select(ifm)):
                identifier_mapping[row["internal_identifier"]] = row[
                    "original_identifier"
                ]

            for row in handle.query(kb.queries.get(Queries.SYNSET)):

                parsed_row = kb.queries.parse_result(name=Queries.SYNSET, row=row)

                assert isinstance(parsed_row, dict)

                identifier = identifier_mapping[parsed_row["identifier"]].replace(
                    "CVCL_", "CVCL:"
                )

                names_descriptions = list(
                    zip(parsed_row["names"], parsed_row["descriptions"])
                )

                title = [p[0] for p in names_descriptions if int(p[1]) == SYMBOL_CODE]

                try:
                    symbol = title[0]
                except IndexError as error:
                    raise RuntimeError(
                        f"Sysnet {parsed_row} has no symbol! This cannot happen!"
                    ) from error

                line = f"{identifier}\t{symbol}"

                names = [p[0] for p in names_descriptions if int(p[1]) != SYMBOL_CODE]

                if len(names) > 0:
                    line += " | " + " | ".join(names)

                fp.write(f"{line}\n")


def main():
    """
    Script
    """

    args = parse_args()

    belb_corpus_to_taggerone_format(
        belb_dir=args.dir,
        out_dir=args.out,
        corpus_name="bioid",
        entity_type="cell_line",
    )

    belb_kb_to_taggerone_format(
        belb_dir=args.dir, out_dir=args.out, db_config=args.db, entity_type="cell_line"
    )


if __name__ == "__main__":
    main()
