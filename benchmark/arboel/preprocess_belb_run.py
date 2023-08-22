#!/usr/bin/env python3

"""
Preprocess all BELB runs in BLINK data schema
"""

import argparse
import json
import os
import pickle

from tqdm import tqdm


def parse_args():
    """
    Parse args
    """

    parser = argparse.ArgumentParser("Preprocess mentions in BLINK format")
    parser.add_argument(
        "--dir",
        required=True,
        type=str,
        help="Directory w/ BELB runs: each sub-directory has files in BLINK (json) format",
    )
    return parser.parse_args()


def preprocess_dictionary(in_dir: str, out_dir: str):
    """
    Pickle into single file
    """

    dictionary = []
    label_ids = set()

    for doc_fname in tqdm(os.listdir(in_dir), desc="Loading docuemnts"):
        assert doc_fname.endswith(".json")
        entity_type = doc_fname.split(".")[0]
        if entity_type in ["train", "test", "val"]:
            continue
        with open(os.path.join(in_dir, doc_fname), "r") as f:
            for _, line in enumerate(f):
                record = {}
                entity = json.loads(line.strip())
                record["cui"] = entity["document_id"]
                record["title"] = entity["title"]
                record["description"] = entity["text"]
                record["type"] = entity_type
                dictionary.append(record)
                label_ids.add(record["cui"])

    assert len(dictionary) == len(label_ids)

    print(f"Finished reading {len(dictionary)} entities")
    print("Saving entity dictionary...")

    with open(os.path.join(out_dir, "dictionary.pickle"), "wb") as write_handle:
        pickle.dump(dictionary, write_handle, protocol=pickle.HIGHEST_PROTOCOL)


def preprocess_mentions(in_dir: str, out_dir: str):
    """
    Preprocess mentions: add context
    """

    # get all of the documents
    documents = {}
    dictionary = {}
    doc_dir = os.path.join(in_dir, "documents")
    for doc_fname in tqdm(os.listdir(doc_dir), desc="Loading docuemnts"):
        assert doc_fname.endswith(".json")
        with open(os.path.join(doc_dir, doc_fname), "r") as f:
            for _, line in enumerate(f):
                one_doc = json.loads(line.strip())
                doc_id = one_doc["document_id"]
                if doc_fname.startswith(("train", "val", "test")):
                    assert doc_id not in documents
                    documents[doc_id] = one_doc
                else:
                    assert doc_id not in dictionary
                    dictionary[doc_id] = one_doc

    # get all of the train mentions
    print("Processing mentions...")

    splits = ["train", "val", "test"]

    for split in splits:
        print(f"Processing split: {split}")
        blink_mentions = []
        with open(os.path.join(in_dir, "mentions", split + ".json"), "r") as f:
            for line in tqdm(f):
                one_mention = json.loads(line.strip())
                label_doc = dictionary[one_mention["label_document_id"]]
                context_doc = documents[one_mention["context_document_id"]]
                start_index = one_mention["start_index"]
                end_index = one_mention["end_index"]
                context_doc_text = context_doc["text"]

                ######################################
                # THIS ASSUMES TEXT IS PRE-TOKENIZED
                ######################################
                # context_tokens = context_doc["text"].split()
                # extracted_mention = " ".join(
                #     context_tokens[start_index : end_index + 1]
                # )
                # assert extracted_mention == one_mention["text"]
                # context_left = " ".join(context_tokens[:start_index])
                # context_right = " ".join(context_tokens[end_index + 1 :])

                transformed_mention = {}
                transformed_mention["mention"] = one_mention["text"]
                transformed_mention["mention_id"] = one_mention["mention_id"]
                transformed_mention["context_left"] = context_doc_text[:start_index]
                transformed_mention["context_right"] = context_doc_text[end_index + 1 :]
                transformed_mention["context_doc_id"] = one_mention[
                    "context_document_id"
                ]
                transformed_mention["type"] = one_mention["category"]
                transformed_mention["label_id"] = one_mention["label_document_id"]
                transformed_mention["label"] = label_doc["text"]
                transformed_mention["label_title"] = label_doc["title"]
                blink_mentions.append(transformed_mention)
        print("Done.")
        # write all of the transformed train mentions
        print(f"Writing {len(blink_mentions)} processed mentions to file...")
        split_to_split = {"val": "valid"}
        with open(
            os.path.join(out_dir, split_to_split.get(split, split) + ".jsonl"), "w"
        ) as f:
            f.write("\n".join([json.dumps(m) for m in blink_mentions]))
        print("Done.")


if __name__ == "__main__":
    args = parse_args()

    for corpus in os.listdir(os.path.join(args.dir, "runs")):
        print(f"PREPROCESSING {corpus}...")

        out_dir = os.path.join(args.dir, "processed", corpus)
        os.makedirs(out_dir, exist_ok=True)

        preprocess_dictionary(
            in_dir=os.path.join(args.dir, "runs", corpus, "documents"), out_dir=out_dir
        )

        preprocess_mentions(
            in_dir=os.path.join(args.dir, "runs", corpus), out_dir=out_dir
        )
