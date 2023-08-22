#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilities for training bi-encoder retrieval model
"""
import argparse
import json
import os
import subprocess
import tempfile
from typing import Union

from loguru import logger


def get_argument_parser():
    """Parse CLI arguments"""

    parser = argparse.ArgumentParser(description="Collect results from baseline")
    parser.add_argument(
        "--run",
        type=str,
        required=True,
        choices=("input", "output"),
        help="Either generate input or parse baseline output files",
    )
    parser.add_argument(
        "--in_dir",
        type=str,
        required=True,
        help="Directory where input to baseline is located ",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        help="Directory where output of baseline is located ",
        default=None,
    )
    parser.add_argument(
        "--belb_dir",
        type=str,
        required=True,
        help="Directory where all BELB data is stored",
    )
    parser.add_argument(
        "--ab3p",
        type=str,
        help="Path to Ab3p",
        default=None,
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=("dev", "test"),
        help="Which corpus split to use",
    )

    return parser


def load_json(path: str, *args, **kwargs) -> dict:
    """Load dict from JSON"""
    with open(path) as fp:
        item = json.load(fp, *args, **kwargs)

    return item


def save_json(path: str, item: Union[dict, list], *args, **kwargs):
    """Save dict to JSON"""
    with open(path, "w") as fp:
        json.dump(item, fp, *args, **kwargs)


def run_ab3p(ab3p_path: str, texts: list[str]) -> dict:
    """Use Ab3p to resolve abbreviations"""
    abbreviations: dict = {}

    full_ab3p_path = os.path.expanduser(ab3p_path)
    word_data_dir = os.path.join(full_ab3p_path, "WordData")

    # Temporarily create path file in the current working directory for Ab3P
    with open(os.path.join(os.getcwd(), "path_Ab3P"), "w") as path_file:
        path_file.write(f"{word_data_dir}{os.path.sep}\n")

    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8") as temp_file:
        for text in texts:
            temp_file.write(f"{text}\n")

        executable = os.path.join(full_ab3p_path, "identify_abbr")

        # Run Ab3P with the temp file containing the dataset
        # https://pylint.pycqa.org/en/latest/user_guide/messages/warning/subprocess-run-check.html
        try:
            out = subprocess.run(
                [executable, temp_file.name],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
            )
        except subprocess.CalledProcessError:
            logger.error(
                (
                    "The Ab3P could not be run on your system.",
                    "To ensure maximum accuracy, please install Ab3P yourself.",
                    "See https://github.com/ncbi-nlp/Ab3P",
                )
            )
            out = None

        if out is not None:
            result = out.stdout.decode("utf-8")
            if "Path file for type cshset does not exist!" in result:
                logger.error(
                    (
                        "A file path_Ab3p needs to exist in your current directory",
                        "with the path to the WordData directory",
                    )
                )
            elif "Cannot open" in result or "failed to open" in result:
                logger.error("Could not open the WordData directory for Ab3P!")

            lines = result.split("\n")

            current = None

            for line in lines:
                elems = line.split("|")

                if len(elems) == 2:
                    eid, _ = elems
                    if current != eid:
                        current = eid

                if current not in abbreviations:
                    abbreviations[current] = {}

                elif len(elems) == 3:
                    sf, lf, _ = elems
                    sf = sf.strip()
                    lf = lf.strip()
                    abbreviations[current][sf] = lf

    return abbreviations
