"""Script to extract NER labels."""
import argparse
from pathlib import Path

import pandas as pd

LABEL_FILE = "labels.txt"


def parse_args():
    parser = argparse.ArgumentParser(
        "Extract labels from NER data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "src", type=Path, help="Directory the train, dev and test NER files"
    )
    parser.add_argument("--suffix", type=str, default=".tsv", help="NER file types")
    return parser.parse_args()


def main(args):
    df = pd.concat(
        [
            pd.read_csv(p, sep="\t", names=["word", "label"])
            for p in args.src.rglob(f"*{args.suffix}")
        ],
        ignore_index=True,
    )

    labels = df["label"].unique().tolist()

    with open(args.src / LABEL_FILE, "w+") as f:
        f.write("\n".join(labels))


if __name__ == "__main__":
    args = parse_args()
    main(args)
