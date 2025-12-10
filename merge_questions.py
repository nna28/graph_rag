import argparse
from pathlib import Path
from typing import List

import pandas as pd


def _load_one(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    if "Question" not in df.columns:
        df = df.rename(columns={df.columns[0]: "Question"})
    if len(df.columns) > 1 and "Ground_Truth" not in df.columns:
        df = df.rename(columns={df.columns[1]: "Ground_Truth"})
    keep_cols = [col for col in ["Question", "Ground_Truth"] if col in df.columns]
    return df[keep_cols]


def merge_csvs(inputs: List[Path], output: Path, drop_duplicates: bool) -> None:
    frames = []
    for path in inputs:
        if path.exists():
            frames.append(_load_one(path))
        else:
            print(f"Warning: {path} not found, skipping.")
    if not frames:
        raise SystemExit("No input files found.")

    merged = pd.concat(frames, ignore_index=True)
    if drop_duplicates:
        merged = merged.drop_duplicates(subset=[col for col in merged.columns])

    merged.to_csv(output, index=False, encoding="utf-8-sig")
    print(f"Saved {len(merged)} rows to {output}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge question CSVs into a single file (default all_questions.csv)."
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=["question_1.csv", "question_2.csv", "questions.csv"],
        help="List of input CSV files to merge.",
    )
    parser.add_argument(
        "--output",
        default="all_questions.csv",
        help="Output CSV filename.",
    )
    parser.add_argument(
        "--drop-duplicates",
        action="store_true",
        help="Drop duplicate rows after merging.",
    )
    args = parser.parse_args()

    input_paths = [Path(p) for p in args.inputs]
    output_path = Path(args.output)

    merge_csvs(input_paths, output_path, args.drop_duplicates)


if __name__ == "__main__":
    main()
