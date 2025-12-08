# src/00_utils/clean_nodes.py
import pandas as pd
import re
import os
import unicodedata

INPUT_FILE = r"C:\Users\ADMIN\Downloads\professors\data\raw\nodes_raw.csv"
OUTPUT_FILE = r"C:\Users\ADMIN\Downloads\professors\data\cleaned\nodes_cleaned.csv"


# ==========================
# UTILS
# ==========================
def strip_accents(s: str) -> str:
    """Remove accents for canonical name."""
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


def clean_wiki_url(url: str) -> str:
    if isinstance(url, str) and url.startswith("/wiki/"):
        return url.strip()
    return ""


def clean_name(name: str) -> str:
    if not isinstance(name, str):
        return ""

    name = name.strip()

    # Remove prefixes “Tạo”, “Create”
    name = re.sub(r"^\s*(Tạo|tao|TẠO|Create|CREATE)\s+", "", name, flags=re.IGNORECASE)

    # Clean percent encoding
    name = re.sub(r"%[0-9A-Fa-f]{2}", " ", name)

    name = re.sub(r"\s+", " ", name).strip()
    return name


def canonical(n: str) -> str:
    """Normalized name used for deduplication."""
    if not isinstance(n, str):
        return ""
    n = n.lower().strip()
    n = strip_accents(n)
    n = re.sub(r"[^a-z0-9\s\-']", " ", n)
    n = re.sub(r"\s+", " ", n).strip()
    return n


# ==========================
# MAIN
# ==========================
def main():
    print("Loading:", INPUT_FILE)
    df = pd.read_csv(INPUT_FILE)

    # Standardize column names
    df.columns = [c.strip().lower() for c in df.columns]

    # Ensure required columns exist
    if "link" not in df.columns:
        raise ValueError("Missing column: link")
    if "name" not in df.columns:
        raise ValueError("Missing column: name")

    # Clean
    df["link"] = df["link"].apply(clean_wiki_url)
    df["name"] = df["name"].apply(clean_name)

    # Generate canonical_name
    df["canonical_name"] = df["name"].apply(canonical)

    # Drop invalid
    df = df[df["link"] != ""]
    df = df[df["name"].str.len() > 1]

    # Correct dedupe
    df = df.drop_duplicates(subset=["canonical_name"], keep="first")

    df = df.reset_index(drop=True)

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)

    print("Cleaned file saved to:", OUTPUT_FILE)
    print("Rows:", len(df))


if __name__ == "__main__":
    main()
