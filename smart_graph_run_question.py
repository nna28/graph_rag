import argparse
import json
import sys
from typing import Iterable, Optional

import pandas as pd
from dotenv import load_dotenv

from src.graph_rag import SmartGraphRAG, load_tiny_vietnamese_llm
from src.init_graph import init


def _iter_questions(csv_path: str) -> Iterable[str]:
    """Yield questions from a CSV file (uses the first column if unnamed)."""
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    question_col = "Question" if "Question" in df.columns else df.columns[0]
    for value in df[question_col].dropna().astype(str):
        question = value.strip()
        if question:
            yield question


def _vector_count(store) -> Optional[int]:
    try:
        return store._collection.count()  # type: ignore[attr-defined]
    except Exception:
        return None


def _ask(rag: SmartGraphRAG, question: str, args) -> str:
    return rag.query(
        question,
        depth=args.depth,
        max_hops=args.max_hops,
        top_k_paths=args.top_k_paths,
        anchor_per_entity=args.anchor_per_entity,
        max_anchors=args.max_anchors,
        neighbor_top_k=args.neighbor_top_k,
        neighbor_candidate_multiplier=args.neighbor_multiplier,
        path_candidate_multiplier=args.path_multiplier,
    )


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Run SmartGraphRAG with a local <1B LLM (Qwen2.5-0.5B-Instruct)."
    )
    parser.add_argument("--question", help="Single question to ask.")
    parser.add_argument(
        "--questions-file",
        help="CSV file containing questions (column name 'Question' or use the first column).",
    )
    parser.add_argument(
        "--output-json",
        default="smart_graph_results.json",
        help="Path to save Q&A JSON (list of {question, answer}).",
    )
    parser.add_argument("--depth", type=int, default=1, help="Ego-graph depth.")
    parser.add_argument("--max-hops", type=int, default=3, help="Max hops for multi-hop paths.")
    parser.add_argument("--top-k-paths", type=int, default=2, help="Number of reranked paths to keep.")
    parser.add_argument(
        "--anchor-per-entity",
        type=int,
        default=3,
        help="Anchors retrieved per extracted entity.",
    )
    parser.add_argument(
        "--max-anchors",
        type=int,
        default=10,
        help="Global cap on anchor nodes.",
    )
    parser.add_argument(
        "--neighbor-top-k",
        type=int,
        default=4,
        help="Nearest edges kept after reranking.",
    )
    parser.add_argument(
        "--neighbor-multiplier",
        type=int,
        default=3,
        help="Oversampling multiplier before neighbor rerank.",
    )
    parser.add_argument(
        "--path-multiplier",
        type=int,
        default=3,
        help="Oversampling multiplier before path rerank.",
    )
    args = parser.parse_args()

    if not args.question and not args.questions_file:
        parser.error("Provide --question or --questions-file.")

    llm = load_tiny_vietnamese_llm()
    rag = SmartGraphRAG(llm_model=llm)
    init(rag)

    count = _vector_count(rag.vector_store)
    if count == 0:
        print(
            "Warning: vector store is empty; run build_embed.py to seed embeddings for better results.",
            file=sys.stderr,
        )

    results = []
    if args.questions_file:
        for question in _iter_questions(args.questions_file):
            answer = _ask(rag, question, args)
            results.append({"question": question, "answer": answer})
            print(f"Q: {question}\nA: {answer}\n")
    if args.question:
        answer = _ask(rag, args.question, args)
        results.append({"question": args.question, "answer": answer})
        print(f"Q: {args.question}\nA: {answer}")

    if results:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"Saved {len(results)} entries to {args.output_json}")


if __name__ == "__main__":
    main()
