import argparse
import json
import sys
from typing import Iterable, List, Tuple

import pandas as pd
import requests
from dotenv import load_dotenv


def _iter_questions(csv_path: str) -> Iterable[str]:
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    question_col = "Question" if "Question" in df.columns else df.columns[0]
    cnt = 0
    for value in df[question_col].dropna().astype(str):
        if cnt > 100: 
            break
        question = value.strip()
        if question:
            yield question
        cnt += 1


def _ask_server(session: requests.Session, base_url: str, question: str, timeout: int) -> Tuple[bool, str]:
    url = base_url.rstrip("/") + "/api/chat"
    try:
        resp = session.post(url, json={"message": question}, timeout=timeout)
    except requests.RequestException as exc:
        return False, f"request_error: {exc}"

    if not resp.ok:
        return False, f"http_{resp.status_code}: {resp.text}"

    try:
        data = resp.json()
    except ValueError:
        return False, f"invalid_json: {resp.text}"

    answer = data.get("response")
    if answer is None:
        return False, f"missing_response_field: {data}"
    return True, str(answer)


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Query SmartGraphRAG via the running FastAPI server (default http://localhost:8000/api/chat)."
    )
    parser.add_argument("--question", help="Single question to ask.")
    parser.add_argument(
        "--questions-file",
        help="CSV file containing questions (column name 'Question' or use the first column).",
    )
    parser.add_argument(
        "--server-url",
        default="http://localhost:8000",
        help="Base URL of the GraphRAG server (without /api/chat).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="Request timeout in seconds.",
    )
    parser.add_argument(
        "--output-json",
        default="smart_graph_server_results.json",
        help="Path to save Q&A JSON (list of {question, answer, ok}).",
    )
    args = parser.parse_args()

    if not args.question and not args.questions_file:
        parser.error("Provide --question or --questions-file.")

    session = requests.Session()

    results: List[dict] = []

    if args.questions_file:
        for question in _iter_questions(args.questions_file):
            ok, answer = _ask_server(session, args.server_url, question, args.timeout)
            results.append({"question": question, "answer": answer, "ok": ok})
            print(f"Q: {question}\nA: {answer}\n")

    if args.question:
        ok, answer = _ask_server(session, args.server_url, args.question, args.timeout)
        results.append({"question": args.question, "answer": answer, "ok": ok})
        print(f"Q: {args.question}\nA: {answer}")

    if results:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"Saved {len(results)} entries to {args.output_json}")


if __name__ == "__main__":
    main()
