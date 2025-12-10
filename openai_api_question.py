import argparse
import json
import os
import sys
from typing import Tuple

import pandas as pd
from dotenv import load_dotenv

try:
    from openai import OpenAI
except ImportError as exc:
    raise SystemExit("Install the openai package to use this script (pip install openai).") from exc


def _iter_questions(csv_path: str) -> Tuple[pd.DataFrame, str]:
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    question_col = "Question" if "Question" in df.columns else df.columns[0]
    return df, question_col


def _ask_openai(client: OpenAI, model: str, question: str, temperature: float) -> str:
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You answer short factual questions in Vietnamese."},
            {"role": "user", "content": question},
        ],
        temperature=temperature,
    )
    return completion.choices[0].message.content.strip()


def _answer_sheet(
    client: OpenAI,
    model: str,
    temperature: float,
    input_path: str,
    output_path: str,
) -> Tuple[pd.DataFrame, str, list]:
    df, question_col = _iter_questions(input_path)

    answers = []
    cleaned_questions = []
    cnt = 0
    for question in df[question_col].fillna("").astype(str):
        q = question.strip()
        if not q:
            answers.append("")
            cleaned_questions.append(q)
            continue
        answers.append(_ask_openai(client, model, q, temperature))
        cleaned_questions.append(q)
        cnt += 1
        if cnt > 100:
            break
        print(answers[-1])
    results = [{"question": q, "answer": a} for q, a in zip(cleaned_questions, answers)]
    return results


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Answer questions directly with OpenAI API (single question or CSV sheet)."
    )
    parser.add_argument("--question", help="Single question to ask.")
    parser.add_argument(
        "--questions-file",
        help="CSV file with a 'Question' column (or first column) to produce an answer sheet.",
    )
    parser.add_argument(
        "--output",
        default="openai_answers.csv",
        help="Output CSV path when using --questions-file.",
    )
    parser.add_argument(
        "--output-json",
        default="openai_answers.json",
        help="Output JSON path for Q&A pairs.",
    )
    parser.add_argument("--model", default="gpt-3.5-turbo", help="OpenAI chat model id.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Generation temperature.")
    args = parser.parse_args()

    if not args.question and not args.questions_file:
        parser.error("Provide --question or --questions-file.")

    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY is missing; load it in .env or your environment.", file=sys.stderr)

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    results = []
    if args.questions_file:
        sheet_results = _answer_sheet(
            client, args.model, args.temperature, args.questions_file, args.output
        )
        results.extend(sheet_results)

    if args.question:
        answer = _ask_openai(client, args.model, args.question, args.temperature)
        results.append({"question": args.question, "answer": answer})
        print(f"Q: {args.question}\nA: {answer}")

    if results:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"Saved {len(results)} entries to {args.output_json}")


if __name__ == "__main__":
    main()
