import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, List

import pandas as pd
from openai import OpenAI
from tqdm.auto import tqdm

RUBRIC_PROMPT = """
You are a severe literary critic. Score the story on seven dimensions
(0‑10 each) and add one short remark about the main weakness.

Dimensions
1. poe_style        – Fidelity to Poe's gothic diction / imagery
2. coherence        – Logical narrative flow
3. suspense         – Intensity and pacing
4. instruction_fit  – Fulfils every element of the prompt
5. creativity       – Originality while staying Poe‑like
6. language_quality – Grammar, rhythm, rhetorical richness
7. redundancy       – Unnecessary repetition (0 = very redundant)

Return STRICT JSON, e.g.:
{
  "poe_style": 8,
  "coherence": 6,
  "suspense": 7,
  "instruction_fit": 9,
  "creativity": 5,
  "language_quality": 8,
  "redundancy": 2,
  "comment": "Atmosphere is gothic but the ending feels rushed"
}
""".strip()

MODEL_NAME = "deepseek-chat" 


def create_client(api_key: str | None = None) -> OpenAI:
    """Return an OpenAI‑compatible DeepSeek client."""
    api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("DeepSeek API key not provided.")
    return OpenAI(api_key=api_key, base_url="https://api.deepseek.com")


def ask_deepseek(client: OpenAI, instruction: str, story: str) -> Dict:
    """
    Send one prompt+story to DeepSeek‑chat; parse JSON result.

    Falls back to regex extraction if stray text surrounds the JSON.
    """
    # Ensure the instruction and story are non‑empty
    user_msg = (
        f"### Prompt:\n{instruction}\n\n"
        f"### Story:\n{story}\n\n"
        "### Task:\nReturn the JSON described above."
    )# Ensure the prompt is well‑formatted
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": RUBRIC_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.0,
        max_tokens=256,
    )
    content = resp.choices[0].message.content
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", content, re.S)
        if not match:
            raise ValueError("DeepSeek returned non‑JSON content.")
        return json.loads(match.group())


def weighted_overall(scores: Dict) -> float:
    """Compute the aggregate score with redundancy as a penalty."""
    return (
        0.25 * scores["poe_style"]
        + 0.20 * scores["coherence"]
        + 0.15 * scores["suspense"]
        + 0.15 * scores["instruction_fit"]
        + 0.15 * scores["language_quality"]
        + 0.10 * scores["creativity"]
        - 0.10 * scores["redundancy"]
    )


def evaluate_file(input_path: Path, output_path: Path, api_key: str | None) -> None:
    """Run DeepSeek evaluation on every record in a JSONL input file."""
    client = create_client(api_key)
    records: List[Dict] = [json.loads(l) for l in input_path.open(encoding="utf-8")]
    # Ensure input is a list of dicts with 'instruction' and 'generated' keys
    results: List[Dict] = []
    for rec in tqdm(records, desc="Judging"):
        scores = ask_deepseek(client, rec["instruction"], rec["generated"])
        scores["overall"] = round(weighted_overall(scores), 2)
        results.append({**rec, **scores})

    # Write JSONL
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for line in results:
            json.dump(line, f, ensure_ascii=False)
            f.write("\n")

    # Mirror to CSV
    pd.DataFrame(results).to_csv(
        output_path.with_suffix(".csv"), index=False, encoding="utf-8"
    )

    print("\n✅ Scoring complete!")
    print("•", output_path)
    print("•", output_path.with_suffix(".csv"))



def parse_args() -> argparse.Namespace:
    """CLI → python poe_deepseek_judge.py --input file.jsonl --output scores.jsonl"""
    p = argparse.ArgumentParser(description="DeepSeek 7‑dimension judge for Poe stories")
    p.add_argument("--input", required=True, help="Path to JSONL with model outputs")
    p.add_argument("--output", required=True, help="Path for JSONL scores")
    p.add_argument("--key", help="DeepSeek API key (optional, else env var)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    evaluate_file(Path(args.input), Path(args.output), args.key)


if __name__ == "__main__":
    main()
