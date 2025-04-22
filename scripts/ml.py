import os
import re
import json
import pickle
import random
from collections import defaultdict, Counter
from pathlib import Path
from typing import Dict, List
import numpy as np
import pandas as pd
import torch  

# Define directories
BASE_DIR   = Path("/content/drive/MyDrive/poe_project")
DATA_DIR   = BASE_DIR / "data"
MODEL_DIR  = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"

# Make sure model & output folders exist
for p in (MODEL_DIR, OUTPUT_DIR):
    p.mkdir(parents=True, exist_ok=True)

# Load dataset
CSV_PATH = DATA_DIR / "enhanced_poe_dataset.csv"
df = pd.read_csv(CSV_PATH)
print(f"Loaded dataset with {len(df)} rows from {CSV_PATH.name}")


class PoeNgramModel:
    """Simple N-gram model (word-level), with temperature-based sampling."""

    def __init__(self, n: int = 3):
        self.n = n
        self.model = defaultdict(Counter)  # store n-gram → next word counts
        self.start_grams = []  # possible sentence starters

    @staticmethod
    def _clean(text: str) -> str:
        return re.sub(r"\s+", " ", text or "").strip()

    def _add_text(self, text: str) -> None:
        """Tokenize and update n-gram counts."""
        text = self._clean(text)
        tokens = text.split()
        if len(tokens) >= self.n:
            self.start_grams.append(tuple(tokens[:self.n]))
        for i in range(len(tokens) - self.n):
            gram = tuple(tokens[i : i + self.n])
            nxt = tokens[i + self.n]
            self.model[gram][nxt] += 1

    def train(self, texts: List[str]) -> None:
        print(f"Training {self.n}-gram model …")
        for txt in texts:
            self._add_text(txt)
        if not self.start_grams:
            self.start_grams = list(self.model.keys())[:10]
        print(f"Model trained with {len(self.model)} n-grams")

    def generate_text(self, max_length=200, temperature=1.0, prompt: str | None = None) -> str:
        """Generate text with optional prompt and adjustable randomness."""
        if not self.model:
            return "[ERROR] Model not trained."

        # Choose starting gram
        if prompt:
            tokens = self._clean(prompt).split()
            current = tuple(tokens[-self.n:]) if len(tokens) >= self.n else random.choice(self.start_grams)
            output = prompt
        else:
            current = random.choice(self.start_grams)
            output = " ".join(current)

        while len(output.split()) < max_length:
            if current not in self.model:
                current = random.choice(self.start_grams)
                output += ". " + " ".join(current)
                continue

            # Sample next word
            cand_words, counts = zip(*self.model[current].items())
            if temperature == 0:
                next_word = cand_words[np.argmax(counts)]
            else:
                probs = np.array(counts) ** (1.0 / temperature)
                probs /= probs.sum()
                next_word = np.random.choice(cand_words, p=probs)

            output += " " + next_word
            current = (*current[1:], next_word)

        return output

    def save(self, path: Path) -> None:
        """Save model to disk."""
        with path.open("wb") as f:
            pickle.dump({
                "n": self.n,
                "model": dict(self.model),
                "start_grams": self.start_grams
            }, f)
        print(f"Model saved → {path}")

    @classmethod
    def load(cls, path: Path) -> "PoeNgramModel":
        """Load model from disk."""
        with path.open("rb") as f:
            data = pickle.load(f)
        inst = cls(n=data["n"])
        for gram, cnt in data["model"].items():
            inst.model[gram].update(cnt)
        inst.start_grams = data["start_grams"]
        return inst


def train_all_ngram_models(n_values=[2, 3, 4]) -> Dict[int, PoeNgramModel]:
    """Train multiple models with different N values."""
    corpus = df["text"].dropna().tolist()
    models = {}
    for n in n_values:
        model = PoeNgramModel(n)
        model.train(corpus)
        model.save(MODEL_DIR / f"poe_{n}gram.pkl")
        models[n] = model
    return models


def extract_metadata(instruction: str) -> Dict[str, str]:
    """Parse bullet-style instruction into metadata dict."""
    meta = {}
    for line in instruction.splitlines():
        if line.startswith("- "):
            k, _, v = line[2:].partition(": ")
            if v and v.lower() != "nan":
                meta[k.lower()] = v
    return meta


def build_prompt(meta: Dict[str, str]) -> str:
    """Generate a stylized prompt based on metadata."""
    prompt = ""

    # Start based on genre
    genre = meta.get("genre", "").lower()
    if "horror" in genre:
        prompt = "In the depths of a shadowy realm, "
    elif "humor" in genre:
        prompt = "With a chuckle that echoed across the chamber, "
    elif "detective" in genre:
        prompt = "The mystery unfurled, drop by drop, "
    elif "essay" in genre:
        prompt = "Consider, if you will, the profound implications of "
    else:
        prompt = "It began, as such tales often do, with "

    # Add tone + setting
    atmos = meta.get("atmosphere", "")
    mood = meta.get("mood", "")
    if atmos and mood:
        prompt += f"a {random.choice(atmos.split(', ')).lower()}, "
        prompt += f"{random.choice(mood.split(', ')).lower()} "

    setting = meta.get("setting", "") or meta.get("primary_setting", "")
    prompt += f"setting in {setting or 'the night'}. "

    # Add character name
    chars = meta.get("characters", "")
    m = re.search(r"([A-Z][a-z]+(?: [A-Z][a-z]+)?)", chars)
    if m:
        prompt += f"{m.group(1)} stood, contemplating the scene before him. "

    # Add theme
    themes = meta.get("themes", "")
    if themes:
        prompt += f"The notion of {random.choice(themes.split(', ')).strip().lower()} weighed heavily upon my mind. "

    return prompt


def generate_batches(model: PoeNgramModel, instructions: List[Dict[str, str]], temperature=1.2, max_len=600) -> List[Dict]:
    """Generate text for each instruction and save results."""
    results = []
    print(f"Generating {len(instructions)} stories …")
    for idx, inst in enumerate(instructions, 1):
        meta = extract_metadata(inst["instruction"])
        prompt = build_prompt(meta)
        story = model.generate_text(max_length=max_len, temperature=temperature, prompt=prompt)
        results.append({
            "instruction": inst["instruction"],
            "metadata": meta,
            "prompt": prompt,
            "generated_text": story
        })
        print(f"  ✓ {idx}/{len(instructions)} finished")

    # Save to JSON
    json_out = OUTPUT_DIR / "generated_stories.json"
    with json_out.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # Save readable text
    txt_out = OUTPUT_DIR / "generated_stories.txt"
    with txt_out.open("w", encoding="utf-8") as f:
        for rec in results:
            f.write("=== Instruction ===\n" + rec["instruction"] + "\n\n")
            f.write("=== Prompt ===\n" + rec["prompt"] + "\n\n")
            f.write("=== Generated ===\n" + rec["generated_text"] + "\n")
            f.write("\n" + "=" * 80 + "\n\n")

    print(f"Saved JSON → {json_out}\nSaved TXT  → {txt_out}")
    return results


if __name__ == "__main__":
    # Step 1: train N-gram models
    models = train_all_ngram_models()

    # Step 2: test generation with one prompt
    sample_meta = {
        "atmosphere": "gloomy",
        "mood": "melancholic",
        "primary_setting": "ancient mansion",
        "themes": "death, madness, isolation",
    }
    print("\nOne-shot 3-gram sample:\n")
    print(models[3].generate_text(prompt=build_prompt(sample_meta), temperature=1.0, max_length=120))

    # Step 3: generate multiple stories
    demo_instructions = [{"instruction": row} for row in df["instruction"].dropna().head(10)]
    generate_batches(models[3], demo_instructions)
