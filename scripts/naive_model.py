import argparse
import json
from pathlib import Path
from typing import List, Dict

import torch
from tqdm.auto import tqdm
from unsloth import FastLanguageModel



def load_base_model(model_name: str = "unsloth/Meta-Llama-3.1-8B",
                    max_seq_len: int = 2048):
    """Load an off‑the‑shelf Llama‑3.1‑8B (4‑bit QLoRA)."""
    print("Loading base model …")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name     = model_name,
        max_seq_length = max_seq_len,
        dtype          = None,     # auto‑detect bfloat16 / float16
        load_in_4bit   = True,     # memory‑friendly
    )
    return model, tokenizer



def generate_story(prompt: str,
                   model,
                   tokenizer,
                   temperature: float = 0.7,
                   max_new_tokens: int = 1024) -> str:
    """
    Return the model’s response (without the prompt prefix).
    """
    # Build an Alpaca‑style prompt
    prefix = f"### Instruction:\n{prompt}\n\n### Response:\n"
    inputs = tokenizer(prefix, return_tensors="pt").to(model.device)

    # Sampling‑based generation
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens      = max_new_tokens,
            temperature         = temperature,
            do_sample           = True,
            top_p               = 0.92,
            top_k               = 50,
            repetition_penalty  = 1.05,
            pad_token_id        = tokenizer.eos_token_id,
            eos_token_id        = tokenizer.eos_token_id,
        )

    # Keep only the text after “### Response:”
    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return full_text.split("### Response:")[-1].strip()



def read_instructions(path: Path) -> List[Dict]:
    """Return a list of {'instruction': …} dicts from a JSONL file."""
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def write_outputs(path: Path, records: List[Dict]) -> None:
    """Write a list of dicts to JSONL (one dict per line)."""
    path.parent.mkdir(parents=True, exist_ok=True)# Ensure parent directories exist
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            json.dump(rec, f, ensure_ascii=False)
            f.write("\n")


# ────────────────────────────
# 4) Main pipeline
# ────────────────────────────
def main(args):
    model, tokenizer = load_base_model()          # load weights/tokenizer
    prompts = read_instructions(Path(args.input)) # read prompts
    print(f"📖 Loaded {len(prompts)} instructions")

    outputs = []
    for item in tqdm(prompts, desc="✍️  Generating"):
        story = generate_story(item["instruction"], model, tokenizer)
        outputs.append({"instruction": item["instruction"], "generated": story})

    write_outputs(Path(args.output), outputs)     # save JSONL
    print(" Done! Baseline outputs saved to", args.output)


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Baseline story generation")
    parser.add_argument("--input",  required=True, help="Path to instructions JSONL")
    parser.add_argument("--output", required=True, help="Path for generated JSONL")
    args = parser.parse_args()

    main(args)
