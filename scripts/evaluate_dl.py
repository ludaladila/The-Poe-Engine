"""
Batch‑generate Poe‑style stories with a fine‑tuned Unsloth model.

This script:
1. Loads the LoRA‑adapted Llama model (`poe_llama_final`)
2. Reads a JSONL file of instructions
3. Generates a response for each instruction
4. Writes all pairs to a new JSONL file
"""

import os
import json
import torch
from tqdm import tqdm
from unsloth import FastLanguageModel


def load_model(model_path: str):
    """Return (model, tokenizer) from a given Unsloth directory."""
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_path,
        max_seq_length=2048,
        dtype=None,          # auto‑detect bf16 / fp16
        load_in_4bit=True,   # VRAM‑friendly
    )
    return model, tokenizer

def inference(
    prompt: str,
    model,
    tokenizer,
    max_new_tokens: int = 1024,
    temperature: float = 0.7,
) -> str:
    """
    Generate one Poe‑style passage from a raw instruction string.
    """
    full_prompt = f"### Instruction:\n{prompt}\n\n### Response:\n"
    inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
    # Generate text with sampling
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,#
            temperature=temperature,
            do_sample=True,
            top_p=0.92,
            top_k=50,
            repetition_penalty=1.05,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    # Decode the output and extract the response
    text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return text.split("### Response:")[-1].strip()

# Constants for file paths
MODEL_PATH  = "/content/drive/MyDrive/poe_project/models/poe_llama_final"
INPUT_FILE  = "/content/drive/MyDrive/poe_project/data/instructions_last10.jsonl"
OUTPUT_FILE = "/content/drive/MyDrive/poe_project/data/generated_last10.jsonl"

# Main function to load model, iterate prompts, and write JSONL
def main() -> None:
    """Load model → iterate prompts → write JSONL."""
    model, tokenizer = load_model(MODEL_PATH)
    # Ensure the model is in evaluation mode
    with open(INPUT_FILE, encoding="utf-8") as f:
        instructions = [json.loads(line) for line in f]
    # Check if instructions are loaded
    results = []
    for item in tqdm(instructions, desc="Generating"):
        story = inference(item["instruction"], model, tokenizer)
        results.append({"instruction": item["instruction"], "generated": story})
    # Write results to JSONL
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for rec in results:
            json.dump(rec, f, ensure_ascii=False)
            f.write("\n")# Ensure each record is on a new line

    print("✅ Generation complete! File saved to:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
