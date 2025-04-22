import os
import json
import gc
from pathlib import Path
from typing import Dict, List

import torch
import pandas as pd
from datasets import Dataset
from tqdm.auto import tqdm
from transformers import TrainingArguments

from unsloth import FastLanguageModel
from trl import SFTTrainer

#path
BASE_DIR   = Path("/content/drive/MyDrive/poe_project")
DATA_DIR   = BASE_DIR / "data"
MODEL_DIR  = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"

for p in (DATA_DIR, MODEL_DIR, OUTPUT_DIR):
    p.mkdir(parents=True, exist_ok=True)


def prepare_training_data(csv_path: Path) -> Dataset:
    """
    Convert the input CSV into a Hugging Face `Dataset` of
    Alpaca‑formatted instruction/response pairs.

    Args
    ----
    csv_path : Path
        Path to the enhanced Poe CSV.

    Returns
    -------
    Dataset
        Hugging Face dataset with `instruction`, `input`, `output`.
    """
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path.name}")

    examples: List[Dict] = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Parsing rows"):
        text = str(row.get("text", "")).strip()
        if len(text) < 100:
            continue  # Skip extremely short or missing stories

        # Build metadata with graceful fallback for NaNs
        meta = {col: str(row.get(col, "")) for col in
                ["title", "classification", "characters",
                 "primary_setting", "location_type",
                 "atmosphere", "mood", "themes"]}

        instr = (
            "Write a short story in the style of Edgar Allan Poe with the "
            "following characteristics:\n"
            f"- Genre: {meta['classification']}\n"
            f"- Characters: {meta['characters']}\n"
            f"- Setting: {meta['primary_setting']}\n"
            f"- Atmosphere: {meta['atmosphere']}\n"
            f"- Mood: {meta['mood']}\n"
            f"- Themes: {meta['themes']}"
        )

        examples.append({
            "instruction": instr,
            "input": "",
            "output": text[:5000]  # cap length
        })

    # Persist raw JSON for reference
    json_path = DATA_DIR / "training_examples.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(examples, f, ensure_ascii=False, indent=2)

    print(f"Created {len(examples)} training examples → {json_path.name}")
    return Dataset.from_list(examples)


def formatting_prompts_func(batch) -> List[str]:
    """Alpaca template: `### Instruction … ### Response …`."""
    return [
        f"### Instruction:\n{ins}\n\n### Response:\n{out}"
        for ins, out in zip(batch["instruction"], batch["output"])
    ]


def train_model(dataset: Dataset,
                save_dir: Path = MODEL_DIR / "poe_llama_final"):
    """
    Fine‑tune Meta‑Llama‑3.1‑8B with QLoRA (8‑rank) on the provided dataset.

    Parameters
    ----------
    dataset : Dataset
        Prepared HF dataset from `prepare_training_data`.
    save_dir : Path
        Target folder to save the adapted model & tokenizer.
    """
    print("🚀 Initialising model …")
    gc.collect(); torch.cuda.empty_cache()

    base_name = "unsloth/Meta-Llama-3.1-8B"
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name     = base_name,
        max_seq_length = 2048,
        dtype          = None,
        load_in_4bit   = True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r                       = 8,
        target_modules          = ["q_proj", "k_proj", "v_proj", "o_proj",
                                   "gate_proj", "up_proj", "down_proj"],
        lora_alpha              = 16,
        lora_dropout            = 0.05,
        bias                    = "none",
        use_gradient_checkpointing = "unsloth",
        random_state            = 3407,
    )

    args = TrainingArguments(
        output_dir                   = MODEL_DIR / "poe_llama_tmp",
        num_train_epochs             = 3,
        per_device_train_batch_size  = 4,
        gradient_accumulation_steps  = 4,
        gradient_checkpointing       = True,
        learning_rate                = 2e-4,
        weight_decay                 = 0.01,
        bf16                         = True,
        max_grad_norm               = 0.3,
        warmup_ratio                 = 0.03,
        lr_scheduler_type            = "cosine",
        save_strategy                = "epoch",
        logging_steps                = 10,
        optim                        = "adamw_torch",
        report_to                    = "none",
        group_by_length              = True,
    )

    trainer = SFTTrainer(
        model           = model,
        args            = args,
        train_dataset   = dataset,
        tokenizer       = tokenizer,
        formatting_func = formatting_prompts_func,
        max_seq_length  = 1024,
        packing         = True,
    )

    print("🛠️  Starting training …")
    trainer.train()

    print(f"💾 Saving final LoRA model to {save_dir}")
    trainer.save_model(str(save_dir))


if __name__ == "__main__":
    CSV_PATH = DATA_DIR / "enhanced_poe_dataset.csv"
    poe_dataset = prepare_training_data(CSV_PATH)
    train_model(poe_dataset)
