#!/usr/bin/env python3
"""Generate the same held-out prompts with the base model and LoRA adapter."""

import argparse
import json
from pathlib import Path


MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sample_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as source:
        rows = [json.loads(line) for line in source]
    if len(rows) < 3:
        raise ValueError("Validation data needs at least three examples")
    return [rows[0], rows[len(rows) // 2], rows[-1]]


def main() -> None:
    args = parse_args()
    # Imports stay here so --help works outside the Axolotl container.
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map="auto"
    ).eval()
    rows = sample_rows(args.validation)

    def generate(row: dict, current_model) -> str:
        messages = row["messages"][:-1]
        inputs = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
        ).to(next(current_model.parameters()).device)
        with torch.inference_mode():
            output = current_model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=96,
                pad_token_id=tokenizer.eos_token_id,
            )
        return tokenizer.decode(output[0, inputs["input_ids"].shape[-1] :], skip_special_tokens=True).strip()

    base_answers = [generate(row, model) for row in rows]
    adapted = PeftModel.from_pretrained(model, str(args.adapter)).eval()
    adapter_answers = [generate(row, adapted) for row in rows]
    comparisons = []
    for row, base, adapter in zip(rows, base_answers, adapter_answers):
        comparisons.append(
            {
                "prompt": row["messages"][-2]["content"],
                "reference": row["messages"][-1]["content"],
                "base": base,
                "adapter": adapter,
            }
        )
    args.output.write_text(json.dumps(comparisons, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for index, item in enumerate(comparisons, start=1):
        print(f"[{index}] Prompt: {item['prompt']}\n    Base: {item['base']}\n    LoRA: {item['adapter']}")
    print(f"Saved comparison: {args.output}")


if __name__ == "__main__":
    main()
