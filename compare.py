#!/usr/bin/env python3
"""Generate the same held-out prompts with the base model and LoRA adapter."""

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path)
    return parser.parse_args()


def load_training_config(path: Path) -> tuple[str, Path]:
    import yaml

    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    return config["base_model"], Path(config["test_datasets"][0]["path"])


def sample_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as source:
        rows = [json.loads(line) for line in source]
    if len(rows) < 3:
        raise ValueError("Validation data needs at least three examples")
    return [rows[0], rows[len(rows) // 2], rows[-1]]


def prompt_messages(row: dict) -> list[dict]:
    messages = row.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        raise ValueError("Comparison rows need at least one user/assistant exchange")
    if any(message.get("role") == "system" for message in messages):
        raise ValueError("Comparison prompts must not contain a system message")
    if messages[-1].get("role") != "assistant":
        raise ValueError("Comparison rows must end with an assistant reference")
    prompt = messages[:-1]
    if not prompt or prompt[-1].get("role") != "user":
        raise ValueError("Comparison prompts must end with a user message")
    return prompt


def main() -> None:
    args = parse_args()
    model_id, validation_path = load_training_config(args.config)
    # Imports stay here so --help works outside the Axolotl container.
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.bfloat16, device_map="auto"
    ).eval()
    rows = sample_rows(validation_path)
    prompts = [prompt_messages(row) for row in rows]

    def generate(messages: list[dict], current_model) -> str:
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

    # Reuse the exact same prompt objects for base and adapter generation.
    base_answers = [generate(prompt, model) for prompt in prompts]
    adapted = PeftModel.from_pretrained(model, str(args.adapter)).eval()
    adapter_answers = [generate(prompt, adapted) for prompt in prompts]
    comparisons = []
    for row, prompt, base, adapter in zip(rows, prompts, base_answers, adapter_answers):
        comparisons.append(
            {
                "prompt": prompt[-1]["content"],
                "reference": row["messages"][-1]["content"],
                "base": base,
                "adapter": adapter,
            }
        )
    args.output.write_text(json.dumps(comparisons, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.markdown_output:
        sections = []
        for index, item in enumerate(comparisons, start=1):
            sections.append(
                f"## Example {index}\n\n"
                f"**Prompt:** {item['prompt']}\n\n"
                f"**Reference:** {item['reference']}\n\n"
                f"**Base model:** {item['base']}\n\n"
                f"**LoRA adapter:** {item['adapter']}\n"
            )
        args.markdown_output.write_text("\n".join(sections), encoding="utf-8")
    for index, item in enumerate(comparisons, start=1):
        print(f"[{index}] Prompt: {item['prompt']}\n    Base: {item['base']}\n    LoRA: {item['adapter']}")
    print(f"Saved comparison: {args.output}")
    if args.markdown_output:
        print(f"Saved readable comparison: {args.markdown_output}")


if __name__ == "__main__":
    main()
