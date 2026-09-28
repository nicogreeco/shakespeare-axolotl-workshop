#!/usr/bin/env python3
"""Prepare Shakespeare or public-domain character dialogue data for Axolotl."""

import argparse
import hashlib
import json
import random
import re
import shutil
import subprocess
import urllib.request
from collections.abc import Iterator
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"

SHAKESPEARE_SOURCE = DATA_DIR / "shakespeare_unformatted.jsonl"
SHAKESPEARE_REVISION = "ea9ab49c0367f9efac40af869c53c1b956c626a5"
SHAKESPEARE_SOURCE_URL = (
    "https://huggingface.co/datasets/chaseharmon/6.7960_Shakespeare/"
    f"resolve/{SHAKESPEARE_REVISION}/shakespeare_unformatted.jsonl"
)
SHAKESPEARE_SOURCE_SHA256 = "d09f91f36fa59230e94519c15bb833a623b38e885de899eb0f420c37f6c8bf5d"
SHAKESPEARE_TRAIN_COUNT = 4000
SHAKESPEARE_VALIDATION_COUNT = 200

PUBLIC_DOMAIN_SOURCE = DATA_DIR / "public_domain" / "train.jsonl.zst"
PUBLIC_DOMAIN_OUTPUT_DIR = DATA_DIR / "public_domain" / "processed"
PUBLIC_DOMAIN_CHARACTERS = {
    "count-dracula": "Count Dracula",
    "sherlock-holmes": "Sherlock Holmes",
    "the-cheshire-cat": "The Cheshire Cat",
    "odysseus": "Odysseus",
    "dorian-gray": "Dorian Gray",
}
PUBLIC_DOMAIN_SEED = 42
PUBLIC_DOMAIN_VALIDATION_RATIO = 0.10
PUBLIC_DOMAIN_TEST_RATIO = 0.10
MAX_EXCHANGES_PER_EXAMPLE = 2
MAX_WORDS_PER_EXAMPLE = 500
MIN_SPOKEN_WORDS_FOR_WINDOW_START = 6
STAGE_DIRECTION = re.compile(r"\*[^*]+\*", re.DOTALL)
STAGE_DIRECTION_ONLY = re.compile(r"\s*(?:\*[^*]+\*\s*)+", re.DOTALL)


def download_shakespeare_source() -> None:
    if not SHAKESPEARE_SOURCE.exists():
        print(f"Downloading {SHAKESPEARE_SOURCE_URL}")
        request = urllib.request.Request(
            SHAKESPEARE_SOURCE_URL,
            headers={"User-Agent": "shakespeare-workshop/1.0"},
        )
        temporary = SHAKESPEARE_SOURCE.with_suffix(".download")
        try:
            with urllib.request.urlopen(request, timeout=60) as response, temporary.open("wb") as output:
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
            temporary.replace(SHAKESPEARE_SOURCE)
        finally:
            temporary.unlink(missing_ok=True)
    else:
        print(f"Using existing source: {SHAKESPEARE_SOURCE}")
    digest = hashlib.sha256(SHAKESPEARE_SOURCE.read_bytes()).hexdigest()
    if digest != SHAKESPEARE_SOURCE_SHA256:
        raise ValueError(
            f"Source checksum mismatch: {digest}; expected {SHAKESPEARE_SOURCE_SHA256}"
        )


def load_shakespeare_pairs() -> tuple[list[tuple[str, str]], int, int]:
    pairs = []
    seen = set()
    empty_count = duplicate_count = 0
    with SHAKESPEARE_SOURCE.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            row = json.loads(line)
            src, trg = row.get("src"), row.get("trg")
            if not isinstance(src, str) or not isinstance(trg, str):
                raise ValueError(f"Line {line_number}: src and trg must be strings")
            pair = (src.strip(), trg.strip())
            if not all(pair):
                empty_count += 1
            elif pair in seen:
                duplicate_count += 1
            else:
                seen.add(pair)
                pairs.append(pair)
    return pairs, empty_count, duplicate_count


def write_shakespeare_messages(path: Path, pairs: list[tuple[str, str]]) -> None:
    with path.open("w", encoding="utf-8") as output:
        for src, trg in pairs:
            row = {
                "messages": [
                    {"role": "user", "content": src},
                    {"role": "assistant", "content": trg},
                ]
            }
            output.write(json.dumps(row, ensure_ascii=False) + "\n")


def prepare_shakespeare() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    download_shakespeare_source()
    pairs, empty_count, duplicate_count = load_shakespeare_pairs()
    required = SHAKESPEARE_TRAIN_COUNT + SHAKESPEARE_VALIDATION_COUNT
    if len(pairs) < required:
        raise ValueError(f"Need {required} usable pairs; found {len(pairs)}")
    # Keep a gap between splits because nearby source rows are consecutive exchanges.
    train = pairs[:SHAKESPEARE_TRAIN_COUNT]
    validation = pairs[-SHAKESPEARE_VALIDATION_COUNT:]
    assert set(train).isdisjoint(validation)
    write_shakespeare_messages(DATA_DIR / "train.jsonl", train)
    write_shakespeare_messages(DATA_DIR / "validation.jsonl", validation)
    print(
        f"Prepared {len(train)} train and {len(validation)} validation examples "
        f"({empty_count} empty, {duplicate_count} duplicate pairs skipped)."
    )


def iter_zstd_jsonl(path: Path) -> Iterator[tuple[int, dict]]:
    if not path.is_file():
        raise FileNotFoundError(f"Public-domain source not found: {path}")
    if shutil.which("zstdcat") is None:
        raise RuntimeError("zstdcat is required to read train.jsonl.zst")

    process = subprocess.Popen(
        ["zstdcat", str(path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
    )
    assert process.stdout is not None
    assert process.stderr is not None
    try:
        for line_number, line in enumerate(process.stdout, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON on source line {line_number}: {error}") from error
            if not isinstance(row, dict):
                raise ValueError(f"Source line {line_number}: expected a JSON object")
            yield line_number, row
    finally:
        process.stdout.close()
        stderr = process.stderr.read()
        return_code = process.wait()
        if return_code:
            raise RuntimeError(f"zstdcat failed with exit code {return_code}: {stderr.strip()}")


def validate_source_conversation(row: dict, line_number: int) -> None:
    conversation = row.get("conversations")
    if not isinstance(conversation, list) or len(conversation) < 3:
        raise ValueError(f"Source line {line_number}: conversations must contain at least three turns")
    if (len(conversation) - 1) % 2:
        raise ValueError(f"Source line {line_number}: conversation has an incomplete exchange")

    expected_roles = ["system"] + [
        "human" if index % 2 else "gpt" for index in range(1, len(conversation))
    ]
    actual_roles = [turn.get("from") for turn in conversation]
    if actual_roles != expected_roles:
        raise ValueError(
            f"Source line {line_number}: expected alternating system/human/gpt roles; "
            f"found {actual_roles}"
        )
    for turn_index, turn in enumerate(conversation):
        content = turn.get("value")
        if not isinstance(content, str) or not content.strip():
            raise ValueError(f"Source line {line_number}, turn {turn_index}: empty content")


def load_public_domain_conversations(character_names: set[str]) -> dict[str, list[dict]]:
    conversations = {name: [] for name in character_names}
    for line_number, row in iter_zstd_jsonl(PUBLIC_DOMAIN_SOURCE):
        character_name = row.get("ai_character")
        if character_name not in character_names:
            continue
        validate_source_conversation(row, line_number)
        conversations[character_name].append(
            {
                "source_line": line_number,
                "turns": row["conversations"],
            }
        )
    return conversations


def split_conversations(conversations: list[dict], seed: int) -> dict[str, list[dict]]:
    if len(conversations) < 3:
        raise ValueError(
            f"Need at least three original conversations for train/validation/test; "
            f"found {len(conversations)}"
        )

    shuffled = list(conversations)
    random.Random(seed).shuffle(shuffled)
    validation_count = max(1, round(len(shuffled) * PUBLIC_DOMAIN_VALIDATION_RATIO))
    test_count = max(1, round(len(shuffled) * PUBLIC_DOMAIN_TEST_RATIO))
    train_count = len(shuffled) - validation_count - test_count
    if train_count < 1:
        raise ValueError(f"Not enough conversations for a non-empty train split: {len(shuffled)}")

    splits = {
        "train": shuffled[:train_count],
        "validation": shuffled[train_count : train_count + validation_count],
        "test": shuffled[train_count + validation_count :],
    }
    source_lines = {
        split_name: {conversation["source_line"] for conversation in split_conversations}
        for split_name, split_conversations in splits.items()
    }
    assert source_lines["train"].isdisjoint(source_lines["validation"])
    assert source_lines["train"].isdisjoint(source_lines["test"])
    assert source_lines["validation"].isdisjoint(source_lines["test"])
    return splits


def word_count(messages: list[dict]) -> int:
    return sum(len(message["content"].split()) for message in messages)


def is_stage_direction_only(content: str) -> bool:
    return STAGE_DIRECTION_ONLY.fullmatch(content) is not None


def has_enough_standalone_context(content: str) -> bool:
    spoken_text = STAGE_DIRECTION.sub(" ", content)
    return len(spoken_text.split()) >= MIN_SPOKEN_WORDS_FOR_WINDOW_START


def build_public_domain_examples(conversations: list[dict]) -> tuple[list[dict], dict[str, int]]:
    examples = []
    stage_only_skipped = 0
    context_dependent_skipped = 0
    overlong_skipped = 0
    assistant_targets = 0

    for conversation in conversations:
        turns = conversation["turns"]
        pending_messages: list[dict] = []
        pending_exchanges = 0

        def flush_pending() -> None:
            nonlocal pending_messages, pending_exchanges
            nonlocal assistant_targets, context_dependent_skipped

            # Without the original scenario, a short reply or an action-only turn
            # is not a useful beginning for a standalone training example.
            while pending_messages and not has_enough_standalone_context(
                pending_messages[0]["content"]
            ):
                pending_messages = pending_messages[2:]
                pending_exchanges -= 1
                context_dependent_skipped += 1

            if not pending_messages:
                pending_exchanges = 0
                return

            examples.append({"messages": list(pending_messages)})
            assistant_targets += pending_exchanges
            pending_messages = []
            pending_exchanges = 0

        for index in range(1, len(turns), 2):
            user_content = turns[index]["value"].strip()
            assistant_content = turns[index + 1]["value"].strip()

            if is_stage_direction_only(assistant_content):
                flush_pending()
                stage_only_skipped += 1
                continue

            exchange = [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": assistant_content},
            ]
            candidate = [*pending_messages, *exchange]
            if pending_messages and word_count(candidate) > MAX_WORDS_PER_EXAMPLE:
                flush_pending()
                candidate = exchange

            if word_count(candidate) > MAX_WORDS_PER_EXAMPLE:
                overlong_skipped += 1
                continue

            pending_messages.extend(exchange)
            pending_exchanges += 1
            if pending_exchanges == MAX_EXCHANGES_PER_EXAMPLE:
                flush_pending()

        flush_pending()

    stats = {
        "examples": len(examples),
        "assistant_targets": assistant_targets,
        "stage_only_skipped": stage_only_skipped,
        "context_dependent_skipped": context_dependent_skipped,
        "overlong_skipped": overlong_skipped,
    }
    return examples, stats


def validate_public_domain_examples(examples: list[dict], expected_targets: int) -> None:
    if not examples:
        raise ValueError("A processed split has no examples")

    assistant_targets = 0
    for example_index, example in enumerate(examples):
        messages = example.get("messages")
        if not isinstance(messages, list) or len(messages) < 2:
            raise ValueError(f"Example {example_index}: messages must contain a complete exchange")
        expected_roles = [
            "user" if index % 2 == 0 else "assistant" for index in range(len(messages))
        ]
        actual_roles = [message.get("role") for message in messages]
        if "system" in actual_roles:
            raise ValueError(f"Example {example_index}: system messages are not allowed")
        if actual_roles != expected_roles or actual_roles[-1] != "assistant":
            raise ValueError(f"Example {example_index}: invalid roles {actual_roles}")
        if not has_enough_standalone_context(messages[0]["content"]):
            raise ValueError(f"Example {example_index}: first user turn lacks standalone context")
        if word_count(messages) > MAX_WORDS_PER_EXAMPLE:
            raise ValueError(f"Example {example_index}: exceeds the word limit")
        for message in messages:
            content = message.get("content")
            if not isinstance(content, str) or not content:
                raise ValueError(f"Example {example_index}: empty message content")
            if message["role"] == "assistant":
                if is_stage_direction_only(content):
                    raise ValueError(f"Example {example_index}: stage-direction-only assistant target")
                assistant_targets += 1

    if assistant_targets != expected_targets:
        raise ValueError(
            f"Assistant target count mismatch: validated {assistant_targets}, "
            f"expected {expected_targets}"
        )

def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def prepare_public_domain(character_slugs: list[str], seed: int) -> list[dict]:
    selected = {slug: PUBLIC_DOMAIN_CHARACTERS[slug] for slug in character_slugs}
    loaded = load_public_domain_conversations(set(selected.values()))
    summaries = []

    for slug, character_name in selected.items():
        originals = loaded[character_name]
        if len(originals) < 3:
            raise ValueError(
                f"{character_name} has too little usable data: "
                f"{len(originals)} original conversations"
            )

        splits = split_conversations(originals, seed)
        output_dir = PUBLIC_DOMAIN_OUTPUT_DIR / slug
        split_stats = {}

        for split_name, split_originals in splits.items():
            examples, stats = build_public_domain_examples(split_originals)
            validate_public_domain_examples(examples, stats["assistant_targets"])
            write_jsonl(output_dir / f"{split_name}.jsonl", examples)
            split_stats[split_name] = {
                "original_conversations": len(split_originals),
                **stats,
            }

        summary = {
            "character": character_name,
            "slug": slug,
            "seed": seed,
            "original_conversations": len(originals),
            "split_ratios": {"train": 0.8, "validation": 0.1, "test": 0.1},
            "windowing": {
                "max_exchanges": MAX_EXCHANGES_PER_EXAMPLE,
                "max_words": MAX_WORDS_PER_EXAMPLE,
                "overlap": False,
                "system_messages": "excluded",
                "minimum_spoken_words_in_first_user_turn": MIN_SPOKEN_WORDS_FOR_WINDOW_START,
                "stage_direction_only_replies": "excluded and treated as a context boundary",
                "context_dependent_window_starts": "excluded",
            },
            "splits": split_stats,
        }
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        summaries.append(summary)

    print("Character | conversations train/validation/test | examples train/validation/test")
    for summary in summaries:
        split_stats = summary["splits"]
        conversations_text = "/".join(
            str(split_stats[split]["original_conversations"])
            for split in ("train", "validation", "test")
        )
        examples_text = "/".join(
            str(split_stats[split]["examples"])
            for split in ("train", "validation", "test")
        )
        print(
            f"{summary['character']} | {summary['original_conversations']} "
            f"({conversations_text}) | {examples_text}"
        )
    return summaries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=("shakespeare", "public-domain"),
        default="shakespeare",
        help="Dataset to prepare (default: shakespeare)",
    )
    parser.add_argument(
        "--character",
        choices=("all", *PUBLIC_DOMAIN_CHARACTERS),
        default="all",
        help="Public-domain character to prepare (default: all five)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=PUBLIC_DOMAIN_SEED,
        help=f"Conversation split seed (default: {PUBLIC_DOMAIN_SEED})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dataset == "shakespeare":
        if args.character != "all":
            raise ValueError("--character can only be used with --dataset public-domain")
        prepare_shakespeare()
        return

    character_slugs = (
        list(PUBLIC_DOMAIN_CHARACTERS) if args.character == "all" else [args.character]
    )
    prepare_public_domain(character_slugs, args.seed)


if __name__ == "__main__":
    main()
