#!/usr/bin/env python3
"""Prepare a small Shakespeare dialogue dataset for Axolotl."""

import hashlib
import json
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
SOURCE = DATA_DIR / "shakespeare_unformatted.jsonl"
REVISION = "ea9ab49c0367f9efac40af869c53c1b956c626a5"
SOURCE_URL = (
    "https://huggingface.co/datasets/chaseharmon/6.7960_Shakespeare/"
    f"resolve/{REVISION}/shakespeare_unformatted.jsonl"
)
SOURCE_SHA256 = "d09f91f36fa59230e94519c15bb833a623b38e885de899eb0f420c37f6c8bf5d"
TRAIN_COUNT = 4000
VALIDATION_COUNT = 200


def download_source() -> None:
    if not SOURCE.exists():
        print(f"Downloading {SOURCE_URL}")
        request = urllib.request.Request(SOURCE_URL, headers={"User-Agent": "shakespeare-workshop/1.0"})
        temporary = SOURCE.with_suffix(".download")
        try:
            with urllib.request.urlopen(request, timeout=60) as response, temporary.open("wb") as output:
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
            temporary.replace(SOURCE)
        finally:
            temporary.unlink(missing_ok=True)
    else:
        print(f"Using existing source: {SOURCE}")
    digest = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    if digest != SOURCE_SHA256:
        raise ValueError(f"Source checksum mismatch: {digest}; expected {SOURCE_SHA256}")


def load_pairs() -> tuple[list[tuple[str, str]], int, int]:
    pairs = []
    seen = set()
    empty_count = duplicate_count = 0
    with SOURCE.open(encoding="utf-8") as source:
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


def write_messages(path: Path, pairs: list[tuple[str, str]]) -> None:
    with path.open("w", encoding="utf-8") as output:
        for src, trg in pairs:
            row = {
                "messages": [
                    {"role": "user", "content": src},
                    {"role": "assistant", "content": trg},
                ]
            }
            output.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    download_source()
    pairs, empty_count, duplicate_count = load_pairs()
    if len(pairs) < TRAIN_COUNT + VALIDATION_COUNT:
        raise ValueError(f"Need {TRAIN_COUNT + VALIDATION_COUNT} usable pairs; found {len(pairs)}")
    # Keep a gap between splits because nearby source rows are consecutive exchanges.
    train, validation = pairs[:TRAIN_COUNT], pairs[-VALIDATION_COUNT:]
    assert set(train).isdisjoint(validation)
    write_messages(DATA_DIR / "train.jsonl", train)
    write_messages(DATA_DIR / "validation.jsonl", validation)
    print(
        f"Prepared {len(train)} train and {len(validation)} validation examples "
        f"({empty_count} empty, {duplicate_count} duplicate pairs skipped)."
    )


if __name__ == "__main__":
    main()
