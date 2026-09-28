#!/usr/bin/env python3
"""Create CSV and SVG loss curves from an Axolotl Trainer state."""

import argparse
import csv
import json
import math
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True, help="Axolotl output directory")
    parser.add_argument("--csv", type=Path, required=True, help="Destination CSV file")
    parser.add_argument("--svg", type=Path, required=True, help="Destination SVG file")
    return parser.parse_args()


def load_latest_history(output_dir: Path) -> tuple[Path, list[dict]]:
    candidates = []
    for path in output_dir.rglob("trainer_state.json"):
        state = json.loads(path.read_text(encoding="utf-8"))
        history = state.get("log_history")
        if isinstance(history, list):
            candidates.append((int(state.get("global_step", -1)), len(history), path, history))
    if not candidates:
        raise FileNotFoundError(f"No trainer_state.json found under {output_dir}")
    _, _, path, history = max(candidates, key=lambda item: (item[0], item[1]))
    return path, history


def metric(entry: dict, *names: str) -> float | None:
    for name in names:
        value = entry.get(name)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
            return float(value)
    return None


def extract_losses(history: list[dict]) -> tuple[dict[int, float], dict[int, float]]:
    training = {}
    validation = {}
    for entry in history:
        step_value = entry.get("step")
        if not isinstance(step_value, (int, float)):
            continue
        step = int(step_value)
        train_loss = metric(entry, "loss", "train/loss")
        validation_loss = metric(entry, "eval_loss", "eval/loss")
        if train_loss is not None:
            training[step] = train_loss
        if validation_loss is not None:
            validation[step] = validation_loss
    if not training:
        raise ValueError("Trainer state contains no training loss values")
    if not validation:
        raise ValueError("Trainer state contains no validation loss values")
    return training, validation


def write_csv(path: Path, training: dict[int, float], validation: dict[int, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(["step", "train_loss", "validation_loss"])
        for step in sorted(training.keys() | validation.keys()):
            writer.writerow(
                [
                    step,
                    training.get(step, ""),
                    validation.get(step, ""),
                ]
            )


def write_svg(path: Path, training: dict[int, float], validation: dict[int, float]) -> None:
    width, height = 1000, 600
    left, right, top, bottom = 90, 35, 55, 75
    plot_width = width - left - right
    plot_height = height - top - bottom

    all_steps = sorted(training.keys() | validation.keys())
    all_losses = list(training.values()) + list(validation.values())
    x_min, x_max = min(all_steps), max(all_steps)
    y_min, y_max = min(all_losses), max(all_losses)
    if x_min == x_max:
        x_max = x_min + 1
    if y_min == y_max:
        y_min -= 0.5
        y_max += 0.5
    else:
        padding = (y_max - y_min) * 0.08
        y_min -= padding
        y_max += padding

    def x(step: int) -> float:
        return left + (step - x_min) / (x_max - x_min) * plot_width

    def y(loss: float) -> float:
        return top + (y_max - loss) / (y_max - y_min) * plot_height

    def polyline(values: dict[int, float]) -> str:
        return " ".join(f"{x(step):.1f},{y(loss):.1f}" for step, loss in sorted(values.items()))

    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="500" y="30" text-anchor="middle" font-family="sans-serif" font-size="22">Training and validation loss</text>',
    ]

    for index in range(6):
        fraction = index / 5
        grid_y = top + fraction * plot_height
        loss_value = y_max - fraction * (y_max - y_min)
        lines.append(
            f'<line x1="{left}" y1="{grid_y:.1f}" x2="{width - right}" y2="{grid_y:.1f}" stroke="#dddddd"/>'
        )
        lines.append(
            f'<text x="{left - 12}" y="{grid_y + 5:.1f}" text-anchor="end" font-family="sans-serif" font-size="13">{loss_value:.4f}</text>'
        )

    for index in range(6):
        fraction = index / 5
        grid_x = left + fraction * plot_width
        step_value = round(x_min + fraction * (x_max - x_min))
        lines.append(
            f'<line x1="{grid_x:.1f}" y1="{top}" x2="{grid_x:.1f}" y2="{height - bottom}" stroke="#eeeeee"/>'
        )
        lines.append(
            f'<text x="{grid_x:.1f}" y="{height - bottom + 25}" text-anchor="middle" font-family="sans-serif" font-size="13">{step_value}</text>'
        )

    lines.extend(
        [
            f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}" stroke="#333333" stroke-width="2"/>',
            f'<line x1="{left}" y1="{height - bottom}" x2="{width - right}" y2="{height - bottom}" stroke="#333333" stroke-width="2"/>',
            f'<polyline points="{polyline(training)}" fill="none" stroke="#2563eb" stroke-width="3"/>',
            f'<polyline points="{polyline(validation)}" fill="none" stroke="#ea580c" stroke-width="3"/>',
        ]
    )

    for step, loss in sorted(validation.items()):
        lines.append(f'<circle cx="{x(step):.1f}" cy="{y(loss):.1f}" r="4" fill="#ea580c"/>')

    best_step, best_loss = min(validation.items(), key=lambda item: item[1])
    lines.extend(
        [
            f'<circle cx="{x(best_step):.1f}" cy="{y(best_loss):.1f}" r="7" fill="#16a34a" stroke="white" stroke-width="2"/>',
            f'<text x="{x(best_step):.1f}" y="{max(top + 15, y(best_loss) - 12):.1f}" text-anchor="middle" font-family="sans-serif" font-size="13" fill="#166534">best: {best_loss:.4f} @ {best_step}</text>',
            '<line x1="690" y1="25" x2="725" y2="25" stroke="#2563eb" stroke-width="3"/>',
            '<text x="735" y="30" font-family="sans-serif" font-size="14">train loss</text>',
            '<line x1="825" y1="25" x2="860" y2="25" stroke="#ea580c" stroke-width="3"/>',
            '<text x="870" y="30" font-family="sans-serif" font-size="14">validation loss</text>',
            f'<text x="{left + plot_width / 2:.1f}" y="{height - 20}" text-anchor="middle" font-family="sans-serif" font-size="15">Training step</text>',
            f'<text x="22" y="{top + plot_height / 2:.1f}" text-anchor="middle" font-family="sans-serif" font-size="15" transform="rotate(-90 22 {top + plot_height / 2:.1f})">Loss</text>',
            '</svg>',
        ]
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    state_path, history = load_latest_history(args.input)
    training, validation = extract_losses(history)
    write_csv(args.csv, training, validation)
    write_svg(args.svg, training, validation)
    print(f"Loss history: {state_path}")
    print(f"Saved {len(training)} training and {len(validation)} validation points")
    print(f"Loss CSV: {args.csv}")
    print(f"Loss plot: {args.svg}")


if __name__ == "__main__":
    main()
