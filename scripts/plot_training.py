from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_PATTERNS = [
    "reward",
    "value loss",
    "surrogate loss",
    "adaptation loss",
    "time elapsed",
]


def _pick_latest_run(models_dir: Path) -> Path:
    runs = sorted([path for path in models_dir.iterdir() if path.is_dir()])
    if not runs:
        raise FileNotFoundError(f"No runs found in {models_dir}")
    return runs[-1]


def _parse_float(value: str):
    value = value.strip()
    try:
        return float(value)
    except ValueError:
        match = re.search(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", value)
        if match:
            try:
                return float(match.group(0))
            except ValueError:
                return None
    return None


def _parse_outputs_txt(path: Path) -> List[Tuple[int, Dict[str, float]]]:
    if not path.exists():
        return []

    samples: List[Tuple[int, Dict[str, float]]] = []
    current: Dict[str, float] = {}
    current_iter = None

    for raw_line in path.read_text(errors="ignore").splitlines():
        if "│" not in raw_line:
            continue
        parts = [part.strip().strip("║│") for part in raw_line.split("│")]
        if len(parts) < 3:
            continue
        key = parts[1].strip()
        value = parts[2].strip()
        if not key:
            continue
        parsed = _parse_float(value)
        if parsed is None:
            continue
        current[key] = parsed
        if key == "iterations":
            current_iter = int(parsed)
            samples.append((current_iter, dict(current)))
            current = {}

    return samples


def _parse_event_scalars(run_dir: Path) -> Dict[str, List[Tuple[int, float]]]:
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except Exception:
        return {}

    event_files = sorted(run_dir.glob("events.out.tfevents*"))
    if not event_files:
        return {}

    accumulator = EventAccumulator(str(run_dir))
    accumulator.Reload()
    scalars = {}
    for tag in accumulator.Tags().get("scalars", []):
        scalars[tag] = [(event.step, float(event.value)) for event in accumulator.Scalars(tag)]
    return scalars


def _match_patterns(name: str, patterns: Sequence[str]) -> bool:
    lowered = name.lower()
    return any(pattern.lower() in lowered for pattern in patterns)


def _series_from_outputs(samples: List[Tuple[int, Dict[str, float]]], patterns: Sequence[str]) -> Dict[str, List[Tuple[int, float]]]:
    if not samples:
        return {}

    keys = set()
    for _, metrics in samples:
        keys.update(metrics.keys())

    selected = [key for key in sorted(keys) if _match_patterns(key, patterns)]
    series: Dict[str, List[Tuple[int, float]]] = {}
    for key in selected:
        values = []
        for iteration, metrics in samples:
            if key in metrics:
                values.append((iteration, metrics[key]))
        if values:
            series[key] = values
    return series


def _series_from_scalars(scalars: Dict[str, List[Tuple[int, float]]], patterns: Sequence[str]) -> Dict[str, List[Tuple[int, float]]]:
    selected = [name for name in sorted(scalars) if _match_patterns(name, patterns)]
    return {name: scalars[name] for name in selected if scalars[name]}


def _resolve_series(run_dir: Path, patterns: Sequence[str]) -> Dict[str, List[Tuple[int, float]]]:
    outputs_path = run_dir / "outputs.txt"
    series = _series_from_outputs(_parse_outputs_txt(outputs_path), patterns)
    if series:
        return series
    return _series_from_scalars(_parse_event_scalars(run_dir), patterns)


def _plot_series(series: Dict[str, List[Tuple[int, float]]], output: Path) -> None:
    if not series:
        raise RuntimeError("No plottable metrics found")

    items = list(series.items())
    cols = 2 if len(items) > 1 else 1
    rows = (len(items) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 4.2 * rows), squeeze=False)
    axes_flat = axes.flatten()

    for ax, (name, values) in zip(axes_flat, items):
        xs = [step for step, _ in values]
        ys = [value for _, value in values]
        ax.plot(xs, ys, linewidth=1.8)
        ax.set_title(name)
        ax.set_xlabel("iteration")
        ax.grid(True, alpha=0.3)
    for ax in axes_flat[len(items):]:
        ax.axis("off")

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160, bbox_inches="tight")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot AlienGo training curves.")
    parser.add_argument("--run_dir", type=Path, help="Path to a single training run directory.")
    parser.add_argument("--models_dir", type=Path, default=Path("models/aliengo_flat"), help="Directory with run folders.")
    parser.add_argument("--output", type=Path, help="Output image path.")
    parser.add_argument("--metric", action="append", dest="metrics", help="Substring filter for metric names. Can be repeated.")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    run_dir = args.run_dir or _pick_latest_run(args.models_dir)
    patterns = args.metrics or DEFAULT_PATTERNS
    series = _resolve_series(run_dir, patterns)
    output = args.output or (run_dir / "training_plot.png")
    _plot_series(series, output)
    print(f"Saved plot to: {output}")


if __name__ == "__main__":
    main()
