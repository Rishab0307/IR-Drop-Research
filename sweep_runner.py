#!/usr/bin/env python3
"""Run the four Vortex-small tuning notebooks as isolated, resumable jobs.

Each combination is materialized as its own notebook and executed by a new
Jupyter kernel.  Successful SSIM values are appended immediately to CSV.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import nbformat


ROOT = Path(__file__).resolve().parent
RUN_ROOT = ROOT / "hyperparameter_sweep_runs"
RESULTS_CSV = RUN_ROOT / "raw_results.csv"

NOTEBOOKS = {
    "MSE": ROOT / "fft-strat-1-MSE.ipynb",
    "MAE": ROOT / "fft-strat-1-MAE.ipynb",
    "Huber": ROOT / "fft-strat-1-Huber.ipynb",
    "LogCosh": ROOT / "fft-strat-1-LogCosh.ipynb",
}
BATCH_SIZES = (8, 16, 32)
LEARNING_RATES = (1e-5, 1e-4, 1e-3)
FINE_TUNED_LAYERS = (2, 4, 6, 8, 10)
FIELDS = (
    "loss",
    "batch_size",
    "learning_rate",
    "fine_tuned_layers",
    "per_chip_range_ssim",
    "unweighted_macro_ssim",
    "elapsed_seconds",
    "run_id",
    "executed_notebook",
)


def rate_label(rate: float) -> str:
    return {1e-5: "1e-5", 1e-4: "1e-4", 1e-3: "1e-3"}[rate]


def run_id(loss: str, batch_size: int, rate: float, layers: int) -> str:
    return f"{loss.lower()}__b{batch_size}__lr{rate_label(rate)}__l{layers}"


def completed_ids() -> set[str]:
    if not RESULTS_CSV.exists():
        return set()
    with RESULTS_CSV.open(newline="") as handle:
        return {row["run_id"] for row in csv.DictReader(handle)}


def append_result(row: dict[str, object]) -> None:
    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    needs_header = not RESULTS_CSV.exists()
    with RESULTS_CSV.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        if needs_header:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()
        os.fsync(handle.fileno())


def replace_exact(source: str, pattern: str, replacement: str, expected: int) -> str:
    updated, count = re.subn(pattern, replacement, source)
    if count != expected:
        raise RuntimeError(
            f"Expected {expected} replacements for {pattern!r}, observed {count}"
        )
    return updated


def parameterize_notebook(
    source_path: Path,
    destination: Path,
    loss: str,
    batch_size: int,
    rate: float,
    layers: int,
) -> None:
    notebook = nbformat.read(source_path, as_version=4)
    lr_text = rate_label(rate)

    batch_replacements = 0
    lr_replacements = 0
    layer_replacements = 0
    save_replacements = 0

    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        source = cell.source

        if re.search(r"(?m)^BATCH_SIZE = \d+\s*$", source):
            source = replace_exact(
                source,
                r"(?m)^BATCH_SIZE = \d+\s*$",
                f"BATCH_SIZE = {batch_size}",
                1,
            )
            batch_replacements += 1

        lr_count = len(re.findall(r"learning_rate=1e-[345]", source))
        if lr_count:
            source = replace_exact(
                source,
                r"learning_rate=1e-[345]",
                f"learning_rate={lr_text}",
                lr_count,
            )
            lr_replacements += lr_count

        if re.search(r"base_model\.layers\[:-\d+\]", source):
            source = replace_exact(
                source,
                r"base_model\.layers\[:-\d+\]",
                f"base_model.layers[:-{layers}]",
                1,
            )
            layer_replacements += 1

        if 'model.save("final_ir_model.keras")' in source:
            source = source.replace(
                'model.save("final_ir_model.keras")',
                'print("Sweep run: final model save skipped; best checkpoints retained.")',
            )
            save_replacements += 1

        cell.source = source

    if (batch_replacements, lr_replacements, layer_replacements, save_replacements) != (
        1,
        2,
        1,
        1,
    ):
        raise RuntimeError(
            "Parameterization audit failed: "
            f"batch={batch_replacements}, lr={lr_replacements}, "
            f"layers={layer_replacements}, save={save_replacements}"
        )

    marker = {
        "loss": loss,
        "batch_size": batch_size,
        "learning_rate": rate,
        "fine_tuned_layers": layers,
    }
    notebook.cells.append(
        nbformat.v4.new_code_cell(
            "import json as _json\n"
            f"_sweep_metadata = {marker!r}\n"
            "_sweep_ssim = float(per_chip_metrics_per_chip['Vortex-small']['SSIM'])\n"
            "# One-chip sweep: the unweighted chip macro equals this chip's SSIM.\n"
            "_sweep_macro_ssim = _sweep_ssim\n"
            "with open('result.json', 'w') as _result_file:\n"
            "    _json.dump({**_sweep_metadata, 'per_chip_range_ssim': _sweep_ssim, 'unweighted_macro_ssim': _sweep_macro_ssim}, _result_file, indent=2)\n"
            "print(f'SWEEP_PER_CHIP_RANGE_SSIM={_sweep_ssim:.9f}')"
        )
    )

    destination.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, destination)


def execute_one(loss: str, batch_size: int, rate: float, layers: int) -> None:
    identifier = run_id(loss, batch_size, rate, layers)
    workdir = RUN_ROOT / identifier
    input_notebook = workdir / "run.ipynb"
    executed_notebook = workdir / "executed.ipynb"
    log_path = workdir / "execution.log"
    result_path = workdir / "result.json"

    parameterize_notebook(
        NOTEBOOKS[loss], input_notebook, loss, batch_size, rate, layers
    )

    env = os.environ.copy()
    env["MPLCONFIGDIR"] = str(workdir / "matplotlib")
    env["TF_CPP_MIN_LOG_LEVEL"] = "2"

    command = [
        sys.executable,
        "-m",
        "jupyter",
        "nbconvert",
        "--to",
        "notebook",
        "--execute",
        input_notebook.name,
        "--output",
        executed_notebook.name,
        "--ExecutePreprocessor.timeout=-1",
        "--ExecutePreprocessor.kernel_name=python3",
    ]

    start = time.time()
    with log_path.open("w") as log:
        completed = subprocess.run(
            command,
            cwd=workdir,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    elapsed = time.time() - start

    if completed.returncode != 0:
        raise RuntimeError(
            f"Run {identifier} failed with exit code {completed.returncode}; "
            f"see {log_path}"
        )
    if not result_path.exists():
        raise RuntimeError(f"Run {identifier} completed without {result_path}")

    result = json.loads(result_path.read_text())
    if any(result[key] != value for key, value in {
        "loss": loss,
        "batch_size": batch_size,
        "learning_rate": rate,
        "fine_tuned_layers": layers,
    }.items()):
        raise RuntimeError(f"Result metadata mismatch for {identifier}: {result}")

    ssim = float(result["per_chip_range_ssim"])
    if not -1.0 <= ssim <= 1.0:
        raise RuntimeError(f"Invalid SSIM for {identifier}: {ssim}")

    append_result(
        {
            "loss": loss,
            "batch_size": batch_size,
            "learning_rate": rate_label(rate),
            "fine_tuned_layers": layers,
            "per_chip_range_ssim": f"{ssim:.9f}",
            "unweighted_macro_ssim": f"{float(result.get('unweighted_macro_ssim', ssim)):.9f}",
            "elapsed_seconds": f"{elapsed:.3f}",
            "run_id": identifier,
            "executed_notebook": str(executed_notebook.relative_to(ROOT)),
        }
    )
    print(f"COMPLETED {identifier}: per-chip-range SSIM={ssim:.9f}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only-loss", choices=tuple(NOTEBOOKS))
    parser.add_argument("--only-batch", type=int, choices=BATCH_SIZES)
    parser.add_argument("--only-lr", type=float, choices=LEARNING_RATES)
    parser.add_argument("--only-layers", type=int, choices=FINE_TUNED_LAYERS)
    parser.add_argument("--limit", type=int, help="Maximum new runs for this invocation")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    done = completed_ids()
    launched = 0

    for loss in NOTEBOOKS:
        for batch_size in BATCH_SIZES:
            for rate in LEARNING_RATES:
                for layers in FINE_TUNED_LAYERS:
                    if args.only_loss and loss != args.only_loss:
                        continue
                    if args.only_batch and batch_size != args.only_batch:
                        continue
                    if args.only_lr and rate != args.only_lr:
                        continue
                    if args.only_layers and layers != args.only_layers:
                        continue

                    identifier = run_id(loss, batch_size, rate, layers)
                    if identifier in done:
                        print(f"SKIP {identifier}: already recorded", flush=True)
                        continue
                    if args.limit is not None and launched >= args.limit:
                        return

                    print(f"START {identifier}", flush=True)
                    execute_one(loss, batch_size, rate, layers)
                    launched += 1


if __name__ == "__main__":
    main()
