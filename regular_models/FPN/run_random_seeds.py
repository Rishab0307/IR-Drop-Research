#!/usr/bin/env python3
"""Execute FPN_6 for seeds 43-51 and maintain a 95% t confidence interval."""

from __future__ import annotations

import csv
import json
import math
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.stats import t


FPN_DIR = Path(__file__).resolve().parent
RESEARCH_DIR = FPN_DIR.parents[1]
SOURCE_NOTEBOOK = FPN_DIR / "FPN_6.ipynb"
RUNS_DIR = FPN_DIR / "random_seed_runs"
RESULTS_CSV = RESEARCH_DIR / "regular_models" / "csv_results" / "fpn_random_seed_results.csv"
SEEDS = list(range(43, 52))
FIELDS = [
    "seed", "status", "weighted_ssim", "macro_ssim",
    "weighted_ci95_lower", "weighted_ci95_upper",
    "macro_ci95_lower", "macro_ci95_upper", "n_completed",
    "started_at", "finished_at", "run_directory", "error",
]


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def read_rows() -> list[dict[str, str]]:
    with RESULTS_CSV.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_rows(rows: list[dict[str, str]]) -> None:
    temporary = RESULTS_CSV.with_suffix(".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(RESULTS_CSV)


def refresh_intervals(rows: list[dict[str, str]]) -> None:
    complete = [row for row in rows if row["status"] == "completed"]
    weighted = np.array([float(row["weighted_ssim"]) for row in complete])
    macro = np.array([float(row["macro_ssim"]) for row in complete])
    n = len(complete)
    if n >= 2:
        critical = float(t.ppf(0.975, df=n - 1))
        weighted_margin = critical * float(weighted.std(ddof=1)) / math.sqrt(n)
        macro_margin = critical * float(macro.std(ddof=1)) / math.sqrt(n)
        values = {
            "weighted_ci95_lower": f"{weighted.mean() - weighted_margin:.6f}",
            "weighted_ci95_upper": f"{weighted.mean() + weighted_margin:.6f}",
            "macro_ci95_lower": f"{macro.mean() - macro_margin:.6f}",
            "macro_ci95_upper": f"{macro.mean() + macro_margin:.6f}",
        }
    else:
        values = {name: "" for name in FIELDS if "ci95" in name}
    for row in rows:
        row["n_completed"] = str(n)
        for name, value in values.items():
            row[name] = value


def update_seed(seed: int, **changes: str) -> None:
    rows = read_rows()
    row = next(item for item in rows if int(item["seed"]) == seed)
    row.update(changes)
    refresh_intervals(rows)
    write_rows(rows)


def make_notebook(seed: int, run_dir: Path) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    notebook = json.loads(SOURCE_NOTEBOOK.read_text(encoding="utf-8"))
    cells = notebook["cells"][:18]

    setup = cells[0]["source"]
    setup_text = "".join(setup)
    setup_text += (
        f"\nimport random\nSEED = {seed}\n"
        "os.environ['PYTHONHASHSEED'] = str(SEED)\n"
        "random.seed(SEED)\n"
        "np.random.seed(SEED)\n"
        "tf.keras.utils.set_random_seed(SEED)\n"
        "try:\n"
        "    tf.config.experimental.enable_op_determinism()\n"
        "except Exception as exc:\n"
        "    print(f'Deterministic TensorFlow operations unavailable: {exc}')\n"
    )
    cells[0]["source"] = setup_text.splitlines(keepends=True)

    for cell in cells:
        source = "".join(cell.get("source", []))
        source = source.replace("random_state=42", "random_state=SEED")
        source = source.replace('"best_fpn.keras"', '"best_fpn_seed_%d.keras"' % seed)
        source = source.replace('"final_fpn_ir_model.keras"', '"final_fpn_ir_model_seed_%d.keras"' % seed)
        cell["source"] = source.splitlines(keepends=True)
        cell["outputs"] = []
        cell["execution_count"] = None

    result_code = f'''# Persist this seed's aggregate per-chip-range SSIM results.
import csv as _csv
from pathlib import Path as _Path

_csv_path = _Path({str(RESULTS_CSV)!r})
with _csv_path.open(newline="", encoding="utf-8") as _handle:
    _rows = list(_csv.DictReader(_handle))
_row = next(_item for _item in _rows if int(_item["seed"]) == {seed})
_row.update({{
    "status": "completed",
    "weighted_ssim": f"{{float(combined_per_chip['SSIM']):.9f}}",
    "macro_ssim": f"{{float(equal_weight_per_chip['SSIM']):.9f}}",
    "finished_at": __import__("datetime").datetime.now().astimezone().isoformat(timespec="seconds"),
    "error": "",
}})
with (_csv_path.with_suffix(".result.tmp")).open("w", newline="", encoding="utf-8") as _handle:
    _writer = _csv.DictWriter(_handle, fieldnames={FIELDS!r})
    _writer.writeheader()
    _writer.writerows(_rows)
(_csv_path.with_suffix(".result.tmp")).replace(_csv_path)
print("Recorded weighted SSIM", _row["weighted_ssim"], "and macro SSIM", _row["macro_ssim"])
'''
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "id": f"save-seed-{seed}",
        "metadata": {"tags": ["random-seed-result"]},
        "outputs": [],
        "source": result_code.splitlines(keepends=True),
    })
    notebook["cells"] = cells
    notebook_path = run_dir / f"FPN_seed_{seed}.ipynb"
    notebook_path.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
    return notebook_path


def run_seed(seed: int) -> None:
    run_dir = RUNS_DIR / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    notebook_path = make_notebook(seed, run_dir)
    log_path = run_dir / f"seed_{seed}.log"
    update_seed(seed, status="running", started_at=now(), finished_at="", error="")
    command = [
        sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook",
        "--execute", notebook_path.name, "--output", notebook_path.name,
        "--ExecutePreprocessor.timeout=-1", "--ExecutePreprocessor.kernel_name=ml-metal",
    ]
    environment = os.environ.copy()
    environment["MPLCONFIGDIR"] = str(run_dir / ".matplotlib")
    try:
        with log_path.open("a", encoding="utf-8") as log:
            subprocess.run(
                command, cwd=run_dir, env=environment, stdout=log,
                stderr=subprocess.STDOUT, check=True,
            )
        rows = read_rows()
        refresh_intervals(rows)
        write_rows(rows)
    except Exception as exc:
        update_seed(seed, status="failed", finished_at=now(), error=str(exc))
        raise


def main() -> None:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    for seed in SEEDS:
        row = next(item for item in read_rows() if int(item["seed"]) == seed)
        if row["status"] == "completed":
            continue
        run_seed(seed)


if __name__ == "__main__":
    main()
