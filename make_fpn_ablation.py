from __future__ import annotations

import copy
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "regular_models" / "FPN" / "FPN_6.ipynb"
OUT_DIR = ROOT / "FPN ablation"

INPUTS = ["power_all", "power_i", "power_sca", "power_s", "power_t"]
COMPLETED_ONLY_RUNS = {"only_pall", "only_pi", "only_ps", "only_psca"}
SHORT = {
    "power_all": "pall",
    "power_i": "pi",
    "power_sca": "psca",
    "power_s": "ps",
    "power_t": "pt",
}


def lines(source: str) -> list[str]:
    return source.splitlines(keepends=True)


def make_config_cell(run_id: str, active_inputs: list[str]) -> str:
    return f'''# Ablation configuration
from pathlib import Path

DATA_ROOT = "/Users/rishab/Research"
TARGET_FOLDER = "IR_drop"
ACTIVE_INPUTS = {active_inputs!r}
CHANNELS_BY_INPUT = {{
    "power_all": 1,
    "power_i": 1,
    "power_sca": 1,
    "power_s": 1,
    "power_t": 20,
}}
INPUT_CHANNELS = sum(CHANNELS_BY_INPUT[name] for name in ACTIVE_INPUTS)

TARGET_SIZE = (320, 320)
BATCH_SIZE = 16
RUN_ID = "{run_id}"
RUN_DIR = Path("artifacts") / RUN_ID
RUN_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_PATH = str(RUN_DIR / f"best_{{RUN_ID}}.keras")
FINAL_MODEL_PATH = str(RUN_DIR / f"final_{{RUN_ID}}.keras")
RESULTS_CSV = "fpn_ablation_results.csv"

print("Run:", RUN_ID)
print("Active inputs:", ACTIVE_INPUTS)
print("Input channels:", INPUT_CHANNELS)
'''


GENERIC_IO = '''def load_input_array(path, input_name):
    arr_file = np.load(path)
    if isinstance(arr_file, np.lib.npyio.NpzFile):
        arr = list(arr_file.values())[0]
        arr_file.close()
    else:
        arr = arr_file
    arr = arr.astype(np.float32)

    if input_name == "power_t":
        if arr.ndim == 3 and arr.shape[0] == 20:
            arr = np.transpose(arr, (1, 2, 0))
    elif arr.ndim == 2:
        arr = arr[..., np.newaxis]

    return arr


def compute_channel_min_max(file_list):
    channel_min = None
    channel_max = None

    for rel_path in file_list:
        chip = rel_path.split(os.sep)[0]
        filename = rel_path.split(os.sep)[-1]
        power_maps = [
            load_input_array(os.path.join(DATA_ROOT, chip, name, filename), name)
            for name in ACTIVE_INPUTS
        ]
        X = np.concatenate(power_maps, axis=-1)
        sample_min = X.min(axis=(0, 1))
        sample_max = X.max(axis=(0, 1))

        if channel_min is None:
            channel_min = sample_min
            channel_max = sample_max
        else:
            channel_min = np.minimum(channel_min, sample_min)
            channel_max = np.maximum(channel_max, sample_max)

    return channel_min, channel_max
'''


FILE_SPLIT = '''# ----------------------------
# File split (identical seed and targets in every ablation)
# ----------------------------
CHIPS = [
    "Vortex-small",
    # "Vortex-large",
    # "nvdla-large",
    "nvdla-small",
    "openc910-1",
    "RISCY",
    "RISCY-FPU",
    "zero-riscy",
]

all_files = []
chip_names = []

for chip in CHIPS:
    ir_dir = os.path.join(DATA_ROOT, chip, TARGET_FOLDER)
    for f in sorted(os.listdir(ir_dir)):
        if all(os.path.exists(os.path.join(DATA_ROOT, chip, name, f)) for name in ACTIVE_INPUTS):
            all_files.append(os.path.join(chip, TARGET_FOLDER, f))
            chip_names.append(chip)

train_files, temp_files, train_labels, temp_labels = train_test_split(
    all_files, chip_names, test_size=0.2, random_state=42, stratify=chip_names
)
val_files, test_files, val_labels, test_labels = train_test_split(
    temp_files, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels
)

CHANNEL_MIN, CHANNEL_MAX = compute_channel_min_max(train_files)
CHANNEL_MIN_TF = tf.constant(CHANNEL_MIN, dtype=tf.float32)
CHANNEL_MAX_TF = tf.constant(CHANNEL_MAX, dtype=tf.float32)

print("CHANNEL_MIN:", CHANNEL_MIN)
print("CHANNEL_MAX:", CHANNEL_MAX)
'''


LOAD_SAMPLE = '''def load_sample(rel_path):
    y_path = tf.strings.join([DATA_ROOT, rel_path], separator=os.sep)
    y = load_npy(y_path)

    chip = tf.strings.split(rel_path, os.sep)[0]
    filename = tf.strings.split(rel_path, os.sep)[-1]
    power_maps = []

    for input_name in ACTIVE_INPUTS:
        x_path = tf.strings.join([DATA_ROOT, chip, input_name, filename], separator=os.sep)

        def _load_active(path, name=input_name):
            return load_input_array(path.decode("utf-8"), name)

        x = tf.numpy_function(_load_active, [x_path], tf.float32)
        x.set_shape([None, None, CHANNELS_BY_INPUT[input_name]])
        x = tf.image.resize(x, TARGET_SIZE)
        power_maps.append(x)

    X = tf.concat(power_maps, axis=-1)
    X.set_shape([TARGET_SIZE[0], TARGET_SIZE[1], INPUT_CHANNELS])
    X = (X - CHANNEL_MIN_TF) / (CHANNEL_MAX_TF - CHANNEL_MIN_TF + 1e-8)
    return X, y
'''


RESULT_CELL = '''# Save one reproducible summary row for this completed run.
import csv

result_row = {
    "run_id": RUN_ID,
    "active_inputs": "+".join(ACTIVE_INPUTS),
    "input_channels": INPUT_CHANNELS,
    "train_samples": len(train_files),
    "val_samples": len(val_files),
    "test_samples": len(test_files),
    "epochs_completed": len(history.history["loss"]),
    "best_epoch": int(np.argmin(history.history["val_loss"])) + 1,
    "best_val_loss": float(np.min(history.history["val_loss"])),
    "train_mse": float(train_metrics[0]),
    "train_mae": float(train_metrics[1]),
    "val_mse": float(val_metrics[0]),
    "val_mae": float(val_metrics[1]),
    "test_mse": float(test_metrics[0]),
    "test_mae": float(test_metrics[1]),
    **{f"test_{k.lower()}": float(v) for k, v in combined_per_chip.items()},
    # Unweighted (macro) average: every chip contributes equally.
    **{f"test_macro_{k.lower()}": float(v) for k, v in equal_weight_per_chip.items()},
    "training_seconds": float(train_end - train_start),
    "total_seconds": float(total_seconds),
}

fieldnames = list(result_row)
existing_rows = []
if os.path.exists(RESULTS_CSV):
    with open(RESULTS_CSV, newline="") as f:
        existing_rows = [row for row in csv.DictReader(f) if row.get("run_id") != RUN_ID]

with open(RESULTS_CSV, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(existing_rows)
    writer.writerow(result_row)

print("Saved results to", RESULTS_CSV)
print(result_row)
'''


def make_notebook(base: dict, run_id: str, active_inputs: list[str]) -> dict:
    nb = copy.deepcopy(base)
    cells = nb["cells"]
    cells[1]["source"] = lines(make_config_cell(run_id, active_inputs))
    cells[2]["source"] = lines(GENERIC_IO)
    cells[3]["source"] = lines(FILE_SPLIT)
    cells[7]["source"] = lines(LOAD_SAMPLE)

    if run_id not in COMPLETED_ONLY_RUNS:
        dataset_source = "".join(cells[8]["source"])
        dataset_source = dataset_source.replace(
            "    .map(load_sample, num_parallel_calls=tf.data.AUTOTUNE)\n"
            "    .shuffle(len(train_files))\n",
            "    .shuffle(len(train_files), reshuffle_each_iteration=True)\n"
            "    .map(load_sample, num_parallel_calls=tf.data.AUTOTUNE)\n",
            1,
        )
        dataset_source = dataset_source.replace(
            ".prefetch(tf.data.AUTOTUNE)",
            ".prefetch(1)",
        )
        cells[8]["source"] = lines(dataset_source)

    model_source = "".join(cells[11]["source"])
    model_source = model_source.replace(
        "input_shape=(320, 320, 24)",
        "input_shape=(320, 320, INPUT_CHANNELS)",
    )
    model_source = model_source.replace(
        "assert fpn_model.input_shape == (None, 320, 320, 24)\nassert fpn_model.output_shape == (None, 320, 320, 1)",
        "assert model.input_shape == (None, 320, 320, INPUT_CHANNELS)\nassert model.output_shape == (None, 320, 320, 1)",
    )
    cells[11]["source"] = lines(model_source)

    train_source = "".join(cells[12]["source"])
    train_source = train_source.replace('"best_fpn.keras"', "CHECKPOINT_PATH")
    train_source = train_source.replace('model.load_weights(CHECKPOINT_PATH)', 'model.load_weights(CHECKPOINT_PATH)')
    train_source = train_source.replace('model.save("final_fpn_ir_model.keras")', 'model.save(FINAL_MODEL_PATH)')
    cells[12]["source"] = lines(train_source)

    # Add the CSV cell directly after the metrics block; omit random visualizations
    # from automated execution to keep result notebooks compact and deterministic.
    nb["cells"] = cells[:16] + [{
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": lines(RESULT_CELL),
    }]

    for cell in nb["cells"]:
        if cell.get("cell_type") == "code":
            cell["execution_count"] = None
            cell["outputs"] = []

    nb.setdefault("metadata", {})["kernelspec"] = {
        "display_name": "Python (ML Metal GPU)",
        "language": "python",
        "name": "ml-metal",
    }
    nb["metadata"].setdefault("language_info", {})["version"] = "3.11.15"

    return nb


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    base = json.loads(SOURCE.read_text())
    specs = []
    for selected in INPUTS:
        specs.append((f"only_{SHORT[selected]}", [selected]))
    for dropped in INPUTS:
        specs.append((f"drop_{SHORT[dropped]}", [name for name in INPUTS if name != dropped]))

    manifest = []
    for run_id, active_inputs in specs:
        path = OUT_DIR / f"{run_id}.ipynb"
        if run_id not in COMPLETED_ONLY_RUNS or not path.exists():
            path.write_text(json.dumps(make_notebook(base, run_id, active_inputs), indent=1))
        manifest.append({"run_id": run_id, "notebook": path.name, "active_inputs": active_inputs})

    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Created {len(manifest)} notebooks in {OUT_DIR}")


if __name__ == "__main__":
    main()
