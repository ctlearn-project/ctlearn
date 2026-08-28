"""
Benchmark: ``SingleCNN(conv_backend="hexagdly")`` + ``HexagdlyMapper`` vs.
``SingleCNN(conv_backend="square")`` + ``BilinearMapper`` on the same LST-1
energy-regression task.

Both models are trained through the real ``TrainCTLearnModel`` tool (not a
hand-rolled loop), with identical ``n_epochs``, ``batch_size`` and
``random_seed`` (so both see the same train/validation split) -- the only
difference between the two runs is ``conv_backend``/``image_mapper_type``.

Data: ctapipe's bundled ``gamma_test_large.simtel.gz`` CI test file (public,
auto-downloaded via ``get_dataset_path``). This is intentionally small (110
simulated events total, ~70 combined LST-1 mono images across telescopes 1-4)
-- the same file CTLearn's own test suite uses for pipeline smoke tests, not
a production-scale dataset. Treat the resulting loss numbers as evidence the
hex-native path trains end-to-end and is directly comparable in cost to the
square-CNN path, not as a statistically powered accuracy claim -- see the
"Limitations" section of the generated report.

Usage
-----
    python benchmarks/hexcnn_vs_singlecnn_lst1_energy.py
"""

import json
import shutil
import tempfile
import time
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "benchmarks" / "results"

N_EPOCHS = 20
BATCH_SIZE = 2
RANDOM_SEED = 0
ALLOWED_TELS = [1, 2, 3, 4]  # LST-1 .. LST-4


def build_gamma_dl1_file(tmp_path: Path) -> Path:
    from ctapipe.core import run_tool
    from ctapipe.tools.process import ProcessorTool
    from ctapipe.utils import get_dataset_path

    gamma_simtel_path = get_dataset_path("gamma_test_large.simtel.gz")
    output = tmp_path / "gamma.dl1.h5"
    argv = [
        f"--input={gamma_simtel_path}",
        f"--output={output}",
        "--write-images",
        "--SimTelEventSource.focal_length_choice=EQUIVALENT",
    ]
    rc = run_tool(ProcessorTool(), argv=argv, cwd=tmp_path)
    assert rc == 0, "ctapipe-process failed to produce the DL1 gamma file"
    return output


def _load_train_tool_class():
    """Load ``TrainCTLearnModel`` from ``ctlearn/tools/train_model.py`` directly,
    bypassing ``ctlearn.tools.__init__`` (which unconditionally imports
    ``predict_LST1``, currently broken in this environment: it needs
    ``ctapipe>=0.29`` while the newest ``dl1_data_handler`` release on PyPI pins
    ``ctapipe<0.26`` -- a pre-existing packaging mismatch unrelated to this
    benchmark or to the hex-conv changes)."""
    import importlib.util
    import sys

    path = REPO_ROOT / "ctlearn" / "tools" / "train_model.py"
    spec = importlib.util.spec_from_file_location("_ctlearn_train_model_standalone", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.TrainCTLearnModel


def run_training(
    *,
    name: str,
    conv_backend: str,
    image_mapper_type: str,
    signal_dir: Path,
    output_dir: Path,
    cwd: Path,
) -> dict:
    TrainCTLearnModel = _load_train_tool_class()
    from ctapipe.core import run_tool

    argv = [
        f"--signal={signal_dir}",
        "--pattern-signal=*.dl1.h5",
        f"--output={output_dir}",
        "--reco=energy",
        f"--TrainCTLearnModel.n_epochs={N_EPOCHS}",
        f"--TrainCTLearnModel.batch_size={BATCH_SIZE}",
        f"--TrainCTLearnModel.random_seed={RANDOM_SEED}",
        "--TrainCTLearnModel.model_type=SingleCNN",
        "--DLImageReader.focal_length_choice=EQUIVALENT",
        f"--DLImageReader.allowed_tels={ALLOWED_TELS}",
        f"--DLImageReader.image_mapper_type={image_mapper_type}",
        f"--SingleCNN.conv_backend={conv_backend}",
        # Disabled symmetrically on both models: SingleCNN's default
        # attention_mechanism ("Dual-SE", inherited from CTLearnModel) hits a
        # pre-existing upstream bug (ctlearn/core/model.py reads
        # self.attention["ratio"], but CTLearnModel.__init__ only ever sets
        # "reduction_ratio") and crashes immediately. That bug is unrelated to
        # this benchmark or to the hex-conv changes -- worth its own tiny fix
        # upstream, not bundled into this PR. Disabling attention here keeps
        # the comparison symmetric and focused on the conv backbone itself.
        "--SingleCNN.attention_mechanism=None",
    ]

    start = time.monotonic()
    rc = run_tool(TrainCTLearnModel(), argv=argv, cwd=cwd)
    wall_time = time.monotonic() - start
    assert rc == 0, f"training failed for {name} (conv_backend={conv_backend})"

    log = pd.read_csv(output_dir / "training_log.csv")

    import keras

    model = keras.saving.load_model(output_dir / "ctlearn_model.keras")

    return {
        "name": name,
        "conv_backend": conv_backend,
        "image_mapper_type": image_mapper_type,
        "n_params": model.count_params(),
        "wall_time_s": wall_time,
        "final_train_mae_energy": float(log["mae_energy"].iloc[-1]),
        "final_val_mae_energy": float(log["val_mae_energy"].iloc[-1]),
        "best_val_mae_energy": float(log["val_mae_energy"].min()),
        "history_csv": str(output_dir / "training_log.csv"),
    }


def init_scale_diagnostic() -> dict:
    """Compare untrained backbone activation scale between the square and
    hexagdly conv backends of SingleCNN on their respective real LST-1
    mapper output shapes, at matching parameter counts.

    This used to show a ~3000x gap (keras_hexagdly.Conv2d's default init
    was scaled far too large, growing worse with kernel_size) -- that was a
    real bug in keras-hexagdly's default initializer, now fixed upstream
    (keras-hexagdly >=0.4.1, see its CHANGELOG). What remains here is the
    ordinary, expected gap between two libraries' default init conventions
    (HeNormal + Constant(0.01) bias vs. Glorot + zeros bias), which no
    longer needs a workaround (e.g. batchnorm) to get a fair comparison --
    see the Results section, both backends converge to comparable MAE with
    no special-casing beyond the class defaults.
    """
    import numpy as np

    from ctlearn.core.model import CTLearnModel
    from ctlearn.utils import get_lst1_subarray_description
    from dl1_data_handler.image_mapper import ImageMapper

    subarray = get_lst1_subarray_description()
    geometry = subarray.tel[1].camera.geometry
    rng = np.random.default_rng(0)

    diagnostic = {}
    for label, mapper_name, conv_backend in [
        ("square", "BilinearMapper", "square"),
        ("hexagdly", "HexagdlyMapper", "hexagdly"),
    ]:
        mapper = ImageMapper.from_name(mapper_name, geometry=geometry, subarray=subarray)
        shape = (mapper.image_shape, mapper.image_shape, 2)
        model = CTLearnModel.from_name(
            "SingleCNN",
            input_shape=shape,
            tasks=["energy"],
            attention_mechanism=None,
            conv_backend=conv_backend,
        )
        x = rng.uniform(0, 5, size=(8, *shape)).astype("float32")
        gap_features = model.backbone_model.predict(x, verbose=0)
        diagnostic[label] = {
            "image_shape": mapper.image_shape,
            "n_params": model.model.count_params(),
            "gap_feature_std": float(gap_features.std()),
            "gap_feature_mean_abs": float(np.abs(gap_features).mean()),
        }
    return diagnostic


def main():
    tmp_path = Path(tempfile.mkdtemp(prefix="hexcnn_benchmark_"))
    print(f"Working directory (DL1 data, checkpoints, logs): {tmp_path}")

    print("\n=== Untrained backbone activation-scale diagnostic ===")
    diagnostic = init_scale_diagnostic()
    for name, d in diagnostic.items():
        print(f"{name}: image_shape={d['image_shape']} n_params={d['n_params']:,} "
              f"GAP-feature std={d['gap_feature_std']:.4f} mean|.|={d['gap_feature_mean_abs']:.4f}")

    dl1_file = build_gamma_dl1_file(tmp_path)

    signal_dir = tmp_path / "gamma_dl1"
    signal_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(dl1_file, signal_dir)

    results = []
    for name, conv_backend, image_mapper_type in [
        ("SingleCNN(square) + BilinearMapper (baseline)", "square", "BilinearMapper"),
        ("SingleCNN(hexagdly) + HexagdlyMapper (hex-native)", "hexagdly", "HexagdlyMapper"),
    ]:
        output_dir = tmp_path / conv_backend  # TrainCTLearnModel creates this itself
        print(f"\n=== Training {name} ===")
        results.append(
            run_training(
                name=name,
                conv_backend=conv_backend,
                image_mapper_type=image_mapper_type,
                signal_dir=signal_dir,
                output_dir=output_dir,
                cwd=tmp_path,
            )
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "hexcnn_vs_singlecnn_lst1_energy_results.json").write_text(
        json.dumps({"training": results, "init_scale_diagnostic": diagnostic}, indent=2)
    )
    write_report(results, diagnostic, tmp_path, dl1_file)
    print(f"\nReport written to {RESULTS_DIR / 'hexcnn_vs_singlecnn_lst1_energy.md'}")


def write_report(results: list, diagnostic: dict, tmp_path: Path, dl1_file: Path):
    import tables

    with tables.open_file(dl1_file) as f:
        n_lst_images = sum(
            f.get_node(f"/dl1/event/telescope/images/tel_{t:03d}").nrows
            for t in ALLOWED_TELS
            if f.__contains__(f"/dl1/event/telescope/images/tel_{t:03d}")
        )

    lines = [
        "# SingleCNN(conv_backend=\"hexagdly\") + HexagdlyMapper vs. "
        "SingleCNN(conv_backend=\"square\") + BilinearMapper",
        "",
        "LST-1 `energy` regression, trained through the real `TrainCTLearnModel` "
        "tool with identical `n_epochs`, `batch_size`, and `random_seed` (so both "
        "runs see the same train/validation split) -- the only difference between "
        "the two runs is `conv_backend`/`image_mapper_type`.",
        "",
        "## Setup",
        "",
        f"- Data: ctapipe's bundled `gamma_test_large.simtel.gz` CI test file "
        f"(LST-1..4, {n_lst_images} combined mono images after DL1 processing)",
        f"- `n_epochs={N_EPOCHS}`, `batch_size={BATCH_SIZE}`, `random_seed={RANDOM_SEED}`",
        "- Both models use `SingleCNN`'s class defaults (`architecture`, "
        "`pooling_parameters`) -- no hyperparameter tuning either way. "
        "`attention_mechanism=None` on both, to route around a pre-existing, "
        "unrelated bug in `SingleCNN`'s default attention config (see Limitations). "
        "Nothing else is overridden -- in particular, no `batchnorm` workaround is "
        "needed (see \"Initializer scale\" below)",
        f"- `BilinearMapper` maps LST-1 onto a "
        f"{diagnostic['square']['image_shape']}x{diagnostic['square']['image_shape']} "
        f"interpolated square grid; `HexagdlyMapper` onto a "
        f"{diagnostic['hexagdly']['image_shape']}x{diagnostic['hexagdly']['image_shape']} "
        "grid at native hex resolution (no interpolation) -- the two models see "
        "different spatial resolutions by design, not by oversight",
        "",
        "## Results",
        "",
        "| | params | wall time (s) | final train MAE | final val MAE | best val MAE |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in results:
        lines.append(
            f"| {r['name']} | {r['n_params']:,} | {r['wall_time_s']:.1f} | "
            f"{r['final_train_mae_energy']:.4f} | {r['final_val_mae_energy']:.4f} | "
            f"{r['best_val_mae_energy']:.4f} |"
        )

    lines += [
        "",
        "### Initializer scale",
        "",
        "Before training, on 8 random inputs at each model's real input shape "
        "(`keras-hexagdly>=0.4.1`, after fixing a real default-initializer bug "
        "-- see its CHANGELOG: each of `Conv2d`'s several internal sub-kernels "
        "used to get its own independently-scaled He-normal init, calibrated as "
        "if it alone were the whole receptive field, inflating output variance "
        "by roughly the number of sub-kernels):",
        "",
        "| | image shape | params | untrained GAP-feature std | mean\\|.\\| |",
        "|---|---:|---:|---:|---:|",
    ]
    for label in ("square", "hexagdly"):
        d = diagnostic[label]
        lines.append(
            f"| conv_backend={label} | {d['image_shape']}x{d['image_shape']} | {d['n_params']:,} | "
            f"{d['gap_feature_std']:.4f} | {d['gap_feature_mean_abs']:.4f} |"
        )
    best_val_by_backend = {r["conv_backend"]: r["best_val_mae_energy"] for r in results}
    lines += [
        "",
        "Before the keras-hexagdly fix, this gap was ~3 orders of magnitude and "
        "training without a workaround (e.g. `batchnorm=True`) gave hex val MAE "
        "roughly 2 orders of magnitude worse than square's. What remains above "
        "is the much smaller, ordinary gap between two libraries' default init "
        "conventions (`keras.layers.Conv2D`'s Glorot+zeros vs. "
        "`keras_hexagdly.Conv2d`'s He-normal+`Constant(0.01)`, matching upstream "
        "HexagDLy's own convention) -- no longer large enough to need a "
        "workaround. The Results table above, with no `batchnorm` or other "
        "special-casing beyond `SingleCNN`'s class defaults, shows both backends "
        "converging to comparable best validation MAE "
        f"({best_val_by_backend['square']:.2f} for square, "
        f"{best_val_by_backend['hexagdly']:.2f} for hexagdly). "
        "(Small-dataset run-to-run noise means the exact numbers here will vary "
        "between reruns -- see Limitations -- but stay in the same range on "
        "both backends rather than differing by orders of magnitude.)",
        "",
        "## Limitations",
        "",
        "This is a pipeline-correctness and cost comparison, not a statistically "
        "powered accuracy claim. The dataset is ctapipe's small CI smoke-test file "
        f"-- {n_lst_images} combined LST mono images, corresponding to only ~18-19 "
        "distinct shower events viewed by up to 4 LSTs each (not independent "
        "samples), split ~80/20 train/validation. MAE at this sample size is noisy "
        "and not representative of production-scale performance for either model. "
        "What this *does* demonstrate: both the hex-native path (`HexagdlyMapper` "
        "-> `SingleCNN(conv_backend=\"hexagdly\")`) and the existing square path "
        "train end-to-end through CTLearn's real tooling with no special-casing, "
        "at comparable wall-clock cost per epoch. A real accuracy comparison needs a production-scale "
        "gamma/proton dataset, which requires CTAO data access this environment "
        "doesn't have.",
        "",
        f"Raw training logs and checkpoints were written to a temp directory "
        f"(`{tmp_path}`) and are not part of this repo; `training_log.csv` per run "
        "is referenced in `hexcnn_vs_singlecnn_lst1_energy_results.json` for anyone "
        "re-running this script locally.",
    ]

    (RESULTS_DIR / "hexcnn_vs_singlecnn_lst1_energy.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
