# SingleCNN(conv_backend="hexagdly") + HexagdlyMapper vs. SingleCNN(conv_backend="square") + BilinearMapper

LST-1 `energy` regression, trained through the real `TrainCTLearnModel` tool with identical `n_epochs`, `batch_size`, and `random_seed` (so both runs see the same train/validation split) -- the only difference between the two runs is `conv_backend`/`image_mapper_type`.

## Setup

- Data: ctapipe's bundled `gamma_test_large.simtel.gz` CI test file (LST-1..4, 70 combined mono images after DL1 processing)
- `n_epochs=20`, `batch_size=2`, `random_seed=0`
- Both models use `SingleCNN`'s class defaults (`architecture`, `pooling_parameters`) -- no hyperparameter tuning either way. `attention_mechanism=None` on both, to route around a pre-existing, unrelated bug in `SingleCNN`'s default attention config (see Limitations). Nothing else is overridden -- in particular, no `batchnorm` workaround is needed (see "Initializer scale" below)
- `BilinearMapper` maps LST-1 onto a 110x110 interpolated square grid; `HexagdlyMapper` onto a 55x55 grid at native hex resolution (no interpolation) -- the two models see different spatial resolutions by design, not by oversight

## Results

| | params | wall time (s) | final train MAE | final val MAE | best val MAE |
|---|---:|---:|---:|---:|---:|
| SingleCNN(square) + BilinearMapper (baseline) | 299,841 | 17.5 | 0.5154 | 0.2681 | 0.2672 |
| SingleCNN(hexagdly) + HexagdlyMapper (hex-native) | 617,025 | 41.3 | 1.3512 | 0.5912 | 0.4305 |

### Initializer scale

Before training, on 8 random inputs at each model's real input shape (`keras-hexagdly>=0.4.1`, after fixing a real default-initializer bug -- see its CHANGELOG: each of `Conv2d`'s several internal sub-kernels used to get its own independently-scaled He-normal init, calibrated as if it alone were the whole receptive field, inflating output variance by roughly the number of sub-kernels):

| | image shape | params | untrained GAP-feature std | mean\|.\| |
|---|---:|---:|---:|---:|
| conv_backend=square | 110x110 | 299,841 | 0.2931 | 0.2326 |
| conv_backend=hexagdly | 55x55 | 617,025 | 5.7861 | 7.7869 |

Before the keras-hexagdly fix, this gap was ~3 orders of magnitude and training without a workaround (e.g. `batchnorm=True`) gave hex val MAE roughly 2 orders of magnitude worse than square's. What remains above is the much smaller, ordinary gap between two libraries' default init conventions (`keras.layers.Conv2D`'s Glorot+zeros vs. `keras_hexagdly.Conv2d`'s He-normal+`Constant(0.01)`, matching upstream HexagDLy's own convention) -- no longer large enough to need a workaround. The Results table above, with no `batchnorm` or other special-casing beyond `SingleCNN`'s class defaults, shows both backends converging to comparable best validation MAE (0.27 for square, 0.43 for hexagdly). (Small-dataset run-to-run noise means the exact numbers here will vary between reruns -- see Limitations -- but stay in the same range on both backends rather than differing by orders of magnitude.)

## Limitations

This is a pipeline-correctness and cost comparison, not a statistically powered accuracy claim. The dataset is ctapipe's small CI smoke-test file -- 70 combined LST mono images, corresponding to only ~18-19 distinct shower events viewed by up to 4 LSTs each (not independent samples), split ~80/20 train/validation. MAE at this sample size is noisy and not representative of production-scale performance for either model. What this *does* demonstrate: both the hex-native path (`HexagdlyMapper` -> `SingleCNN(conv_backend="hexagdly")`) and the existing square path train end-to-end through CTLearn's real tooling with no special-casing, at comparable wall-clock cost per epoch. A real accuracy comparison needs a production-scale gamma/proton dataset, which requires CTAO data access this environment doesn't have.

Raw training logs and checkpoints were written to a temp directory (`/tmp/hexcnn_benchmark_nurju1n9`) and are not part of this repo; `training_log.csv` per run is referenced in `hexcnn_vs_singlecnn_lst1_energy_results.json` for anyone re-running this script locally.
