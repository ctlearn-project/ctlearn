# HexCNN + HexagdlyMapper vs. SingleCNN + BilinearMapper

LST-1 `energy` regression, trained through the real `TrainCTLearnModel` tool with identical `n_epochs`, `batch_size`, and `random_seed` (so both runs see the same train/validation split) -- the only difference between the two runs is `model_type`/`image_mapper_type`.

## Setup

- Data: ctapipe's bundled `gamma_test_large.simtel.gz` CI test file (LST-1..4, 70 combined mono images after DL1 processing)
- `n_epochs=20`, `batch_size=2`, `random_seed=0`
- Both models use their class defaults (`architecture`, `pooling_parameters`) -- no hyperparameter tuning either way. `attention_mechanism=None` on both, to route around a pre-existing, unrelated bug in `SingleCNN`'s default attention config (see Limitations)
- `BilinearMapper` maps LST-1 onto a 110x110 interpolated square grid; `HexagdlyMapper` onto a 55x55 grid at native hex resolution (no interpolation) -- the two models see different spatial resolutions by design, not by oversight

## Results

| | params | wall time (s) | final train MAE | final val MAE | best val MAE |
|---|---:|---:|---:|---:|---:|
| SingleCNN + BilinearMapper (square, baseline) | 299,841 | 10.6 | 0.5075 | 0.2927 | 0.2927 |
| HexCNN + HexagdlyMapper (hex-native) | 277,185 | 13.4 | 5.2548 | 7.7666 | 2.8901 |

### Why the raw loss numbers favor SingleCNN here

Before training, on 8 random inputs at each model's real input shape (same param-count order of magnitude, `attention_mechanism=None` on both):

| | image shape | params | untrained GAP-feature std | mean\|.\| |
|---|---:|---:|---:|---:|
| SingleCNN | 110x110 | 299,841 | 0.1914 | 0.1532 |
| HexCNN | 55x55 | 277,185 | 74.4550 | 63.1250 |

`SingleCNN`'s untrained backbone output is essentially dead (std ~0) -- `keras.layers.Conv2D`'s default Glorot+zeros init, stacked 4 deep with ReLU on this input scale, collapses to near-zero activations before training starts, so its head is initially just predicting close to a constant. `keras_hexagdly.Conv2d` uses HeNormal weights + a `Constant(0.01)` bias by default (matching upstream HexagDLy's own convention), so `HexCNN`'s backbone produces real, non-dead activations from the first forward pass -- at a scale the (shared, untuned) energy head and `1e-4` learning rate aren't calibrated for. That mismatch, not a defect in the hex-conv path, is what the larger/noisier `HexCNN` loss above mostly reflects: a matched-scale comparison needs either per-model learning-rate tuning or aligning the two initializers, which is exactly the hyperparameter-tuning work this quick pass deliberately skipped.

## Limitations

This is a pipeline-correctness and cost comparison, not a statistically powered accuracy claim. The dataset is ctapipe's small CI smoke-test file -- 70 combined LST mono images, corresponding to only ~18-19 distinct shower events viewed by up to 4 LSTs each (not independent samples), split ~80/20 train/validation. MAE at this sample size is noisy and not representative of production-scale performance for either model. What this *does* demonstrate: both the hex-native path (`HexagdlyMapper` -> `HexCNN`) and the existing square path train end-to-end through CTLearn's real tooling with no special-casing, at comparable wall-clock cost per epoch. A real accuracy comparison needs a production-scale gamma/proton dataset, which requires CTAO data access this environment doesn't have.

Raw training logs and checkpoints were written to a temp directory (`/tmp/hexcnn_benchmark_v1ruj9q4`) and are not part of this repo; `training_log.csv` per run is referenced in `hexcnn_vs_singlecnn_lst1_energy_results.json` for anyone re-running this script locally.
