ctlearn v0.11.0 (2026-10-30)
============================


API Changes
-----------

- Introduced framework-specific model and training APIs, including the public PyTorch classes under ``ctlearn.core.pytorch`` and training tool under ``ctlearn.tools.pytorch``. [`#276 <https://github.com/ctlearn-project/ctlearn/pull/276>`__]

- Removed legacy, non-core PyTorch architectures and utilities. Code importing those modules must migrate to the supported model and training APIs. [`#280 <https://github.com/ctlearn-project/ctlearn/pull/280>`__]

- Split prediction implementations into dedicated mono-, stereo-, and LST-1 modules. Update direct imports of moved implementations to their new module paths. [`#298 <https://github.com/ctlearn-project/ctlearn/pull/298>`__]

- With ``conv_backend="hexagdly"``, ``kernel_size`` and the pooling size are now read as a square-kernel width, like for the square backend, instead of a number of hexagonal rings. The same configuration therefore describes comparable models on both grids: a ``kernel_size`` of 3 is the 7-cell hexagon and 1 is a plain 1x1, which also makes the hardcoded ``kernel_size=1`` projections of the ResNet true 1x1 convolutions. A hexagonal model built from a configuration now has smaller kernels than before, so it must be trained again; models saved with an earlier version still load unchanged. A hexagonal window is centred on a cell, so it has no even size: an even ``kernel_size`` or pooling size is rounded up to the next odd one (2 gives the 7-cell window of 3, not the identity of 1), and the hexagonal layers say so with a warning. Note that the default pooling size of 2 triggers it with ``conv_backend="hexagdly"``. Requires ``keras-hexagdly>=0.6.1`` and ``pytorch-hexagdly>=0.4.1``. [`#312 <https://github.com/ctlearn-project/ctlearn/pull/312>`__]


Bug Fixes
---------

- Fixes the SingleCNN attention ratio configuration being read from the wrong key. [`#287 <https://github.com/ctlearn-project/ctlearn/pull/287>`__]

- Raise a clear ``ToolConfigurationError`` when signal or background file patterns match no files, instead of failing later during training. [`#299 <https://github.com/ctlearn-project/ctlearn/pull/299>`__]

- Fixes the tensorboard output files, so that ``Keras`` and ``PyTorch`` models can be compared in the same tensorboard plots. [`#306 <https://github.com/ctlearn-project/ctlearn/pull/306>`__]

- Fixed CI dependency installation order by installing CTLearn from source before the matrix-selected TensorFlow and PyTorch versions. This ensures the intended framework versions remain installed instead of being upgraded during the source installation. [`#313 <https://github.com/ctlearn-project/ctlearn/pull/313>`__]

- Raise a ``ToolConfigurationError`` during ``TrainCTLearnModel.setup()`` when no reconstruction task (``--reco`` / ``reco_tasks``) is selected, instead of failing later in the training loop. [`#315 <https://github.com/ctlearn-project/ctlearn/pull/315>`__]


Data Model Changes
------------------

- Added support for the new CTAO data product metadata model provided by ctapipe > v0.32.0 and ctao_datamodel >= v1.4.0. [`#317 <https://github.com/ctlearn-project/ctlearn/pull/317>`__]


New Features
------------

- Added PyTorch support for model training and prediction, including model architectures, data loading, and dedicated training tools. [`#276 <https://github.com/ctlearn-project/ctlearn/pull/276>`__]

- Added native hexagonal convolutions to PyTorch models through ``pytorch-hexagdly``. [`#305 <https://github.com/ctlearn-project/ctlearn/pull/305>`__]

- Adds the changelog system to the CTLearn repository following ctapipe. The changelog is generated from the ``ctlearn/docs/changes``  directory, which contains ``{PR_number}.{type}.rst`` files that describe new features, bug fixes, and other changes in a structured format. [`#308 <https://github.com/ctlearn-project/ctlearn/pull/308>`__]

- Added a script (``scripts/overwrite_metadata.py``) to migrate HDF5 product metadata to the current CTAO data model. [`#317 <https://github.com/ctlearn-project/ctlearn/pull/317>`__]


Maintenance
-----------

- Added caching to CI to reuse dependencies and test data between runs. [`#292 <https://github.com/ctlearn-project/ctlearn/pull/292>`__]

- Write the provenance log and tool configuration additionally to the output directory. To ensure reproducibility, the provenance log includes information about the software versions, parameters used, and any relevant metadata. The tool configuration specifies the settings and options used during the execution of the training tool to be reused for the inference phase. Both the provenance log and tool configuration are saved in separate files in a structured format (e.g., JSON or YAML) within the output directory for easy access and reference. [`#304 <https://github.com/ctlearn-project/ctlearn/pull/304>`__]

- Aligns the logging between ``Keras`` and ``PyTorch`` training tools by adopting the ctapipe logger. [`#306 <https://github.com/ctlearn-project/ctlearn/pull/306>`__]

- Added pre-commit checks for spelling, Ruff linting, formatting, and common file issues. [`#310 <https://github.com/ctlearn-project/ctlearn/pull/310>`__]

- Added unit test coverage reporting to CI and a coverage badge to the README. [`#311 <https://github.com/ctlearn-project/ctlearn/pull/311>`__]

- Raised the minimum TensorFlow dependency to 2.18 and specified minimum TensorFlow and PyTorch versions for each supported Python version in the ``pyproject.toml``. Added also a macOS job in the CI. [`#313 <https://github.com/ctlearn-project/ctlearn/pull/313>`__]

- Bumped the minimum supported ``dl1_data_handler`` version to 0.14.12. [`#317 <https://github.com/ctlearn-project/ctlearn/pull/317>`__]


Refactoring and Optimization
----------------------------

- Introduces ``ConvBackend`` classes that hold properties of the convolution backend. This allows selecting the square (standard) or hexagonal (``hexagdly``) model backbones for both frameworks (``Keras`` or ``PyTorch``). [`#307 <https://github.com/ctlearn-project/ctlearn/pull/307>`__]
