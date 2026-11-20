%YAML 1.1
---
cwlVersion: v1.2
class: CommandLineTool
baseCommand: []
label: Train CTLearn Deep Learning Model
doc: |
  Train a deep learning model for RECO reconstruction tasks using CTLearn.

  Supports training for the following reconstruction tasks:
  - Classification of the primary particle type (gamma/proton)
  - Regression of the primary particle energy
  - Regression of the primary particle arrival direction (camera or sky coordinates)

  This tool wraps both Keras and PyTorch training backends. The framework
  is selected via the ``framework`` parameter, and all configuration details
  are specified in the configuration files.

inputs:
  framework:
    type:
      type: enum
      symbols:
        - keras
        - pytorch
      doc: |
        Deep learning framework to use for training.
        - keras: Use Keras/TensorFlow backend (ctlearn-train-keras-model)
        - pytorch: Use PyTorch backend (ctlearn-train-pytorch-model)
    doc: |
      Deep learning framework to use for training.
    inputBinding:
      position: -1
      valueFrom: ctlearn-train-$(self)-model

  signal_input:
    type: Directory
    doc: |
      Path to directory containing signal (gamma-ray) DL1 input data files.
    inputBinding:
      prefix: --signal

  pattern_signal:
    type:
      type: array
      items: string
    doc: |
      Glob patterns to select signal event files. Multiple patterns can be specified.
      Examples: "gamma_*_run00.dl1.h5", "gamma_*_run*.dl1.h5"
    inputBinding:
      prefix: --pattern-signal

  background_input:
    type: Directory?
    doc: |
      Path to directory containing background (proton/electron) DL1 input data files.
      Required for classification tasks, optional for regression tasks.
    inputBinding:
      prefix: --background

  pattern_background:
    type:
      type: array
      items: string
    default: []
    doc: |
      Glob patterns to select background event files. Multiple patterns can be specified.
      Required if background_input is provided.
    inputBinding:
      prefix: --pattern-background

  model_path:
    type: string?
    doc: |
      Optional destination for a copy of the trained model, relative to the
      CWL working directory. Must end in .keras for Keras or .pth for PyTorch.
    inputBinding:
      prefix: --model-path

  output_directory:
    type: string
    doc: |
      Path to output directory where trained model and logs will be saved.
    default: "./"
    inputBinding:
      prefix: --output

  config_files:
    type: File[]
    doc: |
      Configuration files for training. Typically includes:
      - DL1 data handler configuration (dl1dh_example_config.json)
      - CTLearn training configuration (ctlearn_train_model_example_config.json)
      - Model architecture configuration (ctlearn_models_example_config.json)
    inputBinding:
      prefix: --config

  reconstruction_task:
    type:
      type: enum
      symbols:
        - type
        - energy
        - cameradirection
        - skydirection
    doc: |
      Reconstruction task to train for:
      - type: Particle type classification (gamma vs proton)
      - energy: Regression of primary particle energy
      - cameradirection: Regression of arrival direction in camera coordinates
      - skydirection: Regression of arrival direction in sky coordinates
    inputBinding:
      prefix: --reco

  provenance_log_filename:
    type: string?
    doc: file in which to write the local provenance.
    default: ctlearn-train-model.provenance.log
    inputBinding:
      prefix: --provenance-log

  csv_summary_filename:
    type: string?
    doc: |
      Destination path relative to the CWL working directory for the additional
      CSV training summary; must end in .csv.
    inputBinding:
      prefix: --csv-summary

  tensorboard_train_log_filename:
    type: string?
    doc: |
      Destination path relative to the CWL working directory for the additional
      TensorBoard training event file.
      Include "tfevents" in the filename for TensorBoard to discover it.
    inputBinding:
      prefix: --tensorboard-train-log

  tensorboard_val_log_filename:
    type: string?
    doc: |
      Destination path relative to the CWL working directory for the additional
      TensorBoard validation event file.
      Include "tfevents" in the filename for TensorBoard to discover it.
    inputBinding:
      prefix: --tensorboard-val-log

outputs:
  model_directory:
    type: Directory
    doc: |
      Output directory containing the trained model and associated logs.
    outputBinding:
      glob: $(inputs.output_directory)

  trained_model:
    type: File?
    doc: |
      Trained model file.
      For Keras: .keras format
      For PyTorch: .pth format
    outputBinding:
      glob: $(inputs.model_path)

  provenance_log:
    type: File?
    doc: ctapipe-format provenance log for this step
    outputBinding:
      glob: $(inputs.provenance_log_filename)

  csv_summary:
    type: File?
    doc: CSV training summary.
    outputBinding:
      glob: $(inputs.csv_summary_filename)

  tensorboard_train_log:
    type: File?
    doc: TensorBoard training event file.
    outputBinding:
      glob: $(inputs.tensorboard_train_log_filename)

  tensorboard_val_log:
    type: File?
    doc: TensorBoard validation event file.
    outputBinding:
      glob: $(inputs.tensorboard_val_log_filename)
