"""
Tools to train a ``CTLearnModel` (in Keras or PyTorch) on R1/DL1a data using the ``DLDataReader`` and ``DLDataLoader``.
"""

import atexit
import pandas as pd
import numpy as np
import shutil
import tensorflow as tf
import sys
from ctapipe.core import Tool
from ctapipe.core.tool import ToolConfigurationError
from ctapipe.core.traits import (
    Bool,
    CaselessStrEnum,
    Path,
    Float,
    Int,
    List,
    Dict,
    classes_with_traits,
    ComponentName,
    Unicode,
)
from dl1_data_handler.reader import DLDataReader
from ctlearn.core.loader import DLDataLoader
from ctlearn.core.model import CTLearnModel
from ctlearn.utils import validate_trait_dict
from ctlearn.tools.pytorch.train_pytorch_model import TrainPyTorchModel
from ctlearn.tools.keras.train_keras_model import TrainKerasModel
# from ctlearn.tools.base_train_model import TrainCTLearnModel
import importlib.util

from enum import Enum
class FrameworkType(Enum):
    KERAS = 1
    PYTORCH = 2


class DLFrameWork(Tool):
    name = "dlframework"
    framework_type = CaselessStrEnum(
        ["pytorch", "keras"],
        default_value="pytorch",
        help="Framework to use",
    ).tag(config=True)

    aliases = {
        "signal": "TrainCTLearnModel.input_dir_signal",
        "background": "TrainCTLearnModel.input_dir_background",
        "pattern-signal": "TrainCTLearnModel.file_pattern_signal",
        "pattern-background": "TrainCTLearnModel.file_pattern_background",
        "reco": "TrainCTLearnModel.reco_tasks",
        ("o", "output"): "TrainCTLearnModel.output_dir",
    }

    flags = {
        "overwrite": (
            {"TrainCTLearnModel": {"overwrite": True}},
            "Overwrite existing files",
        ),
    }

    classes = (
        [
            CTLearnModel,
            DLDataReader,
        ]
        + classes_with_traits(CTLearnModel)
        + classes_with_traits(DLDataReader)
    )

    def setup(self):
        # Check if the output directory exists and if it should be overwritten
        if self.output_dir.exists():
            if not self.overwrite:
                raise ToolConfigurationError(
                    f"Output directory {self.output_dir} already exists. Use --overwrite to overwrite."
                )
            else:
                # Remove the output directory if it exists
                self.log.info("Removing existing output directory %s", self.output_dir)
                shutil.rmtree(self.output_dir)
        # Create a MirroredStrategy.
        self.strategy = tf.distribute.MirroredStrategy()
        atexit.register(self.strategy._extended._collective_ops._lock.locked)  # type: ignore
        self.log.info("Number of devices: %s", self.strategy.num_replicas_in_sync)
        # Get signal input files
        self.input_url_signal = []
        for signal_pattern in self.file_pattern_signal:
            self.input_url_signal.extend(self.input_dir_signal.glob(signal_pattern))
        # Get bkg input files
        self.input_url_background = []
        if self.input_dir_background is not None:
            for background_pattern in self.file_pattern_background:
                self.input_url_background.extend(self.input_dir_background.glob(background_pattern))

        # Set up the data reader
        self.log.info("Loading data:")
        self.log.info("For a large dataset, this may take a while...")
        if self.dl1dh_reader_type == "DLFeatureVectorReader":
            raise NotImplementedError(
                "'DLFeatureVectorReader' is not supported in CTLearn yet. "
                "Missing stereo CTLearnModel implementation."
            )
        self.dl1dh_reader = DLDataReader.from_name(
            self.dl1dh_reader_type,
            input_url_signal=sorted(self.input_url_signal),
            input_url_background=sorted(self.input_url_background),
            parent=self,
        )
        self.log.info("Number of events loaded: %s", self.dl1dh_reader._get_n_events())
        if "type" in self.reco_tasks:
            self.log.info("Number of signal events: %d", self.dl1dh_reader.n_signal_events)
            self.log.info("Number of background events: %d", self.dl1dh_reader.n_bkg_events)
        # Check if the number of events is enough to form a batch
        if self.dl1dh_reader._get_n_events() < self.batch_size:
            raise ValueError(
                f"{self.dl1dh_reader._get_n_events()} events are not enough "
                f"to form a batch of size {self.batch_size}. Reduce the batch size."
            )
        # Check if there are at least two classes in the reader for the particle classification
        if self.dl1dh_reader.class_weight is None and "type" in self.reco_tasks:
            raise ValueError(
                "Classification task selected but less than two classes are present in the data."
            )
        # Check if stereo mode is selected for stacking telescope images
        if self.stack_telescope_images and self.dl1dh_reader.mode == "mono":
            raise ToolConfigurationError(
                f"Cannot stack telescope images in mono mode. Use stereo mode for stacking."
            )
        # Ckeck if only one telescope type is selected for stacking telescope images
        if self.stack_telescope_images and len(list(self.dl1dh_reader.selected_telescopes)) > 1:
            raise ToolConfigurationError(
                f"Cannot stack telescope images from multiple telescope types. Use only one telescope type."
            )
        # Check if sorting by intensity is disabled for stacking telescope images
        if self.stack_telescope_images and self.sort_by_intensity:
            raise ToolConfigurationError(
                f"Cannot stack telescope images when sorting by intensity. Disable sorting by intensity."
            )

        # Set up the data loaders for training and validation
        indices = list(range(self.dl1dh_reader._get_n_events()))
        # Shuffle the indices before the training/validation split
        np.random.seed(self.random_seed)
        np.random.shuffle(indices)
        n_validation_examples = int(self.validation_split * self.dl1dh_reader._get_n_events())
        training_indices = indices[n_validation_examples:]
        validation_indices = indices[:n_validation_examples]
        self.training_loader = DLDataLoader(
            self.dl1dh_reader,
            training_indices,
            tasks=self.reco_tasks,
            batch_size=self.batch_size*self.strategy.num_replicas_in_sync,
            random_seed=self.random_seed,
            sort_by_intensity=self.sort_by_intensity,
            stack_telescope_images=self.stack_telescope_images,
        )
        self.validation_loader = DLDataLoader(
            self.dl1dh_reader,
            validation_indices,
            tasks=self.reco_tasks,
            batch_size=self.batch_size*self.strategy.num_replicas_in_sync,
            random_seed=self.random_seed,
            sort_by_intensity=self.sort_by_intensity,
            stack_telescope_images=self.stack_telescope_images,
        )

    def finish(self):
        pass

    @classmethod
    def string_to_type(self,str_type:str)->FrameworkType:

        type_=None
        str_type = str.upper(str_type)
        try:
            type_ = FrameworkType[str_type]
        except KeyError:
            print(f"'{str_type}' is not a valid enum type.")
        return type_
    
    def get_framework(self, framework_type: FrameworkType):
        if framework_type == FrameworkType.KERAS:
            if not self.is_package_available("tensorflow"):
               raise ImportError("TensorFlow is not installed. Cannot run Keras framework.")
            else: 
                fw = TrainKerasModel()

        elif framework_type == FrameworkType.PYTORCH:
            if not self.is_package_available("torch"):
                raise ImportError("PyTorch is not installed. Cannot run PyTorch framework.")
            else:            
                fw = TrainPyTorchModel()
        else:
            raise ValueError("Unknown Framework")

        return fw
    
    def is_package_available(self, package_name: str) -> bool:
        return importlib.util.find_spec(package_name) is not None

if __name__ == "__main__":

    DLFrameWork().launch_instance()
 