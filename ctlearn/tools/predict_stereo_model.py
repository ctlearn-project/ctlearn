"""
Tool to predict the gammaness, energy and arrival direction in stereoscopic mode using ``CTLearnModel`` on R1/DL1 data using the ``DLDataReader`` and ``KerasSequence``/``PyTorchDataset``.
"""

__all__ = [
    "StereoPredictCTLearnModel",
] 

import numpy as np
from astropy.table import (
    Table,
    vstack,
    join,
    setdiff,
    unique,
)

from ctapipe.containers import (
    ParticleClassificationContainer,
    ReconstructedGeometryContainer,
    ReconstructedEnergyContainer,
)
from ctapipe.io import read_table, write_table
from ctapipe.io.hdf5dataformat import (
    DL1_SUBARRAY_GROUP,
    DL1_SUBARRAY_POINTING_GROUP,
    DL1_TEL_TRIGGER_TABLE,
    DL2_EVENT_STATISTICS_GROUP,
    SIMULATION_RUN_TABLE,
    DL2_SUBARRAY_PARTICLETYPE_GROUP,
    DL2_SUBARRAY_ENERGY_GROUP,
    DL2_SUBARRAY_GEOMETRY_GROUP,
)
from ctapipe.reco.utils import add_defaults_and_meta
from ctlearn.tools.predict_model import PredictCTLearnModel
from dl1_data_handler.reader import ProcessType

# Convienient constants for column names and table keys
SUBARRAY_EVENT_KEYS = ["obs_id", "event_id"]


class StereoPredictCTLearnModel(PredictCTLearnModel):
    """
    Tool to predict the gammaness, energy and arrival direction from R1/DL1 stereoscopic data using CTLearn models.

    This tool extends the ``PredictCTLearnModel`` to specifically handle stereoscopic R1/DL1 data. The prediction
    is performed using the CTLearn models. The data is stored in the output file following the ctapipe DL2 data format.
    It also stores the telescope/subarray pointing monitoring and DL1 feature vectors (if selected) in the output file.
    By default, waveforms and images are not copied to the DL2 output file unless explicitly enabled in the configuration.

    Attributes
    ----------
    name : str
        Name of the tool.
    description : str
        Description of the tool.
    examples : str
        Examples of how to use the tool.

    Methods
    -------
    start()
        Start the tool.
    _store_mc_subarray_pointing(all_identifiers)
        Store the subarray pointing table for the stereo mode for MC simulation.
    """

    name = "ctlearn-predict-stereo-model"
    description = __doc__

    examples = """
    To predict from pixel-wise image data in stereo mode using trained CTLearn models:
    > ctlearn-predict-stereo-model \
        --input_url input.dl1.h5 \
        --PredictCTLearnModel.batch_size=16 \
        --PredictCTLearnModel.dl1dh_reader_type=DLImageReader \
        --DLImageReader.channels=cleaned_image \
        --DLImageReader.channels=cleaned_relative_peak_time \
        --DLImageReader.image_mapper_type=BilinearMapper \
        --DLImageReader.mode=stereo \
        --DLImageReader.min_telescopes=2 \
        --PredictCTLearnModel.stack_telescope_images=True \
        --type_model="/path/to/your/stereo/type/ctlearn_model(.keras/.pth)" \
        --energy_model="/path/to/your/stereo/energy/ctlearn_model(.keras/.pth)" \
        --skydirection_model="/path/to/your/stereo/skydirection/ctlearn_model(.keras/.pth)" \
        --output output.dl2.h5 \
    """

    def start(self):
        """
        Execute the core logic of the tool.

        Depending on the tool, this method either orchestrates the training and validation 
        loops across epochs, or it iterates through the input dataset to generate 
        and save model predictions to the output file.
        """
        self.log.info("Processing the telescope pointings...")
        # Retrieve the IDs from the dl1dh for the prediction tables
        example_identifiers = self.dl1dh_reader.unique_example_identifiers.copy()
        example_identifiers.keep_columns(SUBARRAY_EVENT_KEYS)
        all_identifiers = read_table(
            self.output_path,
            DL1_TEL_TRIGGER_TABLE,
        )
        all_identifiers.keep_columns(SUBARRAY_EVENT_KEYS + ["time"])
        # Unique example identifiers by events
        all_identifiers = unique(all_identifiers, keys=SUBARRAY_EVENT_KEYS)
        nonexample_identifiers = setdiff(
            all_identifiers, example_identifiers, keys=SUBARRAY_EVENT_KEYS
        )
        nonexample_identifiers.remove_column("time")
        # Construct the survival telescopes for each event of the example_identifiers
        survival_telescopes = []
        for subarray_event in self.dl1dh_reader.example_identifiers_grouped.groups:
            survival_mask = np.zeros(len(self.dl1dh_reader.tel_ids), dtype=bool)
            survival_tels = [
                self.dl1dh_reader.subarray.tel_indices[tel_id]
                for tel_id in subarray_event["tel_id"].data
            ]
            survival_mask[survival_tels] = True
            survival_telescopes.append(survival_mask)
        # Add the survival telescopes to the example_identifiers
        example_identifiers.add_column(
            survival_telescopes, name=f"{self.prefixes['all']}_telescopes"
        )
        # Pointing table for the stereo mode for MC simulation
        if self.dl1dh_reader.process_type == ProcessType.Simulation:
            pointing_info = self._store_mc_subarray_pointing(all_identifiers)

        # Pointing table for the observation mode
        if self.dl1dh_reader.process_type == ProcessType.Observation:
            pointing_info = super()._store_pointing(all_identifiers)

        self.log.info("Starting the prediction...")
        particletype_feature_vectors = None
        if self.load_type_model_from is not None:
            # Predict the classification of the primary particle
            particletype_table, particletype_feature_vectors = (
                super()._predict_particletype(example_identifiers)
            )
            if self.dl2_subarray:
                particletype_table.rename_column(
                    f"{self.prefixes['all']}_telescopes",
                    f"{self.prefixes['type']}_telescopes",
                )
                # Produce output table with NaNs for missing predictions
                if len(nonexample_identifiers) > 0:
                    nan_table = super()._create_nan_table(
                        nonexample_identifiers,
                        columns=[f"{self.prefixes['type']}_tel_prediction"],
                        shapes=[(len(nonexample_identifiers),)],
                        reco_task="type",
                    )
                    particletype_table = vstack([particletype_table, nan_table])
                # Add is_valid column to the particletype table
                particletype_table.add_column(
                    ~np.isnan(
                        particletype_table[
                            f"{self.prefixes['type']}_tel_prediction"
                        ].data,
                        dtype=bool,
                    ),
                    name=f"{self.prefixes['type']}_is_valid",
                )
                # Rename the columns for the stereo mode
                particletype_table.rename_column(
                    f"{self.prefixes['type']}_tel_prediction",
                    f"{self.prefixes['type']}_prediction",
                )
                # Deduplicate the subarray particletype table to have only one entry per event
                particletype_table = super().deduplicate_first_valid(
                    table=particletype_table,
                    keys=SUBARRAY_EVENT_KEYS,
                    valid_col=f"{self.prefixes['type']}_is_valid",
                )
                particletype_table.sort(SUBARRAY_EVENT_KEYS)
                # Add the default values and meta data to the table
                add_defaults_and_meta(
                    particletype_table,
                    ParticleClassificationContainer,
                    prefix=self.prefixes["type"],
                )
                # Save the prediction to the output file
                write_table(
                    particletype_table,
                    self.output_path,
                    f"{DL2_SUBARRAY_PARTICLETYPE_GROUP}/{self.prefixes['type']}",
                )
                self.log.info(
                    "DL2 prediction data was stored in '%s' under '%s'",
                    self.output_path,
                    f"{DL2_SUBARRAY_PARTICLETYPE_GROUP}/{self.prefixes['type']}",
                )
            # Store the telescope event statistics table
            write_table(
                self.dl1dh_reader.quality_query.to_table(functions=True),
                self.output_path,
                f"{DL2_EVENT_STATISTICS_GROUP}/{self.prefixes['type']}",
                append=True,
            )
            self.log.info(
                "DL2 service telescope event statistics data was stored in '%s' under '%s'",
                self.output_path,
                f"{DL2_EVENT_STATISTICS_GROUP}/{self.prefixes['type']}",
            )
        energy_feature_vectors = None
        if self.load_energy_model_from is not None:
            # Predict the energy of the primary particle
            energy_table, energy_feature_vectors = super()._predict_energy(
                example_identifiers
            )
            if self.dl2_subarray:
                energy_table.rename_column(
                    f"{self.prefixes['all']}_telescopes",
                    f"{self.prefixes['energy']}_telescopes",
                )
                # Produce output table with NaNs for missing predictions
                if len(nonexample_identifiers) > 0:
                    nan_table = super()._create_nan_table(
                        nonexample_identifiers,
                        columns=[f"{self.prefixes['energy']}_tel_energy"],
                        shapes=[(len(nonexample_identifiers),)],
                        reco_task="energy",
                    )
                    energy_table = vstack([energy_table, nan_table])
                # Add is_valid column to the energy table
                energy_table.add_column(
                    ~np.isnan(
                        energy_table[f"{self.prefixes['energy']}_tel_energy"].data,
                        dtype=bool,
                    ),
                    name=f"{self.prefixes['energy']}_is_valid",
                )
                # Rename the columns for the stereo mode
                energy_table.rename_column(
                    f"{self.prefixes['energy']}_tel_energy",
                    f"{self.prefixes['energy']}_energy",
                )
                # Deduplicate the subarray energy table to have only one entry per event
                energy_table = super().deduplicate_first_valid(
                    table=energy_table,
                    keys=SUBARRAY_EVENT_KEYS,
                    valid_col=f"{self.prefixes['energy']}_is_valid",
                )
                energy_table.sort(SUBARRAY_EVENT_KEYS)
                # Add the default values and meta data to the table
                add_defaults_and_meta(
                    energy_table,
                    ReconstructedEnergyContainer,
                    prefix=self.prefixes["energy"],
                )
                # Save the prediction to the output file
                write_table(
                    energy_table,
                    self.output_path,
                    f"{DL2_SUBARRAY_ENERGY_GROUP}/{self.prefixes['energy']}",
                )
                self.log.info(
                    "DL2 prediction data was stored in '%s' under '%s'",
                    self.output_path,
                    f"{DL2_SUBARRAY_ENERGY_GROUP}/{self.prefixes['energy']}",
                )
            # Store the telescope event statistics table
            write_table(
                self.dl1dh_reader.quality_query.to_table(functions=True),
                self.output_path,
                f"{DL2_EVENT_STATISTICS_GROUP}/{self.prefixes['energy']}",
                append=True,
            )
            self.log.info(
                "DL2 service telescope event statistics data was stored in '%s' under '%s'",
                self.output_path,
                f"{DL2_EVENT_STATISTICS_GROUP}/{self.prefixes['energy']}",
            )
        direction_feature_vectors = None
        if self.load_skydirection_model_from is not None:
            # Join the prediction table with the telescope pointing table
            example_identifiers = join(
                left=example_identifiers,
                right=pointing_info,
                keys=SUBARRAY_EVENT_KEYS,
            )
            # Predict the arrival direction of the primary particle
            direction_table, direction_feature_vectors = super()._predict_skydirection(
                example_identifiers
            )
            if self.dl2_subarray:
                direction_table.rename_column(
                    f"{self.prefixes['all']}_telescopes",
                    f"{self.prefixes['skydirection']}_telescopes",
                )
                # Transform the spherical coordinate offsets to sky coordinates
                direction_table = super()._transform_spher_coord_offsets_to_sky(
                    direction_table
                )
                # Produce output table with NaNs for missing predictions
                if len(nonexample_identifiers) > 0:
                    nan_table = super()._create_nan_table(
                        nonexample_identifiers,
                        columns=[
                            f"{self.prefixes['skydirection']}_alt",
                            f"{self.prefixes['skydirection']}_az",
                        ],
                        shapes=[
                            (len(nonexample_identifiers),),
                            (len(nonexample_identifiers),),
                        ],
                        reco_task="skydirection",
                    )
                    direction_table = vstack([direction_table, nan_table])
                # Add is_valid column to the direction table
                direction_table.add_column(
                    ~np.isnan(
                        direction_table[f"{self.prefixes['skydirection']}_alt"].data,
                        dtype=bool,
                    ),
                    name=f"{self.prefixes['skydirection']}_is_valid",
                )
                # Deduplicate the subarray direction table to have only one entry per event
                direction_table = super().deduplicate_first_valid(
                    table=direction_table,
                    keys=SUBARRAY_EVENT_KEYS,
                    valid_col=f"{self.prefixes['skydirection']}_is_valid",
                )
                direction_table.sort(SUBARRAY_EVENT_KEYS)
                # Add the default values and meta data to the table
                add_defaults_and_meta(
                    direction_table,
                    ReconstructedGeometryContainer,
                    prefix=self.prefixes["skydirection"],
                )
                # Save the prediction to the output file
                write_table(
                    direction_table,
                    self.output_path,
                    f"{DL2_SUBARRAY_GEOMETRY_GROUP}/{self.prefixes['skydirection']}",
                )
                self.log.info(
                    "DL2 prediction data was stored in '%s' under '%s'",
                    self.output_path,
                    f"{DL2_SUBARRAY_GEOMETRY_GROUP}/{self.prefixes['skydirection']}",
                )
            # Store the telescope event statistics table
            write_table(
                self.dl1dh_reader.quality_query.to_table(functions=True),
                self.output_path,
                f"{DL2_EVENT_STATISTICS_GROUP}/{self.prefixes['skydirection']}",
                append=True,
            )
            self.log.info(
                "DL2 service telescope event statistics data was stored in '%s' under '%s'",
                self.output_path,
                f"{DL2_EVENT_STATISTICS_GROUP}/{self.prefixes['skydirection']}",
            )
        # Create the feature vector table if the DL1 features are enabled
        if self.dl1_features:
            self.log.info("Processing and storing dl1 feature vectors...")
            feature_vector_table = super()._create_feature_vectors_table(
                example_identifiers,
                nonexample_identifiers,
                particletype_feature_vectors,
                energy_feature_vectors,
                direction_feature_vectors,
            )
            # Loop over the selected telescopes and store the feature vectors
            # for each telescope in the output file. The feature vectors are stored
            # in the DL1_TEL_GROUP/features/{self.prefixes['all']}/tel_{tel_id:03d} table.
            # Rename the columns for the stereo mode
            feature_vector_table.rename_column(
                f"{self.prefixes['all']}_tel_particletype_feature_vectors",
                f"{self.prefixes['all']}_particletype_feature_vectors",
            )
            feature_vector_table.rename_column(
                f"{self.prefixes['all']}_tel_energy_feature_vectors",
                f"{self.prefixes['all']}_energy_feature_vectors",
            )
            feature_vector_table.rename_column(
                f"{self.prefixes['all']}_tel_geometry_feature_vectors",
                f"{self.prefixes['all']}_geometry_feature_vectors",
            )
            feature_vector_table.rename_column(
                f"{self.prefixes['all']}_tel_is_valid",
                f"{self.prefixes['all']}_is_valid",
            )
            feature_vector_table.sort(SUBARRAY_EVENT_KEYS)
            # Save the prediction to the output file
            write_table(
                feature_vector_table,
                self.output_path,
                f"{DL1_SUBARRAY_GROUP}/features/{self.prefixes['all']}",
            )
            self.log.info(
                "DL1 feature vectors was stored in '%s' under '%s'",
                self.output_path,
                f"{DL1_SUBARRAY_GROUP}/features/{self.prefixes['all']}",
            )

    def _store_mc_subarray_pointing(self, all_identifiers):
        """
        Store the subarray pointing table from MC simulation to the output file.

        Parameters:
        -----------
        all_identifiers : astropy.table.Table
            Table containing the subarray pointing information.
        """
        # Read the subarray pointing table
        pointing_info = read_table(
            self.input_url,
            SIMULATION_RUN_TABLE,
        )
        # Assuming min_az = max_az and min_alt = max_alt
        pointing_info.keep_columns(["obs_id", "min_az", "min_alt"])
        pointing_info.rename_column("min_az", "pointing_azimuth")
        pointing_info.rename_column("min_alt", "pointing_altitude")
        # Join the prediction table with the telescope pointing table
        pointing_info = join(
            left=pointing_info,
            right=all_identifiers,
            keys=["obs_id"],
        )
        # TODO: use keep_order for astropy v7.0.0
        pointing_info.sort(SUBARRAY_EVENT_KEYS)
        # Create the pointing table
        pointing_table = Table(
            {
                "time": pointing_info["time"],
                "array_azimuth": pointing_info["pointing_azimuth"],
                "array_altitude": pointing_info["pointing_altitude"],
                "array_ra": np.nan * np.ones(len(pointing_info)),
                "array_dec": np.nan * np.ones(len(pointing_info)),
            }
        )
        # Save the pointing table to the output file
        write_table(
            pointing_table,
            self.output_path,
            DL1_SUBARRAY_POINTING_GROUP,
        )
        self.log.info(
            "DL1 subarray pointing table was stored in '%s' under '%s'",
            self.output_path,
            DL1_SUBARRAY_POINTING_GROUP,
        )
        return pointing_info


def main():
    """
    Main entry point for the command-line tool.

    This function instantiates the Tool class and invokes its `run()` method, 
    which sequentially executes the `setup()`, `start()`, and `finish()` methods.
    """
    # Run the tool
    tool = StereoPredictCTLearnModel()
    tool.run()


if __name__ == "__main__":
    main()