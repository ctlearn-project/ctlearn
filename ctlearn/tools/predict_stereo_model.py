"""
Tools to predict the gammaness, energy and arrival direction in stereoscopic mode using ``CTLearnModel`` on R1/DL1 data using the ``DLDataReader`` and ``DLDataLoader``.
"""

import numpy as np
from astropy.table import (
    Table,
    vstack,
    join,
    setdiff,
)

from ctapipe.containers import (
    ParticleClassificationContainer,
    ReconstructedGeometryContainer,
    ReconstructedEnergyContainer,
)
from ctapipe.reco.utils import add_defaults_and_meta
from dl1_data_handler.reader import ProcessType
from ctlearn.tools.predict_model import PredictCTLearnModel

SIMULATION_CONFIG_TABLE = "/configuration/simulation/run"
FIXED_POINTING_GROUP = "/configuration/telescope/pointing"
POINTING_GROUP = "/dl1/monitoring/telescope/pointing"
SUBARRAY_POINTING_GROUP = "/dl1/monitoring/subarray/pointing"
DL1_TELESCOPE_GROUP = "/dl1/event/telescope"
DL1_SUBARRAY_GROUP = "/dl1/event/subarray"
DL2_SUBARRAY_GROUP = "/dl2/event/subarray"
DL2_TELESCOPE_GROUP = "/dl2/event/telescope"
SUBARRAY_EVENT_KEYS = ["obs_id", "event_id"]
TELESCOPE_EVENT_KEYS = ["obs_id", "event_id", "tel_id"]

__all__ = ["StereoPredictCTLearnModel"]


class StereoPredictCTLearnModel(PredictCTLearnModel):
    """
    Tool to predict the gammaness, energy and arrival direction from R1/DL1 stereoscopic data using CTLearn models.

    This tool extends the ``PredictCTLearnModel`` to specifically handle stereoscopic R1/DL1 data. The prediction
    is performed using the CTLearn models. The data is stored in the output file following the ctapipe DL2 data format.
    It also stores the telescope/subarray pointing monitoring and DL1 feature vectors (if selected) in the output file.

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
    > ctlearn-predict-stereo-model \\
        --input_url input.dl1.h5 \\
        --PredictCTLearnModel.batch_size=16 \\
        --PredictCTLearnModel.dl1dh_reader_type=DLImageReader \\
        --DLImageReader.channels=cleaned_image \\
        --DLImageReader.channels=cleaned_relative_peak_time \\
        --DLImageReader.image_mapper_type=BilinearMapper \\
        --DLImageReader.mode=stereo \\
        --DLImageReader.min_telescopes=2 \\
        --PredictCTLearnModel.stack_telescope_images=True \\
        --type_model="/path/to/your/stereo/type/ctlearn_model.cpk" \\
        --energy_model="/path/to/your/stereo/energy/ctlearn_model.cpk" \\
        --skydirection_model="/path/to/your/stereo/skydirection/ctlearn_model.cpk" \\
        --output output.dl2.h5 \\
        --PredictCTLearnModel.overwrite_tables=True \\
    """

    def start(self):
        self.log.info("Processing the telescope pointings...")
        # Retrieve the IDs from the dl1dh for the prediction tables
        example_identifiers = self.dl1dh_reader.unique_example_identifiers.copy()
        example_identifiers.keep_columns(SUBARRAY_EVENT_KEYS)
        all_identifiers = self.dl1dh_reader.subarray_trigger_table.copy()
        all_identifiers.keep_columns(SUBARRAY_EVENT_KEYS + ["time"])
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
            survival_telescopes, name=f"{self.prefix}_telescopes"
        )
        # Pointing table for the stereo mode for MC simulation
        if self.dl1dh_reader.process_type == ProcessType.Simulation:
            pointing_info = self._store_mc_subarray_pointing(all_identifiers)

        # Pointing table for the observation mode
        if self.dl1dh_reader.process_type == ProcessType.Observation:
            pointing_info = super()._store_pointing(all_identifiers)

        self.log.info("Starting the prediction...")
        classification_feature_vectors = None
        if self.load_type_model_from is not None:
            # Predict the particle type of the primary particle
            classification_table, classification_feature_vectors = (
                super()._predict_classification(example_identifiers)
            )
            if self.dl2_subarray:
                # Produce output table with NaNs for missing predictions
                if len(nonexample_identifiers) > 0:
                    nan_table = super()._create_nan_table(
                        nonexample_identifiers,
                        columns=[f"{self.prefix}_tel_prediction"],
                        shapes=[(len(nonexample_identifiers),)],
                    )
                    classification_table = vstack([classification_table, nan_table])
                # Add is_valid column to the particle type table
                classification_table.add_column(
                    ~np.isnan(
                        classification_table[f"{self.prefix}_tel_prediction"].data,
                        dtype=bool,
                    ),
                    name=f"{self.prefix}_tel_is_valid",
                )
                # Rename the columns for the stereo mode
                classification_table.rename_column(
                    f"{self.prefix}_tel_prediction", f"{self.prefix}_prediction"
                )
                classification_table.rename_column(
                    f"{self.prefix}_tel_is_valid", f"{self.prefix}_is_valid"
                )
                classification_table.sort(SUBARRAY_EVENT_KEYS)
                # Add the default values and meta data to the table
                add_defaults_and_meta(
                    classification_table,
                    ParticleClassificationContainer,
                    prefix=self.prefix,
                )
                # Save the prediction to the output file
                write_table(
                    classification_table,
                    self.output_path,
                    f"{DL2_SUBARRAY_GROUP}/classification/{self.prefix}",
                    overwrite=self.overwrite_tables,
                )
                self.log.info(
                    "DL2 prediction data was stored in '%s' under '%s'",
                    self.output_path,
                    f"{DL2_SUBARRAY_GROUP}/classification/{self.prefix}",
                )
        energy_feature_vectors = None
        if self.load_energy_model_from is not None:
            # Predict the energy of the primary particle
            energy_table, energy_feature_vectors = super()._predict_energy(
                example_identifiers
            )
            if self.dl2_subarray:
                # Produce output table with NaNs for missing predictions
                if len(nonexample_identifiers) > 0:
                    nan_table = super()._create_nan_table(
                        nonexample_identifiers,
                        columns=[f"{self.prefix}_tel_energy"],
                        shapes=[(len(nonexample_identifiers),)],
                    )
                    energy_table = vstack([energy_table, nan_table])
                # Add is_valid column to the energy table
                energy_table.add_column(
                    ~np.isnan(
                        energy_table[f"{self.prefix}_tel_energy"].data, dtype=bool
                    ),
                    name=f"{self.prefix}_tel_is_valid",
                )
                # Rename the columns for the stereo mode
                energy_table.rename_column(
                    f"{self.prefix}_tel_energy", f"{self.prefix}_energy"
                )
                energy_table.rename_column(
                    f"{self.prefix}_tel_is_valid", f"{self.prefix}_is_valid"
                )
                energy_table.sort(SUBARRAY_EVENT_KEYS)
                # Add the default values and meta data to the table
                add_defaults_and_meta(
                    energy_table,
                    ReconstructedEnergyContainer,
                    prefix=self.prefix,
                )
                # Save the prediction to the output file
                write_table(
                    energy_table,
                    self.output_path,
                    f"{DL2_SUBARRAY_GROUP}/energy/{self.prefix}",
                    overwrite=self.overwrite_tables,
                )
                self.log.info(
                    "DL2 prediction data was stored in '%s' under '%s'",
                    self.output_path,
                    f"{DL2_SUBARRAY_GROUP}/energy/{self.prefix}",
                )
        impact_feature_vectors = None
        if self.load_impact_model_from is not None:
            # Predict the impact of the primary particle
            impact_table, impact_feature_vectors = super()._predict_impact(
                example_identifiers
            )
            if self.dl2_subarray:
                # Produce output table with NaNs for missing predictions
                if len(nonexample_identifiers) > 0:
                    nan_table = super()._create_nan_table(
                        nonexample_identifiers,
                        columns=[f"{self.prefix}_tel_impact"],
                        shapes=[(len(nonexample_identifiers),)],
                    )
                    impact_table = vstack([impact_table, nan_table])
                # Add is_valid column to the impact table
                impact_table.add_column(
                    ~np.isnan(
                        impact_table[f"{self.prefix}_tel_impact"].data, dtype=bool
                    ),
                    name=f"{self.prefix}_tel_is_valid",
                )
                # Rename the columns for the stereo mode
                impact_table.rename_column(
                    f"{self.prefix}_tel_impact", f"{self.prefix}_impact"
                )
                impact_table.rename_column(
                    f"{self.prefix}_tel_is_valid", f"{self.prefix}_is_valid"
                )
                impact_table.sort(SUBARRAY_EVENT_KEYS)
                # Save the prediction to the output file
                write_table(
                    impact_table,
                    self.output_path,
                    f"{DL2_SUBARRAY_GROUP}/impact/{self.prefix}",
                    overwrite=self.overwrite_tables,
                )
                self.log.info(
                    "DL2 prediction data was stored in '%s' under '%s'",
                    self.output_path,
                    f"{DL2_SUBARRAY_GROUP}/impact/{self.prefix}",
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
                # Transform the spherical coordinate offsets to sky coordinates
                direction_table = super()._transform_spher_coord_offsets_to_sky(
                    direction_table
                )
                # Produce output table with NaNs for missing predictions
                if len(nonexample_identifiers) > 0:
                    nan_table = super()._create_nan_table(
                        nonexample_identifiers,
                        columns=[f"{self.prefix}_alt", f"{self.prefix}_az"],
                        shapes=[
                            (len(nonexample_identifiers),),
                            (len(nonexample_identifiers),),
                        ],
                    )
                    direction_table = vstack([direction_table, nan_table])
                # Add is_valid column to the direction table
                direction_table.add_column(
                    ~np.isnan(direction_table[f"{self.prefix}_alt"].data, dtype=bool),
                    name=f"{self.prefix}_is_valid",
                )
                direction_table.sort(SUBARRAY_EVENT_KEYS)
                # Add the default values and meta data to the table
                add_defaults_and_meta(
                    direction_table,
                    ReconstructedGeometryContainer,
                    prefix=self.prefix,
                )
                # Save the prediction to the output file
                write_table(
                    direction_table,
                    self.output_path,
                    f"{DL2_SUBARRAY_GROUP}/geometry/{self.prefix}",
                    overwrite=self.overwrite_tables,
                )
                self.log.info(
                    "DL2 prediction data was stored in '%s' under '%s'",
                    self.output_path,
                    f"{DL2_SUBARRAY_GROUP}/geometry/{self.prefix}",
                )

        # Create the feature vector table if the DL1 features are enabled
        if self.dl1_features:
            self.log.info("Processing and storing dl1 feature vectors...")
            feature_vector_table = super()._create_feature_vectors_table(
                example_identifiers,
                nonexample_identifiers,
                classification_feature_vectors,
                energy_feature_vectors,
                direction_feature_vectors,
                impact_feature_vectors,
            )
            # Loop over the selected telescopes and store the feature vectors
            # for each telescope in the output file. The feature vectors are stored
            # in the DL1_TELESCOPE_GROUP/features/{prefix}/tel_{tel_id:03d} table.
            # Rename the columns for the stereo mode
            feature_vector_table.rename_column(
                f"{self.prefix}_tel_classification_feature_vectors",
                f"{self.prefix}_classification_feature_vectors",
            )
            feature_vector_table.rename_column(
                f"{self.prefix}_tel_energy_feature_vectors",
                f"{self.prefix}_energy_feature_vectors",
            )
            feature_vector_table.rename_column(
                f"{self.prefix}_tel_impact_feature_vectors",
                f"{self.prefix}_impact_feature_vectors",
            )
            feature_vector_table.rename_column(
                f"{self.prefix}_tel_geometry_feature_vectors",
                f"{self.prefix}_geometry_feature_vectors",
            )
            feature_vector_table.rename_column(
                f"{self.prefix}_tel_is_valid", f"{self.prefix}_is_valid"
            )
            feature_vector_table.sort(SUBARRAY_EVENT_KEYS)
            # Save the prediction to the output file
            write_table(
                feature_vector_table,
                self.output_path,
                f"{DL1_SUBARRAY_GROUP}/features/{self.prefix}",
                overwrite=self.overwrite_tables,
            )
            self.log.info(
                "DL1 feature vectors was stored in '%s' under '%s'",
                self.output_path,
                f"{DL1_SUBARRAY_GROUP}/features/{self.prefix}",
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
            f"{SIMULATION_CONFIG_TABLE}",
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
            f"{SUBARRAY_POINTING_GROUP}",
            overwrite=self.overwrite_tables,
        )
        self.log.info(
            "DL1 subarray pointing table was stored in '%s' under '%s'",
            self.output_path,
            f"{SUBARRAY_POINTING_GROUP}",
        )
        return pointing_info

def main():
    # Run the tool
    tool = StereoPredictCTLearnModel()
    tool.run()


if __name__ == "main":
    main()
