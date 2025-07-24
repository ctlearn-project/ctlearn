"""
Tools to predict the gammaness, energy and arrival direction in monoscopic and stereoscopic mode using ``CTLearnModel`` on R1/DL1 data using the ``DLDataReader`` and ``DLDataLoader``.
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

from ctapipe.core.traits import ComponentName
from ctapipe.io import write_table
from ctapipe.reco.reconstructor import ReconstructionProperty
from ctapipe.reco.stereo_combination import StereoCombiner
from ctapipe.reco.utils import add_defaults_and_meta
from dl1_data_handler.reader import ProcessType


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

__all__ = ["MonoPredictCTLearnModel"]

from ctlearn.tools.predict.utils.predict_model import PredictCTLearnModel

class MonoPredictCTLearnModel(PredictCTLearnModel):
    """
    Tool to predict the gammaness, energy and arrival direction from monoscopic R1/DL1 data using CTLearn models.

    This tool extends the ``PredictCTLearnModel`` to specifically handle monoscopic R1/DL1 data. The prediction
    is performed using the CTLearn models. The data is stored in the output file following the ctapipe DL2 data format.
    It also stores the telescope pointing monitoring and DL1 feature vectors (if selected) in the output file.

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
    _store_mc_telescope_pointing(all_identifiers)
        Store the telescope pointing table for the mono mode for MC simulation.
    """

    name = "ctlearn-predict-mono-model"
    description = __doc__

    examples = """
    To predict from pixel-wise image data in mono mode using trained CTLearn models:
    > ctlearn-predict-mono-model \\
        --input_url input.dl1.h5 \\
        --PredictCTLearnModel.batch_size=64 \\
        --PredictCTLearnModel.dl1dh_reader_type=DLImageReader \\
        --DLImageReader.channels=cleaned_image \\
        --DLImageReader.channels=cleaned_relative_peak_time \\
        --DLImageReader.image_mapper_type=BilinearMapper \\
        --type_model="/path/to/your/mono/type/ctlearn_model.cpk" \\
        --energy_model="/path/to/your/mono/energy/ctlearn_model.cpk" \\
        --cameradirection_model="/path/to/your/mono/cameradirection/ctlearn_model.cpk" \\
        --dl1-features \\
        --use-HDF5Merger \\
        --no-dl1-images \\
        --no-true-images \\
        --output output.dl2.h5 \\
        --PredictCTLearnModel.overwrite_tables=True \\

    To predict from pixel-wise waveform data in mono mode using trained CTLearn models:
    > ctlearn-predict-mono-model \\
        --input_url input.r1.h5 \\
        --PredictCTLearnModel.dl1dh_reader_type=DLWaveformReader \\
        --DLWaveformReader.sequnce_length=20 \\
        --DLWaveformReader.image_mapper_type=BilinearMapper \\
        --type_model="/path/to/your/mono_waveform/type/ctlearn_model.cpk" \\
        --energy_model="/path/to/your/mono_waveform/energy/ctlearn_model.cpk" \\
        --cameradirection_model="/path/to/your/mono_waveform/cameradirection/ctlearn_model.cpk" \\
        --use-HDF5Merger \\
        --no-r0-waveforms \\
        --no-r1-waveforms \\
        --no-dl1-images \\
        --no-true-images \\
        --output output.dl2.h5 \\
        --PredictCTLearnModel.overwrite_tables=True \\
    """

    stereo_combiner_cls = ComponentName(
        StereoCombiner,
        default_value="StereoMeanCombiner",
        help="Which stereo combination method to use after the monoscopic reconstruction.",
    ).tag(config=True)

    def start(self):
        self.log.info("Processing the telescope pointings...")
        # Retrieve the IDs from the dl1dh for the prediction tables
        example_identifiers = self.dl1dh_reader.example_identifiers.copy()
        example_identifiers.keep_columns(TELESCOPE_EVENT_KEYS)
        all_identifiers = self.dl1dh_reader.tel_trigger_table.copy()
        all_identifiers.keep_columns(TELESCOPE_EVENT_KEYS + ["time"])
        nonexample_identifiers = setdiff(
            all_identifiers, example_identifiers, keys=TELESCOPE_EVENT_KEYS
        )
        nonexample_identifiers.remove_column("time")
        # Pointing table for the mono mode for MC simulation
        if self.dl1dh_reader.process_type == ProcessType.Simulation:
            pointing_info = self._store_mc_telescope_pointing(all_identifiers)

        # Pointing table for the observation mode
        if self.dl1dh_reader.process_type == ProcessType.Observation:
            pointing_info = super()._store_pointing(all_identifiers)

        self.log.info("Starting the prediction...")
        classification_feature_vectors = None
        if self.load_type_model_from is not None:
            self.type_stereo_combiner = StereoCombiner.from_name(
                self.stereo_combiner_cls,
                prefix=self.prefix,
                property=ReconstructionProperty.PARTICLE_TYPE,
                parent=self,
            )
            # Predict the energy of the primary particle
            classification_table, classification_feature_vectors = (
                super()._predict_classification(example_identifiers)
            )
            if self.dl2_telescope:
                # Produce output table with NaNs for missing predictions
                if len(nonexample_identifiers) > 0:
                    nan_table = super()._create_nan_table(
                        nonexample_identifiers,
                        columns=[f"{self.prefix}_tel_prediction"],
                        shapes=[(len(nonexample_identifiers),)],
                    )
                    classification_table = vstack([classification_table, nan_table])
                # Add is_valid column to the energy table
                classification_table.add_column(
                    ~np.isnan(
                        classification_table[f"{self.prefix}_tel_prediction"].data,
                        dtype=bool,
                    ),
                    name=f"{self.prefix}_tel_is_valid",
                )
                # Add the default values and meta data to the table
                add_defaults_and_meta(
                    classification_table,
                    ParticleClassificationContainer,
                    prefix=self.prefix,
                    add_tel_prefix=True,
                )
                for tel_id in self.dl1dh_reader.selected_telescopes[
                    self.dl1dh_reader.tel_type
                ]:
                    # Retrieve the example identifiers for the selected telescope
                    telescope_mask = classification_table["tel_id"] == tel_id
                    classification_tel_table = classification_table[telescope_mask]
                    classification_tel_table.sort(TELESCOPE_EVENT_KEYS)
                    # Save the prediction to the output file for the selected telescope
                    write_table(
                        classification_tel_table,
                        self.output_path,
                        f"{DL2_TELESCOPE_GROUP}/classification/{self.prefix}/tel_{tel_id:03d}",
                        overwrite=self.overwrite_tables,
                    )
                    self.log.info(
                        "DL2 prediction data was stored in '%s' under '%s'",
                        self.output_path,
                        f"{DL2_TELESCOPE_GROUP}/classification/{self.prefix}/tel_{tel_id:03d}",
                    )
            if self.dl2_subarray:
                self.log.info("Processing and storing the subarray type prediction...")
                # Combine the telescope predictions to the subarray prediction using the stereo combiner
                subarray_classification_table = self.type_stereo_combiner.predict_table(
                    classification_table
                )
                # TODO: Remove temporary fix once the stereo combiner returns correct table
                # Check if the table has to be converted to a boolean mask
                if (
                    subarray_classification_table[f"{self.prefix}_telescopes"].dtype
                    != np.bool_
                ):
                    # Create boolean mask for telescopes that participate in the stereo reconstruction combination
                    reco_telescopes = np.zeros(
                        (
                            len(subarray_classification_table),
                            len(self.dl1dh_reader.tel_ids),
                        ),
                        dtype=bool,
                    )
                    # Loop over the table and set the boolean mask for the telescopes
                    for index, tel_id_mask in enumerate(
                        subarray_classification_table[f"{self.prefix}_telescopes"]
                    ):
                        if not tel_id_mask:
                            continue
                        for tel_id in tel_id_mask:
                            reco_telescopes[index][
                                self.dl1dh_reader.subarray.tel_ids_to_indices(tel_id)
                            ] = True
                    # Overwrite the column with the boolean mask with fix length
                    subarray_classification_table[f"{self.prefix}_telescopes"] = (
                        reco_telescopes
                    )
                # Save the prediction to the output file
                write_table(
                    subarray_classification_table,
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
            self.energy_stereo_combiner = StereoCombiner.from_name(
                self.stereo_combiner_cls,
                prefix=self.prefix,
                property=ReconstructionProperty.ENERGY,
                parent=self,
            )
            # Predict the energy of the primary particle
            energy_table, energy_feature_vectors = super()._predict_energy(
                example_identifiers
            )
            if self.dl2_telescope:
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
                # Add the default values and meta data to the table
                add_defaults_and_meta(
                    energy_table,
                    ReconstructedEnergyContainer,
                    prefix=self.prefix,
                    add_tel_prefix=True,
                )
                for tel_id in self.dl1dh_reader.selected_telescopes[
                    self.dl1dh_reader.tel_type
                ]:
                    # Retrieve the example identifiers for the selected telescope
                    telescope_mask = energy_table["tel_id"] == tel_id
                    energy_tel_table = energy_table[telescope_mask]
                    energy_tel_table.sort(TELESCOPE_EVENT_KEYS)
                    # Save the prediction to the output file
                    write_table(
                        energy_tel_table,
                        self.output_path,
                        f"{DL2_TELESCOPE_GROUP}/energy/{self.prefix}/tel_{tel_id:03d}",
                        overwrite=self.overwrite_tables,
                    )
                    self.log.info(
                        "DL2 prediction data was stored in '%s' under '%s'",
                        self.output_path,
                        f"{DL2_TELESCOPE_GROUP}/energy/{self.prefix}/tel_{tel_id:03d}",
                    )
            if self.dl2_subarray:
                self.log.info(
                    "Processing and storing the subarray energy prediction..."
                )
                # Combine the telescope predictions to the subarray prediction using the stereo combiner
                subarray_energy_table = self.energy_stereo_combiner.predict_table(
                    energy_table
                )
                # TODO: Remove temporary fix once the stereo combiner returns correct table
                # Check if the table has to be converted to a boolean mask
                if subarray_energy_table[f"{self.prefix}_telescopes"].dtype != np.bool_:
                    # Create boolean mask for telescopes that participate in the stereo reconstruction combination
                    reco_telescopes = np.zeros(
                        (len(subarray_energy_table), len(self.dl1dh_reader.tel_ids)),
                        dtype=bool,
                    )
                    # Loop over the table and set the boolean mask for the telescopes
                    for index, tel_id_mask in enumerate(
                        subarray_energy_table[f"{self.prefix}_telescopes"]
                    ):
                        if not tel_id_mask:
                            continue
                        for tel_id in tel_id_mask:
                            reco_telescopes[index][
                                self.dl1dh_reader.subarray.tel_ids_to_indices(tel_id)
                            ] = True
                    # Overwrite the column with the boolean mask with fix length
                    subarray_energy_table[f"{self.prefix}_telescopes"] = reco_telescopes
                # Save the prediction to the output file
                write_table(
                    subarray_energy_table,
                    self.output_path,
                    f"{DL2_SUBARRAY_GROUP}/energy/{self.prefix}",
                    overwrite=self.overwrite_tables,
                )
                self.log.info(
                    "DL2 prediction data was stored in '%s' under '%s'",
                    self.output_path,
                    f"{DL2_SUBARRAY_GROUP}/energy/{self.prefix}",
                )
        direction_feature_vectors = None
        if self.load_cameradirection_model_from is not None:
            self.geometry_stereo_combiner = StereoCombiner.from_name(
                self.stereo_combiner_cls,
                prefix=self.prefix,
                property=ReconstructionProperty.GEOMETRY,
                parent=self,
            )
            # Join the prediction table with the telescope pointing table
            example_identifiers = join(
                left=example_identifiers,
                right=pointing_info,
                keys=TELESCOPE_EVENT_KEYS,
            )
            # Predict the arrival direction of the primary particle
            direction_table, direction_feature_vectors = (
                super()._predict_cameradirection(example_identifiers)
            )
            direction_tel_tables = []
            if self.dl2_telescope:
                for tel_id in self.dl1dh_reader.selected_telescopes[
                    self.dl1dh_reader.tel_type
                ]:
                    # Retrieve the example identifiers for the selected telescope
                    telescope_mask = direction_table["tel_id"] == tel_id
                    direction_tel_table = direction_table[telescope_mask]
                    direction_tel_table = super()._transform_cam_coord_offsets_to_sky(
                        direction_tel_table
                    )
                    # Produce output table with NaNs for missing predictions
                    nan_telescope_mask = nonexample_identifiers["tel_id"] == tel_id
                    nonexample_identifiers_tel = nonexample_identifiers[
                        nan_telescope_mask
                    ]
                    if len(nonexample_identifiers_tel) > 0:
                        nan_table = super()._create_nan_table(
                            nonexample_identifiers_tel,
                            columns=[f"{self.prefix}_tel_alt", f"{self.prefix}_tel_az"],
                            shapes=[
                                (len(nonexample_identifiers_tel),),
                                (len(nonexample_identifiers_tel),),
                            ],
                        )
                        direction_tel_table = vstack([direction_tel_table, nan_table])
                    direction_tel_table.sort(TELESCOPE_EVENT_KEYS)
                    # Add is_valid column to the direction table
                    direction_tel_table.add_column(
                        ~np.isnan(
                            direction_tel_table[f"{self.prefix}_tel_alt"].data,
                            dtype=bool,
                        ),
                        name=f"{self.prefix}_tel_is_valid",
                    )
                    # Add the default values and meta data to the table
                    add_defaults_and_meta(
                        direction_tel_table,
                        ReconstructedGeometryContainer,
                        prefix=self.prefix,
                        add_tel_prefix=True,
                    )
                    direction_tel_tables.append(direction_tel_table)
                    # Save the prediction to the output file
                    write_table(
                        direction_tel_table,
                        self.output_path,
                        f"{DL2_TELESCOPE_GROUP}/geometry/{self.prefix}/tel_{tel_id:03d}",
                        overwrite=self.overwrite_tables,
                    )
                    self.log.info(
                        "DL2 prediction data was stored in '%s' under '%s'",
                        self.output_path,
                        f"{DL2_TELESCOPE_GROUP}/geometry/{self.prefix}/tel_{tel_id:03d}",
                    )
            if self.dl2_subarray:
                self.log.info(
                    "Processing and storing the subarray geometry prediction..."
                )
                # Stack the telescope tables to the subarray table
                direction_tel_tables = vstack(direction_tel_tables)
                # Sort the table by the telescope event keys
                direction_tel_tables.sort(TELESCOPE_EVENT_KEYS)
                # Combine the telescope predictions to the subarray prediction using the stereo combiner
                subarray_direction_table = self.geometry_stereo_combiner.predict_table(
                    direction_tel_tables
                )
                # TODO: Remove temporary fix once the stereo combiner returns correct table
                # Check if the table has to be converted to a boolean mask
                if (
                    subarray_direction_table[f"{self.prefix}_telescopes"].dtype
                    != np.bool_
                ):
                    # Create boolean mask for telescopes that participate in the stereo reconstruction combination
                    reco_telescopes = np.zeros(
                        (len(subarray_direction_table), len(self.dl1dh_reader.tel_ids)),
                        dtype=bool,
                    )
                    # Loop over the table and set the boolean mask for the telescopes
                    for index, tel_id_mask in enumerate(
                        subarray_direction_table[f"{self.prefix}_telescopes"]
                    ):
                        if not tel_id_mask:
                            continue
                        for tel_id in tel_id_mask:
                            reco_telescopes[index][
                                self.dl1dh_reader.subarray.tel_ids_to_indices(tel_id)
                            ] = True
                    # Overwrite the column with the boolean mask with fix length
                    subarray_direction_table[f"{self.prefix}_telescopes"] = (
                        reco_telescopes
                    )
                # Save the prediction to the output file
                write_table(
                    subarray_direction_table,
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
            )
            # Loop over the selected telescopes and store the feature vectors
            # for each telescope in the output file. The feature vectors are stored
            # in the DL1_TELESCOPE_GROUP/features/{prefix}/tel_{tel_id:03d} table.
            for tel_id in self.dl1dh_reader.selected_telescopes[
                self.dl1dh_reader.tel_type
            ]:
                # Retrieve the example identifiers for the selected telescope
                telescope_mask = feature_vector_table["tel_id"] == tel_id
                feature_vectors_tel_table = feature_vector_table[telescope_mask]
                feature_vectors_tel_table.sort(TELESCOPE_EVENT_KEYS)
                # Save the prediction to the output file
                write_table(
                    feature_vectors_tel_table,
                    self.output_path,
                    f"{DL1_TELESCOPE_GROUP}/features/{self.prefix}/tel_{tel_id:03d}",
                    overwrite=self.overwrite_tables,
                )
                self.log.info(
                    "DL1 feature vectors was stored in '%s' under '%s'",
                    self.output_path,
                    f"{DL1_TELESCOPE_GROUP}/features/{self.prefix}/tel_{tel_id:03d}",
                )

    def _store_mc_telescope_pointing(self, all_identifiers):
        """
        Store the telescope pointing table from MC simulation to the output file.

        Parameters:
        -----------
        all_identifiers : astropy.table.Table
            Table containing the telescope pointing information.
        """
        # Create the pointing table for each telescope
        pointing_info = []
        for tel_id in self.dl1dh_reader.selected_telescopes[self.dl1dh_reader.tel_type]:
            # Pointing table for the mono mode
            tel_pointing = self.dl1dh_reader.get_tel_pointing(self.input_url, tel_id)
            tel_pointing.rename_column("telescope_pointing_azimuth", "pointing_azimuth")
            tel_pointing.rename_column(
                "telescope_pointing_altitude", "pointing_altitude"
            )
            # Join the prediction table with the telescope pointing table
            tel_pointing = join(
                left=tel_pointing,
                right=all_identifiers,
                keys=["obs_id", "tel_id"],
            )
            # TODO: use keep_order for astropy v7.0.0
            tel_pointing.sort(TELESCOPE_EVENT_KEYS)
            # Retrieve the example identifiers for the selected telescope
            tel_pointing_table = Table(
                {
                    "time": tel_pointing["time"],
                    "azimuth": tel_pointing["pointing_azimuth"],
                    "altitude": tel_pointing["pointing_altitude"],
                }
            )
            write_table(
                tel_pointing_table,
                self.output_path,
                f"{POINTING_GROUP}/tel_{tel_id:03d}",
                overwrite=self.overwrite_tables,
            )
            self.log.info(
                "DL1 telescope pointing table was stored in '%s' under '%s'",
                self.output_path,
                f"{POINTING_GROUP}/tel_{tel_id:03d}",
            )
            pointing_info.append(tel_pointing)
        pointing_info = vstack(pointing_info)
        return pointing_info

def mono_tool():
    # Run the tool
    mono_tool = MonoPredictCTLearnModel()
    mono_tool.run()
    
if __name__ == "main":
    mono_tool()
    
if __name__ == "__main__":
    mono_tool()