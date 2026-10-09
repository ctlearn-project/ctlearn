"""
Tool to predict the gammaness, energy and arrival direction in monoscopic mode using ``CTLearnModel`` on R1/DL1 data using the ``DLDataReader`` and ``KerasSequence``/``PyTorchDataset``.
"""

__all__ = [
    "MonoPredictCTLearnModel",
] 

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
from ctapipe.io import read_table, write_table
from ctapipe.io.hdf5dataformat import (
    DL1_TEL_GROUP,
    DL1_TEL_POINTING_GROUP,
    DL1_TEL_TRIGGER_TABLE,
    DL2_EVENT_STATISTICS_GROUP,
    DL2_TEL_PARTICLETYPE_GROUP,
    DL2_TEL_ENERGY_GROUP,
    DL2_TEL_GEOMETRY_GROUP,
    DL2_SUBARRAY_PARTICLETYPE_GROUP,
    DL2_SUBARRAY_ENERGY_GROUP,
    DL2_SUBARRAY_GEOMETRY_GROUP,
)
from ctapipe.reco.reconstructor import ReconstructionProperty
from ctapipe.reco.stereo_combination import StereoCombiner
from ctapipe.reco.utils import add_defaults_and_meta
from ctlearn.tools.predict_model import PredictCTLearnModel
from dl1_data_handler.reader import ProcessType


# Convienient constants for column names and table keys
SUBARRAY_EVENT_KEYS = ["obs_id", "event_id"]
TEL_EVENT_KEYS = ["obs_id", "event_id", "tel_id"]


class MonoPredictCTLearnModel(PredictCTLearnModel):
    """
    Tool to predict the gammaness, energy and arrival direction from monoscopic R1/DL1 data using CTLearn models.

    This tool extends the ``PredictCTLearnModel`` to specifically handle monoscopic R1/DL1 data. The prediction
    is performed using the CTLearn models. The data is stored in the output file following the ctapipe DL2 data format.
    It also stores the telescope pointing monitoring and DL1 feature vectors (if selected) in the output file.
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
        --type_model="/path/to/your/mono/type/ctlearn_model(.keras/.pth)" \\
        --energy_model="/path/to/your/mono/energy/ctlearn_model(.keras/.pth)" \\
        --cameradirection_model="/path/to/your/mono/cameradirection/ctlearn_model(.keras/.pth)" \\
        --dl1-features \\
        --no-dl1-images \\
        --no-true-images \\
        --output output.dl2.h5 \\

    To predict from pixel-wise waveform data in mono mode using trained CTLearn models:
    > ctlearn-predict-mono-model \\
        --input_url input.r1.h5 \\
        --PredictCTLearnModel.dl1dh_reader_type=DLWaveformReader \\
        --DLWaveformReader.sequnce_length=20 \\
        --DLWaveformReader.image_mapper_type=BilinearMapper \\
        --type_model="/path/to/your/mono_waveform/type/ctlearn_model(.keras/.pth)" \\
        --energy_model="/path/to/your/mono_waveform/energy/ctlearn_model(.keras/.pth)" \\
        --cameradirection_model="/path/to/your/mono_waveform/cameradirection/ctlearn_model(.keras/.pth)" \\
        --no-r0-waveforms \\
        --no-r1-waveforms \\
        --no-dl1-images \\
        --no-true-images \\
        --output output.dl2.h5 \\
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
        example_identifiers.keep_columns(TEL_EVENT_KEYS)
        all_identifiers = read_table(
            self.output_path,
            DL1_TEL_TRIGGER_TABLE,
        )
        all_identifiers.keep_columns(TEL_EVENT_KEYS + ["time"])
        nonexample_identifiers = setdiff(
            all_identifiers, example_identifiers, keys=TEL_EVENT_KEYS
        )
        nonexample_identifiers.remove_column("time")
        # Pointing table for the mono mode for MC simulation
        if self.dl1dh_reader.process_type == ProcessType.Simulation:
            pointing_info = self._store_mc_telescope_pointing(all_identifiers)

        # Pointing table for the observation mode
        if self.dl1dh_reader.process_type == ProcessType.Observation:
            pointing_info = super()._store_pointing(all_identifiers)

        self.log.info("Starting the prediction...")
        particletype_feature_vectors = None
        if self.load_type_model_from is not None:
            # Predict the type of the primary particle
            particletype_table, particletype_feature_vectors = (
                super()._predict_particletype(example_identifiers)
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
            # Add is_valid column to the particle type table
            particletype_table.add_column(
                ~np.isnan(
                    particletype_table[f"{self.prefixes['type']}_tel_prediction"].data,
                    dtype=bool,
                ),
                name=f"{self.prefixes['type']}_tel_is_valid",
            )
            # Add the default values and meta data to the table
            add_defaults_and_meta(
                particletype_table,
                ParticleClassificationContainer,
                prefix=self.prefixes["type"],
                add_tel_prefix=True,
            )
            if self.dl2_telescope:
                for tel_id in self.dl1dh_reader.selected_telescopes[
                    self.dl1dh_reader.tel_type
                ]:
                    # Retrieve the example identifiers for the selected telescope
                    telescope_mask = particletype_table["tel_id"] == tel_id
                    particletype_tel_table = particletype_table[telescope_mask]
                    particletype_tel_table.sort(TEL_EVENT_KEYS)
                    # Save the prediction to the output file for the selected telescope
                    write_table(
                        particletype_tel_table,
                        self.output_path,
                        f"{DL2_TEL_PARTICLETYPE_GROUP}/{self.prefixes['type']}/tel_{tel_id:03d}",
                    )
                    self.log.info(
                        "DL2 prediction data was stored in '%s' under '%s'",
                        self.output_path,
                        f"{DL2_TEL_PARTICLETYPE_GROUP}/{self.prefixes['type']}/tel_{tel_id:03d}",
                    )

            if self.dl2_subarray:
                self.log.info("Processing and storing the subarray type prediction...")
                # If only one telescope is used, copy the particletype table
                # and modify it to subarray format
                if len(self.dl1dh_reader.tel_ids) == 1:
                    particletype_subarray_table = particletype_table.copy()
                    telescope_mask = (
                        particletype_subarray_table["tel_id"]
                        == self.dl1dh_reader.tel_ids[0]
                    )
                    particletype_subarray_table = particletype_subarray_table[
                        telescope_mask
                    ]
                    particletype_subarray_table.remove_column("tel_id")
                    for colname in particletype_subarray_table.colnames:
                        if "_tel_" in colname:
                            particletype_subarray_table.rename_column(
                                colname, colname.replace("_tel", "")
                            )
                    particletype_subarray_table.add_column(
                        [
                            [val]
                            for val in particletype_subarray_table[
                                f"{self.prefixes['type']}_is_valid"
                            ]
                        ],
                        name=f"{self.prefixes['type']}_telescopes",
                    )
                else:
                    self.type_stereo_combiner = StereoCombiner.from_name(
                        self.stereo_combiner_cls,
                        prefix=self.prefixes["type"],
                        property=ReconstructionProperty.PARTICLE_TYPE,
                        parent=self,
                    )
                    # Combine the telescope predictions to the subarray prediction using the stereo combiner
                    particletype_subarray_table = (
                        self.type_stereo_combiner.predict_table(particletype_table)
                    )
                    # TODO: Remove temporary fix once the stereo combiner returns correct table
                    # Check if the table has to be converted to a boolean mask
                    if (
                        particletype_subarray_table[
                            f"{self.prefixes['type']}_telescopes"
                        ].dtype
                        != np.bool_
                    ):
                        # Create boolean mask for telescopes that participate in the stereo reconstruction combination
                        reco_telescopes = np.zeros(
                            (
                                len(particletype_subarray_table),
                                len(self.dl1dh_reader.tel_ids),
                            ),
                            dtype=bool,
                        )
                        # Loop over the table and set the boolean mask for the telescopes
                        for index, tel_id_mask in enumerate(
                            particletype_subarray_table[
                                f"{self.prefixes['type']}_telescopes"
                            ]
                        ):
                            if not tel_id_mask:
                                continue
                            for tel_id in tel_id_mask:
                                reco_telescopes[index][
                                    self.dl1dh_reader.subarray.tel_ids_to_indices(
                                        tel_id
                                    )
                                ] = True
                        # Overwrite the column with the boolean mask with fix length
                        particletype_subarray_table[
                            f"{self.prefixes['type']}_telescopes"
                        ] = reco_telescopes
                # Deduplicate the subarray particletype table to have only one entry per event
                particletype_subarray_table = super().deduplicate_first_valid(
                    table=particletype_subarray_table,
                    keys=SUBARRAY_EVENT_KEYS,
                    valid_col=f"{self.prefixes['type']}_is_valid",
                )
                # Sort the subarray particletype table
                particletype_subarray_table.sort(SUBARRAY_EVENT_KEYS)
                # Save the prediction to the output file
                write_table(
                    particletype_subarray_table,
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
                name=f"{self.prefixes['energy']}_tel_is_valid",
            )
            # Add the default values and meta data to the table
            add_defaults_and_meta(
                energy_table,
                ReconstructedEnergyContainer,
                prefix=self.prefixes["energy"],
                add_tel_prefix=True,
            )
            if self.dl2_telescope:
                for tel_id in self.dl1dh_reader.selected_telescopes[
                    self.dl1dh_reader.tel_type
                ]:
                    # Retrieve the example identifiers for the selected telescope
                    telescope_mask = energy_table["tel_id"] == tel_id
                    energy_tel_table = energy_table[telescope_mask]
                    energy_tel_table.sort(TEL_EVENT_KEYS)
                    # Save the prediction to the output file
                    write_table(
                        energy_tel_table,
                        self.output_path,
                        f"{DL2_TEL_ENERGY_GROUP}/{self.prefixes['energy']}/tel_{tel_id:03d}",
                    )
                    self.log.info(
                        "DL2 prediction data was stored in '%s' under '%s'",
                        self.output_path,
                        f"{DL2_TEL_ENERGY_GROUP}/{self.prefixes['energy']}/tel_{tel_id:03d}",
                    )
            if self.dl2_subarray:
                self.log.info(
                    "Processing and storing the subarray energy prediction..."
                )
                # If only one telescope is used, copy the particletype table
                # and modify it to subarray format
                if len(self.dl1dh_reader.tel_ids) == 1:
                    energy_subarray_table = energy_table.copy()
                    telescope_mask = (
                        energy_subarray_table["tel_id"] == self.dl1dh_reader.tel_ids[0]
                    )
                    energy_subarray_table = energy_subarray_table[telescope_mask]
                    energy_subarray_table.remove_column("tel_id")
                    for colname in energy_subarray_table.colnames:
                        if "_tel_" in colname:
                            energy_subarray_table.rename_column(
                                colname, colname.replace("_tel", "")
                            )
                    energy_subarray_table.add_column(
                        [
                            [val]
                            for val in energy_subarray_table[
                                f"{self.prefixes['energy']}_is_valid"
                            ]
                        ],
                        name=f"{self.prefixes['energy']}_telescopes",
                    )
                else:
                    self.energy_stereo_combiner = StereoCombiner.from_name(
                        self.stereo_combiner_cls,
                        prefix=self.prefixes["energy"],
                        property=ReconstructionProperty.ENERGY,
                        parent=self,
                    )
                    # Combine the telescope predictions to the subarray prediction using the stereo combiner
                    energy_subarray_table = self.energy_stereo_combiner.predict_table(
                        energy_table
                    )
                    # TODO: Remove temporary fix once the stereo combiner returns correct table
                    # Check if the table has to be converted to a boolean mask
                    if (
                        energy_subarray_table[
                            f"{self.prefixes['energy']}_telescopes"
                        ].dtype
                        != np.bool_
                    ):
                        # Create boolean mask for telescopes that participate in the stereo reconstruction combination
                        reco_telescopes = np.zeros(
                            (
                                len(energy_subarray_table),
                                len(self.dl1dh_reader.tel_ids),
                            ),
                            dtype=bool,
                        )
                        # Loop over the table and set the boolean mask for the telescopes
                        for index, tel_id_mask in enumerate(
                            energy_subarray_table[
                                f"{self.prefixes['energy']}_telescopes"
                            ]
                        ):
                            if not tel_id_mask:
                                continue
                            for tel_id in tel_id_mask:
                                reco_telescopes[index][
                                    self.dl1dh_reader.subarray.tel_ids_to_indices(
                                        tel_id
                                    )
                                ] = True
                        # Overwrite the column with the boolean mask with fix length
                        energy_subarray_table[
                            f"{self.prefixes['energy']}_telescopes"
                        ] = reco_telescopes
                # Deduplicate the subarray energy table to have only one entry per event
                energy_subarray_table = super().deduplicate_first_valid(
                    table=energy_subarray_table,
                    keys=SUBARRAY_EVENT_KEYS,
                    valid_col=f"{self.prefixes['energy']}_is_valid",
                )
                # Sort the subarray energy table
                energy_subarray_table.sort(SUBARRAY_EVENT_KEYS)
                # Save the prediction to the output file
                write_table(
                    energy_subarray_table,
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
        if self.load_cameradirection_model_from is not None:
            # Join the prediction table with the telescope pointing table
            example_identifiers = join(
                left=example_identifiers,
                right=pointing_info,
                keys=TEL_EVENT_KEYS,
            )
            # Predict the arrival direction of the primary particle
            direction_table, direction_feature_vectors = (
                super()._predict_cameradirection(example_identifiers)
            )
            direction_tel_tables = []
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
                nonexample_identifiers_tel = nonexample_identifiers[nan_telescope_mask]
                if len(nonexample_identifiers_tel) > 0:
                    nan_table = super()._create_nan_table(
                        nonexample_identifiers_tel,
                        columns=[
                            f"{self.prefixes['cameradirection']}_tel_alt",
                            f"{self.prefixes['cameradirection']}_tel_az",
                        ],
                        shapes=[
                            (len(nonexample_identifiers_tel),),
                            (len(nonexample_identifiers_tel),),
                        ],
                        reco_task="cameradirection",
                    )
                    direction_tel_table = vstack([direction_tel_table, nan_table])
                direction_tel_table.sort(TEL_EVENT_KEYS)
                # Add is_valid column to the direction table
                direction_tel_table.add_column(
                    ~np.isnan(
                        direction_tel_table[
                            f"{self.prefixes['cameradirection']}_tel_alt"
                        ].data,
                        dtype=bool,
                    ),
                    name=f"{self.prefixes['cameradirection']}_tel_is_valid",
                )
                # Add the default values and meta data to the table
                add_defaults_and_meta(
                    direction_tel_table,
                    ReconstructedGeometryContainer,
                    prefix=self.prefixes["cameradirection"],
                    add_tel_prefix=True,
                )
                direction_tel_tables.append(direction_tel_table)
                if self.dl2_telescope:
                    # Save the prediction to the output file
                    write_table(
                        direction_tel_table,
                        self.output_path,
                        f"{DL2_TEL_GEOMETRY_GROUP}/{self.prefixes['cameradirection']}/tel_{tel_id:03d}",
                    )
                    self.log.info(
                        "DL2 prediction data was stored in '%s' under '%s'",
                        self.output_path,
                        f"{DL2_TEL_GEOMETRY_GROUP}/{self.prefixes['cameradirection']}/tel_{tel_id:03d}",
                    )
            if self.dl2_subarray:
                self.log.info(
                    "Processing and storing the subarray geometry prediction..."
                )
                # Stack the telescope tables to the subarray table
                direction_tel_tables = vstack(direction_tel_tables)
                # Sort the table by the telescope event keys
                direction_tel_tables.sort(TEL_EVENT_KEYS)
                # If only one telescope is used, copy the classification table
                # and modify it to subarray format
                if len(self.dl1dh_reader.tel_ids) == 1:
                    direction_subarray_table = direction_tel_tables.copy()
                    telescope_mask = (
                        direction_subarray_table["tel_id"]
                        == self.dl1dh_reader.tel_ids[0]
                    )
                    direction_subarray_table = direction_subarray_table[telescope_mask]
                    direction_subarray_table.remove_column("tel_id")
                    for colname in direction_subarray_table.colnames:
                        if "_tel_" in colname:
                            direction_subarray_table.rename_column(
                                colname, colname.replace("_tel", "")
                            )
                    direction_subarray_table.add_column(
                        [
                            [val]
                            for val in direction_subarray_table[
                                f"{self.prefixes['cameradirection']}_is_valid"
                            ]
                        ],
                        name=f"{self.prefixes['cameradirection']}_telescopes",
                    )
                else:
                    self.geometry_stereo_combiner = StereoCombiner.from_name(
                        self.stereo_combiner_cls,
                        prefix=self.prefixes["cameradirection"],
                        property=ReconstructionProperty.GEOMETRY,
                        parent=self,
                    )
                    # Combine the telescope predictions to the subarray prediction using the stereo combiner
                    direction_subarray_table = (
                        self.geometry_stereo_combiner.predict_table(
                            direction_tel_tables
                        )
                    )
                    # TODO: Remove temporary fix once the stereo combiner returns correct table
                    # Check if the table has to be converted to a boolean mask
                    if (
                        direction_subarray_table[
                            f"{self.prefixes['cameradirection']}_telescopes"
                        ].dtype
                        != np.bool_
                    ):
                        # Create boolean mask for telescopes that participate in the stereo reconstruction combination
                        reco_telescopes = np.zeros(
                            (
                                len(direction_subarray_table),
                                len(self.dl1dh_reader.tel_ids),
                            ),
                            dtype=bool,
                        )
                        # Loop over the table and set the boolean mask for the telescopes
                        for index, tel_id_mask in enumerate(
                            direction_subarray_table[
                                f"{self.prefixes['cameradirection']}_telescopes"
                            ]
                        ):
                            if not tel_id_mask:
                                continue
                            for tel_id in tel_id_mask:
                                reco_telescopes[index][
                                    self.dl1dh_reader.subarray.tel_ids_to_indices(
                                        tel_id
                                    )
                                ] = True
                        # Overwrite the column with the boolean mask with fix length
                        direction_subarray_table[
                            f"{self.prefixes['cameradirection']}_telescopes"
                        ] = reco_telescopes
                # Deduplicate the subarray direction table to have only one entry per event
                direction_subarray_table = super().deduplicate_first_valid(
                    table=direction_subarray_table,
                    keys=SUBARRAY_EVENT_KEYS,
                    valid_col=f"{self.prefixes['cameradirection']}_is_valid",
                )
                # Sort the subarray geometry table
                direction_subarray_table.sort(SUBARRAY_EVENT_KEYS)
                # Save the prediction to the output file
                write_table(
                    direction_subarray_table,
                    self.output_path,
                    f"{DL2_SUBARRAY_GEOMETRY_GROUP}/{self.prefixes['cameradirection']}",
                )
                self.log.info(
                    "DL2 prediction data was stored in '%s' under '%s'",
                    self.output_path,
                    f"{DL2_SUBARRAY_GEOMETRY_GROUP}/{self.prefixes['cameradirection']}",
                )
            # Store the telescope event statistics table
            write_table(
                self.dl1dh_reader.quality_query.to_table(functions=True),
                self.output_path,
                f"{DL2_EVENT_STATISTICS_GROUP}/{self.prefixes['cameradirection']}",
                append=True,
            )
            self.log.info(
                "DL2 service telescope event statistics data was stored in '%s' under '%s'",
                self.output_path,
                f"{DL2_EVENT_STATISTICS_GROUP}/{self.prefixes['cameradirection']}",
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
            # in the DL1_TEL_GROUP/features/{prefix}/tel_{tel_id:03d} table.
            for tel_id in self.dl1dh_reader.selected_telescopes[
                self.dl1dh_reader.tel_type
            ]:
                # Retrieve the example identifiers for the selected telescope
                telescope_mask = feature_vector_table["tel_id"] == tel_id
                feature_vectors_tel_table = feature_vector_table[telescope_mask]
                feature_vectors_tel_table.sort(TEL_EVENT_KEYS)
                # Save the prediction to the output file
                write_table(
                    feature_vectors_tel_table,
                    self.output_path,
                    f"{DL1_TEL_GROUP}/features/{self.prefixes['all']}/tel_{tel_id:03d}",
                )
                self.log.info(
                    "DL1 feature vectors was stored in '%s' under '%s'",
                    self.output_path,
                    f"{DL1_TEL_GROUP}/features/{self.prefixes['all']}/tel_{tel_id:03d}",
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
            tel_pointing.sort(TEL_EVENT_KEYS)
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
                f"{DL1_TEL_POINTING_GROUP}/tel_{tel_id:03d}",
            )
            self.log.info(
                "DL1 telescope pointing table was stored in '%s' under '%s'",
                self.output_path,
                f"{DL1_TEL_POINTING_GROUP}/tel_{tel_id:03d}",
            )
            pointing_info.append(tel_pointing)
        pointing_info = vstack(pointing_info)
        return pointing_info


def main():
    # Run the tool
    tool = MonoPredictCTLearnModel()
    tool.run()


if __name__ == "__main__":
    main()