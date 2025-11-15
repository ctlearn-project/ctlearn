"""
Overwrite the is_valid flags in the hdf5 file.
"""

from astropy.table import join, MaskedColumn
import numpy as np
import os

from ctapipe.io import read_table, write_table, HDF5Merger
from ctapipe.core import Tool, ToolConfigurationError
from ctapipe.core.traits import (
    Path,
    Unicode,
    flag,
    Bool,
    Set,
    CInt,
    List,
)
from ctapipe.instrument import SubarrayDescription

DL2_SUBARRAY_GROUP = "/dl2/event/subarray"
DL2_TELESCOPE_GROUP = "/dl2/event/telescope"
SUBARRAY_EVENT_KEYS = ["obs_id", "event_id"]
TELESCOPE_EVENT_KEYS = ["obs_id", "event_id", "tel_id"]

__all__ = [
    "OverwriteIsValidFlag",
]


class OverwriteIsValidFlag(Tool):
    """
    Overwrite the is_valid flags in the hdf5 file.

    This tool reads the is_valid flags from one ctapipe HDF5 file and overwrites
    them in another ctapipe HDF5 file. The user can specify which reconstruction
    tasks to consider for the overwrite, as well as the prefix used for the
    reconstruction algorithm. The output file will contain the same data as the
    input file to which the is_valid flags were written, but with the is_valid
    flags replaced by those from the other input file.

    Parameters
    ----------
    is_valid_from : str
        Input ctapipe HDF5 files from which the is_valid flags will be taken.
    is_valid_to : str
        Input ctapipe HDF5 files to which the is_valid flags will be overwritten.
    prefix : str
        Name of the reconstruction algorithm used to generate the dl2 data.
    reco_tasks : list
        List of reconstruction tasks to be used for the overwrite of the is_valid flag.
    output_path : str
        Output ctapipe HDF5 files including the overwritten is_valid flags.
    """
    name = "OverwriteIsValidFlag"
    description = "Overwrite the is_valid flags in the hdf5 file."

    is_valid_from = Path(
        help="Input ctapipe HDF5 files from which the is_valid flags will be taken.",
        allow_none=False,
        exists=True,
        directory_ok=False,
        file_ok=True,
    ).tag(config=True)

    is_valid_to = Path(
        help="Input ctapipe HDF5 files to which the is_valid flags will be overwritten.",
        allow_none=False,
        exists=True,
        directory_ok=False,
        file_ok=True,
    ).tag(config=True)

    output_path = Path(
        help="Output ctapipe HDF5 files including the overwritten is_valid flags.",
        allow_none=False,
        exists=False,
        directory_ok=False,
        file_ok=True,
    ).tag(config=True)

    allowed_tels = Set(
        trait=CInt(),
        default_value=None,
        allow_none=True,
        help=(
            "List of allowed tel_ids, others will be ignored. "
            "If None, all telescopes in the input stream will be included."
        ),
    ).tag(config=True)

    prefix = Unicode(
        default_value="CTLearn",
        allow_none=False,
        help="Name of the reconstruction algorithm used to generate the dl2 data.",
    ).tag(config=True)

    reco_tasks = List(
        default_value=["classification", "energy", "geometry"],
        allow_none=False,
        help="List of reconstruction tasks to be used for the overwrite of the is_valid flag.",
    ).tag(config=True)

    dl2_telescope = Bool(
        default_value=True,
        help="Whether to overwrite the is_valid flag in the dl2 telescope group.",
    ).tag(config=True)

    dl2_subarray = Bool(
        default_value=True,
        help="Whether to overwrite the is_valid flag in the dl2 subarray group.",
    ).tag(config=True)

    aliases = {
        ("f", "is-valid-from"): "OverwriteIsValidFlag.is_valid_from",
        ("t", "is-valid-to"): "OverwriteIsValidFlag.is_valid_to",
        ("o", "output"): "OverwriteIsValidFlag.output_path",
        ("p", "prefix"): "OverwriteIsValidFlag.prefix",
        ("r", "reco-tasks"): "OverwriteIsValidFlag.reco_tasks",
    }

    flags = {
        **flag(
            "dl2-telescope",
            "OverwriteIsValidFlag.dl2_telescope",
            "Include overwrite dl2 telescope-event-wise data in the output file",
            "Exclude overwrite dl2 telescope-event-wise data in the output file",
        ),
        **flag(
            "dl2-subarray",
            "OverwriteIsValidFlag.dl2_subarray",
            "Include overwrite dl2 subarray-event-wise data in the output file",
            "Exclude overwrite dl2 subarray-event-wise data in the output file",
        ),
    }

    def setup(self):
        # Check if output file already exists
        if os.path.exists(self.output_path):
            raise ToolConfigurationError(
                f"The output file '{self.output_path}' already exists. Please set "
                "a different output path or manually remove the existing file."
            )
        else:
            # Copy selected tables from the input file to the output file
            self.log.info("Copying to output destination.")
            with HDF5Merger(self.output_path, parent=self) as merger:
                merger(self.is_valid_to)

        # Read the SubarrayDescription from the input file
        self.subarray = SubarrayDescription.from_hdf(self.is_valid_to)
        if self.allowed_tels is not None:
            self.subarray = self.subarray.select_subarray(self.allowed_tels)

    def start(self):
        # Loop over the reconstruction tasks and combine the telescope tables to a subarray table
        for reco_task in self.reco_tasks:
            self.log.info("Processing %s...", reco_task)
            # Read the telescope tables from the input file
            if self.dl2_telescope:
                is_valid_col = f"{self.prefix}_tel_is_valid"
                for tel_id in self.subarray.tel_ids:
                    self.log.info("Processing telescope '%03d' ...", tel_id)
                    input_tel_table = read_table(
                        self.is_valid_from,
                        f"{DL2_TELESCOPE_GROUP}/{reco_task}/{self.prefix}/tel_{tel_id:03d}",
                    )
                    input_tel_table.keep_columns(TELESCOPE_EVENT_KEYS + [is_valid_col])
                    output_tel_table = read_table(
                        self.is_valid_to,
                        f"{DL2_TELESCOPE_GROUP}/{reco_task}/{self.prefix}/tel_{tel_id:03d}",
                    )
                    output_tel_table.remove_columns([is_valid_col])
                    if len(input_tel_table) != len(output_tel_table):
                        self.log.warning(
                            "Input and output telescope tables (tel_id '%03d') have different lengths: %d vs %d",
                            tel_id,
                            len(input_tel_table),
                            len(output_tel_table),
                        )
                    joined_tel_table = join(
                        input_tel_table,
                        output_tel_table,
                        keys=TELESCOPE_EVENT_KEYS,
                        join_type="right",
                    )
                    # Fill missing values in the is_valid column with False if necessary
                    if isinstance(joined_tel_table[is_valid_col], MaskedColumn):
                        joined_tel_table[is_valid_col] = joined_tel_table[is_valid_col].filled(False)
                    # Sort the table by the telescope event keys
                    joined_tel_table.sort(TELESCOPE_EVENT_KEYS)
                    write_table(
                        joined_tel_table,
                        self.output_path,
                        f"{DL2_TELESCOPE_GROUP}/{reco_task}/{self.prefix}/tel_{tel_id:03d}",
                        overwrite=True,
                    )
                    self.log.info(
                        "DL2 prediction data was stored in '%s' under '%s'",
                        self.output_path,
                        f"{DL2_TELESCOPE_GROUP}/{reco_task}/{self.prefix}/tel_{tel_id:03d}",
                    )

            if self.dl2_subarray:
                self.log.info("Processing subarray ...")
                is_valid_col = f"{self.prefix}_is_valid"
                input_subarray_table = read_table(
                    self.is_valid_from,
                    f"{DL2_SUBARRAY_GROUP}/{reco_task}/{self.prefix}",
                )
                input_subarray_table.keep_columns(SUBARRAY_EVENT_KEYS + [is_valid_col])
                output_subarray_table = read_table(
                    self.is_valid_to,
                    f"{DL2_SUBARRAY_GROUP}/{reco_task}/{self.prefix}",
                )
                output_subarray_table.remove_columns([is_valid_col])
                if len(input_subarray_table) != len(output_subarray_table):
                    self.log.warning(
                        "Input and output subarray tables have different lengths: %d vs %d",
                        len(input_subarray_table),
                        len(output_subarray_table),
                    )
                joined_subarray_table = join(
                    input_subarray_table,
                    output_subarray_table,
                    keys=SUBARRAY_EVENT_KEYS,
                    join_type="right",
                )
                # Fill missing values in the is_valid column with False if necessary
                if isinstance(joined_subarray_table[is_valid_col], MaskedColumn):
                    joined_subarray_table[is_valid_col] = joined_subarray_table[is_valid_col].filled(False)
                # Sort the table by the subarray event keys
                joined_subarray_table.sort(SUBARRAY_EVENT_KEYS)
                # Save the prediction to the file
                write_table(
                    joined_subarray_table,
                    self.output_path,
                    f"{DL2_SUBARRAY_GROUP}/{reco_task}/{self.prefix}",
                    overwrite=True,
                )
                self.log.info(
                    "DL2 prediction data was stored in '%s' under '%s'",
                    self.output_path,
                    f"{DL2_SUBARRAY_GROUP}/{reco_task}/{self.prefix}",
                )

    def finish(self):
        # Shutting down the tool
        self.log.info("Tool is shutting down")

def main():
    # Run the tool
    tool = OverwriteIsValidFlag()
    tool.run()


if __name__ == "__main__":
    main()
