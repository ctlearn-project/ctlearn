
import sys
from argparse import ArgumentParser
from astropy.table import join, unique, vstack
import numpy as np
from pathlib import Path

from ctapipe.io import read_table, write_table
from ctapipe.containers import (
    ParticleClassificationContainer,
    ReconstructedGeometryContainer,
    ReconstructedEnergyContainer,
)
from ctapipe.core import Tool, traits
from ctapipe.core.traits import (
    Unicode,
    Bool,
    List,
)
from ctapipe.instrument import SubarrayDescription
from ctapipe.reco.utils import add_defaults_and_meta


DL2_SUBARRAY_GROUP = "/dl2/event/subarray"
SUBARRAY_EVENT_KEYS = ["obs_id", "event_id"]
TELESCOPE_EVENT_KEYS = ["obs_id", "event_id", "tel_id"]

__all__ = [
    "MergeSubarrayTables",
]


class MergeSubarrayTables(Tool):
    """
    Append a subarray table to the hdf5 file after the monoscopic predictions.

    This tool reads the monoscopic predictions from the input hdf5 file and combines them
    to a subarray table using the ctapipe stereo combiner. The subarray table is then written
    to the hdf5 file.

    Parameters
    ----------
    input_url : str
        Input ctapipe HDF5 files including monoscopic predictions.
    prefix : str
        Name of the reconstruction algorithm used to generate the dl2 data.
    reco_tasks : list
        List of reconstruction tasks to be used for the stereo combination.
    overwrite_tables : bool
        Overwrite the table in the hdf5 file if it exists.
    """
    name = "MergeSubarrayTables"
    description = "Merge subarray tables to a single subarray table using the ctapipe stereo combiner."

    input_url = Path(
        help="Input ctapipe HDF5 files including stereoscopic predictions.",
        allow_none=False,
        exists=True,
        directory_ok=False,
        file_ok=True,
    ).tag(config=True)

    input_dir = Path(
        default_value=None,
        help="Input directory",
        allow_none=True,
        exists=True,
        directory_ok=True,
        file_ok=False,
    ).tag(config=True)

    input_files = List(
        traits.Path(exists=True, directory_ok=False),
        default_value=[],
        help="Input ctapipe HDF5 files including stereoscopic predictions.",
    ).tag(config=True)

    file_pattern = Unicode(
        default_value="*.h5",
        help="Give a specific file pattern for matching files in ``input_dir``",
    ).tag(config=True)

    output_path = traits.Path(
        help="Output ctapipe HDF5 file for the merged stereoscopic predictions.",
        allow_none=False,
        exists=False,
        directory_ok=False,
        file_ok=True,
    ).tag(config=True)

    prefix = Unicode(
        default_value="CTLearn",
        allow_none=False,
        help="Name of the reconstruction algorithm used to generate the dl2 data.",
    ).tag(config=True)

    reco_tasks = List(
        default_value=["classification", "energy", "geometry"],
        allow_none=False,
        help="List of reconstruction tasks to be used for the stereo combination.",
    ).tag(config=True)

    overwrite = Bool(
        default_value=True,
        allow_none=False,
        help="Overwrite the table in the hdf5 file if it exists",
    ).tag(config=True)

    parser = ArgumentParser()
    parser.add_argument("input_files", nargs="*", type=Path)

    aliases = {
        ("i", "input-dir"): "MergeSubarrayTables.input_dir",
        ("o", "output"): "MergeSubarrayTables.output_path",
        ("p", "pattern"): "MergeSubarrayTables.file_pattern",
    }

    def setup(self):
        # Set up the containers and colnames based on the reco tasks
        self.reco_containers = {
            "geometry": ReconstructedGeometryContainer,
            "energy": ReconstructedEnergyContainer,
            "classification": ParticleClassificationContainer,
            
        }
            
        self.reco_colnames = {
            "geometry": [f"{self.prefix}_alt", f"{self.prefix}_az"],
            "energy": [f"{self.prefix}_energy"],
            "classification": [f"{self.prefix}_prediction"],
        }
        
        # Get input Files
        args = self.parser.parse_args(self.extra_args)
        self.input_files.extend(args.input_files)
        if self.input_dir is not None:
            self.input_files.extend(sorted(self.input_dir.glob(self.file_pattern)))

        if not self.input_files:
            self.log.critical(
                "No input files provided, either provide --input-dir "
                "or input files as positional arguments"
            )
            sys.exit(1)
        # Read the SubarrayDescription from the first input file
        self.subarray = SubarrayDescription.read(self.input_files[0])

    def start(self):
        # Loop over the reconstruction tasks and combine the telescope tables to a subarray table
        for reco_task in self.reco_tasks:
            self.log.info("Processing %s...", reco_task)
            
            # Read the subarray tables from the input files
            subarray_tables = []
            for input_file in self.input_files:
                subarray_tables.append(
                    read_table(
                        input_file,
                        f"{DL2_SUBARRAY_GROUP}/{reco_task}/{self.prefix}",
                    )
                )
            # Stack the telescope tables to a common table
            subarray_tables = vstack(subarray_tables)
            # Deep copy the table to avoid modifying the original table
            predictions = subarray_tables.copy()
            # Keep only the relevant columns for the mean calculation
            predictions.keep_columns(
                SUBARRAY_EVENT_KEYS + self.reco_colnames[reco_task]
            )
            # Group the predictions by the subarray event keys
            predictions_grouped = predictions.group_by(SUBARRAY_EVENT_KEYS)
            # Calculate the mean predictions for each subarray event
            mean_predictions = predictions_grouped.groups.aggregate(np.mean)
            # Sort the mean prediction table by the subarray event keys
            mean_predictions.sort(SUBARRAY_EVENT_KEYS)
            # Unique the subarray tables to avoid duplicates
            subarray_table = unique(
                subarray_tables, keys=SUBARRAY_EVENT_KEYS
            )
            # Remove the columns that will be replace by the mean predictions
            subarray_table.remove_columns(self.reco_colnames[reco_task])
            # Join the mean predictions to the subarray table
            subarray_table = join(
                left=subarray_table,
                right=mean_predictions,
                keys=SUBARRAY_EVENT_KEYS,
            )
            # Sort the table by the subarray event keys
            subarray_table.sort(SUBARRAY_EVENT_KEYS)
            # Add the default values and meta data to the table
            add_defaults_and_meta(
                subarray_table,
                self.reco_containers[reco_task],
                prefix=self.prefix,
                add_tel_prefix=False,
            )
            # Save the prediction to the file
            write_table(
                subarray_table,
                self.output_path,
                f"{DL2_SUBARRAY_GROUP}/{reco_task}/{self.prefix}",
                overwrite=self.overwrite,
            )
            self.log.info(
                "DL2 prediction data was stored in '%s' under '%s'",
                self.output_path,
                f"{DL2_SUBARRAY_GROUP}/{reco_task}/{self.prefix}",
            )

    def finish(self):
        # Write the SubarrayDescription to the output file
        self.output_path.to_hdf(self.subarray, overwrite=self.overwrite)
        # Shutting down the tool
        self.log.info("Tool is shutting down")

def main():
    # Run the tool
    tool = MergeSubarrayTables()
    tool.run()


if __name__ == "__main__":
    main()
