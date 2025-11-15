"""
Append a subarray table to the hdf5 file after the monoscopic predictions.
"""

from astropy.table import vstack
import numpy as np

from ctapipe.io import read_table, write_table
from ctapipe.core import Tool
from ctapipe.core.traits import (
    ComponentName,
    Path,
    Unicode,
    Bool,
    List,
)
from ctapipe.instrument import SubarrayDescription
from ctapipe.reco.reconstructor import ReconstructionProperty
from ctapipe.reco.stereo_combination import StereoCombiner

DL2_SUBARRAY_GROUP = "/dl2/event/subarray"
DL2_TELESCOPE_GROUP = "/dl2/event/telescope"
SUBARRAY_EVENT_KEYS = ["obs_id", "event_id"]
TELESCOPE_EVENT_KEYS = ["obs_id", "event_id", "tel_id"]

__all__ = [
    "AppendSubarrayTable",
]


class AppendSubarrayTable(Tool):
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
    stereo_combiner_cls : str
        Which ctapipe stereo combination method to use after the monoscopic reconstruction.
    overwrite_tables : bool
        Overwrite the table in the hdf5 file if it exists.
    """
    name = "AppendSubarrayTable"
    description = "Append a subarray table to the hdf5 file after the monoscopic predictions."

    input_url = Path(
        help="Input ctapipe HDF5 files including monoscopic predictions.",
        allow_none=False,
        exists=True,
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

    stereo_combiner_cls = ComponentName(
        StereoCombiner,
        default_value="StereoMeanCombiner",
        help="Which ctapipe stereo combination method to use after the monoscopic reconstruction.",
    ).tag(config=True)

    overwrite_tables = Bool(
        default_value=True,
        allow_none=False,
        help="Overwrite the table in the hdf5 file if it exists",
    ).tag(config=True)

    aliases = {
        ("i", "input_url"): "AppendSubarrayTable.input_url",
        ("p", "prefix"): "AppendSubarrayTable.prefix",
        ("r", "reco-tasks"): "AppendSubarrayTable.reco_tasks",
        ("o", "overwrite-tables"): "AppendSubarrayTable.overwrite_tables",
        ("s", "stereo-combiner-cls"): "AppendSubarrayTable.stereo_combiner_cls",
    }

    def setup(self):
        # Set up the reconstruction properties for the stereo combiner
        self.reco_properties = {
            "geometry": ReconstructionProperty.GEOMETRY,
            "energy": ReconstructionProperty.ENERGY,
            "classification": ReconstructionProperty.PARTICLE_TYPE,
        }
        # Read the SubarrayDescription from the input file
        self.subarray = SubarrayDescription.from_hdf(self.input_url)


    def start(self):
        # Loop over the reconstruction tasks and combine the telescope tables to a subarray table
        for reco_task in self.reco_tasks:
            self.log.info("Processing %s...", reco_task)
            # Instantiate the stereo combiner
            self.stereo_combiner = StereoCombiner.from_name(
                self.stereo_combiner_cls,
                prefix=self.prefix,
                property=self.reco_properties[reco_task],
                parent=self,
            )
            # Read the telescope tables from the input file
            tel_tables = []
            for tel_id in self.subarray.tel_ids:
                tel_tables.append(
                    read_table(
                        self.input_url,
                        f"{DL2_TELESCOPE_GROUP}/{reco_task}/{self.prefix}/tel_{tel_id:03}",
                    )
                )
            # Stack the telescope tables to a common table
            tel_tables = vstack(tel_tables)
            # Sort the table by the telescope event keys
            tel_tables.sort(TELESCOPE_EVENT_KEYS)
            # Combine the telescope predictions to the subarray prediction using the ctapipe stereo combiner
            subarray_table = self.stereo_combiner.predict_table(
                tel_tables
            )
            # TODO: Remove temporary fix once the stereo combiner returns correct table
            # Check if the table has to be converted to a boolean mask
            if (
                subarray_table[f"{self.prefix}_telescopes"].dtype
                != np.bool_
            ):
                # Create boolean mask for telescopes that participate in the stereo reconstruction combination
                reco_telescopes = np.zeros(
                    (len(subarray_table), len(self.subarray.tel_ids)),
                    dtype=bool,
                )
                # Loop over the table and set the boolean mask for the telescopes
                for index, tel_id_mask in enumerate(
                    subarray_table[f"{self.prefix}_telescopes"]
                ):
                    if not tel_id_mask:
                        continue
                    for tel_id in tel_id_mask:
                        reco_telescopes[index][
                            self.subarray.tel_ids_to_indices(tel_id)
                        ] = True
                # Overwrite the column with the boolean mask with fix length
                subarray_table[f"{self.prefix}_telescopes"] = (
                    reco_telescopes
                )
            # Sort the table by the subarray event keys
            subarray_table.sort(SUBARRAY_EVENT_KEYS)
            # Save the prediction to the file
            write_table(
                subarray_table,
                self.input_url,
                f"{DL2_SUBARRAY_GROUP}/{reco_task}/{self.prefix}",
                overwrite=self.overwrite_tables,
            )
            self.log.info(
                "DL2 prediction data was stored in '%s' under '%s'",
                self.input_url,
                f"{DL2_SUBARRAY_GROUP}/{reco_task}/{self.prefix}",
            )

    def finish(self):
        # Shutting down the tool
        self.log.info("Tool is shutting down")

def main():
    # Run the tool
    tool = AppendSubarrayTable()
    tool.run()


if __name__ == "__main__":
    main()
