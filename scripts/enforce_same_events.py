from astropy.table import vstack, join
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
    "EnforceSameEvents",
]


class EnforceSameEvents(Tool):
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
    name = "EnforceSameEvents"
    description = "Append a subarray table to the hdf5 file after the monoscopic predictions."

    input_url = Path(
        help="Input ctapipe HDF5 files including monoscopic predictions.",
        allow_none=False,
        exists=True,
        directory_ok=False,
        file_ok=True,
    ).tag(config=True)

    output_url = Path(
        help="Output ctapipe HDF5 files including monoscopic predictions.",
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

    dl2_telescope= Bool(
        default_value=True,
        help="Whether to create dl2 telescope group if it does not exist.",
    ).tag(config=True)

    dl2_subarray = Bool(
        default_value=True,
        help="Whether to create dl2 subarray group if it does not exist.",
    ).tag(config=True)

    aliases = {
        ("i", "input"): "EnforceSameEvents.input_url",
        ("o", "output"): "EnforceSameEvents.output_url",
        ("p", "prefix"): "EnforceSameEvents.prefix",
        ("r", "reco-tasks"): "EnforceSameEvents.reco_tasks",
        "dl2-telescope": "EnforceSameEvents.dl2_telescope",
        "dl2-subarray": "EnforceSameEvents.dl2_subarray",
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
        
            # Read the telescope tables from the input file
            tel_tables = []
            if self.dl2_telescope:
                for tel_id in self.subarray.tel_ids:
                    input_tel_table = read_table(
                        self.input_url,
                        f"{DL2_TELESCOPE_GROUP}/{reco_task}/{self.prefix}/tel_{tel_id:03}",
                    )
                    output_tel_table = read_table(
                        self.output_url,
                        f"{DL2_TELESCOPE_GROUP}/{reco_task}/{self.prefix}/tel_{tel_id:03}",
                    )
                    joined_table = join(
                        input_tel_table,
                        output_tel_table,
                        keys=TELESCOPE_EVENT_KEYS,
                        join_type="inner",
                    )
                    self.log.info("Input table for telescope %03d has %d rows", tel_id, len(input_tel_table))
                    self.log.info("Output table for telescope %03d has %d rows", tel_id, len(output_tel_table))
                    self.log.info("Joined table for telescope %03d has %d rows", tel_id, len(joined_table))
                    self.log.info(joined_table.colnames)
            # Stack the telescope tables to a common table
            #tel_tables = vstack(tel_tables)
            # Sort the table by the telescope event keys
            #tel_tables.sort(TELESCOPE_EVENT_KEYS)
           
            # Sort the table by the subarray event keys
            #subarray_table.sort(SUBARRAY_EVENT_KEYS)
            # Save the prediction to the file
            #write_table(
            #    subarray_table,
            #    self.input_url,
            #    f"{DL2_SUBARRAY_GROUP}/{reco_task}/{self.prefix}",
            #    overwrite=self.overwrite_tables,
            #)
            #self.log.info(
            #    "DL2 prediction data was stored in '%s' under '%s'",
            #    self.input_url,
            #    f"{DL2_SUBARRAY_GROUP}/{reco_task}/{self.prefix}",
            #)

    def finish(self):
        # Shutting down the tool
        self.log.info("Tool is shutting down")

def main():
    # Run the tool
    tool = EnforceSameEvents()
    tool.run()


if __name__ == "__main__":
    main()
