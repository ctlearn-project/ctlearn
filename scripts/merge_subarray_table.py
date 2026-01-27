
import sys
from argparse import ArgumentParser
from astropy.table import join, unique, vstack, Table, setdiff
import math
import re
import numpy as np
import pathlib 
import astropy.units as u

from ctapipe.io import HDF5Merger, read_table, write_table
from ctapipe.containers import (
    ParticleClassificationContainer,
    ReconstructedGeometryContainer,
    ReconstructedEnergyContainer,
)
from ctapipe.core import Tool
from ctapipe.core.traits import (
    Unicode,
    Bool,
    Path,
    List,
    Enum,
    classes_with_traits,
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
        allow_none=True,
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
        Path(exists=True, directory_ok=False),
        default_value=[],
        help="Input ctapipe HDF5 files including stereoscopic predictions.",
    ).tag(config=True)

    file_pattern = Unicode(
        default_value="*.h5",
        help="Give a specific file pattern for matching files in ``input_dir``",
    ).tag(config=True)

    output_path = Path(
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

    n_telescopes = Enum(
        [3,4],
        default_value=4,
        allow_none=False,
        help="Number of telescopes in the subarray. "
             "This is used to determine the telescope combinations.",
    ).tag(config=True)

    overwrite = Bool(
        default_value=True,
        allow_none=False,
        help="Overwrite the table in the hdf5 file if it exists",
    ).tag(config=True)

    parser = ArgumentParser()
    parser.add_argument("input_files", nargs="*", type=pathlib.Path)

    aliases = {
        ("i", "input-dir"): "MergeSubarrayTables.input_dir",
        ("o", "output"): "MergeSubarrayTables.output_path",
        ("p", "pattern"): "MergeSubarrayTables.file_pattern",
        ("t", "n_telescopes"): "MergeSubarrayTables.n_telescopes",
    }

    flags = {
        "overwrite": (
            {"HDF5Merger": {"overwrite": True}},
            "Overwrite existing files",
        ),
        "append": (
            {"HDF5Merger": {"append": True}},
            "Append to existing files",
        ),
    }

    classes = classes_with_traits(HDF5Merger)

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
        
        # Merge the first input file to the output path
        with HDF5Merger(
            parent=self,
            output_path=self.output_path,
        ) as merger:
            merger(self.input_files[0])
        # Read the SubarrayDescription from the first input file
        #self.subarray = SubarrayDescription.read(self.input_files[0])
        if self.n_telescopes == 4:
            self.tel_id_2_index = {1:0, 2:1, 3:2, 4:3}
            self.tel_combinations = [[1,4], [4,3], [3,2], [2,1], [1,3], [4,2]]
        elif self.n_telescopes == 3:
            self.tel_id_2_index = {1:0, 3:1, 4:2}
            self.tel_combinations = [[1,4], [4,3], [1,3]]
       
        dl1b_parameters = []
        for tel_id in self.tel_id_2_index.keys():
            dl1b_parameters.append(
                read_table(
                    self.input_files[0],
                    f"/dl1/event/telescope/parameters/tel_{tel_id:03d}"
                )
            )

        dl1b_parameter_table = vstack(dl1b_parameters)
        #self.log.info(dl1b_parameter_table)

        dl1b_parameter_table.sort(TELESCOPE_EVENT_KEYS)

        dl1b_parameter_groups = dl1b_parameter_table.group_by(SUBARRAY_EVENT_KEYS)
        self.weights = {}
        self.weights["obs_id"] = np.zeros(
            len(dl1b_parameter_groups.groups),
            dtype=int
        )
        self.weights["event_id"] = np.zeros(
            len(dl1b_parameter_groups.groups),
            dtype=int
        )
        self.weights[f"{self.prefix}_telescopes"] = np.zeros(
            (len(dl1b_parameter_groups.groups), len(self.tel_id_2_index)),
            dtype=bool
        )
        self.weights[f"{self.prefix}_is_valid"] = np.ones(
            len(dl1b_parameter_groups.groups),
            dtype=bool
        )
        for tel_combination in self.tel_combinations:
            self.weights[f"LST{tel_combination[0]}LST{tel_combination[1]}_norm"] = np.zeros(
                len(dl1b_parameter_groups.groups),
                dtype=float
            )
        for g, grp in enumerate(dl1b_parameter_groups.groups):
            # Save the obs_id and event_id for each group
            self.weights["obs_id"][g] = int(grp["obs_id"][0])
            self.weights["event_id"][g] = int(grp["event_id"][0])
            #if self.weights["obs_id"][g] > 2:
            #    break
            # Create boolean array for tel_ids with hillas_intensity > 25
            # tel_id indexing starts from 1
            tel_bool_array = np.zeros(len(self.tel_id_2_index), dtype=bool)                                        
            tel_hillas_array = np.zeros(len(self.tel_id_2_index), dtype=float)                                        
            # Set True for tel_ids with high intensity
            tel_mask = grp["hillas_intensity"] > 25
            for i, survival_tel_id in enumerate(grp["tel_id"][tel_mask]):
                tel_bool_array[self.tel_id_2_index[survival_tel_id]] = True
                tel_hillas_array[self.tel_id_2_index[survival_tel_id]] = grp["hillas_intensity"][tel_mask][i] # Adjust for 0-based indexing
            self.weights[f"{self.prefix}_telescopes"][g] = tel_bool_array

            surviving_tel_ids = set(grp["tel_id"][tel_mask])
            if len(grp["tel_id"][tel_mask]) < 2:
                self.weights[f"{self.prefix}_is_valid"][g] = False
            elif len(grp["tel_id"][tel_mask]) == 2:
                for tel_combination in self.tel_combinations:
                    if all(tel_id in surviving_tel_ids for tel_id in tel_combination):
                        self.weights[f"LST{tel_combination[0]}LST{tel_combination[1]}_norm"][g] = 1.0
            elif len(grp["tel_id"][tel_mask]) > 2:
                hillas_sum = {}
                for tel_combination in self.tel_combinations:
                    if all(tel_id in surviving_tel_ids for tel_id in tel_combination):
                        hillas_sum[f"LST{tel_combination[0]}LST{tel_combination[1]}"] = tel_hillas_array[self.tel_id_2_index[tel_combination[0]]] + tel_hillas_array[self.tel_id_2_index[tel_combination[1]]]
                
                hillas_norms = {key: value / np.sum(list(hillas_sum.values())) for key, value in hillas_sum.items()}
                for key, hillas_norm in hillas_norms.items():
                    self.weights[f"{key}_norm"][g] = hillas_norm
        # Convert to astropy Table
        self.weights = Table(data=self.weights)
        # self.log.info(len(self.weights))

    def start(self):

        # Loop over the reconstruction tasks and combine the telescope tables to a subarray table
        class_table, eng_table, dir_table = None, None, None
        for reco_task in self.reco_tasks:
            self.log.info("Processing %s...", reco_task)
            
            # Read and join the subarray tables from the input files
            subarray_tables = []
            for input_file in self.input_files:

                self.log.info("Reading from file: %s", input_file)
                
                trigger = read_table(
                    input_file,
                    "/dl1/event/subarray/trigger",
                )
                self.log.info(trigger)
                shower = read_table(
                    input_file,
                    "/simulation/event/subarray/shower",
                )
                self.log.info(shower)
                self.log.info(len(trigger))
                self.log.info(len(shower))
                
                dl2_tab = read_table(
                    input_file,
                    f"{DL2_SUBARRAY_GROUP}/{reco_task}/{self.prefix}",
                )

                self.log.info(dl2_tab)
                self.log.info(len(dl2_tab))
                self.log.info(len(trigger))
                self.log.info(len(shower))
                dl2_tab.keep_columns(SUBARRAY_EVENT_KEYS + self.reco_colnames[reco_task])
                dl2_tab = join(
                    left=dl2_tab,
                    right=self.weights,
                    keys=SUBARRAY_EVENT_KEYS,
                )
                # Extract telescope IDs from filename by finding digits after 'LST' substrings               
                tel_ids_comb = [int(match) for match in re.findall(r'LST(\d+)', str(input_file)) if match.isdigit()]
                for col_name in self.reco_colnames[reco_task]:
                    if reco_task == "energy":
                        dl2_tab[col_name] = np.log10((u.Quantity(dl2_tab[col_name], unit=dl2_tab[col_name].unit).to_value(u.GeV)))
                    
                    dl2_tab[col_name] = dl2_tab[col_name].data * dl2_tab[f"LST{tel_ids_comb[0]}LST{tel_ids_comb[1]}_norm"].data
                subarray_tables.append(dl2_tab)
            subarray_table = vstack(subarray_tables)

            subarray_table.keep_columns(SUBARRAY_EVENT_KEYS + self.reco_colnames[reco_task] + [f"{self.prefix}_telescopes", f"{self.prefix}_is_valid"])
            subarray_table.sort(SUBARRAY_EVENT_KEYS)
            if reco_task == "classification":
                class_table = subarray_table.copy()
            elif reco_task == "energy":
                eng_table = subarray_table.copy()
            elif reco_task == "geometry":
                dir_table = subarray_table.copy()
            self.log.info(len(subarray_table))

            # Deep copy the table to avoid modifying the original table
            predictions = subarray_table.copy()
            # Keep only the relevant columns for the mean calculation
            predictions.keep_columns(
                SUBARRAY_EVENT_KEYS + self.reco_colnames[reco_task]
            )
            # Group the predictions by the subarray event keys
            predictions_grouped = predictions.group_by(SUBARRAY_EVENT_KEYS)

            # Calculate the mean predictions for each subarray event
            mean_predictions = predictions_grouped.groups.aggregate(np.nansum)
            if reco_task == "energy":
                mean_predictions[self.reco_colnames[reco_task][0]] = 10 ** mean_predictions[self.reco_colnames[reco_task][0]]

            # Sort the mean prediction table by the subarray event keys
            mean_predictions.sort(SUBARRAY_EVENT_KEYS)

            # Unique the subarray tables to avoid duplicates
            # this is needed because of the vstack above
            final_subarray_table = unique(
                subarray_table, keys=SUBARRAY_EVENT_KEYS
            )

            # Remove the columns that will be replace by the mean predictions
            final_subarray_table.remove_columns(self.reco_colnames[reco_task])

            # Join the mean predictions to the subarray table
            final_subarray_table = join(
                left=final_subarray_table,
                right=mean_predictions,
                keys=SUBARRAY_EVENT_KEYS,
            )
            final_subarray_table.sort(SUBARRAY_EVENT_KEYS)
            #final_subarray_table[f"{self.prefix}_telescopes"] = self.weights[f"{self.prefix}_telescopes"]
            #final_subarray_table[f"{self.prefix}_is_valid"] = self.weights[f"{self.prefix}_is_valid"]

            for col_name in self.reco_colnames[reco_task]:
                # Set the prediction to NaN if the event is not valid
                final_subarray_table[col_name] = np.where(
                    final_subarray_table[f"{self.prefix}_is_valid"],
                    final_subarray_table[col_name],
                    np.nan
                )

                # Add units to the columns
                if reco_task == "energy":
                    final_subarray_table[col_name] = u.Quantity(
                        final_subarray_table[col_name],
                        unit=u.GeV,
                    ).to(u.TeV)

                if reco_task == "geometry":
                    final_subarray_table[col_name] = u.Quantity(
                        final_subarray_table[col_name],
                        unit=u.deg,
                    )

            # Add the default values and meta data to the table
            add_defaults_and_meta(
                final_subarray_table,
                self.reco_containers[reco_task],
                prefix=self.prefix,
                add_tel_prefix=False,
            )

            # Save the prediction to the file
            write_table(
                final_subarray_table,
                self.output_path,
                f"{DL2_SUBARRAY_GROUP}/{reco_task}/{self.prefix}",
                overwrite=self.overwrite,
            )
            self.log.info(
                "DL2 prediction data was stored in '%s' under '%s'",
                self.output_path,
                f"{DL2_SUBARRAY_GROUP}/{reco_task}/{self.prefix}",
            )
        diff_1 = setdiff(
            self.weights, class_table, keys=SUBARRAY_EVENT_KEYS
        )
        self.log.info(diff_1)
        diff_2 = setdiff(
            self.weights, eng_table, keys=SUBARRAY_EVENT_KEYS
        )
        self.log.info(diff_2)
        diff_3 = setdiff(
            self.weights, dir_table, keys=SUBARRAY_EVENT_KEYS
        )
        self.log.info(diff_3)
         

    def finish(self):
        # Shutting down the tool
        self.log.info("Tool is shutting down")

def main():
    # Run the tool
    tool = MergeSubarrayTables()
    tool.run()


if __name__ == "__main__":
    main()
