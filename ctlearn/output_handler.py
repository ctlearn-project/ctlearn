import os
import logging
import numpy as np
import pandas as pd
from astropy.table import Table
from astropy.coordinates import SkyCoord
import astropy.units as u


def write_output(h5file, data, rest_data, reader, predictions, tasks):
    prediction_dir = h5file.replace(f'{h5file.split("/")[-1]}', "")
    if not os.path.exists(prediction_dir):
        os.makedirs(prediction_dir)

    # Store dl2 data
    reco = {}
    if os.path.isfile(h5file):
        with pd.HDFStore(h5file, mode="r") as file:
            h5file_keys = list(file.keys())
            if f"/dl2/reco" in h5file_keys:
                reco = pd.read_hdf(file, key=f"/dl2/reco")

    # Store event and obsveration ids
    if data.evt_pos:
        event_id = data.event_list[data.batch_size :]
        if rest_data:
            event_id = np.concatenate(
                (
                    event_id,
                    rest_data.event_list[rest_data.batch_size :],
                ),
                axis=0,
            )
        reco["event_id"] = event_id
    if data.obs_pos:
        obs_id = data.obs_list[data.batch_size :]
        if rest_data:
            obs_id = np.concatenate(
                (
                    obs_id,
                    rest_data.obs_list[rest_data.batch_size :],
                ),
                axis=0,
            )
        reco["obs_id"] = obs_id

    # Store telescope pointings
    if reader.telescope_pointings is not None:
        for tel_id in reader.telescope_pointings:
            pd.DataFrame(data=reader.telescope_pointings[tel_id].to_pandas()).to_hdf(
                h5file, key=f"/monitoring/telescope/pointing/{tel_id}", mode="a"
            )

    # Store predictions and simulation values
    # Gamma/hadron classification
    if data.prt_pos:
        true_shower_primary_id = data.prt_labels[data.batch_size :]
        if rest_data:
            true_shower_primary_id = np.concatenate(
                (
                    true_shower_primary_id,
                    rest_data.prt_labels[rest_data.batch_size :],
                ),
                axis=0,
            )
        reco["true_shower_primary_id"] = true_shower_primary_id
    if "type" in tasks:
        for n, name in enumerate(data.class_names):
            reco[name + "ness"] = np.array(predictions[:, n])
    # Energy regression
    if data.enr_pos:
        if data.energy_unit == "log(TeV)":
            true_energy = np.power(10, data.enr_labels[data.batch_size :])
            if rest_data:
                true_energy = np.concatenate(
                    (
                        true_energy,
                        np.power(10, rest_data.enr_labels[rest_data.batch_size :]),
                    ),
                    axis=0,
                )
            reco["true_energy"] = true_energy
        else:
            true_energy = data.enr_labels[data.batch_size :]
            if rest_data:
                true_energy = np.concatenate(
                    (
                        true_energy,
                        rest_data.enr_labels[rest_data.batch_size :],
                    ),
                    axis=0,
                )
            reco["true_energy"] = true_energy
    if "energy" in tasks:
        if data.energy_unit == "log(TeV)":
            reco["reco_energy"] = np.power(10, predictions)[:, 0]
        else:
            reco["reco_energy"] = np.array(predictions)[:, 0]
    # Arrival direction regression
    if data.drc_pos:
        true_az_offset = data.az_labels[data.batch_size :]
        true_alt_offset = data.alt_labels[data.batch_size :]
        true_sep = data.sep_labels[data.batch_size :]
        if rest_data:
            true_az_offset = np.concatenate(
                (
                    true_az_offset,
                    rest_data.az_labels[rest_data.batch_size :],
                ),
                axis=0,
            )
            true_alt_offset = np.concatenate(
                (
                    true_alt_offset,
                    rest_data.alt_labels[rest_data.batch_size :],
                ),
                axis=0,
            )
            true_sep = np.concatenate(
                (
                    true_sep,
                    rest_data.sep_labels[rest_data.batch_size :],
                ),
                axis=0,
            )
        reco["true_sep"] = true_sep
        if reader.fix_pointing is None:
            reco["true_az"] = true_az_offset
            reco["true_alt"] = true_alt_offset
        else:
            true_az, true_alt = [], []
            for az_off, alt_off in zip(true_az_offset, true_alt_offset):
                true_direction = reader.fix_pointing.spherical_offsets_by(
                    u.Quantity(az_off, unit=data.drc_unit),
                    u.Quantity(alt_off, unit=data.drc_unit),
                )
                true_az.append(true_direction.az.to_value(data.drc_unit))
                true_alt.append(true_direction.alt.to_value(data.drc_unit))
            reco["true_az"] = np.array(true_az)
            reco["true_alt"] = np.array(true_alt)
    if "direction" in tasks:
        if reader.fix_pointing is None:
            reco["reco_az"] = np.array(predictions[:, 0])
            reco["reco_alt"] = np.array(predictions[:, 1])
            reco["reco_sep"] = np.array(predictions[:, 2])
        else:
            reco_az, reco_alt = [], []
            for az_off, alt_off in zip(predictions[:, 0], predictions[:, 1]):
                reco_direction = reader.fix_pointing.spherical_offsets_by(
                    u.Quantity(az_off, unit=data.drc_unit),
                    u.Quantity(alt_off, unit=data.drc_unit),
                )
                reco_az.append(reco_direction.az.to_value(data.drc_unit))
                reco_alt.append(reco_direction.alt.to_value(data.drc_unit))
            reco["reco_az"] = np.array(reco_az)
            reco["reco_alt"] = np.array(reco_alt)
            reco["reco_sep"] = np.array(predictions[:, 2])
            reco["pointing_az"] = np.full(
                len(reco["reco_az"]), reader.fix_pointing.az.to_value(data.drc_unit)
            )
            reco["pointing_alt"] = np.full(
                len(reco["reco_alt"]), reader.fix_pointing.alt.to_value(data.drc_unit)
            )

    if data.trgpatch_pos:
        cherenkov_photons = data.trgpatch_labels[data.batch_size :]
        if rest_data:
            cherenkov_photons = np.concatenate(
                (
                    cherenkov_photons,
                    rest_data.trgpatch_labels[rest_data.batch_size :],
                ),
                axis=0,
            )
        reco["true_cherenkov_photons"] = cherenkov_photons
    if "cherenkov_photons" in tasks:
        reco["reco_cherenkov_photons"] = np.array(predictions)[:, 0]
    # Dump the dl2 data to hdf5 file
    if reader.include_nsb_patches is None:
        pd.DataFrame(data=reco).to_hdf(h5file, key=f"/dl2/reco", mode="a")
    else:
        pd.DataFrame(data=reco).to_hdf(h5file, key=f"/trigger/reco", mode="a")

    # Store the simulation information for pyirf
    if reader.simulation_info and reader.include_nsb_patches != "all":
        pd.DataFrame(data=reader.simulation_info, index=[0]).to_hdf(
            h5file, key=f"/info/mc_header", mode="a"
        )

    # Store the selected Hillas parameters (dl1b)
    if reader.parameter_list and reader.include_nsb_patches != "all":
        tel_counter = 0
        if reader.mode == "mono":
            tel_type = list(reader.selected_telescopes.keys())[0]
            tel_ids = "tel"
            for tel_id in reader.selected_telescopes[tel_type]:
                tel_ids += f"_{tel_id}"
            parameters = {}
            for p, parameter in enumerate(reader.parameter_list):
                parameter_list = np.array(data.parameter_list)[data.batch_size :, p]
                if rest_data:
                    parameter_list = np.concatenate(
                        (
                            parameter_list,
                            np.array(rest_data.parameter_list)[
                                rest_data.batch_size :, p
                            ],
                        ),
                        axis=0,
                    )
                parameters[parameter] = parameter_list
            pd.DataFrame(data=parameters).to_hdf(
                h5file, key=f"/dl1b/{tel_type}/{tel_ids}", mode="a"
            )
        else:
            for tel_type in reader.selected_telescopes:
                for t, tel_id in enumerate(reader.selected_telescopes[tel_type]):
                    parameters = {}
                    for p, parameter in enumerate(reader.parameter_list):
                        parameter_list = np.array(data.parameter_list)[
                            data.batch_size :, tel_counter + t, p
                        ]
                        if rest_data:
                            parameter_list = np.concatenate(
                                (
                                    parameter_list,
                                    np.array(rest_data.parameter_list)[
                                        rest_data.batch_size :, tel_counter + t, p
                                    ],
                                ),
                                axis=0,
                            )
                        parameters[parameter] = parameter_list
                    pd.DataFrame(data=parameters).to_hdf(
                        h5file, key=f"/dl1b/{tel_type}/tel_{tel_id}", mode="a"
                    )
                tel_counter += len(reader.selected_telescopes[tel_type])
