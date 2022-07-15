import os
import logging
import numpy as np
import pandas as pd


def write_output(h5file, data, rest_data, reader, predictions, tasks):

    prediction_dir = h5file.replace(f'{h5file.split("/")[-1]}', "")
    if not os.path.exists(prediction_dir):
        os.makedirs(prediction_dir)

    # Store the information for observational data
    if reader.instrument_id == "MAGIC" and reader.process_type == "Observation":
        # Dump the information of the run to hdf5 file
        run_info = {}
        run_info["run_number"] = reader._v_attrs["run_number"]
        run_info["magic_number"] = reader._v_attrs["magic_number"]
        run_info["num_events"] = reader._v_attrs["num_events"]
        run_info["run_start_mjd"] = reader._v_attrs["run_start_mjd"]
        run_info["run_start_ms"] = reader._v_attrs["run_start_ms"]
        run_info["run_start_ns"] = reader._v_attrs["run_start_ns"]
        run_info["run_stop_mjd"] = reader._v_attrs["run_stop_mjd"]
        run_info["run_stop_ms"] = reader._v_attrs["run_stop_ms"]
        run_info["run_stop_ns"] = reader._v_attrs["run_stop_ns"]
        pd.DataFrame(data=run_info, index=[0]).to_hdf(
            h5file, key=f"/info/run", mode="a"
        )
        # Dump the information of the obsveration to hdf5 file
        obs_info = {}
        obs_info["source_name"] = reader._v_attrs["source_name"]
        obs_info["project_name"] = reader._v_attrs["project_name"]
        obs_info["observation_mode"] = reader._v_attrs["observation_mode"]
        obs_info["source_dec"] = reader._v_attrs["source_dec"]
        obs_info["source_ra"] = reader._v_attrs["source_ra"]
        obs_info["telescope_dec"] = reader._v_attrs["telescope_dec"]
        obs_info["telescope_ra"] = reader._v_attrs["telescope_ra"]
        pd.DataFrame(data=obs_info, index=[0]).to_hdf(
            h5file, key=f"/info/obs", mode="a"
        )

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
            reco["obs_id"] = np.concatenate(
                (
                    obs_id,
                    rest_data.obs_list[rest_data.batch_size :],
                ),
                axis=0,
            )

    # Store the timestamp
    if data.mjd_pos:
        mjd = data.mjd_list[data.batch_size :]
        if rest_data:
            reco["mjd"] = np.concatenate(
                (
                    mjd,
                    rest_data.mjd_list[rest_data.batch_size :],
                ),
                axis=0,
            )
    if data.milli_pos:
        milli_sec = data.milli_list[data.batch_size :]
        if rest_data:
            reco["milli_sec"] = np.concatenate(
                (
                    milli_sec,
                    rest_data.milli_list[rest_data.batch_size :],
                ),
                axis=0,
            )
    if data.nano_pos:
        nano_sec = data.nano_list[data.batch_size :]
        if rest_data:
            reco["nano_sec"] = np.concatenate(
                (
                    nano_sec,
                    rest_data.nano_list[rest_data.batch_size :],
                ),
                axis=0,
            )

    # Store pointings
    if data.pon_pos:
        pointing_alt = np.array(data.pointing)[data.batch_size :, 0]
        pointing_az = np.array(data.pointing)[data.batch_size :, 1]
        if rest_data:
            pointing_alt = np.concatenate(
                (
                    pointing_alt,
                    np.array(rest_data.pointing)[rest_data.batch_size :, 0],
                ),
                axis=0,
            )
            pointing_az = np.concatenate(
                (
                    pointing_az,
                    np.array(rest_data.pointing)[rest_data.batch_size :, 1],
                ),
                axis=0,
            )
    else:
        pointing_alt = np.array([reader.pointing[0]] * len(reader))
        pointing_az = np.array([reader.pointing[1]] * len(reader))
    reco["pointing_alt"] = pointing_alt
    reco["pointing_az"] = pointing_az

    # Store predictions and simulation values
    # Gamma/hadron classification
    if data.prt_pos:
        true_shower_primary_id = data.prt_labels[data.batch_size :]
        if rest_data:
            reco["true_shower_primary_id"] = np.concatenate(
                (
                    true_shower_primary_id,
                    rest_data.prt_labels[rest_data.batch_size :],
                ),
                axis=0,
            )
    if "particletype" in tasks:
        for n, name in enumerate(data.class_names):
            reco[name + "ness"] = np.array(predictions[:, n])
    # Energy regression
    if data.enr_pos:
        if data.energy_unit == "log(TeV)":
            true_energy = np.power(10, data.enr_labels[data.batch_size :])
            if rest_data:
                reco["true_energy"] = np.concatenate(
                    (
                        true_energy,
                        np.power(10, rest_data.enr_labels[rest_data.batch_size :]),
                    ),
                    axis=0,
                )
        else:
            true_energy = data.enr_labels[data.batch_size :]
            if rest_data:
                reco["true_energy"] = np.concatenate(
                    (
                        true_energy,
                        rest_data.enr_labels[rest_data.batch_size :],
                    ),
                    axis=0,
                )
    if "energy" in tasks:
        if data.energy_unit == "log(TeV)" or np.min(predictions) < 0.0:
            reco["reco_energy"] = np.power(10, predictions)[:, 0]
        else:
            reco["reco_energy"] = np.array(predictions)[:, 0]
    # Arrival direction regression
    if data.drc_pos:
        alt = data.alt_labels[data.batch_size :]
        az = data.az_labels[data.batch_size :]
        if rest_data:
            alt = (
                np.concatenate(
                    (
                        alt,
                        rest_data.alt_labels[rest_data.batch_size :],
                    ),
                    axis=0,
                )
                + pointing_alt
            )
            az = (
                np.concatenate(
                    (
                        az,
                        rest_data.az_labels[rest_data.batch_size :],
                    ),
                    axis=0,
                )
                + pointing_az
            )
        if "corsika_version" not in reader._v_attrs:
            reco["source_alt"] = alt
            reco["source_az"] = az
        else:
            reco["true_alt"] = alt
            reco["true_az"] = az

    if "direction" in tasks:
        reco["reco_alt"] = np.array(predictions[:, 0]) + pointing_alt
        reco["reco_az"] = np.array(predictions[:, 1]) + pointing_az

    # Dump the dl2 data to hdf5 file
    pd.DataFrame(data=reco).to_hdf(h5file, key=f"/dl2/reco", mode="a")

    # Store the simulation information for pyirf
    if reader.simulation_info:
        pd.DataFrame(data=reader.simulation_info, index=[0]).to_hdf(
            h5file, key=f"/info/mc_header", mode="a"
        )

    # Store the selected Hillas parameters (dl1b)
    if reader.parameter_list:
        tel_counter = 0
        if reader.mode == "mono":
            tel_type = list(reader.telescopes.keys())[0]
            tel_ids = "tel"
            for tel_id in reader.telescopes[tel_type]:
                tel_ids += f"_{tel_id}"
            parameters = {}
            for p, parameter in enumerate(reader.parameter_list):
                parameter_list = np.array(data.parameter_list)[data.batch_size :, p]
                if rest_data:
                    parameters[parameter] = np.concatenate(
                        (
                            parameter_list,
                            np.array(rest_data.parameter_list)[
                                rest_data.batch_size :, p
                            ],
                        ),
                        axis=0,
                    )
            pd.DataFrame(data=parameters).to_hdf(
                h5file, key=f"/dl1b/{tel_type}/{tel_ids}", mode="a"
            )
        else:
            for tel_type in reader.telescopes:
                for t, tel_id in enumerate(reader.telescopes[tel_type]):
                    parameters = {}
                    for p, parameter in enumerate(reader.parameter_list):
                        parameter_list = np.array(data.parameter_list)[
                            data.batch_size :, tel_counter + t, p
                        ]
                        if rest_data:
                            parameters[parameter] = np.concatenate(
                                (
                                    parameter_list,
                                    np.array(rest_data.parameter_list)[
                                        rest_data.batch_size :, tel_counter + t, p
                                    ],
                                ),
                                axis=0,
                            )
                    pd.DataFrame(data=parameters).to_hdf(
                        h5file, key=f"/dl1b/{tel_type}/tel_{tel_id}", mode="a"
                    )
                tel_counter += len(reader.telescopes[tel_type])
