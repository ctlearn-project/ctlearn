"""
Build IRFS and sensitivity from CTLearn DL2-like files using pyirf.
Edited from pyirf examples (Credits Noethe et al.):
https://github.com/cta-observatory/pyirf/blob/master/examples/calculate_eventdisplay_irfs.py
"""
import argparse
import glob
import logging
import operator
import os

import numpy as np
from astropy import table
from astropy.table import QTable, MaskedColumn
import astropy.units as u
from astropy.io import fits
import pandas as pd

from pyirf.binning import (
    bin_center,
    create_bins_per_decade,
    add_overflow_bins,
    create_histogram_table,
)
from pyirf.cuts import calculate_percentile_cut, evaluate_binned_cut
from pyirf.sensitivity import calculate_sensitivity, estimate_background
from pyirf.simulations import SimulatedEventsInfo
from pyirf.utils import calculate_theta, calculate_source_fov_offset
from pyirf.benchmarks import energy_bias_resolution, angular_resolution

from pyirf.spectral import (
    calculate_event_weights,
    PowerLaw,
    CRAB_HEGRA,
    IRFDOC_PROTON_SPECTRUM,
    IRFDOC_ELECTRON_SPECTRUM,
)
from pyirf.cut_optimization import optimize_gh_cut

from pyirf.irf import (
    effective_area_per_energy,
    energy_dispersion,
    psf_table,
    background_2d,
)

from pyirf.io import (
    create_aeff2d_hdu,
    create_psf_table_hdu,
    create_energy_dispersion_hdu,
    create_rad_max_hdu,
    create_background_2d_hdu,
)


log = logging.getLogger("pyirf")

# Map the particle ids to the particle information
particles = {
    1: {
        "name": "gamma",
        "target_spectrum": CRAB_HEGRA,
        "mc_header": pd.DataFrame(),
        "events": QTable(),
    },
    0: {
        "name": "proton",
        "target_spectrum": IRFDOC_PROTON_SPECTRUM,
        "mc_header": pd.DataFrame(),
        "events": QTable(),
    },
    2: {
        "name": "electron",
        "target_spectrum": IRFDOC_ELECTRON_SPECTRUM,
        "mc_header": pd.DataFrame(),
        "events": QTable(),
    },
}

# Map units
unit_mapping = {
    "true_energy": u.TeV,
    "reco_energy": u.TeV,
    "pointing_alt": u.rad,
    "pointing_az": u.rad,
    "true_alt": u.rad,
    "true_az": u.rad,
    "reco_alt": u.rad,
    "reco_az": u.rad,
}


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Build IRFS and sensitivity from CTLearn DL2-like files using pyirf."
        )
    )
    parser.add_argument(
        "--input",
        "-i",
        help="Input directories",
        default=["./"],
        nargs="+",
    )
    parser.add_argument(
        "--pattern",
        "-p",
        help="Pattern to mask unwanted files from the data input directory",
        default=["*.h5"],
        nargs="+",
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Output file",
        default="pyirf.fits.gz",
    )
    parser.add_argument(
        "--energy_range",
        "-e",
        help="Energy range in TeV",
        default = [0.03, 30.0],
        nargs="+",
        type=float,
    )
    parser.add_argument(
        "--theta_range",
        "-t",
        help="Theta cut range in deg",
        default = [0.05, 0.3],
        nargs="+",
        type=float,
    )
    parser.add_argument(
        "--obstime",
        help="Observation time in hours",
        default=50,
    )
    parser.add_argument(
        "--alpha",
        help="Scaling between on and off region",
        default=0.2,
    )
    parser.add_argument(
        "--max_bg_radius",
        help="Maximum background radius in deg",
        default=1.0,
    )
    parser.add_argument(
        "--max_gh_cut_eff",
        help="Maximum gamma/hadron cut efficiency",
        default=0.9,
    )
    parser.add_argument(
        "--gh_cut_eff_step",
        help="Gamma/hadron cut efficiency step",
        default=0.01,
    )
    parser.add_argument(
        "--init_gh_cut_eff",
        help="Initial gamma/hadron cut efficiency",
        default=0.4,
    )
    parser.add_argument(
        "--quality_cuts",
        "-c",
        help="String of the quality cuts",
        type=str,
    )
    parser.add_argument(
        "--size_cut",
        "-z",
        help="Minimum size values",
        nargs="+",
        type=float,
    )
    parser.add_argument(
        "--leakage_cut",
        "-l",
        help="Maximum leakage2 intensity values",
        nargs="+",
        type=float,
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logging.getLogger("pyirf").setLevel(logging.DEBUG)

    T_OBS = args.obstime * u.hour

    # scaling between on and off region.
    # (Default) Make off region 5 times larger than on region for better
    # background statistics
    ALPHA = args.alpha

    # Radius to use for calculating bg rate
    MAX_BG_RADIUS = args.max_bg_radius * u.deg
    MAX_GH_CUT_EFFICIENCY = args.max_gh_cut_eff
    GH_CUT_EFFICIENCY_STEP = args.gh_cut_eff_step

    # gh cut used for first calculation of the binned theta cuts
    INITIAL_GH_CUT_EFFICENCY = args.init_gh_cut_eff

    MIN_ENERGY = args.energy_range[0] * u.TeV
    MAX_ENERGY = args.energy_range[-1] * u.TeV

    MIN_THETA_CUT = args.theta_range[0] * u.deg
    MAX_THETA_CUT = args.theta_range[-1] * u.deg

    for input in args.input:
        abs_file_dir = os.path.abspath(input)
        for pattern in args.pattern:
            files = glob.glob(os.path.join(abs_file_dir, pattern))
            if not files:
                continue

            for file in np.sort(files):
                with pd.HDFStore(file, mode="r") as f:
                    file_keys = list(f.keys())
                    events = f["/dl2/reco"]
                    particle_type = int(events["true_shower_primary_id"][0])
                    drop_cols = ["event_id", "obs_id", "true_shower_primary_id"]
                    for k in [key for key in file_keys if key.startswith("/dl1b/")]:
                        parameters = f[k].rename(
                            lambda x: f'{k.split("/")[-1]}_' + x, axis="columns"
                        )
                        drop_cols.extend(parameters.keys())
                        events = pd.concat([events, parameters], axis=1)

                    # Apply quality cuts
                    mask = None
                    if args.quality_cuts:
                        mask = args.quality_cuts

                    if args.size_cut:
                        for s, size in enumerate(args.size_cut):
                            if mask:
                                mask += f"& tel_{s+1}_hillas_intensity > {size} "
                            else:
                                mask = f"tel_{s+1}_hillas_intensity > {size} "
                    if args.leakage_cut:
                        for l, leakage in enumerate(args.leakage_cut):
                            if mask:
                                mask += f"& tel_{l+1}_leakage_intensity_width_2 > {leakage} "
                            else:
                                mask = (
                                    f"tel_{l+1}_leakage_intensity_width_2 > {leakage} "
                                )
                    if mask:
                        events.query(mask, inplace=True)
                    events = events.drop(drop_cols, axis=1)
                    events = table.QTable.from_pandas(events)
                    for k, v in unit_mapping.items():
                        events[k] *= v

                    particles[particle_type]["events"] = table.vstack(
                        [particles[particle_type]["events"], events]
                    )

                    # Sims info
                    mc_header = f["/info/mc_header"]
                    # Check if ringwobbles then set the viewcone radius to zero
                    if (
                        particles[particle_type]["name"] == "gamma"
                        and np.around(mc_header["max_viewcone_radius"][0], decimals=1)
                        == 0.4
                    ):
                        mc_header["max_viewcone_radius"][0] = 0
                    particles[particle_type]["mc_header"] = pd.concat(
                        [particles[particle_type]["mc_header"], mc_header],
                        ignore_index=True,
                    )

    for particle_type, p in particles.items():
        log.info(f'Simulated {p["name"]} Events:')

        simulation_info = SimulatedEventsInfo(
            n_showers=int(p["mc_header"]["num_showers"].sum()),
            energy_min=u.Quantity(p["mc_header"]["energy_range_min"].min(), u.TeV),
            energy_max=u.Quantity(p["mc_header"]["energy_range_max"].max(), u.TeV),
            spectral_index=p["mc_header"]["spectral_index"][0],
            max_impact=u.Quantity(p["mc_header"]["max_scatter_range"].max(), u.m),
            viewcone=u.Quantity(p["mc_header"]["max_viewcone_radius"][0], u.deg),
        )
        p["simulation_info"] = simulation_info
        p["simulated_spectrum"] = PowerLaw.from_simulation(simulation_info, T_OBS)
        p["events"]["weight"] = MaskedColumn(
            data=calculate_event_weights(
                p["events"]["true_energy"],
                p["target_spectrum"],
                p["simulated_spectrum"],
            )
        )
        for prefix in ("true", "reco"):
            k = f"{prefix}_source_fov_offset"
            p["events"][k] = calculate_source_fov_offset(p["events"], prefix=prefix)

        # calculate theta / distance between reco and true direction of the gamma-ray
        p["events"]["theta"] = calculate_theta(
            p["events"],
            assumed_source_az=p["events"]["true_az"],
            assumed_source_alt=p["events"]["true_alt"],
        )
        log.info(simulation_info)
        log.info("")

    gammas = particles[1]["events"]
    # background table composed of both electrons and protons
    background = table.vstack([particles[0]["events"], particles[2]["events"]])

    INITIAL_GH_CUT = np.quantile(gammas["gh_score"], (1 - INITIAL_GH_CUT_EFFICENCY))
    log.info(f"Using fixed G/H cut of {INITIAL_GH_CUT} to calculate theta cuts")

    # event display uses much finer bins for the theta cut than
    # for the sensitivity
    theta_bins = add_overflow_bins(
        create_bins_per_decade(MIN_ENERGY, MAX_ENERGY, 50)
    )
    # same bins as event display uses
    sensitivity_bins = add_overflow_bins(
        create_bins_per_decade(
            MIN_ENERGY, MAX_ENERGY, bins_per_decade=5
        )
    )

    # theta cut is 68 percent containmente of the gammas
    # for now with a fixed global, unoptimized score cut
    # the cut is calculated in the same bins as the sensitivity,
    # but then interpolated to 10x the resolution.
    mask_theta_cuts = gammas["gh_score"] >= INITIAL_GH_CUT
    theta_cuts_coarse = calculate_percentile_cut(
        gammas["theta"][mask_theta_cuts],
        gammas["reco_energy"][mask_theta_cuts],
        bins=sensitivity_bins,
        min_value=MIN_THETA_CUT,
        fill_value=MAX_THETA_CUT,
        max_value=MAX_THETA_CUT,
        percentile=68,
    )

    # interpolate to 50 bins per decade
    theta_center = bin_center(theta_bins)
    inter_center = bin_center(sensitivity_bins)
    theta_cuts = table.QTable(
        {
            "low": theta_bins[:-1],
            "high": theta_bins[1:],
            "center": theta_center,
            "cut": np.interp(
                np.log10(theta_center / u.TeV),
                np.log10(inter_center / u.TeV),
                theta_cuts_coarse["cut"],
            ),
        }
    )

    log.info("Optimizing G/H separation cut for best sensitivity")
    gh_cut_efficiencies = np.arange(
        GH_CUT_EFFICIENCY_STEP,
        MAX_GH_CUT_EFFICIENCY + GH_CUT_EFFICIENCY_STEP / 2,
        GH_CUT_EFFICIENCY_STEP,
    )
    sensitivity, gh_cuts = optimize_gh_cut(
        gammas,
        background,
        reco_energy_bins=sensitivity_bins,
        gh_cut_efficiencies=gh_cut_efficiencies,
        op=operator.ge,
        theta_cuts=theta_cuts,
        alpha=ALPHA,
        background_radius=MAX_BG_RADIUS,
    )

    # now that we have the optimized gh cuts, we recalculate the theta
    # cut as 68 percent containment on the events surviving these cuts.
    log.info("Recalculating theta cut for optimized GH Cuts")
    for tab in (gammas, background):
        tab["selected_gh"] = evaluate_binned_cut(
            tab["gh_score"], tab["reco_energy"], gh_cuts, operator.ge
        )

    gammas["selected_theta"] = evaluate_binned_cut(
        gammas["theta"], gammas["reco_energy"], theta_cuts, operator.le
    )
    gammas["selected"] = gammas["selected_theta"] & gammas["selected_gh"]

    # scale relative sensitivity by Crab flux to get the flux sensitivity
    spectrum = particles[1]["target_spectrum"]
    sensitivity["flux_sensitivity"] = sensitivity["relative_sensitivity"] * spectrum(
        sensitivity["reco_energy_center"]
    )

    log.info("Calculating IRFs")
    hdus = [
        fits.PrimaryHDU(),
        fits.BinTableHDU(sensitivity, name="SENSITIVITY"),
        fits.BinTableHDU(theta_cuts, name="THETA_CUTS"),
        fits.BinTableHDU(gh_cuts, name="GH_CUTS"),
    ]

    masks = {
        "": gammas["selected"],
        "_NO_CUTS": slice(None),
        "_ONLY_GH": gammas["selected_gh"],
        "_ONLY_THETA": gammas["selected_theta"],
    }

    # binnings for the irfs
    true_energy_bins = add_overflow_bins(
        create_bins_per_decade(MIN_ENERGY, MAX_ENERGY, 10)
    )
    reco_energy_bins = add_overflow_bins(
        create_bins_per_decade(MIN_ENERGY, MAX_ENERGY, 5)
    )
    fov_offset_bins = [0, 0.5] * u.deg
    source_offset_bins = np.arange(0, 1 + 1e-4, 1e-3) * u.deg
    energy_migration_bins = np.geomspace(0.2, 5, 200)

    for label, mask in masks.items():
        effective_area = effective_area_per_energy(
            gammas[mask],
            particles[1]["simulation_info"],
            true_energy_bins=true_energy_bins,
        )
        hdus.append(
            create_aeff2d_hdu(
                effective_area[..., np.newaxis],  # add one dimension for FOV offset
                true_energy_bins,
                fov_offset_bins,
                extname="EFFECTIVE_AREA" + label,
            )
        )
        edisp = energy_dispersion(
            gammas[mask],
            true_energy_bins=true_energy_bins,
            fov_offset_bins=fov_offset_bins,
            migration_bins=energy_migration_bins,
        )
        hdus.append(
            create_energy_dispersion_hdu(
                edisp,
                true_energy_bins=true_energy_bins,
                migration_bins=energy_migration_bins,
                fov_offset_bins=fov_offset_bins,
                extname="ENERGY_DISPERSION" + label,
            )
        )

    bias_resolution = energy_bias_resolution(
        gammas[gammas["selected"]], reco_energy_bins, energy_type="reco"
    )
    ang_res = angular_resolution(
        gammas[gammas["selected_gh"]], reco_energy_bins, energy_type="reco"
    )
    psf = psf_table(
        gammas[gammas["selected_gh"]],
        true_energy_bins,
        fov_offset_bins=fov_offset_bins,
        source_offset_bins=source_offset_bins,
    )

    background_rate = background_2d(
        background[background["selected_gh"]],
        reco_energy_bins,
        fov_offset_bins=np.arange(0, 11) * u.deg,
        t_obs=T_OBS,
    )

    hdus.append(
        create_background_2d_hdu(
            background_rate,
            reco_energy_bins,
            fov_offset_bins=np.arange(0, 11) * u.deg,
        )
    )
    hdus.append(
        create_psf_table_hdu(
            psf,
            true_energy_bins,
            source_offset_bins,
            fov_offset_bins,
        )
    )
    hdus.append(
        create_rad_max_hdu(
            theta_cuts["cut"][:, np.newaxis], theta_bins, fov_offset_bins
        )
    )
    hdus.append(fits.BinTableHDU(ang_res, name="ANGULAR_RESOLUTION"))
    hdus.append(fits.BinTableHDU(bias_resolution, name="ENERGY_BIAS_RESOLUTION"))

    if not args.output.endswith(".fits.gz"):
        args.output += ".fits.gz"
    log.info(f"Writing outputfile in {args.output}")
    fits.HDUList(hdus).writeto(args.output, overwrite=True)


if __name__ == "__main__":
    main()
