"""Replot MPAS forward trajectory state pickles."""

import pickle

import numpy as np

from .plume_mpas import (
    plot_mpas_deposited_parcels_by_hour,
    plot_mpas_hourly_snapshots,
    plot_mpas_parcel_trajectories,
    plot_mpas_vertical_distribution,
)


def main(args):
    with open(args.pickle_file, "rb") as fh:
        state = pickle.load(fh)

    grid = state["grid"]
    trajectories = state["trajectories"]
    script_args = state.get("args", {})
    lat = np.asarray(grid["lat_deg"])
    lon = np.asarray(grid["lon_deg"])
    traj_lon = np.asarray(trajectories["lon"])
    traj_lat = np.asarray(trajectories["lat"])
    traj_z = np.asarray(trajectories["z"])
    traj_active = np.asarray(trajectories["active"])
    traj_times = np.asarray(trajectories["times"])
    map_extent = tuple(args.map_extent) if args.map_extent is not None else script_args.get("map_extent")
    if map_extent is not None:
        map_extent = tuple(map_extent)
    dpi = max(50, int(args.figure_dpi if args.figure_dpi is not None else script_args.get("figure_dpi", 200)))

    if args.hourly_figures:
        plot_mpas_hourly_snapshots(
            lat,
            lon,
            traj_lon,
            traj_lat,
            traj_active,
            traj_z,
            trajectories.get("time_indices", np.arange(traj_lon.shape[0])),
            args.hourly_output_dir,
            figure_dpi=dpi,
            map_extent=map_extent,
        )

    if args.deposition_figure is not None:
        plot_mpas_deposited_parcels_by_hour(
            lat,
            lon,
            traj_lon,
            traj_lat,
            traj_z,
            traj_active,
            traj_times,
            args.deposition_figure,
            figure_dpi=dpi,
            map_extent=map_extent,
        )

    init_heights = trajectories.get("initial_height_m")
    if init_heights is None:
        init_heights = state.get("initial_parcels", {}).get("z_init")
    if init_heights is None:
        init_heights = np.zeros(traj_lon.shape[1])

    plot_mpas_parcel_trajectories(
        traj_lon,
        traj_lat,
        traj_active,
        np.arange(traj_lon.shape[1]),
        np.asarray(init_heights) / 1000.0,
        lat,
        lon,
        args.initial_height_figure,
        title="Aggregated trajectories colored by initial height",
        colorbar_label="Initial height (km)",
        figure_dpi=dpi,
        map_extent=map_extent,
        cmap_name="rainbow",
    )

    age_hours = np.maximum((traj_times[-1] - traj_times[0]) / np.timedelta64(1, "h"), 0.0)
    plot_mpas_parcel_trajectories(
        traj_lon,
        traj_lat,
        traj_active,
        np.arange(traj_lon.shape[1]),
        np.full(traj_lon.shape[1], float(age_hours)),
        lat,
        lon,
        args.age_figure,
        title="Aggregated trajectories colored by age",
        colorbar_label="Age (hours)",
        figure_dpi=dpi,
        map_extent=map_extent,
        cmap_name="gist_ncar",
    )

    if args.seeds_vertical_figure is not None:
        initial_parcels = state.get("initial_parcels")
        if initial_parcels is None:
            raise ValueError("Pickle does not contain initial parcels.")
        plot_mpas_vertical_distribution(
            initial_parcels,
            args.seeds_vertical_figure,
            script_args.get("z_min"),
            script_args.get("z_max"),
            dpi,
        )
