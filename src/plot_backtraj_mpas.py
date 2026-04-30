"""Replot MPAS backward trajectory state pickles."""

import pickle

import numpy as np

from .plume_mpas import (
    plot_emission_matrix,
    plot_mpas_column_and_parcels,
    plot_mpas_trajectories,
    plot_mpas_vertical_distribution,
)


def main(args):
    with open(args.pickle_file, "rb") as fh:
        state = pickle.load(fh)

    grid = state["grid"]
    column_state = state["column"]
    trajectories = state["trajectories"]
    script_args = state["args"]
    metadata = state.get("metadata", {})

    lat = np.asarray(grid["lat_deg"])
    lon = np.asarray(grid["lon_deg"])
    column = np.asarray(column_state["field"])
    traj_lon = np.asarray(trajectories["lon"])
    traj_lat = np.asarray(trajectories["lat"])
    traj_active = np.asarray(trajectories["active"])
    arrived = np.asarray(trajectories.get("arrived_mask", np.zeros(traj_lon.shape[1], dtype=bool)))
    arrived_idx = np.where(arrived)[0]
    map_extent = tuple(args.map_extent) if args.map_extent is not None else script_args.get("map_extent")
    if map_extent is not None:
        map_extent = tuple(map_extent)
    seed_bbox = tuple(script_args["seed_bbox"]) if script_args.get("seed_bbox") else None
    dpi = int(script_args.get("figure_dpi", 200))

    plot_mpas_column_and_parcels(
        column,
        lat,
        lon,
        state.get("initial_parcels", {}),
        "aggregated_parcel_locations.png",
        threshold=script_args.get("threshold"),
        receptor_lat=script_args.get("receptor_lat"),
        receptor_lon=script_args.get("receptor_lon"),
        receptor_radius_m=script_args.get("receptor_radius"),
        seed_bbox=seed_bbox,
        title="Aggregated Parcel seeds",
        colorbar_label=column_state.get("colorbar_label", "Column value"),
        figure_dpi=dpi,
        map_extent=map_extent,
    )

    init_heights = trajectories.get("initial_height_m")
    if init_heights is None:
        init_heights = state.get("initial_parcels", {}).get("z_init")
    if init_heights is None:
        init_heights = np.zeros(traj_lon.shape[1])
    plot_mpas_trajectories(
        column,
        lat,
        lon,
        traj_lon,
        traj_lat,
        traj_active,
        arrived_idx,
        np.asarray(init_heights)[arrived_idx] if arrived_idx.size else np.array([]),
        "aggregated_trajectories.png",
        threshold=script_args.get("threshold"),
        receptor_lat=script_args.get("receptor_lat"),
        receptor_lon=script_args.get("receptor_lon"),
        receptor_radius_m=script_args.get("receptor_radius"),
        seed_bbox=seed_bbox,
        title="Aggregated parcel trajectories",
        colorbar_label=column_state.get("colorbar_label", "Column value"),
        figure_dpi=dpi,
        map_extent=map_extent,
    )

    arrival_age = np.asarray(trajectories.get("arrival_age_hours", np.array([])), dtype=float)
    if arrival_age.size:
        plot_mpas_trajectories(
            column,
            lat,
            lon,
            traj_lon,
            traj_lat,
            traj_active,
            arrived_idx,
            arrival_age[arrived] if arrival_age.size == arrived.size else arrival_age,
            "aggregated_trajectory_ages.png",
            threshold=script_args.get("threshold"),
            receptor_lat=script_args.get("receptor_lat"),
            receptor_lon=script_args.get("receptor_lon"),
            receptor_radius_m=script_args.get("receptor_radius"),
            seed_bbox=seed_bbox,
            title="Aggregated trajectory ages",
            colorbar_label="Age, h",
            figure_dpi=dpi,
            map_extent=map_extent,
        )

    emission = state.get("emission", {})
    if emission.get("matrix") is not None:
        plot_emission_matrix(
            emission["matrix"],
            np.asarray(emission["time_edges"]),
            np.asarray(emission["z_bins"]),
            "aggregated_emission_matrix.png",
            figure_dpi=dpi,
        )

    if state.get("initial_parcels") and script_args.get("seeds_vertical_figure"):
        plot_mpas_vertical_distribution(
            state["initial_parcels"],
            script_args["seeds_vertical_figure"],
            script_args.get("z_min"),
            script_args.get("z_max"),
            dpi,
        )
