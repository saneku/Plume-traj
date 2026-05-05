import argparse
import pickle
from pathlib import Path
import sys
import numpy as np

# Add parent directory to path to import backend helpers
sys.path.append(str(Path(__file__).resolve().parent.parent))


'''Aggregate and plot trajectories from multiple forward run pickles.
Supports both WRF and MPAS run pickles.
Writes the shared aggregate figure names used by both backends where the
diagnostic is the same.

Example usage:

python misc/aggregate_forwtraj.py \
  --pattern "./forward_run_*.pkl" \
  --height-figure plume_height_colored_aggregate.png \
  --age-figure plume_age_colored_aggregate.png \
  --seeds-vertical-figure seeds_vertical_aggregate.png \
  --deposition-figure deposited_by_hour_aggregate.png \
  --hourly-output-dir ./hourly_maps_aggregate \
  --map-extent 30 5 65 30
'''


from src.plume_wrf import (
    plot_trajectories_by_height,
    plot_trajectories_by_age,
    plot_deposited_parcels_by_hour,
    plot_hourly_parcel_snapshots,
    plot_seed_vertical_distribution,
)
from src.plume_mpas import (
    plot_mpas_deposited_parcels_by_hour,
    plot_mpas_hourly_snapshots,
    plot_mpas_parcel_trajectories,
    plot_mpas_vertical_distribution,
)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Aggregate and plot trajectories from multiple forward run pickles."
    )
    parser.add_argument(
        "--pattern",
        required=True,
        help="Glob pattern for forward run pickle files.",
    )
    parser.add_argument(
        "--height-figure",
        default="plume_height_colored_aggregate.png",
        help="Output PNG for trajectories colored by height.",
    )
    parser.add_argument(
        "--age-figure",
        default="plume_age_colored_aggregate.png",
        help="Output PNG for trajectories colored by age.",
    )
    parser.add_argument(
        "--deposition-figure",
        default=None,
        help=(
            "Optional output PNG for deposited parcels only, "
            "colored by deposition hour since release."
        ),
    )
    parser.add_argument(
        "--seeds-vertical-figure",
        default=None,
        help="Optional PNG for the initial vertical distribution of parcels.",
    )
    parser.add_argument(
        "--hourly-output-dir",
        default=None,
        help=(
            "If set, save parcel-location maps at each whole hour since release to "
            "this output directory."
        ),
    )
    parser.add_argument(
        "--map-extent",
        nargs=4,
        type=float,
        metavar=("WEST", "SOUTH", "EAST", "NORTH"),
        help="Optional map extent override (west/south/east/north bounds) for all plots.",
    )
    parser.add_argument(
        "--figure-dpi",
        type=int,
        default=None,
        help="Optional DPI override for all saved figures.",
    )
    return parser.parse_args()


def _as_utc_times(state):
    """Return trajectory timestamps as datetime64[s] for one run-state."""
    traj_times = np.asarray(state["trajectories"]["times"])
    if traj_times.ndim != 1:
        raise ValueError("Trajectory times must be a 1-D array.")

    if traj_times.dtype.kind == "M":
        return traj_times.astype("datetime64[s]")

    meta = state.get("metadata", {})
    ref_time = meta.get("start_time", meta.get("advection_start_time"))
    if ref_time is None:
        raise ValueError("Cannot recover UTC timestamps: metadata.start_time is missing.")
    ref_time = np.datetime64(ref_time).astype("datetime64[s]")

    traj_sec = np.asarray(traj_times, dtype=float)
    if traj_sec.size == 0:
        return np.asarray([], dtype="datetime64[s]")
    rel_sec = np.rint(traj_sec - traj_sec[0]).astype(np.int64)
    return ref_time + rel_sec.astype("timedelta64[s]")


def _align_2d_to_master(arr, run_idx_for_master, as_bool=False):
    """Reindex a [time, parcel] array onto master timeline."""
    arr = np.asarray(arr)
    if arr.ndim != 2:
        raise ValueError("Expected a 2-D [time, parcel] array.")

    t_master = run_idx_for_master.size
    n_parcels = arr.shape[1]
    if as_bool:
        out = np.zeros((t_master, n_parcels), dtype=bool)
    else:
        if np.issubdtype(arr.dtype, np.floating):
            out = np.full((t_master, n_parcels), np.nan, dtype=arr.dtype)
        else:
            arr = arr.astype(float)
            out = np.full((t_master, n_parcels), np.nan, dtype=float)

    valid = run_idx_for_master >= 0
    if np.any(valid):
        out[valid] = arr[run_idx_for_master[valid]]
    return out


def _align_state_to_master_times(state, target, master_times):
    """Align one state's trajectory time axis to master UTC timeline."""
    traj = state["trajectories"]
    run_times = _as_utc_times(state)
    run_sec = run_times.astype("datetime64[s]").astype(np.int64)
    master_sec = master_times.astype("datetime64[s]").astype(np.int64)
    time_to_idx = {int(sec): idx for idx, sec in enumerate(run_sec)}
    run_idx_for_master = np.array([time_to_idx.get(int(sec), -1) for sec in master_sec], dtype=int)

    if target == "wrf":
        required = ("i", "j", "active")
        optional = ("k", "height_hist_m")
    else:
        required = ("lon", "lat", "z", "active")
        optional = ("height_hist_m",)

    for key in required:
        if key not in traj:
            raise KeyError(f"Trajectory key '{key}' missing in run-state.")
        traj[key] = _align_2d_to_master(
            traj[key],
            run_idx_for_master,
            as_bool=(key == "active"),
        )

    for key in optional:
        if key in traj and traj[key] is not None:
            traj[key] = _align_2d_to_master(traj[key], run_idx_for_master, as_bool=False)

    traj["times_utc"] = master_times.copy()
    if target == "wrf":
        traj["times"] = ((master_times - master_times[0]) / np.timedelta64(1, "s")).astype(float)
    else:
        traj["times"] = master_times.copy()
    traj["time_indices"] = np.arange(master_times.size, dtype=int)

    meta = state.setdefault("metadata", {})
    meta["start_time"] = master_times[0]
    meta["end_time"] = master_times[-1]


def _concat_optional_time_parcel(agg_traj, new_traj, key):
    """Concatenate optional [time, parcel] arrays, padding missing side with NaN."""
    t_master = agg_traj["active"].shape[0]
    n_agg = agg_traj["active"].shape[1]
    n_new = new_traj["active"].shape[1]
    agg_arr = agg_traj.get(key)
    new_arr = new_traj.get(key)
    if agg_arr is None and new_arr is None:
        return
    if agg_arr is None:
        agg_arr = np.full((t_master, n_agg), np.nan, dtype=float)
    if new_arr is None:
        new_arr = np.full((t_master, n_new), np.nan, dtype=float)
    agg_traj[key] = np.concatenate([np.asarray(agg_arr), np.asarray(new_arr)], axis=1)


def _concat_initial_parcels(aggregated_state, state, keys):
    if "initial_parcels" not in aggregated_state or not aggregated_state["initial_parcels"]:
        return
    new_ip = state.get("initial_parcels")
    if not new_ip:
        return
    agg_ip = aggregated_state["initial_parcels"]
    for key in keys:
        if key in agg_ip and key in new_ip and agg_ip[key] is not None and new_ip[key] is not None:
            agg_ip[key] = np.concatenate([np.asarray(agg_ip[key]), np.asarray(new_ip[key])])


def _non_deposited_parcel_mask(trajectory_k, trajectory_active, tol=1e-6):
    """
    Return parcel mask that excludes parcels deposited at the lower boundary.

    A parcel is treated as deposited if its first inactive snapshot is at/under
    k=0 (within tolerance).
    """
    if trajectory_k is None:
        return None

    traj_k = np.asarray(trajectory_k, dtype=float)
    active_hist = np.asarray(trajectory_active, dtype=bool)
    if traj_k.ndim != 2 or active_hist.ndim != 2 or traj_k.shape != active_hist.shape:
        return None

    n_parcels = traj_k.shape[1]
    keep = np.ones(n_parcels, dtype=bool)
    for p in range(n_parcels):
        inactive_idx = np.where(~active_hist[:, p])[0]
        if inactive_idx.size == 0:
            continue
        idx = int(inactive_idx[0])
        kval = traj_k[idx, p]
        if np.isfinite(kval) and kval <= tol:
            keep[p] = False
    return keep


def _load_and_aggregate_common(pickle_paths, target):
    states = []
    run_times_list = []
    for pkl_path in pickle_paths:
        print(f"[diag] Loading {pkl_path}...")
        with open(pkl_path, "rb") as fh:
            state = pickle.load(fh)
        states.append(state)
        run_times_list.append(_as_utc_times(state))

    if not states:
        return None

    master_times = np.unique(np.concatenate(run_times_list)).astype("datetime64[s]")
    print(
        "[diag] UTC alignment: "
        f"{len(states)} run(s), master timeline has {master_times.size} snapshot(s) "
        f"from {master_times[0]} to {master_times[-1]}."
    )

    for state in states:
        _align_state_to_master_times(state, target, master_times)

    aggregated_state = states[0]
    agg_traj = aggregated_state["trajectories"]
    for state in states[1:]:
        new_traj = state["trajectories"]

        if target == "wrf":
            for key in ("i", "j", "active"):
                agg_traj[key] = np.concatenate([np.asarray(agg_traj[key]), np.asarray(new_traj[key])], axis=1)
            _concat_optional_time_parcel(agg_traj, new_traj, "k")
            _concat_optional_time_parcel(agg_traj, new_traj, "height_hist_m")
            _concat_initial_parcels(aggregated_state, state, ("i", "j", "z_init"))
        else:
            for key in ("lon", "lat", "z", "active"):
                agg_traj[key] = np.concatenate([np.asarray(agg_traj[key]), np.asarray(new_traj[key])], axis=1)
            _concat_optional_time_parcel(agg_traj, new_traj, "height_hist_m")
            _concat_initial_parcels(aggregated_state, state, ("lon", "lat", "z_init", "cell"))

    aggregated_state.setdefault("metadata", {})
    aggregated_state["metadata"]["start_time"] = master_times[0]
    aggregated_state["metadata"]["end_time"] = master_times[-1]
    return aggregated_state


def load_and_aggregate(pickle_paths):
    return _load_and_aggregate_common(pickle_paths, target="wrf")


def load_and_aggregate_mpas(pickle_paths):
    return _load_and_aggregate_common(pickle_paths, target="mpas")


def main():
    args = parse_args()
    pickle_paths = sorted(Path().glob(args.pattern))
    if not pickle_paths:
        raise FileNotFoundError("No pickle files matched the given pattern.")

    with open(pickle_paths[0], "rb") as fh:
        sample_state = pickle.load(fh)
    target = sample_state.get("args", {}).get("target", "wrf")

    state = load_and_aggregate_mpas(pickle_paths) if target == "mpas" else load_and_aggregate(pickle_paths)
    if state is None:
        raise ValueError("No valid pickle files loaded.")

    grid = state["grid"]
    trajectories = state["trajectories"]
    script_args = state.get("args", {})
    seed_bbox = script_args.get("seed_bbox")
    if seed_bbox is not None:
        seed_bbox = tuple(seed_bbox)

    if args.figure_dpi is not None:
        fig_dpi = max(50, int(args.figure_dpi))
    else:
        fig_dpi = max(50, int(script_args.get("figure_dpi", 200)))
    z_min = script_args.get("z_min")
    z_max = script_args.get("z_max")
    source_lat = script_args.get("source_lat")
    source_lon = script_args.get("source_lon")
    if args.map_extent is not None:
        map_extent = tuple(args.map_extent)
    else:
        map_extent = script_args.get("map_extent")
        if map_extent is not None:
            map_extent = tuple(map_extent)

    if target == "mpas":
        lat_deg = np.asarray(grid["lat_deg"])
        lon_deg = np.asarray(grid["lon_deg"])
        traj_lon = np.asarray(trajectories["lon"])
        traj_lat = np.asarray(trajectories["lat"])
        traj_active = np.asarray(trajectories["active"])
        traj_z = np.asarray(trajectories["z"])
        traj_times = np.asarray(trajectories["times"])
        init_heights = trajectories.get("initial_height_m")
        if init_heights is None:
            init_heights = state.get("initial_parcels", {}).get("z_init")
        if init_heights is not None:
            init_heights = np.asarray(init_heights)

        if args.hourly_output_dir:
            n_hourly = plot_mpas_hourly_snapshots(
                lat_deg,
                lon_deg,
                traj_lon,
                traj_lat,
                traj_active,
                traj_z,
                traj_times,
                trajectories.get("time_indices", np.arange(traj_lon.shape[0])),
                args.hourly_output_dir,
                figure_dpi=fig_dpi,
                map_extent=map_extent,
                source_lat=script_args.get("source_lat"),
                source_lon=script_args.get("source_lon"),
                tail_enabled=True,
                tail_steps=6,
            )
            print(
                "[diag] Hourly snapshots saved: "
                f"{n_hourly} file(s) in '{args.hourly_output_dir}' "
                "using 'parcel_positions_hour.XXXX.png' naming."
            )

        if args.deposition_figure is not None:
            saved_dep = plot_mpas_deposited_parcels_by_hour(
                lat_deg,
                lon_deg,
                traj_lon,
                traj_lat,
                traj_z,
                traj_active,
                traj_times,
                args.deposition_figure,
                figure_dpi=fig_dpi,
                map_extent=map_extent,
            )
            if saved_dep:
                print(f"[diag] Deposition-hour figure saved to '{args.deposition_figure}'.")

        if args.height_figure:
            plot_mpas_parcel_trajectories(
                traj_lon,
                traj_lat,
                traj_active,
                np.arange(traj_lon.shape[1]),
                np.asarray(init_heights) / 1000.0 if init_heights is not None else np.zeros(traj_lon.shape[1]),
                lat_deg,
                lon_deg,
                args.height_figure,
                title="Aggregated trajectories colored by initial height",
                colorbar_label="Initial height (km)",
                figure_dpi=fig_dpi,
                map_extent=map_extent,
                cmap_name="rainbow",
            )

        if args.age_figure:
            ages = np.maximum((traj_times[-1] - traj_times[0]) / np.timedelta64(1, "h"), 0.0)
            plot_mpas_parcel_trajectories(
                traj_lon,
                traj_lat,
                traj_active,
                np.arange(traj_lon.shape[1]),
                np.full(traj_lon.shape[1], float(ages)),
                lat_deg,
                lon_deg,
                args.age_figure,
                title="Aggregated trajectories colored by age",
                colorbar_label="Age (hours)",
                figure_dpi=fig_dpi,
                map_extent=map_extent,
                cmap_name="gist_ncar",
            )

        if args.seeds_vertical_figure is not None:
            initial_parcels = state.get("initial_parcels")
            if initial_parcels is None:
                raise ValueError("Pickle does not contain initial parcels.")
            plot_mpas_vertical_distribution(
                parcels=initial_parcels,
                out_path=args.seeds_vertical_figure,
                z_min=z_min,
                z_max=z_max,
                figure_dpi=fig_dpi,
            )
        return

    xlat = np.asarray(grid["xlat"])
    xlon = np.asarray(grid["xlon"])
    height_hist = trajectories.get("height_hist_m")
    if height_hist is None:
        raise ValueError("Aggregated trajectories do not contain height history.")
    traj_i_all = np.asarray(trajectories["i"])
    traj_j_all = np.asarray(trajectories["j"])
    traj_active_all = np.asarray(trajectories["active"])
    traj_k_all = trajectories.get("k")
    if traj_k_all is not None:
        traj_k_all = np.asarray(traj_k_all)
    height_hist_all = np.asarray(height_hist)

    if args.hourly_output_dir:
        traj_times_utc = None
        start_time_utc = state.get("metadata", {}).get("start_time")
        if start_time_utc is not None:
            try:
                t0 = np.datetime64(start_time_utc).astype("datetime64[s]")
                traj_seconds = np.rint(np.asarray(trajectories["times"], dtype=float)).astype(
                    np.int64
                )
                traj_times_utc = t0 + traj_seconds.astype("timedelta64[s]")
            except Exception:
                traj_times_utc = None

        n_hourly = plot_hourly_parcel_snapshots(
            xlat=xlat,
            xlon=xlon,
            trajectory_i=traj_i_all,
            trajectory_j=traj_j_all,
            trajectory_active=traj_active_all,
            trajectory_k=traj_k_all,
            trajectory_times_sec=trajectories["times"],
            out_dir=args.hourly_output_dir,
            height_hist_m=height_hist_all,
            figure_dpi=fig_dpi,
            seed_bbox=seed_bbox,
            source_lat=source_lat,
            source_lon=source_lon,
            map_extent=map_extent,
            trajectory_times_utc=traj_times_utc,
            tail_enabled=True,
            tail_steps=6,
        )
        print(
            "[diag] Hourly snapshots saved: "
            f"{n_hourly} file(s) in '{args.hourly_output_dir}' "
            "using 'parcel_positions_hour.XXXX.png' naming."
        )

    if args.deposition_figure is not None:
        saved_dep = plot_deposited_parcels_by_hour(
            xlat=xlat,
            xlon=xlon,
            trajectory_i=traj_i_all,
            trajectory_j=traj_j_all,
            trajectory_active=traj_active_all,
            trajectory_k=traj_k_all,
            trajectory_times_sec=trajectories["times"],
            out_path=args.deposition_figure,
            figure_dpi=fig_dpi,
            seed_bbox=seed_bbox,
            source_lat=source_lat,
            source_lon=source_lon,
            map_extent=map_extent,
        )
        if saved_dep:
            print(f"[diag] Deposition-hour figure saved to '{args.deposition_figure}'.")

    keep_non_dep = _non_deposited_parcel_mask(traj_k_all, traj_active_all)
    if keep_non_dep is None:
        i_plot = traj_i_all
        j_plot = traj_j_all
        active_plot = traj_active_all
        k_plot = traj_k_all
        height_plot = height_hist_all
    else:
        n_removed = int((~keep_non_dep).sum())
        n_total = int(keep_non_dep.size)
        print(
            "[diag] Excluding deposited parcels from aggregate age/height plots: "
            f"{n_removed}/{n_total} removed."
        )
        i_plot = traj_i_all[:, keep_non_dep]
        j_plot = traj_j_all[:, keep_non_dep]
        active_plot = traj_active_all[:, keep_non_dep]
        height_plot = height_hist_all[:, keep_non_dep]
        k_plot = traj_k_all[:, keep_non_dep] if traj_k_all is not None else None

    if i_plot.shape[1] == 0:
        print("[diag] No non-deposited parcels remain; skipping aggregate age/height plots.")
    else:
        plot_trajectories_by_height(
            xlat=xlat,
            xlon=xlon,
            trajectory_i=i_plot,
            trajectory_j=j_plot,
            trajectory_active=active_plot,
            trajectory_k=k_plot,
            height_hist_m=height_plot,
            out_path=args.height_figure,
            height_min=None,
            height_max=None,
            figure_dpi=fig_dpi,
            seed_bbox=seed_bbox,
            source_lat=source_lat,
            source_lon=source_lon,
            map_extent=map_extent,
        )

        plot_trajectories_by_age(
            xlat=xlat,
            xlon=xlon,
            trajectory_i=i_plot,
            trajectory_j=j_plot,
            trajectory_active=active_plot,
            trajectory_k=k_plot,
            trajectory_times_sec=trajectories["times"],
            out_path=args.age_figure,
            figure_dpi=fig_dpi,
            seed_bbox=seed_bbox,
            source_lat=source_lat,
            source_lon=source_lon,
            map_extent=map_extent,
        )

    if args.seeds_vertical_figure is not None:
        initial_parcels = state.get("initial_parcels")
        if initial_parcels is None:
            raise ValueError("Pickle does not contain initial parcels.")
        plot_seed_vertical_distribution(
            parcels=initial_parcels,
            out_path=args.seeds_vertical_figure,
            z_min=z_min,
            z_max=z_max,
            figure_dpi=fig_dpi,
            x_coords=initial_parcels.get("i"),
        )


if __name__ == "__main__":
    main()
