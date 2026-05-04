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


def _pad_time(arr, target, fill):
    if arr.shape[0] >= target:
        return arr
    pad_w = [(0, target - arr.shape[0])] + [(0, 0)] * (arr.ndim - 1)
    return np.pad(arr, pad_w, mode="constant", constant_values=fill)


def _pad_time_with_last(arr, target):
    """
    Pad time axis by repeating the final available snapshot.
    """
    if arr.shape[0] >= target:
        return arr
    n_add = target - arr.shape[0]
    tail = np.repeat(arr[-1:, ...], n_add, axis=0)
    return np.concatenate([arr, tail], axis=0)


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


def load_and_aggregate(pickle_paths):
    aggregated_state = None

    for pkl_path in pickle_paths:
        print(f"[diag] Loading {pkl_path}...")
        with open(pkl_path, "rb") as fh:
            state = pickle.load(fh)

        if aggregated_state is None:
            aggregated_state = state
            return_state = aggregated_state
            traj = return_state["trajectories"]
            traj["i"] = np.asarray(traj["i"])
            traj["j"] = np.asarray(traj["j"])
            traj["active"] = np.asarray(traj["active"])
            if "k" in traj:
                traj["k"] = np.asarray(traj["k"])
            if "height_hist_m" in traj:
                traj["height_hist_m"] = np.asarray(traj["height_hist_m"])
            if "times" in traj:
                traj["times"] = np.asarray(traj["times"])
            if "initial_parcels" in return_state and return_state["initial_parcels"]:
                ip = return_state["initial_parcels"]
                ip["i"] = np.asarray(ip["i"])
                ip["j"] = np.asarray(ip["j"])
                if "z_init" in ip:
                    ip["z_init"] = np.asarray(ip["z_init"])
            continue

        agg_traj = aggregated_state["trajectories"]
        new_traj = state["trajectories"]
        new_traj["i"] = np.asarray(new_traj["i"])
        new_traj["j"] = np.asarray(new_traj["j"])
        new_traj["active"] = np.asarray(new_traj["active"])
        if "k" in new_traj:
            new_traj["k"] = np.asarray(new_traj["k"])
        if "height_hist_m" in new_traj:
            new_traj["height_hist_m"] = np.asarray(new_traj["height_hist_m"])

        t_agg = agg_traj["i"].shape[0]
        t_new = new_traj["i"].shape[0]
        if t_agg != t_new:
            t_max = max(t_agg, t_new)
            print(f"[diag] Time dimension mismatch: {t_agg} vs {t_new}. Padding to {t_max}.")
            # Keep final parcel position/state visible after a shorter run ends.
            agg_traj["i"] = _pad_time_with_last(agg_traj["i"], t_max)
            agg_traj["j"] = _pad_time_with_last(agg_traj["j"], t_max)
            agg_traj["active"] = _pad_time(agg_traj["active"], t_max, False)
            if "k" in agg_traj:
                agg_traj["k"] = _pad_time_with_last(agg_traj["k"], t_max)
            if "height_hist_m" in agg_traj:
                agg_traj["height_hist_m"] = _pad_time_with_last(
                    agg_traj["height_hist_m"], t_max
                )

            new_traj["i"] = _pad_time_with_last(new_traj["i"], t_max)
            new_traj["j"] = _pad_time_with_last(new_traj["j"], t_max)
            new_traj["active"] = _pad_time(new_traj["active"], t_max, False)
            if "k" in new_traj:
                new_traj["k"] = _pad_time_with_last(new_traj["k"], t_max)
            if "height_hist_m" in new_traj:
                new_traj["height_hist_m"] = _pad_time_with_last(
                    new_traj["height_hist_m"], t_max
                )

        agg_traj["i"] = np.concatenate([agg_traj["i"], new_traj["i"]], axis=1)
        agg_traj["j"] = np.concatenate([agg_traj["j"], new_traj["j"]], axis=1)
        agg_traj["active"] = np.concatenate([agg_traj["active"], new_traj["active"]], axis=1)
        if "k" in agg_traj and "k" in new_traj:
            agg_traj["k"] = np.concatenate([agg_traj["k"], new_traj["k"]], axis=1)
        elif "k" in agg_traj and "k" not in new_traj:
            # Keep shape consistent when mixing old/new pickles by padding missing k.
            missing_k = np.full(new_traj["i"].shape, np.nan, dtype=float)
            agg_traj["k"] = np.concatenate([agg_traj["k"], missing_k], axis=1)
        elif "k" not in agg_traj and "k" in new_traj:
            # Backfill previous parcels with NaN k if earlier pickles had no k.
            old_cols = agg_traj["i"].shape[1] - new_traj["i"].shape[1]
            backfill_k = np.full((agg_traj["i"].shape[0], old_cols), np.nan, dtype=float)
            agg_traj["k"] = np.concatenate([backfill_k, new_traj["k"]], axis=1)

        if "height_hist_m" in agg_traj and "height_hist_m" in new_traj:
            agg_traj["height_hist_m"] = np.concatenate(
                [agg_traj["height_hist_m"], new_traj["height_hist_m"]], axis=1
            )

        if "initial_parcels" in aggregated_state and aggregated_state["initial_parcels"]:
            ip = aggregated_state["initial_parcels"]
            nip = state.get("initial_parcels")
            if nip:
                ip["i"] = np.concatenate([ip["i"], np.asarray(nip["i"])])
                ip["j"] = np.concatenate([ip["j"], np.asarray(nip["j"])])
                if "z_init" in ip and "z_init" in nip:
                    ip["z_init"] = np.concatenate([ip["z_init"], np.asarray(nip["z_init"])])

    return aggregated_state


def load_and_aggregate_mpas(pickle_paths):
    aggregated_state = None

    for pkl_path in pickle_paths:
        print(f"[diag] Loading {pkl_path}...")
        with open(pkl_path, "rb") as fh:
            state = pickle.load(fh)

        if aggregated_state is None:
            aggregated_state = state
            traj = aggregated_state["trajectories"]
            for key in ("lon", "lat", "z", "active", "times", "time_indices", "height_hist_m"):
                if key in traj and traj[key] is not None:
                    traj[key] = np.asarray(traj[key])
            if "initial_parcels" in aggregated_state and aggregated_state["initial_parcels"]:
                ip = aggregated_state["initial_parcels"]
                for key in ("lon", "lat", "z_init", "cell"):
                    if key in ip and ip[key] is not None:
                        ip[key] = np.asarray(ip[key])
            continue

        agg_traj = aggregated_state["trajectories"]
        new_traj = state["trajectories"]
        for key in ("lon", "lat", "z", "active", "times", "time_indices", "height_hist_m"):
            if key in new_traj and new_traj[key] is not None:
                new_traj[key] = np.asarray(new_traj[key])

        t_agg = agg_traj["lon"].shape[0]
        t_new = new_traj["lon"].shape[0]
        if t_agg != t_new:
            t_max = max(t_agg, t_new)
            print(f"[diag] Time dimension mismatch: {t_agg} vs {t_new}. Padding to {t_max}.")
            agg_traj["lon"] = _pad_time_with_last(agg_traj["lon"], t_max)
            agg_traj["lat"] = _pad_time_with_last(agg_traj["lat"], t_max)
            agg_traj["z"] = _pad_time_with_last(agg_traj["z"], t_max)
            agg_traj["active"] = _pad_time(agg_traj["active"], t_max, False)

            new_traj["lon"] = _pad_time_with_last(new_traj["lon"], t_max)
            new_traj["lat"] = _pad_time_with_last(new_traj["lat"], t_max)
            new_traj["z"] = _pad_time_with_last(new_traj["z"], t_max)
            new_traj["active"] = _pad_time(new_traj["active"], t_max, False)

        agg_traj["lon"] = np.concatenate([agg_traj["lon"], new_traj["lon"]], axis=1)
        agg_traj["lat"] = np.concatenate([agg_traj["lat"], new_traj["lat"]], axis=1)
        agg_traj["z"] = np.concatenate([agg_traj["z"], new_traj["z"]], axis=1)
        agg_traj["active"] = np.concatenate([agg_traj["active"], new_traj["active"]], axis=1)

        for key in ("final_heights_m", "height_hist_m"):
            if key in agg_traj and key in new_traj and agg_traj[key] is not None and new_traj[key] is not None:
                agg_traj[key] = np.concatenate([np.asarray(agg_traj[key]), np.asarray(new_traj[key])], axis=0 if np.asarray(agg_traj[key]).ndim == 1 else 1)

        if "initial_parcels" in aggregated_state and aggregated_state["initial_parcels"]:
            ip = aggregated_state["initial_parcels"]
            nip = state.get("initial_parcels")
            if nip:
                for key in ("lon", "lat", "z_init"):
                    if key in ip and key in nip:
                        ip[key] = np.concatenate([np.asarray(ip[key]), np.asarray(nip[key])])

    return aggregated_state


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
