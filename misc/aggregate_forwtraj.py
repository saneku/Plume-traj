import argparse
import pickle
from pathlib import Path
import sys
import numpy as np

# Add parent directory to path to import plume_forwtraj
sys.path.append(str(Path(__file__).resolve().parent.parent))


'''Aggregate and plot trajectories from multiple forward run pickles.

Usage:

python misc/aggregate_forwtraj.py \
  --pattern "./forward_run_*.pkl" \
  --height-figure plume_height_colored_aggregate.png \
  --age-figure plume_age_colored_aggregate.png \
  --seeds-vertical-figure seeds_vertical_aggregate.png \
  --map-extent 35 5 65 30
'''


from plume_forwtraj import (
    plot_trajectories_by_height,
    plot_trajectories_by_age,
    plot_seed_vertical_distribution,
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
        "--seeds-vertical-figure",
        default=None,
        help="Optional PNG for the initial vertical distribution of parcels.",
    )
    parser.add_argument(
        "--map-extent",
        nargs=4,
        type=float,
        metavar=("WEST", "SOUTH", "EAST", "NORTH"),
        help="Optional map extent override (west/south/east/north bounds) for all plots.",
    )
    return parser.parse_args()


def _pad_time(arr, target, fill):
    if arr.shape[0] >= target:
        return arr
    pad_w = [(0, target - arr.shape[0])] + [(0, 0)] * (arr.ndim - 1)
    return np.pad(arr, pad_w, mode="constant", constant_values=fill)


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

        t_agg = agg_traj["i"].shape[0]
        t_new = new_traj["i"].shape[0]
        if t_agg != t_new:
            t_max = max(t_agg, t_new)
            print(f"[diag] Time dimension mismatch: {t_agg} vs {t_new}. Padding to {t_max}.")
            agg_traj["i"] = _pad_time(agg_traj["i"], t_max, np.nan)
            agg_traj["j"] = _pad_time(agg_traj["j"], t_max, np.nan)
            agg_traj["active"] = _pad_time(agg_traj["active"], t_max, False)
            if "height_hist_m" in agg_traj:
                agg_traj["height_hist_m"] = _pad_time(agg_traj["height_hist_m"], t_max, np.nan)

            new_traj["i"] = _pad_time(new_traj["i"], t_max, np.nan)
            new_traj["j"] = _pad_time(new_traj["j"], t_max, np.nan)
            new_traj["active"] = _pad_time(new_traj["active"], t_max, False)
            if "height_hist_m" in new_traj:
                new_traj["height_hist_m"] = _pad_time(new_traj["height_hist_m"], t_max, np.nan)

        agg_traj["i"] = np.concatenate([agg_traj["i"], new_traj["i"]], axis=1)
        agg_traj["j"] = np.concatenate([agg_traj["j"], new_traj["j"]], axis=1)
        agg_traj["active"] = np.concatenate([agg_traj["active"], new_traj["active"]], axis=1)

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


def main():
    args = parse_args()
    pickle_paths = sorted(Path().glob(args.pattern))
    if not pickle_paths:
        raise FileNotFoundError("No pickle files matched the given pattern.")

    state = load_and_aggregate(pickle_paths)
    if state is None:
        raise ValueError("No valid pickle files loaded.")

    grid = state["grid"]
    trajectories = state["trajectories"]
    script_args = state.get("args", {})

    xlat = np.asarray(grid["xlat"])
    xlon = np.asarray(grid["xlon"])
    seed_bbox = script_args.get("seed_bbox")
    if seed_bbox is not None:
        seed_bbox = tuple(seed_bbox)

    fig_dpi = int(script_args.get("figure_dpi", 200))
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

    plot_trajectories_by_height(
        xlat=xlat,
        xlon=xlon,
        trajectory_i=trajectories["i"],
        trajectory_j=trajectories["j"],
        trajectory_active=trajectories["active"],
        height_hist_m=trajectories["height_hist_m"],
        out_path=args.height_figure,
        height_min=z_min,
        height_max=z_max,
        figure_dpi=fig_dpi,
        seed_bbox=seed_bbox,
        source_lat=source_lat,
        source_lon=source_lon,
        map_extent=map_extent,
    )

    plot_trajectories_by_age(
        xlat=xlat,
        xlon=xlon,
        trajectory_i=trajectories["i"],
        trajectory_j=trajectories["j"],
        trajectory_active=trajectories["active"],
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
