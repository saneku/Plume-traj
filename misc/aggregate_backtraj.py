#python aggregate_backtraj.py ../4km

import argparse
import pickle
import sys
import numpy as np
from pathlib import Path

'''Aggregate and plot trajectories from multiple back-trajectory run directories.
Example usage:

python misc/aggregate_backtraj.py ./4km --map-extent 30 5 65 30

'''

# Add parent directory to path to import plume_backtraj
sys.path.append(str(Path(__file__).resolve().parent.parent))

from plume_backtraj import (
    plot_parcel_locations,
    plot_parcel_trajectories,
    plot_parcel_age_map,
    plot_parcel_arrival_height_map,
    plot_parcel_emission_time_map,
    plot_missed_parcel_trajectories,
    plot_emission_matrix,
    _format_time_str,
    compute_height_edges,
)

def parse_args():
    parser = argparse.ArgumentParser(description="Aggregate and plot trajectories from multiple back-trajectory run directories.")
    parser.add_argument(
        "dir_prefix",
        help="Directory prefix containing the ash run directories (e.g., '../4km')."
    )
    parser.add_argument(
        "--map-extent",
        nargs=4,
        type=float,
        metavar=("WEST", "SOUTH", "EAST", "NORTH"),
        help="Optional map extent override (west/south/east/north bounds) for all plots.",
    )
    return parser.parse_args()

def load_and_aggregate(ash_dirs):
    aggregated_state = None
    total_parcels_offset = 0

    for d in ash_dirs:
        pkl_path = d / "run_ash.pkl"
        if not pkl_path.exists():
            print(f"[diag] {pkl_path} does not exist. Skipping.")
            continue
        
        print(f"[diag] Loading {pkl_path}...")
        with open(pkl_path, "rb") as fh:
            state = pickle.load(fh)
        
        if aggregated_state is None:
            aggregated_state = state
            
            # Ensure lists are converted to numpy arrays for concatenation
            traj = aggregated_state["trajectories"]
            if "indices_in_bins" in traj:
                traj["indices_in_bins"] = np.asarray(traj["indices_in_bins"])
            if "arrival_age_hours" in traj:
                traj["arrival_age_hours"] = np.asarray(traj["arrival_age_hours"])
            if "emission_time_hours" in traj and traj["emission_time_hours"] is not None:
                traj["emission_time_hours"] = np.asarray(traj["emission_time_hours"])
            if "arrival_height_m" in traj:
                traj["arrival_height_m"] = np.asarray(traj["arrival_height_m"])
            if "initial_height_m" in traj and traj["initial_height_m"] is not None:
                traj["initial_height_m"] = np.asarray(traj["initial_height_m"])
            if "final_z" in traj:
                traj["final_z"] = np.asarray(traj["final_z"])

            if "initial_parcels" in aggregated_state and aggregated_state["initial_parcels"]:
                ip = aggregated_state["initial_parcels"]
                ip["i"] = np.asarray(ip["i"])
                ip["j"] = np.asarray(ip["j"])
                ip["z_init"] = np.asarray(ip["z_init"])

            total_parcels_offset = traj["i"].shape[1]
            continue

        # Aggregate trajectories
        traj = aggregated_state["trajectories"]
        new_traj = state["trajectories"]

        # Handle potential mismatch in time dimension (axis 0)
        t_agg = traj["i"].shape[0]
        t_new = new_traj["i"].shape[0]
        if t_agg != t_new:
            t_max = max(t_agg, t_new)
            print(f"[diag] Time dimension mismatch: {t_agg} vs {t_new}. Padding to {t_max}.")
            
            def pad_t(arr, target, fill):
                if arr.shape[0] >= target: return arr
                pad_w = [(0, target - arr.shape[0])] + [(0,0)]*(arr.ndim-1)
                return np.pad(arr, pad_w, mode='constant', constant_values=fill)

            if t_agg < t_max:
                traj["i"] = pad_t(traj["i"], t_max, np.nan)
                traj["j"] = pad_t(traj["j"], t_max, np.nan)
                traj["active"] = pad_t(traj["active"], t_max, False)
                if "times" in new_traj and len(new_traj["times"]) == t_max:
                    traj["times"] = new_traj["times"]
            
            if t_new < t_max:
                new_traj["i"] = pad_t(new_traj["i"], t_max, np.nan)
                new_traj["j"] = pad_t(new_traj["j"], t_max, np.nan)
                new_traj["active"] = pad_t(new_traj["active"], t_max, False)
        
        # Concatenate arrays along axis 1 (parcels) or axis 0 (1D arrays)
        traj["i"] = np.concatenate([traj["i"], new_traj["i"]], axis=1)
        traj["j"] = np.concatenate([traj["j"], new_traj["j"]], axis=1)
        traj["active"] = np.concatenate([traj["active"], new_traj["active"]], axis=1)
        traj["arrived_mask"] = np.concatenate([traj["arrived_mask"], new_traj["arrived_mask"]])
        
        def _agg_dense(key):
            val_new = new_traj.get(key)
            val_curr = traj.get(key)
            
            if val_new is None and val_curr is None:
                return

            # If new is present but current is None
            if val_curr is None:
                val_new = np.asarray(val_new)
                if total_parcels_offset > 0:
                    # Pad missing previous data with NaNs
                    shape_curr = (total_parcels_offset,) + val_new.shape[1:]
                    val_curr_arr = np.full(shape_curr, np.nan)
                    traj[key] = np.concatenate([val_curr_arr, val_new])
                else:
                    traj[key] = val_new
                return

            # If current is present but new is None
            if val_new is None:
                n_new = new_traj["i"].shape[1]
                shape_new = (n_new,) + val_curr.shape[1:]
                val_new_arr = np.full(shape_new, np.nan)
                traj[key] = np.concatenate([val_curr, val_new_arr])
                return

            traj[key] = np.concatenate([val_curr, np.asarray(val_new)])

        def _agg_sparse(key):
            val_new = new_traj.get(key)
            val_curr = traj.get(key)
            
            # Treat None as empty
            arr_curr = np.asarray(val_curr) if val_curr is not None else np.array([])
            arr_new = np.asarray(val_new) if val_new is not None else np.array([])
            
            if arr_curr.size == 0 and arr_new.size == 0:
                return

            traj[key] = np.concatenate([arr_curr, arr_new])

        _agg_dense("initial_height_m")
        _agg_dense("final_z")
        _agg_sparse("arrival_age_hours")
        # Skip emission_time_hours aggregation to force recalculation in main
        # _agg_sparse("emission_time_hours") 
        _agg_sparse("arrival_height_m")

        # indices_in_bins: offset by the total parcels processed so far
        new_indices = new_traj.get("indices_in_bins")
        if new_indices is not None:
            new_indices = np.asarray(new_indices) + total_parcels_offset
            curr_indices = traj.get("indices_in_bins")
            if curr_indices is None:
                traj["indices_in_bins"] = new_indices
            else:
                traj["indices_in_bins"] = np.concatenate([curr_indices, new_indices])
        
        n_parcels_new = new_traj["i"].shape[1]
        total_parcels_offset += n_parcels_new

        # Aggregate initial parcels
        if "initial_parcels" in aggregated_state and aggregated_state["initial_parcels"]:
             ip = aggregated_state["initial_parcels"]
             nip = state.get("initial_parcels")
             if nip:
                 ip["i"] = np.concatenate([ip["i"], np.asarray(nip["i"])])
                 ip["j"] = np.concatenate([ip["j"], np.asarray(nip["j"])])
                 ip["z_init"] = np.concatenate([ip["z_init"], np.asarray(nip["z_init"])])

        # Aggregate emission matrix
        if "emission" in aggregated_state:
            em = aggregated_state["emission"]
            nem = state["emission"]
            em["matrix"] += nem["matrix"]
            if "mass_matrix" in em and "mass_matrix" in nem:
                if em["mass_matrix"] is not None and nem["mass_matrix"] is not None:
                    em["mass_matrix"] += nem["mass_matrix"]
            if "total_parcels" in em and "total_parcels" in nem:
                 em["total_parcels"] += nem["total_parcels"]

    # Force recalculation of emission_time_hours to ensure consistency
    if aggregated_state and "trajectories" in aggregated_state:
        aggregated_state["trajectories"].pop("emission_time_hours", None)

    return aggregated_state

def main():
    args = parse_args()
    dir_prefix = Path(args.dir_prefix)
    
    # Identify ash directories (ash5 to ash10)
    ash_types = [f"ash{i}" for i in range(5, 11)]
    ash_dirs = [dir_prefix / f"{ash_type}_run" for ash_type in ash_types]
    
    state = load_and_aggregate(ash_dirs)
    if state is None:
        print("No valid run_state.pkl files found.")
        return

    column = state["column"]["field"]
    xlat = state["grid"]["xlat"]
    xlon = state["grid"]["xlon"]
    script_args = state["args"]
    dpi = script_args.get("figure_dpi", 100)
    seed_bbox = tuple(script_args["seed_bbox"]) if script_args.get("seed_bbox") else None
    if args.map_extent is not None:
        map_extent = tuple(args.map_extent)
    else:
        map_extent = script_args.get("map_extent")
        if map_extent is not None:
            map_extent = tuple(map_extent)

    # Approximate initial-height array from trajectories
    n_parcels = state["trajectories"]["i"].shape[1]
    init_heights = state["trajectories"].get("initial_height_m")
    if init_heights is None:
        init_heights = (
            state.get("initial_parcels", {}).get("z_init")
            if state.get("initial_parcels")
            else None
        )
    if init_heights is not None:
        init_heights = np.asarray(init_heights)
    if init_heights is None or init_heights.shape[0] != n_parcels:
        init_heights = None

    print("[diag] Plotting aggregated trajectories...")
    plot_parcel_trajectories(
        column2d=column,
        xlat=xlat,
        xlon=xlon,
        trajectory_times=state["trajectories"]["times"],
        trajectory_i=state["trajectories"]["i"],
        trajectory_j=state["trajectories"]["j"],
        trajectory_active=state["trajectories"]["active"],
        arrived_flags=state["trajectories"]["arrived_mask"],
        threshold=script_args["threshold"],
        receptor_lat=script_args["receptor_lat"],
        receptor_lon=script_args["receptor_lon"],
        receptor_radius_m=script_args["receptor_radius"],
        seed_bbox=seed_bbox,
        out_path="aggregated_trajectories.png",
        colorbar_label=state["column"]["colorbar_label"],
        initial_heights=init_heights,
        z_min=script_args.get("z_min"),
        z_max=script_args.get("z_max"),
        figure_dpi=dpi,
        map_extent=map_extent,
    )

    if "initial_parcels" in state and state["initial_parcels"]:
        print("[diag] Plotting aggregated parcel locations...")
        plot_parcel_locations(
            column2d=column,
            xlat=xlat,
            xlon=xlon,
            parcels=state["initial_parcels"],
            out_path="aggregated_parcel_locations.png",
            threshold=script_args["threshold"],
            receptor_lat=script_args["receptor_lat"],
            receptor_lon=script_args["receptor_lon"],
            receptor_radius_m=script_args["receptor_radius"],
            seed_bbox=seed_bbox,
            colorbar_label=state["column"]["colorbar_label"],
            title=(
                "Aggregated Parcel seeds at "
                f"{_format_time_str(state['metadata'].get('start_time', ''))}\n"
                f"(WRF index {state['metadata']['start_time_index']})"
            ),
            figure_dpi=dpi,
            map_extent=map_extent,
        )

    print("[diag] Plotting aggregated age map...")
    plot_parcel_age_map(
        column2d=column,
        xlat=xlat,
        xlon=xlon,
        trajectory_i=state["trajectories"]["i"],
        trajectory_j=state["trajectories"]["j"],
        trajectory_active=state["trajectories"]["active"],
        parcel_indices=state["trajectories"]["indices_in_bins"],
        parcel_ages_hours=state["trajectories"]["arrival_age_hours"],
        threshold=script_args["threshold"],
        receptor_lat=script_args["receptor_lat"],
        receptor_lon=script_args["receptor_lon"],
        receptor_radius_m=script_args["receptor_radius"],
        out_path="aggregated_trajectory_ages.png",
        colorbar_label=state["column"]["colorbar_label"],
        figure_dpi=dpi,
        seed_bbox=seed_bbox,
        map_extent=map_extent,
    )

    emission_time_hours = state["trajectories"].get("emission_time_hours")
    emission_start_time = state.get("metadata", {}).get("emission_start")
    if emission_start_time is None:
        emission_start_time = np.datetime64("2025-11-23T08:30:00")
        print(f"[diag] emission_start missing in metadata, using default: {emission_start_time}")

    if emission_time_hours is None:
        arrival_age_hours = state["trajectories"].get("arrival_age_hours")
        start_time = state.get("metadata", {}).get("start_time")
        
        # Fallback if start_time is missing in metadata
        if start_time is None and "times" in state["trajectories"] and len(state["trajectories"]["times"]) > 0:
            start_time = state["trajectories"]["times"][0]
            print(f"[diag] start_time missing in metadata, using trajectories['times'][0]: {start_time}")
        
        if start_time is not None:
            start_time = np.datetime64(start_time)

        if arrival_age_hours is not None and start_time is not None:
            start_time_sec = (
                (start_time - emission_start_time) / np.timedelta64(1, "s")
            ).astype(float)
            emission_time_hours = start_time_sec / 3600.0 - np.asarray(arrival_age_hours, dtype=float)
            print("[diag] Derived emission-time data from arrival ages.")
        else:
            print("[diag] No emission-time data available.")

    if emission_time_hours is not None:
        print("[diag] Plotting aggregated emission time map...")
        plot_parcel_emission_time_map(
            column2d=column,
            xlat=xlat,
            xlon=xlon,
            trajectory_i=state["trajectories"]["i"],
            trajectory_j=state["trajectories"]["j"],
            trajectory_active=state["trajectories"]["active"],
            parcel_indices=state["trajectories"]["indices_in_bins"],
            parcel_emission_time_hours=emission_time_hours,
            threshold=script_args["threshold"],
            receptor_lat=script_args["receptor_lat"],
            receptor_lon=script_args["receptor_lon"],
            receptor_radius_m=script_args["receptor_radius"],
            out_path="aggregated_trajectory_emission_time.png",
            colorbar_label=state["column"]["colorbar_label"],
            figure_dpi=dpi,
            emission_start_time=emission_start_time,
            seed_bbox=seed_bbox,
            map_extent=map_extent,
        )
    else:
        print("[diag] No emission-time plot generated (data unavailable).")

    print("[diag] Plotting aggregated arrival height map...")
    plot_parcel_arrival_height_map(
        column2d=column,
        xlat=xlat,
        xlon=xlon,
        trajectory_i=state["trajectories"]["i"],
        trajectory_j=state["trajectories"]["j"],
        trajectory_active=state["trajectories"]["active"],
        parcel_indices=state["trajectories"]["indices_in_bins"],
        parcel_arrival_height_m=state["trajectories"]["arrival_height_m"],
        threshold=script_args["threshold"],
        receptor_lat=script_args["receptor_lat"],
        receptor_lon=script_args["receptor_lon"],
        receptor_radius_m=script_args["receptor_radius"],
        out_path="aggregated_trajectory_arrival_height.png",
        colorbar_label=state["column"]["colorbar_label"],
        figure_dpi=dpi,
        height_min=script_args.get("z_min"),
        height_max=script_args.get("z_max"),
        seed_bbox=seed_bbox,
        map_extent=map_extent,
    )

    missed_indices = np.where(~state["trajectories"]["arrived_mask"])[0]
    if missed_indices.size:
        print("[diag] Plotting aggregated missed trajectories...")
        plot_missed_parcel_trajectories(
            column2d=column,
            xlat=xlat,
            xlon=xlon,
            trajectory_i=state["trajectories"]["i"],
            trajectory_j=state["trajectories"]["j"],
            trajectory_active=state["trajectories"]["active"],
            arrived_flags=state["trajectories"]["arrived_mask"],
            initial_heights=init_heights,
            final_heights=state["trajectories"]["final_z"],
            threshold=script_args["threshold"],
            receptor_lat=script_args["receptor_lat"],
            receptor_lon=script_args["receptor_lon"],
            receptor_radius_m=script_args["receptor_radius"],
            out_path="aggregated_missed_trajectories.png",
            z_min=script_args.get("z_min"),
            z_max=script_args.get("z_max"),
            figure_dpi=dpi,
            seed_bbox=seed_bbox,
            map_extent=map_extent,
        )

    print("[diag] Plotting aggregated emission matrix...")
    plot_emission_matrix(
        emission=state["emission"]["matrix"],
        time_edges=state["emission"]["time_edges"],
        z_bins=state["emission"]["z_bins"],
        z_edges=state["emission"]["z_edges"],
        time_labels=state["emission"]["time_labels"],
        out_path="aggregated_emission_matrix.png",
        time_axis_mode=state["emission"]["time_axis_mode"],
        total_parcels=state["emission"]["total_parcels"],
        figure_dpi=dpi,
    )
    
    # Save the emission matrix to a text file
    emission_matrix = state["emission"]["matrix"]
    time_labels = state["emission"]["time_labels"]
    z_bins = state["emission"]["z_bins"]
    nz_bins = z_bins.size
    n_time_bins = len(time_labels)
    
    with open("aggregated_emission_matrix.txt", "w") as f:
        f.write("time " + " ".join(time_labels) + "\n")
        f.write("height " + " ".join(f"{z:.2f}" for z in z_bins) + "\n")
        for iz in range(nz_bins - 1, -1, -1):
            row_vals = " ".join(f"{int(round(emission_matrix[iz, jt]))}" for jt in range(n_time_bins))
            f.write(row_vals + "\n")
    print("Aggregated Time–height emission series written to 'aggregated_emission_matrix.txt'.")

    if "mass_matrix" in state["emission"] and state["emission"]["mass_matrix"] is not None:
        print("[diag] Plotting aggregated mass-weighted emission matrix...")
        plot_emission_matrix(
            emission=state["emission"]["mass_matrix"],
            time_edges=state["emission"]["time_edges"],
            z_bins=state["emission"]["z_bins"],
            z_edges=state["emission"]["z_edges"],
            time_labels=state["emission"]["time_labels"],
            out_path="aggregated_mass_matrix.png",
            time_axis_mode=state["emission"]["time_axis_mode"],
            total_parcels=None,
            colorbar_label="Parcel mass",
            figure_dpi=dpi,
        )
        
        mass_matrix = state["emission"]["mass_matrix"]
        with open("aggregated_mass_matrix.txt", "w") as f:
            f.write("time " + " ".join(time_labels) + "\n")
            f.write("height " + " ".join(f"{z:.2f}" for z in z_bins) + "\n")
            for iz in range(nz_bins - 1, -1, -1):
                row_vals = " ".join(
                    f"{mass_matrix[iz, jt]:.6e}" for jt in range(n_time_bins)
                )
                f.write(row_vals + "\n")
        print("Aggregated Mass-weighted emission series written to 'aggregated_mass_matrix.txt'.")

if __name__ == "__main__":
    main()
