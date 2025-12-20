#!/usr/bin/env python
import argparse
import warnings
import pickle
from glob import glob

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import BoundaryNorm
import numpy as np
import cartopy.crs as ccrs
from scipy.interpolate import RegularGridInterpolator

''' 
Example usage:'

python plume_forwtraj.py \
  --wrfout /scratch/ukhova/SandBox/WRF/run_hayligubbi/ERA5/ERA5_100km/wrfout_d01_2025-1* \
  --start-time 2025-11-23T08:30:00 \
  --end-time 2025-11-24T12:00:00 \
  --aer-type ash9 \
  --integration-dt 15 \
  --source-lat 13.51 \
  --source-lon 40.71 \
  --z-min 1000 \
  --z-max 23000 \
  --n-vert 30 \
  --age-figure plume_age_colored.png \
  --height-figure plume_height_colored.png \
  --seeds-vertical-figure parcel_initial_vertical_distribution.png \
  --state-pickle forward_run.pkl \
  --map-extent 35 5 65 30
  ##--seed-bbox 40.0 10.0 50.1 20.1 --n-columns 25 \
  
  
'''


from plume_backtraj import (
    _diag,
    _format_time_str,
    _get_velocity_at_time,
    _init_geo_axes,
    _plot_seed_bbox,
    TRAJECTORY_LINESTYLE,
    TRAJECTORY_LINEWIDTH,
    plot_seed_vertical_distribution,
    read_wrf_geometry_and_winds,
)
from misc.settling_velocity_data import SETTLING_VEL_MS, Z_M

TRAJECTORY_ALPHA = 1.0


def _parse_time_arg(time_str):
    if time_str is None:
        return None
    s = time_str.strip()
    if "T" not in s and "_" in s:
        s = s.replace("_", "T")
    return np.datetime64(s)


def _combine_wrf_outputs(wrfout_paths):
    grids = []
    for path in wrfout_paths:
        grids.append(read_wrf_geometry_and_winds(path))

    base = grids[0]
    xlat = base["xlat"]
    xlon = base["xlon"]
    dx_m = base["dx_m"]
    dy_m = base["dy_m"]
    area = base["area"]

    u_list = []
    v_list = []
    w_list = []
    z_list = []
    dz_list = []
    time_list = []

    for idx, grid in enumerate(grids):
        if grid["xlat"].shape != xlat.shape or grid["xlon"].shape != xlon.shape:
            raise ValueError(f"WRF grid shape mismatch in file {wrfout_paths[idx]}.")
        if not np.allclose(grid["xlat"], xlat, equal_nan=True):
            raise ValueError(f"WRF XLAT mismatch in file {wrfout_paths[idx]}.")
        if not np.allclose(grid["xlon"], xlon, equal_nan=True):
            raise ValueError(f"WRF XLONG mismatch in file {wrfout_paths[idx]}.")

        u_list.append(grid["u"])
        v_list.append(grid["v"])
        w_list.append(grid["w"])
        z_list.append(grid["z_center"])
        dz_list.append(grid["dz"])
        time_list.append(grid["times"])

    u = np.concatenate(u_list, axis=0)
    v = np.concatenate(v_list, axis=0)
    w = np.concatenate(w_list, axis=0)
    z_center = np.concatenate(z_list, axis=0)
    dz = np.concatenate(dz_list, axis=0)
    times = np.concatenate(time_list, axis=0)

    if times.dtype.kind == "M":
        diffs = np.diff(times.astype("datetime64[s]"))
        if np.any(diffs <= np.timedelta64(0, "s")):
            _diag("WRF times are not strictly increasing; check file order.")

    return dict(
        xlat=xlat,
        xlon=xlon,
        dx_m=dx_m,
        dy_m=dy_m,
        area=area,
        u=u,
        v=v,
        w=w,
        z_center=z_center,
        dz=dz,
        times=times,
    )


def _expand_wrfout_paths(wrfout_args):
    expanded = []
    for entry in wrfout_args:
        if any(ch in entry for ch in ["*", "?", "["]):
            matches = sorted(glob(entry))
            expanded.extend(matches)
        else:
            expanded.append(entry)
    expanded = [path for path in expanded if path]
    if not expanded:
        raise ValueError("No WRF output files matched the provided pattern(s).")
    return expanded


def _select_seed_columns(xlat, xlon, seed_bbox, n_columns, rng):
    if seed_bbox is None:
        return None
    lon_min, lat_min, lon_max, lat_max = seed_bbox
    if lon_min > lon_max:
        lon_min, lon_max = lon_max, lon_min
    if lat_min > lat_max:
        lat_min, lat_max = lat_max, lat_min
    mask = (
        (xlon >= lon_min)
        & (xlon <= lon_max)
        & (xlat >= lat_min)
        & (xlat <= lat_max)
    )
    jj, ii = np.where(mask)
    if jj.size == 0:
        raise ValueError("seed-bbox does not overlap the WRF grid.")
    if n_columns > jj.size:
        raise ValueError(
            f"seed-bbox contains only {jj.size} cells; cannot sample {n_columns} columns."
        )
    choice = rng.choice(jj.size, size=n_columns, replace=False)
    return np.column_stack((jj[choice], ii[choice]))


def _select_source_cell(xlat, xlon, source_lat, source_lon):
    dist2 = (xlat - source_lat) ** 2 + (xlon - source_lon) ** 2
    return np.unravel_index(np.argmin(dist2), dist2.shape)


def generate_parcels_from_source(
    z_center,
    time_index,
    seed_cells,
    n_vert,
    z_min,
    z_max,
):
    if n_vert <= 0:
        return {"j": np.array([]), "i": np.array([]), "k": np.array([]), "z_init": np.array([])}

    z_profile = z_center[time_index]
    n_columns = seed_cells.shape[0]
    n_parcels = n_columns * n_vert

    j_p = np.empty(n_parcels, dtype=float)
    i_p = np.empty(n_parcels, dtype=float)
    k_p = np.empty(n_parcels, dtype=float)
    z_init = np.empty(n_parcels, dtype=float)

    idx = 0
    for j0, i0 in seed_cells:
        z_prof = z_profile[:, j0, i0]
        k_levels = np.arange(z_prof.size, dtype=float)
        if n_vert == 1:
            z_targets = np.array([0.5 * (z_min + z_max)])
        else:
            z_targets = np.linspace(z_min, z_max, num=n_vert)
        k_targets = np.interp(z_targets, z_prof, k_levels)

        for kk in range(n_vert):
            j_p[idx] = float(j0)
            i_p[idx] = float(i0)
            k_p[idx] = k_targets[kk]
            z_init[idx] = z_targets[kk]
            idx += 1

    return {"j": j_p, "i": i_p, "k": k_p, "z_init": z_init}


def advect_parcels_forward_wrf(
    parcels,
    times,
    u,
    v,
    w,
    dz,
    z_center,
    dx_m,
    dy_m,
    xlat,
    xlon,
    start_time_index,
    end_time_index,
    integration_dt,
    settling_profile=None,
    settling_recalc_interval=300.0,
):
    u = np.asarray(u)
    v = np.asarray(v)
    w = np.asarray(w)
    dz = np.asarray(dz)
    z_center = np.asarray(z_center)
    dx_m = np.asarray(dx_m)
    dy_m = np.asarray(dy_m)
    xlat = np.asarray(xlat)
    xlon = np.asarray(xlon)

    nt, nz, ny, nx = u.shape
    if integration_dt <= 0:
        raise ValueError("integration_dt must be positive.")

    times = np.asarray(times)
    if times.dtype.kind == "M":
        t_ref = times[0]
        t_sec = (times - t_ref) / np.timedelta64(1, "s")
        t_sec = t_sec.astype(float)
    else:
        t_sec = np.asarray(times, dtype=float)

    it_start = int(np.clip(start_time_index, 0, nt - 1))
    it_end = int(np.clip(end_time_index, 0, nt - 1))
    if it_end <= it_start:
        raise ValueError("end_time must be later than start_time.")

    j_p = parcels["j"].copy()
    i_p = parcels["i"].copy()
    k_p = parcels["k"].copy()
    n_parcels = j_p.size

    active = np.ones(n_parcels, dtype=bool)
    trajectory_times = []
    trajectory_time_indices = []
    trajectory_i = []
    trajectory_j = []
    trajectory_k = []
    trajectory_active = []

    def record_trajectory(time_value, time_index):
        trajectory_times.append(time_value)
        trajectory_time_indices.append(int(time_index))
        trajectory_i.append(i_p.copy())
        trajectory_j.append(j_p.copy())
        trajectory_k.append(k_p.copy())
        trajectory_active.append(active.copy())

    record_trajectory(t_sec[it_start], it_start)

    settling_enabled = settling_profile is not None
    if settling_enabled:
        settling_heights = np.asarray(settling_profile["heights_m"], dtype=float)
        settling_velocity = np.asarray(settling_profile["velocity_ms"], dtype=float)
        if settling_heights.ndim != 1 or settling_velocity.ndim != 1:
            raise ValueError("settling_profile arrays must be 1-D.")
        if settling_heights.size != settling_velocity.size:
            raise ValueError("settling_profile heights and velocity must align.")
        settle_steps = max(1, int(round(settling_recalc_interval / integration_dt)))
        steps_until_settle_update = 0
        parcel_settle = np.zeros(n_parcels, dtype=float)
    else:
        settle_steps = None
        steps_until_settle_update = None
        parcel_settle = None

    j_coords = np.arange(ny)
    i_coords = np.arange(nx)
    k_coords = np.arange(nz)

    interp_dx = RegularGridInterpolator(
        (j_coords, i_coords), dx_m, bounds_error=False, fill_value=None
    )
    interp_dy = RegularGridInterpolator(
        (j_coords, i_coords), dy_m, bounds_error=False, fill_value=None
    )

    dt_values = []
    it_finish = it_start
    current_time_sec = t_sec[it_start]

    for it in range(it_start, it_end):
        if not active.any():
            break

        dt_total = t_sec[it + 1] - t_sec[it]
        if dt_total <= 0:
            continue

        _diag(
            "Processing time index "
            f"{it} ({_format_time_str(times[it])}) -> "
            f"{it + 1} ({_format_time_str(times[it + 1])})."
        )

        u_lo = u[it]
        v_lo = v[it]
        w_lo = w[it]
        dz_lo = dz[it]
        z_lo = z_center[it]

        u_hi = u[it + 1]
        v_hi = v[it + 1]
        w_hi = w[it + 1]
        dz_hi = dz[it + 1]
        z_hi = z_center[it + 1]

        interp_u_lo = RegularGridInterpolator(
            (k_coords, j_coords, i_coords), u_lo, bounds_error=False, fill_value=np.nan
        )
        interp_v_lo = RegularGridInterpolator(
            (k_coords, j_coords, i_coords), v_lo, bounds_error=False, fill_value=np.nan
        )
        interp_w_lo = RegularGridInterpolator(
            (k_coords, j_coords, i_coords), w_lo, bounds_error=False, fill_value=np.nan
        )
        interp_dz_lo = RegularGridInterpolator(
            (k_coords, j_coords, i_coords), dz_lo, bounds_error=False, fill_value=np.nan
        )

        interp_u_hi = RegularGridInterpolator(
            (k_coords, j_coords, i_coords), u_hi, bounds_error=False, fill_value=np.nan
        )
        interp_v_hi = RegularGridInterpolator(
            (k_coords, j_coords, i_coords), v_hi, bounds_error=False, fill_value=np.nan
        )
        interp_w_hi = RegularGridInterpolator(
            (k_coords, j_coords, i_coords), w_hi, bounds_error=False, fill_value=np.nan
        )
        interp_dz_hi = RegularGridInterpolator(
            (k_coords, j_coords, i_coords), dz_hi, bounds_error=False, fill_value=np.nan
        )
        interp_z_lo = RegularGridInterpolator(
            (k_coords, j_coords, i_coords), z_lo, bounds_error=False, fill_value=np.nan
        )
        interp_z_hi = RegularGridInterpolator(
            (k_coords, j_coords, i_coords), z_hi, bounds_error=False, fill_value=np.nan
        )

        sub_time = t_sec[it]
        dt_remaining = dt_total

        while dt_remaining > 0 and active.any():
            dt_step = min(integration_dt, dt_remaining)
            dt_values.append(dt_step)

            idxs = np.where(active)[0]
            if idxs.size == 0:
                break

            t1 = sub_time
            pos1 = np.column_stack((k_p[idxs], j_p[idxs], i_p[idxs]))

            if settling_enabled:
                steps_until_settle_update -= 1
                if steps_until_settle_update <= 0:
                    frac_hi_settle = np.clip((sub_time - t_sec[it]) / dt_total, 0.0, 1.0)
                    frac_lo_settle = 1.0 - frac_hi_settle
                    heights_now = (
                        frac_hi_settle * interp_z_hi(pos1)
                        + frac_lo_settle * interp_z_lo(pos1)
                    )
                    settle_vals_now = np.interp(
                        heights_now,
                        settling_heights,
                        settling_velocity,
                        left=settling_velocity[0],
                        right=settling_velocity[-1],
                    )
                    parcel_settle[idxs] = settle_vals_now
                    steps_until_settle_update = settle_steps
                settle_vals_current = parcel_settle[idxs]
            else:
                settle_vals_current = None

            v1, bad1 = _get_velocity_at_time(
                t1,
                pos1,
                t_sec[it],
                t_sec[it + 1],
                interp_u_lo,
                interp_u_hi,
                interp_v_lo,
                interp_v_hi,
                interp_w_lo,
                interp_w_hi,
                interp_dx,
                interp_dy,
                interp_dz_lo,
                interp_dz_hi,
                settle_vals=settle_vals_current,
            )
            k1 = dt_step * v1

            t2 = sub_time + 0.5 * dt_step
            pos2 = pos1 + 0.5 * k1
            v2, bad2 = _get_velocity_at_time(
                t2,
                pos2,
                t_sec[it],
                t_sec[it + 1],
                interp_u_lo,
                interp_u_hi,
                interp_v_lo,
                interp_v_hi,
                interp_w_lo,
                interp_w_hi,
                interp_dx,
                interp_dy,
                interp_dz_lo,
                interp_dz_hi,
                settle_vals=settle_vals_current,
            )
            k2 = dt_step * v2

            t3 = sub_time + 0.5 * dt_step
            pos3 = pos1 + 0.5 * k2
            v3, bad3 = _get_velocity_at_time(
                t3,
                pos3,
                t_sec[it],
                t_sec[it + 1],
                interp_u_lo,
                interp_u_hi,
                interp_v_lo,
                interp_v_hi,
                interp_w_lo,
                interp_w_hi,
                interp_dx,
                interp_dy,
                interp_dz_lo,
                interp_dz_hi,
                settle_vals=settle_vals_current,
            )
            k3 = dt_step * v3

            t4 = sub_time + dt_step
            pos4 = pos1 + k3
            v4, bad4 = _get_velocity_at_time(
                t4,
                pos4,
                t_sec[it],
                t_sec[it + 1],
                interp_u_lo,
                interp_u_hi,
                interp_v_lo,
                interp_v_hi,
                interp_w_lo,
                interp_w_hi,
                interp_dx,
                interp_dy,
                interp_dz_lo,
                interp_dz_hi,
                settle_vals=settle_vals_current,
            )
            k4 = dt_step * v4

            bad_any = bad1 | bad2 | bad3 | bad4
            if bad_any.any():
                active[idxs[bad_any]] = False
                good_mask = ~bad_any
                if not good_mask.any():
                    dt_remaining -= dt_step
                    sub_time += dt_step
                    current_time_sec = sub_time
                    continue

                idxs = idxs[good_mask]
                k1 = k1[good_mask]
                k2 = k2[good_mask]
                k3 = k3[good_mask]
                k4 = k4[good_mask]
                if settling_enabled:
                    settle_vals_current = settle_vals_current[good_mask]

            delta_pos = (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
            i_new = i_p[idxs] + delta_pos[:, 0]
            j_new = j_p[idxs] + delta_pos[:, 1]
            k_new = k_p[idxs] + delta_pos[:, 2]

            i_p[idxs] = i_new
            j_p[idxs] = j_new
            k_p[idxs] = k_new

            out = (
                (i_new < 0)
                | (i_new > nx - 1)
                | (j_new < 0)
                | (j_new > ny - 1)
                | (k_new < 0)
                | (k_new > nz - 1)
            )
            if out.any():
                active[idxs[out]] = False

            dt_remaining -= dt_step
            sub_time += dt_step
            current_time_sec = sub_time

        it_finish = it + 1
        time_idx = int(np.argmin(np.abs(t_sec - current_time_sec)))
        record_trajectory(current_time_sec, time_idx)

    result = dict(parcels)
    result["trajectory_times"] = np.array(trajectory_times, dtype=float)
    result["trajectory_time_indices"] = np.array(trajectory_time_indices, dtype=int)
    result["trajectory_i"] = np.stack(trajectory_i, axis=0)
    result["trajectory_j"] = np.stack(trajectory_j, axis=0)
    result["trajectory_k"] = np.stack(trajectory_k, axis=0)
    result["trajectory_active"] = np.stack(trajectory_active, axis=0)
    result["advection_start_index"] = it_start
    result["advection_finish_index"] = it_finish
    result["advection_start_time"] = times[it_start]
    result["advection_finish_time"] = times[it_finish]

    if dt_values:
        dt_stats = dict(
            min=float(np.min(dt_values)),
            max=float(np.max(dt_values)),
            mean=float(np.mean(dt_values)),
        )
    else:
        dt_stats = dict(min=float("nan"), max=float("nan"), mean=float("nan"))
    result["dt_seconds_stats"] = dt_stats
    return result


def compute_last_active_indices(trajectory_active):
    active_hist = np.asarray(trajectory_active, dtype=bool)
    last_active_time_idx = active_hist.T.sum(axis=1) - 1
    return np.clip(last_active_time_idx, 0, active_hist.shape[0] - 1)


def compute_final_heights(
    trajectory_i,
    trajectory_j,
    trajectory_k,
    last_active_time_idx,
    trajectory_time_indices,
    z_center,
):
    traj_i = np.asarray(trajectory_i)
    traj_j = np.asarray(trajectory_j)
    traj_k = np.asarray(trajectory_k)
    time_indices = np.asarray(trajectory_time_indices, dtype=int)

    n_snap, n_parcels = traj_i.shape
    last_active_time_idx = np.clip(last_active_time_idx, 0, n_snap - 1)

    final_i = traj_i[last_active_time_idx, np.arange(n_parcels)]
    final_j = traj_j[last_active_time_idx, np.arange(n_parcels)]
    final_k = traj_k[last_active_time_idx, np.arange(n_parcels)]
    final_time_idx = time_indices[last_active_time_idx]

    final_z = np.full(n_parcels, np.nan, dtype=float)
    unique_times = np.unique(final_time_idx)
    for t_idx in unique_times:
        mask = final_time_idx == t_idx
        if not mask.any():
            continue
        z_slice = np.asarray(z_center[t_idx])
        k_coords = np.arange(z_slice.shape[0])
        j_coords = np.arange(z_slice.shape[1])
        i_coords = np.arange(z_slice.shape[2])
        interp_z = RegularGridInterpolator(
            (k_coords, j_coords, i_coords), z_slice, bounds_error=False, fill_value=np.nan
        )
        pts = np.column_stack((final_k[mask], final_j[mask], final_i[mask]))
        final_z[mask] = interp_z(pts)

    return final_z


def compute_height_history(
    trajectory_i,
    trajectory_j,
    trajectory_k,
    trajectory_time_indices,
    z_center,
):
    traj_i = np.asarray(trajectory_i)
    traj_j = np.asarray(trajectory_j)
    traj_k = np.asarray(trajectory_k)
    time_indices = np.asarray(trajectory_time_indices, dtype=int)

    n_snap, n_parcels = traj_i.shape
    height_hist = np.full((n_snap, n_parcels), np.nan, dtype=float)

    unique_times = np.unique(time_indices)
    interp_cache = {}
    for t_idx in unique_times:
        z_slice = np.asarray(z_center[t_idx])
        k_coords = np.arange(z_slice.shape[0])
        j_coords = np.arange(z_slice.shape[1])
        i_coords = np.arange(z_slice.shape[2])
        interp_cache[t_idx] = RegularGridInterpolator(
            (k_coords, j_coords, i_coords), z_slice, bounds_error=False, fill_value=np.nan
        )

    for snap_idx in range(n_snap):
        t_idx = time_indices[snap_idx]
        interp_z = interp_cache[t_idx]
        pts = np.column_stack((traj_k[snap_idx], traj_j[snap_idx], traj_i[snap_idx]))
        height_hist[snap_idx] = interp_z(pts)

    return height_hist


def _plot_trajectory_map_base(xlat, xlon, map_extent=None):
    lon_min, lon_max = float(np.nanmin(xlon)), float(np.nanmax(xlon))
    lat_min, lat_max = float(np.nanmin(xlat)), float(np.nanmax(xlat))
    lon_pad = max((lon_max - lon_min) * 0.05, 0.1)
    lat_pad = max((lat_max - lat_min) * 0.05, 0.1)
    fig, ax = _init_geo_axes(
        lon_min, lon_max, lat_min, lat_max, lon_pad, lat_pad, map_extent=map_extent
    )
    gl = ax.gridlines(draw_labels=True, linewidth=0.2, color="gray", alpha=0.5, linestyle="--")
    gl.top_labels = False
    gl.right_labels = False
    return fig, ax


def _trajectory_latlon_history(xlat, xlon, trajectory_i, trajectory_j):
    traj_i = np.asarray(trajectory_i)
    traj_j = np.asarray(trajectory_j)

    ny, nx = xlat.shape
    j_coords = np.arange(ny)
    i_coords = np.arange(nx)
    interp_lat = RegularGridInterpolator(
        (j_coords, i_coords), xlat, bounds_error=False, fill_value=np.nan
    )
    interp_lon = RegularGridInterpolator(
        (j_coords, i_coords), xlon, bounds_error=False, fill_value=np.nan
    )

    lat_hist = np.empty_like(traj_j, dtype=float)
    lon_hist = np.empty_like(traj_i, dtype=float)
    for t in range(traj_j.shape[0]):
        pts = np.column_stack((traj_j[t, :], traj_i[t, :]))
        lat_hist[t, :] = interp_lat(pts)
        lon_hist[t, :] = interp_lon(pts)
    return lat_hist, lon_hist


def plot_trajectories_by_height(
    xlat,
    xlon,
    trajectory_i,
    trajectory_j,
    trajectory_active,
    height_hist_m,
    out_path,
    height_min=None,
    height_max=None,
    figure_dpi=200,
    seed_bbox=None,
    source_lat=None,
    source_lon=None,
    map_extent=None,
):
    """Plot trajectories colored by parcel height along the path."""
    heights_m = np.asarray(height_hist_m, dtype=float)
    if not np.isfinite(heights_m).any():
        print("[diag] No height history available; skipping height figure.")
        return False

    traj_i = np.asarray(trajectory_i)
    traj_j = np.asarray(trajectory_j)
    active_hist = np.asarray(trajectory_active, dtype=bool)
    lat_hist, lon_hist = _trajectory_latlon_history(xlat, xlon, traj_i, traj_j)

    fig, ax = _plot_trajectory_map_base(xlat, xlon, map_extent=map_extent)

    n_bins = 10
    data_min_m = float(np.nanmin(heights_m))
    data_max_m = float(np.nanmax(heights_m))
    if np.isclose(data_min_m, data_max_m):
        data_max_m = data_min_m + 50.0
    if (
        height_min is not None
        and height_max is not None
        and np.isfinite(height_min)
        and np.isfinite(height_max)
        and height_max > height_min
    ):
        range_min_m = float(height_min)
        range_max_m = float(height_max)
    else:
        range_min_m = data_min_m
        range_max_m = data_max_m
    if not np.isfinite(range_min_m):
        range_min_m = data_min_m
    if not np.isfinite(range_max_m) or range_max_m <= range_min_m:
        range_max_m = range_min_m + 50.0

    h_bins_m = np.linspace(range_min_m, range_max_m, n_bins + 1)
    cmap = plt.get_cmap("rainbow", n_bins)
    palette = cmap(np.arange(n_bins))

    segments = np.array([
        np.stack([lon_hist[:-1, :], lat_hist[:-1, :]], axis=2),
        np.stack([lon_hist[1:, :], lat_hist[1:, :]], axis=2),
    ]).transpose(2, 1, 0, 3)

    active_mask = (active_hist[:-1, :] & active_hist[1:, :]).T
    if active_mask.any():
        heights_seg = heights_m[:-1, :].T
        color_idx = np.digitize(heights_seg, h_bins_m) - 1
        color_idx = np.clip(color_idx, 0, n_bins - 1)
        colors_all = palette[color_idx]
        colors_to_plot = colors_all[active_mask]
        segments_to_plot = segments[active_mask]
        lc = LineCollection(
            segments_to_plot,
            colors=colors_to_plot,
            linestyle=TRAJECTORY_LINESTYLE,
            linewidth=TRAJECTORY_LINEWIDTH,
            alpha=TRAJECTORY_ALPHA,
            transform=ccrs.PlateCarree(),
        )
        ax.add_collection(lc)

    if source_lat is not None and source_lon is not None:
        ax.scatter(
            source_lon,
            source_lat,
            marker="^",
            s=80,
            c="red",
            edgecolors="black",
            linewidths=0.4,
            zorder=7,
            transform=ccrs.PlateCarree(),
        )

    _plot_seed_bbox(ax, seed_bbox)

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    min_km = float(np.nanmin(heights_m)) / 1000.0
    max_km = float(np.nanmax(heights_m)) / 1000.0
    ax.set_title(
        "Parcel trajectories coloured by height "
        f"(min={min_km:.2f} km, max={max_km:.2f} km)"
    )

    cax = fig.add_axes([0.2, 0.05, 0.6, 0.03])
    h_bins_km = h_bins_m / 1000.0
    height_centers = h_bins_km[:-1] + 0.5 * np.diff(h_bins_km)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=BoundaryNorm(h_bins_km, cmap.N))
    sm.set_array([])
    cb = plt.colorbar(
        sm,
        cax=cax,
        orientation="horizontal",
        boundaries=h_bins_km,
        ticks=height_centers,
    )
    cb.set_ticklabels([f"{val:.1f}" for val in height_centers])
    cb.set_label("Height (km)")

    fig.savefig(out_path, dpi=figure_dpi)
    plt.close(fig)
    return True


def plot_trajectories_by_age(
    xlat,
    xlon,
    trajectory_i,
    trajectory_j,
    trajectory_active,
    trajectory_times_sec,
    out_path,
    figure_dpi=200,
    seed_bbox=None,
    source_lat=None,
    source_lon=None,
    map_extent=None,
):
    """Plot trajectories colored by parcel age since release."""
    traj_i = np.asarray(trajectory_i)
    traj_j = np.asarray(trajectory_j)
    active_hist = np.asarray(trajectory_active, dtype=bool)
    lat_hist, lon_hist = _trajectory_latlon_history(xlat, xlon, traj_i, traj_j)

    traj_times = np.asarray(trajectory_times_sec, dtype=float)
    if traj_times.size < 2:
        print("[diag] Not enough trajectory times for age plot.")
        return False
    age_hours = (traj_times - traj_times[0]) / 3600.0

    fig, ax = _plot_trajectory_map_base(xlat, xlon, map_extent=map_extent)

    n_bins = 10
    age_min = float(np.nanmin(age_hours))
    age_max = float(np.nanmax(age_hours))
    if np.isclose(age_min, age_max):
        age_max = age_min + 0.1
    age_bins = np.linspace(age_min, age_max, n_bins + 1)
    cmap = plt.get_cmap("gist_ncar", n_bins)
    palette = cmap(np.arange(n_bins))

    segments = np.array([
        np.stack([lon_hist[:-1, :], lat_hist[:-1, :]], axis=2),
        np.stack([lon_hist[1:, :], lat_hist[1:, :]], axis=2),
    ]).transpose(2, 1, 0, 3)

    active_mask = (active_hist[:-1, :] & active_hist[1:, :]).T
    if active_mask.any():
        age_seg = age_hours[:-1]
        color_idx = np.digitize(age_seg, age_bins) - 1
        color_idx = np.clip(color_idx, 0, n_bins - 1)
        colors_per_step = palette[color_idx]
        colors_all = np.repeat(colors_per_step[None, :, :], traj_i.shape[1], axis=0)
        colors_to_plot = colors_all[active_mask]
        segments_to_plot = segments[active_mask]
        lc = LineCollection(
            segments_to_plot,
            colors=colors_to_plot,
            linestyle=TRAJECTORY_LINESTYLE,
            linewidth=TRAJECTORY_LINEWIDTH,
            alpha=TRAJECTORY_ALPHA,
            transform=ccrs.PlateCarree(),
        )
        ax.add_collection(lc)

    if source_lat is not None and source_lon is not None:
        ax.scatter(
            source_lon,
            source_lat,
            marker="^",
            s=80,
            c="red",
            edgecolors="black",
            linewidths=0.4,
            zorder=7,
            transform=ccrs.PlateCarree(),
        )

    _plot_seed_bbox(ax, seed_bbox)

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Parcel trajectories coloured by age since release")

    cax = fig.add_axes([0.2, 0.05, 0.6, 0.03])
    age_centers = age_bins[:-1] + 0.5 * np.diff(age_bins)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=BoundaryNorm(age_bins, cmap.N))
    sm.set_array([])
    cb = plt.colorbar(
        sm,
        cax=cax,
        orientation="horizontal",
        boundaries=age_bins,
        ticks=age_centers,
    )
    cb.set_ticklabels([f"{val:.1f} h" for val in age_centers])
    cb.set_label("Parcel age (hours)")

    fig.savefig(out_path, dpi=figure_dpi)
    plt.close(fig)
    return True


def parse_args():
    parser = argparse.ArgumentParser(
        description="Forward-advect parcels from a source column in WRF winds."
    )
    parser.add_argument(
        "--wrfout",
        nargs="+",
        required=True,
        help="One or more WRF output files (ordered in time). Wildcards are supported.",
    )
    parser.add_argument(
        "--start-time",
        required=True,
        help="UTC start time for forward trajectories (e.g., 2021-04-10T15:00:00).",
    )
    parser.add_argument(
        "--end-time",
        required=True,
        help="UTC end time for forward trajectories (e.g., 2021-04-10T21:00:00).",
    )
    parser.add_argument(
        "--source-lat",
        type=float,
        default=None,
        help="Source latitude for the release column [deg].",
    )
    parser.add_argument(
        "--source-lon",
        type=float,
        default=None,
        help="Source longitude for the release column [deg].",
    )
    parser.add_argument(
        "--seed-bbox",
        nargs=4,
        type=float,
        metavar=("LON_MIN", "LAT_MIN", "LON_MAX", "LAT_MAX"),
        help="Optional lon/lat box restricting where the source column is placed.",
    )
    parser.add_argument(
        "--n-columns",
        type=int,
        default=1,
        help="Number of random source columns to sample within --seed-bbox.",
    )
    parser.add_argument(
        "--n-vert",
        type=int,
        default=30,
        help="Number of parcels along the vertical release column.",
    )
    parser.add_argument(
        "--z-min",
        type=float,
        default=2000.0,
        help="Minimum release height [m].",
    )
    parser.add_argument(
        "--z-max",
        type=float,
        default=25000.0,
        help="Maximum release height [m].",
    )
    parser.add_argument(
        "--integration-dt",
        type=float,
        default=15.0,
        help="Forward-advection sub-step in seconds (e.g., 15).",
    )
    parser.add_argument(
        "--aer-type",
        choices=sorted(SETTLING_VEL_MS.keys()),
        default=None,
        help="Optional aerosol type for gravitational settling (keys from SETTLING_VEL_MS).",
    )
    parser.add_argument(
        "--height-figure",
        default="parcel_heights.png",
        help="Output PNG for trajectories colored by height along the path.",
    )
    parser.add_argument(
        "--age-figure",
        default="parcel_ages.png",
        help="Output PNG for trajectories colored by age since release.",
    )
    parser.add_argument(
        "--seeds-vertical-figure",
        default=None,
        help="Optional PNG for the initial vertical distribution of parcels.",
    )
    parser.add_argument(
        "--figure-dpi",
        type=int,
        default=200,
        help="DPI resolution for saved figures.",
    )
    parser.add_argument(
        "--map-extent",
        nargs=4,
        type=float,
        metavar=("WEST", "SOUTH", "EAST", "NORTH"),
        help="Optional map extent override (west/south/east/north bounds) for all plots.",
    )
    parser.add_argument(
        "--state-pickle",
        default=None,
        help="Optional path to store a pickle with inputs/results for re-plotting.",
    )
    return parser.parse_args()


def main(args):
    wrfout_paths = _expand_wrfout_paths(args.wrfout)
    start_time_dt = _parse_time_arg(args.start_time)
    end_time_dt = _parse_time_arg(args.end_time)

    if start_time_dt is None or end_time_dt is None:
        raise ValueError("Both --start-time and --end-time are required.")

    if start_time_dt >= end_time_dt:
        raise ValueError("--end-time must be later than --start-time.")

    grid = _combine_wrf_outputs(wrfout_paths)
    xlat = grid["xlat"]
    xlon = grid["xlon"]
    u = grid["u"]
    v = grid["v"]
    w = grid["w"]
    z_center = grid["z_center"]
    dz = grid["dz"]
    dx_m = grid["dx_m"]
    dy_m = grid["dy_m"]
    times = grid["times"]

    times_arr = np.asarray(times)
    if start_time_dt < times_arr[0] or start_time_dt > times_arr[-1]:
        raise ValueError("Start time is outside the WRF time range.")
    if end_time_dt < times_arr[0] or end_time_dt > times_arr[-1]:
        raise ValueError("End time is outside the WRF time range.")

    time_diffs_start = np.abs(times_arr - start_time_dt)
    it_start = int(np.argmin(time_diffs_start))
    if time_diffs_start[it_start] > np.timedelta64(1, "m"):
        warnings.warn(
            f"Provided start time {start_time_dt} is more than 1 minute away from "
            f"the closest WRF time {times_arr[it_start]}. Using the closest time."
        )

    time_diffs_end = np.abs(times_arr - end_time_dt)
    it_end = int(np.argmin(time_diffs_end))
    if time_diffs_end[it_end] > np.timedelta64(1, "m"):
        warnings.warn(
            f"Provided end time {end_time_dt} is more than 1 minute away from "
            f"the closest WRF time {times_arr[it_end]}. Using the closest time."
        )

    if it_end <= it_start:
        raise ValueError("End time index is not later than start time index.")

    model_max_height = float(np.nanmax(z_center))
    if args.z_max is not None and np.isfinite(args.z_max) and args.z_max > model_max_height:
        raise ValueError(
            f"Requested z_max={args.z_max:.1f} m exceeds model top {model_max_height:.1f} m."
        )

    rng = np.random.default_rng()
    seed_cells = _select_seed_columns(xlat, xlon, args.seed_bbox, args.n_columns, rng)
    if seed_cells is None:
        if args.source_lat is None or args.source_lon is None:
            raise ValueError("Provide --source-lat/--source-lon or --seed-bbox.")
        j0, i0 = _select_source_cell(xlat, xlon, args.source_lat, args.source_lon)
        seed_cells = np.array([[j0, i0]], dtype=int)

    parcels_init = generate_parcels_from_source(
        z_center=z_center,
        time_index=it_start,
        seed_cells=seed_cells,
        n_vert=args.n_vert,
        z_min=args.z_min,
        z_max=args.z_max,
    )

    if args.seeds_vertical_figure:
        plot_seed_vertical_distribution(
            parcels=parcels_init,
            out_path=args.seeds_vertical_figure,
            z_min=args.z_min,
            z_max=args.z_max,
            figure_dpi=max(50, int(args.figure_dpi)),
            x_coords=parcels_init["i"],
        )
        print(f"[diag] Parcel vertical distribution saved to '{args.seeds_vertical_figure}'.")

    settling_profile = None
    if args.aer_type is not None:
        settling_profile = dict(
            heights_m=np.asarray(Z_M, dtype=float),
            velocity_ms=np.asarray(SETTLING_VEL_MS[args.aer_type], dtype=float),
        )
        _diag(f"Applying settling profile for '{args.aer_type}'.")

    result = advect_parcels_forward_wrf(
        parcels=parcels_init,
        times=times,
        u=u,
        v=v,
        w=w,
        dz=dz,
        z_center=z_center,
        dx_m=dx_m,
        dy_m=dy_m,
        xlat=xlat,
        xlon=xlon,
        start_time_index=it_start,
        end_time_index=it_end,
        integration_dt=args.integration_dt,
        settling_profile=settling_profile,
    )

    print(
        "[diag] Advection window: index "
        f"{result['advection_start_index']} ({_format_time_str(result['advection_start_time'])}) -> "
        f"{result['advection_finish_index']} ({_format_time_str(result['advection_finish_time'])})."
    )
    dt_stats = result["dt_seconds_stats"]
    if np.isfinite(dt_stats["mean"]):
        print(
            "[diag] Time-step stats (s): "
            f"min={dt_stats['min']:.1f}, max={dt_stats['max']:.1f}, "
            f"mean={dt_stats['mean']:.1f}."
        )

    last_active_time_idx = compute_last_active_indices(result["trajectory_active"])
    height_hist = compute_height_history(
        trajectory_i=result["trajectory_i"],
        trajectory_j=result["trajectory_j"],
        trajectory_k=result["trajectory_k"],
        trajectory_time_indices=result["trajectory_time_indices"],
        z_center=z_center,
    )
    final_heights = height_hist[last_active_time_idx, np.arange(height_hist.shape[1])]

    fig_dpi = max(50, int(args.figure_dpi))
    map_extent = tuple(args.map_extent) if args.map_extent is not None else None
    saved_height = plot_trajectories_by_height(
        xlat=xlat,
        xlon=xlon,
        trajectory_i=result["trajectory_i"],
        trajectory_j=result["trajectory_j"],
        trajectory_active=result["trajectory_active"],
        height_hist_m=height_hist,
        out_path=args.height_figure,
        height_min=args.z_min,
        height_max=args.z_max,
        figure_dpi=fig_dpi,
        seed_bbox=args.seed_bbox,
        source_lat=args.source_lat,
        source_lon=args.source_lon,
        map_extent=map_extent,
    )
    if saved_height:
        print(f"[diag] Height-colored figure saved to '{args.height_figure}'.")

    saved_age = plot_trajectories_by_age(
        xlat=xlat,
        xlon=xlon,
        trajectory_i=result["trajectory_i"],
        trajectory_j=result["trajectory_j"],
        trajectory_active=result["trajectory_active"],
        trajectory_times_sec=result["trajectory_times"],
        out_path=args.age_figure,
        figure_dpi=fig_dpi,
        seed_bbox=args.seed_bbox,
        source_lat=args.source_lat,
        source_lon=args.source_lon,
        map_extent=map_extent,
    )
    if saved_age:
        print(f"[diag] Age-colored figure saved to '{args.age_figure}'.")

    if args.state_pickle:
        args_dict = {k: getattr(args, k) for k in vars(args)}
        payload = dict(
            args=args_dict,
            grid=dict(xlat=xlat, xlon=xlon),
            initial_parcels=parcels_init,
            trajectories=dict(
                times=result["trajectory_times"],
                time_indices=result["trajectory_time_indices"],
                i=result["trajectory_i"],
                j=result["trajectory_j"],
                k=result["trajectory_k"],
                active=result["trajectory_active"],
                final_heights_m=final_heights,
                height_hist_m=height_hist,
            ),
            metadata=dict(
                start_time=times_arr[it_start],
                end_time=times_arr[it_end],
                start_time_index=it_start,
                end_time_index=it_end,
            ),
        )
        with open(args.state_pickle, "wb") as fh:
            pickle.dump(payload, fh)
        print(f"[diag] Saved processing state to '{args.state_pickle}'.")


if __name__ == "__main__":
    main(parse_args())
