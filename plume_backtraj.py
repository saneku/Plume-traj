#!/usr/bin/env python
import argparse
import warnings
from pathlib import Path
import pickle
warnings.filterwarnings("ignore", category=DeprecationWarning, message=r".*`np\.bool` is a deprecated alias.*")

import matplotlib
matplotlib.use('Agg')
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from netCDF4 import Dataset
from scipy.interpolate import RegularGridInterpolator
#from datetime import datetime

from settling_velocity_data import SETTLING_VEL_MS, Z_M



def _format_time_str(val):
    arr = np.asarray(val)
    if arr.dtype.kind == "M":
        return str(val)
    return f"{val}"


def _diag(msg: str) -> None:
    """Lightweight diagnostic logger."""
    print(f"[diag] {msg}")


PARCEL_MARKER_SIZE = 3.0
PARCEL_MARKER_ALPHA = 0.5
PARCEL_MARKER_EDGE = (0.78, 0.78, 0.78)
PARCEL_MARKER_LINEWIDTH = 0.2
TRAJECTORY_LINEWIDTH = 0.6
TRAJECTORY_LINESTYLE = "-"
TRAJECTORY_ALPHA = 0.5

'''
nohup 
WRFOUT='/scratch/ukhova/SandBox/WRF/run_hayligubbi/ERA5_4km/wrfout_d01_2025-11-23_06:00:00'
COLUMN='/lustre2/project/k10022/ukhova/Volcano/Hayli_Gubbi/operRSmerged_SO2_/sulfurdioxide_total_vertical_column_15km/4km/merged_sulfurdioxide_total_vertical_column_15km_2025-Nov-24.nc'


#SO2

outdir="./so2_run"
mkdir -p "$outdir"
python plume_backtraj.py \
 --wrfout "$WRFOUT" \
 --start-time '2025-11-24T10:00:00' \
 --column "$COLUMN" --integration-dt 15\
 --so2-efolding-days 35 \
 --column-var 'sulfurdioxide_total_vertical_column_15km' --column-coef 2242.95 --threshold 0.1 --colorbar-label 'SO2, DU'\
 --n-columns 3000 --n-vert 30 --parcel-radius 10000 --z-min 1000 --z-max 30000 \
 --emission-start '2025-11-23T08:30:00' --emission-end '2025-11-24T10:00:00' \
 --receptor-lat 13.51 --receptor-lon 40.71 \
 --receptor-radius 20000 --receptor-min-h 1000 --receptor-max-h 30000 \
 --arrival-bin-minutes 30 \
 --output-txt "$outdir/so2_emission_time_height.txt" --output-figure "$outdir/so2_emission_time_height.png" \
 --trajectory-figure "$outdir/so2_trajectories.png" --trajectory-age "$outdir/so2_trajectory_ages.png"  \
 --trajectory-emission-time-figure "$outdir/so2_trajectory_emission_time.png" \
 --seeds-figure "$outdir/parcel_initial_locations.png" \
 --mass-figure "$outdir/mass_matrix.png" --mass-output-txt "$outdir/mass_emission_time_height.txt" \
 --trajectory-arrival-height-figure "$outdir/so2_trajectory_arrival_heights.png" \
 --missed-trajectory-figure "$outdir/so2_missed_trajectories.png" \
 --figure-dpi 300 --state-pickle "$outdir/run_so2.pkl"


#Aerosols
WRFOUT='/scratch/ukhova/SandBox/WRF/run_hayligubbi/ERA5_4km/wrfout_d01_2025-11-23_06:00:00'
COLUMN='/lustre2/project/k10022/ukhova/Volcano/Hayli_Gubbi/operRSmerged_SO2_/AOD_AI_HEIGHT/4km/merged_aerosol_index_354_388_2025-NOV-24.nc'

for aer in sulf ash10 ash9 ash8 ash7 ash6; do
    outdir="./${aer}_run"
    mkdir -p "$outdir"
    echo " "
    python plume_backtraj.py \
        --wrfout "$WRFOUT" \
        --aer-type "$aer" \
        --start-time '2025-11-24T10:00:00' \
        --column "$COLUMN" --integration-dt 15 \
        --column-var 'aerosol_index_354_388' --column-coef 1 --threshold 0.10 --colorbar-label 'Aerosol Index' \
        --n-columns 50 --n-vert 30 --parcel-radius 10000 --z-min 1000 --z-max 30000 \
        --emission-start '2025-11-23T08:30:00' --emission-end   '2025-11-24T10:00:00' \
        --receptor-lat 13.51 --receptor-lon 40.71 \
        --receptor-radius 10000 --receptor-min-h 1000 --receptor-max-h 30000 \
        --arrival-bin-minutes 30 \
        --output-txt              "$outdir/emission_time_height.txt" \
        --output-figure           "$outdir/emission_time_height.png" \
        --trajectory-figure       "$outdir/trajectories.png" \
        --trajectory-age          "$outdir/trajectory_ages.png" \
        --trajectory-emission-time-figure "$outdir/so2_trajectory_emission_time.png" \
        --seeds-figure "$outdir/parcel_initial_locations.png" \
        --mass-figure "$outdir/mass_matrix.png" --mass-output-txt "$outdir/mass_emission_time_height.txt" \
        --trajectory-arrival-height-figure "$outdir/trajectory_arrival_heights.png" \
        --missed-trajectory-figure        "$outdir/missed_trajectories.png" \
        --figure-dpi 300 --state-pickle            "$outdir/run_state.pkl"
done


#todo: add so2 oxidation in reverse direction

 --aer-type can be 'sulf', 'ash1', … 'ash10'
 --seed-bbox 37 9 43 11
 >& run.log &

# --hourly-figures
#--seeds-vertical-figure "$outdir/parcel_initial_vertical_distribution.png" \

1 mol/m² ≈ 6.022e23 / 2.687e20 ≈ 2243 DU
'''

PLATE_CARREE = ccrs.PlateCarree()


def _init_geo_axes(lon_min, lon_max, lat_min, lat_max, lon_pad, lat_pad, figsize=(10, 8)):
    """Initialize a Cartopy PlateCarree map with coastlines, borders, and gridlines."""
    fig, ax = plt.subplots(
        figsize=figsize,
        subplot_kw={"projection": PLATE_CARREE},
    )
    ax.set_extent(
        [lon_min - lon_pad, lon_max + lon_pad, lat_min - lat_pad, lat_max + lat_pad],
        crs=PLATE_CARREE,
    )
    ax.coastlines(resolution="50m", linewidth=0.6, color="gray")
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.4, edgecolor="gray")
    gl = ax.gridlines(
        draw_labels=True,
        linewidth=0.3,
        color="gray",
        alpha=0.4,
        linestyle="--",
    )
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {"size": 10}
    gl.ylabel_style = {"size": 10}
    return fig, ax


def read_wrf_geometry_and_winds(wrfout_path):
    """
    Read geometry and winds from a WRF output file.

    Returns
    -------
    dict with keys:
      xlat, xlon : 2D (south_north, west_east)
      dx_m, dy_m : 2D (south_north, west_east) physical grid spacing [m]
      area       : 2D (south_north, west_east) grid cell area [m2]
      u, v, w    : 4D (time, z, south_north, west_east) destaggered winds [m/s]
      z_center   : 4D (time, z, south_north, west_east) cell center heights [m]
      dz         : 4D (time, z, south_north, west_east) layer thickness [m]
      times      : 1D np.datetime64 array (time,)
    """
    _diag(f"Opening WRF output '{wrfout_path}'.")
    ds = Dataset(wrfout_path, "r")

    # Horizontal grid / map factors
    xlat = ds.variables["XLAT"][0, :, :]
    xlon = ds.variables["XLONG"][0, :, :]
    _diag(f"Loaded horizontal grid: shape={xlat.shape}.")
    mapfac_mx = ds.variables["MAPFAC_MX"][0, :, :]
    mapfac_my = ds.variables["MAPFAC_MY"][0, :, :]
    dx = getattr(ds, "DX")
    dy = getattr(ds, "DY")

    dx_m = dx / mapfac_mx
    dy_m = dy / mapfac_my
    area = dx_m * dy_m
    _diag(
        f"Computed physical spacing: dx range {dx_m.min():.1f}–{dx_m.max():.1f} m, "
        f"dy range {dy_m.min():.1f}–{dy_m.max():.1f} m."
    )

    # Winds (destaggered to mass points)
    U = ds.variables["U"][:]  # (time, z, y, x_stag)
    V = ds.variables["V"][:]  # (time, z, y_stag, x)
    W = ds.variables["W"][:]  # (time, z_stag, y, x)

    u = 0.5 * (U[:, :, :, :-1] + U[:, :, :, 1:])
    v = 0.5 * (V[:, :, :-1, :] + V[:, :, 1:, :])
    w = 0.5 * (W[:, :-1, :, :] + W[:, 1:, :, :])
    _diag(
        f"Destaggered winds to mass grid: u/v/w shape {u.shape} (time, z, y, x)."
    )

    # Vertical coordinate from PH + PHB
    PH = ds.variables["PH"][:]   # (time, z_stag, y, x)
    PHB = ds.variables["PHB"][:]
    z_stag = (PH + PHB) / 9.81   # geopotential -> m

    z_center = 0.5 * (z_stag[:, :-1, :, :] + z_stag[:, 1:, :, :])  # (time, z, y, x)
    dz = z_stag[:, 1:, :, :] - z_stag[:, :-1, :, :]
    _diag(
        f"Computed vertical centers/thicknesses: nz={z_center.shape[1]} levels, "
        f"height span {z_center.min():.0f}–{z_center.max():.0f} m."
    )

    # Time
    tvar = ds.variables["Times"]  # char array (time, 19)
    times = []
    for it in range(tvar.shape[0]):
        t_str = tvar[it].tobytes().decode("ascii").strip()
        t_iso = t_str.replace("_", "T")
        times.append(np.datetime64(t_iso))
    times = np.array(times)
    _diag(
        f"WRF time axis: {times.size} steps from "
        f"{_format_time_str(times[0])} to {_format_time_str(times[-1])}."
    )

    ds.close()

    return dict(
        xlat=xlat,xlon=xlon,dx_m=dx_m,dy_m=dy_m,
        area=area,u=u,v=v,w=w,
        z_center=z_center,dz=dz,times=times)


def generate_parcels_from_column_wrf(
    column2d,
    xlat,
    xlon,
    z_profile_3d,
    time_index,
    threshold,
    n_columns,
    n_vert,
    z_min=2000.0,
    z_max=25000.0,
    rng=None,
    seed_bbox=None,
):
    """
    Sample random plume columns and create vertical parcel lines.

    column2d : 2D (y, x) column loading [e.g. kg/m2]
    xlat,xlon: 2D (y, x) grid cell center coordinates
    z_profile_3d : 4D (time, z, y, x) cell center heights [m]
    time_index   : int, which time index in z_profile_3d to use
    threshold    : float, plume mask threshold on column2d
    n_columns    : number of random plume columns
    n_vert       : parcels per column
    z_min,z_max  : initial vertical range [m]
    """
    if rng is None:
        rng = np.random.default_rng()

    column2d = np.asarray(column2d)

    mask = column2d >= threshold
    if seed_bbox is not None:
        lon_min, lat_min, lon_max, lat_max = seed_bbox
        if lon_min > lon_max:
            lon_min, lon_max = lon_max, lon_min
        if lat_min > lat_max:
            lat_min, lat_max = lat_max, lat_min
        bbox_mask = (xlon >= lon_min) & (xlon <= lon_max) & (xlat >= lat_min) & (xlat <= lat_max)
        mask &= bbox_mask
        _diag(
            "Restricting seed region to box "
            f"lon[{lon_min:.3f},{lon_max:.3f}], lat[{lat_min:.3f},{lat_max:.3f}]."
        )
    plume_cells = np.argwhere(mask)  # list of (j, i)

    if plume_cells.size == 0:
        raise ValueError("No grid cells above threshold; plume is empty.")

    ny, nx = column2d.shape
    _, nz, _, _ = z_profile_3d.shape

    n_available = plume_cells.shape[0]
    if n_columns > n_available:
        chosen_idx = rng.integers(0, n_available, size=n_columns)
    else:
        chosen_idx = rng.choice(n_available, size=n_columns, replace=False)

    n_parcels = n_columns * n_vert
    j_p = np.empty(n_parcels, dtype=float)
    i_p = np.empty(n_parcels, dtype=float)
    k_p = np.empty(n_parcels, dtype=float)
    z_init = np.empty(n_parcels, dtype=float)

    z_t = z_profile_3d[time_index]  # (z, y, x)

    idx = 0
    for idx_col in chosen_idx:
        j0, i0 = plume_cells[idx_col]
        z_prof = z_t[:, j0, i0]  # 1D (z,)
        k_levels = np.arange(z_prof.size, dtype=float)

        z_targets = rng.uniform(z_min, z_max, size=n_vert)
        z_targets.sort()
        k_targets = np.interp(z_targets, z_prof, k_levels)

        for kk in range(n_vert):
            j_p[idx] = float(j0)
            i_p[idx] = float(i0)
            k_p[idx] = k_targets[kk]
            z_init[idx] = z_targets[kk]
            idx += 1

    return {"j": j_p, "i": i_p, "k": k_p, "z_init": z_init}


def _get_velocity_at_time(
    t_sec,
    pts_3d,
    t_sec_lo,
    t_sec_hi,
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
    settle_vals=None,
):
    """Interpolate velocity components at a specific time and for multiple points."""
    dt_total = t_sec_hi - t_sec_lo
    if dt_total <= 0:
        dt_total = 1.0 # Avoid division by zero if t_sec_hi == t_sec_lo

    frac_hi = np.clip((t_sec - t_sec_lo) / dt_total, 0.0, 1.0)
    frac_lo = 1.0 - frac_hi

    # Spatial interpolation for both time steps
    u_lo_val = interp_u_lo(pts_3d)
    v_lo_val = interp_v_lo(pts_3d)
    w_lo_val = interp_w_lo(pts_3d)
    dz_lo_val = interp_dz_lo(pts_3d)

    u_hi_val = interp_u_hi(pts_3d)
    v_hi_val = interp_v_hi(pts_3d)
    w_hi_val = interp_w_hi(pts_3d)
    dz_hi_val = interp_dz_hi(pts_3d)

    # Temporal interpolation
    u_val = frac_hi * u_hi_val + frac_lo * u_lo_val
    v_val = frac_hi * v_hi_val + frac_lo * v_lo_val
    w_val = frac_hi * w_hi_val + frac_lo * w_lo_val
    dz_val = frac_hi * dz_hi_val + frac_lo * dz_lo_val

    if settle_vals is not None:
        w_val = w_val - settle_vals

    # Interpolate grid spacing
    pts_2d = pts_3d[:, 1:3]
    dx_val = interp_dx(pts_2d)
    dy_val = interp_dy(pts_2d)

    # Check for bad values
    bad = (
        np.isnan(u_val) | np.isnan(v_val) | np.isnan(w_val) | np.isnan(dz_val) |
        (dx_val <= 0) | (dy_val <= 0) | (dz_val <= 0)
    )

    # Convert to grid-index velocity (di/dt, dj/dt, dk/dt)
    # The negative sign for backward advection will be applied in the main loop
    with np.errstate(divide='ignore', invalid='ignore'):
        di_dt = u_val / dx_val
        dj_dt = v_val / dy_val
        dk_dt = w_val / dz_val

    # Set bad values to zero and return in (i, j, k) order
    di_dt[bad] = 0.0
    dj_dt[bad] = 0.0
    dk_dt[bad] = 0.0

    return np.stack([di_dt, dj_dt, dk_dt], axis=-1), bad

def advect_parcels_backward_wrf(
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
    receptor_lat,
    receptor_lon,
    receptor_radius_m,
    parcel_radius_m=0.0,
    receptor_min_h=0.0,
    receptor_max_h=30000.0,
    emission_start_time=None,
    emission_end_time=None,
    integration_dt=15.0,
    snapshot_config=None,
    settling_profile=None,
    settling_recalc_interval=300.0,
):
    """
    Backward advection of parcels in WRF index space.

    parcels : dict with 'j', 'i', 'k' (1D arrays)
    times   : 1D np.datetime64, shape (nt,)
    u,v,w   : 4D (time, z, y, x) ms-1
    dz      : 4D (time, z, y, x) layer thickness [m]
    z_center: 4D (time, z, y, x) cell center height [m]
    dx_m,dy_m: 2D (y, x) horizontal cell sizes [m]
    xlat,xlon: 2D (y, x) for mapping i,j -> lat,lon
    receptor_lat, receptor_lon: cylinder center [deg]
    receptor_radius_m: cylinder radius [m]
    receptor_min_h,receptor_max_h: vertical bounds of cylinder [m]
    emission_start_time, emission_end_time: np.datetime64 or None
    integration_dt: backward-advection sub-step [s]
    snapshot_config: optional dict containing plotting settings
    settling_profile: dict with 'heights_m' and 'velocity_ms' or None
    settling_recalc_interval: seconds between settling refreshes
    """
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
    emission_end_sec = None
    if times.dtype.kind == "M":
        if emission_start_time is not None:
            t_ref = emission_start_time
        else:
            t_ref = times[0]
        t_sec = (times - t_ref) / np.timedelta64(1, "s")
        t_sec = t_sec.astype(float)
        if emission_end_time is not None:
            emission_end_sec = (
                (emission_end_time - t_ref) / np.timedelta64(1, "s")
            ).astype(float)
    else:
        t_sec = np.asarray(times, dtype=float)

    def _sec_to_time(sec_val):
        if times.dtype.kind == "M":
            # Allow NaN/None values
            if np.isnan(sec_val):
                return np.datetime64("NaT")
            sec_int = int(np.round(sec_val))
            return t_ref + np.timedelta64(sec_int, "s")
        return sec_val

    # Determine starting time index for advection
    it_start = nt - 1
    it_finish = it_start
    dt_values = []
    finish_time_sec = t_sec[it_start]

    j_p = parcels["j"].copy()
    i_p = parcels["i"].copy()
    k_p = parcels["k"].copy()
    n_parcels = j_p.size

    active = np.ones(n_parcels, dtype=bool)
    arrived = np.zeros(n_parcels, dtype=bool)
    arrival_time = np.full(n_parcels, np.nan, dtype=float)
    arrival_z = np.full(n_parcels, np.nan, dtype=float)

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

    # Find closest grid cell to receptor coordinates
    dist2 = (xlat - receptor_lat) ** 2 + (xlon - receptor_lon) ** 2
    j_rec, i_rec = np.unravel_index(np.argmin(dist2), dist2.shape)
    rec_dx = dx_m[j_rec, i_rec]
    rec_dy = dy_m[j_rec, i_rec]
    radius_eff = max(receptor_radius_m + parcel_radius_m, 0.0)
    radius_sq = radius_eff ** 2

    reached_start = False
    current_time_sec = t_sec[it_start]

    trajectory_times = []
    trajectory_i = []
    trajectory_j = []
    trajectory_active = []

    def record_trajectory(time_value):
        trajectory_times.append(time_value)
        trajectory_i.append(i_p.copy())
        trajectory_j.append(j_p.copy())
        trajectory_active.append(active.copy())

    record_trajectory(current_time_sec)

    for step_idx, it in enumerate(range(it_start, 0, -1)):
        if not active.any():
            break

        print(
            "[diag] Processing time index "
            f"{it} ({_format_time_str(times[it])}) -> "
            f"{it - 1} ({_format_time_str(times[it - 1])})."
        )

        dt_total = t_sec[it] - t_sec[it - 1]
        if dt_total <= 0:
            continue

        u_hi = u[it]
        v_hi = v[it]
        w_hi = w[it]
        dz_hi = dz[it]
        z_hi = z_center[it]

        u_lo = u[it - 1]
        v_lo = v[it - 1]
        w_lo = w[it - 1]
        dz_lo = dz[it - 1]
        z_lo = z_center[it - 1]

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
        interp_z_hi = RegularGridInterpolator(
            (k_coords, j_coords, i_coords), z_hi, bounds_error=False, fill_value=np.nan
        )

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
        interp_z_lo = RegularGridInterpolator(
            (k_coords, j_coords, i_coords), z_lo, bounds_error=False, fill_value=np.nan
        )

        sub_time = t_sec[it]
        dt_remaining = dt_total

        while dt_remaining > 0 and active.any():
            dt_step = min(integration_dt, dt_remaining)
            dt_values.append(dt_step)

            idxs = np.where(active)[0]
            if idxs.size == 0:
                break

            # RK4 integration
            t1 = sub_time
            pos1 = np.column_stack((k_p[idxs], j_p[idxs], i_p[idxs]))

            if settling_enabled:
                steps_until_settle_update -= 1
                if steps_until_settle_update <= 0:
                    frac_hi_settle = np.clip((sub_time - t_sec[it - 1]) / dt_total, 0.0, 1.0)
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
                t_sec[it - 1],
                t_sec[it],
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
            k1 = -dt_step * v1

            t2 = sub_time - 0.5 * dt_step
            pos2 = pos1 + 0.5 * k1
            v2, bad2 = _get_velocity_at_time(
                t2,
                pos2,
                t_sec[it - 1],
                t_sec[it],
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
            k2 = -dt_step * v2

            t3 = sub_time - 0.5 * dt_step
            pos3 = pos1 + 0.5 * k2
            v3, bad3 = _get_velocity_at_time(
                t3,
                pos3,
                t_sec[it - 1],
                t_sec[it],
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
            k3 = -dt_step * v3

            t4 = sub_time - dt_step
            pos4 = pos1 + k3
            v4, bad4 = _get_velocity_at_time(
                t4,
                pos4,
                t_sec[it - 1],
                t_sec[it],
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
            k4 = -dt_step * v4

            # Check for any bad interpolations during RK4 steps
            bad_any = bad1 | bad2 | bad3 | bad4
            if bad_any.any():
                active[idxs[bad_any]] = False
                good_mask = ~bad_any
                if not good_mask.any():
                    dt_remaining -= dt_step
                    sub_time -= dt_step
                    current_time_sec = sub_time
                    continue

                # Filter out bad indices from further calculations
                idxs = idxs[good_mask]
                k1 = k1[good_mask]
                k2 = k2[good_mask]
                k3 = k3[good_mask]
                k4 = k4[good_mask]
                if settling_enabled:
                    settle_vals_current = settle_vals_current[good_mask]

            # Update parcel positions
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


            idxs_after = np.where(active)[0]
            if idxs_after.size == 0:
                dt_remaining -= dt_step
                sub_time -= dt_step
                current_time_sec = sub_time
                break

            new_time = sub_time - dt_step
            pts_3d_after = np.column_stack((k_p[idxs_after], j_p[idxs_after], i_p[idxs_after]))
            frac_hi_after = np.clip((new_time - t_sec[it - 1]) / dt_total, 0.0, 1.0)
            frac_lo_after = 1.0 - frac_hi_after
            z_p = (
                frac_hi_after * interp_z_hi(pts_3d_after)
                + frac_lo_after * interp_z_lo(pts_3d_after)
            )

            di_h = i_p[idxs_after] - float(i_rec)
            dj_h = j_p[idxs_after] - float(j_rec)
            dist_sq = (di_h * rec_dx) ** 2 + (dj_h * rec_dy) ** 2

            in_cyl_h = dist_sq <= radius_sq
            in_cyl_z = (z_p >= receptor_min_h) & (z_p <= receptor_max_h)
            hit = in_cyl_h & in_cyl_z

            if hit.any():
                hit_idxs_all = idxs_after[hit]
                z_hit = z_p[hit]
                valid_hit = np.ones(hit_idxs_all.size, dtype=bool)
                if emission_end_sec is not None:
                    valid_hit &= new_time <= emission_end_sec + 1e-6
                if emission_start_time is not None:
                    valid_hit &= new_time >= -1e-6
                if np.any(valid_hit):
                    hit_idxs = hit_idxs_all[valid_hit]
                    arrived[hit_idxs] = True
                    arrival_time[hit_idxs] = np.maximum(new_time, 0.0)
                    arrival_z[hit_idxs] = z_hit[valid_hit]
                    active[hit_idxs] = False

            dt_remaining -= dt_step
            sub_time = new_time
            current_time_sec = sub_time

            if emission_start_time is not None and sub_time <= 0.0:
                reached_start = True
                break

        it_finish = it - 1
        finish_time_sec = sub_time
        current_time_sec = sub_time
        record_trajectory(current_time_sec)

        if snapshot_config is not None and snapshot_config.get("column2d") is not None:
            mask_plot = active | arrived
            parcels_snapshot = dict(j=j_p[mask_plot], i=i_p[mask_plot])
            total_steps = snapshot_config.get("total_steps", it_start)
            reverse_index = max(total_steps - step_idx, 0)
            snapshot_path = (
                snapshot_config["output_dir"]
                / f"{snapshot_config['prefix']}{reverse_index:03d}.png"
            )
            snapshot_title = (
                f"Parcels at {_format_time_str(_sec_to_time(sub_time))}\n"
                f"Index {reverse_index}"
            )
            plot_parcel_locations(
                column2d=snapshot_config["column2d"],
                xlat=snapshot_config["xlat"],
                xlon=snapshot_config["xlon"],
                parcels=parcels_snapshot,
                out_path=str(snapshot_path),
                title=snapshot_title,
                threshold=snapshot_config.get("threshold"),
                receptor_lat=snapshot_config.get("receptor_lat"),
                receptor_lon=snapshot_config.get("receptor_lon"),
                receptor_radius_m=snapshot_config.get("receptor_radius_m"),
                figure_dpi=figure_dpi,
            )
            print(f"[diag] Snapshot saved to '{snapshot_path}'.")

        if reached_start:
            break

    # Calculate final heights for all parcels
    final_z = np.full(n_parcels, np.nan, dtype=float)
    final_time_sec = np.full(n_parcels, np.nan, dtype=float)

    # For parcels that arrived, the final state is the arrival state
    final_z[arrived] = arrival_z[arrived]
    final_time_sec[arrived] = arrival_time[arrived]

    # For parcels that did not arrive, find their last active state
    not_arrived_indices = np.where(~arrived)[0]
    if not_arrived_indices.size > 0:
        last_active_time_idx = np.array(trajectory_active).T[not_arrived_indices].sum(axis=1) - 1
        final_time_sec[not_arrived_indices] = np.array(trajectory_times)[last_active_time_idx]

        final_pts = np.column_stack((k_p[not_arrived_indices], j_p[not_arrived_indices], i_p[not_arrived_indices]))
        final_z[not_arrived_indices] = interp_z_lo(final_pts) # Use last available interpolator

    result = dict(parcels)
    result["arrived"] = arrived
    result["arrival_time"] = arrival_time
    result["arrival_z"] = arrival_z
    if dt_values:
        dt_stats = dict(
            min=float(np.min(dt_values)),
            max=float(np.max(dt_values)),
            mean=float(np.mean(dt_values)),
        )
    else:
        dt_stats = dict(min=float("nan"), max=float("nan"), mean=float("nan"))
    result["advection_start_index"] = it_start
    result["final_j"] = j_p
    result["final_i"] = i_p
    result["final_k"] = k_p
    result["final_z"] = final_z
    result["final_time"] = final_time_sec
    result["advection_finish_index"] = it_finish
    result["advection_start_time"] = times[it_start]
    if times.dtype.kind == "M":
        result["advection_finish_time"] = _sec_to_time(finish_time_sec)
    else:
        result["advection_finish_time"] = times[it_finish]
    result["dt_seconds_stats"] = dt_stats
    result["trajectory_times"] = np.array(trajectory_times, dtype=float)
    result["trajectory_i"] = np.stack(trajectory_i, axis=0)
    result["trajectory_j"] = np.stack(trajectory_j, axis=0)
    result["trajectory_active"] = np.stack(trajectory_active, axis=0)
    return result


def plot_parcel_locations(
    column2d,
    xlat,
    xlon,
    parcels,
    out_path,
    title=None,
    threshold=None,
    receptor_lat=None,
    receptor_lon=None,
    receptor_radius_m=None,
    colorbar_label="Column value",
    figure_dpi=200,
):
    """Plot sampled parcel locations on top of the column field."""
    column2d = np.asarray(column2d)
    if np.ma.isMaskedArray(column2d):
        column2d = column2d.filled(np.nan)
    xlat = np.asarray(xlat)
    if np.ma.isMaskedArray(xlat):
        xlat = xlat.filled(np.nan)
    xlon = np.asarray(xlon)
    if np.ma.isMaskedArray(xlon):
        xlon = xlon.filled(np.nan)
    ny, nx = column2d.shape
    j_coords = np.arange(ny)
    i_coords = np.arange(nx)

    interp_lat = RegularGridInterpolator(
        (j_coords, i_coords), np.asarray(xlat), bounds_error=False, fill_value=np.nan
    )
    interp_lon = RegularGridInterpolator(
        (j_coords, i_coords), np.asarray(xlon), bounds_error=False, fill_value=np.nan
    )

    parcel_points = np.column_stack((parcels["j"], parcels["i"]))
    lat_p = interp_lat(parcel_points)
    lon_p = interp_lon(parcel_points)

    valid = ~np.isnan(lat_p)
    lat_p = lat_p[valid]
    lon_p = lon_p[valid]

    lon_min = float(np.nanmin(xlon))
    lon_max = float(np.nanmax(xlon))
    lat_min = float(np.nanmin(xlat))
    lat_max = float(np.nanmax(xlat))
    lon_pad = max((lon_max - lon_min) * 0.05, 0.1)
    lat_pad = max((lat_max - lat_min) * 0.05, 0.1)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.set_extent([lon_min - lon_pad, lon_max + lon_pad, lat_min - lat_pad, lat_max + lat_pad], crs=ccrs.PlateCarree())

    mesh = ax.pcolormesh(xlon, xlat, column2d, shading="auto", cmap="plasma", transform=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE, color="gray", linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, color="gray", linewidth=0.4)
    gl = ax.gridlines(draw_labels=True, linewidth=0.2, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False

    if threshold is not None:
        ax.contour(
            xlon,
            xlat,
            column2d,
            levels=[threshold],
            colors="white",
            linewidths=1.0,
            linestyles="-.",
            transform=ccrs.PlateCarree(),
        )
    if receptor_lat is not None and receptor_lon is not None:
        ax.scatter(
            receptor_lon,
            receptor_lat,
            marker="^",
            s=80,
            c="red",
            edgecolors="black",
            linewidths=0.4,
            zorder=6,
            transform=ccrs.PlateCarree(),
        )
        if receptor_radius_m is not None and receptor_radius_m > 0:
            ang = np.linspace(0, 2 * np.pi, 181)
            lat_scale = 111320.0
            lon_scale = np.maximum(np.cos(np.deg2rad(receptor_lat)) * 111320.0, 1e-6)
            lat_circle = receptor_lat + (receptor_radius_m / lat_scale) * np.sin(ang)
            lon_circle = receptor_lon + (receptor_radius_m / lon_scale) * np.cos(ang)
            ax.plot(lon_circle, lat_circle, color="red", linestyle="--", linewidth=1.0, transform=ccrs.Geodetic())
    if lat_p.size:
        ax.scatter(
            lon_p,
            lat_p,
            s=PARCEL_MARKER_SIZE,
            c="red",
            edgecolors=PARCEL_MARKER_EDGE,
            linewidths=PARCEL_MARKER_LINEWIDTH,
            zorder=5,
            transform=ccrs.PlateCarree(),
            alpha=PARCEL_MARKER_ALPHA,
        )

    # Add a horizontal colorbar at the bottom
    cax = fig.add_axes([0.2, 0.05, 0.6, 0.03])
    cbar = plt.colorbar(mesh, cax=cax, label=colorbar_label, orientation="horizontal")

    # cbar = plt.colorbar(mesh, ax=ax, label=colorbar_label, orientation="vertical", pad=0.1, shrink=0.8)
    if title:
        ax.set_title(title)

    text_str = f"Total parcels initialized: {parcels['j'].size}"
    if threshold is not None:
        text_str += f" within contour of {threshold:.2f} {colorbar_label}"
    ax.text(
        0.01,
        0.98,
        text_str,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox=dict(facecolor="white", alpha=0.6, edgecolor="none"),
    )
    fig.savefig(out_path, dpi=figure_dpi)
    plt.close(fig)


def plot_seed_vertical_distribution(
    parcels, out_path, z_min, z_max, figure_dpi=200, x_coords=None
):
    """Plot the initial vertical distribution of parcels."""
    heights = np.asarray(parcels.get("z_init"))
    if heights.size == 0:
        _diag("No heights available for seed vertical plot.")
        return

    if x_coords is None:
        x_vals = np.arange(heights.size, dtype=float)
    else:
        x_vals = np.asarray(x_coords, dtype=float)
        if x_vals.size != heights.size:
            _diag("x_coords size mismatch; falling back to rank-based x-axis.")
            x_vals = np.arange(heights.size, dtype=float)

    cmap = plt.get_cmap("turbo")
    norm = plt.Normalize(z_min, z_max)

    fig, ax = plt.subplots(figsize=(10, 8))
    sc = ax.scatter(
        x_vals,
        heights / 1000.0,
        c=heights,
        cmap=cmap,
        norm=norm,
        s=14,
        linewidths=0.15,
        edgecolors="black",
    )
    ax.set_xlabel("Grid-column index")
    ax.set_ylabel("Initial altitude (km)")
    ax.set_title("Initial vertical distribution of parcels")
    ax.grid(True, linestyle=":", linewidth=0.4, alpha=0.6)
    ax.text(
        0.01,
        0.97,
        f"Total parcels: {heights.size}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox=dict(facecolor="white", alpha=0.6, edgecolor="none"),
    )
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Initial altitude (km)")
    if np.isfinite(z_min) and np.isfinite(z_max) and z_max > z_min:
        ticks = np.linspace(z_min, z_max, num=6)
        cbar.set_ticks(ticks)
        cbar.set_ticklabels([f"{val/1000.0:.1f}" for val in ticks])

    #fig.tight_layout()
    fig.savefig(out_path, dpi=figure_dpi)
    plt.close(fig)


def compute_height_edges(z_bins):
    """Compute vertical edges from bin centers."""
    z_bins = np.asarray(z_bins, dtype=float)
    if z_bins.size >= 2:
        dh = np.diff(z_bins)
        edges = np.empty(z_bins.size + 1, dtype=float)
        edges[1:-1] = z_bins[:-1] + 0.5 * dh
        edges[0] = z_bins[0] - 0.5 * (z_bins[1] - z_bins[0])
        edges[-1] = z_bins[-1] + 0.5 * (z_bins[-1] - z_bins[-2])
        edges[0] = max(0.0, edges[0])
    elif z_bins.size == 1:
        width = max(z_bins[0] * 0.1, 500.0)
        edges = np.array([max(0.0, z_bins[0] - width), z_bins[0] + width])
    else:
        edges = np.array([0.0, 1.0])
    return edges


def plot_emission_matrix(
    emission,
    time_edges,
    z_bins,
    z_edges,
    time_labels,
    out_path,
    time_axis_mode,
    total_parcels=None,
    colorbar_label="Parcel count",
    figure_dpi=200,
):
    """Plot the emission (parcel-count) matrix as time vs height."""
    emission = np.asarray(emission)
    if emission.size == 0:
        print("[diag] Emission matrix empty; skipping figure generation.")
        return

    time_edges_arr = np.asarray(time_edges)
    z_edges_arr = np.asarray(z_edges, dtype=float)
    z_edges_km = z_edges_arr / 1000.0
    z_bins_km = np.asarray(z_bins, dtype=float) / 1000.0

    if time_axis_mode == "datetime":
        def _dt64_to_py(dt):
            return dt.tolist()

        time_edges_dt = [_dt64_to_py(np.datetime64(t)) for t in time_edges_arr]
        time_edges_num = mdates.date2num(time_edges_dt)
        tick_positions_all = mdates.date2num(time_edges_dt[:-1])
        tick_labels_all = [dt.strftime("%H:%M") for dt in time_edges_dt[:-1]]
    else:
        time_edges_num = np.asarray(time_edges_arr, dtype=float)
        tick_positions_all = time_edges_num[:-1]
        tick_labels_all = [str(lbl) for lbl in time_labels]

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.pcolormesh(
        time_edges_num,
        z_edges_km,
        emission,
        alpha=0.08,
        zorder=2,
        edgecolors="grey",
        linewidth=0.2,
        shading="flat",
    )
    max_count = float(np.nanmax(emission))
    if not np.isfinite(max_count) or max_count <= 0:
        max_count = 1.0
    n_bins = min(10, max(3, int(np.ceil(max_count))))
    boundaries = np.linspace(0.0, max_count, n_bins + 1)

    # Custom colormap: viridis with white for highest values, then swapped
    original_cmap = plt.get_cmap("viridis", n_bins)
    new_colors = original_cmap(np.linspace(0, 1, n_bins))
    new_colors[-1] = (1, 1, 1, 1)  # Set the highest value color to white

    # Swap first and last colors as per user request
    first_color = new_colors[0].copy()
    new_colors[0] = new_colors[-1]
    new_colors[-1] = first_color

    cmap = ListedColormap(new_colors)

    norm = BoundaryNorm(boundaries, cmap.N, clip=True)
    cs = ax.pcolormesh(
        time_edges_num,
        z_edges_km,
        emission,
        cmap=cmap,
        norm=norm,
        shading="flat",
    )

    ax.set_ylabel("Altitude (km)")
    ax.set_xlabel("Time (UTC)" if time_axis_mode == "datetime" else "Time")
    if total_parcels is not None:
        ax.text(
            0.01,
            0.98,
            f"Total parcels reaching receptor: {total_parcels}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox=dict(facecolor="white", alpha=0.6, edgecolor="none"),
        )
    ax.set_ylim(z_edges_km[0], z_edges_km[-1])

    if tick_labels_all and len(tick_positions_all) > 0:
        max_labels = 18
        if len(tick_labels_all) > max_labels:
            step = int(np.ceil(len(tick_labels_all) / max_labels))
            tick_positions = tick_positions_all[::step]
            tick_labels = tick_labels_all[::step]
        else:
            tick_positions = tick_positions_all
            tick_labels = tick_labels_all
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, rotation=90, ha="center", fontsize=6)
    ax.set_yticks(
        z_bins_km
        if z_bins_km.size <= 20
        else z_bins_km[:: max(1, z_bins_km.size // 20)]
    )

    tick_vals = boundaries
    if tick_vals.size > 15:
        step = int(np.ceil(tick_vals.size / 15))
        tick_vals = tick_vals[::step]
    cbar = plt.colorbar(
        cs,
        ax=ax,
        label=colorbar_label,
        boundaries=boundaries,
        ticks=tick_vals,
        spacing="proportional",
    )

    #fig.tight_layout()
    fig.savefig(out_path, dpi=figure_dpi)
    plt.close(fig)


def plot_parcel_trajectories(
    column2d,
    xlat,
    xlon,
    trajectory_times,
    trajectory_i,
    trajectory_j,
    trajectory_active,
    arrived_flags,
    threshold,
    receptor_lat,
    receptor_lon,
    receptor_radius_m,
    out_path,
    colorbar_label="Column value",
    initial_heights=None,
    z_min=None,
    z_max=None,
    figure_dpi=200,
):
    """Plot parcel trajectories using stored hourly positions."""
    column2d = np.asarray(column2d)
    if np.ma.isMaskedArray(column2d):
        column2d = column2d.filled(np.nan)
    xlat = np.asarray(xlat)
    if np.ma.isMaskedArray(xlat):
        xlat = xlat.filled(np.nan)
    xlon = np.asarray(xlon)
    if np.ma.isMaskedArray(xlon):
        xlon = xlon.filled(np.nan)

    traj_i = np.asarray(trajectory_i)
    traj_j = np.asarray(trajectory_j)
    active_hist = np.asarray(trajectory_active, dtype=bool)
    arrived_flags = np.asarray(arrived_flags, dtype=bool)

    if not arrived_flags.any():
        print("[diag] No parcels reached the receptor; trajectory figure will be empty.")

    ny, nx = column2d.shape
    j_coords = np.arange(ny)
    i_coords = np.arange(nx)
    interp_lat = RegularGridInterpolator(
        (j_coords, i_coords), xlat, bounds_error=False, fill_value=np.nan
    )
    interp_lon = RegularGridInterpolator(
        (j_coords, i_coords), xlon, bounds_error=False, fill_value=np.nan
    )

    def indices_to_latlon(j_arr, i_arr):
        pts = np.column_stack((j_arr, i_arr))
        return interp_lat(pts), interp_lon(pts)

    lat_hist = np.empty_like(traj_j, dtype=float)
    lon_hist = np.empty_like(traj_i, dtype=float)
    for t in range(traj_j.shape[0]):
        lat_hist[t, :], lon_hist[t, :] = indices_to_latlon(traj_j[t, :], traj_i[t, :])
    n_snap = traj_j.shape[0]

    lon_min, lon_max = float(np.nanmin(xlon)), float(np.nanmax(xlon))
    lat_min, lat_max = float(np.nanmin(xlat)), float(np.nanmax(xlat))
    lon_pad = max((lon_max - lon_min) * 0.05, 0.1)
    lat_pad = max((lat_max - lat_min) * 0.05, 0.1)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.set_extent([lon_min - lon_pad, lon_max + lon_pad, lat_min - lat_pad, lat_max + lat_pad], crs=ccrs.PlateCarree())

    ax.add_feature(cfeature.COASTLINE, color="gray", linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, color="gray", linewidth=0.4)
    gl = ax.gridlines(draw_labels=True, linewidth=0.2, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False

    if threshold is not None:
        ax.contour(xlon, xlat, column2d, levels=[threshold], colors="black", linewidths=1.2, transform=ccrs.PlateCarree())

    init_heights = np.asarray(initial_heights, dtype=float) if initial_heights is not None else None

    cmap_colors = None
    color_bins = None
    if (
        init_heights is not None
        and z_min is not None
        and z_max is not None
        and z_max > z_min
    ):
        n_intervals = 10
        color_bins = np.linspace(z_min, z_max, n_intervals + 1)
        base_cmap = plt.get_cmap("rainbow", n_intervals)
        palette = base_cmap(np.arange(n_intervals))
        cmap_colors = ListedColormap(palette)
        color_idx = np.digitize(init_heights, color_bins) - 1
        color_idx = np.clip(color_idx, 0, n_intervals - 1)
        color_lookup = palette[color_idx]
    else:
        color_lookup = np.tile(np.array([[0.0, 0.0, 0.0, 1.0]]), (traj_i.shape[1], 1))

    parcel_indices = np.where(arrived_flags)[0]
    init_heights_subset = None
    if init_heights is not None and parcel_indices.size > 0:
        subset_vals = init_heights[parcel_indices]
        subset_vals = subset_vals[np.isfinite(subset_vals)]
        if subset_vals.size > 0:
            init_heights_subset = subset_vals
    if parcel_indices.size > 0:
        lon = lon_hist[:, parcel_indices]
        lat = lat_hist[:, parcel_indices]

        # Create line segments
        segments = np.array([
            np.stack([lon[:-1, :], lat[:-1, :]], axis=2),
            np.stack([lon[1:, :], lat[1:, :]], axis=2)
        ]).transpose(2, 1, 0, 3) # (n_parcels, n_segments, 2, 2)

        # Filter for active segments
        active_mask = (active_hist[:-1, parcel_indices] & active_hist[1:, parcel_indices]).T

        # Get colors for each parcel and repeat for each segment
        colors_for_parcels = color_lookup[parcel_indices]

        # This check is important if active_mask is not rectangular
        if active_mask.any():
            segments_to_plot = segments[active_mask]
            colors_to_plot = np.repeat(colors_for_parcels, active_mask.sum(axis=1), axis=0)

            lc = LineCollection(
                segments_to_plot,
                colors=colors_to_plot,
                linestyle=TRAJECTORY_LINESTYLE,
                linewidth=TRAJECTORY_LINEWIDTH,
                alpha=TRAJECTORY_ALPHA,
                transform=ccrs.PlateCarree(),
            )
            ax.add_collection(lc)

        start_lon = lon_hist[0, parcel_indices]
        start_lat = lat_hist[0, parcel_indices]
        start_colors = colors_for_parcels

        ax.scatter(
            start_lon,
            start_lat,
            s=PARCEL_MARKER_SIZE,
            c=start_colors,
            edgecolors=PARCEL_MARKER_EDGE,
            linewidths=PARCEL_MARKER_LINEWIDTH,
            zorder=6,
            transform=ccrs.PlateCarree(),
            alpha=PARCEL_MARKER_ALPHA,
        )

    if receptor_lat is not None and receptor_lon is not None:
        ax.scatter(receptor_lon, receptor_lat, marker="^", s=80, c="red", edgecolors="black", linewidths=0.4, zorder=7, transform=ccrs.PlateCarree())
        if receptor_radius_m is not None and receptor_radius_m > 0:
            ang = np.linspace(0, 2 * np.pi, 181)
            lat_scale = 111320.0
            lon_scale = np.maximum(np.cos(np.deg2rad(receptor_lat)) * 111320.0, 1e-6)
            lat_circle = receptor_lat + (receptor_radius_m / lat_scale) * np.sin(ang)
            lon_circle = receptor_lon + (receptor_radius_m / lon_scale) * np.cos(ang)
            ax.plot(lon_circle, lat_circle, color="red", linestyle="--", linewidth=1.0, transform=ccrs.Geodetic())

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    min_init = float(np.nanmin(init_heights_subset)) if init_heights_subset is not None else float("nan")
    max_init = float(np.nanmax(init_heights_subset)) if init_heights_subset is not None else float("nan")
    title_suffix = ""
    if np.isfinite(min_init) and np.isfinite(max_init):
        title_suffix = f" (min={min_init/1000.0:.2f} km, max={max_init/1000.0:.2f} km)"
    ax.set_title("Parcel Trajectories colored by Initial Plume Height" + title_suffix)

    if color_bins is not None and cmap_colors is not None and parcel_indices.size > 0:
        norm = BoundaryNorm(color_bins, cmap_colors.N)
        sm = plt.cm.ScalarMappable(cmap=cmap_colors, norm=norm)
        sm.set_array([])
        cax = fig.add_axes([0.2, 0.05, 0.6, 0.03])
        tick_centers = color_bins[:-1] + 0.5 * np.diff(color_bins)
        cb = plt.colorbar(
            sm,
            cax=cax,
            orientation="horizontal",
            boundaries=color_bins,
            ticks=tick_centers,
        )
        cb.set_ticklabels([f"{val/1000:.1f}" for val in tick_centers])
        cb.set_label("Initial Height in Plume (km)")

    fig.savefig(out_path, dpi=figure_dpi)
    plt.close(fig)


def plot_parcel_age_map(
    column2d,
    xlat,
    xlon,
    trajectory_i,
    trajectory_j,
    trajectory_active,
    parcel_indices,
    parcel_ages_hours,
    threshold,
    receptor_lat,
    receptor_lon,
    receptor_radius_m,
    out_path,
    colorbar_label="Column value",
    figure_dpi=200,
):
    """Plot parcel trajectories coloured by arrival age."""
    parcel_indices = np.asarray(parcel_indices, dtype=int)
    ages = np.asarray(parcel_ages_hours, dtype=float)
    valid = np.isfinite(ages)
    parcel_indices = parcel_indices[valid]
    ages = ages[valid]

    if parcel_indices.size == 0:
        print("[diag] No parcel-age data available; skipping trajectory-age figure.")
        return False

    column2d = np.asarray(column2d)
    if np.ma.isMaskedArray(column2d):
        column2d = column2d.filled(np.nan)
    xlat = np.asarray(xlat)
    if np.ma.isMaskedArray(xlat):
        xlat = xlat.filled(np.nan)
    xlon = np.asarray(xlon)
    if np.ma.isMaskedArray(xlon):
        xlon = xlon.filled(np.nan)

    traj_i = np.asarray(trajectory_i)
    traj_j = np.asarray(trajectory_j)
    active_hist = np.asarray(trajectory_active, dtype=bool)

    ny, nx = column2d.shape
    j_coords = np.arange(ny)
    i_coords = np.arange(nx)
    interp_lat = RegularGridInterpolator(
        (j_coords, i_coords), xlat, bounds_error=False, fill_value=np.nan
    )
    interp_lon = RegularGridInterpolator(
        (j_coords, i_coords), xlon, bounds_error=False, fill_value=np.nan
    )

    def indices_to_latlon(j_arr, i_arr):
        pts = np.column_stack((j_arr, i_arr))
        return interp_lat(pts), interp_lon(pts)

    lat_hist = np.empty_like(traj_j, dtype=float)
    lon_hist = np.empty_like(traj_i, dtype=float)
    for t in range(traj_j.shape[0]):
        lat_hist[t, :], lon_hist[t, :] = indices_to_latlon(traj_j[t, :], traj_i[t, :])

    lon_min, lon_max = float(np.nanmin(xlon)), float(np.nanmax(xlon))
    lat_min, lat_max = float(np.nanmin(xlat)), float(np.nanmax(xlat))
    lon_pad = max((lon_max - lon_min) * 0.05, 0.1)
    lat_pad = max((lat_max - lat_min) * 0.05, 0.1)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.set_extent([lon_min - lon_pad, lon_max + lon_pad, lat_min - lat_pad, lat_max + lat_pad], crs=ccrs.PlateCarree())

    ax.add_feature(cfeature.COASTLINE, color="gray", linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, color="gray", linewidth=0.4)
    gl = ax.gridlines(draw_labels=True, linewidth=0.2, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False

    if threshold is not None:
        ax.contour(xlon, xlat, column2d, levels=[threshold], colors="black", linewidths=1.2, transform=ccrs.PlateCarree())

    ages_subset = ages[np.isfinite(ages)]
    n_bins = 10
    age_min = float(np.nanmin(ages))
    age_max = float(np.nanmax(ages))
    if np.isclose(age_min, age_max):
        age_max = age_min + 0.1
    age_bins = np.linspace(age_min, age_max, n_bins + 1)
    cmap = plt.get_cmap("gist_ncar", n_bins)
    palette = cmap(np.arange(n_bins))
    color_idx = np.digitize(ages, age_bins) - 1
    color_idx = np.clip(color_idx, 0, n_bins - 1)

    parcel_colors = palette[color_idx]

    lon = lon_hist[:, parcel_indices]
    lat = lat_hist[:, parcel_indices]

    segments = np.array([
        np.stack([lon[:-1, :], lat[:-1, :]], axis=2),
        np.stack([lon[1:, :], lat[1:, :]], axis=2)
    ]).transpose(2, 1, 0, 3)

    active_mask = (active_hist[:-1, parcel_indices] & active_hist[1:, parcel_indices]).T
    masked_segments = segments[active_mask]

    colors_for_segments = np.repeat(parcel_colors, active_mask.sum(axis=1), axis=0)

    lc = LineCollection(
        masked_segments,
        colors=colors_for_segments,
        linestyle=TRAJECTORY_LINESTYLE,
        linewidth=TRAJECTORY_LINEWIDTH,
        alpha=TRAJECTORY_ALPHA,
        transform=ccrs.PlateCarree(),
    )
    ax.add_collection(lc)

    start_lon = lon_hist[0, parcel_indices]
    start_lat = lat_hist[0, parcel_indices]
    ax.scatter(
        start_lon,
        start_lat,
        s=PARCEL_MARKER_SIZE,
        c=parcel_colors,
        edgecolors=PARCEL_MARKER_EDGE,
        linewidths=PARCEL_MARKER_LINEWIDTH,
        zorder=6,
        transform=ccrs.PlateCarree(),
        alpha=PARCEL_MARKER_ALPHA,
    )

    if receptor_lat is not None and receptor_lon is not None:
        ax.scatter(receptor_lon, receptor_lat, marker="^", s=80, c="red", edgecolors="black", linewidths=0.4, zorder=7, transform=ccrs.PlateCarree())
        if receptor_radius_m is not None and receptor_radius_m > 0:
            ang = np.linspace(0, 2 * np.pi, 181)
            lat_scale = 111320.0
            lon_scale = np.maximum(np.cos(np.deg2rad(receptor_lat)) * 111320.0, 1e-6)
            lat_circle = receptor_lat + (receptor_radius_m / lat_scale) * np.sin(ang)
            lon_circle = receptor_lon + (receptor_radius_m / lon_scale) * np.cos(ang)
            ax.plot(lon_circle, lat_circle, color="red", linestyle="--", linewidth=1.0, transform=ccrs.Geodetic())

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    min_age = float(np.nanmin(ages_subset)) if ages_subset.size else float("nan")
    max_age = float(np.nanmax(ages_subset)) if ages_subset.size else float("nan")
    title_suffix = ""
    if np.isfinite(min_age) and np.isfinite(max_age):
        title_suffix = f" (min={min_age:.2f} h, max={max_age:.2f} h)"
    ax.set_title("Parcel trajectories coloured by arrival age" + title_suffix)

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


def plot_parcel_emission_time_map(
    column2d,
    xlat,
    xlon,
    trajectory_i,
    trajectory_j,
    trajectory_active,
    parcel_indices,
    parcel_emission_time_hours,
    threshold,
    receptor_lat,
    receptor_lon,
    receptor_radius_m,
    out_path,
    colorbar_label="Column value",
    figure_dpi=200,
    emission_start_time=None,
):
    """Plot parcel trajectories coloured by emission time (hours since emission start)."""
    parcel_indices = np.asarray(parcel_indices, dtype=int)
    emission_hours = np.asarray(parcel_emission_time_hours, dtype=float)
    valid = np.isfinite(emission_hours)
    parcel_indices = parcel_indices[valid]
    emission_hours = emission_hours[valid]

    if parcel_indices.size == 0:
        print("[diag] No emission-time data available; skipping emission-time figure.")
        return False

    column2d = np.asarray(column2d)
    if np.ma.isMaskedArray(column2d):
        column2d = column2d.filled(np.nan)
    xlat = np.asarray(xlat)
    if np.ma.isMaskedArray(xlat):
        xlat = xlat.filled(np.nan)
    xlon = np.asarray(xlon)
    if np.ma.isMaskedArray(xlon):
        xlon = xlon.filled(np.nan)

    traj_i = np.asarray(trajectory_i)
    traj_j = np.asarray(trajectory_j)
    active_hist = np.asarray(trajectory_active, dtype=bool)

    ny, nx = column2d.shape
    j_coords = np.arange(ny)
    i_coords = np.arange(nx)
    interp_lat = RegularGridInterpolator(
        (j_coords, i_coords), xlat, bounds_error=False, fill_value=np.nan
    )
    interp_lon = RegularGridInterpolator(
        (j_coords, i_coords), xlon, bounds_error=False, fill_value=np.nan
    )

    def indices_to_latlon(j_arr, i_arr):
        pts = np.column_stack((j_arr, i_arr))
        return interp_lat(pts), interp_lon(pts)

    lat_hist = np.empty_like(traj_j, dtype=float)
    lon_hist = np.empty_like(traj_i, dtype=float)
    for t in range(traj_j.shape[0]):
        lat_hist[t, :], lon_hist[t, :] = indices_to_latlon(traj_j[t, :], traj_i[t, :])

    lon_min, lon_max = float(np.nanmin(xlon)), float(np.nanmax(xlon))
    lat_min, lat_max = float(np.nanmin(xlat)), float(np.nanmax(xlat))
    lon_pad = max((lon_max - lon_min) * 0.05, 0.1)
    lat_pad = max((lat_max - lat_min) * 0.05, 0.1)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.set_extent(
        [lon_min - lon_pad, lon_max + lon_pad, lat_min - lat_pad, lat_max + lat_pad],
        crs=ccrs.PlateCarree(),
    )

    ax.add_feature(cfeature.COASTLINE, color="gray", linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, color="gray", linewidth=0.4)
    gl = ax.gridlines(draw_labels=True, linewidth=0.2, color="gray", alpha=0.5, linestyle="--")
    gl.top_labels = False
    gl.right_labels = False

    if threshold is not None:
        ax.contour(
            xlon,
            xlat,
            column2d,
            levels=[threshold],
            colors="black",
            linewidths=1.2,
            transform=ccrs.PlateCarree(),
        )

    n_bins = 10
    t_min = float(np.nanmin(emission_hours))
    t_max = float(np.nanmax(emission_hours))
    if np.isclose(t_min, t_max):
        t_max = t_min + 0.1
    time_bins = np.linspace(t_min, t_max, n_bins + 1)
    cmap = plt.get_cmap("gist_ncar", n_bins)
    palette = cmap(np.arange(n_bins))
    color_idx = np.digitize(emission_hours, time_bins) - 1
    color_idx = np.clip(color_idx, 0, n_bins - 1)

    lon = lon_hist[:, parcel_indices]
    lat = lat_hist[:, parcel_indices]

    segments = np.array(
        [
            np.stack([lon[:-1, :], lat[:-1, :]], axis=2),
            np.stack([lon[1:, :], lat[1:, :]], axis=2),
        ]
    ).transpose(2, 1, 0, 3)

    active_mask = (active_hist[:-1, parcel_indices] & active_hist[1:, parcel_indices]).T
    masked_segments = segments[active_mask]

    parcel_colors = palette[color_idx]
    colors_for_segments = np.repeat(parcel_colors, active_mask.sum(axis=1), axis=0)

    lc = LineCollection(
        masked_segments,
        colors=colors_for_segments,
        linestyle=TRAJECTORY_LINESTYLE,
        linewidth=TRAJECTORY_LINEWIDTH,
        alpha=TRAJECTORY_ALPHA,
        transform=ccrs.PlateCarree(),
    )
    ax.add_collection(lc)

    start_lon = lon_hist[0, parcel_indices]
    start_lat = lat_hist[0, parcel_indices]
    ax.scatter(
        start_lon,
        start_lat,
        s=PARCEL_MARKER_SIZE,
        c=parcel_colors,
        edgecolors=PARCEL_MARKER_EDGE,
        linewidths=PARCEL_MARKER_LINEWIDTH,
        zorder=6,
        transform=ccrs.PlateCarree(),
        alpha=PARCEL_MARKER_ALPHA,
    )

    if receptor_lat is not None and receptor_lon is not None:
        ax.scatter(
            receptor_lon,
            receptor_lat,
            marker="^",
            s=80,
            c="red",
            edgecolors="black",
            linewidths=0.4,
            zorder=7,
            transform=ccrs.PlateCarree(),
        )
        if receptor_radius_m is not None and receptor_radius_m > 0:
            ang = np.linspace(0, 2 * np.pi, 181)
            lat_scale = 111320.0
            lon_scale = np.maximum(np.cos(np.deg2rad(receptor_lat)) * 111320.0, 1e-6)
            lat_circle = receptor_lat + (receptor_radius_m / lat_scale) * np.sin(ang)
            lon_circle = receptor_lon + (receptor_radius_m / lon_scale) * np.cos(ang)
            ax.plot(
                lon_circle,
                lat_circle,
                color="red",
                linestyle="--",
                linewidth=1.0,
                transform=ccrs.Geodetic(),
            )

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    min_t = float(np.nanmin(emission_hours))
    max_t = float(np.nanmax(emission_hours))
    title_suffix = ""
    if np.isfinite(min_t) and np.isfinite(max_t):
        title_suffix = f" (min={min_t:.2f} h, max={max_t:.2f} h)"
    ax.set_title("Parcel trajectories coloured by emission time" + title_suffix)

    cax = fig.add_axes([0.2, 0.05, 0.6, 0.03])
    time_centers = time_bins[:-1] + 0.5 * np.diff(time_bins)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=BoundaryNorm(time_bins, cmap.N))
    sm.set_array([])
    cb = plt.colorbar(
        sm,
        cax=cax,
        orientation="horizontal",
        boundaries=time_bins,
        ticks=time_centers,
    )
    cb.set_ticklabels([f"{val:.1f} h" for val in time_centers])
    if emission_start_time is not None:
        cb.set_label(
            f"Hours since emission start ({np.datetime_as_string(emission_start_time, unit='m')})"
        )
    else:
        cb.set_label("Emission time since reference (hours)")

    fig.savefig(out_path, dpi=figure_dpi)
    plt.close(fig)
    return True


def plot_parcel_arrival_height_map(
    column2d,
    xlat,
    xlon,
    trajectory_i,
    trajectory_j,
    trajectory_active,
    parcel_indices,
    parcel_arrival_height_m,
    threshold,
    receptor_lat,
    receptor_lon,
    receptor_radius_m,
    out_path,
    colorbar_label="Column value",
    figure_dpi=200,
    height_min=None,
    height_max=None,
):
    """Plot parcel trajectories coloured by their arrival heights."""
    parcel_indices = np.asarray(parcel_indices, dtype=int)
    heights_m = np.asarray(parcel_arrival_height_m, dtype=float)
    heights_km = heights_m / 1000.0
    valid = np.isfinite(heights_km)
    parcel_indices = parcel_indices[valid]
    heights_m = heights_m[valid]
    heights_km = heights_km[valid]
    heights_subset_km = heights_km.copy()

    if parcel_indices.size == 0:
        print("[diag] No parcel height data available; skipping arrival-height figure.")
        return False

    column2d = np.asarray(column2d)
    if np.ma.isMaskedArray(column2d):
        column2d = column2d.filled(np.nan)
    xlat = np.asarray(xlat)
    if np.ma.isMaskedArray(xlat):
        xlat = xlat.filled(np.nan)
    xlon = np.asarray(xlon)
    if np.ma.isMaskedArray(xlon):
        xlon = xlon.filled(np.nan)

    traj_i = np.asarray(trajectory_i)
    traj_j = np.asarray(trajectory_j)
    active_hist = np.asarray(trajectory_active)

    ny, nx = column2d.shape
    j_coords = np.arange(ny)
    i_coords = np.arange(nx)
    interp_lat = RegularGridInterpolator(
        (j_coords, i_coords), xlat, bounds_error=False, fill_value=np.nan
    )
    interp_lon = RegularGridInterpolator(
        (j_coords, i_coords), xlon, bounds_error=False, fill_value=np.nan
    )

    def indices_to_latlon(j_arr, i_arr):
        pts = np.column_stack((j_arr, i_arr))
        return interp_lat(pts), interp_lon(pts)

    lat_hist = []
    lon_hist = []
    for j_row, i_row in zip(traj_j, traj_i):
        lat_vals, lon_vals = indices_to_latlon(j_row, i_row)
        lat_hist.append(lat_vals)
        lon_hist.append(lon_vals)
    lat_hist = np.vstack(lat_hist)
    lon_hist = np.vstack(lon_hist)

    lon_min = float(np.nanmin(xlon))
    lon_max = float(np.nanmax(xlon))
    lat_min = float(np.nanmin(xlat))
    lat_max = float(np.nanmax(xlat))
    lon_pad = max((lon_max - lon_min) * 0.05, 0.1)
    lat_pad = max((lat_max - lat_min) * 0.05, 0.1)

    fig, ax = _init_geo_axes(
        lon_min,
        lon_max,
        lat_min,
        lat_max,
        lon_pad,
        lat_pad,
        figsize=(10, 8),
    )

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
    color_idx = np.digitize(heights_m, h_bins_m) - 1
    color_idx = np.clip(color_idx, 0, n_bins - 1)

    if threshold is not None:
        ax.contour(
            xlon,
            xlat,
            column2d,
            levels=[threshold],
            colors="black",
            linewidths=1.2,
            transform=ccrs.PlateCarree(),
        )

    if receptor_lat is not None and receptor_lon is not None:
        ax.scatter(
            receptor_lon,
            receptor_lat,
            marker="^",
            s=80,
            c="red",
            edgecolors="black",
            linewidths=0.4,
            zorder=7,
            transform=ccrs.PlateCarree(),
        )
        if receptor_radius_m is not None and receptor_radius_m > 0:
            ang = np.linspace(0, 2 * np.pi, 181)
            lat_scale = 111320.0
            lon_scale = np.maximum(np.cos(np.deg2rad(receptor_lat)) * 111320.0, 1e-6)
            lat_circle = receptor_lat + (receptor_radius_m / lat_scale) * np.sin(ang)
            lon_circle = receptor_lon + (receptor_radius_m / lon_scale) * np.cos(ang)
            ax.plot(
                lon_circle,
                lat_circle,
                color="red",
                linestyle="--",
                linewidth=1.0,
                transform=ccrs.PlateCarree(),
            )

    n_snap = lat_hist.shape[0]
    if parcel_indices.size > 0:
        parcel_colors = palette[color_idx]
        active_sub = active_hist[:, parcel_indices]
        lon_sub = lon_hist[:, parcel_indices]
        lat_sub = lat_hist[:, parcel_indices]

        # Create line segments
        segments = np.array([
            np.stack([lon_sub[:-1, :], lat_sub[:-1, :]], axis=2),
            np.stack([lon_sub[1:, :], lat_sub[1:, :]], axis=2)
        ]).transpose(2, 1, 0, 3)  # (n_parcels, n_segments, 2, 2)

        # Filter for active segments
        active_mask = (active_sub[:-1, :] & active_sub[1:, :]).T

        if active_mask.any():
            segments_to_plot = segments[active_mask]
            colors_to_plot = np.repeat(parcel_colors, active_mask.sum(axis=1), axis=0)

            lc = LineCollection(
                segments_to_plot,
                colors=colors_to_plot,
                linestyle=TRAJECTORY_LINESTYLE,
                linewidth=TRAJECTORY_LINEWIDTH,
                alpha=TRAJECTORY_ALPHA,
                transform=PLATE_CARREE,
            )
            ax.add_collection(lc)

        start_lon = lon_sub[0, :]
        start_lat = lat_sub[0, :]
        ax.scatter(
            start_lon,
            start_lat,
            s=PARCEL_MARKER_SIZE,
            c=parcel_colors,
            edgecolors=PARCEL_MARKER_EDGE,
            linewidths=PARCEL_MARKER_LINEWIDTH,
            zorder=6,
            transform=PLATE_CARREE,
            alpha=PARCEL_MARKER_ALPHA,
        )

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    min_km = float(np.nanmin(heights_subset_km)) if heights_subset_km.size else float("nan")
    max_km = float(np.nanmax(heights_subset_km)) if heights_subset_km.size else float("nan")
    ax.set_title(
        f"Parcel trajectories coloured by arrival height "
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
    cb.set_label("Arrival height (km)")

    fig.savefig(out_path, dpi=figure_dpi)
    plt.close(fig)
    return True


def plot_missed_parcel_trajectories(
    column2d,
    xlat,
    xlon,
    trajectory_i,
    trajectory_j,
    trajectory_active,
    arrived_flags,
    initial_heights,
    final_heights,
    threshold,
    receptor_lat,
    receptor_lon,
    receptor_radius_m,
    out_path,
    z_min=None,
    z_max=None,
    figure_dpi=200,
):
    """Plot trajectories of parcels that did not reach the receptor."""
    missed_flags = ~np.asarray(arrived_flags, dtype=bool)
    parcel_indices = np.where(missed_flags)[0]

    if parcel_indices.size == 0:
        print("[diag] All parcels reached the receptor; skipping missed-trajectory figure.")
        return

    print(f"[diag] Plotting {parcel_indices.size} missed parcel trajectories.")

    column2d = np.asarray(column2d)
    xlat = np.asarray(xlat)
    xlon = np.asarray(xlon)
    traj_i = np.asarray(trajectory_i)
    traj_j = np.asarray(trajectory_j)
    active_hist = np.asarray(trajectory_active, dtype=bool)

    ny, nx = column2d.shape
    j_coords = np.arange(ny)
    i_coords = np.arange(nx)
    interp_lat = RegularGridInterpolator((j_coords, i_coords), xlat, bounds_error=False, fill_value=np.nan)
    interp_lon = RegularGridInterpolator((j_coords, i_coords), xlon, bounds_error=False, fill_value=np.nan)

    def indices_to_latlon(j_arr, i_arr):
        pts = np.column_stack((j_arr, i_arr))
        return interp_lat(pts), interp_lon(pts)

    lat_hist = np.empty_like(traj_j, dtype=float)
    lon_hist = np.empty_like(traj_i, dtype=float)
    for t in range(traj_j.shape[0]):
        lat_hist[t, :], lon_hist[t, :] = indices_to_latlon(traj_j[t, :], traj_i[t, :])

    lon_min, lon_max = float(np.nanmin(xlon)), float(np.nanmax(xlon))
    lat_min, lat_max = float(np.nanmin(xlat)), float(np.nanmax(xlat))
    lon_pad = max((lon_max - lon_min) * 0.05, 0.1)
    lat_pad = max((lat_max - lat_min) * 0.05, 0.1)

    fig, ax = _init_geo_axes(lon_min, lon_max, lat_min, lat_max, lon_pad, lat_pad)

    if threshold is not None:
        ax.contour(xlon, xlat, column2d, levels=[threshold], colors="black", linewidths=1.2, transform=ccrs.PlateCarree())

    # Color mapping based on altitude range
    n_intervals = 10
    z_range_min = z_min if z_min is not None else 0
    z_range_max = z_max if z_max is not None else 30000
    if not np.isfinite(z_range_min):
        z_range_min = 0.0
    if not np.isfinite(z_range_max) or z_range_max <= z_range_min:
        z_range_max = z_range_min + 1.0
    color_bins = np.linspace(z_range_min, z_range_max, n_intervals + 1)
    cmap = plt.get_cmap("rainbow", n_intervals)
    listed_cmap = ListedColormap(cmap(np.arange(n_intervals)))
    norm = BoundaryNorm(color_bins, listed_cmap.N, clip=True)

    total_parcels = traj_i.shape[1]
    if initial_heights is None:
        init_heights_arr = np.full(parcel_indices.size, np.nan)
    else:
        init_heights_raw = np.asarray(initial_heights, dtype=float)
        if init_heights_raw.size == total_parcels:
            init_heights_arr = init_heights_raw[parcel_indices]
        elif init_heights_raw.size == parcel_indices.size:
            init_heights_arr = init_heights_raw
        else:
            init_heights_arr = np.full(parcel_indices.size, np.nan)
            count = min(init_heights_raw.size, init_heights_arr.size)
            if count > 0:
                init_heights_arr[:count] = init_heights_raw[:count]

    default_height = 0.5 * (z_range_min + z_range_max)
    heights_for_color = np.where(
        np.isfinite(init_heights_arr), init_heights_arr, default_height
    )
    color_idx = np.digitize(heights_for_color, color_bins) - 1
    color_idx = np.clip(color_idx, 0, n_intervals - 1)
    color_lookup = np.array(listed_cmap.colors)[color_idx]
    missing_mask = ~np.isfinite(init_heights_arr)
    if missing_mask.any():
        color_lookup[missing_mask] = np.array([0.6, 0.6, 0.6, 1.0])

    # --- Plot Trajectories ---
    n_snap = lat_hist.shape[0]
    lon = lon_hist[:, parcel_indices]
    lat = lat_hist[:, parcel_indices]
    active_sub = active_hist[:, parcel_indices]

    segments = np.array([
        np.stack([lon[:-1, :], lat[:-1, :]], axis=2),
        np.stack([lon[1:, :], lat[1:, :]], axis=2)
    ]).transpose(2, 1, 0, 3)

    active_mask = (active_sub[:-1, :] & active_sub[1:, :]).T

    if active_mask.any():
        segments_to_plot = segments[active_mask]
        colors_to_plot = np.repeat(color_lookup, active_mask.sum(axis=1), axis=0)

        lc = LineCollection(
            segments_to_plot,
            colors=colors_to_plot,
            linestyle=TRAJECTORY_LINESTYLE,
            linewidth=TRAJECTORY_LINEWIDTH,
            alpha=TRAJECTORY_ALPHA,
            transform=PLATE_CARREE,
        )
        ax.add_collection(lc)

    # --- Plot Finish Markers (colored by initial height) ---
    last_active_time_idx = active_hist[:, parcel_indices].sum(axis=0) - 1
    last_active_time_idx = np.clip(last_active_time_idx, 0, lat_hist.shape[0] - 1)

    # Need to get the correct indices for finish_lon and finish_lat
    row_indices = last_active_time_idx
    col_indices = np.arange(parcel_indices.size)

    finish_lon = lon_hist[row_indices, parcel_indices]
    finish_lat = lat_hist[row_indices, parcel_indices]
    finish_colors = color_lookup

    valid_finish = np.isfinite(finish_lon) & np.isfinite(finish_lat)
    ax.scatter(finish_lon[valid_finish], finish_lat[valid_finish], s=PARCEL_MARKER_SIZE, c=finish_colors[valid_finish],
               marker='o', edgecolors=PARCEL_MARKER_EDGE, linewidths=PARCEL_MARKER_LINEWIDTH, zorder=7,
               transform=ccrs.PlateCarree(), alpha=PARCEL_MARKER_ALPHA,
               label="Final Position (colored by initial height)")

    # --- Receptor Visualization ---
    if receptor_lat is not None and receptor_lon is not None:
        ax.scatter(receptor_lon, receptor_lat, marker="^", s=100, c="red", edgecolors="black", linewidths=0.5, zorder=8, transform=ccrs.PlateCarree())
        if receptor_radius_m is not None and receptor_radius_m > 0:
            ang = np.linspace(0, 2 * np.pi, 181)
            lat_scale = 111320.0
            lon_scale = np.maximum(np.cos(np.deg2rad(receptor_lat)) * 111320.0, 1e-6)
            lat_circle = receptor_lat + (receptor_radius_m / lat_scale) * np.sin(ang)
            lon_circle = receptor_lon + (receptor_radius_m / lon_scale) * np.cos(ang)
            ax.plot(lon_circle, lat_circle, color="red", linestyle="--", linewidth=1.2, transform=ccrs.Geodetic(), zorder=8)

    # --- Labels and Colorbar ---
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    finite_init = init_heights_arr[np.isfinite(init_heights_arr)]
    title_suffix = ""
    if finite_init.size:
        min_init = float(np.min(finite_init))
        max_init = float(np.max(finite_init))
        title_suffix = f" (min={min_init/1000.0:.2f} km, max={max_init/1000.0:.2f} km)"
    ax.set_title("Trajectories of Parcels Missing the Receptor" + title_suffix)

    # Create a single colorbar for altitude (km), horizontal like main plot
    sm = plt.cm.ScalarMappable(cmap=listed_cmap, norm=norm)
    sm.set_array([])
    cax = fig.add_axes([0.2, 0.05, 0.6, 0.03])
    tick_centers = color_bins[:-1] + 0.5 * np.diff(color_bins)
    cb = plt.colorbar(sm, cax=cax, orientation="horizontal", boundaries=color_bins, ticks=tick_centers)
    cb.set_ticklabels([f"{val/1000:.1f}" for val in tick_centers])
    cb.set_label("Initial Height in Plume (km)")

    # The legend is now implicit through the markers and colorbar
    # from matplotlib.lines import Line2D
    # legend_elements = [
    #     Line2D([0], [0], marker='o', color='w', label='Final Position (colored by initial height)', markerfacecolor='grey', markersize=10)
    # ]
    # ax.legend(handles=legend_elements, loc='upper left', fontsize=8)

    fig.savefig(out_path, dpi=figure_dpi)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Back-advect SO2 plume parcels in WRF winds."
    )
    parser.add_argument(
        "--wrfout",
        required=True,
        help="Path to WRF output file containing winds and geometry.",
    )
    parser.add_argument(
        "--column",
        required=True,
        help="NetCDF file containing SO2 column on WRF grid.",
    )
    parser.add_argument(
        "--column-var",
        default="SO2_COLUMN",
        help="Variable name for SO2 column in the column file.",
    )
    parser.add_argument(
        "--column-coef",
        type=float,
        default=1.0,
        help="Scaling coefficient applied to the SO2 column (e.g., mol/m2 -> DU).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.1,
        help="Threshold on column loading to define the plume.",
    )
    parser.add_argument(
        "--n-columns",
        type=int,
        default=200,
        help="Number of random plume columns to sample.",
    )
    parser.add_argument(
        "--n-vert",
        type=int,
        default=30,
        help="Number of parcels along each vertical line.",
    )
    parser.add_argument(
        "--seed-bbox",
        nargs=4,
        type=float,
        metavar=("LON_MIN", "LAT_MIN", "LON_MAX", "LAT_MAX"),
        help="Optional bounding box (lon/lat) restricting where parcels are initialized.",
    )
    parser.add_argument(
        "--z-min",
        type=float,
        default=2000.0,
        help="Minimum initial height of parcels [m].",
    )
    parser.add_argument(
        "--z-max",
        type=float,
        default=25000.0,
        help="Maximum initial height of parcels [m].",
    )
    parser.add_argument(
        "--receptor-lat",
        type=float,
        required=True,
        help="Latitude of receptor (center of vertical cylinder) [deg].",
    )
    parser.add_argument(
        "--receptor-lon",
        type=float,
        required=True,
        help="Longitude of receptor (center of vertical cylinder) [deg].",
    )
    parser.add_argument(
        "--receptor-radius",
        type=float,
        default=10000.0,
        help="Horizontal radius of receptor cylinder [m].",
    )
    parser.add_argument(
        "--parcel-radius",
        type=float,
        default=0.0,
        help="Horizontal radius of each parcel (m) for cylinder-contact checks.",
    )
    parser.add_argument(
        "--receptor-min-h",
        type=float,
        default=0.0,
        help="Minimum height of receptor cylinder [m].",
    )
    parser.add_argument(
        "--receptor-max-h",
        type=float,
        default=30000.0,
        help="Maximum height of receptor cylinder [m].",
    )
    parser.add_argument(
        "--integration-dt",
        type=float,
        default=15.0,
        help="Backward-advection sub-step in seconds (e.g., 15).",
    )
    parser.add_argument(
        "--start-time",
        type=str,
        default=None,
        help=(
            "UTC start time for back-trajectories, "
            "e.g. 2021-04-10T15:00:00 or 2021-04-10_15:00:00. "
            "If not provided, the last time step from the WRF file is used."
        ),
    )
    parser.add_argument(
        "--aer-type",
        choices=sorted(SETTLING_VEL_MS.keys()),
        default=None,
        help="Optional aerosol type for gravitational settling (keys from SETTLING_VEL_MS).",
    )
    parser.add_argument(
        "--hourly-figures",
        action="store_true",
        help="Save hourly parcel location plots during back-trajectory integration.",
    )
    parser.add_argument(
        "--arrival-bin-minutes",
        type=float,
        default=60.0,
        help=(
            "Bin width for grouping arriving particles in time [minutes]. "
            "Set to 0 to disable time binning."
        ),
    )
    parser.add_argument(
        "--output-txt",
        default="emission_time_height.txt",
        help="Path to output TXT file with time–height emission series.",
    )
    parser.add_argument(
        "--mass-output-txt",
        default=None,
        help="Optional path for a TXT file storing the mass-weighted emission matrix.",
    )
    parser.add_argument(
        "--output-figure",
        default="emission_time_height.png",
        help="Path to PNG figure for the emission (parcel-count) matrix.",
    )
    parser.add_argument(
        "--mass-figure",
        default=None,
        help="Optional PNG figure for the emission matrix weighted by parcel mass.",
    )
    parser.add_argument(
        "--colorbar-label",
        default="Parcel count",
        help="Colorbar label for the emission matrix plot.",
    )
    parser.add_argument(
        "--trajectory-figure",
        default="parcel_trajectories.png",
        help="Path to PNG figure showing parcel trajectories.",
    )
    parser.add_argument(
        "--seeds-figure",
        default=None,
        help="Optional PNG file for the initial parcel-location map.",
    )
    parser.add_argument(
        "--seeds-vertical-figure",
        default=None,
        help="Optional PNG for the initial vertical distribution of parcels.",
    )
    parser.add_argument(
        "--trajectory-age",
        default=None,
        help="Optional PNG file for the parcel-age scatter plot (hours vs altitude).",
    )
    parser.add_argument(
        "--trajectory-emission-time-figure",
        default=None,
        help="Optional PNG file for parcel trajectories coloured by emission time.",
    )
    parser.add_argument(
        "--trajectory-arrival-height-figure",
        default=None,
        help="Optional PNG file for parcel trajectories coloured by arrival height.",
    )
    parser.add_argument(
        "--missed-trajectory-figure",
        default=None,
        help="Optional PNG file for trajectories of parcels that missed the receptor.",
    )
    parser.add_argument(
        "--figure-dpi",
        type=int,
        default=200,
        help="DPI resolution for all saved figures.",
    )
    parser.add_argument(
        "--state-pickle",
        default=None,
        help="Optional path to store a pickle with all inputs/results for re-plotting.",
    )
    parser.add_argument(
        "--emission-start",
        type=str,
        default=None,
        help=(
            "UTC start time of emission, "
            "e.g. 2021-04-10T15:00:00 or 2021-04-10_15:00:00. "
            "Advection stops at this time, and emission times are "
            "reported relative to it."
        ),
    )
    parser.add_argument(
        "--emission-end",
        type=str,
        default=None,
        help=(
            "UTC end time of emission (same format as --emission-start). "
            "Used to define fixed time bins and ignore arrivals after emission ceased."
        ),
    )
    parser.add_argument(
        "--so2-efolding-days",
        type=float,
        default=None,
        help="e-folding time for SO2 mass decay in days. If set, mass is increased backwards in time.",
    )
    return parser.parse_args()


def main(args):
    wrfout_path = args.wrfout
    column_file = args.column
    column_varname = args.column_var

    threshold = args.threshold
    n_columns = args.n_columns
    n_vert = args.n_vert
    z_min = args.z_min
    z_max = args.z_max
    seed_bbox = tuple(args.seed_bbox) if args.seed_bbox is not None else None

    receptor_lat = args.receptor_lat
    receptor_lon = args.receptor_lon
    receptor_radius_m = args.receptor_radius
    receptor_min_h = args.receptor_min_h
    receptor_max_h = args.receptor_max_h
    integration_dt = args.integration_dt

    settling_profile = None
    if args.aer_type is not None:
        settling_profile = dict(
            heights_m=np.asarray(Z_M, dtype=float),
            velocity_ms=np.asarray(SETTLING_VEL_MS[args.aer_type], dtype=float),
        )
        _diag(f"Applying settling profile for '{args.aer_type}'.")
        _diag(f"Settling heights (m): {settling_profile['heights_m']}")
        _diag(f"Settling velocities (m/s): {settling_profile['velocity_ms']}")
    figure_dpi = max(50, int(args.figure_dpi))

    grid = read_wrf_geometry_and_winds(wrfout_path)
    xlat = grid["xlat"]
    xlon = grid["xlon"]
    area = grid["area"]
    u = grid["u"]
    v = grid["v"]
    w = grid["w"]
    z_center = grid["z_center"]
    dz = grid["dz"]
    dx_m = grid["dx_m"]
    dy_m = grid["dy_m"]
    times = grid["times"]

    times_arr = np.asarray(times)

    # Parse eruption start time (optional)
    emission_start_time = None
    if args.emission_start is not None:
        s = args.emission_start.strip()
        if "T" not in s and "_" in s:
            s_iso = s.replace("_", "T")
        else:
            s_iso = s
        emission_start_time = np.datetime64(s_iso)

        if emission_start_time < times_arr[0] or emission_start_time > times_arr[-1]:
            raise ValueError(
                f"Eruption start time {emission_start_time} outside WRF time range "
                f"[{times_arr[0]}, {times_arr[-1]}]."
            )

    emission_end_time = None
    if args.emission_end is not None:
        if emission_start_time is None:
            raise ValueError("--emission-end requires --emission-start to be set.")
        s = args.emission_end.strip()
        if "T" not in s and "_" in s:
            s_iso = s.replace("_", "T")
        else:
            s_iso = s
        emission_end_time = np.datetime64(s_iso)
        if emission_end_time <= emission_start_time:
            raise ValueError("emission-end must be later than emission-start.")
        if emission_end_time < times_arr[0] or emission_end_time > times_arr[-1]:
            raise ValueError(
                f"Eruption end time {emission_end_time} outside WRF time range "
                f"[{times_arr[0]}, {times_arr[-1]}]."
            )

    # Find closest grid cell to receptor for vertical profile
    dist2_rec = (xlat - receptor_lat) ** 2 + (xlon - receptor_lon) ** 2
    j_rec, i_rec = np.unravel_index(np.argmin(dist2_rec), dist2_rec.shape)

    nt = u.shape[0]
    if args.start_time:
        s = args.start_time.strip()
        if "T" not in s and "_" in s:
            s_iso = s.replace("_", "T")
        else:
            s_iso = s
        start_time_dt = np.datetime64(s_iso)
        time_diffs = np.abs(times_arr - start_time_dt)
        it_start = np.argmin(time_diffs)
        if time_diffs[it_start] > np.timedelta64(1, 'm'):
             warnings.warn(f"Provided start time {start_time_dt} is more than 1 minute away from the closest WRF time {times_arr[it_start]}. Using the closest time.")
        _diag(f"Using start time {times_arr[it_start]} at index {it_start} (closest to requested {start_time_dt}).")
    else:
        it_start = nt - 1
        _diag(f"No start time provided. Using last available time: {times_arr[it_start]} at index {it_start}.")

    with Dataset(column_file, "r") as ds_col:
        col_var = ds_col.variables[column_varname]
        column_data = np.asarray(col_var[:])
        if column_data.ndim == 3:
            n_col_times = column_data.shape[0]
            col_time_index = min(it_start, n_col_times - 1)
            if it_start >= n_col_times:
                print(
                    "[diag] Column file has only "
                    f"{n_col_times} time(s); using index {col_time_index} instead of {it_start}."
                )
            column2d = np.asarray(column_data[col_time_index, :, :])
        elif column_data.ndim == 2:
            column2d = np.asarray(column_data)
        else:
            raise ValueError(
                f"Unsupported column array shape {column_data.shape}; expected 2D or 3D."
            )

    column_coef = args.column_coef
    if column_coef != 1.0:
        print(f"[diag] Scaling column field by factor {column_coef}.")
        column2d = column2d * column_coef
    else:
        print("[diag] Column scaling factor = 1.0 (no change).")

    parcels_init = generate_parcels_from_column_wrf(
        column2d=column2d,
        xlat=xlat,
        xlon=xlon,
        z_profile_3d=z_center,
        time_index=it_start,
        threshold=threshold,
        n_columns=n_columns,
        n_vert=n_vert,
        z_min=z_min,
        z_max=z_max,
        seed_bbox=seed_bbox,
    )

    if n_vert > 0:
        cell_mass = np.asarray(column2d, dtype=float) * np.asarray(area, dtype=float)
        cell_mass = np.where(np.isfinite(cell_mass), cell_mass, 0.0)
        j_idx = parcels_init["j"].astype(int)
        i_idx = parcels_init["i"].astype(int)
        parcel_mass = cell_mass[j_idx, i_idx] / float(n_vert)
    else:
        parcel_mass = np.zeros_like(parcels_init["j"], dtype=float)
    parcels_init["mass"] = parcel_mass

    print(
        "[diag] Initialized "
        f"{parcels_init['j'].size} parcels at time index {it_start} "
        f"({_format_time_str(times_arr[it_start])})."
    )

    if args.seeds_figure:
        plot_title = (
            f"Parcel seeds at WRF time index {it_start}\n"
            f"{_format_time_str(times_arr[it_start])}"
        )
        plot_parcel_locations(
            column2d=column2d,
            xlat=xlat,
            xlon=xlon,
            parcels=parcels_init,
            out_path=args.seeds_figure,
            threshold=threshold,
            receptor_lat=receptor_lat,
            receptor_lon=receptor_lon,
            receptor_radius_m=receptor_radius_m,
            colorbar_label=args.colorbar_label,
            title=plot_title,
            figure_dpi=figure_dpi,
        )
        print(f"[diag] Parcel-location map saved to '{args.seeds_figure}'.")

    if args.seeds_vertical_figure:
        plot_seed_vertical_distribution(
            parcels=parcels_init,
            out_path=args.seeds_vertical_figure,
            z_min=z_min,
            z_max=z_max,
            figure_dpi=figure_dpi,
            x_coords=parcels_init["i"],
        )
        print(f"[diag] Parcel vertical distribution saved to '{args.seeds_vertical_figure}'.")

    snapshot_config = None
    if args.hourly_figures:
        snapshot_config = dict(
            column2d=column2d,
            xlat=xlat,
            xlon=xlon,
            threshold=threshold,
            receptor_lat=receptor_lat,
            receptor_lon=receptor_lon,
            receptor_radius_m=receptor_radius_m,
            output_dir=Path("."),
            prefix="parcel_positions_hour_",
            total_steps=it_start,
        )

    result = advect_parcels_backward_wrf(
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
        receptor_lat=receptor_lat,
        receptor_lon=receptor_lon,
        receptor_radius_m=receptor_radius_m,
        parcel_radius_m=args.parcel_radius,
        receptor_min_h=receptor_min_h,
        receptor_max_h=receptor_max_h,
        emission_start_time=emission_start_time,
        emission_end_time=emission_end_time,
        integration_dt=integration_dt,
        snapshot_config=snapshot_config,
        settling_profile=settling_profile,
    )

    plot_parcel_trajectories(
        column2d=column2d,
        xlat=xlat,
        xlon=xlon,
        trajectory_times=result["trajectory_times"],
        trajectory_i=result["trajectory_i"],
        trajectory_j=result["trajectory_j"],
        trajectory_active=result["trajectory_active"],
        arrived_flags=result["arrived"],
        threshold=threshold,
        receptor_lat=receptor_lat,
        receptor_lon=receptor_lon,
        receptor_radius_m=receptor_radius_m,
        out_path=args.trajectory_figure,
        colorbar_label=args.colorbar_label,
        initial_heights=parcels_init.get("z_init"),
        z_min=z_min,
        z_max=z_max,
        figure_dpi=figure_dpi,
    )
    print(f"[diag] Trajectory figure saved to '{args.trajectory_figure}'.")

    if args.missed_trajectory_figure:
        plot_missed_parcel_trajectories(
            column2d=column2d,
            xlat=xlat,
            xlon=xlon,
            trajectory_i=result["trajectory_i"],
            trajectory_j=result["trajectory_j"],
            trajectory_active=result["trajectory_active"],
            arrived_flags=result["arrived"],
            initial_heights=parcels_init.get("z_init"),
            final_heights=result["final_z"],
            threshold=threshold,
            receptor_lat=receptor_lat,
            receptor_lon=receptor_lon,
            receptor_radius_m=receptor_radius_m,
            out_path=args.missed_trajectory_figure,
            z_min=z_min,
            z_max=z_max,
            figure_dpi=figure_dpi,
        )
        print(f"[diag] Missed trajectory figure saved to '{args.missed_trajectory_figure}'.")

    start_idx = result["advection_start_index"]
    finish_idx = result["advection_finish_index"]
    print(
        "[diag] Advection window: index "
        f"{start_idx} ({_format_time_str(result['advection_start_time'])}) -> "
        f"{finish_idx} ({_format_time_str(result['advection_finish_time'])})."
    )
    dt_stats = result["dt_seconds_stats"]
    if np.isfinite(dt_stats["mean"]):
        print(
            "[diag] Time-step stats (s): "
            f"min={dt_stats['min']:.1f}, max={dt_stats['max']:.1f}, "
            f"mean={dt_stats['mean']:.1f}."
        )
    else:
        print("[diag] Time-step stats unavailable (no backward integration steps executed).")

    # Keep only parcels that actually reached the cylinder
    arrived_mask_full = result["arrived"].copy()
    arrived_mask = arrived_mask_full
    arrived_indices = np.where(arrived_mask)[0]
    for key in ["j", "i", "k", "arrival_time", "arrival_z", "arrived"]:
        result[key] = result[key][arrived_mask]

    arrival_time_sec = result["arrival_time"]
    arrival_z = result["arrival_z"]
    arrival_counts = np.ones(arrival_time_sec.shape, dtype=float)
    parcels_mass_all = parcels_init.get("mass")
    if parcels_mass_all is not None:
        parcels_mass_all = np.asarray(parcels_mass_all, dtype=float)
    if parcels_mass_all is not None and parcels_mass_all.size == parcels_init["j"].size:
        arrival_mass = parcels_mass_all[arrived_indices]
    else:
        # This case might be hit if mass was not in parcels_init, which shouldn't happen
        # with current logic but is a safe fallback.
        arrival_mass = np.ones_like(arrival_time_sec, dtype=float)

    if times_arr.dtype.kind == "M":
        if emission_start_time is not None:
            t0 = emission_start_time
        else:
            t0 = times_arr[0]
        start_time_sec = (
            (times_arr[it_start] - t0) / np.timedelta64(1, "s")
        ).astype(float)
        start_time_sec = float(start_time_sec)
        arrival_time = t0 + arrival_time_sec.astype("timedelta64[s]")
    else:
        arrival_time = arrival_time_sec
        start_time_sec = float(times_arr[it_start])

    print(f"Total parcels reaching receptor: {result['j'].size}")
    if args.arrival_bin_minutes <= 0:
        print("arrival_bin_minutes <= 0, skipping time–height series.")
        return

    # --- SO2 mass correction for oxidation ---
    if args.so2_efolding_days is not None and args.so2_efolding_days > 0:
        print("[diag] Applying SO2 mass correction for oxidation...")
        efolding_time_days = args.so2_efolding_days
        efolding_time_sec = efolding_time_days * 86400.0
        original_mass_sum = arrival_mass.sum()

        # Calculate age and correction factor for each parcel individually
        corrected_masses = []
        for i in range(len(arrival_time_sec)):
            parcel_age_sec = start_time_sec - arrival_time_sec[i]
            # mass_emission = mass_receptor * exp(age / tau)
            mass_correction_factor = np.exp(parcel_age_sec / efolding_time_sec)
            corrected_mass = arrival_mass[i] * mass_correction_factor
            corrected_masses.append(corrected_mass)

        arrival_mass = np.array(corrected_masses)
        corrected_mass_sum = arrival_mass.sum()

        print(f"[diag] Applied SO2 mass correction with e-folding time of {efolding_time_days} days.")
        print(f"[diag] Total mass changed from {original_mass_sum:.3e} to {corrected_mass_sum:.3e}.")

    # Vertical discretisation: use WRF vertical layers over receptor cell at start time
    z_profile_receptor = z_center[it_start, :, j_rec, i_rec]
    nz = z_profile_receptor.size

    valid_z_mask = (z_profile_receptor >= receptor_min_h) & (z_profile_receptor <= receptor_max_h)
    k_valid = np.where(valid_z_mask)[0]
    if k_valid.size == 0:
        print("No vertical levels within cylinder bounds at receptor; nothing to output.")
        return

    z_bins = z_profile_receptor[k_valid]
    nz_bins = z_bins.size
    z_edges = compute_height_edges(z_bins)
    lower_edge = z_edges[0]
    upper_edge = z_edges[-1]

    # Map arriving parcels to vertical bins using their physical arrival height
    valid_vert = (arrival_z >= lower_edge) & (arrival_z <= upper_edge)
    if not np.any(valid_vert):
        print("No arrivals fall within vertical bins; emission grid will be zeros.")
    arrival_time_sec = arrival_time_sec[valid_vert]
    arrival_counts = arrival_counts[valid_vert]
    arrival_z = arrival_z[valid_vert]
    arrival_mass = arrival_mass[valid_vert]
    valid_parcel_indices = arrived_indices[valid_vert]

    z_bin = np.digitize(arrival_z, z_edges, right=False) - 1
    z_bin = np.clip(z_bin, 0, nz_bins - 1)

    emission_time_hours = np.maximum(arrival_time_sec, 0.0) / 3600.0
    arrival_age_hours = np.maximum(
        (start_time_sec - arrival_time_sec) / 3600.0, 0.0
    )

    # Time discretisation
    bin_width_sec = args.arrival_bin_minutes * 60.0
    emission = None
    time_labels = []
    time_edges_plot = None
    time_axis_mode = "numeric"

    if (
        emission_start_time is not None
        and emission_end_time is not None
        and times_arr.dtype.kind == "M"
    ):
        duration_sec = (
            (emission_end_time - emission_start_time) / np.timedelta64(1, "s")
        ).astype(float)
        if duration_sec <= 0:
            raise ValueError("Eruption duration must be positive.")
        
        n_time_bins = int(np.ceil(duration_sec / bin_width_sec))
        if n_time_bins == 0 and duration_sec > 0:
            n_time_bins = 1

        if n_time_bins <= 0:
            raise ValueError("Computed zero or negative time bins; adjust eruption window or bin width.")

        emission = np.zeros((nz_bins, n_time_bins), dtype=float)
        mass_emission = np.zeros_like(emission)
        time_axis_mode = "datetime"
        if arrival_time_sec.size > 0:
            # Parcels arriving after the eruption ends are still counted if they are within the last time bin
            valid_window = (arrival_time_sec >= 0.0) & (arrival_time_sec < duration_sec + bin_width_sec)
            arrival_time_win = arrival_time_sec[valid_window]
            arrival_counts_win = arrival_counts[valid_window]
            arrival_mass_win = arrival_mass[valid_window]
            z_bin_win = z_bin[valid_window]
            if arrival_time_win.size > 0:
                t_bin = np.floor(arrival_time_win / bin_width_sec).astype(int)
                t_bin = np.clip(t_bin, 0, n_time_bins - 1)
                np.add.at(emission, (z_bin_win, t_bin), arrival_counts_win)
                np.add.at(mass_emission, (z_bin_win, t_bin), arrival_mass_win)
        
        start_edges_sec = np.arange(n_time_bins + 1) * bin_width_sec
        time_edges_plot = emission_start_time + start_edges_sec.astype("timedelta64[s]")
        time_labels = [str(t) for t in time_edges_plot[:-1]]
    else:
        if arrival_time_sec.size == 0:
            print("No arrivals to bin; writing zeros based on single column.")
            n_time_bins = 1
            emission = np.zeros((nz_bins, n_time_bins), dtype=float)
            mass_emission = np.zeros_like(emission)
            if times_arr.dtype.kind == "M":
                time_axis_mode = "datetime"
                if emission_start_time is not None:
                    t_ref = emission_start_time
                else:
                    t_ref = times_arr[0]
                duration = max(bin_width_sec, 3600.0)
                time_edges_plot = np.array(
                    [t_ref, t_ref + np.timedelta64(int(duration), "s")],
                    dtype="datetime64[s]",
                )
                time_labels = [str(time_edges_plot[0])]
            else:
                step_width = max(bin_width_sec, 3600.0)
                time_edges_plot = np.array([0.0, step_width])
                time_labels = ["0.0"]
        else:
            t_bin_float = np.floor(arrival_time_sec / bin_width_sec)
            t_bin_int = t_bin_float.astype(int)
            t_bin_min = t_bin_int.min()
            t_bin_max = t_bin_int.max()
            n_time_bins = t_bin_max - t_bin_min + 1
            t_bin_shifted = t_bin_int - t_bin_min
            emission = np.zeros((nz_bins, n_time_bins), dtype=float)
            mass_emission = np.zeros_like(emission)
            np.add.at(emission, (z_bin, t_bin_shifted), arrival_counts)
            np.add.at(mass_emission, (z_bin, t_bin_shifted), arrival_mass)
            start_edges_sec = (t_bin_min + np.arange(n_time_bins + 1)) * bin_width_sec
            if times_arr.dtype.kind == "M":
                time_axis_mode = "datetime"
                if emission_start_time is not None:
                    t_ref = emission_start_time
                else:
                    t_ref = times_arr[0]
                time_edges_plot = (
                    t_ref + start_edges_sec.astype("timedelta64[s]")
                ).astype("datetime64[s]")
                time_labels = [str(t) for t in time_edges_plot[:-1]]
            else:
                time_edges_plot = start_edges_sec.astype(float)
                time_labels = [f"{sec:.1f}" for sec in time_edges_plot[:-1]]

    print(f"[diag] Parcel counts accumulated in matrix: {emission.sum():.0f}")
    print(f"[diag] Parcel mass accumulated in matrix: {mass_emission.sum():.3e}")

    # Write time–height emission series to TXT file
    out_path = args.output_txt
    with open(out_path, "w") as f:
        f.write("time " + " ".join(time_labels) + "\n")
        f.write("height " + " ".join(f"{z:.2f}" for z in z_bins) + "\n")
        for iz in range(nz_bins - 1, -1, -1):
            row_vals = " ".join(f"{int(round(emission[iz, jt]))}" for jt in range(n_time_bins))
            f.write(row_vals + "\n")

    print(f"Time–height emission series written to '{out_path}'.")
    if args.mass_output_txt:
        out_mass_path = args.mass_output_txt
        with open(out_mass_path, "w") as f:
            f.write("time " + " ".join(time_labels) + "\n")
            f.write("height " + " ".join(f"{z:.2f}" for z in z_bins) + "\n")
            for iz in range(nz_bins - 1, -1, -1):
                row_vals = " ".join(
                    f"{mass_emission[iz, jt]:.6e}" for jt in range(n_time_bins)
                )
                f.write(row_vals + "\n")
        print(f"Mass-weighted emission series written to '{out_mass_path}'.")

    z_edges = compute_height_edges(z_bins)
    if args.trajectory_age:
        saved = plot_parcel_age_map(
            column2d=column2d,
            xlat=xlat,
            xlon=xlon,
            trajectory_i=result["trajectory_i"],
            trajectory_j=result["trajectory_j"],
            trajectory_active=result["trajectory_active"],
            parcel_indices=valid_parcel_indices,
            parcel_ages_hours=arrival_age_hours,
            threshold=threshold,
            receptor_lat=receptor_lat,
            receptor_lon=receptor_lon,
            receptor_radius_m=receptor_radius_m,
            out_path=args.trajectory_age,
            colorbar_label=args.colorbar_label,
            figure_dpi=figure_dpi,
        )
        if saved:
            print(f"[diag] Parcel-age figure saved to '{args.trajectory_age}'.")

    if args.trajectory_emission_time_figure:
        saved_emission = plot_parcel_emission_time_map(
            column2d=column2d,
            xlat=xlat,
            xlon=xlon,
            trajectory_i=result["trajectory_i"],
            trajectory_j=result["trajectory_j"],
            trajectory_active=result["trajectory_active"],
            parcel_indices=valid_parcel_indices,
            parcel_emission_time_hours=emission_time_hours,
            threshold=threshold,
            receptor_lat=receptor_lat,
            receptor_lon=receptor_lon,
            receptor_radius_m=receptor_radius_m,
            out_path=args.trajectory_emission_time_figure,
            colorbar_label=args.colorbar_label,
            figure_dpi=figure_dpi,
            emission_start_time=emission_start_time,
        )
        if saved_emission:
            print(
                "[diag] Parcel emission-time figure saved to "
                f"'{args.trajectory_emission_time_figure}'."
            )

    if args.trajectory_arrival_height_figure:
        saved_height = plot_parcel_arrival_height_map(
            column2d=column2d,
            xlat=xlat,
            xlon=xlon,
            trajectory_i=result["trajectory_i"],
            trajectory_j=result["trajectory_j"],
            trajectory_active=result["trajectory_active"],
            parcel_indices=valid_parcel_indices,
            parcel_arrival_height_m=arrival_z,
            threshold=threshold,
            receptor_lat=receptor_lat,
            receptor_lon=receptor_lon,
            receptor_radius_m=receptor_radius_m,
            out_path=args.trajectory_arrival_height_figure,
            colorbar_label=args.colorbar_label,
            figure_dpi=figure_dpi,
            height_min=z_min,
            height_max=z_max,
        )
        if saved_height:
            print(
                "[diag] Parcel arrival-height figure saved to "
                f"'{args.trajectory_arrival_height_figure}'."
            )

    figure_path = args.output_figure
    if time_edges_plot is None:
        print("[diag] No time edges available; skipping emission figure.")
    else:
        plot_emission_matrix(
            emission=emission,
            time_edges=time_edges_plot,
            z_bins=z_bins,
            z_edges=z_edges,
            time_labels=time_labels,
            out_path=figure_path,
            time_axis_mode=time_axis_mode,
            total_parcels=result["j"].size,
            figure_dpi=figure_dpi,
        )
        print(f"Emission matrix figure saved to '{figure_path}'.")
        if args.mass_figure:
            plot_emission_matrix(
                emission=mass_emission,
                time_edges=time_edges_plot,
                z_bins=z_bins,
                z_edges=z_edges,
                time_labels=time_labels,
                out_path=args.mass_figure,
                time_axis_mode=time_axis_mode,
                total_parcels=None,
                figure_dpi=figure_dpi,
                colorbar_label="Parcel mass",
            )
            print(f"[diag] Mass-weighted emission figure saved to '{args.mass_figure}'.")
    if args.mass_figure and time_edges_plot is None:
        print("[diag] No time edges available; skipping mass-weighted emission figure.")

    if args.state_pickle:
        args_dict = {k: getattr(args, k) for k in vars(args)}
        pickle_payload = dict(
            args=args_dict,
            grid=dict(
                xlat=xlat,
                xlon=xlon,
            ),
            initial_parcels=dict(
                j=parcels_init["j"],
                i=parcels_init["i"],
                k=parcels_init["k"],
                z_init=parcels_init.get("z_init"),
                mass=parcels_init.get("mass"),
            ),
            column=dict(
                field=column2d,
                threshold=threshold,
                colorbar_label=args.colorbar_label,
            ),
            trajectories=dict(
                times=result["trajectory_times"],
                i=result["trajectory_i"],
                j=result["trajectory_j"],
                active=result["trajectory_active"],
                arrived_mask=arrived_mask_full,
                arrival_indices=arrived_indices,
                indices_in_bins=valid_parcel_indices,
                emission_time_hours=emission_time_hours,
                arrival_age_hours=arrival_age_hours,
                final_z=result["final_z"],
                arrival_z=arrival_z,
                arrival_height_m=arrival_z,
                initial_height_m=parcels_init.get("z_init"),
            ),
            emission=dict(
                matrix=emission,
                mass_matrix=mass_emission,
                time_edges=time_edges_plot,
                time_labels=time_labels,
                z_bins=z_bins,
                z_edges=z_edges,
                time_axis_mode=time_axis_mode,
                total_parcels=int(arrived_mask_full.sum()),
            ),
            metadata=dict(
                start_time=times_arr[it_start],
                start_time_index=it_start,
                finish_time_index=finish_idx,
                parcels_initialized=parcels_init["j"].size,
                receptor=dict(lat=receptor_lat, lon=receptor_lon, radius_m=receptor_radius_m),
                receptor_min_h=receptor_min_h,
                receptor_max_h=receptor_max_h,
                emission_start=emission_start_time,
                emission_end=emission_end_time,
            ),
        )
        with open(args.state_pickle, "wb") as fh:
            pickle.dump(pickle_payload, fh)
        print(f"[diag] Saved processing state to '{args.state_pickle}'.")


if __name__ == "__main__":
    main(parse_args())
