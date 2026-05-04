import math
import pickle
from glob import glob
from pathlib import Path

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import BoundaryNorm, ListedColormap
from netCDF4 import Dataset
from scipy.spatial import cKDTree

from misc.map_style import apply_map_style
from misc.settling_velocity_data import SETTLING_VEL_MS, Z_M
from .plume_base import PlumeBackend
from .plotting_base import plot_seed_bbox


PLATE_CARREE = ccrs.PlateCarree()
EARTH_RADIUS_M = 6371229.0
PARCEL_MARKER_SIZE = 10.0
PARCEL_MARKER_ALPHA = 0.75
PARCEL_MARKER_EDGE = (0.1, 0.1, 0.1)
PARCEL_MARKER_LINEWIDTH = 0.2
TRAJECTORY_LINEWIDTH = 0.7
TRAJECTORY_LINESTYLE = "-"
TRAJECTORY_ALPHA = 0.65


def _diag(msg):
    print(f"[diag] {msg}")


def _ensure_parent_dir(path):
    if not path:
        return
    Path(path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)


def _expand_input_paths(input_args):
    expanded = []
    for entry in input_args:
        if any(ch in entry for ch in ["*", "?", "["]):
            expanded.extend(sorted(glob(entry)))
        else:
            expanded.append(entry)
    expanded = [p for p in expanded if p]
    if not expanded:
        raise ValueError("No input files matched the provided pattern(s).")
    return expanded


def _as_time_array(ds):
    if "xtime" not in ds.variables:
        if "Time" in ds.variables and np.asarray(ds.variables["Time"][:]).dtype.kind == "M":
            return np.asarray(ds.variables["Time"][:])
        raise ValueError("MPAS history file does not contain xtime or datetime Time.")
    xtime = ds.variables["xtime"][:]
    out = []
    for row in xtime:
        if hasattr(row, "tobytes"):
            s = row.tobytes().decode("ascii").strip()
        else:
            s = bytes(row).decode("ascii").strip()
        out.append(np.datetime64(s.replace("_", "T")))
    return np.asarray(out)


def _normalize_lons_deg(lon_deg):
    return ((lon_deg + 180.0) % 360.0) - 180.0


def _sphere_xyz(lat_deg, lon_deg):
    lat = np.deg2rad(np.asarray(lat_deg, dtype=float))
    lon = np.deg2rad(np.asarray(lon_deg, dtype=float))
    clat = np.cos(lat)
    return np.column_stack((clat * np.cos(lon), clat * np.sin(lon), np.sin(lat)))


def _build_tree(lat_deg, lon_deg):
    return cKDTree(_sphere_xyz(lat_deg, lon_deg))


def _nearest_cells(tree, lat_deg, lon_deg):
    pts = _sphere_xyz(lat_deg, lon_deg)
    return tree.query(pts, k=1)[1]


def _read_mpas_file(path):
    with Dataset(path, "r") as ds:
        if "latCell" not in ds.variables or "lonCell" not in ds.variables:
            raise ValueError(f"MPAS file '{path}' is missing latCell/lonCell.")
        lat = np.asarray(ds.variables["latCell"][:], dtype=float)
        lon = np.asarray(ds.variables["lonCell"][:], dtype=float)
        lat_deg = np.rad2deg(lat)
        lon_deg = _normalize_lons_deg(np.rad2deg(lon))
        area = np.asarray(ds.variables["areaCell"][:], dtype=float) if "areaCell" in ds.variables else np.ones_like(lat_deg)

        if "zgrid" not in ds.variables:
            raise ValueError(f"MPAS file '{path}' is missing zgrid.")
        zgrid = np.asarray(ds.variables["zgrid"][:], dtype=float)
        if zgrid.ndim != 2:
            raise ValueError("MPAS zgrid must be 2-D (nCells, nVertLevelsP1).")
        zmid = 0.5 * (zgrid[:, :-1] + zgrid[:, 1:])
        dz = np.diff(zgrid, axis=1)
        if np.any(dz <= 0):
            raise ValueError("MPAS zgrid must increase monotonically with height.")

        if "uReconstructZonal" not in ds.variables or "uReconstructMeridional" not in ds.variables:
            raise ValueError(f"MPAS file '{path}' is missing uReconstructZonal/uReconstructMeridional.")
        u = np.asarray(ds.variables["uReconstructZonal"][:], dtype=float)
        v = np.asarray(ds.variables["uReconstructMeridional"][:], dtype=float)
        if "w" not in ds.variables:
            raise ValueError(f"MPAS file '{path}' is missing w.")
        w = np.asarray(ds.variables["w"][:], dtype=float)
        if w.shape[-1] == zgrid.shape[1]:
            w = 0.5 * (w[..., :-1] + w[..., 1:])
        if w.shape[-1] != zmid.shape[1]:
            raise ValueError("MPAS w vertical dimension does not match zgrid.")

        times = _as_time_array(ds)
        if times.size > 0:
            _diag(
                "MPAS time axis: "
                f"{times.size} steps from {times[0]} to {times[-1]}."
            )
        return dict(
            lat_deg=lat_deg,
            lon_deg=lon_deg,
            area=area,
            zgrid=zgrid,
            zmid=zmid,
            dz=dz,
            u=u,
            v=v,
            w=w,
            times=times,
        )


def read_mpas_history(input_args):
    paths = _expand_input_paths(input_args)
    _diag(f"Reading {len(paths)} MPAS history file(s).")
    for path in paths:
        _diag(f"  input: {path}")
    parts = [_read_mpas_file(path) for path in paths]
    base = parts[0]
    for idx, part in enumerate(parts[1:], start=1):
        if part["lat_deg"].shape != base["lat_deg"].shape:
            raise ValueError(f"MPAS nCells mismatch in file {paths[idx]}.")
        if not np.allclose(part["lat_deg"], base["lat_deg"], equal_nan=True):
            raise ValueError(f"MPAS latCell mismatch in file {paths[idx]}.")
        if not np.allclose(part["lon_deg"], base["lon_deg"], equal_nan=True):
            raise ValueError(f"MPAS lonCell mismatch in file {paths[idx]}.")
        if not np.allclose(part["zgrid"], base["zgrid"], equal_nan=True):
            raise ValueError(f"MPAS zgrid mismatch in file {paths[idx]}.")

    out = dict(base)
    out["u"] = np.concatenate([p["u"] for p in parts], axis=0)
    out["v"] = np.concatenate([p["v"] for p in parts], axis=0)
    out["w"] = np.concatenate([p["w"] for p in parts], axis=0)
    out["times"] = np.concatenate([p["times"] for p in parts], axis=0)
    _diag(
        "MPAS concatenated time axis: "
        f"{out['times'].size} steps from {out['times'][0]} to {out['times'][-1]}."
    )
    out["tree"] = _build_tree(out["lat_deg"], out["lon_deg"])
    out["cell_xyz"] = _sphere_xyz(out["lat_deg"], out["lon_deg"])
    out["triangulation"] = mtri.Triangulation(out["lon_deg"], out["lat_deg"])
    return out


def _interp_profile(z, profile, values):
    if profile.size < 2:
        return np.nan
    return np.interp(z, profile, values, left=np.nan, right=np.nan)


def _find_time_bounds(times, time_value):
    if time_value <= times[0]:
        return 0, 0, 0.0
    if time_value >= times[-1]:
        last = times.size - 1
        return last, last, 0.0
    hi = int(np.searchsorted(times, time_value, side="right"))
    lo = hi - 1
    span = (times[hi] - times[lo]) / np.timedelta64(1, "s")
    frac = 0.0 if span <= 0 else float((time_value - times[lo]) / np.timedelta64(1, "s") / span)
    return lo, hi, frac


def _prepare_time_seconds(times, reference=None):
    if times.dtype.kind != "M":
        return np.asarray(times, dtype=float), reference
    if reference is None:
        reference = times[0]
    return ((times - reference) / np.timedelta64(1, "s")).astype(float), reference


def generate_parcels_from_column(column, lat_deg, lon_deg, zmid, time_index, threshold, n_columns, n_vert, z_min, z_max, seed_bbox=None, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    column = np.asarray(column, dtype=float).reshape(-1)
    mask = np.isfinite(column) & (column >= threshold)
    if seed_bbox is not None:
        lon_min, lat_min, lon_max, lat_max = seed_bbox
        if lon_min > lon_max:
            lon_min, lon_max = lon_max, lon_min
        if lat_min > lat_max:
            lat_min, lat_max = lat_max, lat_min
        mask &= (lon_deg >= lon_min) & (lon_deg <= lon_max) & (lat_deg >= lat_min) & (lat_deg <= lat_max)
    cells = np.where(mask)[0]
    if cells.size == 0:
        raise ValueError("No MPAS cells above threshold.")
    choice = rng.choice(cells, size=n_columns, replace=(n_columns > cells.size))
    n_parcels = n_columns * n_vert
    lon = np.empty(n_parcels, dtype=float)
    lat = np.empty(n_parcels, dtype=float)
    z = np.empty(n_parcels, dtype=float)
    cell = np.empty(n_parcels, dtype=int)
    idx = 0
    for c in choice:
        z_targets = np.sort(rng.uniform(z_min, z_max, size=n_vert))
        for zz in z_targets:
            lon[idx] = lon_deg[c]
            lat[idx] = lat_deg[c]
            z[idx] = zz
            cell[idx] = c
            idx += 1
    return dict(lon=lon, lat=lat, z=z, cell=cell, z_init=z.copy())


def generate_parcels_from_point(lon_deg, lat_deg, cell_idx, n_vert, z_min, z_max, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    if n_vert <= 0:
        return {"lon": np.array([]), "lat": np.array([]), "z": np.array([]), "cell": np.array([], dtype=int), "z_init": np.array([])}
    if n_vert == 1:
        z_targets = np.array([0.5 * (z_min + z_max)], dtype=float)
    else:
        z_targets = np.linspace(z_min, z_max, num=n_vert, dtype=float)
    return dict(
        lon=np.full(n_vert, float(lon_deg), dtype=float),
        lat=np.full(n_vert, float(lat_deg), dtype=float),
        z=z_targets.copy(),
        cell=np.full(n_vert, int(cell_idx), dtype=int),
        z_init=z_targets.copy(),
    )


def _velocities(lat, lon, z, current_time_sec, times_sec, time_index_lo, time_index_hi, u, v, w, zmid, tree):
    t0 = float(times_sec[time_index_lo])
    t1 = float(times_sec[time_index_hi])
    current_time = float(current_time_sec)
    span = max(t1 - t0, 1.0)
    frac = np.clip((current_time - t0) / span, 0.0, 1.0)
    idx = _nearest_cells(tree, lat, lon)
    u0 = np.empty(lat.size, dtype=float)
    v0 = np.empty_like(u0)
    w0 = np.empty_like(u0)
    u1 = np.empty_like(u0)
    v1 = np.empty_like(u0)
    w1 = np.empty_like(u0)
    for p in range(lat.size):
        c = idx[p]
        u0[p] = _interp_profile(z[p], zmid[c], u[time_index_lo, c])
        v0[p] = _interp_profile(z[p], zmid[c], v[time_index_lo, c])
        w0[p] = _interp_profile(z[p], zmid[c], w[time_index_lo, c])
        u1[p] = _interp_profile(z[p], zmid[c], u[time_index_hi, c])
        v1[p] = _interp_profile(z[p], zmid[c], v[time_index_hi, c])
        w1[p] = _interp_profile(z[p], zmid[c], w[time_index_hi, c])
    uu = (1.0 - frac) * u0 + frac * u1
    vv = (1.0 - frac) * v0 + frac * v1
    ww = (1.0 - frac) * w0 + frac * w1
    dlat = vv / EARTH_RADIUS_M * 180.0 / math.pi
    dlon = uu / (EARTH_RADIUS_M * np.cos(np.deg2rad(np.clip(lat, -89.9, 89.9)))) * 180.0 / math.pi
    dz = ww
    bad = ~np.isfinite(dlat) | ~np.isfinite(dlon) | ~np.isfinite(dz)
    dlat[bad] = 0.0
    dlon[bad] = 0.0
    dz[bad] = 0.0
    return np.column_stack((dlon, dlat, dz)), bad


def advect_parcels_forward(parcels, times, u, v, w, zmid, tree, start_time_index, end_time_index, integration_dt):
    times_sec, t_ref = _prepare_time_seconds(times)
    lon = parcels["lon"].copy()
    lat = parcels["lat"].copy()
    z = parcels["z"].copy()
    active = np.ones(lon.size, dtype=bool)
    lon_hist = [lon.copy()]
    lat_hist = [lat.copy()]
    z_hist = [z.copy()]
    active_hist = [active.copy()]
    time_indices = [int(start_time_index)]
    current_sec = times_sec[start_time_index]
    for it in range(int(start_time_index), int(end_time_index)):
        _diag(
            "Processing time index "
            f"{it} ({times[it]}) -> {it + 1} ({times[it + 1]})."
        )
        dt_total = times_sec[it + 1] - times_sec[it]
        if dt_total <= 0:
            continue
        sub_time = current_sec
        dt_remaining = dt_total
        while dt_remaining > 0 and active.any():
            dt_step = min(integration_dt, dt_remaining)
            idxs = np.where(active)[0]
            pos = np.column_stack((lon[idxs], lat[idxs], z[idxs]))
            v1, bad1 = _velocities(lat[idxs], lon[idxs], z[idxs], sub_time, times_sec, it, it + 1, u, v, w, zmid, tree)
            k1 = dt_step * v1
            v2, bad2 = _velocities(lat[idxs] + 0.5 * k1[:, 1], lon[idxs] + 0.5 * k1[:, 0], z[idxs] + 0.5 * k1[:, 2], sub_time + 0.5 * dt_step, times_sec, it, it + 1, u, v, w, zmid, tree)
            k2 = dt_step * v2
            v3, bad3 = _velocities(lat[idxs] + 0.5 * k2[:, 1], lon[idxs] + 0.5 * k2[:, 0], z[idxs] + 0.5 * k2[:, 2], sub_time + 0.5 * dt_step, times_sec, it, it + 1, u, v, w, zmid, tree)
            k3 = dt_step * v3
            v4, bad4 = _velocities(lat[idxs] + k3[:, 1], lon[idxs] + k3[:, 0], z[idxs] + k3[:, 2], sub_time + dt_step, times_sec, it, it + 1, u, v, w, zmid, tree)
            k4 = dt_step * v4
            bad = bad1 | bad2 | bad3 | bad4
            if bad.any():
                active[idxs[bad]] = False
                good = ~bad
                if not good.any():
                    dt_remaining -= dt_step
                    sub_time += dt_step
                    continue
                idxs = idxs[good]
                k1 = k1[good]
                k2 = k2[good]
                k3 = k3[good]
                k4 = k4[good]
            delta = (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
            lon[idxs] += delta[:, 0]
            lat[idxs] += delta[:, 1]
            z[idxs] += delta[:, 2]
            out = (z[idxs] < 0.0)
            if out.any():
                z[idxs[out]] = 0.0
                active[idxs[out]] = False
            dt_remaining -= dt_step
            sub_time += dt_step
        current_sec = sub_time
        lon_hist.append(lon.copy())
        lat_hist.append(lat.copy())
        z_hist.append(z.copy())
        active_hist.append(active.copy())
        time_indices.append(it + 1)
    return dict(
        lon=lon,
        lat=lat,
        z=z,
        trajectory_times=np.asarray(times)[np.asarray(time_indices, dtype=int)],
        trajectory_lon=np.stack(lon_hist, axis=0),
        trajectory_lat=np.stack(lat_hist, axis=0),
        trajectory_z=np.stack(z_hist, axis=0),
        trajectory_active=np.stack(active_hist, axis=0),
        trajectory_time_indices=np.asarray(time_indices, dtype=int),
        advection_start_index=int(start_time_index),
        advection_finish_index=int(end_time_index),
        advection_start_time=times[start_time_index],
        advection_finish_time=times[end_time_index],
    )


def advect_parcels_backward(parcels, times, u, v, w, zmid, tree, receptor_lat, receptor_lon, receptor_radius_m, receptor_min_h, receptor_max_h, start_time_index, integration_dt, emission_start_time=None, emission_end_time=None):
    times_sec, t_ref = _prepare_time_seconds(times)
    lon = parcels["lon"].copy()
    lat = parcels["lat"].copy()
    z = parcels["z"].copy()
    active = np.ones(lon.size, dtype=bool)
    arrived = np.zeros(lon.size, dtype=bool)
    arrival_time = np.full(lon.size, np.nan, dtype=float)
    arrival_z = np.full(lon.size, np.nan, dtype=float)
    lon_hist = [lon.copy()]
    lat_hist = [lat.copy()]
    z_hist = [z.copy()]
    active_hist = [active.copy()]
    time_indices = [int(start_time_index)]
    time_hist = [times[start_time_index]]
    current_sec = times_sec[start_time_index]
    for it in range(int(start_time_index), 0, -1):
        _diag(
            "Processing time index "
            f"{it} ({times[it]}) -> {it - 1} ({times[it - 1]})."
        )
        dt_total = times_sec[it] - times_sec[it - 1]
        if dt_total <= 0:
            continue
        sub_time = current_sec
        dt_remaining = dt_total
        while dt_remaining > 0 and active.any():
            dt_step = min(integration_dt, dt_remaining)
            idxs = np.where(active)[0]
            v1, bad1 = _velocities(lat[idxs], lon[idxs], z[idxs], sub_time, times_sec, it - 1, it, u, v, w, zmid, tree)
            k1 = -dt_step * v1
            v2, bad2 = _velocities(lat[idxs] + 0.5 * k1[:, 1], lon[idxs] + 0.5 * k1[:, 0], z[idxs] + 0.5 * k1[:, 2], sub_time - 0.5 * dt_step, times_sec, it - 1, it, u, v, w, zmid, tree)
            k2 = -dt_step * v2
            v3, bad3 = _velocities(lat[idxs] + 0.5 * k2[:, 1], lon[idxs] + 0.5 * k2[:, 0], z[idxs] + 0.5 * k2[:, 2], sub_time - 0.5 * dt_step, times_sec, it - 1, it, u, v, w, zmid, tree)
            k3 = -dt_step * v3
            v4, bad4 = _velocities(lat[idxs] + k3[:, 1], lon[idxs] + k3[:, 0], z[idxs] + k3[:, 2], sub_time - dt_step, times_sec, it - 1, it, u, v, w, zmid, tree)
            k4 = -dt_step * v4
            bad = bad1 | bad2 | bad3 | bad4
            if bad.any():
                active[idxs[bad]] = False
                good = ~bad
                if not good.any():
                    dt_remaining -= dt_step
                    sub_time -= dt_step
                    continue
                idxs = idxs[good]
                k1 = k1[good]
                k2 = k2[good]
                k3 = k3[good]
                k4 = k4[good]
            delta = (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
            lon[idxs] += delta[:, 0]
            lat[idxs] += delta[:, 1]
            z[idxs] += delta[:, 2]
            out = z[idxs] < 0.0
            if out.any():
                z[idxs[out]] = 0.0
                active[idxs[out]] = False
            dt_remaining -= dt_step
            sub_time -= dt_step
            # receptor hit test
            dlat = np.deg2rad(lat[idxs] - receptor_lat)
            dlon = np.deg2rad(lon[idxs] - receptor_lon) * np.cos(np.deg2rad(receptor_lat))
            horiz = np.sqrt((EARTH_RADIUS_M * dlat) ** 2 + (EARTH_RADIUS_M * dlon) ** 2)
            hit = (horiz <= receptor_radius_m) & (z[idxs] >= receptor_min_h) & (z[idxs] <= receptor_max_h)
            if hit.any():
                hit_idxs = idxs[hit]
                arrived[hit_idxs] = True
                arrival_time[hit_idxs] = max(sub_time, 0.0)
                arrival_z[hit_idxs] = z[hit_idxs]
                active[hit_idxs] = False
        current_sec = sub_time
        lon_hist.append(lon.copy())
        lat_hist.append(lat.copy())
        z_hist.append(z.copy())
        active_hist.append(active.copy())
        time_indices.append(it - 1)
        time_hist.append(times[it - 1])
    return dict(
        lon=lon,
        lat=lat,
        z=z,
        arrived=arrived,
        arrival_time=arrival_time,
        arrival_z=arrival_z,
        trajectory_lon=np.stack(lon_hist, axis=0),
        trajectory_lat=np.stack(lat_hist, axis=0),
        trajectory_z=np.stack(z_hist, axis=0),
        trajectory_active=np.stack(active_hist, axis=0),
        trajectory_time_indices=np.asarray(time_indices, dtype=int),
        trajectory_times=np.asarray(time_hist),
        advection_start_index=int(start_time_index),
        advection_finish_index=0,
        advection_start_time=times[start_time_index],
        advection_finish_time=times[0],
    )


def _setup_geo_axes(lat_deg, lon_deg, map_extent=None, figsize=(10, 8)):
    fig, ax = plt.subplots(figsize=figsize, subplot_kw={"projection": PLATE_CARREE})
    lon_min, lon_max = float(np.nanmin(lon_deg)), float(np.nanmax(lon_deg))
    lat_min, lat_max = float(np.nanmin(lat_deg)), float(np.nanmax(lat_deg))
    pad_lon = max((lon_max - lon_min) * 0.05, 0.1)
    pad_lat = max((lat_max - lat_min) * 0.05, 0.1)
    if map_extent is None:
        ax.set_extent([lon_min - pad_lon, lon_max + pad_lon, lat_min - pad_lat, lat_max + pad_lat], crs=PLATE_CARREE)
    else:
        west, south, east, north = map_extent
        ax.set_extent([west, east, south, north], crs=PLATE_CARREE)
    apply_map_style(ax, draw_labels=False, label_size=10)
    return fig, ax


def _plot_background(ax, lon_deg, lat_deg, values, cmap="plasma", levels=30, threshold=None):
    values = np.asarray(values, dtype=float)
    mask = np.isfinite(values) & np.isfinite(lon_deg) & np.isfinite(lat_deg)
    tri = mtri.Triangulation(lon_deg[mask], lat_deg[mask])
    mesh = ax.tricontourf(tri, values[mask], levels=levels, cmap=cmap, transform=PLATE_CARREE)
    if threshold is not None:
        ax.tricontour(tri, values[mask], levels=[threshold], colors="white", linewidths=1.0, transform=PLATE_CARREE)
    return mesh


def plot_mpas_column_and_parcels(column, lat_deg, lon_deg, parcels, out_path, threshold=None, receptor_lat=None, receptor_lon=None, receptor_radius_m=None, seed_bbox=None, title=None, colorbar_label="Column value", figure_dpi=200, map_extent=None):
    fig, ax = _setup_geo_axes(lat_deg, lon_deg, map_extent=map_extent)
    mesh = _plot_background(ax, lon_deg, lat_deg, column, threshold=threshold)
    ax.scatter(parcels["lon"], parcels["lat"], s=PARCEL_MARKER_SIZE, c="red", edgecolors=PARCEL_MARKER_EDGE, linewidths=PARCEL_MARKER_LINEWIDTH, alpha=PARCEL_MARKER_ALPHA, transform=PLATE_CARREE, zorder=5)
    if (
        receptor_lat is not None
        and receptor_lon is not None
        and receptor_radius_m is not None
        and receptor_radius_m > 0
    ):
        ang = np.linspace(0, 2 * np.pi, 181)
        lat_scale = 111320.0
        lon_scale = max(np.cos(np.deg2rad(receptor_lat)) * 111320.0, 1e-6)
        ax.plot(
            receptor_lon + (receptor_radius_m / lon_scale) * np.cos(ang),
            receptor_lat + (receptor_radius_m / lat_scale) * np.sin(ang),
            color="red",
            linewidth=2.0,
            transform=PLATE_CARREE,
        )
    plot_seed_bbox(ax, seed_bbox)
    cax = fig.add_axes([0.2, 0.05, 0.6, 0.03])
    plt.colorbar(mesh, cax=cax, orientation="horizontal", label=colorbar_label)
    text_str = f"Total parcels initialized: {parcels['lon'].size}"
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
    if title:
        ax.set_title(title)
    fig.savefig(out_path, dpi=figure_dpi)
    plt.close(fig)


def plot_mpas_trajectories(column, lat_deg, lon_deg, traj_lon, traj_lat, traj_active, parcel_indices, parcel_values, out_path, threshold=None, receptor_lat=None, receptor_lon=None, receptor_radius_m=None, seed_bbox=None, title=None, colorbar_label="Value", figure_dpi=200, map_extent=None, value_unit_label=None):
    fig, ax = _setup_geo_axes(lat_deg, lon_deg, map_extent=map_extent)
    _plot_background(ax, lon_deg, lat_deg, column, threshold=threshold)
    parcel_indices = np.asarray(parcel_indices, dtype=int)
    values = np.asarray(parcel_values, dtype=float)
    bins = None
    cmap = None
    tick_centers = None
    if parcel_indices.size == 0 or values.size == 0:
        parcel_indices = np.array([], dtype=int)
        values = np.array([], dtype=float)
    else:
        n = min(parcel_indices.size, values.size)
        parcel_indices = parcel_indices[:n]
        values = values[:n]
        finite = np.isfinite(values)
        parcel_indices = parcel_indices[finite]
        values = values[finite]
    if parcel_indices.size > 0:
        n_bins = min(10, max(2, int(np.sqrt(parcel_indices.size))))
        vmin = float(np.nanmin(values))
        vmax = float(np.nanmax(values))
        if np.isclose(vmin, vmax):
            vmax = vmin + 1.0
        bins = np.linspace(vmin, vmax, n_bins + 1)
        cmap = plt.get_cmap("rainbow", n_bins)
        palette = cmap(np.arange(n_bins))
        color_idx = np.clip(np.digitize(values, bins) - 1, 0, n_bins - 1)
        colors = palette[color_idx]
        lon = traj_lon[:, parcel_indices]
        lat = traj_lat[:, parcel_indices]
        active = np.asarray(traj_active, dtype=bool)[:, parcel_indices]
        segments = np.array([
            np.stack([lon[:-1, :], lat[:-1, :]], axis=2),
            np.stack([lon[1:, :], lat[1:, :]], axis=2),
        ]).transpose(2, 1, 0, 3)
        active_mask = (active[:-1, :] & active[1:, :]).T
        segs = segments[active_mask]
        seg_colors = np.repeat(colors, active_mask.sum(axis=1), axis=0)
        lc = LineCollection(segs, colors=seg_colors, linewidths=TRAJECTORY_LINEWIDTH, alpha=TRAJECTORY_ALPHA, transform=PLATE_CARREE)
        ax.add_collection(lc)
        ax.scatter(lon[0, :], lat[0, :], s=PARCEL_MARKER_SIZE, c=colors, edgecolors=PARCEL_MARKER_EDGE, linewidths=PARCEL_MARKER_LINEWIDTH, alpha=PARCEL_MARKER_ALPHA, transform=PLATE_CARREE, zorder=6)
        tick_centers = bins[:-1] + 0.5 * np.diff(bins)
    if (
        receptor_lat is not None
        and receptor_lon is not None
        and receptor_radius_m is not None
        and receptor_radius_m > 0
    ):
        ang = np.linspace(0, 2 * np.pi, 181)
        lat_scale = 111320.0
        lon_scale = max(np.cos(np.deg2rad(receptor_lat)) * 111320.0, 1e-6)
        ax.plot(
            receptor_lon + (receptor_radius_m / lon_scale) * np.cos(ang),
            receptor_lat + (receptor_radius_m / lat_scale) * np.sin(ang),
            color="red",
            linewidth=2.0,
            transform=PLATE_CARREE,
        )
    plot_seed_bbox(ax, seed_bbox)
    cax = fig.add_axes([0.2, 0.05, 0.6, 0.03])
    if bins is not None and cmap is not None and tick_centers is not None:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=BoundaryNorm(bins, cmap.N))
        sm.set_array([])
        cb = plt.colorbar(
            sm,
            cax=cax,
            orientation="horizontal",
            boundaries=bins,
            ticks=tick_centers,
        )
        cb.set_ticklabels([f"{val:.2f}" for val in tick_centers])
        cb.set_label(colorbar_label)
    if title:
        ax.set_title(title)
    fig.savefig(out_path, dpi=figure_dpi)
    plt.close(fig)


def plot_mpas_missed_trajectories(column, lat_deg, lon_deg, traj_lon, traj_lat, traj_active, arrived_mask, initial_heights_m, out_path, threshold=None, receptor_lat=None, receptor_lon=None, receptor_radius_m=None, seed_bbox=None, z_min=None, z_max=None, figure_dpi=200, map_extent=None):
    missed_flags = ~np.asarray(arrived_mask, dtype=bool)
    parcel_indices = np.where(missed_flags)[0]
    if parcel_indices.size == 0:
        _diag("All parcels reached the receptor; skipping missed-trajectory figure.")
        return False

    fig, ax = _setup_geo_axes(lat_deg, lon_deg, map_extent=map_extent)
    _ = _plot_background(ax, lon_deg, lat_deg, column, threshold=threshold)

    z_range_min = z_min if z_min is not None else 0.0
    z_range_max = z_max if z_max is not None else 30000.0
    if not np.isfinite(z_range_min):
        z_range_min = 0.0
    if not np.isfinite(z_range_max) or z_range_max <= z_range_min:
        z_range_max = z_range_min + 1.0

    n_bins = 10
    color_bins = np.linspace(z_range_min, z_range_max, n_bins + 1)
    cmap = plt.get_cmap("rainbow", n_bins)
    palette = cmap(np.arange(n_bins))

    init_heights_arr = np.asarray(initial_heights_m, dtype=float)[parcel_indices]
    default_height = 0.5 * (z_range_min + z_range_max)
    heights_for_color = np.where(np.isfinite(init_heights_arr), init_heights_arr, default_height)
    color_idx = np.digitize(heights_for_color, color_bins) - 1
    color_idx = np.clip(color_idx, 0, n_bins - 1)
    color_lookup = palette[color_idx]
    missing_mask = ~np.isfinite(init_heights_arr)
    if missing_mask.any():
        color_lookup[missing_mask] = np.array([0.6, 0.6, 0.6, 1.0])

    lon = np.asarray(traj_lon)[:, parcel_indices]
    lat = np.asarray(traj_lat)[:, parcel_indices]
    active_sub = np.asarray(traj_active, dtype=bool)[:, parcel_indices]
    segments = np.array([
        np.stack([lon[:-1, :], lat[:-1, :]], axis=2),
        np.stack([lon[1:, :], lat[1:, :]], axis=2),
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

    last_active_time_idx = np.asarray(traj_active, dtype=bool)[:, parcel_indices].sum(axis=0) - 1
    last_active_time_idx = np.clip(last_active_time_idx, 0, np.asarray(traj_lat).shape[0] - 1)
    finish_lon = np.asarray(traj_lon)[last_active_time_idx, parcel_indices]
    finish_lat = np.asarray(traj_lat)[last_active_time_idx, parcel_indices]
    valid_finish = np.isfinite(finish_lon) & np.isfinite(finish_lat)
    if valid_finish.any():
        ax.scatter(
            finish_lon[valid_finish],
            finish_lat[valid_finish],
            s=PARCEL_MARKER_SIZE,
            c=color_lookup[valid_finish],
            marker="o",
            edgecolors=PARCEL_MARKER_EDGE,
            linewidths=PARCEL_MARKER_LINEWIDTH,
            zorder=7,
            transform=PLATE_CARREE,
            alpha=PARCEL_MARKER_ALPHA,
        )

    if (
        receptor_lat is not None
        and receptor_lon is not None
        and receptor_radius_m is not None
        and receptor_radius_m > 0
    ):
        ang = np.linspace(0, 2 * np.pi, 181)
        lat_scale = 111320.0
        lon_scale = max(np.cos(np.deg2rad(receptor_lat)) * 111320.0, 1e-6)
        ax.plot(
            receptor_lon + (receptor_radius_m / lon_scale) * np.cos(ang),
            receptor_lat + (receptor_radius_m / lat_scale) * np.sin(ang),
            color="red",
            linewidth=2.0,
            transform=PLATE_CARREE,
            zorder=8,
        )

    plot_seed_bbox(ax, seed_bbox)
    ax.set_xlabel("")
    ax.set_ylabel("")
    finite_init = init_heights_arr[np.isfinite(init_heights_arr)]
    title_suffix = ""
    if finite_init.size:
        min_init = float(np.min(finite_init))
        max_init = float(np.max(finite_init))
        title_suffix = f" (min={min_init/1000.0:.2f} km, max={max_init/1000.0:.2f} km)"
    ax.set_title("Trajectories of Parcels Missing the Receptor" + title_suffix)

    sm = plt.cm.ScalarMappable(cmap=ListedColormap(palette), norm=BoundaryNorm(color_bins, len(palette)))
    sm.set_array([])
    cax = fig.add_axes([0.2, 0.05, 0.6, 0.03])
    tick_centers = color_bins[:-1] + 0.5 * np.diff(color_bins)
    cb = plt.colorbar(sm, cax=cax, orientation="horizontal", boundaries=color_bins, ticks=tick_centers)
    cb.set_ticklabels([f"{val/1000.0:.1f}" for val in tick_centers])
    cb.set_label("Initial Height in Plume (km)")

    fig.savefig(out_path, dpi=figure_dpi)
    plt.close(fig)
    return True


def plot_mpas_parcel_trajectories(traj_lon, traj_lat, traj_active, parcel_indices, parcel_values, lat_deg, lon_deg, out_path, title=None, colorbar_label="Value", figure_dpi=200, map_extent=None, cmap_name="rainbow", source_lat=None, source_lon=None):
    fig, ax = _setup_geo_axes(lat_deg, lon_deg, map_extent=map_extent)
    parcel_indices = np.asarray(parcel_indices, dtype=int)
    values = np.asarray(parcel_values, dtype=float)
    if parcel_indices.size == 0 or values.size == 0:
        fig.savefig(out_path, dpi=figure_dpi)
        plt.close(fig)
        return

    n = min(parcel_indices.size, values.size)
    parcel_indices = parcel_indices[:n]
    values = values[:n]
    finite = np.isfinite(values)
    parcel_indices = parcel_indices[finite]
    values = values[finite]
    if parcel_indices.size == 0:
        fig.savefig(out_path, dpi=figure_dpi)
        plt.close(fig)
        return

    n_bins = 10
    vmin = float(np.nanmin(values))
    vmax = float(np.nanmax(values))
    if np.isclose(vmin, vmax):
        vmax = vmin + 1.0
    bins = np.linspace(vmin, vmax, n_bins + 1)
    cmap = plt.get_cmap(cmap_name, n_bins)
    palette = cmap(np.arange(n_bins))
    color_idx = np.clip(np.digitize(values, bins) - 1, 0, n_bins - 1)
    parcel_colors = palette[color_idx]

    lon = np.asarray(traj_lon)[:, parcel_indices]
    lat = np.asarray(traj_lat)[:, parcel_indices]
    active = np.asarray(traj_active, dtype=bool)[:, parcel_indices]

    segments = np.array([
        np.stack([lon[:-1, :], lat[:-1, :]], axis=2),
        np.stack([lon[1:, :], lat[1:, :]], axis=2),
    ]).transpose(2, 1, 0, 3)
    active_mask = (active[:-1, :] & active[1:, :]).T
    segments_to_plot = segments[active_mask]
    if segments_to_plot.size > 0:
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

    ax.scatter(
        lon[0, :],
        lat[0, :],
        s=PARCEL_MARKER_SIZE,
        c=parcel_colors,
        edgecolors=PARCEL_MARKER_EDGE,
        linewidths=PARCEL_MARKER_LINEWIDTH,
        alpha=PARCEL_MARKER_ALPHA,
        transform=PLATE_CARREE,
        zorder=6,
    )
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
            transform=PLATE_CARREE,
        )

    ax.set_title(title or "Parcel trajectories")

    ax.set_xlabel("")
    ax.set_ylabel("")

    cax = fig.add_axes([0.2, 0.05, 0.6, 0.03])
    tick_centers = bins[:-1] + 0.5 * np.diff(bins)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=BoundaryNorm(bins, cmap.N))
    sm.set_array([])
    cb = plt.colorbar(
        sm,
        cax=cax,
        orientation="horizontal",
        boundaries=bins,
        ticks=tick_centers,
    )
    cb.set_ticklabels([f"{val:.1f}" for val in tick_centers])
    cb.set_label(colorbar_label)

    fig.savefig(out_path, dpi=figure_dpi)
    plt.close(fig)


def plot_mpas_trajectories_by_age(
    traj_lon,
    traj_lat,
    traj_active,
    traj_z,
    traj_times,
    lat_deg,
    lon_deg,
    out_path,
    figure_dpi=200,
    map_extent=None,
    source_lat=None,
    source_lon=None,
):
    """Plot trajectories colored by parcel age since release, matching WRF logic."""
    lon_hist = np.asarray(traj_lon, dtype=float)
    lat_hist = np.asarray(traj_lat, dtype=float)
    active_hist = np.asarray(traj_active, dtype=bool)
    z_hist = np.asarray(traj_z, dtype=float)
    times_arr = np.asarray(traj_times)
    if lon_hist.ndim != 2 or lat_hist.ndim != 2 or active_hist.ndim != 2:
        raise ValueError("MPAS trajectory arrays must be 2-D.")
    if lon_hist.shape != lat_hist.shape or lon_hist.shape != active_hist.shape:
        raise ValueError("Trajectory lon/lat/active arrays must have matching shape.")
    if times_arr.ndim != 1 or times_arr.size != lon_hist.shape[0]:
        raise ValueError("traj_times must be 1-D with one entry per trajectory snapshot.")
    if times_arr.size < 2:
        print("[diag] Not enough trajectory times for age plot.")
        return False

    if np.issubdtype(times_arr.dtype, np.datetime64):
        age_hours = (times_arr - times_arr[0]) / np.timedelta64(1, "h")
    else:
        age_hours = (np.asarray(times_arr, dtype=float) - float(times_arr[0])) / 3600.0
    age_hours = np.asarray(age_hours, dtype=float)

    fig, ax = _setup_geo_axes(lat_deg, lon_deg, map_extent=map_extent)

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

    # Match WRF logic: draw segments while parcel is active at segment start.
    active_mask = active_hist[:-1, :].T
    if active_mask.any():
        age_seg = age_hours[:-1]
        color_idx = np.digitize(age_seg, age_bins) - 1
        color_idx = np.clip(color_idx, 0, n_bins - 1)
        colors_per_step = palette[color_idx]
        colors_all = np.repeat(colors_per_step[None, :, :], lon_hist.shape[1], axis=0)
        colors_to_plot = colors_all[active_mask]
        segments_to_plot = segments[active_mask]
        lc = LineCollection(
            segments_to_plot,
            colors=colors_to_plot,
            linestyle=TRAJECTORY_LINESTYLE,
            linewidth=TRAJECTORY_LINEWIDTH,
            alpha=TRAJECTORY_ALPHA,
            transform=PLATE_CARREE,
        )
        ax.add_collection(lc)

    # Draw deposited parcel stop points in black, analogous to WRF.
    n_parcels = lon_hist.shape[1]
    stop_idx = np.full(n_parcels, -1, dtype=int)
    for p in range(n_parcels):
        inactive = np.where(~active_hist[:, p])[0]
        if inactive.size == 0:
            continue
        idx = int(inactive[0])
        if np.isfinite(z_hist[idx, p]) and z_hist[idx, p] <= 0.0:
            stop_idx[p] = idx
    deposited = stop_idx >= 0
    if np.any(deposited):
        cols = np.where(deposited)[0]
        rows = stop_idx[cols]
        lon_stop = lon_hist[rows, cols]
        lat_stop = lat_hist[rows, cols]
        valid_stop = np.isfinite(lon_stop) & np.isfinite(lat_stop)
        if valid_stop.any():
            ax.scatter(
                lon_stop[valid_stop],
                lat_stop[valid_stop],
                s=16.0,
                c="black",
                edgecolors="none",
                transform=PLATE_CARREE,
                zorder=8,
            )

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
            transform=PLATE_CARREE,
        )

    ax.set_xlabel("")
    ax.set_ylabel("")
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


def plot_mpas_vertical_distribution(parcels, out_path, z_min, z_max, figure_dpi=200):
    heights = np.asarray(parcels.get("z_init", []), dtype=float)
    if heights.size == 0:
        return
    x_vals = np.asarray(parcels.get("cell", np.arange(heights.size)), dtype=float)
    if x_vals.size != heights.size:
        x_vals = np.arange(heights.size, dtype=float)
    cmap = plt.get_cmap("turbo")
    norm = plt.Normalize(z_min, z_max)
    fig, ax = plt.subplots(figsize=(10, 8))
    sc = ax.scatter(x_vals, heights / 1000.0, s=14, c=heights, cmap=cmap, norm=norm, edgecolors="black", linewidths=0.15)
    ax.set_xlabel("Grid-cell index")
    ax.set_ylabel("Initial altitude, km")
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
    if np.isfinite(z_min) and np.isfinite(z_max) and z_max > z_min:
        ax.set_ylim(z_min / 1000.0, z_max / 1000.0)
    fig.savefig(out_path, dpi=figure_dpi)
    plt.close(fig)


def plot_mpas_seed_locations(lat_deg, lon_deg, parcels, out_path, title=None, figure_dpi=200, map_extent=None):
    fig, ax = _setup_geo_axes(lat_deg, lon_deg, map_extent=map_extent)
    ax.scatter(parcels["lon"], parcels["lat"], s=PARCEL_MARKER_SIZE, c="red", edgecolors=PARCEL_MARKER_EDGE, linewidths=PARCEL_MARKER_LINEWIDTH, alpha=PARCEL_MARKER_ALPHA, transform=PLATE_CARREE, zorder=5)
    if title:
        ax.set_title(title)
    fig.savefig(out_path, dpi=figure_dpi)
    plt.close(fig)


def plot_mpas_hourly_snapshots(
    lat_deg,
    lon_deg,
    traj_lon,
    traj_lat,
    traj_active,
    traj_z,
    traj_times,
    time_indices,
    out_dir,
    figure_dpi=200,
    map_extent=None,
    source_lat=None,
    source_lon=None,
    tail_enabled=False,
    tail_steps=6,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    times = np.asarray(time_indices, dtype=int)
    traj_times = np.asarray(traj_times)
    traj_active = np.asarray(traj_active, dtype=bool)
    traj_z_km = np.asarray(traj_z, dtype=float) / 1000.0
    active_heights = traj_z_km[traj_active & np.isfinite(traj_z_km)]
    if active_heights.size > 0:
        zmin = float(np.nanmin(active_heights))
        zmax = float(np.nanmax(active_heights))
        if zmax <= zmin:
            zmax = zmin + 1.0
    else:
        zmin = 0.0
        zmax = 1.0
    n_bins = 10
    h_bins = np.linspace(zmin, zmax, n_bins + 1)
    cmap = plt.get_cmap("rainbow", n_bins)
    norm = BoundaryNorm(h_bins, cmap.N)
    height_centers = h_bins[:-1] + 0.5 * np.diff(h_bins)
    traj_lon = np.asarray(traj_lon, dtype=float)
    traj_lat = np.asarray(traj_lat, dtype=float)

    def _draw_fancy_tail(ax, snap_idx):
        if not tail_enabled or snap_idx <= 0:
            return
        max_tail = min(int(tail_steps), int(snap_idx))
        for lag in range(max_tail, 0, -1):
            t0 = snap_idx - lag
            t1 = t0 + 1
            lon0 = traj_lon[t0]
            lat0 = traj_lat[t0]
            lon1 = traj_lon[t1]
            lat1 = traj_lat[t1]
            active_pair = traj_active[t0] & traj_active[t1]
            valid = (
                np.isfinite(lon0)
                & np.isfinite(lat0)
                & np.isfinite(lon1)
                & np.isfinite(lat1)
                & active_pair
            )
            if not np.any(valid):
                continue

            z0 = traj_z_km[t0]
            valid &= np.isfinite(z0)
            if not np.any(valid):
                continue

            segs = np.stack(
                [
                    np.stack([lon0[valid], lat0[valid]], axis=1),
                    np.stack([lon1[valid], lat1[valid]], axis=1),
                ],
                axis=1,
            )
            colors = cmap(norm(z0[valid]))
            alpha_tail = 0.08 + 0.42 * ((max_tail - lag + 1) / max_tail)
            colors_glow = colors.copy()
            colors_glow[:, 3] = np.clip(alpha_tail * 0.45, 0.05, 0.35)
            colors[:, 3] = np.clip(alpha_tail, 0.12, 0.65)
            lw_glow = 2.8 + 1.6 * ((max_tail - lag + 1) / max_tail)
            lw_core = 1.4 + 1.0 * ((max_tail - lag + 1) / max_tail)

            lc_glow = LineCollection(
                segs,
                colors=colors_glow,
                linewidths=lw_glow,
                capstyle="round",
                transform=PLATE_CARREE,
                zorder=4,
            )
            ax.add_collection(lc_glow)
            lc_core = LineCollection(
                segs,
                colors=colors,
                linewidths=lw_core,
                capstyle="round",
                transform=PLATE_CARREE,
                zorder=5,
            )
            ax.add_collection(lc_core)
    n_saved = 0
    for snap_idx, time_idx in enumerate(times):
        fig, ax = _setup_geo_axes(lat_deg, lon_deg, map_extent=map_extent)
        lon = traj_lon[snap_idx]
        lat = traj_lat[snap_idx]
        active = traj_active[snap_idx]
        z = traj_z_km[snap_idx]
        _draw_fancy_tail(ax, snap_idx)
        ax.scatter(
            lon[active],
            lat[active],
            s=PARCEL_MARKER_SIZE,
            c=z[active],
            cmap=cmap,
            norm=norm,
            edgecolors="none",
            linewidths=0.0,
            alpha=PARCEL_MARKER_ALPHA,
            transform=PLATE_CARREE,
            zorder=5,
        )
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
                transform=PLATE_CARREE,
            )
        if traj_times.size > snap_idx:
            snapshot_time = traj_times[snap_idx]
            if np.issubdtype(traj_times.dtype, np.datetime64):
                utc_label = str(snapshot_time.astype("datetime64[s]"))
                hour = int(np.rint((snapshot_time - traj_times[0]) / np.timedelta64(1, "h")))
            else:
                utc_label = str(snapshot_time)
                hour = int(time_idx)
            ax.set_title(f"Parcel positions at +{hour:02d} h (UTC: {utc_label})")
        else:
            ax.set_title(f"Parcel positions at snapshot {time_idx}")
        cax = fig.add_axes([0.2, 0.05, 0.6, 0.03])
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cb = plt.colorbar(
            sm,
            cax=cax,
            orientation="horizontal",
            boundaries=h_bins,
            ticks=height_centers,
        )
        cb.set_ticklabels([f"{val:.1f}" for val in height_centers])
        cb.set_label("Height (km)")
        fig.savefig(out_dir / f"parcel_positions_hour.{snap_idx:04d}.png", dpi=figure_dpi)
        plt.close(fig)
        n_saved += 1
    return n_saved


def plot_mpas_deposited_parcels_by_hour(
    lat_deg,
    lon_deg,
    traj_lon,
    traj_lat,
    traj_z,
    traj_active,
    traj_times,
    out_path,
    figure_dpi=200,
    map_extent=None,
    source_lat=None,
    source_lon=None,
):
    traj_lon = np.asarray(traj_lon, dtype=float)
    traj_lat = np.asarray(traj_lat, dtype=float)
    traj_z = np.asarray(traj_z, dtype=float)
    traj_active = np.asarray(traj_active, dtype=bool)
    traj_times = np.asarray(traj_times)
    if traj_lon.ndim != 2 or traj_lat.ndim != 2 or traj_z.ndim != 2 or traj_active.ndim != 2:
        raise ValueError("MPAS trajectory arrays must be 2-D.")
    if traj_lon.shape != traj_lat.shape or traj_lon.shape != traj_z.shape or traj_lon.shape != traj_active.shape:
        raise ValueError("MPAS trajectory arrays must have matching shape.")
    if traj_times.ndim != 1 or traj_times.size != traj_lon.shape[0]:
        raise ValueError("traj_times must be 1-D with one entry per snapshot.")

    n_parcels = traj_lon.shape[1]
    stop_idx = np.full(n_parcels, -1, dtype=int)
    for p in range(n_parcels):
        inactive = np.where(~traj_active[:, p])[0]
        if inactive.size == 0:
            continue
        idx = int(inactive[0])
        if np.isfinite(traj_z[idx, p]) and traj_z[idx, p] <= 0.0:
            stop_idx[p] = idx

    deposited = stop_idx >= 0
    if not np.any(deposited):
        print("[diag] No deposited parcels found; skipping deposition-hour figure.")
        return False

    dep_hours = np.maximum((traj_times[stop_idx[deposited]] - traj_times[0]) / np.timedelta64(1, "h"), 0.0)
    lon_dep = traj_lon[stop_idx[deposited], np.where(deposited)[0]]
    lat_dep = traj_lat[stop_idx[deposited], np.where(deposited)[0]]

    fig, ax = _setup_geo_axes(lat_deg, lon_deg, map_extent=map_extent)
    n_bins = 10
    hmin = float(np.nanmin(dep_hours))
    hmax = float(np.nanmax(dep_hours))
    if np.isclose(hmin, hmax):
        hmax = hmin + 0.1
    h_bins = np.linspace(hmin, hmax, n_bins + 1)
    cmap = plt.get_cmap("plasma", n_bins)
    norm = BoundaryNorm(h_bins, cmap.N)
    ax.scatter(
        lon_dep,
        lat_dep,
        c=dep_hours,
        cmap=cmap,
        norm=norm,
        s=18.0,
        alpha=0.9,
        edgecolors="none",
        transform=PLATE_CARREE,
        zorder=8,
    )
    if source_lat is not None and source_lon is not None:
        ax.scatter(
            source_lon,
            source_lat,
            marker="^",
            s=80,
            c="red",
            edgecolors="black",
            linewidths=0.4,
            zorder=9,
            transform=PLATE_CARREE,
        )
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title(
        "Deposited parcels coloured by deposition hour "
        f"(n={lon_dep.size}, min={hmin:.1f} h, max={hmax:.1f} h)"
    )
    cax = fig.add_axes([0.2, 0.05, 0.6, 0.03])
    hour_centers = h_bins[:-1] + 0.5 * np.diff(h_bins)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = plt.colorbar(sm, cax=cax, orientation="horizontal", boundaries=h_bins, ticks=hour_centers)
    cb.set_ticklabels([f"{val:.1f}" for val in hour_centers])
    cb.set_label("Deposition hour since release (h)")
    fig.savefig(out_path, dpi=figure_dpi)
    plt.close(fig)
    return True


def plot_emission_matrix(emission, time_edges, z_edges_km, out_path, figure_dpi=200, colorbar_label="Parcel count", total_parcels=None):
    emission = np.asarray(emission, dtype=float)
    if emission.size == 0:
        _diag("Emission matrix empty; skipping figure generation.")
        return
    time_edges = np.asarray(time_edges, dtype=float)
    z_edges_km = np.asarray(z_edges_km, dtype=float)
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.pcolormesh(
        time_edges,
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
    original_cmap = plt.get_cmap("viridis", n_bins)
    new_colors = original_cmap(np.linspace(0, 1, n_bins))
    new_colors[-1] = (1, 1, 1, 1)
    first_color = new_colors[0].copy()
    new_colors[0] = new_colors[-1]
    new_colors[-1] = first_color
    cmap = ListedColormap(new_colors)
    norm = BoundaryNorm(boundaries, cmap.N, clip=True)
    cs = ax.pcolormesh(
        time_edges,
        z_edges_km,
        emission,
        cmap=cmap,
        norm=norm,
        shading="flat",
    )
    ax.set_ylabel("Altitude (km)")
    ax.set_xlabel("Time")
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
    ax.set_xticks(time_edges[:-1])
    ax.set_xticklabels([f"{val:.1f}" for val in time_edges[:-1]], rotation=90, ha="center", fontsize=6)
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
    fig.savefig(out_path, dpi=figure_dpi)
    plt.close(fig)


def run_backtraj(args):
    data = read_mpas_history(args.input)
    times = data["times"]
    lat = data["lat_deg"]
    lon = data["lon_deg"]
    zmid = data["zmid"]
    area = data["area"]
    tree = data["tree"]
    times_arr = np.asarray(times)
    _diag(
        "MPAS dimensions: "
        f"nCells={lat.size}, nVertLevels={zmid.shape[1]}, nTimes={times_arr.size}."
    )

    start_time = None
    if args.start_time is not None:
        s = args.start_time.replace("_", "T")
        start_time = np.datetime64(s)
    if start_time is None:
        start_time = times_arr[-1]
    if start_time < times_arr[0] or start_time > times_arr[-1]:
        raise ValueError("start time outside MPAS time range.")
    start_idx = int(np.argmin(np.abs(times_arr - start_time)))
    _diag(
        "Using start time "
        f"{times_arr[start_idx]} at index {start_idx} "
        f"(closest to requested {start_time})."
    )

    with Dataset(args.column, "r") as ds_col:
        _diag(f"Reading MPAS column field '{args.column_var}' from {args.column}.")
        if args.column_var not in ds_col.variables:
            raise KeyError(f"Column variable '{args.column_var}' not found.")
        col_var = ds_col.variables[args.column_var][:]
        col_var = np.asarray(col_var, dtype=float)
        if col_var.ndim == 1:
            column = col_var
        elif col_var.ndim == 2:
            if col_var.shape[0] == times_arr.size:
                column = col_var[min(start_idx, col_var.shape[0] - 1), :]
            else:
                column = np.nansum(col_var, axis=-1)
        elif col_var.ndim == 3:
            time_idx = min(start_idx, col_var.shape[0] - 1)
            column = np.nansum(col_var[time_idx, :, :], axis=-1)
        else:
            raise ValueError(f"Unsupported MPAS column shape: {col_var.shape}")
        column = np.asarray(column, dtype=float).reshape(-1)
    if args.column_coef != 1.0:
        _diag(f"Scaling column field by factor {args.column_coef}.")
        column = column * float(args.column_coef)
    else:
        _diag("Column scaling factor = 1.0 (no change).")

    seed_bbox = tuple(args.seed_bbox) if args.seed_bbox is not None else None
    for out_path in (
        args.seeds_figure,
        args.seeds_vertical_figure,
        args.trajectory_figure,
        args.trajectory_age,
        args.trajectory_emission_time_figure,
        args.trajectory_arrival_height_figure,
        args.missed_trajectory_figure,
        args.output_txt,
        args.output_figure,
        args.mass_output_txt,
        args.mass_figure,
        args.state_pickle,
    ):
        _ensure_parent_dir(out_path)

    parcels_init = generate_parcels_from_column(column, lat, lon, zmid, start_idx, args.threshold, args.n_columns, args.n_vert, args.z_min, args.z_max, seed_bbox=seed_bbox)
    _diag(
        "Initialized "
        f"{parcels_init['lon'].size} MPAS parcels at time index {start_idx} "
        f"({times_arr[start_idx]})."
    )
    if args.n_vert > 0:
        cell_mass = np.asarray(column, dtype=float) * np.asarray(area, dtype=float)
        parcel_mass = cell_mass[parcels_init["cell"]] / float(args.n_vert)
        parcels_init["mass"] = parcel_mass
    else:
        parcels_init["mass"] = np.zeros_like(parcels_init["z"], dtype=float)

    if args.seeds_figure:
        plot_mpas_column_and_parcels(column, lat, lon, parcels_init, args.seeds_figure, threshold=args.threshold, receptor_lat=args.receptor_lat, receptor_lon=args.receptor_lon, receptor_radius_m=args.receptor_radius, seed_bbox=seed_bbox, title=f"Parcel seeds at {times_arr[start_idx]}", colorbar_label=args.colorbar_label, figure_dpi=args.figure_dpi, map_extent=tuple(args.map_extent) if args.map_extent is not None else None)
        _diag(f"Parcel-location map saved to '{args.seeds_figure}'.")
    if args.seeds_vertical_figure:
        plot_mpas_vertical_distribution(parcels_init, args.seeds_vertical_figure, args.z_min, args.z_max, args.figure_dpi)
        _diag(f"Parcel vertical distribution saved to '{args.seeds_vertical_figure}'.")

    settling_profile = None
    if args.aer_type is not None:
        settling_profile = dict(heights_m=np.asarray(Z_M, dtype=float), velocity_ms=np.asarray(SETTLING_VEL_MS[args.aer_type], dtype=float))
        _diag(f"Applying settling profile for '{args.aer_type}'.")

    _diag("Starting MPAS backward advection.")
    result = advect_parcels_backward(
        parcels_init,
        times,
        data["u"],
        data["v"],
        data["w"],
        zmid,
        data["tree"],
        args.receptor_lat,
        args.receptor_lon,
        args.receptor_radius,
        args.receptor_min_h,
        args.receptor_max_h,
        start_idx,
        args.integration_dt,
        emission_start_time=np.datetime64(args.emission_start.replace("_", "T")) if args.emission_start else None,
        emission_end_time=np.datetime64(args.emission_end.replace("_", "T")) if args.emission_end else None,
    )
    _diag(
        "MPAS backward advection complete: "
        f"{int(result['arrived'].sum())}/{result['arrived'].size} parcels reached the receptor."
    )

    if args.trajectory_figure:
        init_h_km = np.asarray(parcels_init["z_init"], dtype=float) / 1000.0
        finite_init_h = init_h_km[np.isfinite(init_h_km)]
        title_suffix = ""
        if finite_init_h.size:
            title_suffix = f" (min={float(np.nanmin(finite_init_h)):.2f} km, max={float(np.nanmax(finite_init_h)):.2f} km)"
        plot_mpas_trajectories(column, lat, lon, result["trajectory_lon"], result["trajectory_lat"], result["trajectory_active"], np.where(result["arrived"])[0], init_h_km, args.trajectory_figure, threshold=args.threshold, receptor_lat=args.receptor_lat, receptor_lon=args.receptor_lon, receptor_radius_m=args.receptor_radius, seed_bbox=seed_bbox, title="Parcel Trajectories colored by Initial Plume Height" + title_suffix, colorbar_label="Initial height in plume (km)", figure_dpi=args.figure_dpi, map_extent=tuple(args.map_extent) if args.map_extent is not None else None)
        _diag(f"Trajectory figure saved to '{args.trajectory_figure}'.")

    if args.trajectory_age:
        start_sec = float((times_arr[start_idx] - times_arr[0]) / np.timedelta64(1, "s"))
        ages = np.maximum((start_sec - result["arrival_time"]) / 3600.0, 0.0)
        age_subset = ages[result["arrived"]]
        age_suffix = ""
        if age_subset.size:
            age_suffix = f" (min={float(np.nanmin(age_subset)):.2f} h, max={float(np.nanmax(age_subset)):.2f} h)"
        plot_mpas_trajectories(column, lat, lon, result["trajectory_lon"], result["trajectory_lat"], result["trajectory_active"], np.where(result["arrived"])[0], age_subset, args.trajectory_age, threshold=args.threshold, receptor_lat=args.receptor_lat, receptor_lon=args.receptor_lon, receptor_radius_m=args.receptor_radius, seed_bbox=seed_bbox, title="Parcel trajectories coloured by arrival age" + age_suffix, colorbar_label="Arrival age (hours)", figure_dpi=args.figure_dpi, map_extent=tuple(args.map_extent) if args.map_extent is not None else None)
        _diag(f"Parcel-age figure saved to '{args.trajectory_age}'.")

    if args.hourly_output_dir:
        n_hourly = plot_mpas_hourly_snapshots(
            lat,
            lon,
            result["trajectory_lon"],
            result["trajectory_lat"],
            result["trajectory_active"],
            result["trajectory_z"],
            result["trajectory_times"],
            result["trajectory_time_indices"],
            args.hourly_output_dir,
            figure_dpi=args.figure_dpi,
            map_extent=tuple(args.map_extent) if args.map_extent is not None else None,
        )
        _diag(f"Hourly snapshots saved: {n_hourly} file(s) in '{args.hourly_output_dir}'.")

    arrived_mask = np.asarray(result["arrived"], dtype=bool)
    arrived_indices = np.where(arrived_mask)[0]
    arrival_time_sec = np.asarray(result["arrival_time"], dtype=float)[arrived_mask]
    arrival_z_m = np.asarray(result["arrival_z"], dtype=float)[arrived_mask]

    start_sec = float((times_arr[start_idx] - times_arr[0]) / np.timedelta64(1, "s"))
    arrival_age_hours = np.maximum((start_sec - arrival_time_sec) / 3600.0, 0.0)
    emission_time_hours = np.maximum(arrival_time_sec, 0.0) / 3600.0

    parcel_mass_all = np.asarray(parcels_init.get("mass", np.ones(parcels_init["z"].size)), dtype=float)
    arrival_mass = parcel_mass_all[arrived_mask]
    if args.efolding_days is not None and args.efolding_days > 0:
        efolding_time_sec = float(args.efolding_days) * 86400.0
        parcel_age_sec = np.maximum(start_sec - arrival_time_sec, 0.0)
        mass_correction_factor = np.exp(parcel_age_sec / efolding_time_sec)
        arrival_mass = arrival_mass * mass_correction_factor
        _diag(f"Applied mass correction with e-folding time of {args.efolding_days} days.")

    if args.trajectory_emission_time_figure:
        em_subset = emission_time_hours[np.isfinite(emission_time_hours)]
        em_suffix = ""
        if em_subset.size:
            em_suffix = f" (min={float(np.nanmin(em_subset)):.2f} h, max={float(np.nanmax(em_subset)):.2f} h)"
        plot_mpas_trajectories(
            column,
            lat,
            lon,
            result["trajectory_lon"],
            result["trajectory_lat"],
            result["trajectory_active"],
            arrived_indices,
            emission_time_hours,
            args.trajectory_emission_time_figure,
            threshold=args.threshold,
            receptor_lat=args.receptor_lat,
            receptor_lon=args.receptor_lon,
            receptor_radius_m=args.receptor_radius,
            seed_bbox=seed_bbox,
            title="Parcel trajectories coloured by emission time" + em_suffix,
            colorbar_label="Emission time since reference (hours)",
            figure_dpi=args.figure_dpi,
            map_extent=tuple(args.map_extent) if args.map_extent is not None else None,
        )
        _diag(f"Parcel emission-time figure saved to '{args.trajectory_emission_time_figure}'.")

    if args.trajectory_arrival_height_figure:
        ah_subset = arrival_z_m[np.isfinite(arrival_z_m)] / 1000.0
        ah_suffix = ""
        if ah_subset.size:
            ah_suffix = f" (min={float(np.nanmin(ah_subset)):.2f} km, max={float(np.nanmax(ah_subset)):.2f} km)"
        plot_mpas_trajectories(
            column,
            lat,
            lon,
            result["trajectory_lon"],
            result["trajectory_lat"],
            result["trajectory_active"],
            arrived_indices,
            arrival_z_m / 1000.0,
            args.trajectory_arrival_height_figure,
            threshold=args.threshold,
            receptor_lat=args.receptor_lat,
            receptor_lon=args.receptor_lon,
            receptor_radius_m=args.receptor_radius,
            seed_bbox=seed_bbox,
            title="Parcel trajectories coloured by arrival height" + ah_suffix,
            colorbar_label="Arrival height (km)",
            figure_dpi=args.figure_dpi,
            map_extent=tuple(args.map_extent) if args.map_extent is not None else None,
        )
        _diag(f"Parcel arrival-height figure saved to '{args.trajectory_arrival_height_figure}'.")

    if args.missed_trajectory_figure:
        saved_missed = plot_mpas_missed_trajectories(
            column,
            lat,
            lon,
            result["trajectory_lon"],
            result["trajectory_lat"],
            result["trajectory_active"],
            arrived_mask,
            np.asarray(parcels_init["z_init"], dtype=float),
            args.missed_trajectory_figure,
            threshold=args.threshold,
            receptor_lat=args.receptor_lat,
            receptor_lon=args.receptor_lon,
            receptor_radius_m=args.receptor_radius,
            seed_bbox=seed_bbox,
            z_min=args.z_min,
            z_max=args.z_max,
            figure_dpi=args.figure_dpi,
            map_extent=tuple(args.map_extent) if args.map_extent is not None else None,
        )
        if saved_missed:
            _diag(f"Missed trajectory figure saved to '{args.missed_trajectory_figure}'.")
        else:
            _diag("No missed parcels found; skipping missed-trajectory figure.")

    if args.output_txt:
        arrival_z = arrival_z_m
        arrival_time = arrival_time_sec
        z_bins = zmid[np.argmin(np.abs(lat - args.receptor_lat) + np.abs(lon - args.receptor_lon))]
        z_edges = np.concatenate(([max(0.0, z_bins[0] - 0.5 * (z_bins[1] - z_bins[0]))], 0.5 * (z_bins[:-1] + z_bins[1:]), [z_bins[-1] + 0.5 * (z_bins[-1] - z_bins[-2])]))
        t_sec = np.maximum(arrival_time, 0.0)
        t_edges = np.arange(0.0, max(float(np.nanmax(t_sec)) if t_sec.size else 0.0, args.arrival_bin_minutes * 60.0) + args.arrival_bin_minutes * 60.0, args.arrival_bin_minutes * 60.0)
        if t_edges.size < 2:
            t_edges = np.array([0.0, args.arrival_bin_minutes * 60.0])
        emission = np.zeros((z_edges.size - 1, t_edges.size - 1), dtype=float)
        mass_emission = np.zeros_like(emission)
        if t_sec.size:
            t_bin = np.clip(np.digitize(t_sec, t_edges) - 1, 0, t_edges.size - 2)
            z_bin = np.clip(np.digitize(arrival_z, z_edges) - 1, 0, z_edges.size - 2)
            np.add.at(emission, (z_bin, t_bin), 1.0)
            np.add.at(mass_emission, (z_bin, t_bin), arrival_mass)
        with open(args.output_txt, "w", encoding="utf-8") as fh:
            fh.write("time " + " ".join(f"{v:.0f}" for v in t_edges[:-1]) + "\n")
            fh.write("height " + " ".join(f"{v:.1f}" for v in z_edges[:-1] / 1000.0) + "\n")
            for row in emission[::-1]:
                fh.write(" ".join(f"{int(v)}" for v in row) + "\n")
        _diag(f"Time-height emission series written to '{args.output_txt}'.")
        if args.output_figure:
            plot_emission_matrix(emission, t_edges / 3600.0, z_edges / 1000.0, args.output_figure, args.figure_dpi, colorbar_label=args.colorbar_label, total_parcels=int(np.count_nonzero(arrived_mask)))
            _diag(f"Emission matrix figure saved to '{args.output_figure}'.")
        if args.mass_output_txt:
            with open(args.mass_output_txt, "w", encoding="utf-8") as fh:
                fh.write("time " + " ".join(f"{v:.0f}" for v in t_edges[:-1]) + "\n")
                fh.write("height " + " ".join(f"{v:.1f}" for v in z_edges[:-1] / 1000.0) + "\n")
                for row in mass_emission[::-1]:
                    fh.write(" ".join(f"{v:.6e}" for v in row) + "\n")
            _diag(f"Mass-weighted emission series written to '{args.mass_output_txt}'.")
        if args.mass_figure:
            plot_emission_matrix(mass_emission, t_edges / 3600.0, z_edges / 1000.0, args.mass_figure, args.figure_dpi, colorbar_label="Parcel mass")
            _diag(f"Mass-weighted emission figure saved to '{args.mass_figure}'.")

    if args.state_pickle:
        args_dict = {k: getattr(args, k) for k in vars(args)}
        emission_matrix = locals().get("emission")
        t_edges = locals().get("t_edges")
        z_edges = locals().get("z_edges")
        pickle_payload = dict(
            args=args_dict,
            grid=dict(lat_deg=lat, lon_deg=lon, zgrid=zmid),
            initial_parcels=dict(
                lon=parcels_init["lon"],
                lat=parcels_init["lat"],
                z_init=parcels_init["z_init"],
                cell=parcels_init.get("cell"),
                mass=parcels_init.get("mass"),
            ),
            column=dict(
                field=column,
                threshold=args.threshold,
                colorbar_label=args.colorbar_label,
            ),
            trajectories=dict(
                times=result["trajectory_times"],
                time_indices=result["trajectory_time_indices"],
                lon=result["trajectory_lon"],
                lat=result["trajectory_lat"],
                z=result["trajectory_z"],
                active=result["trajectory_active"],
                arrived_mask=result["arrived"],
                arrival_time_sec=result["arrival_time"],
                arrival_age_hours=np.maximum(result["arrival_time"], 0.0) / 3600.0,
                arrival_z=result["arrival_z"],
                arrival_height_m=result["arrival_z"],
                initial_height_m=parcels_init.get("z_init"),
                final_z=result["z"],
            ),
            emission=dict(
                matrix=emission_matrix,
                mass_matrix=locals().get("mass_emission"),
                time_edges=(t_edges / 3600.0) if t_edges is not None else None,
                z_bins=(z_edges / 1000.0) if z_edges is not None else None,
                total_parcels=int(result["arrived"].sum()),
            ),
            metadata=dict(
                start_time=times_arr[start_idx],
                start_time_index=start_idx,
                finish_time_index=0,
                parcels_initialized=parcels_init["lon"].size,
                receptor=dict(lat=args.receptor_lat, lon=args.receptor_lon, radius_m=args.receptor_radius),
                receptor_min_h=args.receptor_min_h,
                receptor_max_h=args.receptor_max_h,
                emission_start=np.datetime64(args.emission_start.replace("_", "T")) if args.emission_start else None,
                emission_end=np.datetime64(args.emission_end.replace("_", "T")) if args.emission_end else None,
                target="mpas",
            ),
        )
        with open(args.state_pickle, "wb") as fh:
            pickle.dump(pickle_payload, fh)
        print(f"[diag] Saved processing state to '{args.state_pickle}'.")


def run_forwtraj(args):
    data = read_mpas_history(args.input)
    times = data["times"]
    lat = data["lat_deg"]
    lon = data["lon_deg"]
    zmid = data["zmid"]
    times_arr = np.asarray(times)
    _diag(
        "MPAS dimensions: "
        f"nCells={lat.size}, nVertLevels={zmid.shape[1]}, nTimes={times_arr.size}."
    )
    if args.start_time is not None:
        start_time = np.datetime64(args.start_time.replace("_", "T"))
        start_idx = int(np.argmin(np.abs(times_arr - start_time)))
    else:
        start_idx = 0
    if args.end_time is not None:
        end_time = np.datetime64(args.end_time.replace("_", "T"))
        end_idx = int(np.argmin(np.abs(times_arr - end_time)))
    else:
        end_idx = times_arr.size - 1
    if args.source_lat is None or args.source_lon is None:
        raise ValueError("MPAS forward mode requires --source-lat and --source-lon.")
    if end_idx <= start_idx:
        raise ValueError("End time index is not later than start time index.")
    _diag(
        "Advection window: index "
        f"{start_idx} ({times_arr[start_idx]}) -> {end_idx} ({times_arr[end_idx]})."
    )
    src_idx = int(_nearest_cells(data["tree"], np.asarray([args.source_lat]), np.asarray([args.source_lon]))[0])
    _diag(
        "Selected MPAS source cell "
        f"{src_idx} at lat={lat[src_idx]:.4f}, lon={lon[src_idx]:.4f} "
        f"for requested lat={args.source_lat:.4f}, lon={args.source_lon:.4f}."
    )
    parcels = generate_parcels_from_point(lon[src_idx], lat[src_idx], src_idx, args.n_vert, args.z_min, args.z_max)
    _diag(f"Initialized {parcels['lon'].size} MPAS parcels.")

    for out_path in (
        getattr(args, "seeds_figure", None),
        args.seeds_vertical_figure,
        args.initial_height_figure,
        args.age_figure,
        args.deposition_figure,
        args.state_pickle,
    ):
        _ensure_parent_dir(out_path)

    _diag("Starting MPAS forward advection.")
    result = advect_parcels_forward(
        parcels,
        times,
        data["u"],
        data["v"],
        data["w"],
        zmid,
        data["tree"],
        start_idx,
        end_idx,
        args.integration_dt,
    )
    _diag("MPAS forward advection complete.")
    seeds_figure = getattr(args, "seeds_figure", None)
    if seeds_figure:
        plot_mpas_seed_locations(lat, lon, parcels, seeds_figure, title="MPAS parcel seeds", figure_dpi=args.figure_dpi, map_extent=tuple(args.map_extent) if args.map_extent is not None else None)
        _diag(f"Parcel seed map saved to '{seeds_figure}'.")
    if args.seeds_vertical_figure:
        plot_mpas_vertical_distribution(parcels, args.seeds_vertical_figure, args.z_min, args.z_max, args.figure_dpi)
        _diag(f"Parcel vertical distribution saved to '{args.seeds_vertical_figure}'.")
    if args.hourly_output_dir:
        n_hourly = plot_mpas_hourly_snapshots(lat, lon, result["trajectory_lon"], result["trajectory_lat"], result["trajectory_active"], result["trajectory_z"], result["trajectory_times"], result["trajectory_time_indices"], args.hourly_output_dir, figure_dpi=args.figure_dpi, map_extent=tuple(args.map_extent) if args.map_extent is not None else None, source_lat=args.source_lat, source_lon=args.source_lon, tail_enabled=True, tail_steps=6)
        _diag(f"Hourly snapshots saved: {n_hourly} file(s) in '{args.hourly_output_dir}'.")
    if args.initial_height_figure:
        init_heights_km = np.asarray(parcels["z_init"], dtype=float) / 1000.0
        init_min_km = float(np.nanmin(init_heights_km))
        init_max_km = float(np.nanmax(init_heights_km))
        title = (
            "Parcel trajectories coloured by initial height "
            f"(min={init_min_km:.2f} km, max={init_max_km:.2f} km)"
        )
        plot_mpas_parcel_trajectories(result["trajectory_lon"], result["trajectory_lat"], result["trajectory_active"], np.arange(result["trajectory_lon"].shape[1]), init_heights_km, lat, lon, args.initial_height_figure, title=title, colorbar_label="Initial height (km)", figure_dpi=args.figure_dpi, map_extent=tuple(args.map_extent) if args.map_extent is not None else None, cmap_name="rainbow", source_lat=args.source_lat, source_lon=args.source_lon)
        _diag(f"Initial-height figure saved to '{args.initial_height_figure}'.")
    if args.age_figure:
        saved_age = plot_mpas_trajectories_by_age(
            result["trajectory_lon"],
            result["trajectory_lat"],
            result["trajectory_active"],
            result["trajectory_z"],
            result["trajectory_times"],
            lat,
            lon,
            args.age_figure,
            figure_dpi=args.figure_dpi,
            map_extent=tuple(args.map_extent) if args.map_extent is not None else None,
            source_lat=args.source_lat,
            source_lon=args.source_lon,
        )
        if saved_age:
            _diag(f"Age-colored figure saved to '{args.age_figure}'.")
    if args.deposition_figure:
        saved_dep = plot_mpas_deposited_parcels_by_hour(
            lat,
            lon,
            result["trajectory_lon"],
            result["trajectory_lat"],
            result["trajectory_z"],
            result["trajectory_active"],
            result["trajectory_times"],
            args.deposition_figure,
            figure_dpi=args.figure_dpi,
            map_extent=tuple(args.map_extent) if args.map_extent is not None else None,
            source_lat=args.source_lat,
            source_lon=args.source_lon,
        )
        if saved_dep:
            _diag(f"Deposition-hour figure saved to '{args.deposition_figure}'.")
    if args.state_pickle:
        args_dict = {k: getattr(args, k) for k in vars(args)}
        payload = dict(
            args=args_dict,
            grid=dict(lat_deg=lat, lon_deg=lon, zgrid=zmid),
            initial_parcels=dict(
                lon=parcels["lon"],
                lat=parcels["lat"],
                z_init=parcels["z_init"],
                cell=parcels.get("cell"),
            ),
            trajectories=dict(
                times=result["trajectory_times"],
                time_indices=result["trajectory_time_indices"],
                lon=result["trajectory_lon"],
                lat=result["trajectory_lat"],
                z=result["trajectory_z"],
                active=result["trajectory_active"],
                final_heights_m=result["trajectory_z"][-1],
                height_hist_m=result["trajectory_z"],
            ),
            metadata=dict(
                start_time=times_arr[start_idx],
                end_time=times_arr[end_idx],
                start_time_index=start_idx,
                end_time_index=end_idx,
                source_lat=args.source_lat,
                source_lon=args.source_lon,
                target="mpas",
            ),
        )
        with open(args.state_pickle, "wb") as fh:
            pickle.dump(payload, fh)
        print(f"[diag] Saved processing state to '{args.state_pickle}'.")


class MPASBackend(PlumeBackend):
    target = "mpas"

    def run_backtraj(self, args):
        run_backtraj(args)

    def run_forwtraj(self, args):
        run_forwtraj(args)

    def plot_backtraj_state(self, args):
        from . import plot_backtraj_mpas

        plot_backtraj_mpas.main(args)

    def plot_forwtraj_state(self, args):
        from . import plot_forwtraj_mpas

        plot_forwtraj_mpas.main(args)
