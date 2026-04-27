import math
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

from misc.settling_velocity_data import SETTLING_VEL_MS, Z_M


PLATE_CARREE = ccrs.PlateCarree()
EARTH_RADIUS_M = 6371229.0
PARCEL_MARKER_SIZE = 10.0
PARCEL_MARKER_ALPHA = 0.75
PARCEL_MARKER_EDGE = (0.1, 0.1, 0.1)
PARCEL_MARKER_LINEWIDTH = 0.2
TRAJECTORY_LINEWIDTH = 0.7
TRAJECTORY_ALPHA = 0.65


def _diag(msg):
    print(f"[diag] {msg}")


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
    z_targets = np.sort(rng.uniform(z_min, z_max, size=n_vert))
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
    current_sec = times_sec[start_time_index]
    for it in range(int(start_time_index), 0, -1):
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
    ax.coastlines(resolution="50m", linewidth=0.6, color="gray")
    try:
        import cartopy.feature as cfeature
        ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.4, edgecolor="gray")
    except Exception:
        pass
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="gray", alpha=0.4, linestyle="--")
    gl.top_labels = False
    gl.right_labels = False
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
    if receptor_lat is not None and receptor_lon is not None:
        ax.scatter(receptor_lon, receptor_lat, marker="^", s=80, c="red", edgecolors="black", linewidths=0.4, zorder=6, transform=PLATE_CARREE)
        if receptor_radius_m is not None and receptor_radius_m > 0:
            ang = np.linspace(0, 2 * np.pi, 181)
            lat_scale = 111320.0
            lon_scale = max(np.cos(np.deg2rad(receptor_lat)) * 111320.0, 1e-6)
            ax.plot(receptor_lon + (receptor_radius_m / lon_scale) * np.cos(ang), receptor_lat + (receptor_radius_m / lat_scale) * np.sin(ang), color="red", linestyle="--", linewidth=1.0, transform=PLATE_CARREE)
    if seed_bbox is not None:
        lon_min, lat_min, lon_max, lat_max = seed_bbox
        ax.plot([lon_min, lon_max, lon_max, lon_min, lon_min], [lat_min, lat_min, lat_max, lat_max, lat_min], color="black", linestyle=":", linewidth=1.0, transform=PLATE_CARREE)
    cax = fig.add_axes([0.2, 0.05, 0.6, 0.03])
    plt.colorbar(mesh, cax=cax, orientation="horizontal", label=colorbar_label)
    if title:
        ax.set_title(title)
    fig.savefig(out_path, dpi=figure_dpi)
    plt.close(fig)


def plot_mpas_trajectories(column, lat_deg, lon_deg, traj_lon, traj_lat, traj_active, parcel_indices, parcel_values, out_path, threshold=None, receptor_lat=None, receptor_lon=None, receptor_radius_m=None, seed_bbox=None, title=None, colorbar_label="Value", figure_dpi=200, map_extent=None, value_unit_label=None):
    fig, ax = _setup_geo_axes(lat_deg, lon_deg, map_extent=map_extent)
    mesh = _plot_background(ax, lon_deg, lat_deg, column, threshold=threshold)
    parcel_indices = np.asarray(parcel_indices, dtype=int)
    values = np.asarray(parcel_values, dtype=float)
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
        cmap = plt.get_cmap("gist_ncar", n_bins)
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
    if receptor_lat is not None and receptor_lon is not None:
        ax.scatter(receptor_lon, receptor_lat, marker="^", s=80, c="red", edgecolors="black", linewidths=0.4, zorder=7, transform=PLATE_CARREE)
        if receptor_radius_m is not None and receptor_radius_m > 0:
            ang = np.linspace(0, 2 * np.pi, 181)
            lat_scale = 111320.0
            lon_scale = max(np.cos(np.deg2rad(receptor_lat)) * 111320.0, 1e-6)
            ax.plot(receptor_lon + (receptor_radius_m / lon_scale) * np.cos(ang), receptor_lat + (receptor_radius_m / lat_scale) * np.sin(ang), color="red", linestyle="--", linewidth=1.0, transform=PLATE_CARREE)
    if seed_bbox is not None:
        lon_min, lat_min, lon_max, lat_max = seed_bbox
        ax.plot([lon_min, lon_max, lon_max, lon_min, lon_min], [lat_min, lat_min, lat_max, lat_max, lat_min], color="black", linestyle=":", linewidth=1.0, transform=PLATE_CARREE)
    cax = fig.add_axes([0.2, 0.05, 0.6, 0.03])
    plt.colorbar(mesh, cax=cax, orientation="horizontal", label=colorbar_label)
    if title:
        ax.set_title(title)
    fig.savefig(out_path, dpi=figure_dpi)
    plt.close(fig)


def plot_mpas_vertical_distribution(parcels, out_path, z_min, z_max, figure_dpi=200):
    heights = np.asarray(parcels.get("z_init", []), dtype=float)
    if heights.size == 0:
        return
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(np.arange(heights.size), heights / 1000.0, s=14, c=heights, cmap="turbo", edgecolors="black", linewidths=0.15)
    ax.set_xlabel("Parcel index")
    ax.set_ylabel("Initial altitude, km")
    ax.grid(True, linestyle=":", linewidth=0.4, alpha=0.6)
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


def plot_mpas_hourly_snapshots(lat_deg, lon_deg, traj_lon, traj_lat, traj_active, traj_z, time_indices, out_dir, figure_dpi=200, map_extent=None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    times = np.asarray(time_indices, dtype=int)
    n_saved = 0
    for snap_idx, time_idx in enumerate(times):
        fig, ax = _setup_geo_axes(lat_deg, lon_deg, map_extent=map_extent)
        lon = traj_lon[snap_idx]
        lat = traj_lat[snap_idx]
        active = np.asarray(traj_active, dtype=bool)[snap_idx]
        z = np.asarray(traj_z[snap_idx], dtype=float)
        ax.scatter(lon[active], lat[active], s=PARCEL_MARKER_SIZE, c=z[active] / 1000.0, cmap="turbo", edgecolors=PARCEL_MARKER_EDGE, linewidths=PARCEL_MARKER_LINEWIDTH, alpha=PARCEL_MARKER_ALPHA, transform=PLATE_CARREE, zorder=5)
        ax.set_title(f"Parcel positions at snapshot {time_idx}")
        fig.savefig(out_dir / f"parcel_positions_hour.{snap_idx:04d}.png", dpi=figure_dpi)
        plt.close(fig)
        n_saved += 1
    return n_saved


def plot_emission_matrix(emission, time_edges, z_bins, out_path, figure_dpi=200, colorbar_label="Parcel count"):
    fig, ax = plt.subplots(figsize=(10, 8))
    mesh = ax.pcolormesh(time_edges, np.asarray(z_bins, dtype=float) / 1000.0, emission, shading="auto", cmap="viridis")
    ax.set_xlabel("Time")
    ax.set_ylabel("Height, km")
    cbar = plt.colorbar(mesh, ax=ax)
    cbar.set_label(colorbar_label)
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

    start_time = None
    if args.start_time is not None:
        s = args.start_time.replace("_", "T")
        start_time = np.datetime64(s)
    if start_time is None:
        start_time = times_arr[-1]
    if start_time < times_arr[0] or start_time > times_arr[-1]:
        raise ValueError("start time outside MPAS time range.")
    start_idx = int(np.argmin(np.abs(times_arr - start_time)))

    with Dataset(args.column, "r") as ds_col:
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
        column = column * float(args.column_coef)

    seed_bbox = tuple(args.seed_bbox) if args.seed_bbox is not None else None
    parcels_init = generate_parcels_from_column(column, lat, lon, zmid, start_idx, args.threshold, args.n_columns, args.n_vert, args.z_min, args.z_max, seed_bbox=seed_bbox)
    if args.n_vert > 0:
        cell_mass = np.asarray(column, dtype=float) * np.asarray(area, dtype=float)
        parcel_mass = cell_mass[parcels_init["cell"]] / float(args.n_vert)
        parcels_init["mass"] = parcel_mass
    else:
        parcels_init["mass"] = np.zeros_like(parcels_init["z"], dtype=float)

    if args.seeds_figure:
        plot_mpas_column_and_parcels(column, lat, lon, parcels_init, args.seeds_figure, threshold=args.threshold, receptor_lat=args.receptor_lat, receptor_lon=args.receptor_lon, receptor_radius_m=args.receptor_radius, seed_bbox=seed_bbox, title=f"Parcel seeds at {times_arr[start_idx]}", colorbar_label=args.colorbar_label, figure_dpi=args.figure_dpi, map_extent=tuple(args.map_extent) if args.map_extent is not None else None)
    if args.seeds_vertical_figure:
        plot_mpas_vertical_distribution(parcels_init, args.seeds_vertical_figure, args.z_min, args.z_max, args.figure_dpi)

    settling_profile = None
    if args.aer_type is not None:
        settling_profile = dict(heights_m=np.asarray(Z_M, dtype=float), velocity_ms=np.asarray(SETTLING_VEL_MS[args.aer_type], dtype=float))

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

    if args.trajectory_figure:
        plot_mpas_trajectories(column, lat, lon, result["trajectory_lon"], result["trajectory_lat"], result["trajectory_active"], np.where(result["arrived"])[0], parcels_init["z_init"], args.trajectory_figure, threshold=args.threshold, receptor_lat=args.receptor_lat, receptor_lon=args.receptor_lon, receptor_radius_m=args.receptor_radius, seed_bbox=seed_bbox, title="MPAS parcel trajectories", colorbar_label=args.colorbar_label, figure_dpi=args.figure_dpi, map_extent=tuple(args.map_extent) if args.map_extent is not None else None)

    if args.trajectory_age:
        ages = np.maximum((times_arr[start_idx] - result["arrival_time"]) / 3600.0, 0.0)
        plot_mpas_trajectories(column, lat, lon, result["trajectory_lon"], result["trajectory_lat"], result["trajectory_active"], np.where(result["arrived"])[0], ages[result["arrived"]], args.trajectory_age, threshold=args.threshold, receptor_lat=args.receptor_lat, receptor_lon=args.receptor_lon, receptor_radius_m=args.receptor_radius, seed_bbox=seed_bbox, title="MPAS trajectories by age", colorbar_label="Age, h", figure_dpi=args.figure_dpi, map_extent=tuple(args.map_extent) if args.map_extent is not None else None)

    if args.output_txt:
        arrived = result["arrived"]
        arrival_z = result["arrival_z"][arrived]
        arrival_time = result["arrival_time"][arrived]
        z_bins = zmid[np.argmin(np.abs(lat - args.receptor_lat) + np.abs(lon - args.receptor_lon))]
        z_edges = np.concatenate(([max(0.0, z_bins[0] - 0.5 * (z_bins[1] - z_bins[0]))], 0.5 * (z_bins[:-1] + z_bins[1:]), [z_bins[-1] + 0.5 * (z_bins[-1] - z_bins[-2])]))
        t_sec = np.maximum(arrival_time, 0.0)
        t_edges = np.arange(0.0, max(float(np.nanmax(t_sec)) if t_sec.size else 0.0, args.arrival_bin_minutes * 60.0) + args.arrival_bin_minutes * 60.0, args.arrival_bin_minutes * 60.0)
        if t_edges.size < 2:
            t_edges = np.array([0.0, args.arrival_bin_minutes * 60.0])
        emission = np.zeros((z_edges.size - 1, t_edges.size - 1), dtype=float)
        if t_sec.size:
            t_bin = np.clip(np.digitize(t_sec, t_edges) - 1, 0, t_edges.size - 2)
            z_bin = np.clip(np.digitize(arrival_z, z_edges) - 1, 0, z_edges.size - 2)
            np.add.at(emission, (z_bin, t_bin), 1.0)
        with open(args.output_txt, "w", encoding="utf-8") as fh:
            fh.write("time " + " ".join(f"{v:.0f}" for v in t_edges[:-1]) + "\n")
            fh.write("height " + " ".join(f"{v:.1f}" for v in z_edges[:-1] / 1000.0) + "\n")
            for row in emission[::-1]:
                fh.write(" ".join(f"{int(v)}" for v in row) + "\n")
        if args.output_figure:
            plot_emission_matrix(emission, t_edges / 3600.0, z_edges / 1000.0, args.output_figure, args.figure_dpi, colorbar_label=args.colorbar_label)


def run_forwtraj(args):
    data = read_mpas_history(args.input)
    times = data["times"]
    lat = data["lat_deg"]
    lon = data["lon_deg"]
    zmid = data["zmid"]
    times_arr = np.asarray(times)
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
    src_idx = int(_nearest_cells(data["tree"], np.asarray([args.source_lat]), np.asarray([args.source_lon]))[0])
    parcels = generate_parcels_from_point(lon[src_idx], lat[src_idx], src_idx, args.n_vert, args.z_min, args.z_max)
    source_field = np.zeros(lat.size, dtype=float)
    source_field[src_idx] = 1.0

    result = advect_parcels_forward(parcels, times, data["u"], data["v"], data["w"], zmid, lat, lon, start_idx, end_idx, args.integration_dt)
    if args.seeds_figure:
        plot_mpas_seed_locations(lat, lon, parcels, args.seeds_figure, title="MPAS parcel seeds", figure_dpi=args.figure_dpi, map_extent=tuple(args.map_extent) if args.map_extent is not None else None)
    if args.seeds_vertical_figure:
        plot_mpas_vertical_distribution(parcels, args.seeds_vertical_figure, args.z_min, args.z_max, args.figure_dpi)
    if args.hourly_figures:
        plot_mpas_hourly_snapshots(lat, lon, result["trajectory_lon"], result["trajectory_lat"], result["trajectory_active"], result["trajectory_z"], result["trajectory_time_indices"], args.hourly_output_dir, figure_dpi=args.figure_dpi, map_extent=tuple(args.map_extent) if args.map_extent is not None else None)
    if args.height_figure:
        plot_mpas_trajectories(source_field, lat, lon, result["trajectory_lon"], result["trajectory_lat"], result["trajectory_active"], np.arange(result["trajectory_lon"].shape[1]), parcels["z_init"], args.height_figure, threshold=None, receptor_lat=args.source_lat, receptor_lon=args.source_lon, receptor_radius_m=0.0, seed_bbox=None, title="MPAS trajectories by initial height", colorbar_label="Height, m", figure_dpi=args.figure_dpi, map_extent=tuple(args.map_extent) if args.map_extent is not None else None)
    if args.age_figure:
        ages = np.maximum((times_arr[end_idx] - times_arr[start_idx]) / np.timedelta64(1, "h"), 0.0)
        plot_mpas_trajectories(source_field, lat, lon, result["trajectory_lon"], result["trajectory_lat"], result["trajectory_active"], np.arange(result["trajectory_lon"].shape[1]), np.full(result["trajectory_lon"].shape[1], float(ages)), args.age_figure, threshold=None, receptor_lat=args.source_lat, receptor_lon=args.source_lon, receptor_radius_m=0.0, seed_bbox=None, title="MPAS trajectories by age", colorbar_label="Age, h", figure_dpi=args.figure_dpi, map_extent=tuple(args.map_extent) if args.map_extent is not None else None)
