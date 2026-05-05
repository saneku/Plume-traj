"""Shared map styling utilities for WRF/MPAS plots."""

from pathlib import Path

import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from shapely.geometry import Point
from shapely.ops import unary_union


LOCAL_CARTOPY_DATA = Path("/scratch/ukhova/iops/cartopy_data")
if LOCAL_CARTOPY_DATA.exists():
    cartopy.config["data_dir"] = str(LOCAL_CARTOPY_DATA)


PLATE_CARREE = ccrs.PlateCarree()


LAND_COLOR = "#d7dfdd"
WATER_COLOR = "#bfe5f2"
BORDER_COLOR = "white"
COAST_COLOR = "#5aaed0"
GRID_COLOR = "#8ec9dd"
WATER_LABEL_COLOR = "#1f7ea8"


COUNTRY_LABELS = [
    ("EGYPT", 31.2, 26.5, 10),
    ("SUDAN", 31.2, 15.0, 10),
    ("ERITREA", 39.0, 15.5, 9),
    ("ETHIOPIA", 39.5, 9.5, 10),
    ("DJIBOUTI", 42.5, 11.7, 7),
    ("SOMALIA", 48.3, 7.0, 10),
    ("SAUDI\nARABIA", 44.0, 22.5, 11),
    ("YEMEN", 45.5, 15.5, 10),
    ("OMAN", 56.5, 20.5, 10),
    ("UNITED ARAB\nEMIRATES", 54.1, 23.4, 6),
    ("QATAR", 51.15, 25.3, 8),
    ("BAHRAIN", 50.55, 26.05, 7),
    ("KUWAIT", 47.7, 29.2, 8),
    ("IRAQ", 45.2, 29.7, 10),
    ("IRAN", 55.0, 28.0, 11),
    ("JORDAN", 36.2, 29.5, 8),
]


WATER_LABELS = [
    ("Red Sea", 38.3, 20.0, 11, "normal", WATER_LABEL_COLOR, -65, "italic"),
    ("Gulf of Aden", 47.3, 12.5, 10, "normal", WATER_LABEL_COLOR, 0, "italic"),
    ("Arabian Gulf", 51.4, 27.5, 10, "normal", WATER_LABEL_COLOR, -48, "italic"),
    ("Gulf of Oman", 59.0, 24.5, 10, "normal", WATER_LABEL_COLOR, -24, "italic"),
    ("ARABIAN\nSEA", 59.8, 13.5, 11, "normal", WATER_LABEL_COLOR, 0, "italic"),
]


_LAND_UNION = None


def _extent_from_ax(ax):
    west, east, south, north = ax.get_extent(crs=PLATE_CARREE)
    return [west, south, east, north]


def _point_in_extent(x, y, extent):
    lon_min, lat_min, lon_max, lat_max = extent
    return (lon_min <= x <= lon_max) and (lat_min <= y <= lat_max)


def _add_label(
    ax,
    text,
    x,
    y,
    size=10,
    weight="bold",
    color="black",
    rotation=0,
    style="normal",
    ha="center",
    va="center",
    alpha=1.0,
    zorder=10,
):
    ax.text(
        x,
        y,
        text,
        transform=PLATE_CARREE,
        fontsize=size,
        fontweight=weight,
        color=color,
        rotation=rotation,
        fontstyle=style,
        ha=ha,
        va=va,
        alpha=alpha,
        zorder=zorder,
        path_effects=[pe.withStroke(linewidth=2.5, foreground="white", alpha=0.55)],
    )


def _build_land_union():
    global _LAND_UNION
    if _LAND_UNION is not None:
        return _LAND_UNION
    land_shp = shpreader.natural_earth(
        resolution="10m",
        category="physical",
        name="land",
    )
    reader = shpreader.Reader(land_shp)
    _LAND_UNION = unary_union(list(reader.geometries()))
    return _LAND_UNION


def _plot_water_only_grid(
    ax,
    extent,
    land_union,
    meridian_step=5,
    parallel_step=5,
    spacing=0.05,
):
    lon_min, lat_min, lon_max, lat_max = extent
    meridians = np.arange(
        np.ceil(lon_min / meridian_step) * meridian_step,
        lon_max + meridian_step,
        meridian_step,
    )
    parallels = np.arange(
        np.ceil(lat_min / parallel_step) * parallel_step,
        lat_max + parallel_step,
        parallel_step,
    )

    for lon in meridians:
        lats = np.arange(lat_min, lat_max + spacing, spacing)
        lons = np.full_like(lats, lon)
        water_mask = np.array([not land_union.contains(Point(x, y)) for x, y in zip(lons, lats)])
        start = None
        for i, is_water in enumerate(water_mask):
            if is_water and start is None:
                start = i
            elif (not is_water or i == len(water_mask) - 1) and start is not None:
                end = i if not is_water else i + 1
                if end - start > 1:
                    ax.plot(
                        lons[start:end],
                        lats[start:end],
                        transform=PLATE_CARREE,
                        color=GRID_COLOR,
                        linewidth=0.45,
                        alpha=0.75,
                        zorder=2.5,
                    )
                start = None

    for lat in parallels:
        lons = np.arange(lon_min, lon_max + spacing, spacing)
        lats = np.full_like(lons, lat)
        water_mask = np.array([not land_union.contains(Point(x, y)) for x, y in zip(lons, lats)])
        start = None
        for i, is_water in enumerate(water_mask):
            if is_water and start is None:
                start = i
            elif (not is_water or i == len(water_mask) - 1) and start is not None:
                end = i if not is_water else i + 1
                if end - start > 1:
                    ax.plot(
                        lons[start:end],
                        lats[start:end],
                        transform=PLATE_CARREE,
                        color=GRID_COLOR,
                        linewidth=0.45,
                        alpha=0.75,
                        zorder=2.5,
                    )
                start = None


def _add_scale_bar(ax, lon0=32.5, lat0=6.2, length_km=750):
    km_per_degree_lon = 111.32 * np.cos(np.deg2rad(lat0))
    length_deg = length_km / km_per_degree_lon
    segments = 5
    seg_km = length_km / segments
    seg_mi = seg_km * 0.621371
    seg_deg = length_deg / segments
    bar_height = 0.25

    for i in range(segments):
        color = "black" if i % 2 == 0 else "white"
        ax.add_patch(
            Rectangle(
                (lon0 + i * seg_deg, lat0),
                seg_deg,
                bar_height,
                facecolor=color,
                edgecolor="black",
                linewidth=0.6,
                transform=PLATE_CARREE,
                zorder=30,
            )
        )

    for i in range(segments + 1):
        x = lon0 + i * seg_deg
        km = int(i * seg_km)
        ax.plot(
            [x, x],
            [lat0, lat0 + 0.45],
            color="black",
            lw=0.6,
            transform=PLATE_CARREE,
            zorder=31,
        )
        ax.text(
            x,
            lat0 + 0.65,
            str(km),
            fontsize=8,
            ha="center",
            transform=PLATE_CARREE,
            zorder=31,
        )
        ax.text(
            x,
            lat0 - 0.55,
            f"{int(round(i * seg_mi))}",
            fontsize=8,
            ha="center",
            transform=PLATE_CARREE,
            zorder=31,
        )

    ax.text(
        lon0 + length_deg + 0.55,
        lat0 + 0.65,
        " km",
        fontsize=8,
        ha="left",
        transform=PLATE_CARREE,
        zorder=31,
    )
    ax.text(
        lon0 + length_deg + 0.55,
        lat0 - 0.55,
        " mi",
        fontsize=8,
        ha="left",
        transform=PLATE_CARREE,
        zorder=31,
    )


def _add_inset(ax, extent):
    lon_min, lat_min, lon_max, lat_max = extent
    lon_center = 0.5 * (lon_min + lon_max)
    lat_center = 0.5 * (lat_min + lat_max)
    inset = ax.inset_axes(
        [0.765, 0.045, 0.19, 0.19],
        projection=ccrs.Orthographic(lon_center, lat_center),
        zorder=50,
    )
    inset.set_global()
    inset.add_feature(
        cfeature.OCEAN.with_scale("110m"),
        facecolor="#d9eef6",
        zorder=1,
    )
    gl = inset.gridlines(
        crs=PLATE_CARREE,
        draw_labels=False,
        linewidth=0.35,
        color=GRID_COLOR,
        alpha=0.75,
        linestyle="-",
        zorder=2,
    )
    gl.xlocator = plt.FixedLocator(np.arange(-180, 181, 30))
    gl.ylocator = plt.FixedLocator(np.arange(-90, 91, 30))
    inset.add_feature(
        cfeature.LAND.with_scale("110m"),
        facecolor="#e6e6e6",
        edgecolor="white",
        linewidth=0.3,
        zorder=3,
    )
    inset.add_feature(
        cfeature.COASTLINE.with_scale("110m"),
        linewidth=0.25,
        edgecolor=COAST_COLOR,
        zorder=4,
    )
    inset.add_patch(
        Rectangle(
            (lon_min, lat_min),
            lon_max - lon_min,
            lat_max - lat_min,
            transform=PLATE_CARREE,
            facecolor="none",
            edgecolor="#e67e22",
            linewidth=1.6,
            zorder=10,
        )
    )
    inset.set_xticks([])
    inset.set_yticks([])
    inset.spines["geo"].set_linewidth(0.8)


def apply_map_style(ax, draw_labels=False, label_size=10):
    """Apply full shared map style from test.py: palette, grid, labels, inset, scale."""
    extent = _extent_from_ax(ax)
    lon_min, lat_min, lon_max, lat_max = extent

    ax.set_facecolor(WATER_COLOR)
    land_union = _build_land_union()

    ax.add_feature(
        cfeature.NaturalEarthFeature(
            "physical",
            "ocean",
            "10m",
            facecolor=WATER_COLOR,
            edgecolor="none",
        ),
        zorder=0,
    )

    _plot_water_only_grid(ax, extent, land_union, meridian_step=5, parallel_step=5, spacing=0.05)

    ax.add_feature(
        cfeature.NaturalEarthFeature(
            "physical",
            "land",
            "10m",
            facecolor=LAND_COLOR,
            edgecolor="none",
        ),
        zorder=3,
    )
    ax.add_feature(
        cfeature.NaturalEarthFeature(
            "physical",
            "lakes",
            "10m",
            facecolor=WATER_COLOR,
            edgecolor=COAST_COLOR,
            linewidth=0.5,
        ),
        zorder=4,
    )
    ax.add_feature(
        cfeature.NaturalEarthFeature(
            "physical",
            "rivers_lake_centerlines",
            "10m",
            facecolor="none",
            edgecolor=COAST_COLOR,
            linewidth=0.35,
        ),
        zorder=5,
    )
    ax.add_feature(
        cfeature.NaturalEarthFeature(
            "physical",
            "coastline",
            "10m",
            facecolor="none",
            edgecolor=COAST_COLOR,
            linewidth=0.6,
        ),
        zorder=6,
    )
    ax.add_feature(
        cfeature.NaturalEarthFeature(
            "cultural",
            "admin_0_boundary_lines_land",
            "10m",
            facecolor="none",
            edgecolor=BORDER_COLOR,
            linewidth=1.0,
        ),
        zorder=7,
    )

    gl = ax.gridlines(
        draw_labels=draw_labels,
        linewidth=0.45,
        color=GRID_COLOR,
        alpha=0.75,
        linestyle="-",
        zorder=2.5,
    )
    gl.top_labels = False
    gl.right_labels = False
    if draw_labels:
        gl.xlabel_style = {"size": label_size}
        gl.ylabel_style = {"size": label_size}

    ax.spines["geo"].set_visible(False)
    if not draw_labels:
        ax.set_xticks([])
        ax.set_yticks([])

    if lon_min <= 66 and lon_max >= 30 and lat_min <= 31 and lat_max >= 5:
        for name, x, y, size in COUNTRY_LABELS:
            if _point_in_extent(x, y, extent):
                _add_label(ax, name, x, y, size=size)
        for text, x, y, size, weight, color, rotation, style in WATER_LABELS:
            if _point_in_extent(x, y, extent):
                _add_label(
                    ax,
                    text,
                    x,
                    y,
                    size=size,
                    weight=weight,
                    color=color,
                    rotation=rotation,
                    style=style,
                )
        _add_scale_bar(ax, lon0=32.5, lat0=6.2, length_km=750)
        _add_inset(ax, extent)

    return gl
