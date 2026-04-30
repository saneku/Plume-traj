"""Small plotting primitives shared by WRF and MPAS backends."""

import cartopy.crs as ccrs


PLATE_CARREE = ccrs.PlateCarree()


def normalize_map_extent(map_extent):
    """Convert (WEST, SOUTH, EAST, NORTH) to Cartopy extent order."""
    west, south, east, north = map_extent
    return [west, east, south, north]


def plot_seed_bbox(ax, seed_bbox):
    """Draw a seed bounding box as a dotted black outline if provided."""
    if seed_bbox is None:
        return
    lon_min, lat_min, lon_max, lat_max = seed_bbox
    if lon_min > lon_max:
        lon_min, lon_max = lon_max, lon_min
    if lat_min > lat_max:
        lat_min, lat_max = lat_max, lat_min
    lons = [lon_min, lon_max, lon_max, lon_min, lon_min]
    lats = [lat_min, lat_min, lat_max, lat_max, lat_min]
    ax.plot(
        lons,
        lats,
        color="black",
        linestyle=":",
        linewidth=1.0,
        transform=PLATE_CARREE,
        zorder=5,
    )
