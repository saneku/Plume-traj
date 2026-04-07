# Misc Utilities (`misc/`)

This folder contains helper scripts used by the main trajectory workflows.

## Scripts

### `aggregate_backtraj.py`
Aggregates multiple back-trajectory run pickles (typically ash runs) into one combined analysis.

- Loads per-run `run_ash.pkl` state files from run directories.
- Concatenates trajectory arrays across runs.
- Sums emission and mass-emission matrices.
- Regenerates combined figures and TXT matrices (trajectories, age, emission-time, arrival-height, missed trajectories, emission heatmaps).

Typical use:
- Compare or merge several ash size-bin back-trajectory runs into one view.

### `aggregate_forwtraj.py`
Aggregates multiple forward-trajectory pickle outputs into one combined trajectory dataset.

- Loads pickles matching a glob pattern.
- Aligns time dimensions (padding when needed).
- Concatenates parcel trajectories and optional height-history arrays.
- Regenerates aggregated height-colored and age-colored trajectory maps.

Typical use:
- Combine forward runs from multiple aerosol bins or scenario variants.

### `netcdf_clean_brush.py`
Interactive NetCDF field editor with a map-based eraser tool.

- Opens all `.nc` files in a folder.
- Displays a selected 2D field on a Cartopy map.
- Lets you erase regions (set to `NaN`) with an adjustable brush radius.
- Saves edits back to the source NetCDF file.

Typical use:
- Manually clean artifacts/noise in satellite-derived 2D fields before trajectory seeding.

### `regrid_wrf.py`
Regrids one or more 2D/3D variables from a source grid onto a destination WRF grid.

- Reads source values and source coordinates.
- Reads destination grid coordinates.
- Interpolates with `scipy.interpolate.griddata` (linear, then nearest fill for edge NaNs).
- Writes a new NetCDF output file on the destination grid.

Typical use:
- Prepare satellite or derived fields so they match the WRF grid used by trajectory scripts.

### `settling_velocity_data.py`
Reference data module for gravitational settling.

- Provides `Z_M`: reference height levels (meters).
- Provides `SETTLING_VEL_MS`: settling velocity profiles by aerosol type (`sulf`, `ash1` ... `ash10`).

Typical use:
- Imported by `plume_backtraj.py` and `plume_forwtraj.py` when `--aer-type` is enabled.

## Notes

- Most scripts here are helpers for the main tools in the repository root:
  - `plume_backtraj.py`
  - `plume_forwtraj.py`
- Aggregation and plotting scripts expect pickle structures produced by the current versions of the main scripts.
