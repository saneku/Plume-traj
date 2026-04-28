# Misc Utilities (`misc/`)

This folder contains helper scripts used by the main trajectory workflows.

## Scripts

### `aggregate_backtraj.py`
Aggregates multiple back-trajectory run pickles into one combined analysis.

- Loads per-run `run_ash.pkl` state files from run directories.
- Concatenates trajectory arrays across runs.
- Sums emission and mass-emission matrices.
- Regenerates combined figures and TXT matrices (trajectories, parcel seeds, age, emission-time, arrival-height, missed trajectories, emission heatmaps).
- Supports both WRF and MPAS run pickles.

Typical use:
- Compare or merge several ash size-bin back-trajectory runs into one view.

### `aggregate_forwtraj.py`
Aggregates multiple forward-trajectory pickle outputs into one combined trajectory dataset.

- Loads pickles matching a glob pattern.
- Aligns time dimensions (padding when needed).
- Concatenates parcel trajectories and optional height-history arrays.
- Regenerates aggregated height-colored and age-colored trajectory maps.
- Optionally saves deposited parcels only, colored by deposition hour (`--deposition-figure`).
- Optionally regenerates hourly parcel-location maps (`--hourly-figures`).
- Supports both WRF and MPAS run pickles.

Typical use:
- Combine forward runs from multiple aerosol bins or scenario variants.

### `netcdf_clean_brush.py`
Interactive NetCDF field editor with a map-based eraser tool.

- Opens all `.nc` files in a folder.
- Displays a selected 2D field on a Cartopy map.
- Lets you erase regions (set to `NaN`) with an adjustable brush radius.
- Saves edits back to the source NetCDF file.

Typical use:
- Manually clean artifacts/noise in satellite-derived 2D fields before regridding to WRF or MPAS.

### `regrid_to_wrf.py`
Regrids one or more 2D/3D variables from a cleaned source grid onto a destination WRF grid.

- Reads source values and source coordinates.
- Reads destination grid coordinates.
- Interpolates with `scipy.interpolate.griddata` (linear, then nearest fill for edge NaNs).
- Writes a new NetCDF output file on the destination grid.

Typical use:
- Prepare satellite or derived fields so they match the WRF grid used by trajectory scripts.

### `regrid_to_mpas.py`
Regrids one or more 2D/3D variables from a cleaned source grid onto an MPAS cell grid.

- Reads source values and source coordinates.
- Reads MPAS cell coordinates (`latCell`, `lonCell`).
- Interpolates with `scipy.interpolate.griddata` (linear, then nearest fill for edge NaNs).
- Writes a new NetCDF output file on the MPAS cell grid.

Typical use:
- Prepare cleaned satellite or derived fields so they match the MPAS mesh or MPAS history-grid layout used by the trajectory scripts.

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
- Forward trajectory runs can now optionally save hourly parcel-location maps via `plume_forwtraj.py --hourly-figures`.
- Aggregation and plotting scripts expect pickle structures produced by the current versions of the main scripts.
- The intended workflow for gridded observations is: clean the source field first, then interpolate it to WRF or MPAS with the matching regrid helper.

## WRF vs MPAS Output Families

When the same diagnostic exists in both modes, the scripts now write the same output names.

| Figure family | WRF | MPAS |
| --- | --- | --- |
| Trajectories | `aggregated_trajectories.png` | `aggregated_trajectories.png` |
| Parcel seeds | `aggregated_parcel_locations.png` | `aggregated_parcel_locations.png` |
| Age map | `aggregated_trajectory_ages.png` | `aggregated_trajectory_ages.png` |
| Emission-time map | `aggregated_trajectory_emission_time.png` | `aggregated_trajectory_emission_time.png` |
| Arrival-height map | `aggregated_trajectory_arrival_height.png` | `aggregated_trajectory_arrival_height.png` |
| Missed trajectories | `aggregated_missed_trajectories.png` | `aggregated_missed_trajectories.png` |
| Emission heatmap | `aggregated_emission_matrix.png` | `aggregated_emission_matrix.png` |
| Mass-weighted heatmap | `aggregated_mass_matrix.png` | `aggregated_mass_matrix.png` |

The difference is not in naming, but in the geometry behind the plots:

- WRF uses structured `i/j/k` fields and WRF grid coordinates.
- MPAS uses cell-based `lon/lat/z` trajectories and unstructured mesh coordinates.
