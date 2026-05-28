# Plume Trajectory Tools

This repository provides two related scripts for Lagrangian parcel tracking using WRF winds or MPAS history files:

- `plume_backtraj.py`: back-trajectory analysis from a 2d field to reconstruct time–height emission scenarios.
- `plume_forwtraj.py`: forward advection of parcels released from a source column, with optional hourly parcel-location snapshots and a deposited-parcel map.

Backend-specific helpers are grouped under `src/`: `src/plume_wrf.py`, `src/plume_mpas.py`, and the shared base class in `src/plume_base.py`. The root scripts remain the user-facing entry points.


## Documentation

- Back-trajectory guide: `plume_backtraj.md`
- Forward-trajectory guide: `plume_forwtraj.md`
- Paper style pdf: `doc/plume-traj.pdf`

## Plotting Helpers

Both workflows can save a pickle file with `--state-pickle`, which can be used to regenerate figures:

- Backward replot: `python plot_backtraj.py run_state.pkl`
- Forward replot: `python plot_forwtraj.py forward_run.pkl`

## Installation

Install required packages (conda example):

```bash
conda install numpy scipy netcdf4 matplotlib cartopy xarray
```

or with `pip`:

```bash
pip install numpy scipy netCDF4 matplotlib cartopy xarray
```

## Quick Start

See the linked guides for full command-line options, inputs, and outputs. They include:

- Required WRF fields
- Full argument lists
- Plot outputs and diagnostics
- Re-plotting from saved pickle state
- Optional `--map-extent WEST SOUTH EAST NORTH` override for custom map bounds
- Optional hourly snapshots by setting `--hourly-output-dir <dir>`
- Time arguments are expected in the same basis as WRF `Times` (normally UTC); no timezone conversion is applied by the scripts
- Select the meteorology backend with `--target wrf` or `--target mpas`
- Forward mode supports `--emission-matrix <txt>` for time-height release schedules
- Forward mode supports `--emission-timeseries <txt>` for time-varying release intensity with uniform vertical distribution from `--z-min/--z-max/--n-vert`
- In forward mode without `--emission-matrix` or `--emission-timeseries`, release can use `--source-lat/--source-lon` or random source columns from `--seed-bbox` with `--n-columns` (WRF and MPAS)
- Forward `--aer-type` applies aerosol gravitational settling for both WRF and MPAS
- Recommended matrix time headers are `time_offset_h` (hours from `--start-time`) or `time_offset_s` (seconds from `--start-time`); legacy `time` is still accepted
- MPAS mode reads `history*.nc` files with `latCell`, `lonCell`, `zgrid`, `uReconstructZonal`, `uReconstructMeridional`, and `w`
- In WRF mode, `--column` is the gridded source field on the WRF mesh; in MPAS mode, it is the source field on the MPAS cell mesh
- If you point MPAS mode at a history file for `--column`, the current backend uses the requested variable as a cell field and collapses any vertical dimension to a 2-D source map by summing over levels

## Output Files

The scripts write PNG figures (trajectories, 2d matrices, and diagnostics) in the current working directory unless you provide explicit output paths. Hourly parcel-location maps are saved only when `--hourly-output-dir` is set, using `parcel_positions_hour.XXXX.png` in that directory. `--deposition-figure` saves a deposited-parcel-only map colored by deposition hour since release. Pickle files (`--state-pickle`) are saved where you specify and can be used to replot without re-running the advection.

## Compact Run Sheet

Backward mode:

```bash
# WRF-Chem
python plume_backtraj.py \
  --target wrf \
  --input wrfout_d01_2025-11* \
  --column SO2_COLUMN.nc \
  --column-var SO2_COLUMN \
  --receptor-lat 13.51 \
  --receptor-lon 40.71 \
  --state-pickle run_state.pkl

# MPAS-Chem
python plume_backtraj.py \
  --target mpas \
  --input history.2025-11-24_*.nc \
  --column mpas_source.nc \
  --column-var so2 \
  --receptor-lat 13.51 \
  --receptor-lon 40.71 \
  --state-pickle run_state.pkl
```

Forward mode:

```bash
# WRF-Chem
python plume_forwtraj.py \
  --target wrf \
  --input wrfout_d01_2025-11* \
  --start-time 2025-11-23T08:30:00 \
  --end-time 2025-11-24T12:00:00 \
  --source-lat 13.51 \
  --source-lon 40.71 \
  --state-pickle forward_run.pkl

# MPAS-Chem
python plume_forwtraj.py \
  --target mpas \
  --input history.2025-11-24_*.nc \
  --start-time 2025-11-23T08:30:00 \
  --end-time 2025-11-24T12:00:00 \
  --source-lat 13.51 \
  --source-lon 40.71 \
  --state-pickle forward_run.pkl

# MPAS-Chem random source columns in bbox (non-emission-matrix mode)
python plume_forwtraj.py \
  --target mpas \
  --input history.2025-11-24_*.nc \
  --start-time 2025-11-23T08:30:00 \
  --end-time 2025-11-24T12:00:00 \
  --seed-bbox 40.0 10.0 40.1 10.1 \
  --n-columns 25 \
  --z-min 1000 \
  --z-max 23000 \
  --n-vert 30 \
  --state-pickle forward_run.pkl

# Forward mode with time-height emission matrix
python plume_forwtraj.py \
  --target wrf \
  --input wrfout_d01_2025-11* \
  --start-time 2025-11-23T08:30:00 \
  --end-time 2025-11-24T12:00:00 \
  --source-lat 13.51 \
  --source-lon 40.71 \
  --emission-matrix emission_matrix.txt \
  --state-pickle forward_run.pkl

# Forward mode with time-intensity emission timeseries
python plume_forwtraj.py \
  --target wrf \
  --input wrfout_d01_2025-11* \
  --start-time 2025-11-23T08:30:00 \
  --end-time 2025-11-24T12:00:00 \
  --source-lat 13.51 \
  --source-lon 40.71 \
  --z-min 100 \
  --z-max 1500 \
  --n-vert 15 \
  --emission-timeseries emission_timeseries.txt \
  --state-pickle forward_run.pkl
```

Emission-matrix notes:
- File format:
  - first non-empty line: `time_offset_h ...` or `time_offset_s ...` (recommended), or legacy `time ...`
  - second non-empty line: `height ...`
  - remaining rows: matrix values (one row per height, from highest height to lowest height)
- In matrix native mode, times/heights/counts come from the file.
- If all three are provided together with `--emission-matrix`: `--z-min --z-max --n-vert`
  - matrix times are still used
  - matrix heights and counts are ignored
  - heights are rebuilt from `z-min..z-max` with `n-vert` levels
  - each (time,height) cell uses one parcel

Emission-timeseries notes:
- File format:
  - first non-empty line: `time_offset_h parcels` or `time_offset_s parcels`
  - remaining rows: one release time and one total parcel count
- `--z-min`, `--z-max`, and `--n-vert` are required.
- Counts are distributed as evenly as possible over uniformly spaced heights between `z-min` and `z-max`.

## Misc Utilities

The `misc/` folder contains helper scripts and data:

- `misc/aggregate_backtraj.py`: aggregate multiple back-trajectory pickle runs and replot combined diagnostics, with optional hourly maps via `--hourly-output-dir`.
- `misc/aggregate_forwtraj.py`: aggregate multiple forward-trajectory pickle runs and replot height/age maps.
- `misc/regrid_to_wrf.py`: utilities for regridding cleaned fields onto a WRF grid.
- `misc/regrid_to_mpas.py`: utilities for regridding cleaned fields onto an MPAS cell grid.
- `misc/netcdf_clean_brush.py`: helper for cleaning 2d source fields before regridding.
- `misc/settling_velocity_data.py`: lookup tables for aerosol (ash and sulfate) settling velocity profiles.

## Output Naming

Where WRF and MPAS produce the same diagnostic figure family, the aggregation scripts now use the same filenames:

- `aggregated_trajectories.png`
- `aggregated_parcel_locations.png`
- `aggregated_trajectory_ages.png`
- `aggregated_trajectory_emission_time.png`
- `aggregated_trajectory_arrival_height.png`
- `aggregated_missed_trajectories.png`
- `aggregated_emission_matrix.png`
- `aggregated_mass_matrix.png`

The grid-specific difference is in the input geometry:

- WRF uses structured grid indices and WRF meteorology files.
- MPAS uses cell-based history files and unstructured mesh coordinates.

## Citation

If you use this code in a publication, please cite as follows:
Ukhov et al., In the Wake of the Hayli Gubbi Eruption, 2026.
