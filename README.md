# Plume Trajectory Tools

This repository provides two related scripts for Lagrangian parcel tracking using WRF winds or MPAS history files:

- `plume_backtraj.py`: back-trajectory analysis from a 2d field to reconstruct time–height emission scenarios.
- `plume_forwtraj.py`: forward advection of parcels released from a source column, with optional hourly parcel-location snapshots and a deposited-parcel map.


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
- Optional forward hourly snapshots with `--hourly-figures`
- Time arguments are expected in the same basis as WRF `Times` (normally UTC); no timezone conversion is applied by the scripts
- Select the meteorology backend with `--target wrf` or `--target mpas`
- MPAS mode reads `history*.nc` files with `latCell`, `lonCell`, `zgrid`, `uReconstructZonal`, `uReconstructMeridional`, and `w`
- In WRF mode, `--column` is the gridded source field on the WRF mesh; in MPAS mode, it is the source field on the MPAS cell mesh
- If you point MPAS mode at a history file for `--column`, the current backend uses the requested variable as a cell field and collapses any vertical dimension to a 2-D source map by summing over levels

## Output Files

The scripts write PNG figures (trajectories, 2d matrices, and diagnostics) in the current working directory unless you provide explicit output paths. In forward mode, `--hourly-figures` saves hourly parcel-location maps named `parcel_positions_hour.XXXX.png` (output location controlled by `--hourly-output-dir`), and `--deposition-figure` saves a deposited-parcel-only map colored by deposition hour since release. Pickle files (`--state-pickle`) are saved where you specify and can be used to replot without re-running the advection.

## Misc Utilities

The `misc/` folder contains helper scripts and data:

- `misc/aggregate_backtraj.py`: aggregate multiple back-trajectory pickle runs and replot combined diagnostics.
- `misc/aggregate_forwtraj.py`: aggregate multiple forward-trajectory pickle runs and replot height/age maps.
- `misc/regrid_wrf.py`: utilities for regridding WRF fields or related inputs.
- `misc/netcdf_clean_brush.py`: helper for cleaning the 2d fields (for example, column loadings).
- `misc/settling_velocity_data.py`: lookup tables for aerosol (ash and sulfate) settling velocity profiles.

## Citation

If you use this code in a publication, please cite as follows:
Ukhov et al., In the Wake of the Hayli Gubbi Eruption, 2026.
