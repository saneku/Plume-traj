# Plume Trajectory Tools

This repository provides two related scripts for Lagrangian parcel tracking using WRF winds:

- `plume_backtraj.py`: back-trajectory analysis from a receptor to reconstruct time–height emission scenarios.
- `plume_forwtraj.py`: forward advection of parcels released from a source column.

Each tool has its own detailed documentation and examples.

## Documentation

- Back-trajectory guide: `plume_backtraj.md`
- Forward-trajectory guide: `plume_forwtraj.md`

## Plotting Helpers

Both workflows can save a pickle file with `--state-pickle`. Use the matching helper to regenerate figures:

- Backward replot: `python plot_backtraj.py run_state.pkl`
- Forward replot: `python plot_forwtraj.py forward_run.pkl`

## Installation

Install required packages (conda example):

```bash
conda install numpy scipy netcdf4 matplotlib cartopy
```

or with `pip`:

```bash
pip install numpy scipy netCDF4 matplotlib cartopy
```

## Quick Start

See the linked guides for full command-line options, inputs, and outputs. They include:

- Required WRF fields and expected file formats
- Full argument lists
- Plot outputs and diagnostics
- Re-plotting from saved pickle state

## Output Files

The scripts write PNG figures (trajectories, matrices, and diagnostics) in the current working directory unless you provide explicit output paths. Pickle files (`--state-pickle`) are saved where you specify and can be used to replot without re-running the advection.

## Misc Utilities

The `misc/` folder contains helper scripts and data:

- `misc/aggregate_backtraj.py`: aggregate multiple back-trajectory pickle runs and replot combined diagnostics.
- `misc/aggregate_forwtraj.py`: aggregate multiple forward-trajectory pickle runs and replot height/age maps.
- `misc/regrid_wrf.py`: utilities for regridding WRF fields or related inputs.
- `misc/clean_files.py`: helper for cleaning intermediate or generated files.
- `misc/settling_velocity_data.py`: lookup tables for aerosol settling velocity profiles.

## Citation

If you use this code in a publication, please cite this repository and describe the WRF configuration and any post-processing choices (e.g., settling, vertical release range).
