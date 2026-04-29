#!/usr/bin/env python
import argparse
import sys
from pathlib import Path
import warnings

from typing import List
import numpy as np
from netCDF4 import Dataset
from scipy.interpolate import griddata

# Suppress common deprecation warnings from numpy and netCDF4
warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*`np\.bool` is a deprecated alias.*")
warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*tostring\(\) is deprecated.*")


def _diag(msg: str) -> None:
    """Lightweight diagnostic logger."""
    print(f"[diag] {msg}")


def regrid_mpas_variable(
    source_data_path: str,
    dest_grid_path: str,
    output_path: str,
    variable_names: List[str],
    time_index: int = 0,
) -> None:
    """
    Clean a source grid first, then interpolate variables to an MPAS cell grid.
    This is the MPAS-side companion to `regrid_to_wrf.py`.

    Args:
        source_data_path: Path to the cleaned source NetCDF file.
        dest_grid_path: Path to the NetCDF file defining the destination MPAS grid.
                        (for example, an MPAS mesh or history file).
        output_path: Path for the newly created NetCDF file with interpolated data.
        variable_names: A list of variable names to interpolate.
        time_index: The time index to process from the source file.


python ./regrid_to_mpas.py \
  --source-file cleaned_source.nc \
  --variables so2 \
  --dest-file-grid MPAS_mesh_or_history.nc \
  --output-file regridded_to_mpas.nc \


python ./regrid_to_mpas.py \
  --source-file cleaned_source.nc \
  --variables aerosol_index_354_388 \
  --dest-file-grid MPAS_mesh_or_history.nc \
  --output-file regridded_to_mpas.nc \

    """
    # --- 1. Read Source Grid and Data ---
    _diag(f"Reading source grid and data from: {source_data_path}")
    with Dataset(source_data_path, "r") as src_ds:
        # Using XLAT_V and XLONG_U for a more robust grid definition if available
        if "XLAT" in src_ds.variables and "XLONG" in src_ds.variables:
            lat_var, lon_var = src_ds.variables["XLAT"], src_ds.variables["XLONG"]
            _diag("Found source coordinate variables: XLAT, XLONG.")
        elif "lat" in src_ds.variables and "lon" in src_ds.variables:
            lat_var, lon_var = src_ds.variables["lat"], src_ds.variables["lon"]
            _diag("Found source coordinate variables: lat, lon.")
        else:
            _diag("Could not find coordinate variables (XLAT/XLONG or lat/lon) in source file.")
            sys.exit(1)

        # Handle both 2D and 3D coordinate variables
        if lat_var.ndim == 3:
            src_lat = lat_var[time_index, :, :].flatten()
        elif lat_var.ndim == 2:
            src_lat = lat_var[:, :].flatten()
        else:
            _diag(f"Unsupported rank for latitude variable '{lat_var.name}': {lat_var.ndim}")
            sys.exit(1)
        src_lon = lon_var[time_index, :, :].flatten() if lon_var.ndim == 3 else lon_var[:, :].flatten()

        source_points = np.column_stack((src_lon, src_lat))
        _diag(f"Source grid has {source_points.shape[0]} points.")

        # Read all specified variables into a dictionary
        source_vars = {}
        for var_name in variable_names:
            if var_name not in src_ds.variables:
                _diag(f"Variable '{var_name}' not found in source file. Skipping.")
                continue
            data_var = src_ds.variables[var_name]
            if data_var.ndim == 3:
                var_data = data_var[time_index, ...]
            elif data_var.ndim == 2:
                var_data = data_var[...]
            else:
                _diag(f"Unsupported rank for data variable '{var_name}': {data_var.ndim}. Skipping.")
                continue
            source_vars[var_name] = var_data.flatten()
            _diag(f"Read source variable '{var_name}' with shape {var_data.shape}.")

    if not source_vars:
        _diag("No valid variables found to process. Exiting.")
        return

    # --- 2. Read Destination Grid ---
    _diag(f"Reading destination grid from: {dest_grid_path}")
    with Dataset(dest_grid_path, "r") as dest_ds:
        if "latCell" in dest_ds.variables and "lonCell" in dest_ds.variables:
            lat_var, lon_var = dest_ds.variables["latCell"], dest_ds.variables["lonCell"]
            _diag("Found destination coordinate variables: latCell, lonCell.")
            dest_lat = np.rad2deg(np.asarray(lat_var[:], dtype=float))
            dest_lon = np.rad2deg(np.asarray(lon_var[:], dtype=float))
        elif "lat" in dest_ds.variables and "lon" in dest_ds.variables:
            lat_var, lon_var = dest_ds.variables["lat"], dest_ds.variables["lon"]
            _diag("Found destination coordinate variables: lat, lon.")
            dest_lat = np.asarray(lat_var[:], dtype=float)
            dest_lon = np.asarray(lon_var[:], dtype=float)
        else:
            _diag("Could not find coordinate variables (latCell/lonCell or lat/lon) in destination file.")
            sys.exit(1)
        if dest_lat.ndim != 1 or dest_lon.ndim != 1:
            dest_lat = np.asarray(dest_lat).reshape(-1)
            dest_lon = np.asarray(dest_lon).reshape(-1)
        _diag(f"Destination grid has {dest_lat.size} cells.")

    # --- 3. Create Output File and Copy Dimensions/Attributes ---
    _diag(f"Creating output file: {output_path}")
    with Dataset(dest_grid_path, "r") as dest_ds, Dataset(
        output_path, "w", format="NETCDF4"
    ) as out_ds:
        # Copy global attributes
        out_ds.setncatts(dest_ds.__dict__)
        out_ds.TITLE = f"Regridded data from {Path(source_data_path).name}"

        # Copy dimensions from destination grid
        for name, dimension in dest_ds.dimensions.items():
            # Create dimension, but don't limit Time if it exists
            out_ds.createDimension(
                name, (len(dimension) if not dimension.isunlimited() else None)
            )

        # Define and copy only essential coordinate variables
        essential_vars = ["latCell", "lonCell", "lat", "lon", "Times", "xtime"]
        for name in essential_vars:
            if name in dest_ds.variables:
                variable = dest_ds.variables[name]
                out_var = out_ds.createVariable(name, variable.datatype, variable.dimensions)
                out_var.setncatts(variable.__dict__)
                if "Time" not in variable.dimensions and "time" not in variable.dimensions:
                    out_var[:] = variable[:]
                else:  # Handle time carefully, only copy the first time step
                    out_var[0] = variable[0]

        # --- 4. Interpolate and Write Variables ---
        for var_name, src_values in source_vars.items():
            _diag(f"Interpolating variable: {var_name}...")
            # Perform interpolation from source points to destination grid
            regridded_data = griddata(
                source_points, src_values, (dest_lon, dest_lat), method="linear"
            )

            # Handle NaNs at the edges by filling with nearest value
            nan_mask = np.isnan(regridded_data)
            if np.any(nan_mask):
                _diag("Filling NaNs from interpolation with nearest-neighbor values.")
                nearest_data = griddata(
                    source_points, src_values, (dest_lon, dest_lat), method="nearest"
                )
                regridded_data[nan_mask] = nearest_data[nan_mask]

            # Create the variable in the output file
            if var_name in dest_ds.variables and any(dim.lower() == "time" for dim in dest_ds.variables[var_name].dimensions):
                out_var = out_ds.createVariable(var_name, "f4", ("Time", "nCells"))
                out_var[0, :] = regridded_data.reshape(-1)
            else:
                out_var = out_ds.createVariable(var_name, "f4", ("nCells",))
                out_var[:] = regridded_data.reshape(-1)

            _diag(f"Finished writing variable: {var_name}")

    _diag(f"Success! Regridded file saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Interpolate cleaned fields onto an MPAS cell grid."
    )
    parser.add_argument(
        "--source-file", required=True, help="Path to the cleaned source file to interpolate."
    )
    parser.add_argument(
        "--dest-file-grid",
        required=True,
        help=(
            "Path to the MPAS mesh or history file defining the target cell grid."
        ),
    )
    parser.add_argument(
        "--output-file", required=True, help="Path for the new NetCDF file to be created."
    )
    parser.add_argument(
        "--variables", required=True, nargs='+', help="Space-separated list of variable names to regrid (e.g., T2 PSFC)."
    )
    parser.add_argument(
        "--time-index", type=int, default=0, help="The time index from the source file to process (default: 0)."
    )
    args = parser.parse_args()

    regrid_mpas_variable(
        source_data_path=args.source_file,
        dest_grid_path=args.dest_file_grid,
        output_path=args.output_file,
        variable_names=args.variables,
        time_index=args.time_index,
    )
