# SO₂ Plume Back-Trajectory Tool (`plume_backtraj.py`)

This script samples a satellite-retrieved SO₂ column on a WRF grid, launches Lagrangian parcels, advects them backward in time using 3‑D WRF winds (with linear interpolation between consecutive WRF times), and reconstructs a time–height SO₂ emission scenario at a prescribed receptor (usually the volcano vent).

Each seeded parcel carries a weight that represents a fraction of the sampled column load (`column2d * cell_area / n_vert`), so the back-trajectory analysis produces both simple parcel counts *and* mass-weighted totals. Both diagnostics can be written to TXT time–height matrices and plotted as PNG heatmaps in a format similar to `PrepEmisSources`’ `emission_scenario.py`.

![Fig2: Backward-trajectory reconstruction (Plume-traj). Ash (left) and SO2 (right) show seeds, time-colored trajectories, and arrival heights; black contours show threshold 0.1; red triangle marks Hayli Gubbi volcano.](Fig2.png)

---

## 1. Dependencies

Install these Python packages in your environment:

- `numpy`
- `netCDF4`
- `scipy`
- `matplotlib`
- `cartopy`

Example (conda):

```bash
conda install numpy netcdf4 scipy
```

or with `pip`:

```bash
pip install numpy netCDF4 scipy
```

---

## 2. Required Inputs

### 2.1 WRF output (`--wrfout`)

One or more WRF NetCDF files (ordered in time) with at least:

- `XLAT`, `XLONG` (2‑D lat/lon)
- `MAPFAC_MX`, `MAPFAC_MY`, `DX`, `DY` (for physical grid spacing)
- `U`, `V`, `W` (staggered 3‑D winds)
- `PH`, `PHB` (for geometric height)
- `Times` (WRF time stamps)

The script destaggers winds, computes cell-center heights and layer thickness (`z_center`, `dz`), and extracts the time axis. When multiple WRF files are provided, their time axes are concatenated (and the combined series is sorted by time).

### 2.2 SO₂ column file (`--column`)

A NetCDF file containing the SO₂ column field.

**Important:** The column data must be regridded onto the exact same horizontal grid as the `--wrfout` file. The script expects matching `(south_north, west_east)` dimensions.

- Dimensions: `(time, south_north, west_east)` or `(south_north, west_east)`
- Variable name: by default `SO2_COLUMN` (changeable via `--column-var`)

If the column field has a time dimension, the script uses the time index matching the chosen WRF start time (the closest WRF output to `--start-time`). If the column file is shorter, the last available column time is used instead.

---

## 3. Command-line Arguments

Run the script with:

```bash
python plume_backtraj.py --wrfout WRFOUT.nc --column SO2_COLUMN.nc --receptor-lat 45.0 --receptor-lon 10.0
```

Full argument list:

- `--wrfout` (required, one or more)  
  One or more WRF output files (ordered in time). Wildcards are supported.

- `--column` (required)  
  NetCDF file containing SO₂ column on the same WRF grid.

- `--column-var` (default: `SO2_COLUMN`)  
  Variable name for the SO₂ column in `--column`.

- `--column-coef` (default: `1.0`)  
  Scalar multiplier applied to the SO₂ column (e.g., convert mol m⁻² to Dobson Units).

- `--threshold` (default: `0.1`)  
  Column loading threshold used to define the SO₂ plume mask (same units as column field).

- `--n-columns` (default: `200`)  
  Number of random plume columns (grid cells above threshold) to sample.

- `--n-vert` (default: `30`)  
  Number of parcels per column, uniformly spaced between `z-min` and `z-max`.

- `--seed-bbox LON_MIN LAT_MIN LON_MAX LAT_MAX` (optional)  
  Restrict parcel seeding to a rectangular lon/lat box. If omitted, the entire plume mask above `--threshold` is used.

- `--z-min` (default: `2000.0` m)  
  Minimum initial parcel height.

- `--z-max` (default: `25000.0` m)  
  Maximum initial parcel height.

- `--receptor-lat`, `--receptor-lon` (required)  
  Latitude and longitude of the receptor (typically the volcano vent), in degrees.

- `--receptor-radius` (default: `10000.0` m)  
  Horizontal radius of the receptor cylinder (distance in meters around the receptor center).

- `--parcel-radius` (default: `0.0` m)  
  Horizontal radius assigned to each parcel when checking for contact with the receptor cylinder (parcel is considered arrived when its circle touches the receptor circle).

- `--receptor-min-h`, `--receptor-max-h` (defaults: `0.0`, `30000.0` m)  
  Vertical bounds of the receptor cylinder.

- `--integration-dt` (default: `15.0` s)  
  Sub-step (seconds) used for backward advection between two consecutive WRF output times. Leave at the native WRF time step for best fidelity.

- `--start-time` (optional, string)
  UTC start time for back-trajectories, e.g. `2021-04-10T15:00:00` or `2021-04-10_15:00:00`. If not provided, the last time step from the WRF file is used.

- `--aer-type` (optional)
  Aerosol type for gravitational settling (e.g., `sulf`, `ash1`, etc.). If provided, applies a size-dependent settling velocity to parcels during advection.

- `--arrival-bin-minutes` (default: `60.0`)  
  Width (minutes) of time bins used to accumulate parcel counts in the emission matrix.  
  Set to `0` to disable binning and skip emission output.

- `--output-txt` (default: `emission_time_height.txt`)  
  Path to the output time–height emission file (first line `time ...`, second line `height ...`, followed by rows of parcel counts).

- `--mass-output-txt` (optional)  
  When provided, writes a second TXT file containing the same headers as `--output-txt` but filled with mass-weighted totals per time–height bin (scientific notation, units inherited from the input column).

- `--output-figure` (default: `emission_time_height.png`)  
  Path to a PNG visualisation of the emission matrix.

- `--mass-figure` (optional)  
  Path to a PNG heatmap that mirrors the emission matrix but displays the accumulated parcel mass using the same binning.

- `--trajectory-figure` (default: `parcel_trajectories.png`)  
  Path for the PNG figure showing parcel trajectories (only parcels that reached the receptor are drawn).

- `--seeds-figure` (optional)
  Path to a PNG map showing the initial locations of all seeded parcels.

- `--seeds-vertical-figure` (optional)
  Path to a PNG plot showing the initial vertical distribution of all seeded parcels.

- `--trajectory-age` (optional)  
  When set, writes a Cartopy map (PNG) where parcel trajectories are coloured by their arrival age (hours). The figure mirrors the main trajectory map, includes the same coastline/border background, and uses 10 discrete intervals from the `gist_ncar` colormap with a horizontal legend.

- `--trajectory-emission-time-figure` (optional)  
  Writes a Cartopy map (PNG) where parcel trajectories are coloured by emission time (hours since `--emission-start`). Uses 10 discrete intervals from `gist_ncar` and labels the colorbar with the emission start time when provided.

- `--trajectory-arrival-height-figure` (optional)  
  Like the age map, but parcels are coloured by the altitude at which they intercepted the receptor cylinder (10 discrete bins, labelled in km).

- `--missed-trajectory-figure` (optional)  
  Stores a Cartopy map of the parcels that never intersected the receptor. Trajectories are plotted with the same initial-height colouring as the main trajectory panel, and final positions are marked with colored points.

- `--figure-dpi` (default: `200`)  
  Resolution (dots per inch) applied to every generated PNG figure.

- `--map-extent WEST SOUTH EAST NORTH` (optional)  
  Override the map extent for all Cartopy plots (west/south/east/north bounds). If omitted, the WRF domain bounds are used.

- `--state-pickle` (optional)  
  Path to a pickle file where the script will store all essential inputs and computed outputs (column field, trajectories, emission matrix, etc.). You can load this file later to regenerate figures without rerunning the back-trajectory calculations.

- `--hourly-figures` (flag)  
  When provided, saves hourly parcel-location maps during the backward advection loop (`parcel_positions_hour_XXX.png`).

- `--colorbar-label` (default: `Parcel count`)  
  Label applied to the column-field colorbars (initial parcel map, hourly snapshots) and to the emission-matrix colorbar. The mass-weighted emission figure always uses the label “Parcel mass”.

- `--emission-start` (optional, string)  
  UTC start time of emission/eruption, e.g.:
  - `2021-04-10T15:00:00` or
  - `2021-04-10_15:00:00`  
  Backward advection stops at this time, and all arrival times are measured relative to it.

- `--emission-end` (optional, string)  
  UTC end time of the eruption/emission. When supplied together with `--emission-start`, the emission matrix uses fixed time bins spanning this window and discards arrivals outside it.

- `--efolding-days` (optional, float)
  e-folding lifetime for mass decay, in days. If set, parcel mass is increased backward in time to account for chemical decay, affecting the mass-weighted emission matrix.

---

## 4. What the Script Does

### 4.1 Read WRF geometry and winds

`read_wrf_geometry_and_winds()`:

- Destaggers `U`, `V`, `W` to mass points.
- Computes:
  - `dx_m`, `dy_m` (physical grid spacing).
  - `area` (cell area).
  - `z_center` (cell center heights).
  - `dz` (layer thickness).
  - `times` (converted from `Times` to `numpy.datetime64`).

### 4.2 Sample SO₂ column and create parcels

`generate_parcels_from_column_wrf()`:

- Builds a plume mask where `column2d >= threshold`.
- Optionally intersects that mask with `--seed-bbox` to limit where parcels are initialized.
- Randomly selects `n-columns` plume grid cells.
- For each selected cell:
  - Draws `n-vert` random heights uniformly between `z-min` and `z-max` (sorted from low to high).
  - Interpolates those heights onto the WRF `k` levels (using the vertical profile at the chosen start time) to get fractional layer indices.
  - Keeps track of the actual initial height (`z_init`) so that diagnostics can colour trajectories by their launch altitude.

Outputs an initial parcel set with:

- `j`, `i`: fractional row/column indices.
- `k`: fractional vertical indices.
- `z_init`: the original launch height (meters).

### 4.3 Backward advection

`advect_parcels_backward_wrf()`:
- Slices the WRF data to start from the time index closest to `--start-time`.
- Converts WRF time axis to seconds relative to:
  - `emission_start_time` (if provided), or
  - first WRF time (`times[0]` otherwise).
- Loops backward from `start_time_index` to earlier times:
  - Interpolates `u`, `v`, `w`, `dz` at each parcel position, *linearly in time between WRF outputs* and spatially within each snapshot.
  - Integrates parcels backward using a 4th-order Runge-Kutta (RK4) scheme in index space:
    - `di = -(u * dt) / dx`
    - `dj = -(v * dt) / dy`
    - `dk = -(w * dt) / dz`
  - Marks parcels that leave the domain as inactive.
  - Checks whether parcels enter the receptor cylinder:
    - Horizontal: grid-index distance converted to meters using `dx`, `dy`, compared to `receptor_radius`.
    - Vertical: height between `z-cyl-min` and `z-cyl-max`.
  - When a parcel hits the cylinder:
    - Records arrival time (seconds since `emission_start_time` or model start).
    - Records arrival height.
    - Marks parcel as arrived and inactive.
    - If `--emission-start`/`--emission-end` are provided, arrivals outside the emission window are ignored.
- Stops integration when:
  - All parcels are inactive, or
  - `emission_start_time` is reached (`t_sec <= 0`).

Returns a dictionary with:

- `j`, `i`, `k`
- `arrived` (bool)
- `arrival_time` (seconds since reference time)
- `arrival_z` (height [m])

The main script then filters to keep only parcels with `arrived == True`.

### 4.4 Build time–height emission series

In `main()`:

1. Converts `arrival_time` to absolute datetimes (if `times` are datetime64).
2. Defines vertical bins:
   - Uses `z_center[start_time, :, j_rec, i_rec]` (vertical profile over receptor grid cell).
   - Selects levels within `[z-cyl-min, z-cyl-max]`.
3. Maps each arriving parcel to a vertical bin based on its physical arrival height. Every parcel contributes:
   - a **count** of `1`, and
   - a **mass weight** inherited from its source column cell (`column value × cell area / n_vert`).
4. Mass correction (optional):
   - If `--efolding-days` is set, the mass of each parcel is increased backward in time based on its age to account for chemical decay.
   - `mass_corrected = mass_at_receptor * exp(age_seconds / e-folding_seconds)`

5. Time discretisation:
   - Uses bins of width `arrival-bin-minutes` (in seconds).
   - Computes integer time bin indices for each arrival (or enforces fixed bins if both `--emission-start` and `--emission-end` are supplied).
6. Accumulates both **counts** and **mass** into 2‑D arrays:
   - `emission[height_bin, time_bin]` stores parcel counts.
   - `mass_emission[height_bin, time_bin]` stores the summed parcel mass for that bin.
7. Writes the TXT file in the format:
   1. `time <labels...>`
   2. `height <z-centers...>`
   3. One row per height (from top to bottom) containing parcel counts per time.
   If `--mass-output-txt` is set, a second file with the same headers stores `mass_emission` using scientific notation.
8. Saves a PNG (path from `--output-figure`) that mirrors the count matrix; if `--mass-figure` is supplied, a second PNG renders the mass-weighted matrix with the same binning.

Cells with no arrivals remain `0.0`. Diagnostic messages print both the total parcel count and the summed mass contained in the matrices.

### 4.5 Diagnostics and figures

- **Initial parcel map** (`parcel_locations_tXXX.png`): shows the column field (scaled by `--column-coef`), the threshold contour (white dash-dot), receptor location/circle, and all initial parcels. The colorbar label for this and any other column-field plot comes from `--colorbar-label`.
- **Hourly snapshots** (`--hourly-figures`): optional sequence of parcel maps rendered after each back-advection step (indices count backward so the numbering matches the WRF time index). This is controlled by the `--hourly-figures` flag.
- **Trajectory figure** (`--trajectory-figure`): plots only those parcels that reached the receptor. Their paths and starting markers are coloured by launch height using 10 evenly spaced intervals between `--z-min` and `--z-max`, with a horizontal rainbow legend for the initial-height bins. Only the start points are highlighted (no terminal markers), emphasizing where each parcel originated. The rendering is heavily optimized using `matplotlib.collections.LineCollection` to draw all trajectories at once, making it efficient even for thousands of parcels.
- **Parcel-age figure** (`--trajectory-age`, optional): second Cartopy panel where the same trajectories are coloured by their arrival age. Ten evenly spaced time bins drive a horizontal `gist_ncar` legend (labelled in hours), giving a quick view of how long each parcel needed to reach the receptor.
- **Emission-time figure** (`--trajectory-emission-time-figure`, optional): Cartopy panel where trajectories are coloured by emission time (hours since `--emission-start`). The colorbar labels the emission reference time when supplied.
- **Arrival-height figure** (`--trajectory-arrival-height-figure`, optional): similar map but colour-codes parcels by the height at which they hit the receptor cylinder, using 10 evenly spaced bins (km) and a horizontal legend.
- **Emission matrix figure** (`--output-figure`): heatmap of parcel counts versus time and altitude with discrete bins, tick labels formatted as HH:MM when actual datetimes are available, and an annotation with the total number of parcels reaching the receptor.
- **Mass-weighted outputs** (`--mass-output-txt`, `--mass-figure`, optional): mirror the count-based TXT/PNG products but display the accumulated parcel mass in each bin, preserving the same axes, tick formatting, and labels (the figure uses the “Parcel mass” colorbar title).
- **Pickled state** (`--state-pickle`, optional): saves a dictionary containing the SO₂ field, trajectories, emission matrix, and metadata so you can replot or post-process without repeating the (expensive) back-trajectory integration.

---

## 5. Re-plotting from a saved state

When `--state-pickle path/to/run_state.pkl` is supplied, the script stores:

- The scaled SO₂ column field and plotting metadata (threshold, colour bar label).
- All parcel trajectories (positions, activity flags, initial heights, arrival ages/heights).
- The emission matrix (both counts and mass) with its time/height edges.
- The original parcel seeds so the initial-location map can be recreated.

To regenerate all plots without re-running the back-trajectory solver, run:

```bash
python plot_backtraj.py run_state.pkl
```

The helper script restores DPI/labels from the pickle and produces the full suite of figures (`trajectories_replot.png`, `trajectory_ages_replot.png`, `trajectory_emission_time_replot.png`, `trajectory_arrival_height_replot.png`, `missed_trajectories_replot.png`, `parcel_locations_replot.png` if seeds were stored, and `emission_matrix_replot.png`). It also writes `emission_matrix_replot.txt` and, if mass data exist, `mass_matrix_replot.png` and `mass_matrix_replot.txt`.

---

## 6. Example Command

Minimal back-trajectory run:

```bash
python plume_backtraj.py \
  --wrfout wrfout_d01_2021-04-10_18:00:00.nc \
  --column so2_column_on_wrf_grid.nc \
  --receptor-lat 45.0 \
  --receptor-lon 10.0
```

Example usage for a volcano at 45°N, 10°E, 10 km receptor radius, 1‑hour time bins:

```bash
python plume_backtraj.py \
  --wrfout wrfout_d01_2021-04-10_18:00:00.nc \
  --start-time '2021-04-10T18:00:00' \
  --column so2_column_on_wrf_grid.nc \
  --column-var SO2_COLUMN \
  --column-coef 2242.95 --threshold 0.1 --colorbar-label 'SO2, DU' \
  --n-columns 3000 \
  --n-vert 30 \
  --z-min 2000 \
  --z-max 25000 \
  --receptor-lat 45.0 \
  --receptor-lon 10.0 \
  --receptor-radius 10000 \
  --receptor-min-h 1000 \
  --receptor-max-h 30000 \
  --arrival-bin-minutes 60 \
  --emission-start 2021-04-10T15:00:00 \
  --emission-end '2021-04-10T21:00:00' \
  --efolding-days 35 \
  --output-txt emission_time_height.txt \
  --mass-output-txt emission_time_height_mass.txt \
  --mass-figure emission_time_height_mass.png
```

You can copy `plume_backtraj.py` and this `plume_backtraj.md` file to another computer, install the required Python packages, and reuse the same command-line interface there.

---

## 7. Notes on Input Data

- The SO2 column file must already be on the WRF grid (`south_north`, `west_east`).
- If the column file has a time dimension, the script uses the time index matching the chosen WRF start time.
- Ensure `--z-min/--z-max` are within the WRF model top; the script will error if `z-max` exceeds it.

---

## 8. Troubleshooting

- **No parcels reach the receptor**: check receptor location/radius and the plume threshold.
- **Cartopy errors**: install `cartopy` with conda if pip wheels are not available for your system.
