#!/usr/bin/env python
"""CLI entry point for backward Plume-Traj runs.

Example WRF SO2 run:

INPUT='/scratch/ukhova/SandBox/WRF/run_hayligubbi/wrfout_d01_2025*'
COLUMN='/lustre2/project/k10022/ukhova/Volcano/Hayli_Gubbi/operRSmerged_SO2_/sulfurdioxide_total_vertical_column_15km/4km/merged_sulfurdioxide_total_vertical_column_15km_2025-Nov-24.nc'
COLUMN_VAR='sulfurdioxide_total_vertical_column_15km'

outdir="./so2_wrf_run"
mkdir -p "$outdir"
mkdir -p "$outdir/hourly"
python plume_backtraj.py \
  --target wrf \
  --input "$INPUT" \
  --start-time '2025-11-24T12:00:00' \
  --column "$COLUMN" --integration-dt 15 \
  --efolding-days 35 \
  --column-var "$COLUMN_VAR" --column-coef 2242.95 --threshold 0.1 --colorbar-label 'SO2, DU' \
  --n-columns 100 --n-vert 30 --parcel-radius 10000 --z-min 1000 --z-max 23000 \
  --emission-start '2025-11-23T08:30:00' --emission-end '2025-11-24T10:00:00' \
  --receptor-lat 13.51 --receptor-lon 40.71 \
  --receptor-radius 20000 --receptor-min-h 1000 --receptor-max-h 30000 \
  --arrival-bin-minutes 30 \
  --output-txt "$outdir/so2_emission_time_height.txt" \
  --output-figure "$outdir/so2_emission_time_height.png" \
  --trajectory-figure "$outdir/so2_trajectories.png" \
  --trajectory-age "$outdir/so2_trajectory_ages.png" \
  --trajectory-emission-time-figure "$outdir/so2_trajectory_emission_time.png" \
  --seeds-figure "$outdir/parcel_initial_locations.png" \
  --mass-figure "$outdir/mass_matrix.png" \
  --mass-output-txt "$outdir/mass_emission_time_height.txt" \
  --trajectory-arrival-height-figure "$outdir/so2_trajectory_arrival_heights.png" \
  --missed-trajectory-figure "$outdir/so2_missed_trajectories.png" \
  --figure-dpi 300 --state-pickle "$outdir/run_so2.pkl" \
  --map-extent 30 5 65 30 \
  --hourly-output-dir "$outdir/hourly" \
  --seeds-vertical-figure "$outdir/parcel_initial_vertical_distribution.png"

Example MPAS SO2 run:

HIST='/scratch/ukhova/MPAS/MPAS-Model/regional/history*.nc'
COLUMN='/scratch/ukhova/MPAS/Plume-traj/so2_regridded_to_mpas.nc'
COLUMN_VAR='sulfurdioxide_total_vertical_column_15km'

outdir="./so2_mpas_run"
mkdir -p "$outdir"
mkdir -p "$outdir/hourly"
python plume_backtraj.py \
  --target mpas \
  --input "$HIST" \
  --start-time '2025-11-24T12:00:00' \
  --column "$COLUMN" --integration-dt 15 \
  --efolding-days 35 \
  --column-var "$COLUMN_VAR" --column-coef 2242.95 --threshold 0.1 --colorbar-label 'SO2, DU' \
  --n-columns 100 --n-vert 30 --parcel-radius 10000 --z-min 1000 --z-max 23000 \
  --emission-start '2025-11-23T08:30:00' --emission-end '2025-11-24T10:00:00' \
  --receptor-lat 13.51 --receptor-lon 40.71 \
  --receptor-radius 20000 --receptor-min-h 1000 --receptor-max-h 30000 \
  --arrival-bin-minutes 30 \
  --output-txt "$outdir/so2_emission_time_height.txt" \
  --output-figure "$outdir/so2_emission_time_height.png" \
  --trajectory-figure "$outdir/so2_trajectories.png" \
  --trajectory-age "$outdir/so2_trajectory_ages.png" \
  --trajectory-emission-time-figure "$outdir/so2_trajectory_emission_time.png" \
  --seeds-figure "$outdir/parcel_initial_locations.png" \
  --mass-figure "$outdir/mass_matrix.png" \
  --mass-output-txt "$outdir/mass_emission_time_height.txt" \
  --trajectory-arrival-height-figure "$outdir/so2_trajectory_arrival_heights.png" \
  --missed-trajectory-figure "$outdir/so2_missed_trajectories.png" \
  --figure-dpi 300 --state-pickle "$outdir/run_so2.pkl" \
  --map-extent 30 5 65 30 \
  --hourly-output-dir "$outdir/hourly" \
  --seeds-vertical-figure "$outdir/parcel_initial_vertical_distribution.png"




Example aerosol loop:

INPUT='/scratch/ukhova/SandBox/WRF/run_hayligubbi/wrfout_d01_2025*'
COLUMN='/lustre2/project/k10022/ukhova/Volcano/Hayli_Gubbi/operRSmerged_SO2_/AOD_AI_HEIGHT/4km/merged_aerosol_index_354_388_2025-NOV-24.nc'
COLUMN_VAR='aerosol_index_354_388'

for aer in sulf ash10 ash9 ash8 ash7 ash6; do
  outdir="./${aer}_run"
  mkdir -p "$outdir"
  mkdir -p "$outdir/hourly"
  python plume_backtraj.py \
    --target wrf \
    --input "$INPUT" \
    --aer-type "$aer" \
    --start-time '2025-11-24T10:00:00' \
    --column "$COLUMN" --integration-dt 15 \
    --column-var "$COLUMN_VAR" --column-coef 1 --threshold 0.10 --colorbar-label 'Aerosol Index' \
    --n-columns 50 --n-vert 30 --parcel-radius 10000 --z-min 1000 --z-max 23000 \
    --emission-start '2025-11-23T08:30:00' --emission-end '2025-11-24T10:00:00' \
    --receptor-lat 13.51 --receptor-lon 40.71 \
    --receptor-radius 10000 --receptor-min-h 1000 --receptor-max-h 30000 \
    --arrival-bin-minutes 30 \
    --output-txt "$outdir/emission_time_height.txt" \
    --output-figure "$outdir/emission_time_height.png" \
    --trajectory-figure "$outdir/trajectories.png" \
    --trajectory-age "$outdir/trajectory_ages.png" \
    --trajectory-emission-time-figure "$outdir/trajectory_emission_time.png" \
    --seeds-figure "$outdir/parcel_initial_locations.png" \
    --mass-figure "$outdir/mass_matrix.png" \
    --mass-output-txt "$outdir/mass_emission_time_height.txt" \
    --trajectory-arrival-height-figure "$outdir/trajectory_arrival_heights.png" \
    --missed-trajectory-figure "$outdir/missed_trajectories.png" \
    --figure-dpi 300 --state-pickle "$outdir/run_state.pkl" \
    --hourly-output-dir "$outdir/hourly" \
    --map-extent 30 5 65 30
done

1 mol/m2 is approximately 2243 DU.
"""

from src.plume_base import backend_from_target
from src.plume_wrf import parse_backtraj_args as parse_args


def main(args=None):
    if args is None:
        args = parse_args()
    backend = backend_from_target(args.target)
    backend.run_backtraj(args)


if __name__ == "__main__":
    main()
