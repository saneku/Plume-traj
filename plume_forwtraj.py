#!/usr/bin/env python
"""CLI entry point for forward Plume-Traj runs.

Example WRF run:

INPUT='/scratch/ukhova/SandBox/WRF/run_hayligubbi/wrfout_d01_2025-*'
INPUT='/scratch/masabas/bandwidth/cylc-run/workforecastmain/run1/work/20260526T1200Z/wrfChem/wrfout_d01_2026-*'

python plume_forwtraj.py \
  --target wrf \
  --input "$INPUT" \
  --start-time 2026-05-28T00:00:00 \
  --end-time 2026-05-28T05:00:00 \
  --aer-type sulf \
  --integration-dt 15 \
  --source-lat 13.51 \
  --source-lon 40.71 \
  --z-min 1000 \
  --z-max 23000 \
  --n-vert 30 \
  --age-figure wrf_plume_age_colored.png \
  --deposition-figure wrf_deposited_by_hour.png \
  --initial-height-figure wrf_plume_initial_height_colored.png \
  --seeds-vertical-figure wrf_parcel_initial_vertical_distribution.png \
--hourly-output-dir wrf_hourly \
--state-pickle wrf_forward_run.pkl \
--map-extent 30 5 65 30 \
--figure-dpi 300

Example WRF run with time-intensity releases:

python plume_forwtraj.py \
  --target wrf \
  --input "$INPUT" \
  --start-time 2026-05-28T00:00:00 \
  --end-time 2026-05-28T05:00:00 \
  --source-lat 13.51 \
  --source-lon 40.71 \
  --z-min 100 \
  --z-max 1500 \
  --n-vert 15 \
  --emission-timeseries emission_timeseries_example_1000_halflife6h.txt \
  --state-pickle wrf_forward_run.pkl
	  

Example MPAS-Chem run:

HIST='/scratch/ukhova/MPAS/MPAS-Model/regional/history*.nc'

python plume_forwtraj.py \
    --target mpas \
    --input "$HIST" \
    --start-time 2026-05-28T00:00:00 \
    --end-time 2026-05-28T05:00:00 \
    --aer-type sulf \
    --source-lat 13.51 \
    --source-lon 40.71 \
    --z-min 1000 \
    --z-max 23000 \
    --n-vert 30 \
    --age-figure mpas_plume_age_colored.png \
    --deposition-figure mpas_deposited_by_hour.png \
    --initial-height-figure mpas_plume_initial_height_colored.png \
    --seeds-vertical-figure mpas_parcel_initial_vertical_distribution.png \
    --hourly-output-dir mpas_hourly \
    --state-pickle mpas_forward_run.pkl \
    --map-extent 30 5 65 30 \
    --figure-dpi 300
"""

from src.plume_base import backend_from_target
from src.plume_wrf import parse_forwtraj_args as parse_args


def main(args=None):
    if args is None:
        args = parse_args()
    backend = backend_from_target(args.target)
    backend.run_forwtraj(args)


if __name__ == "__main__":
    main()
