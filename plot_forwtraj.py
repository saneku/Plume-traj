#!/usr/bin/env python
"""CLI entry point for replotting forward Plume-Traj states.

Example:

python plot_forwtraj.py forward_run.pkl \
  --initial-height-figure my_initial_height.png \
  --age-figure my_age.png \
  --hourly-output-dir hourly_replot \
  --seeds-vertical-figure seeds_vertical.png \
  --deposition-figure deposited_by_hour_replot.png \
  --map-extent 30 5 65 30 \
  --figure-dpi 300
"""

import pickle

from src.plume_base import backend_from_state
from src.plot_forwtraj_wrf import parse_args


def main(args=None):
    if args is None:
        args = parse_args()
    with open(args.pickle_file, "rb") as fh:
        state = pickle.load(fh)
    backend = backend_from_state(state)
    backend.plot_forwtraj_state(args)


if __name__ == "__main__":
    main()
