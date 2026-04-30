#!/usr/bin/env python
"""CLI entry point for replotting backward Plume-Traj states.

Example:


python plot_backtraj.py so2_wrf_run/run_so2.pkl \
    --hourly-figures --hourly-output-dir hourly_replot \
    --map-extent 30 5 65 30 \
    --figure-dpi 300
    
"""

import pickle

from src.plume_base import backend_from_state
from src.plot_backtraj_wrf import parse_args


def main(args=None):
    if args is None:
        args = parse_args()
    with open(args.pickle_file, "rb") as fh:
        state = pickle.load(fh)
    backend = backend_from_state(state)
    backend.plot_backtraj_state(args)


if __name__ == "__main__":
    main()
