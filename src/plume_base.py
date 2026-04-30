"""Shared backend contract and utilities for Plume-Traj."""

from abc import ABC, abstractmethod
import pickle

import numpy as np


class PlumeBackend(ABC):
    """Common interface implemented by the WRF and MPAS backends."""

    target = None

    @staticmethod
    def diag(msg: str) -> None:
        print(f"[diag] {msg}")

    @staticmethod
    def format_time(val) -> str:
        arr = np.asarray(val)
        if arr.dtype.kind == "M":
            return str(val)
        return f"{val}"

    @staticmethod
    def save_state_pickle(path, payload) -> None:
        with open(path, "wb") as fh:
            pickle.dump(payload, fh)
        PlumeBackend.diag(f"Saved processing state to '{path}'.")

    @abstractmethod
    def run_backtraj(self, args) -> None:
        """Run backward trajectories for this backend."""

    @abstractmethod
    def run_forwtraj(self, args) -> None:
        """Run forward trajectories for this backend."""

    @abstractmethod
    def plot_backtraj_state(self, args) -> None:
        """Replot a saved backward trajectory state."""

    @abstractmethod
    def plot_forwtraj_state(self, args) -> None:
        """Replot a saved forward trajectory state."""


def backend_from_target(target):
    if target == "mpas":
        from .plume_mpas import MPASBackend

        return MPASBackend()
    if target == "wrf":
        from .plume_wrf import WRFBackend

        return WRFBackend()
    raise ValueError(f"Unsupported backend target: {target}")


def backend_from_state(state):
    args = state.get("args", {})
    metadata = state.get("metadata", {})
    target = args.get("target") or metadata.get("target") or "wrf"
    return backend_from_target(target)
