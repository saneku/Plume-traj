"""Shared backend contract and utilities for Plume-Traj."""

from abc import ABC, abstractmethod
import pickle
from pathlib import Path

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


def parse_emission_matrix_file(path):
    """
    Parse a time-height emission matrix text file.

    Expected format:
      time_offset_h <dt1> <dt2> ...   # recommended; offsets in hours from --start-time
      or time_offset_s <dt1> <dt2> ...# offsets in seconds from --start-time
      or time <t1> <t2> ...           # legacy: ISO or numeric
      height <h1> <h2> ...
      <row for highest height>
      ...
      <row for lowest height>
    """
    text = Path(path).read_text(encoding="utf-8")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if len(lines) < 3:
        raise ValueError("Emission matrix file must contain time, height, and at least one data row.")

    time_fields = lines[0].split()
    time_key = time_fields[0].lower()
    if time_key not in ("time", "time_offset_s", "time_offset_h"):
        raise ValueError(
            "Emission matrix first non-empty line must start with "
            "'time_offset_h ', 'time_offset_s ', or 'time '."
        )
    if not lines[1].lower().startswith("height "):
        raise ValueError("Emission matrix second non-empty line must start with 'height '.")

    time_tokens = time_fields[1:]
    if not time_tokens:
        raise ValueError("Emission matrix time line has no values.")
    height_tokens = lines[1].split()[1:]
    if not height_tokens:
        raise ValueError("Emission matrix 'height' line has no values.")

    n_t = len(time_tokens)
    n_h = len(height_tokens)

    if time_key == "time_offset_h":
        time_kind = "offset_s"
        time_vals = np.array([float(tok) * 3600.0 for tok in time_tokens], dtype=float)
    elif time_key == "time_offset_s":
        time_kind = "offset_s"
        time_vals = np.array([float(tok) for tok in time_tokens], dtype=float)
    else:
        # Legacy format: ISO datetime preferred; numeric fallback.
        time_kind = "datetime"
        try:
            time_vals = np.array([np.datetime64(tok) for tok in time_tokens], dtype="datetime64[s]")
        except Exception:
            time_kind = "numeric_legacy"
            time_vals = np.array([float(tok) for tok in time_tokens], dtype=float)

    heights = np.array([float(tok) for tok in height_tokens], dtype=float)
    rows = lines[2:]
    if len(rows) != n_h:
        raise ValueError(
            f"Emission matrix row count mismatch: got {len(rows)} data rows, expected {n_h} from height line."
        )

    # Store counts indexed by ascending heights (low->high), while file rows are high->low.
    counts = np.zeros((n_h, n_t), dtype=float)
    for row_idx, line in enumerate(rows):
        vals = [float(v) for v in line.split()]
        if len(vals) != n_t:
            raise ValueError(
                f"Emission matrix row {row_idx + 1} has {len(vals)} columns; expected {n_t} from time line."
            )
        h_idx = n_h - 1 - row_idx
        counts[h_idx, :] = np.asarray(vals, dtype=float)

    return dict(
        path=str(path),
        raw_text=text,
        time_tokens=time_tokens,
        time_key=time_key,
        time_kind=time_kind,
        time_values=time_vals,
        height_values=heights,
        counts=counts,  # shape: [height(low->high), time]
    )


def emission_matrix_to_schedule(matrix, start_time_utc, z_override=None):
    """
    Convert parsed matrix into release schedule arrays.
    Returns release times (datetime64[s]), heights [m], and counts matrix.
    """
    tvals = matrix["time_values"]
    if matrix["time_kind"] == "offset_s":
        offsets_sec = np.asarray(tvals, dtype=float)
    elif matrix["time_kind"] == "datetime":
        t0 = tvals[0].astype("datetime64[s]")
        offsets_sec = ((tvals.astype("datetime64[s]") - t0) / np.timedelta64(1, "s")).astype(float)
    else:
        # Legacy numeric "time" line support: normalize by first value.
        offsets_sec = np.asarray(tvals, dtype=float) - float(np.asarray(tvals, dtype=float)[0])

    release_times = start_time_utc.astype("datetime64[s]") + np.rint(offsets_sec).astype(np.int64).astype("timedelta64[s]")

    if z_override is not None:
        z_min = float(z_override["z_min"])
        z_max = float(z_override["z_max"])
        n_vert = int(z_override["n_vert"])
        if n_vert <= 0:
            raise ValueError("n_vert must be positive in emission-matrix override mode.")
        if n_vert == 1:
            heights_m = np.array([0.5 * (z_min + z_max)], dtype=float)
        else:
            heights_m = np.linspace(z_min, z_max, n_vert, dtype=float)
        counts = np.ones((heights_m.size, release_times.size), dtype=float)
        mode = "override_ones"
        unit_hint = "m"
    else:
        heights_raw = np.asarray(matrix["height_values"], dtype=float)
        # Heuristic: legacy MPAS backtraj writes km-scale heights.
        if np.nanmax(np.abs(heights_raw)) <= 30.0:
            heights_m = heights_raw * 1000.0
            unit_hint = "km_to_m"
        else:
            heights_m = heights_raw
            unit_hint = "m"
        counts = np.asarray(matrix["counts"], dtype=float)
        mode = "matrix_native"

    counts_int = np.rint(np.clip(counts, 0.0, None)).astype(int)
    return dict(
        release_times_utc=release_times,
        heights_m=np.asarray(heights_m, dtype=float),
        counts=counts_int,  # [height(low->high), time]
        mode=mode,
        height_unit_hint=unit_hint,
    )
