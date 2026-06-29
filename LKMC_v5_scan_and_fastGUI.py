"""
LKMC_v5_gui_fast.py
====================
GUI-accelerated version of the 2D lattice KMC electrodeposition simulator.

The simulation *kernel* (Fenwick tree, energy computation, event-rate refresh,
bonding relaxation, and the KMC stepping loop) is compiled to native machine
code via Numba JIT the first time it runs.  The GUI, file I/O, and all output
formats are identical to v4b; only execution speed changes.

Speed-up strategy
-----------------
* Every call to ``SimulationGUI.run_batch`` previously called Python
  ``execute_step()`` 200 times (one Python function call per KMC event).
* Now ``run_batch`` calls a single compiled function ``_nb_run_n_steps`` that
  runs all N steps inside native code with no Python overhead.
* Fenwick-tree, local energy, rate-refresh, and BFS bond-relaxation routines
  are all compiled with ``@njit(cache=True)``.

First-run note
--------------
Numba compiles the simulation kernel on the first call (~5–20 s).  Subsequent
runs load a cached binary and start instantly.  A status-bar message warns the
user.

Fallback
--------
If ``numba`` is not installed the code silently falls back to pure-Python mode
(identical to v4b).  Install with:  pip install numba
"""

from __future__ import annotations

import csv
import itertools
import math
import time
import tkinter as tk
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from tkinter import messagebox, ttk
from typing import Deque, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure

# ---------------------------------------------------------------------------
# Optional Numba import
# ---------------------------------------------------------------------------
try:
    from numba import njit          # noqa: F401
    import numba                    # noqa: F401
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

    def njit(fn=None, **kwargs):    # type: ignore[misc]
        """No-op decorator used when numba is absent."""
        if callable(fn):
            return fn
        def _dec(f):
            return f
        return _dec

# ---------------------------------------------------------------------------
# Lattice state codes  (same as v4b)
# ---------------------------------------------------------------------------
EMPTY      = 0
FREE       = 1
DEPOSITED  = 2
SUBSTRATE  = 3
PASSIVATED = 4

Coord = Tuple[int, int]

# ===========================================================================
# Numba-compiled simulation kernel
# ===========================================================================
# All functions below are module-level so that numba can JIT them.
# They work exclusively on numpy arrays and Python scalars — no class state.

# ---------------------------------------------------------------------------
# Fenwick tree helpers
# ---------------------------------------------------------------------------

@njit(cache=True)
def _nb_fen_update(tree: np.ndarray, size: int, idx: int, delta: float) -> None:
    i = idx + 1
    while i <= size:
        tree[i] += delta
        i += i & (-i)


@njit(cache=True)
def _nb_fen_total(tree: np.ndarray, size: int) -> float:
    i = size
    s = 0.0
    while i > 0:
        s += tree[i]
        i -= i & (-i)
    return s


@njit(cache=True)
def _nb_fen_find(tree: np.ndarray, size: int, target: float) -> int:
    idx = 0
    bit = 1
    while bit < size:
        bit <<= 1
    bit >>= 1
    while bit > 0:
        nxt = idx + bit
        if nxt <= size and tree[nxt] < target:
            target -= tree[nxt]
            idx = nxt
        bit >>= 1
    return idx


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

@njit(cache=True)
def _nb_calc_local_energy(
    lattice: np.ndarray,
    x: int, y: int,
    Ny: int, Nx: int,
    atom_type: int,
    energy_lookup: np.ndarray,
    periodic_x: bool,
) -> float:
    """Sum pair interaction energies for a hypothetical atom of `atom_type` at (x,y)."""
    e = 0.0
    DX = (1, -1, 0, 0)
    DY = (0,  0, 1, -1)
    for k in range(4):
        xx = x + DX[k]
        yy = y + DY[k]
        if periodic_x:
            xx = xx % Nx
        else:
            if xx < 0 or xx >= Nx:
                continue
        if yy < 0 or yy >= Ny:
            continue
        nbr = lattice[yy, xx]
        e += energy_lookup[atom_type, nbr]
    return e


@njit(cache=True)
def _nb_has_empty_neighbor(
    lattice: np.ndarray,
    x: int, y: int,
    Ny: int, Nx: int,
    periodic_x: bool,
) -> bool:
    DX = (1, -1, 0, 0)
    DY = (0,  0, 1, -1)
    for k in range(4):
        xx = x + DX[k]
        yy = y + DY[k]
        if periodic_x:
            xx = xx % Nx
        else:
            if xx < 0 or xx >= Nx:
                continue
        if yy < 0 or yy >= Ny:
            continue
        if lattice[yy, xx] == EMPTY:
            return True
    return False


@njit(cache=True)
def _nb_desired_mobile_state(
    lattice: np.ndarray,
    x: int, y: int,
    Ny: int, Nx: int,
    periodic_x: bool,
) -> int:
    """Return FREE(1) or DEPOSITED(2) based on neighbors, ignoring current site."""
    DX = (1, -1, 0, 0)
    DY = (0,  0, 1, -1)
    for k in range(4):
        xx = x + DX[k]
        yy = y + DY[k]
        if periodic_x:
            xx = xx % Nx
        else:
            if xx < 0 or xx >= Nx:
                continue
        if yy < 0 or yy >= Ny:
            continue
        nbr = lattice[yy, xx]
        if nbr == DEPOSITED or nbr == SUBSTRATE:
            return DEPOSITED
    return FREE


# ---------------------------------------------------------------------------
# Event rate computation
# ---------------------------------------------------------------------------

@njit(cache=True)
def _nb_get_event_rate(
    lattice: np.ndarray,
    kind: int,           # 0=drop, 1=hop, 2=passivate
    src_x: int, src_y: int,
    dst_x_raw: int, dst_y_raw: int,
    Nx: int, Ny: int, periodic_x: bool,
    energy_lookup: np.ndarray,
    d0: float, nu_f: float, nu_d: float, nu_p: float,
    kB: float, T: float,
) -> float:
    if kind == 0:          # drop
        if lattice[dst_y_raw, dst_x_raw] == EMPTY:
            return d0
        return 0.0

    if kind == 2:          # passivate
        if lattice[src_y, src_x] != DEPOSITED:
            return 0.0
        if _nb_has_empty_neighbor(lattice, src_x, src_y, Ny, Nx, periodic_x):
            return nu_p
        return 0.0

    # hop (kind == 1)
    atom_type = lattice[src_y, src_x]
    if atom_type != FREE and atom_type != DEPOSITED:
        return 0.0

    # Resolve destination x
    if periodic_x:
        dst_x = dst_x_raw % Nx
    else:
        if dst_x_raw < 0 or dst_x_raw >= Nx:
            return 0.0
        dst_x = dst_x_raw

    dst_y = dst_y_raw
    if dst_y < 0 or dst_y >= Ny:
        return 0.0
    if lattice[dst_y, dst_x] != EMPTY:
        return 0.0

    nu = nu_f if atom_type == FREE else nu_d

    e_init = _nb_calc_local_energy(lattice, src_x, src_y, Ny, Nx, atom_type, energy_lookup, periodic_x)

    # Temporarily vacate source to compute final energy correctly
    lattice[src_y, src_x] = EMPTY
    final_type = _nb_desired_mobile_state(lattice, dst_x, dst_y, Ny, Nx, periodic_x)
    e_final    = _nb_calc_local_energy(lattice, dst_x, dst_y, Ny, Nx, final_type, energy_lookup, periodic_x)
    lattice[src_y, src_x] = atom_type  # restore

    dE = e_final - e_init
    return nu * math.exp(-dE / (2.0 * kB * T))


# ---------------------------------------------------------------------------
# Single-index rate update
# ---------------------------------------------------------------------------

@njit(cache=True)
def _nb_update_rate_at_idx(
    idx: int,
    lattice: np.ndarray,
    ftree: np.ndarray,
    event_rates: np.ndarray,
    ev_kind: np.ndarray,
    ev_src_x: np.ndarray, ev_src_y: np.ndarray,
    ev_dst_x: np.ndarray, ev_dst_y: np.ndarray,
    max_events: int,
    Nx: int, Ny: int, periodic_x: bool,
    energy_lookup: np.ndarray,
    d0: float, nu_f: float, nu_d: float, nu_p: float,
    kB: float, T: float,
) -> None:
    new_rate = _nb_get_event_rate(
        lattice,
        ev_kind[idx],
        ev_src_x[idx], ev_src_y[idx],
        ev_dst_x[idx], ev_dst_y[idx],
        Nx, Ny, periodic_x,
        energy_lookup, d0, nu_f, nu_d, nu_p, kB, T,
    )
    delta = new_rate - event_rates[idx]
    if abs(delta) > 1.0e-18:
        event_rates[idx] = new_rate
        _nb_fen_update(ftree, max_events, idx, delta)


# ---------------------------------------------------------------------------
# Local rate refresh  (radius-2 neighbourhood)
# ---------------------------------------------------------------------------

@njit(cache=True)
def _nb_refresh_local_rates(
    lattice: np.ndarray,
    ftree: np.ndarray,
    event_rates: np.ndarray,
    ev_kind: np.ndarray,
    ev_src_x: np.ndarray, ev_src_y: np.ndarray,
    ev_dst_x: np.ndarray, ev_dst_y: np.ndarray,
    changed_x: np.ndarray, changed_y: np.ndarray, n_changed: int,
    Nx: int, Ny: int, periodic_x: bool,
    energy_lookup: np.ndarray,
    d0: float, nu_f: float, nu_d: float, nu_p: float,
    kB: float, T: float,
    num_drop_events: int, num_hop_events: int, max_events: int,
    visited: np.ndarray,           # bool[Ny, Nx] — reset inside
) -> None:
    visited[:, :] = False
    top_y = Ny - 1

    for ci in range(n_changed):
        cx = changed_x[ci]
        cy = changed_y[ci]
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                if abs(dx) + abs(dy) > 2:
                    continue
                xx = cx + dx
                yy = cy + dy
                if periodic_x:
                    xx = xx % Nx
                else:
                    if xx < 0 or xx >= Nx:
                        continue
                if yy < 0 or yy >= Ny:
                    continue
                if visited[yy, xx]:
                    continue
                visited[yy, xx] = True

                # Drop event (top row only)
                if yy == top_y:
                    _nb_update_rate_at_idx(
                        xx,
                        lattice, ftree, event_rates,
                        ev_kind, ev_src_x, ev_src_y, ev_dst_x, ev_dst_y,
                        max_events, Nx, Ny, periodic_x,
                        energy_lookup, d0, nu_f, nu_d, nu_p, kB, T,
                    )

                # Hop events (4 directions)
                base = num_drop_events + (yy * Nx + xx) * 4
                for offset in range(4):
                    _nb_update_rate_at_idx(
                        base + offset,
                        lattice, ftree, event_rates,
                        ev_kind, ev_src_x, ev_src_y, ev_dst_x, ev_dst_y,
                        max_events, Nx, Ny, periodic_x,
                        energy_lookup, d0, nu_f, nu_d, nu_p, kB, T,
                    )

                # Passivation event
                pidx = num_drop_events + num_hop_events + yy * Nx + xx
                _nb_update_rate_at_idx(
                    pidx,
                    lattice, ftree, event_rates,
                    ev_kind, ev_src_x, ev_src_y, ev_dst_x, ev_dst_y,
                    max_events, Nx, Ny, periodic_x,
                    energy_lookup, d0, nu_f, nu_d, nu_p, kB, T,
                )


# ---------------------------------------------------------------------------
# BFS bond-state relaxation
# ---------------------------------------------------------------------------

@njit(cache=True)
def _nb_bonding_relaxation(
    lattice: np.ndarray,
    seeds_x: np.ndarray, seeds_y: np.ndarray, n_seeds: int,
    Ny: int, Nx: int, periodic_x: bool,
    queue_x: np.ndarray, queue_y: np.ndarray,   # pre-allocated work buffers
    in_queue: np.ndarray,                        # bool[Ny, Nx]
    out_x: np.ndarray, out_y: np.ndarray,        # output: sites whose state changed
) -> int:
    """Run BFS relaxation starting from seeds + their neighbours.
    Returns the number of changed sites written into out_x / out_y."""
    in_queue[:, :] = False
    queue_head = 0
    queue_tail = 0
    n_changed  = 0
    buf_cap    = len(queue_x)

    DX = (1, -1, 0, 0)
    DY = (0,  0, 1, -1)

    # Seed the queue with each seed site and its 4 neighbours
    for si in range(n_seeds):
        sx = seeds_x[si]
        sy = seeds_y[si]

        # Seed itself
        if not in_queue[sy, sx] and queue_tail < buf_cap:
            queue_x[queue_tail] = sx
            queue_y[queue_tail] = sy
            queue_tail += 1
            in_queue[sy, sx] = True

        # Neighbours of seed
        for k in range(4):
            xx = sx + DX[k]
            yy = sy + DY[k]
            if periodic_x:
                xx = xx % Nx
            else:
                if xx < 0 or xx >= Nx:
                    continue
            if yy < 0 or yy >= Ny:
                continue
            if not in_queue[yy, xx] and queue_tail < buf_cap:
                queue_x[queue_tail] = xx
                queue_y[queue_tail] = yy
                queue_tail += 1
                in_queue[yy, xx] = True

    while queue_head < queue_tail:
        x = queue_x[queue_head]
        y = queue_y[queue_head]
        queue_head += 1
        in_queue[y, x] = False

        current = lattice[y, x]
        if current != FREE and current != DEPOSITED:
            continue

        desired = _nb_desired_mobile_state(lattice, x, y, Ny, Nx, periodic_x)
        if desired == current:
            continue

        # Apply change
        lattice[y, x] = desired
        if n_changed < len(out_x):
            out_x[n_changed] = x
            out_y[n_changed] = y
            n_changed += 1

        # Enqueue neighbours (may need re-evaluation)
        for k in range(4):
            xx = x + DX[k]
            yy = y + DY[k]
            if periodic_x:
                xx = xx % Nx
            else:
                if xx < 0 or xx >= Nx:
                    continue
            if yy < 0 or yy >= Ny:
                continue
            if not in_queue[yy, xx] and queue_tail < buf_cap:
                queue_x[queue_tail] = xx
                queue_y[queue_tail] = yy
                queue_tail += 1
                in_queue[yy, xx] = True

        # Re-enqueue self (ensures stable convergence)
        if not in_queue[y, x] and queue_tail < buf_cap:
            queue_x[queue_tail] = x
            queue_y[queue_tail] = y
            queue_tail += 1
            in_queue[y, x] = True

    return n_changed


# ---------------------------------------------------------------------------
# Main compiled batch stepper  ← THE KEY SPEEDUP
# ---------------------------------------------------------------------------

@njit(cache=True)
def _nb_run_n_steps(
    # Simulation state (modified in-place)
    lattice:       np.ndarray,   # int8[Ny, Nx]
    ftree:         np.ndarray,   # float64[max_events+1]
    event_rates:   np.ndarray,   # float64[max_events]
    # Event table (read-only)
    ev_kind:  np.ndarray,        # int8[max_events]   0=drop,1=hop,2=passivate
    ev_src_x: np.ndarray,        # int32[max_events]
    ev_src_y: np.ndarray,        # int32[max_events]
    ev_dst_x: np.ndarray,        # int32[max_events]  raw x (may need wrapping for hops)
    ev_dst_y: np.ndarray,        # int32[max_events]
    # Model parameters
    Nx: int, Ny: int, periodic_x: bool,
    energy_lookup: np.ndarray,
    d0: float, nu_f: float, nu_d: float, nu_p: float,
    kB: float, T: float,
    max_events: int, num_drop_events: int, num_hop_events: int,
    # Step / time limits
    n_steps_req: int,
    max_steps: int,
    max_time: float,
    log_every: int,
    # Current counters (passed by value; returned updated)
    current_step: int,
    current_time: float,
    # Work buffers (pre-allocated, reused across calls)
    direct_x:  np.ndarray,   # int32, size >= 2
    direct_y:  np.ndarray,
    relax_x:   np.ndarray,   # int32, size >= Nx*Ny
    relax_y:   np.ndarray,
    all_x:     np.ndarray,   # int32, size >= Nx*Ny + 2  (direct + relaxed)
    all_y:     np.ndarray,
    queue_x:   np.ndarray,   # int32, size >= 8*Nx*Ny  (BFS queue)
    queue_y:   np.ndarray,
    in_queue:  np.ndarray,   # bool[Ny, Nx]
    visited:   np.ndarray,   # bool[Ny, Nx]
    # Pre-generated random numbers from Python's self.rng (PCG64)
    # Shape: [2 * n_steps_req]  — two values per step (u1 for dt, u2 for event)
    rand_array: np.ndarray,
    # Per-batch log output (preallocated; numba fills these)
    log_steps: np.ndarray,   # int64
    log_times: np.ndarray,   # float64
    log_free:  np.ndarray,   # int32
    log_dep:   np.ndarray,   # int32
    log_pass:  np.ndarray,   # int32
    max_log:   int,
) -> Tuple[int, int, float, int]:
    """
    Run up to *n_steps_req* KMC steps entirely in compiled code.

    Returns
    -------
    (steps_done, new_step, new_time, n_log_entries)
    """
    n_log     = 0
    steps_done = 0

    for _ in range(n_steps_req):
        if current_step >= max_steps or current_time >= max_time:
            break

        r_tot = _nb_fen_total(ftree, max_events)
        if r_tot <= 0.0:
            break

        # Use pre-generated numbers from Python's PCG64 RNG — identical to original
        u1 = max(rand_array[2 * steps_done],     1.0e-15)
        u2 = max(rand_array[2 * steps_done + 1], 1.0e-15)
        dt = -math.log(u1) / r_tot
        target = u2 * r_tot
        idx = _nb_fen_find(ftree, max_events, target)

        kind  = ev_kind[idx]
        src_x = ev_src_x[idx]
        src_y = ev_src_y[idx]
        dst_x_raw = ev_dst_x[idx]
        dst_y_raw = ev_dst_y[idx]

        n_direct = 0

        if kind == 0:       # drop
            lattice[dst_y_raw, dst_x_raw] = FREE
            direct_x[0] = dst_x_raw
            direct_y[0] = dst_y_raw
            n_direct = 1

        elif kind == 2:     # passivate
            lattice[src_y, src_x] = PASSIVATED
            direct_x[0] = src_x
            direct_y[0] = src_y
            n_direct = 1

        else:               # hop
            if periodic_x:
                dst_x = dst_x_raw % Nx
            else:
                dst_x = dst_x_raw   # valid because rate > 0 implies in-bounds
            dst_y = dst_y_raw
            atom_type = lattice[src_y, src_x]
            lattice[src_y, src_x]  = EMPTY
            lattice[dst_y, dst_x]  = atom_type
            direct_x[0] = src_x;  direct_y[0] = src_y
            direct_x[1] = dst_x;  direct_y[1] = dst_y
            n_direct = 2

        # Bond-state BFS relaxation
        n_relax = _nb_bonding_relaxation(
            lattice,
            direct_x, direct_y, n_direct,
            Ny, Nx, periodic_x,
            queue_x, queue_y, in_queue,
            relax_x, relax_y,
        )

        # Merge changed sites into all_x / all_y
        n_all = n_direct + n_relax
        for i in range(n_direct):
            all_x[i] = direct_x[i]
            all_y[i] = direct_y[i]
        for i in range(n_relax):
            all_x[n_direct + i] = relax_x[i]
            all_y[n_direct + i] = relax_y[i]

        # Refresh event rates in the radius-2 neighbourhood
        _nb_refresh_local_rates(
            lattice, ftree, event_rates,
            ev_kind, ev_src_x, ev_src_y, ev_dst_x, ev_dst_y,
            all_x, all_y, n_all,
            Nx, Ny, periodic_x,
            energy_lookup, d0, nu_f, nu_d, nu_p, kB, T,
            num_drop_events, num_hop_events, max_events,
            visited,
        )

        current_time += dt
        current_step += 1
        steps_done   += 1

        # Record history log entry if due
        if current_step % log_every == 0 and n_log < max_log:
            nf = np.int32(0)
            nd = np.int32(0)
            np_ = np.int32(0)
            for row in range(Ny):
                for col in range(Nx):
                    v = lattice[row, col]
                    if   v == FREE:       nf += np.int32(1)
                    elif v == DEPOSITED:  nd += np.int32(1)
                    elif v == PASSIVATED: np_ += np.int32(1)
            log_steps[n_log] = current_step
            log_times[n_log] = current_time
            log_free [n_log] = nf
            log_dep  [n_log] = nd
            log_pass [n_log] = np_
            n_log += 1

    return steps_done, current_step, current_time, n_log


# ===========================================================================
# Simulation parameters
# ===========================================================================

@dataclass
class KMCParams:
    """All user-controlled simulation parameters."""

    Nx: int   = 40
    Ny: int   = 25
    T:  float = 300.0
    d0: float = 1.0e3
    e0: float = -0.2
    e1: float = -0.5
    nu_f: float = 5.0e9
    nu_d: float = 1.0e9
    nu_p: float = 1.0e2
    kB:   float = 8.617333262145e-5
    max_steps: int   = 400_000
    max_time:  float = 100.0
    rng_seed:  Optional[int] = 394583
    periodic_x: bool = True
    log_every:       int  = 1000
    save_snapshots:  bool = True
    snapshot_every:  int  = 10_000
    save_npy_states: bool = True
    output_dir:       str = "kmc_output"
    history_filename: str = "time_series.csv"
    validation_enabled: bool = False
    validation_every:   int  = 1000
    gui_batch_steps:    int  = 2000         # larger batch → fewer Python↔C crossings
    stop_fill_fraction: Optional[float] = None


# ===========================================================================
# Fenwick tree (Python class — kept for rebuild_all_rates / validation)
# ===========================================================================

class FenwickTree:
    """Binary Indexed Tree for cumulative-rate queries (Python path)."""

    def __init__(self, size: int):
        self.size = size
        self.tree = np.zeros(size + 1, dtype=float)

    def update(self, idx: int, delta: float) -> None:
        i = idx + 1
        while i <= self.size:
            self.tree[i] += delta
            i += i & -i

    def total(self) -> float:
        i = self.size
        s = 0.0
        while i > 0:
            s += self.tree[i]
            i -= i & -i
        return s

    def find_prefix_index(self, target: float) -> int:
        idx = 0
        bit = 1
        while bit < self.size:
            bit <<= 1
        bit >>= 1
        while bit > 0:
            nxt = idx + bit
            if nxt <= self.size and self.tree[nxt] < target:
                target -= self.tree[nxt]
                idx = nxt
            bit >>= 1
        return idx


# ===========================================================================
# KMC simulator
# ===========================================================================

class ElectrodepositionKMC:
    """2-D lattice KMC electrodeposition simulator with optional Numba speedup."""

    def __init__(self, params: KMCParams):
        self.p   = params
        self.rng = np.random.default_rng(self.p.rng_seed)


        # ── validation ────────────────────────────────────────────────────
        if self.p.Nx < 1:           raise ValueError("Nx must be at least 1.")
        if self.p.Ny < 2:           raise ValueError("Ny must be at least 2.")
        if self.p.T  <= 0:          raise ValueError("Temperature must be positive.")
        if self.p.log_every      < 1: raise ValueError("log_every must be >= 1.")
        if self.p.snapshot_every < 1: raise ValueError("snapshot_every must be >= 1.")
        if self.p.gui_batch_steps < 1: raise ValueError("gui_batch_steps must be >= 1.")

        # ── lattice ───────────────────────────────────────────────────────
        self.lattice = np.zeros((self.p.Ny, self.p.Nx), dtype=np.int8)
        self.lattice[0, :] = SUBSTRATE

        # ── interaction table ─────────────────────────────────────────────
        self.energy_lookup = np.zeros((5, 5), dtype=float)
        el = self.energy_lookup
        el[DEPOSITED,  DEPOSITED]  = self.p.e0
        el[DEPOSITED,  SUBSTRATE]  = self.p.e1
        el[SUBSTRATE,  DEPOSITED]  = self.p.e1
        el[SUBSTRATE,  SUBSTRATE]  = self.p.e1
        el[PASSIVATED, DEPOSITED]  = self.p.e0
        el[DEPOSITED,  PASSIVATED] = self.p.e0
        el[PASSIVATED, PASSIVATED] = self.p.e0
        el[PASSIVATED, SUBSTRATE]  = self.p.e1
        el[SUBSTRATE,  PASSIVATED] = self.p.e1

        # ── event indexing ────────────────────────────────────────────────
        self.num_drop_events       = self.p.Nx
        self.num_hop_events        = self.p.Nx * self.p.Ny * 4
        self.num_passivation_events = self.p.Nx * self.p.Ny
        self.max_events = (self.num_drop_events
                           + self.num_hop_events
                           + self.num_passivation_events)

        self.event_rates = np.zeros(self.max_events, dtype=float)
        self.ftree       = FenwickTree(self.max_events)

        # Python event-metadata list (kept for pure-Python path / validation)
        self.idx_to_data: List = [None] * self.max_events
        self._setup_indices()

        # Numba event-metadata arrays (built alongside)
        self._build_event_arrays()

        # Work buffers for the compiled stepper
        if NUMBA_AVAILABLE:
            self._alloc_work_buffers()

        # ── simulation counters ───────────────────────────────────────────
        self.time = 0.0
        self.step = 0

        # ── output paths ──────────────────────────────────────────────────
        self.output_dir  = Path(self.p.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.snapshot_dir = self.output_dir / "snapshots"
        if self.p.save_snapshots:
            self.snapshot_dir.mkdir(parents=True, exist_ok=True)

        self._cmap = ListedColormap(["white", "tab:blue", "tab:orange", "black", "tab:green"])
        self._fig  = None
        self._ax   = None
        self._im   = None
        self._title = None

        self.history: List = []

        # ── initial state ─────────────────────────────────────────────────
        self.rebuild_all_rates()
        self.record_history(label="initial")
        if self.p.save_snapshots:
            self.save_snapshot("initial")
        if self.p.save_npy_states:
            self.save_lattice_npy("initial")

    # ------------------------------------------------------------------
    # Event-table builders
    # ------------------------------------------------------------------

    def _setup_indices(self) -> None:
        """Populate the Python idx_to_data list."""
        top_y = self.p.Ny - 1
        for x in range(self.p.Nx):
            self.idx_to_data[x] = ("drop", None, (x, top_y))

        base  = self.num_drop_events
        dirs  = ((1, 0), (-1, 0), (0, 1), (0, -1))
        for y in range(self.p.Ny):
            for x in range(self.p.Nx):
                offset = (y * self.p.Nx + x) * 4
                for i, (dx, dy) in enumerate(dirs):
                    idx = base + offset + i
                    self.idx_to_data[idx] = ("hop", (x, y), (x + dx, y + dy))

        base = self.num_drop_events + self.num_hop_events
        for y in range(self.p.Ny):
            for x in range(self.p.Nx):
                idx = base + y * self.p.Nx + x
                self.idx_to_data[idx] = ("passivate", (x, y), (x, y))

    def _build_event_arrays(self) -> None:
        """Build numpy arrays encoding the event table for the Numba path."""
        n = self.max_events
        self._ev_kind  = np.zeros(n, dtype=np.int8)
        self._ev_src_x = np.full(n, -1, dtype=np.int32)
        self._ev_src_y = np.full(n, -1, dtype=np.int32)
        self._ev_dst_x = np.zeros(n, dtype=np.int32)
        self._ev_dst_y = np.zeros(n, dtype=np.int32)

        top_y = self.p.Ny - 1

        # Drop events
        for x in range(self.p.Nx):
            self._ev_kind[x]  = 0
            self._ev_dst_x[x] = x
            self._ev_dst_y[x] = top_y

        # Hop events
        base = self.num_drop_events
        dirs = ((1, 0), (-1, 0), (0, 1), (0, -1))
        for y in range(self.p.Ny):
            for x in range(self.p.Nx):
                for i, (dx, dy) in enumerate(dirs):
                    idx = base + (y * self.p.Nx + x) * 4 + i
                    self._ev_kind[idx]  = 1
                    self._ev_src_x[idx] = x
                    self._ev_src_y[idx] = y
                    self._ev_dst_x[idx] = x + dx   # raw; wrapping handled at runtime
                    self._ev_dst_y[idx] = y + dy

        # Passivation events
        base = self.num_drop_events + self.num_hop_events
        for y in range(self.p.Ny):
            for x in range(self.p.Nx):
                idx = base + y * self.p.Nx + x
                self._ev_kind[idx]  = 2
                self._ev_src_x[idx] = x
                self._ev_src_y[idx] = y
                self._ev_dst_x[idx] = x
                self._ev_dst_y[idx] = y

    def _alloc_work_buffers(self) -> None:
        """Pre-allocate numpy buffers reused by the compiled stepper."""
        Nx, Ny = self.p.Nx, self.p.Ny
        max_xy = Nx * Ny

        self._buf_direct_x = np.zeros(2,           dtype=np.int32)
        self._buf_direct_y = np.zeros(2,           dtype=np.int32)
        self._buf_relax_x  = np.zeros(max_xy,      dtype=np.int32)
        self._buf_relax_y  = np.zeros(max_xy,      dtype=np.int32)
        self._buf_all_x    = np.zeros(max_xy + 2,  dtype=np.int32)
        self._buf_all_y    = np.zeros(max_xy + 2,  dtype=np.int32)
        self._buf_queue_x  = np.zeros(8 * max_xy,  dtype=np.int32)
        self._buf_queue_y  = np.zeros(8 * max_xy,  dtype=np.int32)
        self._buf_inqueue  = np.zeros((Ny, Nx),    dtype=np.bool_)
        self._buf_visited  = np.zeros((Ny, Nx),    dtype=np.bool_)

        _LOG = max(self.p.gui_batch_steps // max(1, self.p.log_every) + 4, 64)
        self._log_steps = np.zeros(_LOG, dtype=np.int64)
        self._log_times = np.zeros(_LOG, dtype=np.float64)
        self._log_free  = np.zeros(_LOG, dtype=np.int32)
        self._log_dep   = np.zeros(_LOG, dtype=np.int32)
        self._log_pass  = np.zeros(_LOG, dtype=np.int32)
        self._max_log   = _LOG

    # ------------------------------------------------------------------
    # Geometry helpers (Python path)
    # ------------------------------------------------------------------

    def wrap_x(self, x: int) -> Optional[int]:
        if self.p.periodic_x:
            return x % self.p.Nx
        if 0 <= x < self.p.Nx:
            return x
        return None

    def valid_neighbor_coords(self, x: int, y: int) -> List[Coord]:
        nbrs: List[Coord] = []
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            xx = self.wrap_x(x + dx)
            if xx is None:
                continue
            yy = y + dy
            if 0 <= yy < self.p.Ny:
                nbrs.append((xx, yy))
        return nbrs

    def radius2_sites(self, seeds: Sequence[Coord]) -> List[Coord]:
        seen: set = set()
        out:  List[Coord] = []
        for x0, y0 in seeds:
            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    if abs(dx) + abs(dy) > 2:
                        continue
                    xx = self.wrap_x(x0 + dx)
                    yy = y0 + dy
                    if xx is None or not (0 <= yy < self.p.Ny):
                        continue
                    if (xx, yy) not in seen:
                        seen.add((xx, yy))
                        out.append((xx, yy))
        return out

    # ------------------------------------------------------------------
    # Event index helpers (Python path)
    # ------------------------------------------------------------------

    def drop_index(self, x: int) -> int:
        return x

    def hop_base_index(self, x: int, y: int) -> int:
        return self.num_drop_events + (y * self.p.Nx + x) * 4

    def passivation_index(self, x: int, y: int) -> int:
        return self.num_drop_events + self.num_hop_events + y * self.p.Nx + x

    # ------------------------------------------------------------------
    # Energetics (Python path — used by rebuild_all_rates / validation)
    # ------------------------------------------------------------------

    def calc_local_energy(self, x: int, y: int, atom_type: int) -> float:
        e = 0.0
        for nx, ny in self.valid_neighbor_coords(x, y):
            e += self.energy_lookup[atom_type, int(self.lattice[ny, nx])]
        return e

    def site_has_empty_neighbor(self, x: int, y: int) -> bool:
        for nx, ny in self.valid_neighbor_coords(x, y):
            if int(self.lattice[ny, nx]) == EMPTY:
                return True
        return False

    def desired_mobile_state_at_site(self, x: int, y: int) -> int:
        for nx, ny in self.valid_neighbor_coords(x, y):
            if int(self.lattice[ny, nx]) in (DEPOSITED, SUBSTRATE):
                return DEPOSITED
        return FREE

    def get_event_rate(self, kind: str, src: Optional[Coord], dst: Coord) -> float:
        if kind == "drop":
            x1, y1 = dst
            return self.p.d0 if self.lattice[y1, x1] == EMPTY else 0.0

        if kind == "passivate":
            x0, y0 = src
            if int(self.lattice[y0, x0]) != DEPOSITED:
                return 0.0
            return self.p.nu_p if self.site_has_empty_neighbor(x0, y0) else 0.0

        x0, y0 = src
        atom_type = int(self.lattice[y0, x0])
        if atom_type not in (FREE, DEPOSITED):
            return 0.0
        x1_raw, y1 = dst
        x1 = self.wrap_x(x1_raw)
        if x1 is None or not (0 <= y1 < self.p.Ny):
            return 0.0
        if self.lattice[y1, x1] != EMPTY:
            return 0.0
        nu = self.p.nu_f if atom_type == FREE else self.p.nu_d
        e_init = self.calc_local_energy(x0, y0, atom_type)
        self.lattice[y0, x0] = EMPTY
        final_type = self.desired_mobile_state_at_site(x1, y1)
        e_final = self.calc_local_energy(x1, y1, final_type)
        self.lattice[y0, x0] = atom_type
        return nu * math.exp(-(e_final - e_init) / (2.0 * self.p.kB * self.p.T))

    def update_rate_at_index(self, idx: int) -> None:
        kind, src, dst = self.idx_to_data[idx]
        new_rate = self.get_event_rate(kind, src, dst)
        delta    = new_rate - self.event_rates[idx]
        if abs(delta) > 1.0e-18:
            self.event_rates[idx] = new_rate
            self.ftree.update(idx, delta)

    def rebuild_all_rates(self) -> None:
        self.event_rates[:] = 0.0
        self.ftree = FenwickTree(self.max_events)
        for idx in range(self.max_events):
            self.update_rate_at_index(idx)

    def refresh_local_rates(self, changed_sites: Sequence[Coord]) -> None:
        for x, y in self.radius2_sites(changed_sites):
            if y == self.p.Ny - 1:
                self.update_rate_at_index(self.drop_index(x))
            base = self.hop_base_index(x, y)
            for offset in range(4):
                self.update_rate_at_index(base + offset)
            self.update_rate_at_index(self.passivation_index(x, y))

    # ------------------------------------------------------------------
    # Bonding relaxation (Python path)
    # ------------------------------------------------------------------

    def desired_bond_state(self, x: int, y: int) -> int:
        site_type = int(self.lattice[y, x])
        if site_type not in (FREE, DEPOSITED):
            return site_type
        return self.desired_mobile_state_at_site(x, y)

    def update_bonding_relaxation(self, seeds: Sequence[Coord]) -> List[Coord]:
        q: Deque[Coord] = deque()
        queued  = set()
        changed = set()
        initial = set()
        for sx, sy in seeds:
            initial.add((sx, sy))
            for nbr in self.valid_neighbor_coords(sx, sy):
                initial.add(nbr)
        for site in initial:
            q.append(site);  queued.add(site)
        while q:
            x, y = q.popleft();  queued.discard((x, y))
            current = int(self.lattice[y, x])
            if current not in (FREE, DEPOSITED):
                continue
            desired = self.desired_bond_state(x, y)
            if desired == current:
                continue
            self.lattice[y, x] = desired
            changed.add((x, y))
            for nbr in self.valid_neighbor_coords(x, y):
                if nbr not in queued:
                    q.append(nbr);  queued.add(nbr)
            if (x, y) not in queued:
                q.append((x, y));  queued.add((x, y))
        return list(changed)

    # ------------------------------------------------------------------
    # KMC step (Python path — used when numba unavailable)
    # ------------------------------------------------------------------

    def execute_step(self) -> bool:
        r_tot = self.ftree.total()
        if r_tot <= 0.0:
            return False
        u1 = max(float(self.rng.random()), 1.0e-15)
        dt = -math.log(u1) / r_tot
        u2 = max(float(self.rng.random()), 1.0e-15)
        target = u2 * r_tot
        idx   = self.ftree.find_prefix_index(target)
        kind, src, dst = self.idx_to_data[idx]
        directly_changed: List[Coord] = []
        if kind == "drop":
            x1, y1 = dst
            self.lattice[y1, x1] = FREE
            directly_changed.append((x1, y1))
        elif kind == "passivate":
            x0, y0 = src
            self.lattice[y0, x0] = PASSIVATED
            directly_changed.append((x0, y0))
        else:
            x0, y0 = src
            x1_raw, y1 = dst
            x1 = self.wrap_x(x1_raw)
            atom_type = int(self.lattice[y0, x0])
            self.lattice[y0, x0] = EMPTY
            self.lattice[y1, x1] = atom_type
            directly_changed.append((x0, y0))
            directly_changed.append((x1, y1))
        relaxed = self.update_bonding_relaxation(directly_changed)
        self.refresh_local_rates(list({*directly_changed, *relaxed}))
        self.time += dt
        self.step += 1
        if self.p.validation_enabled and self.step % self.p.validation_every == 0:
            self.validate_against_full_rebuild()
        return True

    # ------------------------------------------------------------------
    # Compiled batch stepper (Numba path)
    # ------------------------------------------------------------------

    def run_n_steps_fast(self, n: int) -> Tuple[int, bool]:
        """
        Run up to *n* steps using the compiled kernel (or fall back to Python).

        Returns
        -------
        (steps_done, finished)
            *finished* is True when max_steps / max_time / no-events is reached.
        """
        if not NUMBA_AVAILABLE:
            # Pure-Python fallback
            for i in range(n):
                if self.step >= self.p.max_steps or self.time >= self.p.max_time:
                    return i, True
                if not self.execute_step():
                    return i, True
                if self.step % self.p.log_every == 0:
                    self.record_history(label="regular")
            return n, False

        # Pre-generate 2*n random numbers using the same PCG64 generator as the
        # original Python mode.  Calling .random(size=2*n) produces the identical
        # sequence as 2*n sequential .random() calls — so trajectories match exactly.
        rand_array = self.rng.random(size=2 * n)

        steps_done, new_step, new_time, n_log = _nb_run_n_steps(
            self.lattice,
            self.ftree.tree,
            self.event_rates,
            self._ev_kind,
            self._ev_src_x, self._ev_src_y,
            self._ev_dst_x, self._ev_dst_y,
            self.p.Nx, self.p.Ny, bool(self.p.periodic_x),
            self.energy_lookup,
            self.p.d0, self.p.nu_f, self.p.nu_d, self.p.nu_p,
            self.p.kB, self.p.T,
            self.max_events, self.num_drop_events, self.num_hop_events,
            n,
            self.p.max_steps, self.p.max_time, self.p.log_every,
            self.step, self.time,
            self._buf_direct_x, self._buf_direct_y,
            self._buf_relax_x,  self._buf_relax_y,
            self._buf_all_x,    self._buf_all_y,
            self._buf_queue_x,  self._buf_queue_y,
            self._buf_inqueue,
            self._buf_visited,
            rand_array,
            self._log_steps, self._log_times,
            self._log_free,  self._log_dep, self._log_pass,
            self._max_log,
        )

        # Flush compiled log entries into Python history
        total_rate_now = float(np.sum(self.event_rates))
        for i in range(n_log):
            nf = int(self._log_free[i])
            nd = int(self._log_dep[i])
            np_ = int(self._log_pass[i])
            self.history.append({
                "label":      "regular",
                "step":       int(self._log_steps[i]),
                "time":       float(self._log_times[i]),
                "free":       nf,
                "deposited":  nd,
                "passivated": np_,
                "total_atoms": nf + nd + np_,
                "total_rate": total_rate_now,   # approximate (end-of-batch value)
            })

        self.step = int(new_step)
        self.time = float(new_time)

        finished = (
            self.step >= self.p.max_steps
            or self.time >= self.p.max_time
            or self.ftree.total() <= 0.0
        )
        return steps_done, finished

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_against_full_rebuild(self) -> None:
        saved_lattice = self.lattice.copy()
        saved_rates   = self.event_rates.copy()
        self.rebuild_all_rates()
        rebuilt = self.event_rates.copy()
        if not np.allclose(saved_rates, rebuilt, rtol=1e-12, atol=1e-12):
            diff = np.max(np.abs(saved_rates - rebuilt))
            raise RuntimeError(f"Validation failed: max diff = {diff}")
        self.lattice[:, :] = saved_lattice
        self.event_rates[:] = saved_rates
        self.ftree = FenwickTree(self.max_events)
        for i, rate in enumerate(saved_rates):
            if abs(rate) > 1.0e-18:
                self.ftree.update(i, rate)

    # ------------------------------------------------------------------
    # Output helpers
    # ------------------------------------------------------------------

    def counts(self) -> Tuple[int, int, int, int]:
        n_free = int(np.count_nonzero(self.lattice == FREE))
        n_dep  = int(np.count_nonzero(self.lattice == DEPOSITED))
        n_pass = int(np.count_nonzero(self.lattice == PASSIVATED))
        return n_free, n_dep, n_pass, n_free + n_dep + n_pass

    def record_history(self, label: str = "regular") -> None:
        n_free, n_dep, n_pass, n_total = self.counts()
        self.history.append({
            "label":       label,
            "step":        self.step,
            "time":        self.time,
            "free":        n_free,
            "deposited":   n_dep,
            "passivated":  n_pass,
            "total_atoms": n_total,
            "total_rate":  float(np.sum(self.event_rates, dtype=float)),
        })

    def write_history_csv(self) -> Path:
        out_path = self.output_dir / self.p.history_filename
        with out_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "label", "step", "time", "free", "deposited",
                "passivated", "total_atoms", "total_rate",
            ])
            writer.writeheader()
            writer.writerows(self.history)
        return out_path

    def lattice_for_display(self) -> np.ndarray:
        return np.flipud(self.lattice)

    def save_snapshot(self, tag: str) -> Path:
        out_path = self.snapshot_dir / f"{tag}.png"
        cell_px  = max(12, min(24, 800 // max(self.p.Nx, self.p.Ny)))
        fig_w    = max(6, self.p.Nx * cell_px / 80)
        fig_h    = max(4, self.p.Ny * cell_px / 80)
        fig, ax  = plt.subplots(figsize=(fig_w + 2, fig_h + 1.5), dpi=100)
        im = ax.imshow(self.lattice_for_display(), cmap=self._cmap,
                       vmin=0, vmax=4, interpolation="nearest", aspect="equal")
        cbar = fig.colorbar(im, ax=ax, ticks=[0, 1, 2, 3, 4], fraction=0.046, pad=0.04)
        cbar.ax.set_yticklabels(["Empty", "Free", "Deposited", "Substrate", "Passivated"])
        cbar.ax.tick_params(labelsize=9)
        ax.set_xlabel("x  (lattice site)", fontsize=10)
        ax.set_ylabel("y  (lattice site)", fontsize=10)
        ax.set_title(
            f"LKMC Electrodeposition — {tag}\n"
            f"Step {self.step:,}    Sim time {self.time:.3e} s    "
            f"T = {self.p.T} K    Nx={self.p.Nx}  Ny={self.p.Ny}",
            fontsize=9, pad=8,
        )
        fig.tight_layout()
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        return out_path

    def save_lattice_npy(self, tag: str) -> Path:
        out_path = self.output_dir / f"lattice_{tag}.npy"
        np.save(out_path, self.lattice)
        return out_path

    def init_gui_figure(self) -> None:
        self._fig   = Figure(figsize=(6, 5))
        self._ax    = self._fig.add_subplot(111)
        self._im    = self._ax.imshow(self.lattice_for_display(), cmap=self._cmap, vmin=0, vmax=4)
        self._title = self._ax.set_title(f"Step {self.step}, Time {self.time:.2e}")
        self._ax.set_xlabel("x"); self._ax.set_ylabel("y")

    def refresh_gui_figure(self) -> None:
        if self._im is None:
            return
        self._im.set_data(self.lattice_for_display())
        self._title.set_text(f"Step {self.step}, Time {self.time:.2e}")

    # ------------------------------------------------------------------
    # Run helpers
    # ------------------------------------------------------------------

    def finalize_outputs(self) -> None:
        self.record_history(label="final")
        if self.p.save_snapshots:
            self.save_snapshot(f"final_step_{self.step}")
        if self.p.save_npy_states:
            self.save_lattice_npy(f"final_step_{self.step}")
        self.write_history_csv()

    def run_cli(self) -> None:
        """Run simulation in CLI / scan mode.

        When Numba is available the inner loop runs entirely in compiled code
        in batches of _CLI_BATCH steps, making scan-mode runs as fast as the
        GUI mode.  The pure-Python fallback is used otherwise.
        """
        total_sites = self.p.Nx * (self.p.Ny - 1)

        if not NUMBA_AVAILABLE:
            # ── Pure-Python path (identical to v4b) ────────────────────────
            while self.step < self.p.max_steps and self.time < self.p.max_time:
                if not self.execute_step():
                    break
                if self.p.stop_fill_fraction is not None:
                    _, n_dep, n_pass, _ = self.counts()
                    if (n_dep + n_pass) / total_sites >= self.p.stop_fill_fraction:
                        break
                if self.step % self.p.log_every == 0:
                    self.record_history(label="regular")
                if self.p.save_snapshots and self.step % self.p.snapshot_every == 0:
                    self.save_snapshot(f"step_{self.step:07d}")
                if self.p.save_npy_states and self.step % self.p.snapshot_every == 0:
                    self.save_lattice_npy(f"step_{self.step:07d}")
            self.finalize_outputs()
            return

        # ── Compiled fast path ─────────────────────────────────────────────
        # Batch size: large enough to keep Python overhead negligible, small
        # enough that stop_fill_fraction and snapshot boundaries aren't missed
        # by more than one batch.  50 000 is a good balance.
        _CLI_BATCH = 50_000

        # Ensure the log buffer is large enough for one full CLI batch.
        cli_log_need = _CLI_BATCH // max(1, self.p.log_every) + 4
        if cli_log_need > self._max_log:
            self._log_steps = np.zeros(cli_log_need, dtype=np.int64)
            self._log_times = np.zeros(cli_log_need, dtype=np.float64)
            self._log_free  = np.zeros(cli_log_need, dtype=np.int32)
            self._log_dep   = np.zeros(cli_log_need, dtype=np.int32)
            self._log_pass  = np.zeros(cli_log_need, dtype=np.int32)
            self._max_log   = cli_log_need

        while self.step < self.p.max_steps and self.time < self.p.max_time:
            old_step = self.step
            batch    = min(_CLI_BATCH, self.p.max_steps - self.step)

            _done, finished = self.run_n_steps_fast(batch)

            # ── Snapshot saving ────────────────────────────────────────────
            # Save at most one snapshot per batch at the batch-end state.
            # Precision: within one batch (≤50 k steps) of the exact interval.
            if self.p.save_snapshots or self.p.save_npy_states:
                snap_ev    = self.p.snapshot_every
                first_snap = (old_step // snap_ev + 1) * snap_ev
                if first_snap <= self.step:
                    tag = f"step_{self.step:07d}"
                    if self.p.save_snapshots:
                        self.save_snapshot(tag)
                    if self.p.save_npy_states:
                        self.save_lattice_npy(tag)

            # ── Fill-fraction stop condition ───────────────────────────────
            if self.p.stop_fill_fraction is not None:
                _, n_dep, n_pass, _ = self.counts()
                if (n_dep + n_pass) / total_sites >= self.p.stop_fill_fraction:
                    break

            if finished:
                break

        self.finalize_outputs()


# ===========================================================================
# GUI
# ===========================================================================

class SimulationGUI:
    """Tkinter front end — identical appearance to v4b, now using compiled kernel."""

    FIELD_SPECS = [
        ("Nx",             int,   "Nx",            "Lattice width (Nx)",        "Number of lattice sites in x direction"),
        ("Ny",             int,   "Ny",            "Lattice height (Ny)",       "Number of lattice sites in y (row 0 = substrate)"),
        ("T",              float, "T",             "Temperature (K)",           "Simulation temperature in Kelvin"),
        ("d0",             float, "d0",            "Drop rate (d₀)",            "Atom dropping rate per empty top-row site (s⁻¹)"),
        ("e0",             float, "e0",            "Bonded energy e₀ (eV)",     "Interaction energy for DEPOSITED/PASSIVATED bonded neighbors (eV)"),
        ("e1",             float, "e1",            "Atom–substrate e₁ (eV)",    "Interaction energy between atom and substrate (eV)"),
        ("nu_f",           float, "nu_f",          "Free attempt freq ν_f",     "Attempt frequency for FREE atoms hopping (s⁻¹)"),
        ("nu_d",           float, "nu_d",          "Dep. attempt freq ν_d",     "Attempt frequency for DEPOSITED atoms hopping (s⁻¹)"),
        ("nu_p",           float, "nu_p",          "Passivation rate ν_p",      "Rate for DEPOSITED atoms with EMPTY neighbor → PASSIVATED (s⁻¹)"),
        ("max_steps",      int,   "max_steps",     "Max steps",                 "Stop simulation after this many KMC steps"),
        ("max_time",       float, "max_time",      "Max sim time (s)",          "Stop simulation after this much simulated time"),
        ("log_every",      int,   "log_every",     "Log interval (steps)",      "Record history every N steps"),
        ("snapshot_every", int,   "snapshot_every","Snapshot interval (steps)", "Save PNG snapshot every N steps"),
        ("periodic_x",     int,   None,            "Periodic x  (1=yes)",       "Use periodic boundary conditions in x"),
        ("rng_seed",       int,   "rng_seed",      "Random seed",               "Seed for reproducibility"),
    ]

    def __init__(self, root: tk.Tk):
        self.root   = root
        self.root.title("LKMC — Lattice KMC Electrodeposition  [Numba accelerated]"
                        if NUMBA_AVAILABLE else
                        "LKMC — Lattice KMC Electrodeposition  [Python mode]")
        self.root.configure(bg="#1e1e2e")

        self.params  = KMCParams()
        self.sim: Optional[ElectrodepositionKMC] = None
        self.paused  = False
        self.stopped = False
        self.canvas: Optional[FigureCanvasTkAgg] = None

        self._hist_steps:      List[int] = []
        self._hist_free:       List[int] = []
        self._hist_deposited:  List[int] = []
        self._hist_passivated: List[int] = []

        self._setup_styles()
        self._build_ui()

    # ── Styling ────────────────────────────────────────────────────────

    def _setup_styles(self) -> None:
        style = ttk.Style()
        style.theme_use("clam")
        BG    = "#1e1e2e"; PANEL = "#2a2a3e"; ACC = "#89b4fa"
        FG    = "#cdd6f4"; ENTRY = "#313244"
        style.configure(".",               background=BG,    foreground=FG,  font=("Segoe UI", 9))
        style.configure("TFrame",          background=BG)
        style.configure("TLabel",          background=BG,    foreground=FG)
        style.configure("TLabelframe",     background=PANEL, foreground=ACC, relief="flat", borderwidth=1)
        style.configure("TLabelframe.Label", background=PANEL, foreground=ACC, font=("Segoe UI", 9, "bold"))
        style.configure("TEntry",          fieldbackground=ENTRY, foreground=FG, insertcolor=FG)
        style.configure("TCheckbutton",    background=PANEL, foreground=FG)
        style.configure("Header.TLabel",   background=BG,    foreground=ACC, font=("Segoe UI", 13, "bold"))
        style.configure("Sub.TLabel",      background=BG,    foreground="#6c7086", font=("Segoe UI", 8))
        style.configure("Stat.TLabel",     background=PANEL, foreground=FG,  font=("Consolas", 9))
        style.configure("StatVal.TLabel",  background=PANEL, foreground=ACC, font=("Consolas", 9, "bold"))
        for name, bg, fg, abg in [
            ("Run.TButton",   "#a6e3a1", "#1e1e2e", "#94d3a2"),
            ("Pause.TButton", "#f9e2af", "#1e1e2e", "#e8d09e"),
            ("Stop.TButton",  "#f38ba8", "#1e1e2e", "#e07090"),
            ("Save.TButton",  ACC,       "#1e1e2e", "#7aa3e8"),
        ]:
            style.configure(name, background=bg, foreground=fg,
                            font=("Segoe UI", 9, "bold"), relief="flat", padding=(8, 5))
            style.map(name, background=[("active", abg)])
        self._colors = {"BG": BG, "PANEL": PANEL, "ACC": ACC, "FG": FG, "ENTRY": ENTRY}

    # ── UI construction ────────────────────────────────────────────────

    def _build_ui(self) -> None:
        C = self._colors
        header = ttk.Frame(self.root)
        header.pack(side="top", fill="x", padx=12, pady=(10, 4))
        ttk.Label(header, text="LKMC Electrodeposition Simulator", style="Header.TLabel").pack(side="left")
        accel_tag = "  ⚡ Numba accelerated" if NUMBA_AVAILABLE else "  ⚠ numba not found — pure Python"
        ttk.Label(header, text=f"Lattice Kinetic Monte Carlo  ·  2D Electrodeposition{accel_tag}",
                  style="Sub.TLabel").pack(side="left", padx=(10, 0))

        body = ttk.Frame(self.root)
        body.pack(fill="both", expand=True, padx=12, pady=4)

        left = ttk.Frame(body)
        left.pack(side="left", fill="y", padx=(0, 8))
        self._build_param_panel(left)
        self._build_control_panel(left)
        self._build_stats_panel(left)

        right = ttk.Frame(body)
        right.pack(side="left", fill="both", expand=True)
        self._build_plot_area(right)

    def _build_param_panel(self, parent: ttk.Frame) -> None:
        groups = [
            ("Lattice Geometry",  self.FIELD_SPECS[0:2]),
            ("Physics",           self.FIELD_SPECS[2:9]),
            ("Run Controls",      self.FIELD_SPECS[9:13]),
            ("Misc",              self.FIELD_SPECS[13:15]),
        ]
        self.entries: dict = {}
        for group_label, specs in groups:
            frame = ttk.LabelFrame(parent, text=f"  {group_label}  ")
            frame.pack(fill="x", pady=(0, 6))
            for row_i, (pname, ptype, attr, label, tooltip) in enumerate(specs):
                default_val = (str(getattr(self.params, attr)) if attr is not None
                               else ("1" if self.params.periodic_x else "0"))
                ttk.Label(frame, text=label, anchor="w").grid(
                    row=row_i, column=0, sticky="w", padx=(8, 4), pady=2)
                entry = ttk.Entry(frame, width=13)
                entry.insert(0, default_val)
                entry.grid(row=row_i, column=1, padx=(0, 8), pady=2)
                entry.bind("<Enter>", lambda e, t=tooltip: self.root.title(f"LKMC  ·  {t}"))
                entry.bind("<Leave>", lambda e: self.root.title(
                    "LKMC — Lattice KMC Electrodeposition"))
                self.entries[pname] = (entry, ptype)

    def _build_control_panel(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="  Controls  ")
        frame.pack(fill="x", pady=(0, 6))
        self.run_btn   = ttk.Button(frame, text="▶  Run",   style="Run.TButton",   command=self.start_simulation)
        self.pause_btn = ttk.Button(frame, text="⏸  Pause", style="Pause.TButton", command=self.toggle_pause)
        self.stop_btn  = ttk.Button(frame, text="⏹  Stop",  style="Stop.TButton",  command=self.stop_simulation)
        self.save_btn  = ttk.Button(frame, text="💾  Save snapshot", style="Save.TButton", command=self.save_now)
        for i, btn in enumerate([self.run_btn, self.pause_btn, self.stop_btn, self.save_btn]):
            btn.grid(row=i // 2, column=i % 2, padx=6, pady=4, sticky="ew")
        frame.columnconfigure(0, weight=1); frame.columnconfigure(1, weight=1)

    def _build_stats_panel(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="  Live Statistics  ")
        frame.pack(fill="x", pady=(0, 6))
        stats = [
            ("Step",       "step_var"),
            ("Sim time",   "time_var"),
            ("Free atoms", "free_var"),
            ("Deposited",  "dep_var"),
            ("Passivated", "pass_var"),
            ("Total rate", "rate_var"),
            ("Speed",      "speed_var"),
        ]
        for i, (label, attr) in enumerate(stats):
            var = tk.StringVar(value="—")
            setattr(self, attr, var)
            ttk.Label(frame, text=f"{label}:", style="Stat.TLabel", width=12, anchor="w").grid(
                row=i, column=0, padx=(8, 2), pady=1, sticky="w")
            ttk.Label(frame, textvariable=var, style="StatVal.TLabel", width=14, anchor="e").grid(
                row=i, column=1, padx=(0, 8), pady=1, sticky="e")
        legend_frame = ttk.Frame(frame)
        legend_frame.grid(row=len(stats), column=0, columnspan=2, pady=(6, 4), padx=8, sticky="w")
        for color, text in [
            ("white",    "Empty"),
            ("#5599dd",  "Free"),
            ("#dd8833",  "Deposited"),
            ("#222222",  "Substrate"),
            ("#66bb6a",  "Passivated"),
        ]:
            tk.Label(legend_frame, text="●", fg=color,
                     bg=self._colors["PANEL"], font=("Segoe UI", 11)).pack(side="left")
            tk.Label(legend_frame, text=f" {text}  ",
                     bg=self._colors["PANEL"], fg=self._colors["FG"],
                     font=("Segoe UI", 8)).pack(side="left")

    def _build_plot_area(self, parent: ttk.Frame) -> None:
        C = self._colors
        self._fig = Figure(figsize=(8, 7), facecolor=C["BG"])
        self._fig.subplots_adjust(hspace=0.35)

        self._ax_lat = self._fig.add_subplot(2, 1, 1)
        self._ax_lat.set_facecolor(C["BG"])
        for sp in self._ax_lat.spines.values(): sp.set_edgecolor(C["ACC"])
        self._ax_lat.tick_params(colors=C["FG"], labelsize=8)
        self._ax_lat.set_xlabel("x  (lattice site)", color=C["FG"], fontsize=8)
        self._ax_lat.set_ylabel("y  (lattice site)", color=C["FG"], fontsize=8)
        self._ax_lat.set_title("Lattice  —  not started", color=C["ACC"], fontsize=9, pad=6)
        blank = np.zeros((self.params.Ny, self.params.Nx), dtype=np.int8)
        self._im = self._ax_lat.imshow(blank, cmap=self._cmap_for_display(),
                                       vmin=0, vmax=4, interpolation="nearest", aspect="auto")
        cbar = self._fig.colorbar(self._im, ax=self._ax_lat,
                                  ticks=[0, 1, 2, 3, 4], fraction=0.03, pad=0.02)
        cbar.ax.set_yticklabels(["Empty", "Free", "Dep.", "Sub.", "Pass."])
        cbar.ax.tick_params(labelsize=7, colors=C["FG"])
        cbar.outline.set_edgecolor(C["ACC"])

        self._ax_cnt = self._fig.add_subplot(2, 1, 2)
        self._ax_cnt.set_facecolor(C["BG"])
        for sp in self._ax_cnt.spines.values(): sp.set_edgecolor(C["ACC"])
        self._ax_cnt.tick_params(colors=C["FG"], labelsize=8)
        self._ax_cnt.set_xlabel("KMC step", color=C["FG"], fontsize=8)
        self._ax_cnt.set_ylabel("Atom count", color=C["FG"], fontsize=8)
        self._ax_cnt.set_title("Atom counts over time", color=C["ACC"], fontsize=9, pad=6)
        self._line_free,  = self._ax_cnt.plot([], [], color="#5599dd", lw=1.5, label="Free")
        self._line_dep,   = self._ax_cnt.plot([], [], color="#dd8833", lw=1.5, label="Deposited")
        self._line_pass,  = self._ax_cnt.plot([], [], color="#66bb6a", lw=1.5, label="Passivated")
        self._ax_cnt.legend(facecolor=C["PANEL"], edgecolor=C["ACC"],
                            labelcolor=C["FG"], fontsize=8, loc="upper left")

        self.canvas = FigureCanvasTkAgg(self._fig, master=parent)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.canvas.draw()

        # Status bar
        self._status_var = tk.StringVar(value="Ready.")
        tk.Label(parent, textvariable=self._status_var, anchor="w",
                 bg=self._colors["BG"], fg=self._colors["FG"],
                 font=("Consolas", 8)).pack(fill="x", padx=4, pady=(0, 2))

        self._wall_t0 = 0.0   # wall-clock reference for speed display

    @staticmethod
    def _cmap_for_display():
        return ListedColormap(["#111111", "#5599dd", "#dd8833", "#222222", "#66bb6a"])

    # ── Parameter parsing ──────────────────────────────────────────────

    def parse_gui_params(self) -> KMCParams:
        values = {}
        for pname, (entry, ptype) in self.entries.items():
            raw = entry.get().strip()
            try:
                values[pname] = ptype(raw)
            except ValueError as exc:
                raise ValueError(f"'{pname}' expects {ptype.__name__} but got '{raw}'.") from exc
        values["periodic_x"] = bool(values["periodic_x"])
        return KMCParams(**values)

    # ── Control callbacks ──────────────────────────────────────────────

    def toggle_pause(self) -> None:
        self.paused = not self.paused
        self.pause_btn.config(text="▶  Resume" if self.paused else "⏸  Pause")

    def stop_simulation(self) -> None:
        self.stopped = True

    def save_now(self) -> None:
        if self.sim is None:
            messagebox.showinfo("Nothing to save", "Start a simulation first.")
            return
        path = self.sim.save_snapshot(f"manual_step_{self.sim.step:07d}")
        messagebox.showinfo("Saved", f"Snapshot saved to:\n{path}")

    def start_simulation(self) -> None:
        try:
            self.params = self.parse_gui_params()
        except ValueError as exc:
            messagebox.showerror("Parameter error", str(exc))
            return

        self.paused  = False
        self.stopped = False
        self.pause_btn.config(text="⏸  Pause")
        self._hist_steps.clear()
        self._hist_free.clear()
        self._hist_deposited.clear()
        self._hist_passivated.clear()

        if NUMBA_AVAILABLE:
            self._status_var.set(
                "Compiling simulation kernel (first run only — may take ~10–20 s)…")
            self.root.update()

        try:
            self.sim = ElectrodepositionKMC(self.params)
        except Exception as exc:
            messagebox.showerror("Simulation error", str(exc))
            return

        blank = np.zeros((self.params.Ny, self.params.Nx), dtype=np.int8)
        self._im.set_data(blank)
        self._im.set_extent([-0.5, self.params.Nx - 0.5,
                              self.params.Ny - 0.5, -0.5])
        self._ax_lat.set_xlim(-0.5, self.params.Nx - 0.5)
        self._ax_lat.set_ylim(self.params.Ny - 0.5, -0.5)
        self._line_free.set_data([], [])
        self._line_dep.set_data([], [])
        self._line_pass.set_data([], [])
        self._ax_cnt.relim()

        self.run_btn.config(state="disabled")
        self._wall_t0 = time.perf_counter()
        self._last_draw_wall = 0.0          # force an immediate first draw
        self._last_step_for_speed = 0
        self._last_wall_for_speed = self._wall_t0
        self.root.after(1, self.run_batch)

    # ── GUI update helpers ─────────────────────────────────────────────

    def _refresh_lattice(self) -> None:
        if self.sim is None:
            return
        self._im.set_data(self.sim.lattice_for_display())
        self._ax_lat.set_title(
            f"Step {self.sim.step:,}    Sim time {self.sim.time:.3e} s    T={self.params.T} K",
            color=self._colors["ACC"], fontsize=9, pad=6)

    def _refresh_stats(self) -> None:
        if self.sim is None:
            return
        nf, nd, npass, nt = self.sim.counts()
        self.step_var.set(f"{self.sim.step:,}")
        self.time_var.set(f"{self.sim.time:.3e} s")
        self.free_var.set(str(nf))
        self.dep_var.set(str(nd))
        self.pass_var.set(str(npass))
        self.rate_var.set(f"{self.sim.ftree.total():.3e}")

        # Speed in KMC steps / second (wall clock)
        now = time.perf_counter()
        dt_wall = now - self._last_wall_for_speed
        if dt_wall > 0.5:
            ksteps = (self.sim.step - self._last_step_for_speed) / dt_wall
            self.speed_var.set(f"{ksteps/1000:.1f} k step/s")
            self._last_step_for_speed = self.sim.step
            self._last_wall_for_speed = now

        self._hist_steps.append(self.sim.step)
        self._hist_free.append(nf)
        self._hist_deposited.append(nd)
        self._hist_passivated.append(npass)
        self._line_free.set_data(self._hist_steps, self._hist_free)
        self._line_dep.set_data(self._hist_steps, self._hist_deposited)
        self._line_pass.set_data(self._hist_steps, self._hist_passivated)
        self._ax_cnt.relim()
        self._ax_cnt.autoscale_view()

    # ── Main stepping loop ─────────────────────────────────────────────

    def run_batch(self) -> None:
        if self.sim is None:
            return

        if self.stopped:
            self.sim.finalize_outputs()
            self.run_btn.config(state="normal")
            self._status_var.set("Stopped.")
            messagebox.showinfo("Stopped",
                f"Simulation stopped at step {self.sim.step:,}.\n"
                f"Output saved to: {self.sim.output_dir}")
            return

        if self.paused:
            self.root.after(50, self.run_batch)
            return

        batch = self.sim.p.gui_batch_steps

        # ── Run compiled (or Python) batch ────────────────────────────
        old_step = self.sim.step
        steps_done, finished = self.sim.run_n_steps_fast(batch)

        # ── Save snapshots for any snapshot steps crossed in this batch ─
        if self.sim.p.save_snapshots or self.sim.p.save_npy_states:
            snap_ev = self.sim.p.snapshot_every
            # Determine if any snapshot step fell inside [old_step+1 .. self.sim.step]
            # We use the current lattice (end-of-batch state) for the nearest step.
            first_snap = (old_step // snap_ev + 1) * snap_ev
            if first_snap <= self.sim.step:
                # Save once using the current lattice state; tag it by step
                tag = f"step_{self.sim.step:07d}"
                if self.sim.p.save_snapshots:
                    self.sim.save_snapshot(tag)
                if self.sim.p.save_npy_states:
                    self.sim.save_lattice_npy(tag)

        # ── Update GUI (throttled to ~20 FPS so matplotlib doesn't dominate) ──
        now = time.perf_counter()
        accel = "numba" if NUMBA_AVAILABLE else "Python"
        self._status_var.set(
            f"Running [{accel}] — step {self.sim.step:,}  |  "
            f"sim time {self.sim.time:.3e} s")

        # Only redraw the heavy matplotlib canvas at most once every 50 ms.
        # Stats labels and status bar update every batch (cheap).
        if now - self._last_draw_wall >= 0.05:
            self._refresh_lattice()
            self._refresh_stats()
            self.canvas.draw_idle()
            self._last_draw_wall = now

        if finished:
            # Always do a final full redraw so the end state is visible
            self._refresh_lattice()
            self._refresh_stats()
            self.canvas.draw_idle()
            self.sim.finalize_outputs()
            self.run_btn.config(state="normal")
            self._status_var.set(
                f"Finished at step {self.sim.step:,}  |  "
                f"wall time {time.perf_counter() - self._wall_t0:.1f} s")
            messagebox.showinfo("Done",
                f"Simulation finished at step {self.sim.step:,}.\n"
                f"Output saved to: {self.sim.output_dir}")
            return

        self.root.after(1, self.run_batch)


# ===========================================================================
# Scan mode  (unchanged from v4b)
# ===========================================================================

def parse_scan_file(filepath: str) -> List[KMCParams]:
    field_types = {
        "Nx": int, "Ny": int, "T": float, "d0": float, "e0": float, "e1": float,
        "nu_f": float, "nu_d": float, "nu_p": float, "kB": float,
        "max_steps": int, "max_time": float, "rng_seed": int,
        "periodic_x": int, "log_every": int,
        "save_snapshots": bool, "snapshot_every": int, "save_npy_states": bool,
        "output_dir": str, "history_filename": str,
        "validation_enabled": bool, "validation_every": int,
        "gui_batch_steps": int, "stop_fill_fraction": float,
    }
    swept: dict = {}
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            name, _, raw = line.partition("=")
            name = name.strip();  raw = raw.strip()
            if name not in field_types:
                print(f"  Warning: unknown parameter '{name}', skipping."); continue
            parts = [p.strip() for p in raw.split(",")]
            typ   = field_types[name]
            parsed = []
            for part in parts:
                if part.lower() in ("none", ""):
                    parsed.append(None)
                elif typ == bool:
                    parsed.append(part.lower() in ("true", "1", "yes"))
                else:
                    parsed.append(typ(part))
            swept[name] = parsed
    if not swept:
        raise ValueError("No valid parameters found in scan file.")
    fixed    = {k: v[0] for k, v in swept.items() if len(v) == 1}
    variable = {k: v    for k, v in swept.items() if len(v) > 1}
    if not variable:
        combos = [{}]; keys = []
    else:
        keys   = list(variable.keys())
        combos = [dict(zip(keys, c)) for c in itertools.product(*variable.values())]
    defaults = KMCParams()
    result = []
    for combo in combos:
        merged = {**vars(defaults), **fixed, **combo}
        if "periodic_x" in merged and not isinstance(merged["periodic_x"], bool):
            merged["periodic_x"] = bool(merged["periodic_x"])
        result.append(KMCParams(**{k: v for k, v in merged.items() if k in vars(defaults)}))
    return result


def run_scan(scan_file: str, scan_output_dir: str = "scan_output") -> None:
    print(f"Reading scan file: {scan_file}")
    param_list = parse_scan_file(scan_file)
    print(f"Total runs: {len(param_list)}")
    base_dir = Path(scan_output_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    for run_idx, params in enumerate(param_list):
        run_label = f"T{params.T}_e0{params.e0}_nu_p{params.nu_p}"
        run_dir   = base_dir / run_label
        run_dir.mkdir(parents=True, exist_ok=True)
        # IMPORTANT: set output_dir BEFORE constructing ElectrodepositionKMC,
        # because __init__ reads it immediately to create snapshot_dir.
        params.output_dir    = str(run_dir)
        params.save_snapshots  = True
        params.save_npy_states = True
        print(f"\n[{run_idx + 1}/{len(param_list)}] {run_label}")
        try:
            t0  = time.time()
            sim = ElectrodepositionKMC(params)
            sim.run_cli()
            wall_time = time.time() - t0
        except Exception as exc:
            print(f"  ERROR in run {run_idx + 1}: {exc}")
            import traceback; traceback.print_exc()
            continue   # skip to next run; don't let one bad run kill the whole scan

        n_free, n_dep, n_pass, n_total = sim.counts()
        total_sites = params.Nx * (params.Ny - 1)
        fill = (n_dep + n_pass) / total_sites

        # Determine why the simulation stopped.
        if params.stop_fill_fraction is not None and fill >= params.stop_fill_fraction:
            stop_reason = "fill_fraction"
        elif sim.step >= params.max_steps:
            stop_reason = "max_steps"
        elif sim.time >= params.max_time:
            stop_reason = "max_time"
        else:
            stop_reason = "no_events"   # rate sum hit zero

        # Pull the final total_rate from the last history record (already computed).
        final_total_rate = sim.history[-1]["total_rate"] if sim.history else float("nan")

        summary_rows.append({
            # --- run identity ---
            "run":              run_idx + 1,
            "label":            run_label,
            # --- input parameters ---
            "Nx":               params.Nx,
            "Ny":               params.Ny,
            "T":                params.T,
            "e0":               params.e0,
            "e1":               params.e1,
            "d0":               params.d0,
            "nu_f":             params.nu_f,
            "nu_d":             params.nu_d,
            "nu_p":             params.nu_p,
            "periodic_x":       int(params.periodic_x),
            "rng_seed":         params.rng_seed,
            # --- stop conditions used ---
            "max_steps":        params.max_steps,
            "max_time":         params.max_time,
            "stop_fill_frac":   params.stop_fill_fraction,
            # --- final simulation state ---
            "final_step":       sim.step,
            "final_time":       sim.time,
            "stop_reason":      stop_reason,
            "n_free":           n_free,
            "n_deposited":      n_dep,
            "n_passivated":     n_pass,
            "n_total_atoms":    n_total,
            "total_sites":      total_sites,
            "fill_fraction":    round(fill, 6),
            "final_total_rate": round(final_total_rate, 4),
            # --- performance ---
            "wall_time_s":      round(wall_time, 2),
        })

        print(f"  Done | stop={stop_reason} | step={sim.step:,} | "
              f"sim_t={sim.time:.3e}s | wall={wall_time:.1f}s | "
              f"fill={fill:.1%} (dep={n_dep} pass={n_pass} free={n_free})")

    # Guard: if every run crashed, report clearly instead of crashing here.
    if not summary_rows:
        print("\nERROR: No runs completed successfully. Check error messages above.")
        return

    # Write summary CSV -- one row per completed run.
    summary_path = base_dir / "scan_summary.csv"
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    n_ok = len(summary_rows)
    n_total_runs = len(param_list)
    print(f"\nScan complete: {n_ok}/{n_total_runs} runs succeeded.")
    print(f"Summary CSV: {summary_path}")

# -----------------------------------------------------------------------------
# Main entry point
# -----------------------------------------------------------------------------
USE_GUI  = False   # Set to False for CLI or scan mode (want GUI)
USE_SCAN = True   # Set to True to run a parameter scan (want SCAN)
SCAN_FILE = "scan_params.txt"   # Path to your scan parameter file

if __name__ == "__main__":
    if USE_SCAN:
        run_scan(SCAN_FILE, scan_output_dir="scan_output")
    elif USE_GUI:
        root = tk.Tk()
        app = SimulationGUI(root)
        root.mainloop()
    else:
        params = KMCParams(
            Nx=40, Ny=25, T=300.0, d0=1.0e3, e0=-0.2, e1=-0.5,
            nu_f=5.0e9, nu_d=1.0e9, nu_p=1.0e3,
            max_steps=400000, max_time=100.0, rng_seed=24153,
            log_every=1000, snapshot_every=10000,
            save_snapshots=True, save_npy_states=True,
            validation_enabled=True, validation_every=5000
        )
        sim = ElectrodepositionKMC(params)
        t0 = time.time()
        sim.run_cli()
        t1 = time.time()
        print(f"Simulation complete at step {sim.step}, "
              f"time {sim.time:.4e}, wall time {t1 - t0:.2f} s")