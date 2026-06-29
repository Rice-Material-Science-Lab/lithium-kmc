from __future__ import annotations  # Allow forward references in type hints.

import csv  # Write history data to CSV files.
import math  # Use log/exp and basic math helpers.
import time  # Small sleep inside the GUI stepping loop when needed.
import tkinter as tk  # Standard GUI toolkit bundled with Python.
from collections import deque  # Efficient queue for iterative local relaxation.
from dataclasses import dataclass  # Convenient container for simulation parameters.
from pathlib import Path  # Safer path handling than raw strings.
from tkinter import messagebox, ttk  # Common Tkinter widgets and pop-up dialogs.
from typing import Deque, List, Optional, Sequence, Tuple  # Type annotations.

import matplotlib.pyplot as plt  # Used for saving lattice snapshots.
import numpy as np  # Array operations and random number generation.
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # Embed plots in Tkinter.
from matplotlib.colors import ListedColormap  # Map integer lattice states to colors.
from matplotlib.figure import Figure  # Matplotlib figure object for GUI embedding.


# -----------------------------------------------------------------------------
# Lattice state codes
# -----------------------------------------------------------------------------
# We keep the integer encoding explicit so the lattice can be stored as a compact
# NumPy integer array while still remaining easy to interpret.
EMPTY = 0       # Empty lattice site.
FREE = 1        # Mobile atom not currently bonded to substrate/deposit.
DEPOSITED = 2   # Atom bonded to substrate or another deposited atom.
SUBSTRATE = 3   # Fixed bottom substrate row.
PASSIVATED = 4  # Immobile atom created from a deposited atom.

# Type alias used throughout the code for lattice coordinates.
Coord = Tuple[int, int]


@dataclass
class KMCParams:
    """All user-controlled simulation parameters."""

    # Lattice dimensions in x and y.
    Nx: int = 40
    Ny: int = 25

    # Temperature in K.
    T: float = 300.0

    # Atom dropping rate for each available top-row site.
    d0: float = 1.0e3

    # Interaction energies.
    # e0: atom-atom interaction energy.
    # e1: atom-substrate interaction energy.
    # This script leaves the sign convention flexible: users may choose positive
    # or negative values, but they should do so consistently with the hop-rate
    # formula they intend to use.
    e0: float = -0.2
    e1: float = -0.5

    # Attempt frequencies for hopping.
    # nu_f applies when the moving atom is FREE.
    # nu_d applies when the moving atom is DEPOSITED.
    nu_f: float = 5.0e9
    nu_d: float = 1.0e9
    nu_p: float = 1.0e2

    # Boltzmann constant in eV/K.
    kB: float = 8.617333262145e-5

    # End conditions.
    max_steps: int = 400000
    max_time: float = 100.0

    # Random seed for reproducibility.
    rng_seed: Optional[int] = 394583

    # Boundary condition in x.
    periodic_x: bool = True

    # How often to append to history and refresh GUI display.
    log_every: int = 1000

    # Snapshot controls.
    save_snapshots: bool = True
    snapshot_every: int = 10000
    save_npy_states: bool = True

    # Output paths.
    output_dir: str = "kmc_output"
    history_filename: str = "time_series.csv"

    # Validation option: when True, periodically compare the local-update event
    # table against a fresh full rebuild.
    validation_enabled: bool = False
    validation_every: int = 1000

    # GUI stepping: number of KMC events to execute per Tkinter callback.
    gui_batch_steps: int = 200

    # Stop when deposited+passivated atoms reach this fraction of non-substrate sites.
    # Set to None to disable. Example: 0.20 means stop at 20% fill.
    stop_fill_fraction: Optional[float] = None


class FenwickTree:
    """Binary Indexed Tree for fast cumulative-rate updates and sampling."""

    def __init__(self, size: int):
        # Fenwick trees are usually stored 1-indexed, so allocate size+1.
        self.size = size
        self.tree = np.zeros(size + 1, dtype=float)


    def update(self, idx: int, delta: float) -> None:
        """Add delta to element idx."""
        # Convert the public 0-based index to internal 1-based indexing.
        i = idx + 1

        # Propagate the delta upward through the Fenwick tree.
        while i <= self.size:
            self.tree[i] += delta
            i += i & -i

    def total(self) -> float:
        """Return the total sum stored in the tree."""
        # Prefix sum up to the last element equals the total rate.
        #
        # In exact arithmetic, this matches the dense sum of all event rates.
        # In floating-point arithmetic, the accumulation order can differ slightly
        # from a dense sum, so tiny discrepancies on the order of ~1e-9 to ~1e-8
        # are normal after many updates.
        i = self.size
        s = 0.0
        while i > 0:
            s += self.tree[i]
            i -= i & -i
        return s

    def find_prefix_index(self, target: float) -> int:
        """Return smallest 0-based index whose prefix sum is >= target."""
        # This is the standard Fenwick-tree search routine.
        # We walk down powers of two from large to small.
        idx = 0

        # Highest power of two not exceeding self.size.
        bit = 1
        while bit < self.size:
            bit <<= 1
        bit >>= 1

        # Descend through the tree.
        while bit > 0:
            nxt = idx + bit
            if nxt <= self.size and self.tree[nxt] < target:
                target -= self.tree[nxt]
                idx = nxt
            bit >>= 1

        # idx is 1-based predecessor; convert to 0-based event index.
        return idx


class ElectrodepositionKMC:
    """Kinetic Monte Carlo simulator for the 2D lattice electrodeposition model."""

    def __init__(self, params: KMCParams):
        # Store user parameters.
        self.p = params

        # Create the random-number generator once so the whole simulation is reproducible.
        self.rng = np.random.default_rng(self.p.rng_seed)

        # Basic input checks.
        if self.p.Nx < 1:
            raise ValueError("Nx must be at least 1.")
        if self.p.Ny < 2:
            raise ValueError("Ny must be at least 2 so one non-substrate row exists.")
        if self.p.T <= 0:
            raise ValueError("Temperature must be positive.")
        if self.p.log_every < 1:
            raise ValueError("log_every must be at least 1.")
        if self.p.snapshot_every < 1:
            raise ValueError("snapshot_every must be at least 1.")
        if self.p.validation_every < 1:
            raise ValueError("validation_every must be at least 1.")
        if self.p.gui_batch_steps < 1:
            raise ValueError("gui_batch_steps must be at least 1.")

        # Create the lattice as a Ny-by-Nx integer array.
        self.lattice = np.zeros((self.p.Ny, self.p.Nx), dtype=np.int8)

        # Set the entire bottom row to be substrate.
        self.lattice[0, :] = SUBSTRATE

        # Build a compact 5x5 interaction lookup table.
        # Unspecified interactions remain zero.
        self.energy_lookup = np.zeros((5, 5), dtype=float)
        self.energy_lookup[DEPOSITED, DEPOSITED] = self.p.e0
        self.energy_lookup[DEPOSITED, SUBSTRATE] = self.p.e1
        self.energy_lookup[SUBSTRATE, DEPOSITED] = self.p.e1
        self.energy_lookup[SUBSTRATE, SUBSTRATE] = self.p.e1
        self.energy_lookup[PASSIVATED, DEPOSITED] = self.p.e0
        self.energy_lookup[DEPOSITED, PASSIVATED] = self.p.e0
        self.energy_lookup[PASSIVATED, PASSIVATED] = self.p.e0
        self.energy_lookup[PASSIVATED, SUBSTRATE] = self.p.e1
        self.energy_lookup[SUBSTRATE, PASSIVATED] = self.p.e1

        # Precompute event indexing.
        # We store all top-row drop events first, then all hop events, then one
        # passivation event per lattice site.
        self.num_drop_events = self.p.Nx
        self.num_hop_events = self.p.Nx * self.p.Ny * 4
        self.num_passivation_events = self.p.Nx * self.p.Ny
        self.max_events = self.num_drop_events + self.num_hop_events + self.num_passivation_events

        # event_rates[idx] holds the current rate of event idx.
        self.event_rates = np.zeros(self.max_events, dtype=float)

        # The Fenwick tree stores the same rates but in a structure suitable for
        # fast cumulative summation and sampling.
        self.ftree = FenwickTree(self.max_events)

        # idx_to_data[idx] tells us what physical event the idx-th event represents.
        # Format:
        #   ("drop", None, (x, y))
        #   ("hop", (x0, y0), (x1, y1))
        #   ("passivate", (x, y), (x, y))
        self.idx_to_data: List[Tuple[str, Optional[Coord], Coord]] = [None] * self.max_events

        # Build the mapping between event indices and physical events.
        self._setup_indices()

        # Simulation time and KMC step counter.
        self.time = 0.0
        self.step = 0

        # Prepare output directories.
        self.output_dir = Path(self.p.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.snapshot_dir = self.output_dir / "snapshots"
        if self.p.save_snapshots:
            self.snapshot_dir.mkdir(parents=True, exist_ok=True)

        # Create a small color map for plotting integer lattice states.
        self._cmap = ListedColormap(["white", "tab:blue", "tab:orange", "black", "tab:green"])

        # GUI-related figure handles are initialized lazily only when needed.
        self._fig = None
        self._ax = None
        self._im = None
        self._title = None

        # History rows are stored as dictionaries, then written to CSV at the end.
        self.history = []

        # Build the initial full event table and rate tree.
        self.rebuild_all_rates()

        # Save the starting state and history entry.
        self.record_history(label="initial")
        if self.p.save_snapshots:
            self.save_snapshot("initial")
        if self.p.save_npy_states:
            self.save_lattice_npy("initial")

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------
    def wrap_x(self, x: int) -> Optional[int]:
        """Apply x-periodicity if enabled; otherwise return None when out of bounds."""
        if self.p.periodic_x:
            return x % self.p.Nx
        if 0 <= x < self.p.Nx:
            return x
        return None

    def valid_neighbor_coords(self, x: int, y: int) -> List[Coord]:
        """Return existing nearest neighbors of site (x, y)."""
        nbrs: List[Coord] = []
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            xx = self.wrap_x(x + dx)
            yy = y + dy
            if xx is None:
                continue
            if 0 <= yy < self.p.Ny:
                nbrs.append((xx, yy))
        return nbrs

    def radius2_sites(self, seeds: Sequence[Coord]) -> List[Coord]:
        """Return unique sites within Manhattan distance <= 2 from any seed."""
        # This larger local region is used when refreshing rates because changing a
        # site can affect events whose source or destination is up to two hops away.
        seen = set()
        out: List[Coord] = []

        for x0, y0 in seeds:
            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    if abs(dx) + abs(dy) > 2:
                        continue
                    xx = self.wrap_x(x0 + dx)
                    yy = y0 + dy
                    if xx is None:
                        continue
                    if not (0 <= yy < self.p.Ny):
                        continue
                    if (xx, yy) not in seen:
                        seen.add((xx, yy))
                        out.append((xx, yy))
        return out

    # ------------------------------------------------------------------
    # Event indexing helpers
    # ------------------------------------------------------------------
    def _setup_indices(self) -> None:
        """Populate idx_to_data with all drop, hop, and passivation event definitions."""
        # Drop events exist only for the top row.
        top_y = self.p.Ny - 1
        for x in range(self.p.Nx):
            self.idx_to_data[x] = ("drop", None, (x, top_y))

        # Hop events exist for every lattice site and every cardinal direction.
        # We still define some events that will always have zero rate, such as hops
        # from substrate sites or hops that leave the domain in nonperiodic x.
        base = self.num_drop_events
        directions = ((1, 0), (-1, 0), (0, 1), (0, -1))
        for y in range(self.p.Ny):
            for x in range(self.p.Nx):
                site_offset = (y * self.p.Nx + x) * 4
                for i, (dx, dy) in enumerate(directions):
                    idx = base + site_offset + i
                    self.idx_to_data[idx] = ("hop", (x, y), (x + dx, y + dy))

        # Passivation events exist for every lattice site.
        base = self.num_drop_events + self.num_hop_events
        for y in range(self.p.Ny):
            for x in range(self.p.Nx):
                idx = base + y * self.p.Nx + x
                self.idx_to_data[idx] = ("passivate", (x, y), (x, y))

    def drop_index(self, x: int) -> int:
        """Return event-table index of the drop event at top-row column x."""
        return x

    def hop_base_index(self, x: int, y: int) -> int:
        """Return index of the first hop event for source site (x, y)."""
        return self.num_drop_events + (y * self.p.Nx + x) * 4

    def passivation_index(self, x: int, y: int) -> int:
        """Return event-table index of the passivation event at site (x, y)."""
        return self.num_drop_events + self.num_hop_events + y * self.p.Nx + x

    # ------------------------------------------------------------------
    # Energetics and event rates
    # ------------------------------------------------------------------
    def calc_local_energy(self, x: int, y: int, atom_type: int) -> float:
        """Compute the local pair interaction energy for a hypothetical atom."""
        e = 0.0
        for nx, ny in self.valid_neighbor_coords(x, y):
            nbr_type = int(self.lattice[ny, nx])
            e += self.energy_lookup[atom_type, nbr_type]
        return e

    def site_has_empty_neighbor(self, x: int, y: int) -> bool:
        """Return True when site (x, y) has at least one empty nearest neighbor."""
        for nx, ny in self.valid_neighbor_coords(x, y):
            if int(self.lattice[ny, nx]) == EMPTY:
                return True
        return False

    def desired_mobile_state_at_site(self, x: int, y: int) -> int:
        """Return the bonded/free state a mobile atom should have at (x, y)."""
        for nx, ny in self.valid_neighbor_coords(x, y):
            if int(self.lattice[ny, nx]) in (DEPOSITED, SUBSTRATE):
                return DEPOSITED
        return FREE

    def get_event_rate(self, kind: str, src: Optional[Coord], dst: Coord) -> float:
        """Return the current rate of a single drop, hop, or passivation event."""
        if kind == "drop":
            # A drop event is active only if the top-row site is empty.
            x1, y1 = dst
            if self.lattice[y1, x1] == EMPTY:
                return self.p.d0
            return 0.0

        if kind == "passivate":
            x0, y0 = src
            if int(self.lattice[y0, x0]) != DEPOSITED:
                return 0.0
            if self.site_has_empty_neighbor(x0, y0):
                return self.p.nu_p
            return 0.0

        # For a hop event, first check whether the source holds a mobile atom.
        x0, y0 = src
        atom_type = int(self.lattice[y0, x0])
        if atom_type not in (FREE, DEPOSITED):
            return 0.0

        # Resolve x-boundary handling at the destination.
        x1_raw, y1 = dst
        x1 = self.wrap_x(x1_raw)
        if x1 is None:
            return 0.0

        # Check y bounds and that the destination site is empty.
        if not (0 <= y1 < self.p.Ny):
            return 0.0
        if self.lattice[y1, x1] != EMPTY:
            return 0.0

        # Pick the appropriate attempt frequency based on the current atom state.
        nu = self.p.nu_f if atom_type == FREE else self.p.nu_d

        # Compute initial local energy of the moving atom at the source.
        e_init = self.calc_local_energy(x0, y0, atom_type)

        # Temporarily remove the atom from the source, then determine what mobile
        # state that atom would have after landing at the destination before
        # evaluating its final energy there.
        self.lattice[y0, x0] = EMPTY
        final_atom_type = self.desired_mobile_state_at_site(x1, y1)
        e_final = self.calc_local_energy(x1, y1, final_atom_type)
        self.lattice[y0, x0] = atom_type

        # Apply the user-specified hopping expression.
        return nu * math.exp(-(e_final - e_init) / (2.0 * self.p.kB * self.p.T))

    def update_rate_at_index(self, idx: int) -> None:
        """Recompute one event rate and propagate its change into the Fenwick tree."""
        kind, src, dst = self.idx_to_data[idx]
        new_rate = self.get_event_rate(kind, src, dst)
        delta = new_rate - self.event_rates[idx]
        if abs(delta) > 1.0e-18:
            self.event_rates[idx] = new_rate
            self.ftree.update(idx, delta)

    def rebuild_all_rates(self) -> None:
        """Full recomputation of all event rates."""
        # Reset both the dense array and the Fenwick tree.
        self.event_rates[:] = 0.0
        self.ftree = FenwickTree(self.max_events)

        # Recompute every event from scratch.
        for idx in range(self.max_events):
            self.update_rate_at_index(idx)

    def refresh_local_rates(self, changed_sites: Sequence[Coord]) -> None:
        """Refresh rates only in a radius-2 neighborhood of changed sites."""
        targets = self.radius2_sites(changed_sites)

        for x, y in targets:
            # Refresh the drop event only if this site is on the top row.
            if y == self.p.Ny - 1:
                self.update_rate_at_index(self.drop_index(x))

            # Refresh all four hop directions whose source is this site.
            base = self.hop_base_index(x, y)
            for offset in range(4):
                self.update_rate_at_index(base + offset)

            # Refresh the passivation event for this site.
            self.update_rate_at_index(self.passivation_index(x, y))

    # ------------------------------------------------------------------
    # Bonding-state updates
    # ------------------------------------------------------------------
    def desired_bond_state(self, x: int, y: int) -> int:
        """Return what state a mobile atom at (x, y) should have after bookkeeping."""
        site_type = int(self.lattice[y, x])
        if site_type not in (FREE, DEPOSITED):
            return site_type

        return self.desired_mobile_state_at_site(x, y)

    def update_bonding_relaxation(self, seeds: Sequence[Coord]) -> List[Coord]:
        """Iteratively relax FREE/DEPOSITED states near seeds until no further changes occur."""
        # We use a queue because changing one atom may change the desired state of
        # its neighbors, which can in turn change the state of their neighbors.
        q: Deque[Coord] = deque()
        queued = set()
        changed = set()

        # Start from each seed and its nearest neighbors.
        initial_sites = set()
        for sx, sy in seeds:
            initial_sites.add((sx, sy))
            for nbr in self.valid_neighbor_coords(sx, sy):
                initial_sites.add(nbr)

        for site in initial_sites:
            q.append(site)
            queued.add(site)

        # Process the queue until the local region is fully relaxed.
        while q:
            x, y = q.popleft()
            queued.discard((x, y))

            # Ignore non-mobile sites.
            current = int(self.lattice[y, x])
            if current not in (FREE, DEPOSITED):
                continue

            # Compute the state the atom should have under the local bonding rule.
            desired = self.desired_bond_state(x, y)

            # If no change is needed, move on.
            if desired == current:
                continue

            # Apply the state change.
            self.lattice[y, x] = desired
            changed.add((x, y))

            # Because this site changed, its neighbors may now need reevaluation.
            for nbr in self.valid_neighbor_coords(x, y):
                if nbr not in queued:
                    q.append(nbr)
                    queued.add(nbr)

            # Re-enqueue this site too, although in this simple two-state model it
            # usually stabilizes immediately. Keeping it here makes the routine more robust.
            if (x, y) not in queued:
                q.append((x, y))
                queued.add((x, y))

        return list(changed)

    # ------------------------------------------------------------------
    # KMC step
    # ------------------------------------------------------------------
    def execute_step(self) -> bool:
        """Execute one rejection-free KMC step. Return False when no event exists."""
        # Get the total event rate.
        r_tot = self.ftree.total()

        # If no events are available, stop the simulation.
        if r_tot <= 0.0:
            return False

        # Draw the KMC time increment.
        # Guard against the zero-edge case by clipping the random number away from 0.
        u1 = max(float(self.rng.random()), 1.0e-15)
        dt = -math.log(u1) / r_tot

        # Draw a second random number to select which event occurs.
        # We sample target in (0, r_tot], not [0, r_tot), to avoid the exact-zero issue.
        u2 = max(float(self.rng.random()), 1.0e-15)
        target = u2 * r_tot
        idx = self.ftree.find_prefix_index(target)

        # Decode the chosen event.
        kind, src, dst = self.idx_to_data[idx]

        # Track sites whose occupancy changed directly.
        directly_changed: List[Coord] = []

        if kind == "drop":
            # Place a free atom at the top-row destination.
            x1, y1 = dst
            self.lattice[y1, x1] = FREE
            directly_changed.append((x1, y1))
        elif kind == "passivate":
            x0, y0 = src
            self.lattice[y0, x0] = PASSIVATED
            directly_changed.append((x0, y0))
        else:
            # Move the atom from source to destination.
            x0, y0 = src
            x1_raw, y1 = dst
            x1 = self.wrap_x(x1_raw)
            atom_type = int(self.lattice[y0, x0])
            self.lattice[y0, x0] = EMPTY
            self.lattice[y1, x1] = atom_type
            directly_changed.append((x0, y0))
            directly_changed.append((x1, y1))

        # Iteratively relax local FREE/DEPOSITED states after the move.
        relaxed_changed = self.update_bonding_relaxation(directly_changed)

        # Refresh rates in a larger neighborhood around everything that changed.
        all_changed = list({*directly_changed, *relaxed_changed})
        self.refresh_local_rates(all_changed)

        # Advance simulation time and step count only after the event is fully applied.
        self.time += dt
        self.step += 1

        # Optionally verify that the local refresh matches a full rebuild.
        if self.p.validation_enabled and self.step % self.p.validation_every == 0:
            self.validate_against_full_rebuild()

        return True

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------
    def validate_against_full_rebuild(self) -> None:
        """Compare current locally updated rates against a clean full rebuild."""
        # Save current state.
        saved_lattice = self.lattice.copy()
        saved_rates = self.event_rates.copy()

        # Rebuild on a clean copy.
        self.rebuild_all_rates()
        rebuilt_rates = self.event_rates.copy()

        # Compare the per-event rate arrays.
        #
        # This is the key validation check. If every event rate agrees with a fresh
        # full rebuild, then the local refresh logic is correct.
        #
        # We intentionally do NOT compare self.ftree.total() against the rebuilt total
        # here. The Fenwick tree and a dense-array sum accumulate floating-point values
        # in different orders, so tiny discrepancies can appear even when every single
        # event rate is correct. Those roundoff-level differences should not cause the
        # validation test to fail.
        if not np.allclose(saved_rates, rebuilt_rates, rtol=1e-12, atol=1e-12):
            diff = np.max(np.abs(saved_rates - rebuilt_rates))
            raise RuntimeError(
                f"Validation failed: local-update rates differ from full rebuild. max diff={diff}"
            )

        # Restore saved state exactly.
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
        """Return counts of free, deposited, passivated, and total non-substrate atoms."""
        n_free = int(np.count_nonzero(self.lattice == FREE))
        n_dep = int(np.count_nonzero(self.lattice == DEPOSITED))
        n_pass = int(np.count_nonzero(self.lattice == PASSIVATED))
        return n_free, n_dep, n_pass, n_free + n_dep + n_pass

    def record_history(self, label: str = "regular") -> None:
        """Append one history record to the in-memory log."""
        n_free, n_dep, n_pass, n_total = self.counts()
        self.history.append(
            {
                "label": label,
                "step": self.step,
                "time": self.time,
                "free": n_free,
                "deposited": n_dep,
                "passivated": n_pass,
                "total_atoms": n_total,
                # Use a dense sum here because it reflects the actual event-rate array
                # directly and avoids tiny order-of-summation differences from the
                # Fenwick tree total.
                "total_rate": float(np.sum(self.event_rates, dtype=float)),
            }
        )

    def write_history_csv(self) -> Path:
        """Write the complete history list to CSV and return the file path."""
        out_path = self.output_dir / self.p.history_filename
        with out_path.open("w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "label",
                    "step",
                    "time",
                    "free",
                    "deposited",
                    "passivated",
                    "total_atoms",
                    "total_rate",
                ],
            )
            writer.writeheader()
            writer.writerows(self.history)
        return out_path

    def lattice_for_display(self) -> np.ndarray:
        """Return a vertically flipped lattice so the top row appears at the top."""
        return np.flipud(self.lattice)

    def save_snapshot(self, tag: str) -> Path:
        """Save a high-quality labeled PNG image of the lattice."""
        out_path = self.snapshot_dir / f"{tag}.png"

        # Scale up so each lattice cell is clearly visible (at least 12px per cell).
        cell_px = max(12, min(24, 800 // max(self.p.Nx, self.p.Ny)))
        fig_w = max(6, self.p.Nx * cell_px / 80)
        fig_h = max(4, self.p.Ny * cell_px / 80)

        fig, ax = plt.subplots(figsize=(fig_w + 2, fig_h + 1.5), dpi=100)
        im = ax.imshow(
            self.lattice_for_display(),
            cmap=self._cmap,
            vmin=0,
            vmax=4,
            interpolation="nearest",   # crisp pixel edges, no blurring
            aspect="equal",
        )

        # Colorbar legend with state labels.
        cbar = fig.colorbar(im, ax=ax, ticks=[0, 1, 2, 3, 4], fraction=0.046, pad=0.04)
        cbar.ax.set_yticklabels(["Empty", "Free", "Deposited", "Substrate", "Passivated"])
        cbar.ax.tick_params(labelsize=9)

        # Axis labels and title.
        ax.set_xlabel("x  (lattice site)", fontsize=10)
        ax.set_ylabel("y  (lattice site)", fontsize=10)
        ax.set_title(
            f"LKMC Electrodeposition — {tag}\n"
            f"Step {self.step:,}    Sim time {self.time:.3e} s    "
            f"T = {self.p.T} K    Nx={self.p.Nx}  Ny={self.p.Ny}",
            fontsize=9,
            pad=8,
        )

        fig.tight_layout()
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        return out_path

    def save_lattice_npy(self, tag: str) -> Path:
        """Save the raw lattice array as a NumPy binary file."""
        out_path = self.output_dir / f"lattice_{tag}.npy"
        np.save(out_path, self.lattice)
        return out_path

    # ------------------------------------------------------------------
    # GUI helpers
    # ------------------------------------------------------------------
    def init_gui_figure(self) -> None:
        """Create the Matplotlib figure used in the Tkinter GUI."""
        self._fig = Figure(figsize=(6, 5))
        self._ax = self._fig.add_subplot(111)
        self._im = self._ax.imshow(self.lattice_for_display(), cmap=self._cmap, vmin=0, vmax=4)
        self._title = self._ax.set_title(f"Step {self.step}, Time {self.time:.2e}")
        self._ax.set_xlabel("x")
        self._ax.set_ylabel("y")

    def refresh_gui_figure(self) -> None:
        """Update the GUI plot after some number of KMC steps."""
        if self._im is None:
            return
        self._im.set_data(self.lattice_for_display())
        self._title.set_text(f"Step {self.step}, Time {self.time:.2e}")

    # ------------------------------------------------------------------
    # Run helpers
    # ------------------------------------------------------------------
    def finalize_outputs(self) -> None:
        """Write final records and files at the end of a run."""
        self.record_history(label="final")
        if self.p.save_snapshots:
            self.save_snapshot(f"final_step_{self.step}")
        if self.p.save_npy_states:
            self.save_lattice_npy(f"final_step_{self.step}")
        self.write_history_csv()

    def run_cli(self) -> None:
        """Run the simulation in command-line mode."""
        total_sites = self.p.Nx * (self.p.Ny - 1)  # exclude substrate row
        while self.step < self.p.max_steps and self.time < self.p.max_time:
            if not self.execute_step():
                break

            # NEW: stop when fill fraction is reached
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


class SimulationGUI:
    """Tkinter front end that drives the simulation in small batches with root.after()."""

    # Human-readable labels and tooltips for each parameter field.
    FIELD_SPECS = [
        # (param_name, type, default_attr, label, tooltip)
        # --- Lattice geometry ---
        ("Nx",            int,   "Nx",            "Lattice width (Nx)",        "Number of lattice sites in x direction"),
        ("Ny",            int,   "Ny",            "Lattice height (Ny)",       "Number of lattice sites in y direction (row 0 = substrate)"),
        # --- Physics ---
        ("T",             float, "T",             "Temperature (K)",           "Simulation temperature in Kelvin"),
        ("d0",            float, "d0",            "Drop rate (d₀)",            "Atom dropping rate per empty top-row site (s⁻¹)"),
        ("e0",            float, "e0",            "Bonded energy e₀ (eV)",     "Interaction energy for DEPOSITED/PASSIVATED bonded neighbors (eV)"),
        ("e1",            float, "e1",            "Atom–substrate e₁ (eV)",    "Interaction energy between atom and substrate (eV)"),
        ("nu_f",          float, "nu_f",          "Free attempt freq ν_f",     "Attempt frequency for FREE atoms hopping (s⁻¹)"),
        ("nu_d",          float, "nu_d",          "Dep. attempt freq ν_d",     "Attempt frequency for DEPOSITED atoms hopping (s⁻¹)"),
        ("nu_p",          float, "nu_p",          "Passivation rate ν_p",      "Rate for DEPOSITED atoms with an EMPTY neighbor to become PASSIVATED (s⁻¹); default 1e3"),
        # --- Run controls ---
        ("max_steps",     int,   "max_steps",     "Max steps",                 "Stop simulation after this many KMC steps"),
        ("max_time",      float, "max_time",      "Max sim time (s)",          "Stop simulation after this much simulated time"),
        ("log_every",     int,   "log_every",     "Log interval (steps)",      "Record history every N steps"),
        ("snapshot_every",int,   "snapshot_every","Snapshot interval (steps)", "Save PNG snapshot every N steps"),
        # --- Misc ---
        ("periodic_x",    int,   None,            "Periodic x  (1=yes)",       "Use periodic boundary conditions in x"),
        ("rng_seed",      int,   "rng_seed",      "Random seed",               "Seed for the random number generator (reproducibility)"),
    ]

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("LKMC — Lattice KMC Electrodeposition")
        self.root.configure(bg="#1e1e2e")

        self.params = KMCParams()
        self.sim: Optional[ElectrodepositionKMC] = None
        self.paused  = False
        self.stopped = False
        self.canvas: Optional[FigureCanvasTkAgg] = None

        # History for the live atom-count chart.
        self._hist_steps:     List[int]   = []
        self._hist_free:      List[int]   = []
        self._hist_deposited: List[int]   = []
        self._hist_passivated: List[int]  = []

        self._setup_styles()
        self._build_ui()

    # ------------------------------------------------------------------
    # Styling
    # ------------------------------------------------------------------
    def _setup_styles(self) -> None:
        style = ttk.Style()
        style.theme_use("clam")

        BG   = "#1e1e2e"
        PANEL= "#2a2a3e"
        ACC  = "#89b4fa"   # soft blue accent
        FG   = "#cdd6f4"
        ENTRY= "#313244"

        style.configure(".",              background=BG,    foreground=FG,   font=("Segoe UI", 9))
        style.configure("TFrame",        background=BG)
        style.configure("TLabel",        background=BG,    foreground=FG)
        style.configure("TLabelframe",   background=PANEL, foreground=ACC,  relief="flat", borderwidth=1)
        style.configure("TLabelframe.Label", background=PANEL, foreground=ACC, font=("Segoe UI", 9, "bold"))
        style.configure("TEntry",        fieldbackground=ENTRY, foreground=FG, insertcolor=FG)
        style.configure("TCheckbutton",  background=PANEL, foreground=FG)
        style.configure("Header.TLabel", background=BG,    foreground=ACC,  font=("Segoe UI", 13, "bold"))
        style.configure("Sub.TLabel",    background=BG,    foreground="#6c7086", font=("Segoe UI", 8))
        style.configure("Stat.TLabel",   background=PANEL, foreground=FG,   font=("Consolas", 9))
        style.configure("StatVal.TLabel",background=PANEL, foreground=ACC,  font=("Consolas", 9, "bold"))

        # Buttons
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

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        C = self._colors

        # ── Top header bar ──────────────────────────────────────────────
        header = ttk.Frame(self.root)
        header.pack(side="top", fill="x", padx=12, pady=(10, 4))
        ttk.Label(header, text="LKMC Electrodeposition Simulator", style="Header.TLabel").pack(side="left")
        ttk.Label(header, text="Lattice Kinetic Monte Carlo  ·  2D Electrodeposition Model",
                  style="Sub.TLabel").pack(side="left", padx=(10, 0))

        # ── Main body ────────────────────────────────────────────────────
        body = ttk.Frame(self.root)
        body.pack(fill="both", expand=True, padx=12, pady=4)

        # Left column: params + controls + stats
        left = ttk.Frame(body)
        left.pack(side="left", fill="y", padx=(0, 8))

        self._build_param_panel(left)
        self._build_control_panel(left)
        self._build_stats_panel(left)

        # Right column: lattice view + atom-count chart
        right = ttk.Frame(body)
        right.pack(side="left", fill="both", expand=True)

        self._build_plot_area(right)

    def _build_param_panel(self, parent: ttk.Frame) -> None:
        """Parameter entry grid, grouped by category."""
        C = self._colors

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
                if attr is not None:
                    default_val = str(getattr(self.params, attr))
                else:
                    # periodic_x is a bool on the dataclass
                    default_val = "1" if self.params.periodic_x else "0"

                lbl = ttk.Label(frame, text=label, anchor="w")
                lbl.grid(row=row_i, column=0, sticky="w", padx=(8, 4), pady=2)

                entry = ttk.Entry(frame, width=13)
                entry.insert(0, default_val)
                entry.grid(row=row_i, column=1, padx=(0, 8), pady=2)

                # Bind tooltip (simple title-bar style).
                entry.bind("<Enter>", lambda e, t=tooltip: self.root.title(
                    f"LKMC  ·  {t}"))
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

        frame.columnconfigure(0, weight=1)
        frame.columnconfigure(1, weight=1)

    def _build_stats_panel(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="  Live Statistics  ")
        frame.pack(fill="x", pady=(0, 6))

        stats = [
            ("Step",      "step_var"),
            ("Sim time",  "time_var"),
            ("Free atoms","free_var"),
            ("Deposited", "dep_var"),
            ("Passivated","pass_var"),
            ("Total rate","rate_var"),
        ]

        for i, (label, attr) in enumerate(stats):
            var = tk.StringVar(value="—")
            setattr(self, attr, var)
            ttk.Label(frame, text=f"{label}:", style="Stat.TLabel", width=12, anchor="w").grid(
                row=i, column=0, padx=(8, 2), pady=1, sticky="w")
            ttk.Label(frame, textvariable=var, style="StatVal.TLabel", width=14, anchor="e").grid(
                row=i, column=1, padx=(0, 8), pady=1, sticky="e")

        # Legend
        legend_frame = ttk.Frame(frame, style="TFrame")
        legend_frame.grid(row=len(stats), column=0, columnspan=2, pady=(6, 4), padx=8, sticky="w")
        legend_items = [
            ("white",       "Empty"),
            ("#5599dd",     "Free"),
            ("#dd8833",     "Deposited"),
            ("#222222",     "Substrate"),
            ("#66bb6a",     "Passivated"),
        ]
        for color, text in legend_items:
            dot = tk.Label(legend_frame, text="●", fg=color,
                           bg=self._colors["PANEL"], font=("Segoe UI", 11))
            dot.pack(side="left")
            tk.Label(legend_frame, text=f" {text}  ", bg=self._colors["PANEL"],
                     fg=self._colors["FG"], font=("Segoe UI", 8)).pack(side="left")

    def _build_plot_area(self, parent: ttk.Frame) -> None:
        C = self._colors
        # Two subplots stacked vertically: lattice view on top, atom count chart below.
        self._fig = Figure(figsize=(8, 7), facecolor=C["BG"])
        self._fig.subplots_adjust(hspace=0.35)

        # Lattice axes (top, bigger)
        self._ax_lat = self._fig.add_subplot(2, 1, 1)
        self._ax_lat.set_facecolor(C["BG"])
        for spine in self._ax_lat.spines.values():
            spine.set_edgecolor(C["ACC"])
        self._ax_lat.tick_params(colors=C["FG"], labelsize=8)
        self._ax_lat.set_xlabel("x  (lattice site)", color=C["FG"], fontsize=8)
        self._ax_lat.set_ylabel("y  (lattice site)", color=C["FG"], fontsize=8)
        self._ax_lat.set_title("Lattice  —  not started", color=C["ACC"], fontsize=9, pad=6)

        # Blank image placeholder.
        blank = np.zeros((self.params.Ny, self.params.Nx), dtype=np.int8)
        self._im = self._ax_lat.imshow(
            blank, cmap=self._cmap_for_display(), vmin=0, vmax=4,
            interpolation="nearest", aspect="auto")

        # Colorbar.
        cbar = self._fig.colorbar(self._im, ax=self._ax_lat,
                                  ticks=[0, 1, 2, 3, 4], fraction=0.03, pad=0.02)
        cbar.ax.set_yticklabels(["Empty", "Free", "Dep.", "Sub.", "Pass."])
        cbar.ax.tick_params(labelsize=7, colors=C["FG"])
        cbar.outline.set_edgecolor(C["ACC"])

        # Atom-count chart (bottom)
        self._ax_cnt = self._fig.add_subplot(2, 1, 2)
        self._ax_cnt.set_facecolor(C["BG"])
        for spine in self._ax_cnt.spines.values():
            spine.set_edgecolor(C["ACC"])
        self._ax_cnt.tick_params(colors=C["FG"], labelsize=8)
        self._ax_cnt.set_xlabel("KMC step", color=C["FG"], fontsize=8)
        self._ax_cnt.set_ylabel("Atom count", color=C["FG"], fontsize=8)
        self._ax_cnt.set_title("Atom counts over time", color=C["ACC"], fontsize=9, pad=6)
        self._line_free, = self._ax_cnt.plot([], [], color="#5599dd", lw=1.5, label="Free")
        self._line_dep,  = self._ax_cnt.plot([], [], color="#dd8833", lw=1.5, label="Deposited")
        self._line_pass, = self._ax_cnt.plot([], [], color="#66bb6a", lw=1.5, label="Passivated")
        self._ax_cnt.legend(facecolor=C["PANEL"], edgecolor=C["ACC"],
                            labelcolor=C["FG"], fontsize=8, loc="upper left")

        # Embed in Tkinter.
        self.plot_frame = parent
        self.canvas = FigureCanvasTkAgg(self._fig, master=parent)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.canvas.draw()

    @staticmethod
    def _cmap_for_display():
        return ListedColormap(["#111111", "#5599dd", "#dd8833", "#222222", "#66bb6a"])

    # ------------------------------------------------------------------
    # Parameter parsing
    # ------------------------------------------------------------------
    def parse_gui_params(self) -> KMCParams:
        values = {}
        for pname, (entry, ptype) in self.entries.items():
            raw = entry.get().strip()
            try:
                values[pname] = ptype(raw)
            except ValueError as exc:
                raise ValueError(
                    f"'{pname}' expects {ptype.__name__} but got '{raw}'.") from exc
        values["periodic_x"] = bool(values["periodic_x"])
        return KMCParams(**values)

    # ------------------------------------------------------------------
    # Control callbacks
    # ------------------------------------------------------------------
    def toggle_pause(self) -> None:
        self.paused = not self.paused
        self.pause_btn.config(text="▶  Resume" if self.paused else "⏸  Pause")

    def stop_simulation(self) -> None:
        self.stopped = True

    def save_now(self) -> None:
        """Save a labeled snapshot of the current lattice immediately."""
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

        # Reset history traces.
        self._hist_steps.clear()
        self._hist_free.clear()
        self._hist_deposited.clear()
        self._hist_passivated.clear()

        # Create simulator.
        try:
            self.sim = ElectrodepositionKMC(self.params)
        except Exception as exc:
            messagebox.showerror("Simulation error", str(exc))
            return

        # Resize blank image to new lattice dimensions.
        blank = np.zeros((self.params.Ny, self.params.Nx), dtype=np.int8)
        self._im.set_data(blank)
        self._im.set_extent([-0.5, self.params.Nx - 0.5,
                              self.params.Ny - 0.5, -0.5])
        self._ax_lat.set_xlim(-0.5, self.params.Nx - 0.5)
        self._ax_lat.set_ylim(self.params.Ny - 0.5, -0.5)

        # Reset atom count chart.
        self._line_free.set_data([], [])
        self._line_dep.set_data([], [])
        self._line_pass.set_data([], [])
        self._ax_cnt.relim()

        self.run_btn.config(state="disabled")
        self.root.after(1, self.run_batch)

    # ------------------------------------------------------------------
    # GUI update helpers
    # ------------------------------------------------------------------
    def _refresh_lattice(self) -> None:
        if self.sim is None:
            return
        self._im.set_data(self.sim.lattice_for_display())
        self._ax_lat.set_title(
            f"Step {self.sim.step:,}    Sim time {self.sim.time:.3e} s    "
            f"T={self.params.T} K",
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

        # Append to history traces.
        self._hist_steps.append(self.sim.step)
        self._hist_free.append(nf)
        self._hist_deposited.append(nd)
        self._hist_passivated.append(npass)

        self._line_free.set_data(self._hist_steps, self._hist_free)
        self._line_dep.set_data(self._hist_steps, self._hist_deposited)
        self._line_pass.set_data(self._hist_steps, self._hist_passivated)
        self._ax_cnt.relim()
        self._ax_cnt.autoscale_view()

    # ------------------------------------------------------------------
    # Main stepping loop
    # ------------------------------------------------------------------
    def run_batch(self) -> None:
        if self.sim is None:
            return

        if self.stopped:
            self.sim.finalize_outputs()
            self.run_btn.config(state="normal")
            messagebox.showinfo("Stopped",
                f"Simulation stopped at step {self.sim.step:,}.\n"
                f"Output saved to: {self.sim.output_dir}")
            return

        if self.paused:
            self.root.after(50, self.run_batch)
            return

        steps_done = 0
        while steps_done < self.sim.p.gui_batch_steps:
            if self.sim.step >= self.sim.p.max_steps or self.sim.time >= self.sim.p.max_time:
                break
            if not self.sim.execute_step():
                break

            if self.sim.step % self.sim.p.log_every == 0:
                self.sim.record_history(label="regular")
            if self.sim.p.save_snapshots and self.sim.step % self.sim.p.snapshot_every == 0:
                self.sim.save_snapshot(f"step_{self.sim.step:07d}")
            if self.sim.p.save_npy_states and self.sim.step % self.sim.p.snapshot_every == 0:
                self.sim.save_lattice_npy(f"step_{self.sim.step:07d}")

            steps_done += 1

        self._refresh_lattice()
        self._refresh_stats()
        self.canvas.draw_idle()

        finished = (
            self.sim.step >= self.sim.p.max_steps
            or self.sim.time >= self.sim.p.max_time
            or self.sim.ftree.total() <= 0.0
        )
        if finished:
            self.sim.finalize_outputs()
            self.run_btn.config(state="normal")
            messagebox.showinfo("Done",
                f"Simulation finished at step {self.sim.step:,}.\n"
                f"Output saved to: {self.sim.output_dir}")
            return

        self.root.after(1, self.run_batch)

# -----------------------------------------------------------------------------
# Scan mode
# -----------------------------------------------------------------------------
import itertools

def parse_scan_file(filepath: str) -> List[KMCParams]:
    """
    Read a scan parameter file and return a list of KMCParams — one per
    combination of swept values.

    File format:
      param_name = value              # single value (fixed)
      param_name = v1, v2, v3        # multiple values (swept)
      # lines starting with # are comments
    """
    # Types for each KMCParams field.
    field_types = {
        "Nx": int, "Ny": int,
        "T": float, "d0": float, "e0": float, "e1": float,
        "nu_f": float, "nu_d": float, "nu_p": float,
        "kB": float,
        "max_steps": int, "max_time": float,
        "rng_seed": int,       # None allowed — handled below
        "periodic_x": int,     # 0 or 1, converted to bool
        "log_every": int,
        "save_snapshots": bool, "snapshot_every": int,
        "save_npy_states": bool,
        "output_dir": str, "history_filename": str,
        "validation_enabled": bool, "validation_every": int,
        "gui_batch_steps": int,
        "stop_fill_fraction": float,   # None allowed — handled below
    }

    swept: dict = {}   # param_name -> list of parsed values

    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            name, _, raw = line.partition("=")
            name = name.strip()
            raw = raw.strip()
            if name not in field_types:
                print(f"  Warning: unknown parameter '{name}', skipping.")
                continue

            parts = [p.strip() for p in raw.split(",")]
            typ = field_types[name]
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

    # Separate swept params (multiple values) from fixed params (one value).
    fixed = {k: v[0] for k, v in swept.items() if len(v) == 1}
    variable = {k: v for k, v in swept.items() if len(v) > 1}

    if not variable:
        # No sweep — just one run.
        combos = [{}]
        keys = []
    else:
        keys = list(variable.keys())
        combos = [dict(zip(keys, combo))
                  for combo in itertools.product(*variable.values())]

    # Build a KMCParams for each combination.
    defaults = KMCParams()
    result = []
    for combo in combos:
        merged = {**vars(defaults), **fixed, **combo}
        # Handle periodic_x: stored as int in file (0/1), needs bool.
        if "periodic_x" in merged and not isinstance(merged["periodic_x"], bool):
            merged["periodic_x"] = bool(merged["periodic_x"])
        result.append(KMCParams(**{k: v for k, v in merged.items()
                                   if k in vars(defaults)}))
    return result


def run_scan(scan_file: str, scan_output_dir: str = "scan_output") -> None:
    """
    Read a scan parameter file, run one simulation per parameter combination,
    and save outputs + a summary CSV.

    Each run saves into its own subfolder:
      scan_output/
        T300.0_e0-0.2_nu_p1000.0/
          initial.png / lattice_initial.npy
          final_step_XXXXXX.png / lattice_final_step_XXXXXX.npy
          time_series.csv
        scan_summary.csv   <- one row per completed run
    """
    print(f"Reading scan file: {scan_file}")
    param_list = parse_scan_file(scan_file)
    print(f"Total runs: {len(param_list)}")

    base_dir = Path(scan_output_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []

    for run_idx, params in enumerate(param_list):
        # Build a short descriptive folder name from the swept values.
        run_label = f"T{params.T}_e0{params.e0}_nu_p{params.nu_p}"
        run_dir = base_dir / run_label
        run_dir.mkdir(parents=True, exist_ok=True)

        # IMPORTANT: set output_dir BEFORE constructing ElectrodepositionKMC,
        # because __init__ reads it immediately to create snapshot_dir.
        params.output_dir = str(run_dir)
        params.save_snapshots = True
        params.save_npy_states = True

        print(f"\n[{run_idx + 1}/{len(param_list)}] {run_label}")

        try:
            t0 = time.time()
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

    # Write summary CSV — one row per completed run.
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
USE_GUI  = True    # Set to False for CLI or scan mode (want GUI)
USE_SCAN = False   # Set to True to run a parameter scan (want SCAN)
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