"""Views that show something other than the pump.

A picture of raw rho is a picture of the forcing. The drive is by far the
loudest thing in the frame, so every absolute view is corduroy and nothing
underneath it can be seen. These views exist to get it out of the way, and to
put time and relationship on the screen instead of a single instant.

Nothing here writes to the world. Every view carries what it removed and what
it was measured against, because a difference image with no stated reference is
not readable.
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass

import numpy as np

SMALL = 256          # resolution kept in the rolling buffer
BUFFER_FRAMES = 600  # how far back co-variation and round-phase can reach


@dataclass
class ViewResult:
    data: np.ndarray
    cycle: int
    quantity: str          # what the picture is OF
    reference: str         # what it is measured AGAINST
    diverging: bool        # symmetric scale around zero
    detail: str = ""       # anything the reader needs to judge it


def _downsample(arr: np.ndarray, target: int) -> np.ndarray:
    h, w = arr.shape
    if max(h, w) <= target:
        return arr.astype(np.float32)
    f = max(1, int(round(max(h, w) / target)))
    if h % f or w % f:
        return arr[::f, ::f].astype(np.float32)
    return arr.reshape(h // f, f, w // f, f).mean(axis=(1, 3)).astype(np.float32)


# --------------------------------------------------------------------- drive
def dominant_modes(arr: np.ndarray, n: int = 2) -> list[tuple[int, int, float]]:
    """The n strongest spatial modes, DC excluded. Measured, not assumed.

    Returns (ky, kx, power) with kx, ky as signed integer wavenumbers.
    """
    F = np.fft.fft2(arr - arr.mean())
    P = np.abs(F) ** 2
    h, w = P.shape
    flat = P.ravel().argsort()[::-1]
    found: list[tuple[int, int, float]] = []
    taken: set[tuple[int, int]] = set()
    for idx in flat:
        iy, ix = divmod(int(idx), w)
        ky = iy - h if iy > h // 2 else iy
        kx = ix - w if ix > w // 2 else ix
        if ky == 0 and kx == 0:
            continue
        key = (abs(ky), abs(kx))          # a real mode and its conjugate are one mode
        if key in taken:
            continue
        taken.add(key)
        found.append((ky, kx, float(P[iy, ix])))
        if len(found) >= n:
            break
    return found


def remove_drive(arr: np.ndarray, n_modes: int = 2, radius: int = 1
                 ) -> tuple[np.ndarray, list[tuple[int, int]], float]:
    """Notch out the strongest spatial modes and return what is left.

    Also returns the fraction of the field's variance those modes carried, so
    the caption can say how much of the picture was the pump.
    """
    mean = arr.mean()
    F = np.fft.fft2(arr - mean)
    h, w = F.shape
    total = float(np.sum(np.abs(F) ** 2))

    removed: list[tuple[int, int]] = []
    for ky, kx, _ in dominant_modes(arr, n_modes):
        removed.append((int(ky), int(kx)))
        for sy, sx in ((ky, kx), (-ky, -kx)):
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    F[(sy + dy) % h, (sx + dx) % w] = 0.0

    kept = float(np.sum(np.abs(F) ** 2))
    fraction = (total - kept) / total if total > 0 else 0.0
    return np.real(np.fft.ifft2(F)).astype(np.float32), removed, fraction


# --------------------------------------------------------------------- store
class FrameStore:
    """Rolling history, so time and relationship can be drawn at all."""

    def __init__(self, row_index: int | None = None, notch_modes: int = 4):
        self._lock = threading.Lock()
        self.notch_modes = notch_modes
        self.notched: list[tuple[int, int]] = []
        self.small: deque = deque(maxlen=BUFFER_FRAMES)      # (cycle, 256^2 frame)
        self.rows: deque = deque(maxlen=BUFFER_FRAMES)       # (cycle, one full-res row)
        self.baseline: tuple[int, np.ndarray] | None = None
        self.row_index = row_index
        self.source_shape: tuple[int, int] | None = None

    def add(self, cycle: int, frame: np.ndarray) -> None:
        # The buffered copy has the drive notched out of it. Correlating raw
        # frames measures the forcing: every cell's history is dominated by the
        # travelling waves, so everything correlates with everything at the
        # drive's phase and nothing else is visible. Measured on a synthetic
        # field with a known correlated pair, raw frames put the true partner
        # at 0.05 and empty background at 0.49. Notching first separates them.
        with self._lock:
            if self.small and self.small[-1][0] == cycle:
                return
            self.source_shape = frame.shape
            r = self.row_index if self.row_index is not None else frame.shape[0] // 2
            r = max(0, min(frame.shape[0] - 1, r))
        small = _downsample(frame, SMALL)
        small, removed, _ = remove_drive(small, n_modes=self.notch_modes)
        with self._lock:
            self.notched = removed
            self.small.append((cycle, small))
            self.rows.append((cycle, frame[r].astype(np.float32).copy()))

    def set_baseline(self, cycle: int, frame: np.ndarray) -> None:
        with self._lock:
            self.baseline = (cycle, frame.astype(np.float32).copy())

    def clear_baseline(self) -> None:
        with self._lock:
            self.baseline = None

    def set_row(self, index: int) -> None:
        with self._lock:
            self.row_index = index
            self.rows.clear()

    def depth(self) -> int:
        with self._lock:
            return len(self.small)

    def span(self) -> tuple[int, int] | None:
        with self._lock:
            if not self.small:
                return None
            return self.small[0][0], self.small[-1][0]

    def snapshot_small(self) -> tuple[list[int], np.ndarray] | None:
        with self._lock:
            if len(self.small) < 8:
                return None
            cycles = [c for c, _ in self.small]
            stack = np.stack([f for _, f in self.small])
        return cycles, stack

    def snapshot_rows(self) -> tuple[list[int], np.ndarray] | None:
        with self._lock:
            if len(self.rows) < 4:
                return None
            cycles = [c for c, _ in self.rows]
            stack = np.stack([r for _, r in self.rows])
        return cycles, stack


# --------------------------------------------------------------------- views
def view_drive_removed(frame: np.ndarray, cycle: int, n_modes: int = 2) -> ViewResult:
    out, removed, fraction = remove_drive(frame, n_modes)
    modes = ", ".join(f"(ky={ky}, kx={kx})" for ky, kx in removed)
    return ViewResult(
        data=out, cycle=cycle,
        quantity="everything except the strongest modes",
        reference=f"notched {modes}",
        diverging=True,
        detail=(f"those modes carried {fraction*100:.1f}% of the field's variance; "
                f"what you are looking at is the remaining {100-fraction*100:.1f}%"),
    )


def view_baseline(frame: np.ndarray, cycle: int, store: FrameStore) -> ViewResult | None:
    if store.baseline is None:
        return None
    b_cycle, b = store.baseline
    if b.shape != frame.shape:
        return None
    return ViewResult(
        data=(frame - b).astype(np.float32), cycle=cycle,
        quantity="change since the baseline",
        reference=f"baseline held at step {b_cycle}",
        diverging=True,
        detail=f"{cycle - b_cycle} steps elapsed since the baseline was pinned",
    )


def view_space_time(store: FrameStore) -> ViewResult | None:
    """One line of the field, stacked downward over time.

    A thing that travels leans. A thing that alternates in place chequers. A
    thing that persists runs straight down. No reasoning about intervals
    required — the answer is the shape.
    """
    got = store.snapshot_rows()
    if got is None:
        return None
    cycles, stack = got
    strip = stack - stack.mean(axis=0, keepdims=True)   # kill the static profile
    # The travelling drive is a diagonal mode of this strip and swamps it.
    strip, notched, fraction = remove_drive(strip, n_modes=4)
    row = store.row_index if store.row_index is not None else (
        store.source_shape[0] // 2 if store.source_shape else 0)
    step = (cycles[-1] - cycles[0]) / max(1, len(cycles) - 1)
    return ViewResult(
        data=strip.astype(np.float32), cycle=cycles[-1],
        quantity=f"row {row} over time (x across, time down)",
        reference=("column averages removed, then the four strongest modes of the "
                   "strip notched — the travelling drive is among them"),
        diverging=True,
        detail=(f"{len(cycles)} frames, steps {cycles[0]} to {cycles[-1]}, "
                f"~{step:.0f} steps per pixel row — anything faster than that "
                f"cannot appear here. The notch removed "
                f"{fraction*100:.1f}% of the strip's variance."),
    )


def view_covariation(store: FrameStore, cx: float, cy: float,
                     radius_frac: float = 0.02) -> ViewResult | None:
    """Where else in the field moves with the place you picked.

    For each cell, the correlation of its history with the history of the
    chosen patch, over the whole buffer. This is a relationship, not a state:
    it answers "what goes with this", which is the only question the medium
    can be asked about association without writing into it.
    """
    got = store.snapshot_small()
    if got is None:
        return None
    cycles, stack = got                      # (T, H, W)
    T, H, W = stack.shape

    px = int(np.clip(cx * W, 0, W - 1))
    py = int(np.clip(cy * H, 0, H - 1))
    r = max(1, int(round(radius_frac * max(H, W))))
    y0, y1 = max(0, py - r), min(H, py + r + 1)
    x0, x1 = max(0, px - r), min(W, px + r + 1)

    seed = stack[:, y0:y1, x0:x1].reshape(T, -1).mean(axis=1)
    seed = seed - seed.mean()
    seed_norm = np.sqrt(np.sum(seed ** 2))
    if seed_norm < 1e-20:
        return None

    flat = stack.reshape(T, -1)
    flat = flat - flat.mean(axis=0, keepdims=True)
    norms = np.sqrt(np.einsum("ij,ij->j", flat, flat))
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = (seed @ flat) / (seed_norm * norms)
    corr = np.nan_to_num(corr, nan=0.0).reshape(H, W).astype(np.float32)

    return ViewResult(
        data=corr, cycle=cycles[-1],
        quantity=f"what moves with ({px * (store.source_shape[1] // W) if store.source_shape else px}, "
                 f"{py * (store.source_shape[0] // H) if store.source_shape else py})",
        reference="correlation over the buffered window, −1 to +1, drive notched first",
        diverging=True,
        detail=(f"{T} frames, steps {cycles[0]} to {cycles[-1]}; patch radius "
                f"{r} cells at {W}² working resolution. The chosen patch reads "
                f"+1 by construction — it is correlated with itself."),
    )
