"""Parameter sweeps — the instrument that produced the results in this repo.

The findings on the repository did not come from looking at one field state.
They came from walking a grid: omega, khra_amp and gixx_amp across a parameter
space, and injections repeated at a site with settling between, reading
coherence and asymmetry at each point. scripts/periodic_table_sweep.sh and
scripts/ab_power_test.sh are that method written by hand. This runs it from the
UI, records every point with the settings that produced it, and writes a CSV.

Settling is counted in the world's own steps, not in seconds. A wall clock
measures your machine; the cycle counter measures the run.
"""

from __future__ import annotations

import csv
import itertools
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

RECORD_FIELDS = [
    "coherence", "asymmetry", "omega", "khra_amp", "gixx_amp",
    "vel_mean", "vel_max", "vel_var", "vorticity_mean",
    "stress_xx", "stress_yy", "stress_xy",
    "gpu_temp_c", "gpu_power_w", "gpu_util_pct",
]

SWEEPABLE = ("omega", "khra_amp", "gixx_amp")


@dataclass
class Injection:
    x: float = 512.0
    y: float = 512.0
    sigma: float = 20.0
    strength: float = 0.06
    repeat: int = 1
    gap_steps: int = 200          # steps between repeated injections


@dataclass
class SweepSpec:
    axes: dict[str, list[float]] = field(default_factory=dict)
    injection: Injection | None = None
    settle_steps: int = 2000
    reset_between: bool = False
    label: str = ""

    def points(self) -> list[dict[str, float]]:
        names = [a for a in SWEEPABLE if self.axes.get(a)]
        if not names:
            return [{}]
        grids = [self.axes[n] for n in names]
        return [dict(zip(names, combo)) for combo in itertools.product(*grids)]


class SweepRunner:
    def __init__(self, bridge, state_dir: str):
        self.bridge = bridge
        self.state_dir = Path(state_dir)
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()

        self.spec: SweepSpec | None = None
        self.rows: list[dict] = []
        self.total = 0
        self.done = 0
        self.state = "idle"          # idle | running | stopping | finished | failed
        self.detail = ""
        self.csv_path: str | None = None
        self.started_at: float | None = None

    # -------------------------------------------------------------- control
    def start(self, spec: SweepSpec) -> dict:
        with self._lock:
            if self.state == "running":
                raise RuntimeError("a sweep is already running")
            self.spec = spec
            self.rows = []
            self.total = len(spec.points())
            self.done = 0
            self.state = "running"
            self.detail = ""
            self.started_at = time.time()
            stamp = time.strftime("%Y%m%d_%H%M%S")
            name = (spec.label.strip().replace(" ", "_") or "sweep")
            self.csv_path = str(self.state_dir / f"{name}_{stamp}.csv")
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="sweep", daemon=True)
        self._thread.start()
        return self.status()

    def stop(self) -> dict:
        with self._lock:
            if self.state == "running":
                self.state = "stopping"
        self._stop.set()
        return self.status()

    def status(self) -> dict:
        with self._lock:
            return {
                "state": self.state,
                "detail": self.detail,
                "done": self.done,
                "total": self.total,
                "rows": self.rows[-500:],
                "csv": self.csv_path,
                "elapsed_s": (time.time() - self.started_at) if self.started_at else None,
                "axes": self.spec.axes if self.spec else {},
            }

    # ------------------------------------------------------------ execution
    def _cycle(self) -> int | None:
        tel = self.bridge.telemetry
        return tel.get("cycle") if tel else None

    def _wait_steps(self, steps: int) -> bool:
        """Wait for the world's own clock to advance. False if stopped/stalled."""
        start = self._cycle()
        if start is None:
            self._stop.wait(2.0)
            return not self._stop.is_set()
        last_change = time.time()
        seen = start
        while not self._stop.is_set():
            now = self._cycle()
            if now is not None and now != seen:
                seen = now
                last_change = time.time()
            if now is not None and now - start >= steps:
                return True
            if time.time() - last_change > 30.0:
                with self._lock:
                    self.detail = ("the world's clock stopped advancing — is the "
                                   "daemon still running?")
                return False
            self._stop.wait(0.2)
        return False

    def _send(self, command: dict) -> bool:
        result = self.bridge.send(command)
        if not result.confirmed:
            with self._lock:
                self.detail = f"{command['cmd']} unconfirmed: {result.detail}"
        return result.confirmed

    def _run(self) -> None:
        spec = self.spec
        assert spec is not None
        try:
            for point in spec.points():
                if self._stop.is_set():
                    break

                if spec.reset_between:
                    self._send({"cmd": "reset_equilibrium"})
                    if not self._wait_steps(200):
                        break

                for name, value in point.items():
                    self._send({"cmd": f"set_{name}", "value": float(value)})

                if spec.injection is not None:
                    inj = spec.injection
                    for i in range(max(1, inj.repeat)):
                        if self._stop.is_set():
                            break
                        self._send({"cmd": "inject_density", "x": inj.x, "y": inj.y,
                                    "sigma": inj.sigma, "strength": inj.strength})
                        if i < inj.repeat - 1 and not self._wait_steps(inj.gap_steps):
                            break

                if not self._wait_steps(spec.settle_steps):
                    break

                tel = dict(self.bridge.telemetry or {})
                row = {"point": self.done, **{k: point.get(k) for k in SWEEPABLE},
                       "cycle": tel.get("cycle")}
                if spec.injection is not None:
                    row.update(inj_x=spec.injection.x, inj_y=spec.injection.y,
                               inj_sigma=spec.injection.sigma,
                               inj_strength=spec.injection.strength,
                               inj_repeat=spec.injection.repeat)
                for k in RECORD_FIELDS:
                    row[f"read_{k}"] = tel.get(k)

                with self._lock:
                    self.rows.append(row)
                    self.done += 1
                self._write_csv()

            with self._lock:
                if self.state != "failed":
                    self.state = "finished" if not self._stop.is_set() else "idle"
                    if not self.detail:
                        self.detail = (f"{self.done} of {self.total} points recorded"
                                       + (f" → {self.csv_path}" if self.csv_path else ""))
        except Exception as exc:            # a sweep must never take the server down
            with self._lock:
                self.state = "failed"
                self.detail = f"{type(exc).__name__}: {exc}"

    def _write_csv(self) -> None:
        with self._lock:
            rows = list(self.rows)
            path = self.csv_path
        if not rows or not path:
            return
        keys: list[str] = []
        for r in rows:
            for k in r:
                if k not in keys:
                    keys.append(k)
        try:
            with open(path, "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=keys)
                w.writeheader()
                w.writerows(rows)
        except OSError as exc:
            with self._lock:
                self.detail = f"could not write {path}: {exc}"
