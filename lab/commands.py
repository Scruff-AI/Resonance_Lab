"""The daemon's command surface, with its real accepted ranges.

Every bound here was read out of the command handler in
cuda/khra_gixx_1024_v5.cu. They matter more than they look:

    THE DAEMON ACKS COMMANDS IT DID NOT APPLY.

set_omega, set_khra_amp and set_gixx_amp silently do nothing when the value is
outside range — no error, no log line — and the generic ACK on 5559 still goes
out naming the command. So an out-of-range set looks exactly like a successful
one from the wire. This module rejects those before they are sent, so the UI
can never show a green tick over a parameter that never moved.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class Param:
    name: str
    kind: str               # "float" | "int" | "str"
    lo: float | None = None
    hi: float | None = None
    default: Any = None
    required: bool = True
    note: str = ""


@dataclass
class Spec:
    cmd: str
    summary: str
    params: list[Param]
    writes_field: bool = False   # does it change the state of the world?
    note: str = ""


SPECS: dict[str, Spec] = {
    s.cmd: s for s in [
        Spec("health_check", "Ask the daemon for cycle, coherence, asymmetry, uptime.", []),
        Spec("snapshot_now", "Publish a density snapshot on 5558 at the next telemetry tick.", []),
        Spec("stress_snapshot_now", "Publish sxx/syy/sxy on 5560 at the next telemetry tick.", []),
        Spec("set_snapshot_interval", "How often density snapshots publish. 0 disables.",
             [Param("interval", "int", 0, 10_000_000, 10)]),
        Spec("set_autosave", "Checkpoint every N cycles. 0 disables.",
             [Param("interval", "int", 0, 10_000_000, 0)]),
        Spec("set_omega", "Relaxation rate. The engine's operating point.",
             [Param("value", "float", 0.5, 1.99, 1.97)],
             writes_field=True,
             note="Changes what equilibrium means. Every calibrated number in "
                  "this project was taken at 1.97 — moving it invalidates "
                  "comparison with anything measured before."),
        Spec("set_khra_amp", "Amplitude of the long forcing wave (wavelength 128).",
             [Param("value", "float", 0.0, 0.2, 0.05)], writes_field=True),
        Spec("set_gixx_amp", "Amplitude of the short forcing wave (wavelength 8).",
             [Param("value", "float", 0.0, 0.1, 0.02)], writes_field=True),
        Spec("inject_density", "Write a Gaussian density perturbation into the field.",
             [Param("x", "float", 0, 1023, 512),
              Param("y", "float", 0, 1023, 512),
              Param("sigma", "float", 1.0, 256.0, 16.0,
                    note="Always sent explicitly. In the daemon inject_sigma is "
                         "a sticky static: omit it and the injection silently "
                         "reuses the previous injection's sigma."),
              Param("strength", "float", -1.0, 1.0, 0.1,
                    note="Also a sticky static, same trap. No density floor "
                         "exists in this build, so large negative strength can "
                         "drive rho below zero.")],
             writes_field=True),
        Spec("reset_equilibrium", "Reset every cell to the equilibrium distribution.",
             [], writes_field=True,
             note="Destroys the current world state. There is no undo — "
                  "save_state first if this run matters."),
        Spec("save_state", "Write the full lattice state to a checkpoint file (KHRG v1).",
             [Param("path", "str", default=".")]),
        Spec("load_state", "Load a checkpoint file back into the running lattice.",
             [Param("path", "str")], writes_field=True,
             note="Replaces the running world."),
        Spec("export_timeseries", "Dump the telemetry ring buffer to a file.",
             [Param("path", "str"),
              Param("last_n", "int", 1, 10_000_000, 10000, required=False)]),
    ]
}


# The full command surface of khra_gixx_1024_v5 is available. This is the
# instrument that produced the results on the repository: the parameter space
# (omega, khra_amp, gixx_amp) swept across a grid, and repeated injections at a
# site with settling between. Range checks stay, because the daemon acks values
# it silently ignores — but nothing is withheld.
WRITES_FIELD = {name for name, spec in SPECS.items() if spec.writes_field}


class Rejected(ValueError):
    """A command that would have been acked without being applied."""


def validate(command: dict) -> dict:
    """Return a normalised command, or raise Rejected with a plain reason."""
    if not isinstance(command, dict):
        raise Rejected("command must be a JSON object")
    name = command.get("cmd")
    if name not in SPECS:
        known = ", ".join(sorted(SPECS))
        raise Rejected(f"unknown command {name!r}. Known commands: {known}")

    spec = SPECS[name]
    out: dict = {"cmd": name}
    allowed = {p.name for p in spec.params}
    extra = set(command) - allowed - {"cmd"}
    if extra:
        raise Rejected(
            f"{name} does not take {sorted(extra)} — the daemon ignores unknown "
            f"fields silently, so this would look like it worked"
        )

    for p in spec.params:
        if p.name not in command:
            if p.required and p.default is None:
                raise Rejected(f"{name} requires {p.name!r}")
            if p.required:
                out[p.name] = p.default
            continue
        raw = command[p.name]
        if p.kind == "str":
            if not isinstance(raw, str) or not raw.strip():
                raise Rejected(f"{name}.{p.name} must be a non-empty string")
            out[p.name] = raw
            continue
        try:
            val = float(raw)
        except (TypeError, ValueError):
            raise Rejected(f"{name}.{p.name} must be a number, got {raw!r}")
        if p.kind == "int":
            if val != int(val):
                raise Rejected(f"{name}.{p.name} must be a whole number, got {raw!r}")
            val = int(val)
        if p.lo is not None and val < p.lo:
            raise Rejected(
                f"{name}.{p.name}={val} is below the daemon's minimum {p.lo}. "
                f"The daemon would ACK this and ignore it."
            )
        if p.hi is not None and val > p.hi:
            raise Rejected(
                f"{name}.{p.name}={val} is above the daemon's maximum {p.hi}. "
                f"The daemon would ACK this and ignore it."
            )
        out[p.name] = val
    return out


def describe() -> list[dict]:
    """Machine-readable command surface, for the UI and the model prompt."""
    return [
        {
            "cmd": s.cmd,
            "writes_field": s.cmd in WRITES_FIELD,
            "summary": s.summary,
            "writes_field": s.writes_field,
            "note": s.note,
            "params": [
                {"name": p.name, "kind": p.kind, "min": p.lo, "max": p.hi,
                 "default": p.default, "required": p.required, "note": p.note}
                for p in s.params
            ],
        }
        for s in SPECS.values()
    ]
