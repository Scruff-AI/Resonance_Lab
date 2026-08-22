"""Resonance Lab configuration.

Everything that used to be hardcoded lives here. Override any value with an
environment variable of the same name, or point RESONANCE_LAB_CONFIG at a JSON
file. Nothing in this package assumes Windows, WSL, or a particular drive.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _env(name: str, default):
    raw = os.environ.get(name)
    if raw is None:
        return default
    if isinstance(default, bool):
        return raw.strip().lower() in ("1", "true", "yes", "on")
    if isinstance(default, int):
        return int(raw)
    if isinstance(default, float):
        return float(raw)
    return raw


@dataclass
class Config:
    # --- daemon wire ---------------------------------------------------
    # The daemon binds all of these on 127.0.0.1, so the lab server must run
    # on the same host as the daemon. Reach the UI from a laptop with:
    #   ssh -N -L 8800:127.0.0.1:8800 you@cluster-node
    daemon_host: str = "127.0.0.1"
    telemetry_port: int = 5556   # daemon PUB  -> we SUB   (JSON, every 10 cycles)
    command_port: int = 5557     # daemon SUB  <- we PUB   (JSON commands)
    snapshot_port: int = 5558    # daemon PUB  -> we SUB   (8B header + NX*NY float32 rho)
    ack_port: int = 5559         # daemon PUB  -> we SUB   (JSON acks / injections / health)
    stress_port: int = 5560      # daemon PUB  -> we SUB   (8B header + 3 * NX*NY float32)
    coarse_port: int = 5561      # observer fork only: 32x32x6 coarse field

    # --- web server ----------------------------------------------------
    http_host: str = "127.0.0.1"
    http_port: int = 8800

    # --- rendering -----------------------------------------------------
    render_size: int = 1024      # downsample target for the browser; raise it on
                                 # hardware that can push more pixels
    render_fps_cap: float = 12.0 # frames pushed per second, regardless of daemon rate
    default_colormap: str = "viridis"
    # A scale recomputed every frame hides anything that holds steady — the same
    # colour stops meaning the same value. Lock it once and keep it.
    fixed_scale: bool = True

    # --- model backend -------------------------------------------------
    model_backend: str = "ollama"          # "ollama" | "openai" | "none"
    ollama_url: str = "http://127.0.0.1:11434"
    ollama_model: str = ""                 # blank -> pick in the UI
    openai_base_url: str = "https://api.openai.com/v1"
    openai_model: str = ""
    openai_api_key_env: str = "RESONANCE_LAB_API_KEY"
    model_temperature: float = 0.8
    model_timeout_s: int = 180

    # --- safety --------------------------------------------------------
    # The model never writes to the lattice directly. Every command it proposes
    # lands in an approval queue and only reaches port 5557 when a human clicks
    # approve. This is deliberate: two writers on one live world contaminate
    # each other's measurements.
    ack_timeout_s: float = 3.0
    # The Navigator is not given the project's names for the readings.
    neutral_telemetry_labels: bool = True

    # --- state files ---------------------------------------------------
    state_dir: str = str(REPO_ROOT / "build" / "states")
    log_dir: str = str(REPO_ROOT / "logs")

    @classmethod
    def load(cls) -> "Config":
        cfg = cls()
        path = os.environ.get("RESONANCE_LAB_CONFIG")
        if path:
            data = json.loads(Path(path).read_text())
            for k, v in data.items():
                if hasattr(cfg, k):
                    setattr(cfg, k, v)
                else:
                    raise KeyError(f"unknown config key in {path}: {k}")
        for k, v in asdict(cfg).items():
            setattr(cfg, k, _env(k.upper(), v))
        Path(cfg.state_dir).mkdir(parents=True, exist_ok=True)
        Path(cfg.log_dir).mkdir(parents=True, exist_ok=True)
        return cfg

    def endpoint(self, port: int) -> str:
        return f"tcp://{self.daemon_host}:{port}"


CONFIG = Config.load()
