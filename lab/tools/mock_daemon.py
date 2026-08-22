"""A stand-in daemon that speaks the real wire protocol, with no GPU.

It exists so the UI can be exercised and verified anywhere, and so a change to
the client can be tested without borrowing time on a real world. It reproduces
the daemon's behaviour deliberately, including the awkward parts:

  * out-of-range set_* values are IGNORED but still acked
  * inject_density sigma/strength are sticky statics
  * snapshots publish every `snapshot_interval` cycles
  * stress publishes only on stress_snapshot_now

It is not physics. The field is two travelling sinusoids plus drifting noise —
enough to look like something and to move when you write into it.

    python -m lab.tools.mock_daemon
"""

from __future__ import annotations

import json
import math
import struct
import sys
import time

import numpy as np
import zmq

NX = NY = 1024
HEADER = struct.Struct("<IHH")


def main() -> None:
    ctx = zmq.Context()
    tel = ctx.socket(zmq.PUB);   tel.bind("tcp://127.0.0.1:5556")
    cmd = ctx.socket(zmq.SUB);   cmd.setsockopt(zmq.SUBSCRIBE, b""); cmd.bind("tcp://127.0.0.1:5557")
    snap = ctx.socket(zmq.PUB);  snap.bind("tcp://127.0.0.1:5558")
    ack = ctx.socket(zmq.PUB);   ack.bind("tcp://127.0.0.1:5559")
    stress = ctx.socket(zmq.PUB); stress.bind("tcp://127.0.0.1:5560")

    print("mock daemon: 5556 telemetry | 5557 commands | 5558 snapshots | "
          "5559 ack | 5560 stress", flush=True)
    print("  hidden in the field: structures at (300,300) and (760,620) share a "
          "signal; (420,800) does not. None are visible in the raw view.",
          flush=True)

    omega, khra_amp, gixx_amp = 1.97, 0.05, 0.02
    inject_sigma, inject_strength = 16.0, 0.1     # sticky, exactly like the real one
    snapshot_interval = 10
    snapshot_now = stress_now = health_pending = False
    injections = 0
    cycle = 0
    start = time.time()

    yy, xx = np.mgrid[0:NY, 0:NX].astype(np.float32)
    marks = np.zeros((NY, NX), dtype=np.float32)

    # Three structures, far apart and invisible under the drive. Two of them
    # breathe on the same slow signal; the third has its own. Nothing in the
    # raw picture distinguishes them. This exists so the co-variation view can
    # be checked against a known answer rather than admired.
    def blob(cx, cy, sigma=26.0):
        return np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma * sigma)))
    PAIR_A = blob(300, 300)
    PAIR_B = blob(760, 620)      # the partner — same signal as A
    LONE_C = blob(420, 800)      # independent

    poller = zmq.Poller()
    poller.register(cmd, zmq.POLLIN)

    while True:
        cycle += 10
        t = cycle

        for _sock, _ in poller.poll(0):
            raw = cmd.recv()
            try:
                m = json.loads(raw.decode())
            except json.JSONDecodeError:
                continue
            name = m.get("cmd")
            status = "ok"
            if name == "set_omega":
                v = float(m.get("value", omega))
                if 0.5 <= v <= 1.99:
                    omega = v
                else:
                    print(f"  (ignored out-of-range omega {v}, still acking)", flush=True)
            elif name == "set_khra_amp":
                v = float(m.get("value", khra_amp))
                if 0.0 <= v <= 0.2: khra_amp = v
            elif name == "set_gixx_amp":
                v = float(m.get("value", gixx_amp))
                if 0.0 <= v <= 0.1: gixx_amp = v
            elif name == "set_snapshot_interval":
                v = int(m.get("interval", snapshot_interval))
                if 0 <= v <= 10_000_000: snapshot_interval = v
            elif name == "snapshot_now":
                snapshot_now = True
            elif name == "stress_snapshot_now":
                stress_now = True
            elif name == "health_check":
                health_pending = True
            elif name == "reset_equilibrium":
                marks[:] = 0.0
            elif name == "inject_density":
                x = min(max(float(m.get("x", 0)), 0), NX - 1)
                y = min(max(float(m.get("y", 0)), 0), NY - 1)
                if "sigma" in m:
                    s = float(m["sigma"])
                    if 1.0 <= s <= 256.0: inject_sigma = s
                if "strength" in m:
                    st = float(m["strength"])
                    if -1.0 <= st <= 1.0: inject_strength = st
                r2 = (xx - x) ** 2 + (yy - y) ** 2
                marks += inject_strength * np.exp(-r2 / (2.0 * inject_sigma ** 2)).astype(np.float32)
                injections += 1
                ack.send_json({"injection_id": injections, "cycle": cycle,
                               "x": x, "y": y, "sigma": inject_sigma,
                               "strength": inject_strength})
                print(f"  inject #{injections} at ({x:.0f},{y:.0f}) "
                      f"sigma={inject_sigma} strength={inject_strength}", flush=True)
                continue
            elif name in ("save_state", "load_state", "export_timeseries", "set_autosave"):
                path = m.get("path", "")
                if name == "save_state" and path:
                    with open(path, "wb") as fh:
                        fh.write(b"KHRG")
                        fh.write(struct.pack("<i", cycle))
                        fh.write(np.zeros(64 - 8, dtype=np.uint8).tobytes())
                        fh.write(np.random.rand(NX * NY * 9).astype("<f4").tobytes())
                    print(f"  wrote checkpoint {path}", flush=True)
            else:
                status = "unknown"
            ack.send_json({"ack": name, "cycle": cycle, "status": status})

        # --- field ------------------------------------------------------
        phase_k = 2 * math.pi * xx / 128.0 + t * 0.025
        phase_g = 2 * math.pi * xx / 8.0 + t * 0.4
        breath = 1.0 + 0.02 * math.sin(2 * math.pi * t / 125.66)
        rho = (1.0
               + khra_amp * np.sin(phase_k) * breath
               + gixx_amp * np.sin(phase_g)
               + 0.004 * np.sin(2 * math.pi * yy / 256.0 + t * 0.01)
               ).astype(np.float32)
        marks *= 0.999
        rho += marks

        shared = 0.010 * math.sin(2 * math.pi * cycle / 3700.0)
        lone = 0.010 * math.sin(2 * math.pi * cycle / 1100.0 + 1.3)
        rho += (shared * (PAIR_A + PAIR_B) + lone * LONE_C).astype(np.float32)
        rho += (0.0006 * np.random.standard_normal((NY, NX))).astype(np.float32)

        coherence = float(0.74 + 0.002 * math.sin(t / 900.0))
        asymmetry = float(12.3 + 0.4 * math.sin(t / 2500.0))
        vel = float(np.abs(rho - 1.0).mean() * 12.0)

        tel.send_string(json.dumps({
            "cycle": cycle, "ts": int(time.time()),
            "coherence": round(coherence, 4), "asymmetry": round(asymmetry, 4),
            "omega": round(omega, 3), "khra_amp": round(khra_amp, 4),
            "gixx_amp": round(gixx_amp, 4), "grid": 1024,
            "vel_mean": round(vel, 6), "vel_max": round(vel * 1.26, 6),
            "vel_var": round(vel * 1e-3, 8),
            "vorticity_mean": round(float(rho.std()) * 1e-3, 6),
            "stress_xx": 0.001234, "stress_yy": 0.001111, "stress_xy": -0.000212,
            "gpu_temp_c": 61, "gpu_power_w": 310.0,
            "gpu_util_pct": 98, "gpu_mem_pct": 22.5,
        }))

        if health_pending:
            ack.send_json({"health": {
                "cycle": cycle, "coherence": round(coherence, 4),
                "asymmetry": round(asymmetry, 4), "omega": round(omega, 3),
                "gpu_temp_c": 61, "gpu_mem_used_mb": 5400.0,
                "uptime_seconds": int(time.time() - start),
                "total_injections": injections, "last_checkpoint_cycle": 0}})
            health_pending = False

        if snapshot_now or (snapshot_interval > 0 and cycle % snapshot_interval == 0):
            snap.send(HEADER.pack(cycle, NX, NY) + rho.tobytes())
            snapshot_now = False

        if stress_now:
            sxx = (rho - 1.0) * 0.01
            syy = (rho - 1.0) * -0.008
            sxy = np.gradient(rho, axis=1).astype(np.float32) * 0.05
            stress.send(HEADER.pack(cycle, NX, NY)
                        + sxx.astype("<f4").tobytes()
                        + syy.astype("<f4").tobytes()
                        + sxy.astype("<f4").tobytes())
            stress_now = False
            print(f"  stress frame at cycle {cycle}", flush=True)

        time.sleep(0.1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
