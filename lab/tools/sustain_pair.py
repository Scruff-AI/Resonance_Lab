"""Sustain a correlated pair and an independent source in a live lattice.

The daemon's inject_density decays (half-life ~23 cycles, gone by ~200), so a
single mark fades out. This tool repeats injections at A and B on the same
schedule — so those two cells breathe together — and at C on a different
schedule, so it is independent. It maintains sources; it does not plant a mark.

This is the client-side half of the co-variation validation: seed a known
correlated pair in a real lattice, then confirm the co-variation view finds it.

Verified against cuda/khra_gixx_1024_v5.cu line 963 that the daemon handles one
command per main-loop iteration, so back-to-back A and B injections do not
overwrite each other — they land one cycle apart.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import zmq

from lab import commands as cmdspec


def _point(s: str) -> tuple[float, float]:
    x, y = s.split(",")
    return float(x), float(y)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--a", type=_point, default=(300.0, 300.0),
                   help="first partner, x,y (default 300,300)")
    p.add_argument("--b", type=_point, default=(700.0, 700.0),
                   help="second partner that breathes with --a, x,y (default 700,700)")
    p.add_argument("--c", type=_point, default=(500.0, 700.0),
                   help="independent source, x,y (default 500,700)")
    p.add_argument("--ab-period", type=int, default=50,
                   help="cycles between A+B injections (default 50)")
    p.add_argument("--c-period", type=int, default=73,
                   help="cycles between C injections; choose coprime to --ab-period (default 73)")
    p.add_argument("--sigma", type=float, default=16.0)
    p.add_argument("--strength", type=float, default=0.1)
    p.add_argument("--seconds", type=float, default=0.0,
                   help="run this long then exit (0 = forever)")
    p.add_argument("--daemon-host", default="127.0.0.1")
    p.add_argument("--telemetry-port", type=int, default=5556)
    p.add_argument("--command-port", type=int, default=5557)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ctx = zmq.Context()

    cmd = ctx.socket(zmq.PUB)
    cmd.connect(f"tcp://{args.daemon_host}:{args.command_port}")

    tel = ctx.socket(zmq.SUB)
    tel.setsockopt(zmq.CONFLATE, 1)  # keep only the newest telemetry message
    tel.setsockopt(zmq.SUBSCRIBE, b"")
    tel.setsockopt(zmq.RCVTIMEO, 2000)
    tel.connect(f"tcp://{args.daemon_host}:{args.telemetry_port}")

    # PUB->SUB drops anything sent before the subscription handshake settles.
    time.sleep(1.0)

    def inject(x: float, y: float) -> None:
        payload = cmdspec.validate({
            "cmd": "inject_density",
            "x": x, "y": y,
            "sigma": args.sigma, "strength": args.strength,
        })
        cmd.send_string(json.dumps(payload, separators=(",", ":")))

    cycle = 0
    last_ab = -1 << 30
    last_c = -1 << 30
    started = time.time()

    try:
        while True:
            # Track the world's own clock from telemetry (published every 10 cycles).
            try:
                msg = tel.recv()
                cycle = int(json.loads(msg).get("cycle", cycle))
            except zmq.Again:
                pass

            if cycle and cycle - last_ab >= args.ab_period:
                inject(*args.a)
                inject(*args.b)
                last_ab = cycle
                print(f"[cycle {cycle}] injected A+B", flush=True)
            if cycle and cycle - last_c >= args.c_period:
                inject(*args.c)
                last_c = cycle
                print(f"[cycle {cycle}] injected C", flush=True)

            if args.seconds and time.time() - started >= args.seconds:
                print("done", flush=True)
                break
            time.sleep(0.02)
    except cmdspec.Rejected as exc:
        print(f"rejected: {exc}", file=sys.stderr)
        raise SystemExit(2)
    except KeyboardInterrupt:
        pass
    finally:
        cmd.close()
        tel.close()
        ctx.term()


if __name__ == "__main__":
    main()
