"""ZMQ bridge to the Khra'gixx daemon.

Wire formats below were read out of cuda/khra_gixx_1024_v5.cu, not assumed:

  5556  daemon PUB, no topic prefix, one JSON object per telemetry tick
        (a tick is every 10 cycles):
          cycle, ts, coherence, asymmetry, omega, khra_amp, gixx_amp, grid,
          vel_mean, vel_max, vel_var, vorticity_mean,
          stress_xx, stress_yy, stress_xy,
          gpu_temp_c, gpu_power_w, gpu_util_pct, gpu_mem_pct

  5557  daemon SUB (it BINDS, we CONNECT and PUB). One JSON object per command:
          {"cmd":"set_omega","value":1.97}
          {"cmd":"inject_density","x":512,"y":512,"sigma":16,"strength":0.1}
          {"cmd":"save_state","path":"..."} / {"cmd":"load_state","path":"..."}
        PUB->SUB drops anything sent before the subscription handshake settles,
        so we connect once at startup and wait before the first send.

  5558  daemon PUB: uint32 cycle | uint16 w | uint16 h | w*h float32 rho
        Fires every `snapshot_interval` cycles (default 10) and on snapshot_now.

  5559  daemon PUB, JSON, three shapes:
          {"ack":"<cmd>","cycle":N,"status":"..."}
          {"injection_id":N,"cycle":N,"x":..,"y":..,"sigma":..,"strength":..}
          {"health":{...}}

  5560  daemon PUB: same 8-byte header, then sxx, syy, sxy each w*h float32.
        Fires only on stress_snapshot_now.

Honesty rule this module enforces: a command is reported CONFIRMED only when a
matching ack came back off 5559. Anything else is reported as UNCONFIRMED with
the reason. It never says "sent" and means "arrived".
"""

from __future__ import annotations

import json
import struct
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import zmq

HEADER = struct.Struct("<IHH")  # cycle, width, height


@dataclass
class Frame:
    cycle: int
    width: int
    height: int
    data: np.ndarray          # float32, shape (height, width)
    received_at: float
    kind: str = "density"


@dataclass
class CommandResult:
    command: dict
    sent_at: float
    confirmed: bool
    ack: dict | None = None
    detail: str = ""

    def to_dict(self) -> dict:
        return {
            "command": self.command,
            "sent_at": self.sent_at,
            "confirmed": self.confirmed,
            "ack": self.ack,
            "detail": self.detail,
        }


class DaemonBridge:
    """Threaded reader/writer for one running daemon."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.ctx = zmq.Context.instance()
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []

        # Only one writer may touch 5557 at a time. Human and model share it.
        self._write_lock = threading.Lock()

        self.telemetry: dict | None = None
        self.telemetry_history: deque = deque(maxlen=3000)
        self.last_telemetry_at: float = 0.0

        self.density: Frame | None = None
        self.stress: dict[str, Frame] | None = None

        self.acks: deque = deque(maxlen=400)
        self.injections: deque = deque(maxlen=400)
        self.health: dict | None = None
        self._ack_event = threading.Condition()

        self.command_log: deque = deque(maxlen=400)
        self._cmd_sock = None
        self._cmd_ready_at = 0.0

        self.on_telemetry: list[Callable[[dict], None]] = []
        self.on_frame: list[Callable[[Frame], None]] = []
        self.on_ack: list[Callable[[dict], None]] = []

    # ------------------------------------------------------------------ life
    def start(self) -> None:
        self._cmd_sock = self.ctx.socket(zmq.PUB)
        self._cmd_sock.setsockopt(zmq.LINGER, 200)
        self._cmd_sock.connect(self.cfg.endpoint(self.cfg.command_port))
        # PUB->SUB slow joiner: sends before the handshake completes vanish.
        self._cmd_ready_at = time.time() + 0.5

        self._spawn(self._telemetry_loop, "telemetry")
        self._spawn(self._snapshot_loop, "snapshot")
        self._spawn(self._ack_loop, "ack")
        self._spawn(self._stress_loop, "stress")

    def _spawn(self, fn, name):
        t = threading.Thread(target=fn, name=f"bridge-{name}", daemon=True)
        t.start()
        self._threads.append(t)

    def stop(self) -> None:
        self._stop.set()
        for t in self._threads:
            t.join(timeout=1.0)
        if self._cmd_sock is not None:
            self._cmd_sock.close(0)

    def _sub(self, port: int):
        s = self.ctx.socket(zmq.SUB)
        s.setsockopt(zmq.SUBSCRIBE, b"")
        s.setsockopt(zmq.RCVHWM, 4)
        s.setsockopt(zmq.LINGER, 0)
        s.setsockopt(zmq.RCVTIMEO, 500)
        s.connect(self.cfg.endpoint(port))
        return s

    # ----------------------------------------------------------------- reads
    def _telemetry_loop(self) -> None:
        sock = self._sub(self.cfg.telemetry_port)
        try:
            while not self._stop.is_set():
                try:
                    raw = sock.recv()
                except zmq.Again:
                    continue
                try:
                    msg = json.loads(raw.decode("utf-8", "replace"))
                except json.JSONDecodeError:
                    continue
                msg["_received_at"] = time.time()
                self.telemetry = msg
                self.last_telemetry_at = msg["_received_at"]
                self.telemetry_history.append(msg)
                for cb in list(self.on_telemetry):
                    try:
                        cb(msg)
                    except Exception:
                        pass
        finally:
            sock.close(0)

    def _decode_frame(self, raw: bytes, planes: int) -> tuple[int, int, int, list[np.ndarray]]:
        if len(raw) < HEADER.size:
            raise ValueError(f"short message: {len(raw)} bytes")
        cycle, w, h = HEADER.unpack_from(raw, 0)
        need = HEADER.size + planes * w * h * 4
        if len(raw) != need:
            raise ValueError(
                f"length mismatch: got {len(raw)}, expected {need} for {planes}x{w}x{h} float32"
            )
        out = []
        off = HEADER.size
        stride = w * h * 4
        for _ in range(planes):
            arr = np.frombuffer(raw, dtype="<f4", count=w * h, offset=off)
            out.append(arr.reshape(h, w))
            off += stride
        return cycle, w, h, out

    def _snapshot_loop(self) -> None:
        sock = self._sub(self.cfg.snapshot_port)
        try:
            while not self._stop.is_set():
                try:
                    raw = sock.recv(copy=True)
                except zmq.Again:
                    continue
                try:
                    cycle, w, h, planes = self._decode_frame(raw, 1)
                except ValueError:
                    continue
                frame = Frame(cycle, w, h, planes[0], time.time(), "density")
                self.density = frame
                for cb in list(self.on_frame):
                    try:
                        cb(frame)
                    except Exception:
                        pass
        finally:
            sock.close(0)

    def _stress_loop(self) -> None:
        sock = self._sub(self.cfg.stress_port)
        try:
            while not self._stop.is_set():
                try:
                    raw = sock.recv(copy=True)
                except zmq.Again:
                    continue
                try:
                    cycle, w, h, planes = self._decode_frame(raw, 3)
                except ValueError:
                    continue
                now = time.time()
                self.stress = {
                    name: Frame(cycle, w, h, plane, now, f"stress_{name}")
                    for name, plane in zip(("xx", "yy", "xy"), planes)
                }
        finally:
            sock.close(0)

    def _ack_loop(self) -> None:
        sock = self._sub(self.cfg.ack_port)
        try:
            while not self._stop.is_set():
                try:
                    raw = sock.recv()
                except zmq.Again:
                    continue
                try:
                    msg = json.loads(raw.decode("utf-8", "replace"))
                except json.JSONDecodeError:
                    continue
                msg["_received_at"] = time.time()
                if "health" in msg:
                    self.health = msg["health"]
                elif "injection_id" in msg:
                    self.injections.append(msg)
                    with self._ack_event:
                        self.acks.append(msg)
                        self._ack_event.notify_all()
                else:
                    with self._ack_event:
                        self.acks.append(msg)
                        self._ack_event.notify_all()
                for cb in list(self.on_ack):
                    try:
                        cb(msg)
                    except Exception:
                        pass
        finally:
            sock.close(0)

    # ---------------------------------------------------------------- writes
    def send(self, command: dict, wait_for_ack: bool = True) -> CommandResult:
        """Publish one command. Blocks briefly for its ack.

        The returned CommandResult says confirmed=True only if the daemon
        actually answered on 5559. No ack means we do not know whether it
        landed, and the result says so.
        """
        name = command.get("cmd")
        if not name:
            raise ValueError("command needs a 'cmd' field")

        payload = json.dumps(command).encode()

        with self._write_lock:
            delay = self._cmd_ready_at - time.time()
            if delay > 0:
                time.sleep(delay)
            mark = len(self.acks)
            sent_at = time.time()
            self._cmd_sock.send(payload)

            if not wait_for_ack:
                result = CommandResult(command, sent_at, False, None,
                                       "sent, ack not awaited")
                self.command_log.append(result)
                return result

            match = self._await_ack(name, mark, self.cfg.ack_timeout_s)

        if match is None:
            result = CommandResult(
                command, sent_at, False, None,
                f"no ack within {self.cfg.ack_timeout_s:g}s — daemon may be down, "
                f"the command may have been rejected, or it landed unacknowledged",
            )
        else:
            # The ACK carries a status field and the daemon uses it — a failed
            # save_checkpoint acks status "error". Treating every ack as success
            # is the same mistake as trusting a set_* that was silently ignored.
            status = match.get("status")
            if status is not None and status != "ok":
                result = CommandResult(
                    command, sent_at, False, match,
                    f"the daemon answered but reported status {status!r} — it "
                    f"received the command and did not succeed",
                )
            else:
                result = CommandResult(command, sent_at, True, match, "acked by daemon")
        self.command_log.append(result)
        return result

    def _await_ack(self, name: str, mark: int, timeout: float) -> dict | None:
        deadline = time.time() + timeout
        # inject_density answers with an injection record, not an "ack" field.
        want_injection = name == "inject_density"
        with self._ack_event:
            while True:
                for msg in list(self.acks)[mark:]:
                    if want_injection and "injection_id" in msg:
                        return msg
                    if not want_injection and msg.get("ack") == name:
                        return msg
                remaining = deadline - time.time()
                if remaining <= 0:
                    return None
                self._ack_event.wait(remaining)

    # ---------------------------------------------------------------- status
    def telemetry_interval(self) -> int:
        """Cycles between telemetry messages, measured rather than assumed."""
        hist = list(self.telemetry_history)[-12:]
        deltas = [
            b["cycle"] - a["cycle"]
            for a, b in zip(hist, hist[1:])
            if isinstance(a.get("cycle"), int) and isinstance(b.get("cycle"), int)
            and b["cycle"] > a["cycle"]
        ]
        if not deltas:
            return 10
        return min(deltas)

    def status(self) -> dict:
        now = time.time()
        tel_age = now - self.last_telemetry_at if self.last_telemetry_at else None
        return {
            "connected": tel_age is not None and tel_age < 5.0,
            "telemetry_age_s": tel_age,
            "telemetry": self.telemetry,
            "density_cycle": self.density.cycle if self.density else None,
            "density_age_s": (now - self.density.received_at) if self.density else None,
            "stress_cycle": self.stress["xx"].cycle if self.stress else None,
            "health": self.health,
            "injections": list(self.injections)[-10:],
            "recent_commands": [c.to_dict() for c in list(self.command_log)[-20:]],
        }
