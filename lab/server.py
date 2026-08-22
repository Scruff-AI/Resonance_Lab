"""Resonance Lab — web server.

Serves one page that shows the live lattice, drives the daemon, and lets a
model of your choosing watch and propose. Run it on the same host as the
daemon (the daemon binds 127.0.0.1) and tunnel the HTTP port to your laptop.

    python -m lab.server
    ssh -N -L 8800:127.0.0.1:8800 you@the-gpu-node
"""

from __future__ import annotations

import asyncio
import os
import json
import time
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from lab import commands as cmdspec
from lab import models as modelbackends
from lab import navigator as nav
from lab import storage
from lab import views as fieldviews
from lab.sweep import SweepRunner, SweepSpec, Injection, SWEEPABLE
from lab import render as renderer
from lab.bridge import DaemonBridge
from lab.config import CONFIG

STATIC = Path(__file__).parent / "static"

app = FastAPI(title="Resonance Lab", version="0.1.0")
bridge = DaemonBridge(CONFIG)
backend = modelbackends.make_backend(CONFIG)

CHAT: list[dict] = []

SYSTEM_PROMPT = nav.NAVIGATOR_PROMPT


class CommandIn(BaseModel):
    command: dict


class ChatIn(BaseModel):
    message: str
    model: str = ""
    attach_field: bool = True
    view: str = "deviation"


class StateIn(BaseModel):
    name: str = ""
    path: str = ""
    dir: str = ""


@app.on_event("startup")
async def _startup() -> None:
    bridge.start()


@app.on_event("shutdown")
async def _shutdown() -> None:
    bridge.stop()


# ----------------------------------------------------------- telemetry labelling
# The Navigator is not told the project's names for these readings. "coherence",
# "omega", "khra_amp" carry a theory with them, and a model handed those words
# will reason about the words. Peppo sees the real names in his own panel; the
# Navigator sees neutral ones and has to describe what the numbers do.
NEUTRAL_LABELS = [
    ("cycle", "step"),
    ("coherence", "reading A"),
    ("asymmetry", "reading B"),
    ("omega", "setting 1"),
    ("khra_amp", "setting 2"),
    ("gixx_amp", "setting 3"),
    ("vel_mean", "reading C (mean)"),
    ("vel_max", "reading C (max)"),
    ("vel_var", "reading C (spread)"),
    ("vorticity_mean", "reading D"),
    ("stress_xx", "reading E1"),
    ("stress_yy", "reading E2"),
    ("stress_xy", "reading E3"),
]


def neutral_telemetry(tel: dict) -> list[tuple[str, object]]:
    out = []
    for key, neutral in NEUTRAL_LABELS:
        if key in tel:
            label = neutral if CONFIG.neutral_telemetry_labels else key
            out.append((label, tel[key]))
    return out


# ----------------------------------------------------------------- field views
STORE = fieldviews.FrameStore()
SEED = {"x": 0.5, "y": 0.5}


def _on_frame(frame):
    if frame.kind == "density":
        STORE.add(frame.cycle, frame.data)


bridge.on_frame.append(_on_frame)

SWEEP = SweepRunner(bridge, CONFIG.state_dir)


def _plane(view: str):
    """Return a ViewResult for a view name, or None if it cannot be drawn yet."""
    d = bridge.density

    if view == "space_time":
        return fieldviews.view_space_time(STORE)

    if view == "covariation":
        return fieldviews.view_covariation(STORE, SEED["x"], SEED["y"])

    if d is None:
        return None

    if view == "drive_removed":
        return fieldviews.view_drive_removed(d.data, d.cycle, n_modes=STORE.notch_modes)

    if view == "baseline":
        return fieldviews.view_baseline(d.data, d.cycle, STORE)

    if view == "deviation":
        return fieldviews.ViewResult(
            d.data - float(d.data.mean()), d.cycle,
            "field minus its own average", "the frame's own mean", True)

    if view == "raw":
        return fieldviews.ViewResult(d.data, d.cycle, "raw field",
                                     "absolute values", False)

    if view.startswith("stress_"):
        key = view.split("_", 1)[1]
        if not bridge.stress or key not in bridge.stress:
            return None
        f = bridge.stress[key]
        return fieldviews.ViewResult(f.data, f.cycle, f"stress {key}",
                                     "its own average", True)
    return None


def _render_view(view: str, cmap: str | None, size: int | None):
    res = _plane(view)
    if res is None:
        return None

    vmin = vmax = None
    if CONFIG.fixed_scale and view in LOCKED_SCALES:
        vmin, vmax = LOCKED_SCALES[view]

    default_cmap = "coolwarm" if res.diverging else "viridis"
    result = renderer.render(
        res.data, res.cycle,
        colormap=cmap or default_cmap,
        target_size=size or CONFIG.render_size,
        center_on_mean=res.diverging and view not in ("covariation",),
        vmin=vmin, vmax=vmax,
    )
    if CONFIG.fixed_scale and view not in LOCKED_SCALES and view not in ("space_time", "covariation"):
        span = (result.vmax - result.vmin) or 1e-12
        pad = span * 0.125
        LOCKED_SCALES[view] = (result.vmin - pad, result.vmax + pad)
        result.fixed_scale = True

    meta = result.meta()
    meta["view"] = view
    meta["quantity"] = res.quantity
    meta["against"] = res.reference
    meta["detail"] = res.detail
    return result, meta


# One locked colour scale per view. Computed from the first frame of that view
# and then held, so the same colour keeps meaning the same value across frames.
LOCKED_SCALES: dict[str, tuple[float, float]] = {}


# ------------------------------------------------------------------ HTTP API
@app.get("/api/status")
async def status():
    return {
        "daemon": bridge.status(),
        "model": await backend.health(),
        "config": {
            "daemon_host": CONFIG.daemon_host,
            "ports": {
                "telemetry": CONFIG.telemetry_port,
                "command": CONFIG.command_port,
                "snapshot": CONFIG.snapshot_port,
                "ack": CONFIG.ack_port,
                "stress": CONFIG.stress_port,
            },
            "require_approval": CONFIG.require_approval,
            "render_size": CONFIG.render_size,
            "state_dir": CONFIG.state_dir,
        },
        "views_available": {
            "density": bridge.density is not None,
            "deviation": bridge.density is not None,
            "stress_xx": bridge.stress is not None,
            "stress_yy": bridge.stress is not None,
            "stress_xy": bridge.stress is not None,
        },
    }


@app.get("/api/commands")
async def command_surface():
    return {"commands": cmdspec.describe()}


@app.post("/api/command")
async def send_command(body: CommandIn):
    try:
        validated = cmdspec.validate(body.command)
    except cmdspec.Rejected as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    result = await asyncio.to_thread(bridge.send, validated)
    return result.to_dict()


@app.get("/api/field.png")
async def field_png(view: str = "deviation", cmap: str = "", size: int = 0):
    got = await asyncio.to_thread(_render_view, view, cmap or None, size or None)
    if got is None:
        raise HTTPException(status_code=404,
                            detail=f"no data yet for view {view!r}")
    result, meta = got
    return Response(content=result.png, media_type="image/png",
                    headers={"X-Field-Meta": json.dumps(meta)})


@app.post("/api/scale/relock")
async def relock_scale(view: str = ""):
    """Drop a locked scale so the next frame of that view sets a new one."""
    if view:
        LOCKED_SCALES.pop(view, None)
    else:
        LOCKED_SCALES.clear()
    return {"locked": {k: list(v) for k, v in LOCKED_SCALES.items()}}


@app.post("/api/baseline")
async def set_baseline(clear: bool = False):
    """Pin the current frame as the reference everything is measured against."""
    if clear:
        STORE.clear_baseline()
        LOCKED_SCALES.pop("baseline", None)
        return {"baseline": None}
    d = bridge.density
    if d is None:
        raise HTTPException(status_code=404, detail="no frame to pin yet")
    STORE.set_baseline(d.cycle, d.data)
    LOCKED_SCALES.pop("baseline", None)
    return {"baseline": d.cycle}


@app.post("/api/seed")
async def set_seed(x: float, y: float):
    """Choose the place co-variation is measured from. Reads only."""
    SEED["x"] = max(0.0, min(1.0, x))
    SEED["y"] = max(0.0, min(1.0, y))
    LOCKED_SCALES.pop("covariation", None)
    return {"seed": SEED, "buffer_frames": STORE.depth()}


@app.post("/api/notch")
async def set_notch(modes: int):
    """How many of the strongest modes to take out before correlating."""
    STORE.notch_modes = max(0, min(12, int(modes)))
    STORE.small.clear()
    LOCKED_SCALES.pop("covariation", None)
    LOCKED_SCALES.pop("drive_removed", None)
    return {"notch_modes": STORE.notch_modes, "buffer_cleared": True}


@app.get("/api/buffer")
async def buffer_state():
    span = STORE.span()
    return {"frames": STORE.depth(), "span": list(span) if span else None,
            "notch_modes": STORE.notch_modes,
            "notched": [list(m) for m in STORE.notched],
            "baseline": STORE.baseline[0] if STORE.baseline else None,
            "seed": SEED, "row": STORE.row_index}


# ------------------------------------------------------------------- sweeps
class SweepIn(BaseModel):
    axes: dict[str, list[float]] = {}
    injection: dict | None = None
    settle_steps: int = 2000
    reset_between: bool = False
    label: str = ""


@app.post("/api/sweep/start")
async def sweep_start(body: SweepIn):
    axes = {}
    for name, values in body.axes.items():
        if name not in SWEEPABLE:
            raise HTTPException(status_code=400,
                                detail=f"{name} is not sweepable. Use one of {list(SWEEPABLE)}.")
        if not values:
            continue
        # Check every value against the daemon's real range up front, so a
        # sweep does not run for an hour silently ignoring half its grid.
        for v in values:
            try:
                cmdspec.validate({"cmd": f"set_{name}", "value": v})
            except cmdspec.Rejected as exc:
                raise HTTPException(status_code=400, detail=str(exc))
        axes[name] = [float(v) for v in values]

    inj = None
    if body.injection:
        inj = Injection(**body.injection)
        try:
            cmdspec.validate({"cmd": "inject_density", "x": inj.x, "y": inj.y,
                              "sigma": inj.sigma, "strength": inj.strength})
        except cmdspec.Rejected as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    spec = SweepSpec(axes=axes, injection=inj,
                     settle_steps=max(0, body.settle_steps),
                     reset_between=body.reset_between, label=body.label)
    if not spec.points():
        raise HTTPException(status_code=400, detail="nothing to sweep")
    try:
        return SWEEP.start(spec)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc))


@app.get("/api/sweep")
async def sweep_status():
    return SWEEP.status()


@app.post("/api/sweep/stop")
async def sweep_stop():
    return SWEEP.stop()


@app.get("/api/telemetry/history")
async def telemetry_history(last_n: int = 600):
    hist = list(bridge.telemetry_history)[-max(1, min(last_n, 3000)):]
    return {"count": len(hist), "rows": hist}


# --------------------------------------------------------------- state files
@app.get("/api/save-targets")
async def save_targets():
    grid = (bridge.telemetry or {}).get("grid", 1024)
    return {"targets": storage.targets(CONFIG.state_dir),
            "checkpoint_bytes": storage.checkpoint_bytes(int(grid))}


def _state_dirs() -> list[Path]:
    dirs = [Path(CONFIG.state_dir)]
    for t in storage.targets(CONFIG.state_dir):
        p = Path(t["path"])
        if p not in dirs:
            dirs.append(p)
    return dirs


@app.get("/api/states")
async def list_states():
    rows = []
    for d in _state_dirs():
        try:
            entries = sorted(d.glob("*.khrg"), key=lambda x: x.stat().st_mtime, reverse=True)
        except OSError:
            continue
        for p in entries[:40]:
            rows.append({"name": p.name, "path": str(p),
                         "bytes": p.stat().st_size, "mtime": p.stat().st_mtime,
                         "dir": str(d)})
    rows.sort(key=lambda r: r["mtime"], reverse=True)
    return {"state_dir": CONFIG.state_dir, "states": rows}


@app.post("/api/state/save")
async def save_state(body: StateIn):
    tel = bridge.telemetry or {}
    cycle = tel.get("cycle", "unknown")
    grid = int(tel.get("grid", 1024))
    directory = body.dir.strip() or CONFIG.state_dir

    name = body.name.strip() or f"state_c{cycle}.khrg"
    if "/" in name or "\\" in name:
        raise HTTPException(status_code=400, detail="name must not contain a path separator")

    d = Path(directory)
    if not d.is_dir():
        raise HTTPException(status_code=400, detail=f"{directory} is not a directory")
    if not os.access(d, os.W_OK):
        raise HTTPException(status_code=400, detail=f"{directory} is not writable")

    needed = storage.checkpoint_bytes(grid)
    warning = storage.check_space(directory, needed)
    if warning:
        raise HTTPException(status_code=400, detail=warning)

    target = str(d / name)
    result = await asyncio.to_thread(bridge.send, {"cmd": "save_state", "path": target})
    out = result.to_dict()
    out["target"] = target

    # An ack is not proof the bytes landed, and a half-written checkpoint on a
    # full stick still acks. Wait for the size to settle, then check it.
    p = Path(target)
    settled = 0
    for _ in range(20):
        await asyncio.sleep(0.4)
        if not p.exists():
            continue
        size = p.stat().st_size
        if size == settled and size > 0:
            break
        settled = size
    out["file_exists"] = p.exists()
    out["file_bytes"] = p.stat().st_size if p.exists() else 0
    out["expected_bytes"] = needed
    if out["file_exists"] and out["file_bytes"] < needed:
        out["detail"] += (f" — the file is {out['file_bytes']} bytes but a "
                          f"{grid}² checkpoint is {needed}. It is truncated.")
        out["file_exists"] = False
    elif not out["file_exists"]:
        out["detail"] += (" — no file at the target path. If the daemon runs "
                          "elsewhere it may have written relative to its own "
                          "working directory.")
    return out


@app.post("/api/state/load")
async def load_state(body: StateIn):
    if not body.path.strip():
        raise HTTPException(status_code=400, detail="path is required")
    result = await asyncio.to_thread(bridge.send, {"cmd": "load_state", "path": body.path})
    return result.to_dict()


# ------------------------------------------------------------------------ chat
@app.get("/api/models")
async def list_models():
    try:
        return {"backend": backend.name, "models": await backend.list_models()}
    except Exception as exc:
        return JSONResponse(
            status_code=200,
            content={"backend": backend.name, "models": [],
                     "error": f"{type(exc).__name__}: {exc}"},
        )


@app.get("/api/chat")
async def chat_history():
    return {"messages": CHAT[-200:]}


@app.post("/api/chat")
async def chat(body: ChatIn):
    tel = bridge.telemetry or {}
    context_lines = [
        f"You are being shown the system's readings. They arrive every "
        f"{bridge.telemetry_interval()} steps of its clock."
    ]
    if tel:
        for label, value in neutral_telemetry(tel):
            context_lines.append(f"  {label} = {value}")
    else:
        context_lines.append("  (nothing is arriving — the system may not be running)")

    images: list[bytes] = []
    if body.attach_field:
        got = await asyncio.to_thread(_render_view, body.view, None, None)
        if got is not None:
            result, meta = got
            images.append(result.png)
            context_lines.append(
                f"An image is attached. It was taken at step {meta['cycle']}, "
                f"reduced from {meta['source_size'][0]}x{meta['source_size'][1]} to "
                f"{meta['size'][0]}x{meta['size'][1]} by {meta['reduction']}. "
                f"Its colour scale runs from {meta['vmin']:.6g} to {meta['vmax']:.6g}"
                + (" and is held fixed between frames, so the same colour always "
                   "means the same value." if meta.get("fixed_scale")
                   else " and is recomputed for each frame, so the same colour "
                        "does NOT mean the same value between frames.")
            )
        else:
            context_lines.append("No image was available to attach.")

    user_turn = {"role": "user",
                 "content": f"{body.message}\n\n" + "\n".join(context_lines)}
    history = [{"role": m["role"], "content": m["content"]} for m in CHAT[-12:]
               if m["role"] in ("user", "assistant")]
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history + [user_turn]

    CHAT.append({"role": "user", "content": body.message, "ts": time.time(),
                 "cycle": tel.get("cycle")})

    try:
        reply = await backend.chat(messages, body.model,
                                   images=images or None,
                                   temperature=CONFIG.model_temperature,
                                   timeout=CONFIG.model_timeout_s)
    except Exception as exc:
        detail = f"{type(exc).__name__}: {exc}"
        CHAT.append({"role": "error", "content": detail, "ts": time.time()})
        raise HTTPException(status_code=502, detail=detail)

    CHAT.append({"role": "assistant", "content": reply, "ts": time.time(),
                 "cycle": tel.get("cycle"),
                 "attached": body.view if images else None})
    return {"reply": reply, "attached_field": bool(images)}


# -------------------------------------------------------------------- welcome
@app.get("/api/welcome")
async def welcome():
    return nav.WELCOME


# ------------------------------------------------------------------ websockets
@app.websocket("/ws/telemetry")
async def ws_telemetry(ws: WebSocket):
    await ws.accept()
    last = None
    try:
        while True:
            tel = bridge.telemetry
            if tel is not None and tel.get("_received_at") != last:
                last = tel.get("_received_at")
                await ws.send_json({"type": "telemetry", "data": tel,
                                    "status": bridge.status()})
            await asyncio.sleep(0.15)
    except (WebSocketDisconnect, RuntimeError):
        return


@app.websocket("/ws/field")
async def ws_field(ws: WebSocket):
    await ws.accept()
    view = "deviation"
    cmap = ""
    last_cycle = -1
    interval = 1.0 / max(0.5, CONFIG.render_fps_cap)
    try:
        while True:
            try:
                msg = await asyncio.wait_for(ws.receive_text(), timeout=interval)
                req = json.loads(msg)
                view = req.get("view", view)
                cmap = req.get("cmap", cmap)
                last_cycle = -1          # force a redraw on view change
            except asyncio.TimeoutError:
                pass
            except (json.JSONDecodeError, KeyError):
                pass

            got = await asyncio.to_thread(_render_view, view, cmap or None, None)
            if got is None:
                await ws.send_json({"type": "no_data", "view": view})
                await asyncio.sleep(0.5)
                continue
            result, meta = got
            if result.cycle == last_cycle:
                continue
            last_cycle = result.cycle
            await ws.send_json({"type": "field_meta", **meta})
            await ws.send_bytes(result.png)
    except (WebSocketDisconnect, RuntimeError):
        return


# ---------------------------------------------------------------------- static
@app.get("/")
async def index():
    return FileResponse(STATIC / "index.html")


app.mount("/static", StaticFiles(directory=STATIC), name="static")


def main() -> None:
    import uvicorn
    print(f"Resonance Lab on http://{CONFIG.http_host}:{CONFIG.http_port}")
    print(f"  daemon at {CONFIG.daemon_host}, telemetry {CONFIG.telemetry_port} / "
          f"commands {CONFIG.command_port} / snapshots {CONFIG.snapshot_port}")
    print(f"  model backend: {CONFIG.model_backend}")
    print(f"  writes into the field: {sorted(cmdspec.WRITES_FIELD)}")
    print(f"  fixed colour scale: {CONFIG.fixed_scale} | "
          f"neutral labels to the Navigator: {CONFIG.neutral_telemetry_labels}")
    uvicorn.run(app, host=CONFIG.http_host, port=CONFIG.http_port, log_level="info")


if __name__ == "__main__":
    main()
