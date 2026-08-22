# Resonance Lab

A browser interface to a running Khra'gixx lattice: live visualiser, direct
controls, checkpoint save/load, and a chat pane you point at whatever model you
have.

**It changes no physics.** Everything here is a client of the daemon's existing
ZMQ ports. `cuda/` is untouched, and no command exists here that the daemon did
not already accept. If you diff this branch against `master`, the only file
outside `lab/` and `scripts/` is nothing at all.

---

## Why the four things you asked for were mostly already there

Three of the four asks needed no engine work, and the fourth was already built:

| You asked for | What already existed |
|---|---|
| browser chat interface | nothing — this is the new part |
| lattice visualiser | port **5558** publishes the full 1024² density field as raw float32, every 10 cycles by default. This just draws it. |
| controls to poke the lattice | port **5557** already accepts `inject_density`, `set_omega`, `set_khra_amp`, `set_gixx_amp`, `reset_equilibrium` and more |
| dump/load lattice state | `save_state` and `load_state` have been in the daemon the whole time. They had no button. |

---

## Getting it running

### 1. Build the daemon for your GPU

The old build scripts hardcode `-arch=sm_89` (an RTX 4090) and absolute
Windows/WSL paths. Use the portable one instead:

```bash
scripts/build_portable.sh              # detects your compute capability
scripts/build_portable.sh observer     # the fork with the extra 5560/5561 streams
```

It reads the compute capability off `nvidia-smi`, takes the **lowest** if you
have several cards so one binary runs on all of them, resolves CUDA from
`$CUDA_HOME` or `PATH`, and writes to `build/` inside the repo.

On a login node with no GPU visible, name the target yourself:

```bash
RESONANCE_ARCH=sm_80 scripts/build_portable.sh     # A100
RESONANCE_ARCH=sm_90 scripts/build_portable.sh     # H100
```

Needs `libzmq` (`apt install libzmq3-dev`) and NVML, which ships with the
driver. The script checks both before it starts compiling and tells you which
is missing.

### 2. Start the daemon

```bash
./build/khra_gixx_1024_v5
```

It binds 5556 telemetry, 5557 commands, 5558 snapshots, 5559 acks, 5560 stress.

### 3. Start the lab

```bash
pip install -r lab/requirements.txt
scripts/run_lab.sh
```

The daemon binds its ports on `127.0.0.1`, so **the lab server has to run on the
same host as the daemon.** From a laptop, tunnel the HTTP port:

```bash
ssh -N -L 8800:127.0.0.1:8800 you@the-gpu-node
```

and open <http://127.0.0.1:8800>.

---

## Choosing a model

Set `MODEL_BACKEND` before starting:

```bash
MODEL_BACKEND=ollama scripts/run_lab.sh        # default; lists whatever you have pulled
MODEL_BACKEND=none   scripts/run_lab.sh        # visualiser and controls only

MODEL_BACKEND=openai \
OPENAI_BASE_URL=https://api.openai.com/v1 \
RESONANCE_LAB_API_KEY=sk-... \
  scripts/run_lab.sh
```

The `openai` backend speaks the OpenAI-compatible chat API, so it also works
against Together, Groq, vLLM, llama.cpp's server, OpenRouter and anything else
with that shape — point `OPENAI_BASE_URL` at it. Models are listed from the
backend and picked in the browser, not in the source.

A vision-capable model gets the current lattice view attached to each message,
along with the scale it was drawn at. A text-only model gets the telemetry and
the caption.

---

## The model does not touch the lattice

It proposes. You approve.

When the model wants to act it writes a line like

```
@cmd {"cmd": "inject_density", "x": 212, "y": 212, "sigma": 25, "strength": 0.01}
```

which appears in the **Proposals** panel with an approve and a reject button.
Nothing reaches port 5557 until you click approve. There is no setting that
turns this off, and that is deliberate: two writers on one live world land
inside each other's measurements, and you cannot tell afterwards which write
produced what.

---

## Two traps this UI protects you from

**1. The daemon acks commands it silently ignored.**

`set_omega`, `set_khra_amp` and `set_gixx_amp` check their range, do nothing if
the value is outside it — no error, no log line — and the generic ACK on 5559
still goes out naming the command. On the wire, an ignored set is
indistinguishable from a successful one.

`lab/commands.py` carries every bound as read out of the daemon's command
handler and refuses out-of-range values here, before they are sent. You get a
red refusal that says what the limit is, instead of a green tick over a
parameter that never moved.

**2. `inject_density` sigma and strength are sticky.**

They are static globals in the daemon. Send an injection without `sigma`, and it
silently reuses the sigma from the last injection — possibly from hours ago,
possibly someone else's. This client always sends both explicitly.

More generally: a command is reported **acked** only when a matching message
came back on 5559. Anything else says **unconfirmed** and gives the reason. It
never says "sent" and means "arrived". `save_state` goes further and stats the
file, because an ack is not proof the bytes landed.

---

## Reading the visualiser

Every frame is captioned with the scale it was drawn at:

```
rho − mean(rho) · cycle 41250 · scale [-6.966e-02, 6.966e-02] · coolwarm ·
1024×1024 → 512×512 by block mean 2x2
```

That caption is not decoration. The lattice's fastest channel flips sign at
period 2, so anything you conclude from a picture depends on what was sampled
and how it was reduced. A screenshot without its ruler is not a measurement.

- **density deviation (ρ − mean)** — the default. Absolute density sits near a
  constant, so on an absolute scale the structure is invisible; this subtracts
  the mean and uses a symmetric diverging scale.
- **density (ρ, absolute)** — the raw field.
- **stress xx / yy / xy** — publishes only on request. Press *request stress
  frame*; it arrives at the next telemetry tick.
- Non-finite cells are painted **magenta**, so a NaN cascade is unmistakable
  rather than looking like an odd colour.

Snapshots publish every 10 cycles by default. `set snapshot every` changes it;
`0` turns them off, which is worth doing if a run needs the clock clean.

---

## Click to inject

Tick **arm click-to-inject** and the lattice becomes a target. The circle that
follows the cursor is 2σ at the sigma you have set, so you can see the size of
what you are about to write before you write it. Clicking sends
`inject_density` at the lattice coordinate under the cursor.

It is off by default and re-arms nothing after a page reload. Writing into a
live world is a real act.

---

## Checkpoints

*Save* writes a KHRG v1 checkpoint into `build/states/` (about 37 MB: 1024² × 9
populations × float32, plus a 64-byte header) and then checks the file is
actually there before telling you it worked. Files in that directory are listed
with a *load* button.

Loading replaces the running world. It asks first.

---

## Testing without a GPU

```bash
python -m lab.tools.mock_daemon      # in one shell
scripts/run_lab.sh                   # in another
```

The mock speaks the real wire protocol on the real ports and reproduces the
awkward behaviours on purpose — the silent out-of-range ignores, the sticky
injection parameters, stress only on request. It is not physics; the field is
two travelling sinusoids and your writes decaying on top. It exists so the
client can be changed and verified without borrowing time on a real world.

---

## Layout

```
lab/
  config.py       every setting, overridable by environment variable
  bridge.py       ZMQ threads; the wire formats, documented from source
  commands.py     the daemon's command surface with its real ranges
  render.py       float32 plane -> PNG, always with its scale
  models.py       Ollama and OpenAI-compatible backends; proposal extraction
  server.py       FastAPI: REST + two websockets
  static/         the single-page UI
  tools/          mock daemon
scripts/
  build_portable.sh   architecture-detecting build
  run_lab.sh          start the UI
```

## Ports

| Port | Direction | Contents |
|---|---|---|
| 5556 | daemon PUB → us | telemetry JSON, one object every 10 cycles |
| 5557 | us PUB → daemon SUB | command JSON |
| 5558 | daemon PUB → us | `uint32 cycle, uint16 w, uint16 h`, then `w*h` float32 ρ |
| 5559 | daemon PUB → us | acks, injection records, health |
| 5560 | daemon PUB → us | same header, then sxx, syy, sxy |
| 5561 | daemon PUB → us | observer fork only: coarse 32×32×6 |

Port 5557 is PUB→SUB, which drops anything sent before the subscription
handshake settles. The bridge connects once at startup and waits before its
first send. If you write your own client, do the same, or your first command
will vanish without a trace.
