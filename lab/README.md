# Resonance Lab

A browser interface to a running Khra'gixx lattice: analysis views that show
something other than the forcing, parameter sweeps, checkpointing, and a chat
pane pointed at whatever model you have.

**It changes no physics.** Everything here is a client of the daemon's existing
ZMQ ports. `cuda/` is untouched, and no command exists here that
`khra_gixx_1024_v5` did not already accept.

> **Verified against a real daemon on an RTX 4090:** telemetry, snapshot
> decoding, save_state/load_state, parameter changes, injection and a sweep
> all confirmed on the real wire. Still unverified: no A100 build has been run,
> and the analysis views have never met a real signal — on an unseeded world
> drive removed leaves noise, because there is nothing planted to find.

---

## Getting it running

### 1. Build the daemon for your GPU

The original build scripts hardcode `-arch=sm_89` (an RTX 4090) and absolute
Windows/WSL paths. Use the portable one:

```bash
scripts/build_portable.sh              # detects your compute capability
scripts/build_portable.sh observer     # the fork with the extra 5560/5561 streams
```

It reads compute capability off `nvidia-smi`, takes the **lowest** if several
cards are present so one binary runs on all of them, resolves CUDA from
`$CUDA_HOME` or `PATH`, and writes to `build/`.

On a login node with no GPU visible, name the target yourself:

```bash
RESONANCE_ARCH=sm_80 scripts/build_portable.sh     # A100
RESONANCE_ARCH=sm_90 scripts/build_portable.sh     # H100
```

Needs `libzmq` (`apt install libzmq3-dev`) and NVML, which ships with the
driver. The script checks both before compiling. Built and verified on an RTX 4090 (sm_89), on the observer variant, and with
RESONANCE_ARCH=sm_80 for the A100 target. On some systems the link needs
-l:libstdc++.so.6, which the script now passes.

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

The daemon binds on `127.0.0.1`, so **the lab server must run on the same host
as the daemon.** From a laptop, tunnel the HTTP port:

```bash
ssh -N -L 8800:127.0.0.1:8800 you@the-gpu-node
```

and open <http://127.0.0.1:8800>.

### Testing with no GPU at all

```bash
python -m lab.tools.mock_daemon      # one shell
scripts/run_lab.sh                   # another
```

The mock speaks the real wire protocol on the real ports and reproduces the
daemon's awkward behaviours deliberately — the silent out-of-range ignores, the
sticky injection parameters, stress only on request. It also carries **three
hidden structures**: two that breathe on the same slow signal, one that has its
own. None are visible in the raw view. That is there so the co-variation view
can be checked against a known answer rather than admired.

---

## The views

A picture of raw ρ is a picture of the pump. The forcing carries about 99% of
the field's variance, so any absolute view is corduroy and nothing underneath it
is visible. These exist to get it out of the way.

| View | What it is | What it is measured against |
|---|---|---|
| **what moves with a place** | correlation of every cell's history with the history of a place you click | the buffered window, −1 to +1, drive notched first |
| **drive removed** | the field with its strongest spatial modes taken out | the modes are named in the caption, with the share of variance they carried |
| **space–time** | one row of the lattice stacked downward over cycles | column averages removed, then the strip's own strongest modes notched |
| **change since baseline** | difference from a frame you pinned | the pinned frame, named by its step |
| **field minus its mean** | deviation from the frame's own average | that frame's mean |
| **raw** | absolute values | nothing |
| **stress xx / yy / xy** | publishes only on request | their own average |

**Reading space–time:** a thing that travels leans, a thing that alternates in
place chequers, a thing that persists runs straight down. It answers "is that
moving or just flickering" by looking, without anyone having to reason about
sampling intervals.

**Reading co-variation:** click a place, and everywhere that moves with it goes
red. The chosen patch reads +1 by construction — it is correlated with itself.
Two structures far apart both reading strongly is the interesting case.

Every frame is captioned with **what it is** and **what it is measured
against**, plus the scale, the reduction and the step. That caption is not
decoration. The lattice's fastest channel flips sign at period 2, so anything
you conclude from a picture depends on what was sampled and how it was reduced.
A screenshot without its ruler is not a measurement.

Colour scales are **locked per view** rather than recomputed per frame. A scale
that follows the frame hides anything holding steady, because the same colour
stops meaning the same value. *re-scale* sets a new one from the current frame.
Non-finite cells are painted **magenta**, so a NaN cascade is unmistakable.

`remove` sets how many of the strongest modes get notched before correlating.
Correlating raw frames measures the drive: every cell's history is dominated by
the travelling waves, so everything correlates with everything at the drive's
phase. On a synthetic field carrying a known correlated pair, raw frames put the
true partner at 0.05 and empty background at 0.49 — worse than useless. Notching
first put the partner at 0.999, an independent structure at −0.06 and background
at −0.35.

---

## Sweeps

The results in the main engine repository came from walking a grid of `omega`,
`khra_amp` and `gixx_amp` and reading coherence and asymmetry at each point.
The Sweep panel automates that.

If you go looking at the hand-written sweep scripts in `Resonance_Engine`, note
that `scripts/ab_power_test.sh` is broken in two ways: it connects to 5557 with
a `zmq.PUSH` socket, which delivers nothing to the daemon's SUB, and it sets
`omega` 6.00, above the 1.99 cap. Do not copy that pattern.

The **Sweep** panel runs that: any combination of the three axes, optional
repeated injections at each point, settling counted in the world's own steps
rather than in seconds, every point written to CSV alongside the settings that
produced it. The whole grid is range-checked before the first point runs, so a
sweep cannot spend an hour silently ignoring half its values.

---

## Choosing a model

```bash
MODEL_BACKEND=ollama scripts/run_lab.sh        # default; lists whatever you have pulled
MODEL_BACKEND=none   scripts/run_lab.sh        # views and controls only

MODEL_BACKEND=openai \
OPENAI_BASE_URL=https://api.openai.com/v1 \
RESONANCE_LAB_API_KEY=<your key> \
  scripts/run_lab.sh
```

The `openai` backend speaks the OpenAI-compatible chat API, so it also works
against Together, Groq, vLLM, llama.cpp's server, OpenRouter and anything with
that shape. Models are listed from the backend and picked in the browser, not in
the source. No key is stored in this repo or written to disk by it — the backend
reads whatever environment variable `openai_api_key_env` names.

A vision-capable model gets the current view attached to each message, with the
scale it was drawn at. A text-only model gets the readings and the caption.

**What the Navigator knows.** It is told what kind of thing it is inside — a
synthetic universe running continuously — and the hypothesis the world was built
on: reality as a discrete nodal network, nodes as anchor points where standing
waves form, gravity as a gradient in node density, nothing meaning anything
except in comparison with something else. It is told plainly that this is a
hypothesis under test and that it is not there to defend it.

It is not told anything else. No channel is labelled and no view is explained,
and the readings reach it relabelled — it sees "reading A", not "coherence" —
because a model handed a measurement's name reasons about the name. You still
see the real names in your own panel.

It has no commands. You drive; it watches, and it can ask for a different view,
a shorter interval or a longer look.

---

## Traps in the daemon that this client works around

These are verified in `cuda/khra_gixx_1024_v5.cu`, not assumed.

**Out-of-range `set_*` values are ignored and acked anyway.** `set_omega`,
`set_khra_amp` and `set_gixx_amp` check their range, do nothing if the value
falls outside, log nothing — and the generic ACK on 5559 still goes out naming
the command with `status: "ok"`. On the wire an ignored set is indistinguishable
from a successful one. `lab/commands.py` carries every bound from the handler
and refuses out-of-range values before they are sent.

**`inject_density` sigma and strength are sticky statics.** Omit either and the
injection silently reuses the previous one's value. This client always sends
both.

**`save_state` takes a directory, not a file path.** The daemon builds the
filename itself (`ckpt_<stamp>_c<cycle>.bin`). Passing a file path makes it
`fopen("<file>/ckpt_....bin")`, which fails with `ENOTDIR` and acks
`status: "error"`. The lab sends the directory and then finds the file the
daemon actually created.

**An ACK is not a success.** The status field is real and the daemon uses it. A
command is reported *acked* only on a matching message with `status: "ok"`;
anything else says *unconfirmed* with the reason. It never says "sent" and means
"arrived".

**Every publisher has a send high-water mark of 1.** If this client stalls, the
daemon drops older frames with no error anywhere — the only evidence is a jump
in the cycle header. The buffer measures those gaps and the co-variation view
says on the picture when its window is not continuous, rather than quietly
correlating across a hole.

**Port 5557 is PUB→SUB.** A `PUSH` or `REQ` socket connects successfully and
delivers nothing. PUB→SUB also drops anything sent before the subscription
handshake settles, so the bridge connects at startup and waits before its first
send. If you write your own client, do both, or your commands vanish without a
trace.

---

## Checkpoints

*Save* writes into a directory you choose, including removable media, with free
space shown — a 1024² checkpoint is about 36 MB (1024² × 9 populations ×
float32, plus a 64-byte header). It waits for the file size to settle and
reports a truncated write as a failure rather than a green tick. *Load* replaces
the running world and asks first.

---

## Layout

```
lab/
  config.py       every setting, overridable by environment variable
  bridge.py       ZMQ threads; the wire formats, documented from source
  commands.py     the daemon's command surface with its real ranges
  views.py        drive removal, space-time, baseline, co-variation
  sweep.py        parameter grids, settled on the world's own clock
  render.py       float32 plane -> PNG, always carrying its scale
  storage.py      save targets including removable media
  models.py       Ollama and OpenAI-compatible backends
  navigator.py    the Navigator's prompt and the first-run card
  server.py       FastAPI: REST + two websockets
  static/         the single-page UI
  tools/          mock daemon with three hidden structures
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
| 5560 | daemon PUB → us | same header, then sxx, syy, sxy contiguous |
| 5561 | daemon PUB → us | observer fork only: coarse 32×32×6 |
