# Resonance Engine — Lab

Everything needed to build, run and drive a Khra'gixx lattice from a browser.
Clone this, build it, run two commands, open a page.

---

## What this is

A 1024×1024 lattice-Boltzmann fluid running continuously on a GPU, driven by two
travelling waves, plus a browser interface for watching it and asking questions
of it.

It was built to explore a hypothesis: that reality is a discrete nodal network
rather than a continuum — nodes as places where standing waves can form, the
wave carrying the identity and the node only anchoring it; node density
determining what can happen locally, with gravity as a gradient in that density
rather than curvature of a background; and nothing meaning anything except in
comparison with something else.

That frame is a hypothesis under test. Nothing here depends on you accepting it.

The interface exists because a raw picture of this medium is useless — the
forcing carries about 99% of the field's variance, so every absolute view is
corduroy and anything underneath it is invisible. The views here get the pump
out of the way and put relationships on the screen instead.

> **First real run: passed.** Built and run against the actual daemon on an RTX
> 4090 — telemetry, snapshot decoding at 1024×1024, `save_state` and
> `load_state`, parameter changes, injection, a sweep to CSV, all confirmed on
> the real wire rather than against the mock.
>
> Two things are still unverified. Nobody has built it on an A100 yet, though
> the `sm_80` path compiles. And the analysis views have never met a real
> signal: on an unseeded world **drive removed** leaves noise, because there is
> nothing in there to find. The planted structures are a feature of the mock.

---

## Before you start

| Need | Check | If missing |
|---|---|---|
| NVIDIA GPU + driver | `nvidia-smi` | — |
| CUDA toolkit | `nvcc --version` | `module load cuda` on a cluster, or install the toolkit |
| ZeroMQ headers | `ls /usr/include/zmq.h` | `apt install libzmq3-dev` (or `module load zeromq`) |
| NVML | ships with the driver | — |
| Python 3.10+ | `python3 --version` | — |
| A model, for the chat pane | `ollama list` | optional — see [The Navigator](#the-navigator) |

Linux or WSL. The build script tells you which of these is missing before it
starts compiling, so if you are not sure, just run it.

---

## Get it

```bash
git clone https://github.com/Scruff-AI/Resonance_Lab.git
cd Resonance_Lab
pip install -r lab/requirements.txt
```

On Ubuntu 23.10 and later, `pip` refuses to install outside a virtual
environment (PEP 668). Either use a venv:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r lab/requirements.txt
```

or override it: `pip install --break-system-packages -r lab/requirements.txt`.

---

## Build

```bash
scripts/build_portable.sh
```

It reads your compute capability off `nvidia-smi` and compiles for it — `sm_80`
on an A100, `sm_89` on a 4090, `sm_90` on an H100. With several cards it takes
the **lowest** so one binary runs on all of them. CUDA comes from `$CUDA_HOME`
or `PATH`. Output lands in `build/`.

On a login node with no GPU visible, name the target yourself:

```bash
RESONANCE_ARCH=sm_80 scripts/build_portable.sh     # A100
```

There is a second variant that publishes extra streams — per-cell stress on
5560 and a coarse 32×32×6 field on 5561:

```bash
scripts/build_portable.sh observer
```

The physics is identical in both. `cuda/khra_gixx_1024_v5.cu` is the engine and
should not be edited — if you change it, you are no longer comparing against
anything anyone else has run.

---

## Run

Two processes, both on the **same machine**. The daemon binds `127.0.0.1`, so
nothing reaches it from another host.

```bash
./build/khra_gixx_1024_v5          # one terminal
scripts/run_lab.sh                  # another
```

Then open <http://127.0.0.1:8800>.

**From a laptop**, tunnel the web port to wherever the daemon is running:

```bash
ssh -N -L 8800:127.0.0.1:8800 you@the-gpu-node
```

and open `http://127.0.0.1:8800` locally.

**On a scheduled cluster**, both processes must live inside the same
allocation — the lab server talks to the daemon over loopback. Something like:

```bash
srun --gres=gpu:1 --pty bash
# inside the allocation:
./build/khra_gixx_1024_v5 &
scripts/run_lab.sh
```

then tunnel to the compute node.

### Two A100s, one node (optional)

If the machine has two GPUs, you can put the lattice on one and the model on the
other. Only two processes touch a GPU — the daemon and Ollama — so pin those and
leave the lab (pure CPU) alone:

```bash
CUDA_VISIBLE_DEVICES=1 ./build/khra_gixx_1024_v5   # lattice on the second GPU
CUDA_VISIBLE_DEVICES=0 ollama serve                # model on the first GPU
scripts/run_lab.sh                                  # UI (CPU), same host
```

Everything still talks over `127.0.0.1`, so this is for two cards in **one
machine** — not two separate machines.


### No GPU to hand?

```bash
python -m lab.tools.mock_daemon      # one terminal
scripts/run_lab.sh                    # another
```

The mock speaks the real wire protocol on the real ports and reproduces the
daemon's awkward behaviours on purpose. It also carries **three hidden
structures** — two breathing on the same slow signal, one independent, none
visible in the raw view — so the co-variation view can be checked against a
known answer instead of admired. Worth ten minutes before you book time on the
real thing.

---

## Your first ten minutes

1. Watch **Readings** until the step counter is climbing. That means the daemon
   is alive and the client is decoding it.
2. Switch the view to **raw**. You will see stripes and nothing else. That is
   correct — it is the forcing.
3. Switch to **drive removed**. The caption names the modes it took out and
   what share of the variance they carried. Whatever is left is not the pump.
4. Switch to **what moves with a place** and let the buffer fill for a minute or
   two — `frames buffered` is shown under the picture. Then click somewhere.
   Everywhere that moves with that spot goes red. The place you clicked reads
   +1 by construction.
5. Switch to **space–time**. Anything travelling leans; anything alternating in
   place chequers; anything persistent runs straight down.
6. Ask the Navigator what it notices. It gets the picture you are looking at,
   with the scale it was drawn at.

Every frame is captioned with what it is **and what it is measured against**.
That caption is not decoration — the fastest channel in this medium flips sign
every second step, so what you can conclude depends entirely on what was
sampled and how it was reduced.

---

## Saving and resuming

If your time on the machine is limited, this is the part that matters.

**Checkpoints** hold the entire lattice — about 36 MB at 1024². The
**Checkpoints** panel lists writable destinations including removable media,
with free space shown, so you can save straight to a USB stick and walk away
with the run. It waits for the write to finish and reports a truncated file as
a failure rather than a green tick.

**Load** puts a saved world back and picks up where you left off.

**Sweeps** write a CSV per run into the same place, one row per grid point with
the settings that produced it.

---

## The Navigator

A chat pane pointed at a model of your choosing.

```bash
MODEL_BACKEND=ollama scripts/run_lab.sh     # default; lists whatever you have pulled
MODEL_BACKEND=none   scripts/run_lab.sh     # views and controls only, no chat

MODEL_BACKEND=openai \
OPENAI_BASE_URL=https://api.openai.com/v1 \
RESONANCE_LAB_API_KEY=<your key> \
  scripts/run_lab.sh
```

The `openai` backend speaks the OpenAI-compatible API, so it also works against
Together, Groq, OpenRouter, vLLM or llama.cpp's server — point `OPENAI_BASE_URL`
at whichever. Models are listed from the backend and picked in the browser. No
key is stored in this repo or written to disk by it.

A vision-capable model gets the current view attached to every message, with the
scale it was drawn at. A text-only model gets the readings and the caption.

**It knows what it is inside** — a synthetic universe, and the nodal hypothesis
above — and is told plainly that this is a hypothesis under test and it is not
there to defend it. It is not told anything else: no channel is labelled, no
view is explained, and the readings reach it relabelled, so it sees "reading A"
rather than "coherence". Hand a model a measurement's name and it reasons about
the name.

It has no controls. You drive; it watches, and it can ask for a different view,
a shorter interval or a longer look.

---

## What this does not do

- **The lattice is 1024², hardcoded** as `#define NX/NY` in the CUDA. 2048² is
  not implemented.
- **One GPU.** A second card sits idle; there is no multi-GPU support.
- **No authentication.** The web server binds loopback by default. Do not move
  it to `0.0.0.0` on a shared machine — the API accepts commands that change the
  running world.

---

## Traps in the daemon this client works around

All verified in `cuda/khra_gixx_1024_v5.cu`, not assumed. They matter if you
write your own client.

**Out-of-range `set_*` values are ignored and acknowledged anyway.** `set_omega`,
`set_khra_amp` and `set_gixx_amp` check their range, do nothing if the value
falls outside, log nothing — and still send back `status: "ok"`. On the wire an
ignored set is indistinguishable from a successful one. This client carries
every bound from the handler and refuses out-of-range values before sending.

**`inject_density` sigma and strength are sticky.** Omit either and the
injection silently reuses the previous one's value. This client always sends
both.

**`save_state` takes a directory, not a file path.** The daemon builds the
filename itself.

**An acknowledgement is not a success.** The `status` field is real and the
daemon uses it.

**Every publisher has a send high-water mark of 1.** A stalled client loses
frames with no error anywhere — the only evidence is a jump in the cycle
counter. The buffer measures those gaps and says so on the picture.

**Port 5557 is PUB→SUB.** A `PUSH` or `REQ` socket connects successfully and
delivers nothing at all.

---

## Reference

| Port | Direction | Contents |
|---|---|---|
| 5556 | daemon → you | telemetry JSON, one object every 10 cycles |
| 5557 | you → daemon | command JSON (PUB→SUB) |
| 5558 | daemon → you | `uint32 cycle, uint16 w, uint16 h`, then `w*h` float32 ρ |
| 5559 | daemon → you | acknowledgements, injection records, health |
| 5560 | daemon → you | same header, then sxx, syy, sxy — observer build only |
| 5561 | daemon → you | coarse 32×32×6 — observer build only |

```
cuda/       the engine. Do not edit.
lab/        the client — bridge, views, sweeps, server, UI
scripts/    build and run
```

Deeper notes on the views, the sweep design and the client internals are in
[`lab/README.md`](lab/README.md).

The theory, the papers and the parameter-sweep findings live in the main engine
repository: <https://github.com/Scruff-AI/Resonance_Engine>.
