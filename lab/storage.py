"""Where checkpoints can be written.

Peppo works on university machines he only has for a while, so the useful
destination is usually a stick he can walk away with. This enumerates the
plausible targets and reports free space, because a 2048² checkpoint is about
150 MB and a full stick fails halfway through with a truncated file.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

# Where removable media normally appears on Linux.
REMOVABLE_ROOTS = ("/media", "/run/media", "/mnt")


def _free(path: Path) -> tuple[int, int]:
    try:
        usage = shutil.disk_usage(path)
        return usage.free, usage.total
    except OSError:
        return 0, 0


def _writable(path: Path) -> bool:
    return path.is_dir() and os.access(path, os.W_OK)


def _human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} TB"


def targets(default_dir: str) -> list[dict]:
    """Local default first, then anything removable that is writable."""
    out: list[dict] = []
    seen: set[str] = set()

    d = Path(default_dir)
    d.mkdir(parents=True, exist_ok=True)
    free, total = _free(d)
    out.append({
        "path": str(d),
        "label": f"local disk — {d}  ({_human(free)} free)",
        "removable": False,
        "free_bytes": free,
        "total_bytes": total,
    })
    seen.add(str(d.resolve()))

    user = os.environ.get("USER") or os.environ.get("USERNAME") or ""
    candidates: list[Path] = []
    for root in REMOVABLE_ROOTS:
        r = Path(root)
        if not r.is_dir():
            continue
        for child in sorted(r.iterdir()):
            if not child.is_dir():
                continue
            # /media/<user>/<stick> is one level deeper than /mnt/<stick>
            if child.name == user:
                candidates.extend(c for c in sorted(child.iterdir()) if c.is_dir())
            else:
                candidates.append(child)

    for c in candidates:
        rp = str(c.resolve())
        if rp in seen or not _writable(c):
            continue
        free, total = _free(c)
        if total == 0:
            continue
        seen.add(rp)
        out.append({
            "path": rp,
            "label": f"{c.name} — {rp}  ({_human(free)} free of {_human(total)})",
            "removable": True,
            "free_bytes": free,
            "total_bytes": total,
        })
    return out


def check_space(directory: str, needed_bytes: int) -> str | None:
    """Return a warning string if the target cannot hold the checkpoint."""
    free, _ = _free(Path(directory))
    if free and free < needed_bytes * 1.05:
        return (f"{_human(free)} free at {directory}, but the checkpoint needs "
                f"about {_human(needed_bytes)}. It would be written truncated.")
    return None


def checkpoint_bytes(grid: int, populations: int = 9, header: int = 64) -> int:
    return header + grid * grid * populations * 4
