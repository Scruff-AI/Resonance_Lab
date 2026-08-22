"""Turn a raw float32 lattice plane into a PNG for the browser.

Every rendered frame carries the scale it was drawn at (vmin/vmax, the
reduction used, the cycle). A picture of a field without its scale is not a
measurement, and the UI shows the numbers next to the image for that reason.
"""

from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np
from PIL import Image

try:  # exact colormaps when matplotlib is present
    from matplotlib import colormaps as _mpl_colormaps
    _HAVE_MPL = True
except Exception:  # pragma: no cover - fallback path
    _HAVE_MPL = False

# Fallback ramps if matplotlib is not installed. Named -fallback in the UI so a
# screenshot is never mistaken for a standard colormap.
_FALLBACK_ANCHORS = {
    "viridis": [(0.267, 0.005, 0.329), (0.229, 0.322, 0.545),
                (0.128, 0.567, 0.551), (0.369, 0.789, 0.383),
                (0.993, 0.906, 0.144)],
    "inferno": [(0.001, 0.000, 0.014), (0.341, 0.063, 0.429),
                (0.729, 0.212, 0.333), (0.969, 0.549, 0.098),
                (0.988, 0.998, 0.645)],
    "gray":    [(0.0, 0.0, 0.0), (1.0, 1.0, 1.0)],
    "coolwarm": [(0.230, 0.299, 0.754), (0.865, 0.865, 0.865),
                 (0.706, 0.016, 0.150)],
}

DIVERGING = {"coolwarm", "bwr", "RdBu_r", "seismic"}


@dataclass
class RenderResult:
    png: bytes
    cycle: int
    vmin: float
    vmax: float
    colormap: str
    reduction: str
    fixed_scale: bool
    width: int
    height: int
    source_width: int
    source_height: int

    def meta(self) -> dict:
        return {
            "cycle": self.cycle,
            "vmin": self.vmin,
            "vmax": self.vmax,
            "colormap": self.colormap,
            "reduction": self.reduction,
            "fixed_scale": self.fixed_scale,
            "size": [self.width, self.height],
            "source_size": [self.source_width, self.source_height],
        }


def _lut(name: str, n: int = 256) -> np.ndarray:
    if _HAVE_MPL:
        try:
            cm = _mpl_colormaps[name]
            xs = np.linspace(0.0, 1.0, n)
            return (np.asarray(cm(xs))[:, :3] * 255.0).astype(np.uint8)
        except KeyError:
            pass
    anchors = _FALLBACK_ANCHORS.get(name) or _FALLBACK_ANCHORS["viridis"]
    a = np.asarray(anchors, dtype=np.float64)
    src = np.linspace(0.0, 1.0, len(a))
    xs = np.linspace(0.0, 1.0, n)
    rgb = np.stack([np.interp(xs, src, a[:, c]) for c in range(3)], axis=1)
    return (rgb * 255.0).astype(np.uint8)


def _downsample(arr: np.ndarray, target: int) -> tuple[np.ndarray, str]:
    h, w = arr.shape
    if max(h, w) <= target:
        return arr, "none (1:1)"
    factor = max(1, int(round(max(h, w) / target)))
    if h % factor or w % factor:
        # Not an exact tiling: fall back to striding and say so.
        return arr[::factor, ::factor], f"stride {factor}x (not block-averaged)"
    view = arr.reshape(h // factor, factor, w // factor, factor)
    return view.mean(axis=(1, 3)), f"block mean {factor}x{factor}"


def render(
    arr: np.ndarray,
    cycle: int,
    colormap: str = "viridis",
    target_size: int = 512,
    vmin: float | None = None,
    vmax: float | None = None,
    center_on_mean: bool = False,
    percentile: float = 0.5,
) -> RenderResult:
    """Render one plane.

    center_on_mean subtracts the field mean and forces a symmetric scale — the
    right view for a field like density that sits near a constant with small
    structure on top, where an absolute scale shows nothing but flat colour.
    """
    scale_was_given = vmin is not None and vmax is not None
    src_h, src_w = arr.shape
    small, reduction = _downsample(np.asarray(arr, dtype=np.float32), target_size)

    finite = np.isfinite(small)
    if not finite.all():
        # NaN in the field is a real event in this system, not a render bug.
        small = np.where(finite, small, np.nan)

    if center_on_mean:
        mean = float(np.nanmean(small))
        small = small - mean

    if vmin is None or vmax is None:
        lo = float(np.nanpercentile(small, percentile))
        hi = float(np.nanpercentile(small, 100.0 - percentile))
        if center_on_mean or colormap in DIVERGING:
            m = max(abs(lo), abs(hi)) or 1e-12
            lo, hi = -m, m
        if hi <= lo:
            hi = lo + 1e-12
        vmin = lo if vmin is None else vmin
        vmax = hi if vmax is None else vmax

    norm = (small - vmin) / (vmax - vmin)
    norm = np.clip(np.nan_to_num(norm, nan=0.0), 0.0, 1.0)
    idx = (norm * 255.0).astype(np.uint8)

    rgb = _lut(colormap)[idx]
    # Paint non-finite cells magenta so a NaN cascade is unmistakable.
    if not finite.all():
        bad = ~np.isfinite(small)
        rgb[bad] = (255, 0, 255)

    buf = io.BytesIO()
    Image.fromarray(rgb, mode="RGB").save(buf, format="PNG", optimize=False, compress_level=1)

    h, w = idx.shape
    return RenderResult(
        png=buf.getvalue(),
        cycle=cycle,
        vmin=float(vmin),
        vmax=float(vmax),
        colormap=colormap if _HAVE_MPL else f"{colormap}-fallback",
        fixed_scale=scale_was_given,
        reduction=reduction,
        width=w,
        height=h,
        source_width=src_w,
        source_height=src_h,
    )
