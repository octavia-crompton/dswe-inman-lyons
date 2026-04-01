"""hysteresis.py – loop direction, circularity, and phase-loop metrics."""
from __future__ import annotations

import numpy as np


def loop_direction_and_circularity(
    xv,
    yv,
    normalize: bool = True,
) -> tuple[str, float]:
    """Compute direction (CW/CCW) and circularity of a closed (x, y) phase loop.

    Parameters
    ----------
    xv, yv : array-like
        Ordered x and y coordinates of the loop.
    normalize : bool
        If True, rescale x and y to [0, 1] before computing circularity
        so the metric is independent of units and aspect ratio.

    Returns
    -------
    direction : str
        ``"CCW"``, ``"CW"``, or ``"NA"`` (fewer than 3 finite points or zero area).
    circularity : float
        Isoperimetric ratio 4πA/P² ∈ [0, 1]; equals 1 for a perfect circle.
        ``nan`` when undefined.
    """
    x = np.asarray(xv, dtype=float)
    y = np.asarray(yv, dtype=float)

    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 3:
        return "NA", np.nan

    if normalize:
        xr = np.ptp(x)
        yr = np.ptp(y)
        if xr > 0:
            x = (x - x.min()) / xr
        if yr > 0:
            y = (y - y.min()) / yr

    # Close the loop
    x2 = np.r_[x, x[0]]
    y2 = np.r_[y, y[0]]

    # Signed area (shoelace formula)
    area = 0.5 * np.sum(x2[:-1] * y2[1:] - x2[1:] * y2[:-1])

    # Perimeter
    perim = np.sum(np.hypot(np.diff(x2), np.diff(y2)))

    direction = "CCW" if area > 0 else ("CW" if area < 0 else "NA")
    circ = (
        np.nan
        if perim <= 0
        else float(np.clip(4 * np.pi * abs(area) / perim**2, 0, 1))
    )
    return direction, circ
