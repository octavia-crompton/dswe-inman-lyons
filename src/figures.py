"""
Centralized figure-saving with an auto-updating CSV registry.

Usage (notebook or script)
--------------------------
    from src.figures import save_figure

    fig, ax = plt.subplots()
    ax.plot(...)
    save_figure(fig, "grdc_timeseries/1357100_timeseries.png",
                source=__file__,          # or the notebook path
                description="Mohembo daily-Q time series")

The registry lives at ``figures/registry.csv`` and is appended to (or
created) each time ``save_figure`` is called.  Columns:

    filename, source, description, timestamp, dpi, size_bytes
"""

from __future__ import annotations

import csv
import datetime as _dt
import inspect
import os
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import matplotlib.figure

# ---------------------------------------------------------------------------
# Resolve project root (works whether imported from notebooks/ or scripts/)
# ---------------------------------------------------------------------------
_THIS_DIR = Path(__file__).resolve().parent          # …/src
PROJECT_ROOT = _THIS_DIR.parent                       # …/dswe-inman-lyons
FIGURES_DIR = PROJECT_ROOT / "figures"
REGISTRY_CSV = FIGURES_DIR / "registry.csv"

_REGISTRY_COLUMNS = [
    "filename",
    "source",
    "description",
    "timestamp",
    "dpi",
    "size_bytes",
]


def _ensure_registry() -> None:
    """Create the figures dir and registry CSV header if they don't exist."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    if not REGISTRY_CSV.exists():
        with open(REGISTRY_CSV, "w", newline="") as fh:
            csv.writer(fh).writerow(_REGISTRY_COLUMNS)


def _guess_source() -> str:
    """Best-effort detection of the calling notebook / script path."""
    # Walk the stack looking for a frame whose file is NOT inside src/
    for info in inspect.stack():
        fpath = info.filename
        # Skip frames inside this package
        if "src/" in fpath and ("figures.py" in fpath or "__init__" in fpath):
            continue
        # IPython / Jupyter kernels expose the notebook via the global
        # __session__ or __vsc_ipynb_file__ (VS Code) variable.
        globs = info.frame.f_globals
        for key in ("__vsc_ipynb_file__", "__session__"):
            nb = globs.get(key)
            if nb:
                return str(nb)
        # Regular .py caller
        if fpath and fpath != "<stdin>" and not fpath.startswith("<"):
            return fpath
    return "<unknown>"


def _relative(path: str | Path) -> str:
    """Return a path relative to project root, if possible."""
    try:
        return str(Path(path).resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def save_figure(
    fig: "matplotlib.figure.Figure",
    name: str | Path,
    *,
    source: str | Path | None = None,
    description: str = "",
    dpi: int = 150,
    bbox_inches: str = "tight",
    **savefig_kw,
) -> Path:
    """Save *fig* to ``figures/<name>`` and log it in the registry.

    Parameters
    ----------
    fig : matplotlib Figure
    name : str or Path
        Relative path under ``figures/``.  Parent dirs are created
        automatically (e.g. ``"grdc_timeseries/1357100.png"``).
    source : str or Path, optional
        Notebook or script that generated this figure.  Auto-detected
        if omitted.
    description : str
        Free-text note stored in the registry.
    dpi, bbox_inches, **savefig_kw
        Forwarded to ``fig.savefig()``.

    Returns
    -------
    Path to the saved figure (absolute).
    """
    _ensure_registry()

    out_path = FIGURES_DIR / name
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(out_path, dpi=dpi, bbox_inches=bbox_inches, **savefig_kw)

    size = out_path.stat().st_size
    src = _relative(source) if source else _relative(_guess_source())

    row = {
        "filename": str(Path(name)),          # normalised relative path
        "source": src,
        "description": description,
        "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
        "dpi": dpi,
        "size_bytes": size,
    }

    with open(REGISTRY_CSV, "a", newline="") as fh:
        csv.DictWriter(fh, fieldnames=_REGISTRY_COLUMNS).writerow(row)

    print(f"Figure saved → {out_path}  ({size:,} bytes)")
    return out_path


def list_figures() -> "list[dict]":
    """Return the current registry as a list of dicts."""
    _ensure_registry()
    with open(REGISTRY_CSV, newline="") as fh:
        return list(csv.DictReader(fh))
