"""
Centralized figure-saving with an auto-updating text registry.

Usage (notebook or script)
--------------------------
    from src.figures import save_figure

    fig, ax = plt.subplots()
    ax.plot(...)
    save_figure(fig, "grdc_timeseries/1357100_timeseries.png",
                source=__file__,          # or the notebook path
                description="Mohembo daily-Q time series")

The registry lives at ``figures/registry.txt`` and is appended to (or
created) each time ``save_figure`` is called.  Each entry is one line:

    timestamp | filename | dpi dpi | size_bytes bytes | source | description
"""

from __future__ import annotations

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
REGISTRY_FILE = FIGURES_DIR / "registry.txt"

_REGISTRY_HEADER = "# timestamp | filename | dpi | size_bytes | source | description"


def _ensure_registry() -> None:
    """Create the figures dir and registry text file if they don't exist."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    if not REGISTRY_FILE.exists():
        REGISTRY_FILE.write_text(_REGISTRY_HEADER + "\n")


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

    ts = _dt.datetime.now().isoformat(timespec="seconds")
    line = " | ".join([
        ts,
        str(Path(name)),
        f"{dpi} dpi",
        f"{size:,} bytes",
        src,
        description,
    ])

    with open(REGISTRY_FILE, "a") as fh:
        fh.write(line + "\n")

    print(f"Figure saved → {out_path}  ({size:,} bytes)")
    return out_path


def list_figures() -> "list[dict]":
    """Return the current registry as a list of dicts."""
    _ensure_registry()
    keys = ["timestamp", "filename", "dpi", "size_bytes", "source", "description"]
    entries = []
    for line in REGISTRY_FILE.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.split(" | ", maxsplit=5)]
        entries.append(dict(zip(keys, parts)))
    return entries
