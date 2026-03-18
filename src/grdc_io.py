"""
grdc_io.py — shared utilities for reading GRDC *.Cmd.txt station files.

Usage
-----
    from src.grdc_io import (
        read_grdc_cmd_file,
        load_grdc_folder,
        pick_discharge_column,
        plot_station_map,
        plot_timeseries_per_station,
    )
"""
from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    _HAS_CARTOPY = True
except ImportError:
    _HAS_CARTOPY = False


META_RE = re.compile(r"^#\s*(?P<key>[^:]+):\s*(?P<val>.*)\s*$")


# ── Low-level helpers ─────────────────────────────────────────────────────────

def _to_float(x) -> float:
    """Robust float parsing — handles commas, whitespace, None."""
    if x is None:
        return np.nan
    s = str(x).strip().replace(",", ".")
    return pd.to_numeric(s, errors="coerce")


def _parse_filename(fp: Path) -> dict:
    """Extract station id + variable tag from a filename like 1357100_Q_Day.Cmd.txt."""
    m = re.match(r"^(?P<id>\d+)_(?P<tag>.+?)\.Cmd\.txt$", fp.name)
    if not m:
        return {"file_station_id": np.nan, "file_tag": fp.stem}
    return {
        "file_station_id": pd.to_numeric(m.group("id"), errors="coerce"),
        "file_tag": m.group("tag"),
    }


# ── Main I/O ──────────────────────────────────────────────────────────────────

def read_grdc_cmd_file(
    fp: str | Path,
    encoding: str = "latin-1",
) -> tuple[dict, pd.DataFrame]:
    """
    Read a GRDC ``*.Cmd.txt`` file.

    Returns
    -------
    meta : dict
        Metadata dict with both raw header keys and standardised fields
        (``grdc_no``, ``lat``, ``lon``, ``station_name``, ``unit``, …).
    ts : pd.DataFrame
        Time series with a ``DatetimeIndex`` and one numeric column per
        data field in the file.  Missing sentinel values (−999.*) are
        replaced with ``NaN``.
    """
    fp = Path(fp)

    # ── metadata ────────────────────────────────────────────────────────────
    meta_raw: dict = {"source_file": fp.name, "source_path": str(fp)}
    meta_raw.update(_parse_filename(fp))

    with open(fp, "r", encoding=encoding, errors="replace") as f:
        for line in f:
            if not line.startswith("#"):
                continue
            m = META_RE.match(line)
            if m:
                meta_raw[m.group("key").strip()] = m.group("val").strip()

    keymap = {
        "GRDC-No.":                      "grdc_no",
        "River":                          "river",
        "Station":                        "station_raw",
        "Country":                        "country",
        "Latitude (DD)":                  "lat",
        "Longitude (DD)":                 "lon",
        "Catchment area (km²)":          "catchment_area_km2",
        "Catchment area (km\ufffd)":      "catchment_area_km2",   # mojibake variant
        "Catchment area (km\xb2)":        "catchment_area_km2",   # cp1252 variant
        "Altitude (m ASL)":               "altitude_m",
        "file generation date":           "file_generation_date",
        "Last update":                    "last_update",
        "Data Set Content":               "dataset_content",
        "Unit of measure":                "unit",
        "Time series":                    "time_series",
        "Data lines":                     "data_lines",
    }

    meta = dict(meta_raw)
    for raw_k, std_k in keymap.items():
        if raw_k in meta_raw:
            meta[std_k] = meta_raw[raw_k]

    # Station name / code split, e.g. "MOHEMBO/MTAEMBO (67932112)"
    if isinstance(meta.get("station_raw"), str):
        m = re.match(r"^(.*?)\s*\((.*?)\)\s*$", meta["station_raw"])
        if m:
            meta["station_name"] = m.group(1).strip()
            meta["station_code"] = m.group(2).strip()
        else:
            meta["station_name"] = meta["station_raw"].strip()

    if "grdc_no" in meta:
        meta["grdc_no"] = pd.to_numeric(str(meta["grdc_no"]).strip(), errors="coerce")
    for k in ("lat", "lon", "catchment_area_km2", "altitude_m"):
        if k in meta:
            meta[k] = _to_float(meta[k])
    if "data_lines" in meta:
        meta["data_lines"] = pd.to_numeric(str(meta["data_lines"]).strip(), errors="coerce")

    # ── time-series data ─────────────────────────────────────────────────────
    df = pd.read_csv(
        fp,
        sep=";",
        comment="#",
        skip_blank_lines=True,
        dtype=str,
        encoding=encoding,
    )
    df.columns = [c.strip() for c in df.columns]
    for c in df.columns:
        df[c] = df[c].astype(str).str.strip()

    date_col = df.columns[0]
    time_col = df.columns[1] if len(df.columns) >= 2 else None

    date = pd.to_datetime(df[date_col], format="%Y-%m-%d", errors="coerce")

    if time_col is not None:
        has_times = (~df[time_col].isin(["--:--", "--", "", "nan", "None"])).any()
        if has_times:
            t = df[time_col].replace({"--:--": "00:00", "--": "00:00", "": "00:00"})
            dt = pd.to_datetime(df[date_col] + " " + t, format="%Y-%m-%d %H:%M", errors="coerce")
        else:
            dt = date
    else:
        dt = date

    drop_cols = [date_col] + ([time_col] if time_col is not None else [])
    data_cols = [c for c in df.columns if c not in drop_cols]

    out = df[data_cols].replace(
        {"-999.000": np.nan, "-999.00": np.nan, "-999.0": np.nan,
         "-999": np.nan, "": np.nan, "nan": np.nan, "None": np.nan}
    )
    for c in data_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    out.insert(0, "datetime", dt)
    out = out.dropna(subset=["datetime"]).set_index("datetime").sort_index()

    return meta, out


def pick_discharge_column(ts: pd.DataFrame) -> str:
    """
    Return the name of the most likely discharge column in a GRDC time series.

    Priority: single numeric column → column matching Q/discharge/flow/value →
    first numeric column.
    """
    numeric_cols = [c for c in ts.columns if pd.api.types.is_numeric_dtype(ts[c])]
    if not numeric_cols:
        raise ValueError("No numeric columns found in the time series table.")
    if len(numeric_cols) == 1:
        return numeric_cols[0]
    for pat in [r"^\s*Q\s*$", r"discharge", r"\bflow\b", r"value"]:
        for c in numeric_cols:
            if re.search(pat, c, flags=re.IGNORECASE):
                return c
    return numeric_cols[0]


def load_grdc_folder(
    folder: str | Path,
    pattern: str = "*.Cmd.txt",
) -> tuple[pd.DataFrame, dict]:
    """
    Load all GRDC files matching *pattern* from *folder*.

    Returns
    -------
    meta_df : pd.DataFrame
        One row per file with all metadata fields.
    ts_dict : dict
        ``{(grdc_no, file_tag, source_filename): pd.DataFrame}``
    """
    folder = Path(folder)
    meta_rows: list[dict] = []
    ts_dict: dict = {}

    for fp in sorted(folder.glob(pattern)):
        meta, ts = read_grdc_cmd_file(fp)
        meta_rows.append(meta)
        grdc_no = meta.get("grdc_no", meta.get("file_station_id"))
        tag = meta.get("file_tag", fp.stem)
        ts_dict[(grdc_no, tag, fp.name)] = ts

    meta_df = pd.DataFrame(meta_rows)
    if "grdc_no" in meta_df.columns:
        meta_df = meta_df.sort_values(
            ["grdc_no", "source_file"], na_position="last"
        ).reset_index(drop=True)

    return meta_df, ts_dict


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_station_map(meta_df: pd.DataFrame):
    """Plot station locations on a Cartopy PlateCarree map."""
    if not _HAS_CARTOPY:
        raise ImportError("cartopy is required for plot_station_map")

    stations = meta_df.dropna(subset=["lat", "lon"]).copy()
    pad = 5

    fig, ax = plt.subplots(
        figsize=(10, 7),
        subplot_kw={"projection": ccrs.PlateCarree()},
    )
    ax.add_feature(cfeature.LAND, linewidth=0.2)
    ax.add_feature(cfeature.OCEAN, linewidth=0.2)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.set_extent(
        [stations["lon"].min() - pad, stations["lon"].max() + pad,
         stations["lat"].min() - pad, stations["lat"].max() + pad],
        crs=ccrs.PlateCarree(),
    )
    ax.scatter(
        stations["lon"], stations["lat"],
        s=25, transform=ccrs.PlateCarree(), zorder=3,
    )
    ax.set_title("Station locations")
    return fig, ax


def plot_timeseries_per_station(
    meta_df: pd.DataFrame,
    ts_dict: dict,
    out_dir: str | Path | None = None,
) -> None:
    """
    One figure per GRDC station.  Multiple files for the same station are
    overlaid on the same axes.  Pass *out_dir* to save PNGs.
    """
    by_station: dict = {}
    for (grdc_no, tag, fname), ts in ts_dict.items():
        by_station.setdefault(grdc_no, []).append((tag, fname, ts))

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

    name_lookup: dict = {}
    if "grdc_no" in meta_df.columns and "station_name" in meta_df.columns:
        for _, r in meta_df.dropna(subset=["grdc_no"]).iterrows():
            if pd.notna(r.get("station_name")):
                name_lookup[r["grdc_no"]] = r["station_name"]

    for grdc_no, series_list in by_station.items():
        fig, ax = plt.subplots(figsize=(14, 4))

        for tag, _fname, ts in sorted(series_list, key=lambda x: str(x[0])):
            numeric_cols = [c for c in ts.columns if pd.api.types.is_numeric_dtype(ts[c])]
            if not numeric_cols:
                continue
            ts[numeric_cols].plot(
                ax=ax, linewidth=0.8,
                label=[f"{tag}:{c}" for c in numeric_cols],
            )

        station_name = name_lookup.get(grdc_no, "")
        title = f"GRDC {int(grdc_no) if pd.notna(grdc_no) else grdc_no}"
        if station_name:
            title += f" — {station_name}"
        ax.set_title(title)
        ax.set_xlabel("Date")
        ax.set_ylabel("Value")
        ax.grid(True, linewidth=0.2)
        ax.legend(loc="best", fontsize=8, ncols=2)
        fig.tight_layout()

        if out_dir is not None:
            safe_id = "unknown" if pd.isna(grdc_no) else str(int(grdc_no))
            fig.savefig(out_dir / f"{safe_id}_timeseries.png", dpi=150)
            plt.close(fig)
