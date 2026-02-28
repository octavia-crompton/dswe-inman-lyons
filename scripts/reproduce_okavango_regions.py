"""
reproduce_okavango_regions.py
─────────────────────────────
Reproduces data/regions/okavango_regions.gpkg from the publicly available
HydroBASINS Africa Level-4 dataset.

Primary source: Google Earth Engine asset
    ``WWF/HydroSHEDS/v1/Basins/hybas_4``
    (requires a GEE-authenticated Python environment, e.g. ``ee-map`` conda env)

Fallback source: local HydroBASINS shapefile or zip
    Supply via ``--hydrobasins-path /path/to/hybas_af_lev04_v1c.shp`` (or .zip)

Steps
-----
1. Fetch the 5 target HydroBASINS polygons (by HYBAS_ID) from GEE or a local file.
2. Assign region names.
3. Split HYBAS_ID 1041515680 at ``--split-lon`` (default 21.0 E) into
   "upper_okavango" (west half) and "east_lower_okavango" (east half),
   matching the original hand-edited file.
4. Write a GeoPackage to ``--out``.
5. Optionally save a preview map PNG (``--plot``).

Usage
-----
    # GEE (default):
    python scripts/reproduce_okavango_regions.py

    # Local shapefile fallback:
    python scripts/reproduce_okavango_regions.py \
        --hydrobasins-path /path/to/hybas_af_lev04_v1c.shp

    # Custom output + map preview:
    python scripts/reproduce_okavango_regions.py \
        --out /tmp/okavango_regions_repro.gpkg --plot
"""

import argparse
import zipfile
from pathlib import Path

import geopandas as gpd
import shapely.geometry as geom

# ── Configuration ─────────────────────────────────────────────────────────────

# HydroBASINS IDs present in the original file (5 unique source basins)
TARGET_HYBAS_IDS = [1041477980, 1041515680, 1041479540, 1041515780, 1041473950]

# Name mapping for the 4 unambiguous basins
HYBAS_NAMES = {
    1041477980: "lowland_cuito",
    1041479540: "lowland_cubango",
    1041515780: "west_lower_okavango",
    1041473950: "seronga",
}

# GEE asset identifier
GEE_ASSET = "WWF/HydroSHEDS/v1/Basins/hybas_4"

DEFAULT_OUT = Path("data/regions/okavango_regions.gpkg")


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_via_gee() -> gpd.GeoDataFrame:
    """Fetch the 5 target HydroBASINS polygons from GEE."""
    import ee
    try:
        ee.Initialize()
    except Exception:
        ee.Authenticate()
        ee.Initialize()

    fc = (
        ee.FeatureCollection(GEE_ASSET)
        .filter(ee.Filter.inList("HYBAS_ID", TARGET_HYBAS_IDS))
    )
    geojson = fc.getInfo()
    gdf = gpd.GeoDataFrame.from_features(geojson["features"], crs="EPSG:4326")
    gdf["HYBAS_ID"] = gdf["HYBAS_ID"].astype(int)
    print(f"Loaded {len(gdf)} polygons from GEE asset '{GEE_ASSET}'")
    return gdf


def load_via_file(path: str) -> gpd.GeoDataFrame:
    """Load HydroBASINS from a local .shp or .zip and filter to target IDs."""
    p = Path(path)
    if p.suffix == ".zip":
        with zipfile.ZipFile(p) as z:
            shp_names = [n for n in z.namelist() if n.endswith(".shp")]
            assert shp_names, "No .shp found inside zip"
            gdf = gpd.read_file(f"zip://{p}/{shp_names[0]}")
    else:
        gdf = gpd.read_file(p)

    gdf = gdf[gdf["HYBAS_ID"].astype(int).isin(TARGET_HYBAS_IDS)].copy()
    gdf["HYBAS_ID"] = gdf["HYBAS_ID"].astype(int)
    print(f"Loaded {len(gdf)} polygons from {p}")
    return gdf


def split_polygon(poly, split_lon: float):
    """Return (west_part, east_part) of *poly* split at *split_lon*."""
    minx, miny, maxx, maxy = poly.bounds
    west_box = geom.box(minx - 1, miny - 1, split_lon, maxy + 1)
    east_box = geom.box(split_lon, miny - 1, maxx + 1, maxy + 1)
    return poly.intersection(west_box), poly.intersection(east_box)


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args):
    # 1. Acquire base polygons
    if args.hydrobasins_path:
        hb = load_via_file(args.hydrobasins_path)
    else:
        hb = load_via_gee()

    if hb.crs is None:
        hb = hb.set_crs("EPSG:4326")
    src_crs = hb.crs

    # 2. Assign names to unambiguous basins
    hb["name"] = hb["HYBAS_ID"].map(HYBAS_NAMES)

    # 3. Split HYBAS_ID 1041515680
    split_rows = hb[hb["HYBAS_ID"] == 1041515680]
    other_rows = hb[hb["HYBAS_ID"] != 1041515680].copy()

    if len(split_rows) == 0:
        print("WARNING: HYBAS_ID 1041515680 not found - skipping split step.")
        result = other_rows
    else:
        base = split_rows.iloc[0]
        west_geom, east_geom = split_polygon(base.geometry, args.split_lon)

        row_upper = base.copy()
        row_upper["geometry"] = west_geom
        row_upper["name"] = "upper_okavango"

        row_east = base.copy()
        row_east["geometry"] = east_geom
        row_east["name"] = "east_lower_okavango"

        result = gpd.GeoDataFrame(
            [row_upper, row_east, *[other_rows.iloc[i] for i in range(len(other_rows))]],
            crs=src_crs,
        )

    result = result.reset_index(drop=True)

    print("\nFinal regions:")
    for _, r in result.iterrows():
        print(f"  {r['name']:30s}  HYBAS_ID={r['HYBAS_ID']}")

    # 4. Save
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    result.to_file(out, driver="GPKG")
    print(f"\nSaved {len(result)} regions -> {out}")

    # 5. Optional plot
    if args.plot:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        fig, ax = plt.subplots(figsize=(8, 7))
        cmap = plt.get_cmap("tab10", len(result))
        for i, (_, row) in enumerate(result.iterrows()):
            gpd.GeoSeries([row.geometry], crs=src_crs).plot(
                ax=ax, color=cmap(i), edgecolor="black", linewidth=0.8, alpha=0.7
            )
        patches = [
            mpatches.Patch(color=cmap(i), label=row["name"])
            for i, (_, row) in enumerate(result.iterrows())
        ]
        ax.legend(handles=patches, loc="lower right", fontsize=8)
        ax.set_title("Reproduced okavango_regions.gpkg")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        # Use centralized figure registry
        from src.figures import save_figure
        save_figure(
            fig,
            "regions/okavango_regions.png",
            source=__file__,
            description="Preview map of reproduced okavango_regions.gpkg",
        )
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--hydrobasins-path", default=None, metavar="PATH",
        help="Path to hybas_af_lev04_v1c.shp or a .zip containing it. "
             "If omitted the script fetches polygons from GEE.",
    )
    parser.add_argument(
        "--out", default=str(DEFAULT_OUT), metavar="PATH",
        help=f"Output GeoPackage path (default: {DEFAULT_OUT})",
    )
    parser.add_argument(
        "--split-lon", type=float, default=21.0, metavar="LON",
        help="Longitude (E) at which to split HYBAS_ID 1041515680 into "
             "upper_okavango (west) and east_lower_okavango (east). Default: 21.0",
    )
    parser.add_argument(
        "--plot", action="store_true",
        help="Save a preview map PNG alongside the output GeoPackage.",
    )
    main(parser.parse_args())
