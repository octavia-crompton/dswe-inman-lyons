"""mapping.py – reusable Okavango Delta basemap for folium notebooks."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import folium
import geopandas as gpd


# Default delta shapefile path (relative to repo root)
_DELTA_SHP = Path(__file__).resolve().parent.parent / "data" / "regions" / "Delta_UCB_WGS84" / "Delta_UCB_WGS84.shp"


def okavango_basemap(
    *,
    delta_shp: str | Path | None = None,
    zoom_start: int = 8,
    tiles: str | None = None,
    extra_tile_layers: Sequence[str] = ("OpenStreetMap", "CartoDB positron"),
    delta_style: dict | None = None,
    show_layer_control: bool = True,
) -> folium.Map:
    """Return a folium Map centred on the Okavango Delta with standard layers.

    Parameters
    ----------
    delta_shp : path, optional
        Path to the delta polygon shapefile.  Defaults to
        ``data/regions/Delta_UCB_WGS84/Delta_UCB_WGS84.shp``.
    zoom_start : int
        Initial zoom level (default 8).
    tiles : str or None
        Base tile layer passed to ``folium.Map``.  *None* means no default
        basemap (useful when adding multiple tile layers).
    extra_tile_layers : sequence of str
        Additional named tile layers to add (e.g. ``"OpenStreetMap"``).
    delta_style : dict, optional
        Style overrides for the delta GeoJSON overlay.
    show_layer_control : bool
        If *True*, add a ``folium.LayerControl``.
    """
    shp = Path(delta_shp) if delta_shp else _DELTA_SHP
    gdf = gpd.read_file(shp).to_crs(epsg=4326)
    centroid = gdf.unary_union.centroid

    m = folium.Map(location=[centroid.y, centroid.x], zoom_start=zoom_start, tiles=tiles)

    for tl in extra_tile_layers:
        folium.TileLayer(tl, name=tl).add_to(m)

    style = {"color": "yellow", "weight": 3, "fillColor": "yellow", "fillOpacity": 0.1}
    if delta_style:
        style.update(delta_style)

    folium.GeoJson(
        gdf.__geo_interface__,
        name="Delta polygon",
        style_function=lambda feat, _s=style: _s,
    ).add_to(m)

    if show_layer_control:
        folium.LayerControl(collapsed=False).add_to(m)

    return m
