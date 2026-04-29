from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
import streamlit as st
import xarray as xr
from shapely.geometry import Point
from shapely.prepared import prep

from config import CHUNKSIZE, DEFAULT_MASK_NC_PATH, FILE_GLOB


def _fix_year_time(ds: xr.Dataset, filepath: str) -> xr.Dataset:
    """Rebuild a clean daily time axis from the filename year.
    CDO remapping corrupts time coordinates in IMERG and ERA5 files."""
    m = re.search(r"(\d{4})", Path(filepath).stem)
    if m is None:
        return ds
    year = int(m.group(1))
    n = ds.sizes.get("time", 0)
    if n == 0:
        return ds
    return ds.assign_coords(
        time=pd.date_range(start=f"{year}-01-01", periods=n, freq="D")
    )


_NEEDS_TIME_FIX: set = {"IMERG", "ERA5"}


@st.cache_resource
def open_folder(folder: str) -> Tuple[xr.Dataset, List[str]]:
    data_dir = Path(folder).expanduser().resolve()
    files = sorted(data_dir.glob(FILE_GLOB))
    if not files:
        raise FileNotFoundError(f"No {FILE_GLOB} files in {data_dir}")

    if data_dir.name.upper() in _NEEDS_TIME_FIX:
        fixed = []
        for f in files:
            ds_f = xr.open_dataset(
                str(f), chunks={"time": CHUNKSIZE}, engine=None
            )
            fixed.append(_fix_year_time(ds_f, str(f)))
        ds = xr.concat(fixed, dim="time") if len(fixed) > 1 else fixed[0]
    else:
        ds = xr.open_mfdataset(
            [str(f) for f in files],
            combine="by_coords",
            parallel=True,
            chunks={"time": CHUNKSIZE},
            engine=None,
        )
    return ds, [str(f) for f in files]


def _find_coord(ds, candidates):
    for name in candidates:
        if name in ds.coords or name in ds.dims:
            return name
    return None


def infer_coords(ds: xr.Dataset) -> Tuple[str, str, str]:
    lat = _find_coord(ds, ["latitude", "lat", "y"])
    lon = _find_coord(ds, ["longitude", "lon", "x"])
    time = _find_coord(ds, ["time", "date"])
    if lat is None or lon is None or time is None:
        raise ValueError(
            f"Cannot infer coords. coords={list(ds.coords)}, dims={list(ds.dims)}"
        )
    return lat, lon, time


def normalize_lons(ds: xr.Dataset, lon_name: str) -> xr.Dataset:
    if np.nanmax(ds[lon_name].values) > 180:
        new_lon = ((ds[lon_name] + 180) % 360) - 180
        ds = ds.assign_coords({lon_name: new_lon}).sortby(lon_name)
    return ds


@st.cache_data(show_spinner=True)
def load_emi_csv(path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(path, index_col=0)
    df.index = df.index.astype(str).str.strip()
    if "LON" not in df.index:
        raise ValueError("Station CSV missing 'LON' row.")
    lat_row = (
        "DAILY/LAT"
        if "DAILY/LAT" in df.index
        else ("LAT" if "LAT" in df.index else None)
    )
    if lat_row is None:
        raise ValueError("Station CSV missing latitude row.")
    meta = pd.DataFrame(
        {
            "lat": df.loc[lat_row].astype(float),
            "lon": df.loc["LON"].astype(float),
        }
    )
    meta.index.name = "station"
    df_data = df.drop(index=["LON", lat_row])
    df_data.index = pd.to_datetime(
        df_data.index, format="%Y%m%d", errors="raise"
    )
    ts = (
        df_data.sort_index()
        .apply(pd.to_numeric, errors="coerce")
        .replace(-99, np.nan)
        .astype(float)
    )
    return meta, ts


@st.cache_resource
def load_mask() -> xr.DataArray:
    if not DEFAULT_MASK_NC_PATH.exists():
        raise FileNotFoundError(f"Mask not found: {DEFAULT_MASK_NC_PATH}")
    ds = xr.open_dataset(str(DEFAULT_MASK_NC_PATH))
    return ds[list(ds.data_vars)[0]]


def apply_mask(
    Z: np.ndarray, lat_vals, lon_vals, mask_da: xr.DataArray
) -> np.ndarray:
    mask_ds = mask_da.to_dataset(name="mask")
    if "time" in mask_ds.coords or "time" in mask_ds.dims:
        mlat, mlon, _ = infer_coords(mask_ds)
    else:
        mlat = _find_coord(mask_ds, ["latitude", "lat", "y"])
        mlon = _find_coord(mask_ds, ["longitude", "lon", "x"])
    if mlat is None or mlon is None:
        raise ValueError("Cannot infer lat/lon in mask .nc")
    mask_ds = normalize_lons(mask_ds, mlon)
    mlv = np.asarray(mask_ds[mlat].values)
    mlov = np.asarray(mask_ds[mlon].values)
    mg = np.asarray(mask_ds["mask"].values)
    if mg.ndim == 3:
        mg = mg[0]
    if mlv[0] > mlv[-1]:
        mlv, mg = mlv[::-1], mg[::-1, :]
    if mlov[0] > mlov[-1]:
        mlov, mg = mlov[::-1], mg[:, ::-1]

    def _nearest(src, tgt):
        src, tgt = np.asarray(src), np.asarray(tgt)
        if len(src) < 2:
            return np.zeros_like(tgt, dtype=int)
        idx = np.clip(np.searchsorted(src, tgt, side="left"), 1, len(src) - 1)
        return np.where((tgt - src[idx - 1]) <= (src[idx] - tgt), idx - 1, idx)

    m2 = mg[
        np.ix_(
            _nearest(mlv, np.asarray(lat_vals)),
            _nearest(mlov, np.asarray(lon_vals)),
        )
    ].astype(float)
    return np.where(np.isfinite(m2) & (m2 > 0.5), Z, np.nan)


def dataset_year_range(ds: xr.Dataset, time_name: str) -> Tuple[int, int]:
    y = ds[time_name].dt.year
    lo = int(y.min().compute()) if hasattr(y.min(), "compute") else int(y.min())
    hi = int(y.max().compute()) if hasattr(y.max(), "compute") else int(y.max())
    return lo, hi


def dataset_time_range(
    ds: xr.Dataset, time_name: str
) -> Tuple[pd.Timestamp, pd.Timestamp]:
    tmin = pd.to_datetime(ds[time_name].min().compute().values)
    tmax = pd.to_datetime(ds[time_name].max().compute().values)
    return tmin, tmax


@st.cache_resource
def load_boundary() -> gpd.GeoDataFrame:
    p = Path("data/ethiopia.geojson")
    if not p.exists():
        raise FileNotFoundError("data/ethiopia.geojson not found")
    return gpd.read_file(p).to_crs("EPSG:4326").dissolve()


def inside_mask(lat_vals, lon_vals, eth_geom) -> np.ndarray:
    g = prep(eth_geom)
    try:
        from shapely import contains_xy

        Lon, Lat = np.meshgrid(lon_vals, lat_vals)
        return contains_xy(eth_geom, Lon, Lat)
    except Exception:
        mask = np.zeros((len(lat_vals), len(lon_vals)), dtype=bool)
        for i, la in enumerate(lat_vals):
            for j, lo in enumerate(lon_vals):
                mask[i, j] = g.contains(Point(float(lo), float(la)))
        return mask
