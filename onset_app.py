from __future__ import annotations

from pathlib import Path
from typing import Tuple, Optional, List, Literal, Dict

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import streamlit as st
import geopandas as gpd
from shapely.geometry import Point
from shapely.prepared import prep
from matplotlib.colors import ListedColormap, BoundaryNorm, TwoSlopeNorm

st.set_page_config(layout="wide")

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
LINEWIDTH_ETH = 2.0
DATASET_FOLDERS: Dict[str, str] = {
    "CHIRPS": "CHIRPS",
    "ENACTS": "ENACTS",
    "IMERG": "IMERG",
    "ERA5": "ERA5",
}
# Per-dataset rainfall variable names inside the NetCDF files
RAIN_VAR_BY_KEY: Dict[str, str] = {
    "CHIRPS": "precip",
    "ENACTS": "precip",
    "IMERG": "precip",  # update here if IMERG uses a different var name
    "ERA5": "precip",
}
RAIN_VAR = "precip"  # fallback default
FILE_GLOB = "*.nc"
CHUNKSIZE = 365
APP_DIR = Path(__file__).resolve().parent
STATION_KEY = "EMI Stations"
STATION_CSV_PATH = APP_DIR / "RF_Station_Grid_Format.csv"
DEFAULT_MASK_NC_PATH = APP_DIR / "chirps_jjas_seasonal_mask_ethiopia_0p25.nc"

DATASET_COLORS = {
    "CHIRPS": {"rain": "steelblue", "wet": "#1a1aff", "onset": "#000080"},
    "ENACTS": {"rain": "darkorange", "wet": "#cc0000", "onset": "#660000"},
    "IMERG": {"rain": "#9467bd", "wet": "#6a0dad", "onset": "#3b0066"},
    "ERA5": {"rain": "#17becf", "wet": "#0a6e7a", "onset": "#003d44"},
    STATION_KEY: {"rain": "#2ca02c", "wet": "#7fff00", "onset": "#006400"},
}

# Datasets with partial-year coverage: map key → human-readable note shown in UI
SEASONAL_COVERAGE_NOTE: Dict[str, str] = {
    "ENACTS": "May–Oct only",
}

# ─────────────────────────────────────────────
# Sidebar: onset parameters
# ─────────────────────────────────────────────
st.sidebar.header("Onset / Wet spell parameters")
wet_day_mm = st.sidebar.number_input(
    "Wet day threshold (mm/day)", min_value=0.0, value=1.0, step=0.1
)
accum_days = st.sidebar.number_input(
    "Accumulation window (days)", min_value=1, value=3, step=1
)
accum_mm = st.sidebar.number_input(
    "Accumulation threshold (mm)", min_value=0.0, value=20.0, step=1.0
)
dry_spell_days = st.sidebar.number_input(
    "Dry spell length (days)", min_value=1, value=7, step=1
)
lookahead_days = st.sidebar.number_input(
    "Lookahead window (days)", min_value=1, value=21, step=1
)

st.sidebar.subheader("Search window")
start_month = st.sidebar.selectbox("Search start month", list(range(1, 13)), index=0)
start_day = st.sidebar.number_input(
    "Search start day", min_value=1, max_value=31, value=1, step=1
)
end_month = st.sidebar.selectbox("Search end month", list(range(1, 13)), index=11)
end_day = st.sidebar.number_input(
    "Search end day", min_value=1, max_value=31, value=31, step=1
)


# ─────────────────────────────────────────────
# Data loading helpers
# ─────────────────────────────────────────────


def _fix_year_time(ds: "xr.Dataset", filepath: str) -> "xr.Dataset":
    """Fix corrupted time coordinates on datasets processed by CDO remapping
    (affects IMERG and ERA5). Recovers correct dates from the filename year."""
    import re as _re

    fname = Path(filepath).stem
    m = _re.search(r"(\d{4})", fname)
    if m is None:
        return ds
    year = int(m.group(1))
    n_times = ds.sizes.get("time", 0)
    if n_times == 0:
        return ds
    correct_times = pd.date_range(start=f"{year}-01-01", periods=n_times, freq="D")
    ds = ds.assign_coords(time=correct_times)
    return ds


# Datasets that need per-file time correction (CDO remapping corrupts their time axis)
_NEEDS_TIME_FIX: set = {"IMERG", "ERA5"}


@st.cache_resource
def open_dataset_folder(folder: str) -> Tuple[xr.Dataset, List[str]]:
    data_dir = Path(folder).expanduser().resolve()
    files = sorted(data_dir.glob(FILE_GLOB))
    if not files:
        raise FileNotFoundError(f"No {FILE_GLOB} files in {data_dir}")

    folder_key = data_dir.name.upper()

    if folder_key in _NEEDS_TIME_FIX:
        # Open each file individually, fix its time axis, then combine
        fixed = []
        for f in files:
            ds_f = xr.open_dataset(str(f), chunks={"time": CHUNKSIZE}, engine=None)
            ds_f = _fix_year_time(ds_f, str(f))
            fixed.append(ds_f)
        ds_ = xr.concat(fixed, dim="time") if len(fixed) > 1 else fixed[0]
    else:
        ds_ = xr.open_mfdataset(
            [str(f) for f in files],
            combine="by_coords",
            parallel=True,
            chunks={"time": CHUNKSIZE},
            engine=None,
        )
    return ds_, [str(f) for f in files]


def _find_coord_name(ds, candidates):
    for name in candidates:
        if name in ds.coords or name in ds.dims:
            return name
    return None


def guess_lat_lon_time_names(ds):
    lat = _find_coord_name(ds, ["latitude", "lat", "y"])
    lon = _find_coord_name(ds, ["longitude", "lon", "x"])
    time = _find_coord_name(ds, ["time", "date"])
    if lat is None or lon is None or time is None:
        raise ValueError(
            f"Cannot infer coords. coords={list(ds.coords)}, dims={list(ds.dims)}"
        )
    return lat, lon, time


def maybe_normalize_longitudes(ds, lon_name):
    lon = ds[lon_name]
    if np.nanmax(lon.values) > 180:
        new_lon = ((lon + 180) % 360) - 180
        ds = ds.assign_coords({lon_name: new_lon}).sortby(lon_name)
    return ds


@st.cache_data(show_spinner=True)
def load_emi_station_csv(path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
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
    lon = df.loc["LON"].astype(float)
    lat = df.loc[lat_row].astype(float)
    meta = pd.DataFrame({"lat": lat, "lon": lon})
    meta.index.name = "station"
    df_data = df.drop(index=["LON", lat_row])
    df_data.index = pd.to_datetime(df_data.index, format="%Y%m%d", errors="raise")
    df_data = df_data.sort_index()
    ts = (
        df_data.apply(pd.to_numeric, errors="coerce").replace(-99, np.nan).astype(float)
    )
    return meta, ts


@st.cache_resource
def load_default_mask_da() -> xr.DataArray:
    if not DEFAULT_MASK_NC_PATH.exists():
        raise FileNotFoundError(f"Mask not found: {DEFAULT_MASK_NC_PATH}")
    ds = xr.open_dataset(str(DEFAULT_MASK_NC_PATH))
    return ds[list(ds.data_vars)[0]]


def apply_nc_mask_to_Z(Z, lat_vals, lon_vals, mask_da):
    mask_ds = mask_da.to_dataset(name="mask")
    if "time" in mask_ds.coords or "time" in mask_ds.dims:
        mlat, mlon, _ = guess_lat_lon_time_names(mask_ds)
    else:
        mlat = _find_coord_name(mask_ds, ["latitude", "lat", "y"])
        mlon = _find_coord_name(mask_ds, ["longitude", "lon", "x"])
    if mlat is None or mlon is None:
        raise ValueError("Cannot infer lat/lon in mask .nc")
    mask_ds = maybe_normalize_longitudes(mask_ds, mlon)
    mlv = np.asarray(mask_ds[mlat].values)
    mlov = np.asarray(mask_ds[mlon].values)
    mg = np.asarray(mask_ds["mask"].values)
    if mg.ndim == 3:
        mg = mg[0]
    if mlv[0] > mlv[-1]:
        mlv = mlv[::-1]
        mg = mg[::-1, :]
    if mlov[0] > mlov[-1]:
        mlov = mlov[::-1]
        mg = mg[:, ::-1]

    def nearest_idx(src, tgt):
        src = np.asarray(src)
        tgt = np.asarray(tgt)
        if len(src) < 2:
            return np.zeros_like(tgt, dtype=int)
        idx = np.clip(np.searchsorted(src, tgt, side="left"), 1, len(src) - 1)
        left = src[idx - 1]
        right = src[idx]
        return np.where((tgt - left) <= (right - tgt), idx - 1, idx)

    li = nearest_idx(mlv, np.asarray(lat_vals))
    lj = nearest_idx(mlov, np.asarray(lon_vals))
    m2 = mg[np.ix_(li, lj)].astype(float)
    return np.where(np.isfinite(m2) & (m2 > 0.5), Z, np.nan)


def dataset_year_range(ds, time_name):
    y = ds[time_name].dt.year
    lo = int(y.min().compute()) if hasattr(y.min(), "compute") else int(y.min())
    hi = int(y.max().compute()) if hasattr(y.max(), "compute") else int(y.max())
    return lo, hi


def dataset_time_range(ds, time_name):
    tmin = pd.to_datetime(ds[time_name].min().compute().values)
    tmax = pd.to_datetime(ds[time_name].max().compute().values)
    return tmin, tmax


# ─────────────────────────────────────────────
# Onset / wet spell detection
# ─────────────────────────────────────────────
def detect_onset(
    dates,
    rain,
    wet_day_mm,
    accum_days,
    accum_mm,
    dry_spell_days,
    lookahead_days,
    start_month,
    start_day,
    end_month,
    end_day,
):
    r = np.asarray(rain, dtype=float)
    d = pd.to_datetime(dates).reset_index(drop=True)
    wet = r >= wet_day_mm
    year = int(d.iloc[0].year)
    t0 = pd.Timestamp(year=year, month=start_month, day=start_day)
    t1 = pd.Timestamp(year=year, month=end_month, day=end_day)
    si = int(np.searchsorted(d.values, np.datetime64(t0)))
    ei = int(np.searchsorted(d.values, np.datetime64(t1), side="right"))
    n = len(r)
    last = n - accum_days - lookahead_days
    if last <= si:
        return None, None
    stop = min(last, ei)
    if stop <= si:
        return None, None
    for t in range(si, stop):
        win = r[t : t + accum_days]
        if np.nansum(win) >= accum_mm and np.all(win >= wet_day_mm):
            fut = wet[t + accum_days : t + accum_days + lookahead_days]
            dry_run, ok = 0, True
            for w in fut:
                if not w:
                    dry_run += 1
                    if dry_run >= dry_spell_days:
                        ok = False
                        break
                else:
                    dry_run = 0
            if ok:
                return d.iloc[t], t
    return None, None


def detect_first_wet_spell(
    dates,
    rain,
    wet_day_mm,
    accum_days,
    accum_mm,
    start_month,
    start_day,
    end_month,
    end_day,
):
    r = np.asarray(rain, dtype=float)
    d = pd.to_datetime(dates).reset_index(drop=True)
    year = int(d.iloc[0].year)
    t0 = pd.Timestamp(year=year, month=start_month, day=start_day)
    t1 = pd.Timestamp(year=year, month=end_month, day=end_day)
    si = int(np.searchsorted(d.values, np.datetime64(t0)))
    ei = int(np.searchsorted(d.values, np.datetime64(t1), side="right"))
    stop = min(len(r) - accum_days + 1, ei)
    for t in range(si, max(si, stop)):
        win = r[t : t + accum_days]
        if np.nansum(win) >= accum_mm and not np.any(win < wet_day_mm):
            return d.iloc[t], t
    return None, None


def wet_spell_start_from_idx(dates, rain, idx, wet_day_mm):
    if idx is None:
        return None
    r = np.asarray(rain, dtype=float)
    d = pd.to_datetime(dates).reset_index(drop=True)
    if idx < 0 or idx >= len(r):
        return None
    if r[idx] < wet_day_mm:
        return d.iloc[idx]
    j = idx
    while j - 1 >= 0 and r[j - 1] >= wet_day_mm:
        j -= 1
    return d.iloc[j]


# ─────────────────────────────────────────────
# Ethiopia boundary
# ─────────────────────────────────────────────
@st.cache_resource
def load_ethiopia_boundary():
    p = Path("data/ethiopia.geojson")
    if not p.exists():
        raise FileNotFoundError("data/ethiopia.geojson not found")
    return gpd.read_file(p).to_crs("EPSG:4326").dissolve()


def make_inside_mask(lat_vals, lon_vals, eth_geom):
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


# ─────────────────────────────────────────────
# JJAS DOY colormap
# ─────────────────────────────────────────────
def build_jjas_doy_cmap():
    month_doys = {
        "May": (121, 151),
        "Jun": (152, 181),
        "Jul": (182, 212),
        "Aug": (213, 243),
        "Sep": (244, 265),
    }
    month_cmaps = {
        "May": plt.cm.YlOrBr,
        "Jun": plt.cm.Greens,
        "Jul": plt.cm.GnBu,
        "Aug": plt.cm.BuPu,
        "Sep": plt.cm.RdPu,
    }
    colors, bounds = [], []
    N = 8
    for m in ["May", "Jun", "Jul", "Aug", "Sep"]:
        d0, d1 = month_doys[m]
        colors.extend(month_cmaps[m](np.linspace(0.35, 0.95, N)))
        bounds.extend(np.linspace(d0, d1, N, endpoint=False))
    bounds.append(month_doys["Sep"][1])
    cmap = ListedColormap(colors, name="JJAS_piecewise")
    norm = BoundaryNorm(bounds, cmap.N)
    tpos = [
        121,
        128,
        136,
        143,
        152,
        159,
        166,
        173,
        182,
        189,
        197,
        204,
        213,
        220,
        228,
        235,
        244,
        251,
        258,
        265,
    ]
    tlbl = [
        "May 01",
        "May 08",
        "May 16",
        "May 23",
        "Jun 01",
        "Jun 08",
        "Jun 15",
        "Jun 22",
        "Jul 01",
        "Jul 08",
        "Jul 16",
        "Jul 23",
        "Aug 01",
        "Aug 08",
        "Aug 16",
        "Aug 23",
        "Sep 01",
        "Sep 08",
        "Sep 15",
        "Sep 22",
    ]
    return cmap, norm, tpos, tlbl


# ─────────────────────────────────────────────
# Cached extraction helpers
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=True)
def extract_series_single_year(
    dataset_key, ds_files, lat_name, lon_name, time_name, year, lat0, lon0
):
    ds = xr.open_mfdataset(
        ds_files,
        combine="by_coords",
        parallel=True,
        chunks={time_name: 365},
        engine=None,
    )
    ds = maybe_normalize_longitudes(ds, lon_name)
    rain_var = RAIN_VAR_BY_KEY.get(dataset_key, RAIN_VAR)
    da = ds[rain_var].sel({time_name: slice(f"{year}-01-01", f"{year}-12-31")})
    da = da.sel({lat_name: lat0, lon_name: lon0}, method="nearest").compute()
    p_lat = float(da[lat_name].values)
    p_lon = float(da[lon_name].values)
    df = da.to_dataframe(name="rain").reset_index()
    df[time_name] = pd.to_datetime(df[time_name])
    df = (
        df.rename(columns={time_name: "time"})
        .sort_values("time")
        .reset_index(drop=True)
    )
    df["dataset"] = dataset_key
    df["snapped_lat"] = p_lat
    df["snapped_lon"] = p_lon
    return df


@st.cache_data(show_spinner=True)
def extract_climatology(
    dataset_key, ds_files, lat_name, lon_name, time_name, y0, y1, lat0, lon0, agg_mode
):
    ds = xr.open_mfdataset(
        ds_files,
        combine="by_coords",
        parallel=True,
        chunks={time_name: 365},
        engine=None,
    )
    ds = maybe_normalize_longitudes(ds, lon_name)
    rain_var = RAIN_VAR_BY_KEY.get(dataset_key, RAIN_VAR)
    da = ds[rain_var].sel({time_name: slice(f"{y0}-01-01", f"{y1}-12-31")})
    da = da.sel({lat_name: lat0, lon_name: lon0}, method="nearest").compute()
    p_lat = float(da[lat_name].values)
    p_lon = float(da[lon_name].values)
    doy = da[time_name].dt.dayofyear
    clim = (
        da.groupby(doy).mean(dim=time_name, skipna=True)
        if agg_mode == "Mean"
        else da.groupby(doy).median(dim=time_name, skipna=True)
    )
    dv = clim[doy.name].values.astype(int)
    rv = clim.values
    keep = dv <= 365
    dv, rv = dv[keep], rv[keep]
    dates = pd.to_datetime("2001-01-01") + pd.to_timedelta(dv - 1, unit="D")
    return pd.DataFrame(
        {
            "time": dates,
            "rain": rv,
            "dataset": dataset_key,
            "snapped_lat": p_lat,
            "snapped_lon": p_lon,
        }
    )


# ─────────────────────────────────────────────
# Map computation
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=True)
def compute_single_year_doy_maps(
    ds_files,
    var,
    lat_name,
    lon_name,
    time_name,
    year,
    wet_day_mm,
    accum_days,
    accum_mm,
    dry_spell_days,
    lookahead_days,
    start_month,
    start_day,
    end_month,
    end_day,
):
    ds = xr.open_mfdataset(
        ds_files,
        combine="by_coords",
        parallel=True,
        chunks={time_name: 365},
        engine=None,
    )
    ds = maybe_normalize_longitudes(ds, lon_name)
    da = ds[var].sel({time_name: slice(f"{year}-01-01", f"{year}-12-31")}).compute()
    lv = da[lat_name].values
    lov = da[lon_name].values
    R = da.values
    times = pd.to_datetime(da[time_name].values)
    wm = np.full((len(lv), len(lov)), np.nan)
    om = np.full((len(lv), len(lov)), np.nan)
    for i in range(R.shape[1]):
        for j in range(R.shape[2]):
            rij = R[:, i, j]
            _, wi = detect_first_wet_spell(
                pd.Series(times),
                rij,
                float(wet_day_mm),
                int(accum_days),
                float(accum_mm),
                int(start_month),
                int(start_day),
                int(end_month),
                int(end_day),
            )
            ws = wet_spell_start_from_idx(pd.Series(times), rij, wi, float(wet_day_mm))
            if ws is not None:
                wm[i, j] = float(pd.Timestamp(ws).dayofyear)
            od, _ = detect_onset(
                pd.Series(times),
                rij,
                float(wet_day_mm),
                int(accum_days),
                float(accum_mm),
                int(dry_spell_days),
                int(lookahead_days),
                int(start_month),
                int(start_day),
                int(end_month),
                int(end_day),
            )
            if od is not None:
                om[i, j] = float(pd.Timestamp(od).dayofyear)
    return lv, lov, np.rint(wm), np.rint(om)


@st.cache_data(show_spinner=True)
def compute_agg_doy_maps(
    ds_files,
    var,
    lat_name,
    lon_name,
    time_name,
    y0,
    y1,
    agg_mode,
    wet_day_mm,
    accum_days,
    accum_mm,
    dry_spell_days,
    lookahead_days,
    start_month,
    start_day,
    end_month,
    end_day,
):
    ds = xr.open_mfdataset(
        ds_files,
        combine="by_coords",
        parallel=True,
        chunks={time_name: 365},
        engine=None,
    )
    ds = maybe_normalize_longitudes(ds, lon_name)
    da_all = ds[var].sel({time_name: slice(f"{y0}-01-01", f"{y1}-12-31")})
    lv = da_all[lat_name].values
    lov = da_all[lon_name].values
    wl, ol = [], []
    for yy in range(y0, y1 + 1):
        da = da_all.sel({time_name: slice(f"{yy}-01-01", f"{yy}-12-31")}).compute()
        R = da.values
        times = pd.to_datetime(da[time_name].values)
        wm = np.full((len(lv), len(lov)), np.nan)
        om = np.full((len(lv), len(lov)), np.nan)
        for i in range(R.shape[1]):
            for j in range(R.shape[2]):
                rij = R[:, i, j]
                _, wi = detect_first_wet_spell(
                    pd.Series(times),
                    rij,
                    float(wet_day_mm),
                    int(accum_days),
                    float(accum_mm),
                    int(start_month),
                    int(start_day),
                    int(end_month),
                    int(end_day),
                )
                ws = wet_spell_start_from_idx(
                    pd.Series(times), rij, wi, float(wet_day_mm)
                )
                if ws is not None:
                    wm[i, j] = float(pd.Timestamp(ws).dayofyear)
                od, _ = detect_onset(
                    pd.Series(times),
                    rij,
                    float(wet_day_mm),
                    int(accum_days),
                    float(accum_mm),
                    int(dry_spell_days),
                    int(lookahead_days),
                    int(start_month),
                    int(start_day),
                    int(end_month),
                    int(end_day),
                )
                if od is not None:
                    om[i, j] = float(pd.Timestamp(od).dayofyear)
        wl.append(wm)
        ol.append(om)
    ws_stk = np.stack(wl, axis=0)
    os_stk = np.stack(ol, axis=0)
    aw = (
        np.nanmedian(ws_stk, axis=0)
        if agg_mode == "Median"
        else np.nanmean(ws_stk, axis=0)
    )
    ao = (
        np.nanmedian(os_stk, axis=0)
        if agg_mode == "Median"
        else np.nanmean(os_stk, axis=0)
    )
    s0, s1 = 121, 265
    aw = np.where((np.rint(aw) >= s0) & (np.rint(aw) <= s1), np.rint(aw), np.nan)
    ao = np.where((np.rint(ao) >= s0) & (np.rint(ao) <= s1), np.rint(ao), np.nan)
    return lv, lov, aw, ao


def clip_and_mask_map(Z, lat_vals, lon_vals, eth_geom, eth_bounds, mask_da):
    lo_mn, la_mn, lo_mx, la_mx = eth_bounds
    # Use integer indices to avoid np.ix_ shape mismatches
    lat_idx = np.where((lat_vals >= la_mn) & (lat_vals <= la_mx))[0]
    lon_idx = np.where((lon_vals >= lo_mn) & (lon_vals <= lo_mx))[0]
    # Clamp to actual array bounds
    lat_idx = lat_idx[lat_idx < Z.shape[0]]
    lon_idx = lon_idx[lon_idx < Z.shape[1]]
    Z = Z[np.ix_(lat_idx, lon_idx)]
    lat_vals = lat_vals[lat_idx]
    lon_vals = lon_vals[lon_idx]
    inside = make_inside_mask(lat_vals, lon_vals, eth_geom)
    Z = np.where(inside, Z, np.nan)
    if mask_da is not None:
        Z = apply_nc_mask_to_Z(Z, lat_vals, lon_vals, mask_da)
    return lat_vals, lon_vals, Z


def _nanminmax(a):
    fin = a[np.isfinite(a)]
    if len(fin) == 0:
        return 0.0, 1.0
    lo, hi = float(fin.min()), float(fin.max())
    return (lo, hi) if lo != hi else (lo, lo + 1e-6)


# ─────────────────────────────────────────────
# Map plot helpers (matplotlib — maps stay static)
# ─────────────────────────────────────────────
def plot_doy_map(
    ax, Lon, Lat, Z, eth, title, cb_label, cmap_jjas, norm_jjas, tpos, tlbl, fig
):
    pcm = ax.pcolormesh(Lon, Lat, Z, shading="auto", cmap=cmap_jjas, norm=norm_jjas)
    eth.boundary.plot(ax=ax, linewidth=LINEWIDTH_ETH, edgecolor="black", zorder=10)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("Lon")
    ax.set_ylabel("Lat")
    cb = fig.colorbar(pcm, ax=ax)
    cb.set_label(cb_label, fontsize=8)
    cb.set_ticks(tpos)
    cb.set_ticklabels(tlbl, fontsize=6)


def plot_rain_map(
    ax, Lon, Lat, Z, eth, title, cb_label, fig, vmin=None, vmax=None, cmap="Blues"
):
    if vmin is None or vmax is None:
        vmin, vmax = _nanminmax(Z)
    pcm = ax.pcolormesh(Lon, Lat, Z, shading="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    eth.boundary.plot(ax=ax, linewidth=LINEWIDTH_ETH, edgecolor="black", zorder=10)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("Lon")
    ax.set_ylabel("Lat")
    cb = fig.colorbar(pcm, ax=ax)
    cb.set_label(cb_label, fontsize=8)


# ─────────────────────────────────────────────
# Plotly timeseries helpers (interactive legend)
# ─────────────────────────────────────────────
def build_plotly_ts(title: str) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Rainfall (mm/day)",
        legend=dict(
            orientation="v",
            x=1.01,
            xanchor="left",
            y=1,
            bgcolor="rgba(0,0,0,0)",
            bordercolor="white",
            borderwidth=1,
            itemclick="toggle",
            itemdoubleclick="toggleothers",
            font=dict(color="white", size=13),
        ),
        hovermode="x unified",
        height=500,
        margin=dict(r=280),
        yaxis2=dict(
            overlaying="y",
            range=[0, 1],
            showticklabels=False,
            showgrid=False,
            zeroline=False,
            fixedrange=True,
        ),
    )
    return fig


def add_rain_trace(fig, times, rain_vals, label, color, show_markers=True):
    fig.add_trace(
        go.Scatter(
            x=list(times),
            y=list(rain_vals),
            mode="lines+markers" if show_markers else "lines",
            name=label,
            line=dict(color=color, width=1.4),
            marker=dict(size=3, color=color) if show_markers else dict(),
            hovertemplate="%{y:.2f} mm<extra>" + label + "</extra>",
        )
    )


def add_wetspell_onset_traces(fig, df_ts, dataset_key, label_prefix):
    """Add wet spell start and onset as toggleable vertical line traces."""
    colors = DATASET_COLORS.get(dataset_key, {"wet": "gray", "onset": "black"})
    times = df_ts["time"]
    rain = df_ts["rain"].values

    # guard: need at least one non-NaN value
    if not np.any(np.isfinite(rain)):
        return

    _, wi = detect_first_wet_spell(
        times,
        rain,
        float(wet_day_mm),
        int(accum_days),
        float(accum_mm),
        int(start_month),
        int(start_day),
        int(end_month),
        int(end_day),
    )
    ws = wet_spell_start_from_idx(times, rain, wi, float(wet_day_mm))

    od, _ = detect_onset(
        times,
        rain,
        float(wet_day_mm),
        int(accum_days),
        float(accum_mm),
        int(dry_spell_days),
        int(lookahead_days),
        int(start_month),
        int(start_day),
        int(end_month),
        int(end_day),
    )

    if ws is not None:
        lbl = f"{label_prefix} wet spell ({ws.strftime('%d %b')})"
        fig.add_trace(
            go.Scatter(
                x=[ws, ws],
                y=[0, 1],
                mode="lines",
                name=lbl,
                line=dict(color=colors["wet"], width=2, dash="dash"),
                yaxis="y2",
                hovertemplate=f"Wet spell start: {ws.date()}<extra></extra>",
                showlegend=True,
            )
        )

    if od is not None:
        lbl = f"{label_prefix} onset ({od.strftime('%d %b')})"
        fig.add_trace(
            go.Scatter(
                x=[od, od],
                y=[0, 1],
                mode="lines",
                name=lbl,
                line=dict(color=colors["onset"], width=2.5, dash="solid"),
                yaxis="y2",
                hovertemplate=f"Onset: {od.date()}<extra></extra>",
                showlegend=True,
            )
        )


def add_wet_threshold_trace(fig, wet_thresh, times):
    t0 = times.iloc[0] if hasattr(times, "iloc") else times[0]
    t1 = times.iloc[-1] if hasattr(times, "iloc") else times[-1]
    # Use legendgroup so even if called multiple times only one entry appears
    fig.add_trace(
        go.Scatter(
            x=[t0, t1],
            y=[wet_thresh, wet_thresh],
            mode="lines",
            name=f"Wet threshold ({wet_thresh:g} mm/day)",
            legendgroup="wet_threshold",
            showlegend=True,
            line=dict(color="gray", width=1.2, dash="dot"),
            hoverinfo="skip",
        )
    )


def build_station_map(meta: pd.DataFrame, selected: Optional[str]) -> go.Figure:
    """Interactive station map styled to match the matplotlib weather maps:
    white background, black Ethiopia boundary, lat/lon axes, red triangle markers.
    All labels grey by default; selected station label turns black+bold."""
    import json

    eth_path = Path("data/ethiopia.geojson")
    with open(eth_path) as f:
        eth_geojson = json.load(f)

    lats = meta["lat"].tolist()
    lons = meta["lon"].tolist()
    names = meta.index.tolist()

    fig = go.Figure()

    # ── Ethiopia boundary (black outline, white fill) ──
    geom = eth_geojson["features"][0]["geometry"]
    polys = [geom["coordinates"]] if geom["type"] == "Polygon" else geom["coordinates"]
    for poly in polys:
        rings = [poly] if not isinstance(poly[0][0], list) else poly
        for ring in rings:
            xs = [pt[0] for pt in ring] + [None]
            ys = [pt[1] for pt in ring] + [None]
            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=ys,
                    mode="lines",
                    line=dict(color="black", width=2),
                    fill="toself",
                    fillcolor="white",
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

    # ── All stations: grey labels ──
    unsel_names = [n for n in names if n != selected]
    unsel_lats = [lats[i] for i, n in enumerate(names) if n != selected]
    unsel_lons = [lons[i] for i, n in enumerate(names) if n != selected]

    fig.add_trace(
        go.Scatter(
            x=unsel_lons,
            y=unsel_lats,
            text=unsel_names,
            customdata=unsel_names,
            mode="markers+text",
            marker=dict(
                symbol="triangle-up",
                size=9,
                color="red",
                line=dict(color="white", width=0.6),
            ),
            textposition="top center",
            textfont=dict(size=9, color="#888888"),
            hovertemplate="<b>%{text}</b><br>%{y:.3f}°N, %{x:.3f}°E<extra></extra>",
            showlegend=False,
        )
    )

    # ── Selected station: black bold label, drawn on top ──
    if selected and selected in meta.index:
        si = names.index(selected)
        fig.add_trace(
            go.Scatter(
                x=[lons[si]],
                y=[lats[si]],
                text=[selected],
                customdata=[selected],
                mode="markers+text",
                marker=dict(
                    symbol="triangle-up",
                    size=11,
                    color="red",
                    line=dict(color="white", width=0.8),
                ),
                textposition="top center",
                textfont=dict(size=10, color="black", family="Arial Black"),
                hovertemplate=f"<b>{selected}</b><br>{lats[si]:.3f}°N, {lons[si]:.3f}°E<extra></extra>",
                showlegend=False,
            )
        )

    # ── Axes styled like the matplotlib maps ──
    lon_ticks = [34, 36, 38, 40, 42, 44, 46]
    lat_ticks = [4, 6, 8, 10, 12, 14]

    fig.update_layout(
        xaxis=dict(
            range=[32.5, 48.0],
            tickvals=lon_ticks,
            ticktext=[f"{v}°E" for v in lon_ticks],
            tickfont=dict(size=11, color="black"),
            title=dict(text="Lon", font=dict(size=12, color="black")),
            showgrid=False,
            zeroline=False,
            scaleanchor="y",
            scaleratio=1,
            linecolor="black",
            linewidth=1,
            mirror=True,
        ),
        yaxis=dict(
            range=[3.2, 15.2],
            tickvals=lat_ticks,
            ticktext=[f"{v}°N" for v in lat_ticks],
            tickfont=dict(size=11, color="black"),
            title=dict(text="Lat", font=dict(size=12, color="black")),
            showgrid=False,
            zeroline=False,
            linecolor="black",
            linewidth=1,
            mirror=True,
        ),
        margin=dict(l=60, r=20, t=40, b=50),
        height=500,
        title=dict(
            text="Rainfall Station Locations — click to select",
            font=dict(size=13, color="black"),
            x=0.5,
            xanchor="center",
        ),
        paper_bgcolor="white",
        plot_bgcolor="white",
        dragmode=False,
    )
    return fig


# ─────────────────────────────────────────────
# Dataset registry
# ─────────────────────────────────────────────
DS_BY_KEY: Dict[str, xr.Dataset] = {}
FILES_BY_KEY: Dict[str, List[str]] = {}
COORDS_BY_KEY: Dict[str, Tuple[str, str, str]] = {}


def ensure_grid_loaded(key: str):
    if key in DS_BY_KEY:
        return
    ds_i, fi = open_dataset_folder(DATASET_FOLDERS[key])
    la, lo, ti = guess_lat_lon_time_names(ds_i)
    ds_i = maybe_normalize_longitudes(ds_i, lo)
    DS_BY_KEY[key] = ds_i
    FILES_BY_KEY[key] = fi
    COORDS_BY_KEY[key] = (la, lo, ti)


# ═════════════════════════════════════════════
# APP STARTS
# ═════════════════════════════════════════════
st.title("Rainfall Onset Explorer — Ethiopia")

# ── ① Dataset selection ──
st.subheader("① Select dataset(s)")
selected_datasets = st.multiselect(
    "Choose dataset(s)",
    options=list(DATASET_FOLDERS.keys()) + [STATION_KEY],
    default=["CHIRPS"],
)
if not selected_datasets:
    st.warning("Please select at least one dataset to continue.")
    st.stop()

has_chirps = "CHIRPS" in selected_datasets
has_enacts = "ENACTS" in selected_datasets
has_station = STATION_KEY in selected_datasets
selected_grids = [k for k in selected_datasets if k in DATASET_FOLDERS]

for key in selected_grids:
    ensure_grid_loaded(key)

emi_meta: Optional[pd.DataFrame] = None
emi_ts: Optional[pd.DataFrame] = None
if has_station:
    if not STATION_CSV_PATH.exists():
        st.error(f"EMI Stations selected but file not found: {STATION_CSV_PATH.name}")
        st.stop()
    emi_meta, emi_ts = load_emi_station_csv(STATION_CSV_PATH)

# ── Shared year range ──
year_ranges = {}
for key in selected_grids:
    _, _, ti = COORDS_BY_KEY[key]
    year_ranges[key] = dataset_year_range(DS_BY_KEY[key], ti)
if has_station and emi_ts is not None:
    year_ranges[STATION_KEY] = (
        int(emi_ts.index.min().year),
        int(emi_ts.index.max().year),
    )

if not year_ranges:
    st.error("No data available.")
    st.stop()

range_list = list(year_ranges.values())
year_min = max(r[0] for r in range_list)
year_max = min(r[1] for r in range_list)

if year_min > year_max:
    st.error(
        "Selected datasets have no overlapping years. Their individual ranges are:"
    )
    for key, (y0, y1) in year_ranges.items():
        st.error(f"  • {key}: {y0}–{y1}")
    st.stop()

# Warn if the intersection is narrower than any individual dataset
if len(year_ranges) > 1:
    msgs = []
    for key, (y0, y1) in year_ranges.items():
        if y0 > year_min or y1 < year_max:
            msgs.append(f"**{key}**: {y0}–{y1}")
    if msgs:
        st.info(
            f"Year selector limited to the overlapping range **{year_min}–{year_max}** "
            f"across all selected datasets. Individual ranges: {', '.join(msgs)}."
        )

# Note any datasets with partial-year (seasonal) coverage
seasonal_notes = [
    f"**{key}** ({note})"
    for key, note in SEASONAL_COVERAGE_NOTE.items()
    if key in year_ranges
]
if seasonal_notes:
    st.warning(
        f"Note: {', '.join(seasonal_notes)} — data outside this window will appear "
        "as missing. Onset/wet spell detection and JJAS maps are unaffected, but "
        "full-year timeseries and non-JJAS maps will have gaps."
    )

year_list = list(range(year_min, year_max + 1))
# Pick a sensible default: prefer a recent-ish year within range
year_default = min(max(year_max - 5, year_min), year_max)

# ── Grid domain (only needed when grids are selected) ──
if selected_grids:
    lat_ranges, lon_ranges, tmin_list, tmax_list = [], [], [], []
    for key in selected_grids:
        la, lo, ti = COORDS_BY_KEY[key]
        ds_i = DS_BY_KEY[key]
        lat_ranges.append(
            (float(ds_i[la].min().compute()), float(ds_i[la].max().compute()))
        )
        lon_ranges.append(
            (float(ds_i[lo].min().compute()), float(ds_i[lo].max().compute()))
        )
        tm0, tm1 = dataset_time_range(ds_i, ti)
        tmin_list.append(tm0)
        tmax_list.append(tm1)
    lat_min = max(r[0] for r in lat_ranges)
    lat_max = min(r[1] for r in lat_ranges)
    lon_min = max(r[0] for r in lon_ranges)
    lon_max = min(r[1] for r in lon_ranges)
    tmin_common = max(tmin_list)
    tmax_common = min(tmax_list)


# ════════════════════════════════════════════
# CASE A — Grids only (CHIRPS and/or ENACTS)
# ════════════════════════════════════════════
if selected_grids and not has_station:

    # ── ② Select View(s) ──
    st.subheader("② Select View(s)")
    show_ts = st.checkbox("Timeseries", value=True, key="show_ts_a")
    show_map = st.checkbox("Map (Ethiopia)", value=True, key="show_map_a")
    if not show_ts and not show_map:
        st.warning("Select at least one view.")
        st.stop()

    # Link toggle — only shown when both views are active
    if show_ts and show_map:
        link_periods = st.checkbox(
            "🔗 Link Timeseries and Map Time Period",
            value=True,
            key="link_periods_a",
        )
        st.caption(
            "When linked, one shared time period applies to both views. "
            "Unlink to set independent time periods for the timeseries and map."
        )
    else:
        link_periods = True

    # helper: renders year / year-range widgets, returns (y0, y1, agg_mode, year_mode_str)
    def _year_selector(label_yr, key_prefix):
        yr_mode = st.radio(
            label_yr,
            ["Single year", "Year range"],
            horizontal=True,
            key=f"{key_prefix}_yrmode",
        )
        if yr_mode == "Single year":
            y = int(
                st.selectbox(
                    "Year",
                    year_list,
                    index=year_list.index(year_default),
                    key=f"{key_prefix}_yr",
                )
            )
            return y, y, "Mean", "Single year"
        else:
            cA, cB = st.columns(2)
            with cA:
                y0 = int(
                    st.selectbox(
                        "Start year", year_list, index=0, key=f"{key_prefix}_y0"
                    )
                )
            with cB:
                y1 = int(
                    st.selectbox(
                        "End year",
                        year_list,
                        index=len(year_list) - 1,
                        key=f"{key_prefix}_y1",
                    )
                )
            if y1 < y0:
                st.error("End year must be ≥ start year.")
                st.stop()
            agg = st.radio(
                "Aggregation (climatology / DOY maps)",
                ["Mean", "Median"],
                horizontal=True,
                key=f"{key_prefix}_agg",
            )
            return y0, y1, agg, "Year range"

    # ── Time period selection ──
    st.subheader("③ Time period")

    if link_periods or not (show_ts and show_map):
        # Single shared selector
        ts_y0, ts_y1, agg_mode_ts, ts_year_mode = _year_selector(
            "Year selection", "a_shared"
        )
        map_y0, map_y1, agg_mode_map, map_year_mode = (
            ts_y0,
            ts_y1,
            agg_mode_ts,
            ts_year_mode,
        )
    else:
        # Separate selectors
        if show_ts:
            st.markdown("**Timeseries time period**")
            ts_y0, ts_y1, agg_mode_ts, ts_year_mode = _year_selector(
                "Timeseries year selection", "a_ts"
            )
        if show_map:
            st.markdown("**Map time period**")
            map_y0, map_y1, agg_mode_map, map_year_mode = _year_selector(
                "Map year selection", "a_map"
            )

    # ── Point selection — only needed for timeseries ──
    if show_ts:
        st.subheader("④ Point selection")
        c1, c2 = st.columns(2)
        with c1:
            lat0 = st.number_input(
                "Latitude",
                min_value=float(lat_min),
                max_value=float(lat_max),
                value=float(np.clip(10.0, lat_min, lat_max)),
                format="%.4f",
                step=0.25,
            )
        with c2:
            lon0 = st.number_input(
                "Longitude",
                min_value=float(lon_min),
                max_value=float(lon_max),
                value=float(np.clip(40.0, lon_min, lon_max)),
                format="%.4f",
                step=0.05,
            )
        for key in selected_grids:
            la, lo, _ = COORDS_BY_KEY[key]
            ds_i = DS_BY_KEY[key]
            sl = float(ds_i[la].sel({la: lat0}, method="nearest").values)
            slo = float(ds_i[lo].sel({lo: lon0}, method="nearest").values)
            st.caption(f"{key} snapped to nearest grid cell: ({sl:.4f}°N, {slo:.4f}°E)")
    else:
        # map-only: set dummy point values (not used)
        lat0 = float(np.clip(10.0, lat_min, lat_max))
        lon0 = float(np.clip(40.0, lon_min, lon_max))

    # ── Map options — only when map is active ──
    if show_map:
        st.subheader("⑤ Map options")
        use_default_mask = st.checkbox(
            "Apply Ethiopia .nc mask", value=True, key="mask_a"
        )
        map_kind = st.radio(
            "Map type",
            [
                "Seasonal mean rainfall (JJAS)",
                "Daily rainfall map (selected date)",
                "Wet spell date (DOY)",
                "Onset date (DOY)",
            ],
            horizontal=True,
            key="a_map_kind",
        )
        date_sel = None
        if map_kind == "Daily rainfall map (selected date)":
            dflt = min(
                max(pd.Timestamp(year=map_y0, month=7, day=15), tmin_common),
                tmax_common,
            )
            date_in = st.date_input(
                "Select date",
                value=dflt.date(),
                min_value=tmin_common.date(),
                max_value=tmax_common.date(),
            )
            date_sel = pd.Timestamp(date_in)

    # ── TIMESERIES PANEL ──
    if show_ts:
        st.subheader("Timeseries")
        title_ts = (
            f"Daily rainfall — ({lat0:.3f}°N, {lon0:.3f}°E) — {ts_y0}"
            if ts_year_mode == "Single year"
            else f"{agg_mode_ts} climatology — ({lat0:.3f}°N, {lon0:.3f}°E) — {ts_y0}–{ts_y1}"
        )
        fig_ts = build_plotly_ts(title_ts)
        last_times = None
        for key in selected_grids:
            la, lo, ti = COORDS_BY_KEY[key]
            fi = FILES_BY_KEY[key]
            clr = DATASET_COLORS.get(key, {}).get("rain", "black")
            if ts_year_mode == "Single year":
                df_ts = extract_series_single_year(
                    key, tuple(fi), la, lo, ti, ts_y0, lat0, lon0
                )
                add_rain_trace(
                    fig_ts,
                    df_ts["time"],
                    df_ts["rain"],
                    label=f"{key} daily rainfall",
                    color=clr,
                )
                add_wetspell_onset_traces(fig_ts, df_ts, key, key)
                last_times = df_ts["time"]
            else:
                df_c = extract_climatology(
                    key, tuple(fi), la, lo, ti, ts_y0, ts_y1, lat0, lon0, agg_mode_ts
                )
                add_rain_trace(
                    fig_ts,
                    df_c["time"],
                    df_c["rain"],
                    label=f"{key} {agg_mode_ts} ({ts_y0}–{ts_y1})",
                    color=clr,
                    show_markers=True,
                )
                add_wetspell_onset_traces(fig_ts, df_c, key, key)
                last_times = df_c["time"]
        # Threshold added once after all datasets
        if last_times is not None:
            add_wet_threshold_trace(fig_ts, float(wet_day_mm), last_times)
        st.plotly_chart(fig_ts, use_container_width=True)

    # ── MAP PANEL ──
    if show_map:
        st.subheader("Map view")
        eth = load_ethiopia_boundary()
        eth_geom = eth.geometry.iloc[0]
        eth_bounds = tuple(map(float, eth.total_bounds))
        mask_da = load_default_mask_da() if use_default_mask else None
        cmap_jjas, norm_jjas, tpos, tlbl = build_jjas_doy_cmap()
        yr_label = str(map_y0) if map_y0 == map_y1 else f"{map_y0}–{map_y1}"

        def get_map_Z(key):
            la, lo, ti = COORDS_BY_KEY[key]
            fi = FILES_BY_KEY[key]
            ds_i = DS_BY_KEY[key]
            if map_kind == "Seasonal mean rainfall (JJAS)":
                _rv = RAIN_VAR_BY_KEY.get(key, RAIN_VAR)
                da = ds_i[_rv].sel({ti: slice(f"{map_y0}-01-01", f"{map_y1}-12-31")})
                da = (
                    da.where(da[ti].dt.month.isin([6, 7, 8, 9]), drop=True)
                    .mean(dim=ti)
                    .compute()
                )
                return (
                    clip_and_mask_map(
                        da.values,
                        da[la].values,
                        da[lo].values,
                        eth_geom,
                        eth_bounds,
                        mask_da,
                    ),
                    "rain",
                )
            elif map_kind == "Daily rainfall map (selected date)":
                _rv = RAIN_VAR_BY_KEY.get(key, RAIN_VAR)
                da = ds_i[_rv].sel({ti: date_sel}, method="nearest").compute()
                return (
                    clip_and_mask_map(
                        da.values,
                        da[la].values,
                        da[lo].values,
                        eth_geom,
                        eth_bounds,
                        mask_da,
                    ),
                    "rain",
                )
            else:
                if map_year_mode == "Single year":
                    _rv = RAIN_VAR_BY_KEY.get(key, RAIN_VAR)
                    lv, lov, wd, od = compute_single_year_doy_maps(
                        tuple(fi),
                        _rv,
                        la,
                        lo,
                        ti,
                        map_y0,
                        float(wet_day_mm),
                        int(accum_days),
                        float(accum_mm),
                        int(dry_spell_days),
                        int(lookahead_days),
                        int(start_month),
                        int(start_day),
                        int(end_month),
                        int(end_day),
                    )
                else:
                    _rv = RAIN_VAR_BY_KEY.get(key, RAIN_VAR)
                    lv, lov, wd, od = compute_agg_doy_maps(
                        tuple(fi),
                        _rv,
                        la,
                        lo,
                        ti,
                        map_y0,
                        map_y1,
                        agg_mode_map,
                        float(wet_day_mm),
                        int(accum_days),
                        float(accum_mm),
                        int(dry_spell_days),
                        int(lookahead_days),
                        int(start_month),
                        int(start_day),
                        int(end_month),
                        int(end_day),
                    )
                Z0 = wd if map_kind == "Wet spell date (DOY)" else od
                return (
                    clip_and_mask_map(Z0, lv, lov, eth_geom, eth_bounds, mask_da),
                    "doy",
                )

        if len(selected_grids) == 1:
            key = selected_grids[0]
            (lv, lov, Z), kind = get_map_Z(key)
            fig_m, ax = plt.subplots(figsize=(10, 6))
            Lon, Lat = np.meshgrid(lov, lv)
            if kind == "rain":
                plot_rain_map(
                    ax,
                    Lon,
                    Lat,
                    Z,
                    eth,
                    f"{key} — {map_kind} — {yr_label}",
                    "Rainfall (mm/day)",
                    fig_m,
                )
            else:
                cb_lbl = (
                    "Wet Spell DOY"
                    if map_kind == "Wet spell date (DOY)"
                    else "Onset DOY"
                )
                plot_doy_map(
                    ax,
                    Lon,
                    Lat,
                    Z,
                    eth,
                    f"{key} — {map_kind} — {yr_label}",
                    cb_lbl,
                    cmap_jjas,
                    norm_jjas,
                    tpos,
                    tlbl,
                    fig_m,
                )
            st.pyplot(fig_m)

        else:
            # Generic N-grid layout: individual maps + pairwise differences, all in one row
            from itertools import combinations

            results = {key: get_map_Z(key) for key in selected_grids}

            n = len(selected_grids)
            pairs = list(combinations(selected_grids, 2))
            n_cols = n + len(pairs)
            # All maps use the same fixed size so they render identically in their columns
            fig_w, fig_h = 4, 4

            all_Z_rain = [
                results[k][0][2] for k in selected_grids if results[k][1] == "rain"
            ]
            vmin_shared = vmax_shared = None
            if len(all_Z_rain) > 1:
                vmin_shared, vmax_shared = _nanminmax(
                    np.concatenate([z.ravel() for z in all_Z_rain])
                )

            # Build a flat ordered list of all panels: individual maps first, then diffs
            # Each entry: ("individual", key) or ("diff", kA, kB)
            panels = [("individual", key) for key in selected_grids]
            panels += [("diff", kA, kB) for kA, kB in pairs]

            MAX_COLS = 3
            # Render in rows of at most MAX_COLS
            for row_start in range(0, len(panels), MAX_COLS):
                row_panels = panels[row_start : row_start + MAX_COLS]
                row_cols = st.columns(len(row_panels))

                for col_idx, panel in enumerate(row_panels):
                    with row_cols[col_idx]:
                        if panel[0] == "individual":
                            key = panel[1]
                            (lv, lov, Z), kind = results[key]
                            st.markdown(f"**{key}**")
                            fig_m, ax = plt.subplots(figsize=(fig_w, fig_h))
                            Lon, Lat = np.meshgrid(lov, lv)
                            if kind == "rain":
                                plot_rain_map(
                                    ax,
                                    Lon,
                                    Lat,
                                    Z,
                                    eth,
                                    f"{key} — {yr_label}",
                                    "mm/day",
                                    fig_m,
                                    vmin_shared,
                                    vmax_shared,
                                )
                            else:
                                cb_lbl = (
                                    "Wet Spell DOY"
                                    if map_kind == "Wet spell date (DOY)"
                                    else "Onset DOY"
                                )
                                plot_doy_map(
                                    ax,
                                    Lon,
                                    Lat,
                                    Z,
                                    eth,
                                    f"{key} — {yr_label}",
                                    cb_lbl,
                                    cmap_jjas,
                                    norm_jjas,
                                    tpos,
                                    tlbl,
                                    fig_m,
                                )
                            st.pyplot(fig_m)

                        else:  # diff panel
                            _, kA, kB = panel
                            (lvA, lovA, ZA), kindA = results[kA]
                            (lvB, lovB, ZB), _ = results[kB]
                            st.markdown(f"**{kA} − {kB}**")

                            # Auto-regrid finer onto coarser if shapes differ
                            resample_note = None
                            if ZA.shape != ZB.shape:
                                from scipy.interpolate import RegularGridInterpolator

                                if ZA.size >= ZB.size:
                                    interp = RegularGridInterpolator(
                                        (lvA, lovA),
                                        ZA,
                                        method="linear",
                                        bounds_error=False,
                                        fill_value=np.nan,
                                    )
                                    LonG, LatG = np.meshgrid(lovB, lvB)
                                    ZA = interp(
                                        np.stack([LatG.ravel(), LonG.ravel()], axis=1)
                                    ).reshape(LonG.shape)
                                    lvA, lovA = lvB, lovB
                                    resample_note = f"ℹ️ {kA} resampled to {kB} resolution for difference map"
                                else:
                                    interp = RegularGridInterpolator(
                                        (lvB, lovB),
                                        ZB,
                                        method="linear",
                                        bounds_error=False,
                                        fill_value=np.nan,
                                    )
                                    LonG, LatG = np.meshgrid(lovA, lvA)
                                    ZB = interp(
                                        np.stack([LatG.ravel(), LonG.ravel()], axis=1)
                                    ).reshape(LonG.shape)
                                    resample_note = f"ℹ️ {kB} resampled to {kA} resolution for difference map"

                            Zdiff = ZA - ZB
                            fig_m, ax = plt.subplots(figsize=(fig_w, fig_h))
                            Lon, Lat = np.meshgrid(lovA, lvA)
                            if kindA == "rain":
                                ma = float(np.nanmax(np.abs(Zdiff))) or 1e-6
                                pcm = ax.pcolormesh(
                                    Lon,
                                    Lat,
                                    Zdiff,
                                    shading="auto",
                                    cmap="RdBu_r",
                                    norm=TwoSlopeNorm(vcenter=0.0, vmin=-ma, vmax=ma),
                                )
                                eth.boundary.plot(
                                    ax=ax,
                                    linewidth=LINEWIDTH_ETH,
                                    edgecolor="black",
                                    zorder=10,
                                )
                                ax.set_title(f"{kA}−{kB} — {yr_label}", fontsize=9)
                                cb = fig_m.colorbar(pcm, ax=ax)
                                cb.set_label(f"mm/day ({kA}−{kB})", fontsize=8)
                            else:
                                lvls = np.arange(-30, 31, 5)
                                pcm = ax.pcolormesh(
                                    Lon,
                                    Lat,
                                    np.clip(Zdiff, -30, 30),
                                    shading="auto",
                                    cmap=plt.cm.get_cmap("RdBu_r", len(lvls) - 1),
                                    norm=BoundaryNorm(lvls, len(lvls) - 1),
                                )
                                eth.boundary.plot(
                                    ax=ax,
                                    linewidth=LINEWIDTH_ETH,
                                    edgecolor="black",
                                    zorder=10,
                                )
                                ax.set_title(
                                    f"DOY diff {kA}−{kB} — {yr_label}", fontsize=9
                                )
                                cb = fig_m.colorbar(pcm, ax=ax, ticks=lvls)
                                cb.set_label(f"DOY ({kA}−{kB})", fontsize=8)
                            ax.set_xlabel("Lon")
                            ax.set_ylabel("Lat")
                            st.pyplot(fig_m)
                            # Caption appears below the map
                            if resample_note:
                                st.caption(resample_note)

# ════════════════════════════════════════════
# CASE B — EMI Stations only
# ════════════════════════════════════════════
elif has_station and not selected_grids:

    st.info("Map view is not available for EMI Stations. Showing timeseries only.")
    # ── Interactive station map picker ──
    st.subheader("② Select EMI station")
    station_list = list(emi_meta.index)

    # No default — None until user clicks
    if "selected_station" not in st.session_state:
        st.session_state["selected_station"] = None

    map_fig = build_station_map(emi_meta, st.session_state["selected_station"])
    clicked = st.plotly_chart(
        map_fig, use_container_width=True, on_select="rerun", key="station_map"
    )

    # Handle click: extract station name from customdata
    if clicked and clicked.get("selection", {}).get("points"):
        pt = clicked["selection"]["points"][0]
        cd = pt.get("customdata")
        if isinstance(cd, list):
            cd = cd[0]
        if cd and cd in station_list:
            st.session_state["selected_station"] = cd

    station_name = st.session_state["selected_station"]
    if station_name is None:
        st.info("Click a station on the map to select it.")
        st.stop()

    st_lat = float(emi_meta.loc[station_name, "lat"])
    st_lon = float(emi_meta.loc[station_name, "lon"])
    st.caption(f"Selected: **{station_name}** — ({st_lat:.4f}°N, {st_lon:.4f}°E)")

    ts_year_mode = st.radio(
        "Year selection",
        ["Single year", "Year range"],
        horizontal=True,
        key="b_ts_mode",
    )
    if ts_year_mode == "Single year":
        ts_year = int(
            st.selectbox(
                "Year", year_list, index=year_list.index(year_default), key="b_yr"
            )
        )
        ts_y0 = ts_y1 = ts_year
    else:
        cA, cB = st.columns(2)
        with cA:
            ts_y0 = int(st.selectbox("Start year", year_list, index=0, key="b_y0"))
        with cB:
            ts_y1 = int(
                st.selectbox(
                    "End year", year_list, index=len(year_list) - 1, key="b_y1"
                )
            )
        if ts_y1 < ts_y0:
            st.error("End year must be ≥ start year.")
            st.stop()

    s = emi_ts[station_name].loc[f"{ts_y0}-01-01":f"{ts_y1}-12-31"]
    df_st = s.rename("rain").to_frame().reset_index()
    df_st.columns = ["time", "rain"]

    if ts_year_mode == "Single year":
        fig_ts = build_plotly_ts(
            f"Station: {station_name} ({st_lat:.3f}°N, {st_lon:.3f}°E) — {ts_y0}"
        )
        add_rain_trace(
            fig_ts,
            df_st["time"],
            df_st["rain"],
            label=f"{station_name} daily rainfall",
            color=DATASET_COLORS[STATION_KEY]["rain"],
            show_markers=True,
        )
        add_wetspell_onset_traces(fig_ts, df_st, STATION_KEY, station_name)
        add_wet_threshold_trace(fig_ts, float(wet_day_mm), df_st["time"])
    else:
        df_st["doy"] = df_st["time"].dt.dayofyear
        clim = df_st.groupby("doy")["rain"].mean().reset_index()
        clim = clim[clim["doy"] <= 365].copy()
        clim["time"] = pd.to_datetime("2001-01-01") + pd.to_timedelta(
            clim["doy"] - 1, unit="D"
        )
        fig_ts = build_plotly_ts(
            f"Station climatology: {station_name} — {ts_y0}–{ts_y1}"
        )
        add_rain_trace(
            fig_ts,
            clim["time"],
            clim["rain"],
            label=f"{station_name} mean climatology ({ts_y0}–{ts_y1})",
            color=DATASET_COLORS[STATION_KEY]["rain"],
            show_markers=True,
        )
        add_wetspell_onset_traces(fig_ts, clim, STATION_KEY, station_name)
        add_wet_threshold_trace(fig_ts, float(wet_day_mm), clim["time"])

    st.plotly_chart(fig_ts, use_container_width=True)


# ════════════════════════════════════════════
# CASE C — EMI Stations + grids
# ════════════════════════════════════════════
elif has_station and selected_grids:

    st.info(
        "Map view is not available when EMI Stations are selected. Showing timeseries only."
    )
    # ── Interactive station map picker ──
    st.subheader("② Select EMI station")
    station_list = list(emi_meta.index)

    # No default — None until user clicks
    if "selected_station" not in st.session_state:
        st.session_state["selected_station"] = None

    map_fig = build_station_map(emi_meta, st.session_state["selected_station"])
    clicked = st.plotly_chart(
        map_fig, use_container_width=True, on_select="rerun", key="station_map"
    )

    # Handle click: extract station name from customdata
    if clicked and clicked.get("selection", {}).get("points"):
        pt = clicked["selection"]["points"][0]
        cd = pt.get("customdata")
        if isinstance(cd, list):
            cd = cd[0]
        if cd and cd in station_list:
            st.session_state["selected_station"] = cd

    station_name = st.session_state["selected_station"]
    if station_name is None:
        st.info("Click a station on the map to select it.")
        st.stop()

    st_lat = float(emi_meta.loc[station_name, "lat"])
    st_lon = float(emi_meta.loc[station_name, "lon"])
    st.caption(f"Selected: **{station_name}** — ({st_lat:.4f}°N, {st_lon:.4f}°E)")

    for key in selected_grids:
        la, lo, _ = COORDS_BY_KEY[key]
        ds_i = DS_BY_KEY[key]
        sl = float(ds_i[la].sel({la: st_lat}, method="nearest").values)
        slo = float(ds_i[lo].sel({lo: st_lon}, method="nearest").values)
        st.caption(f"{key} nearest grid cell: ({sl:.4f}°N, {slo:.4f}°E)")

    ts_year_mode = st.radio(
        "Year selection",
        ["Single year", "Year range"],
        horizontal=True,
        key="c_ts_mode",
    )
    if ts_year_mode == "Single year":
        ts_year = int(
            st.selectbox(
                "Year", year_list, index=year_list.index(year_default), key="c_yr"
            )
        )
        ts_y0 = ts_y1 = ts_year
        agg_mode_ts = "Mean"
    else:
        cA, cB = st.columns(2)
        with cA:
            ts_y0 = int(st.selectbox("Start year", year_list, index=0, key="c_y0"))
        with cB:
            ts_y1 = int(
                st.selectbox(
                    "End year", year_list, index=len(year_list) - 1, key="c_y1"
                )
            )
        if ts_y1 < ts_y0:
            st.error("End year must be ≥ start year.")
            st.stop()
        agg_mode_ts = st.radio(
            "Climatology aggregation", ["Mean", "Median"], horizontal=True, key="c_agg"
        )

    s = emi_ts[station_name].loc[f"{ts_y0}-01-01":f"{ts_y1}-12-31"]
    df_st = s.rename("rain").to_frame().reset_index()
    df_st.columns = ["time", "rain"]

    if ts_year_mode == "Single year":
        fig_ts = build_plotly_ts(
            f"Station vs gridded — {station_name} ({st_lat:.3f}°N, {st_lon:.3f}°E) — {ts_y0}"
        )

        # Station series (with dots)
        add_rain_trace(
            fig_ts,
            df_st["time"],
            df_st["rain"],
            label=f"{station_name} daily rainfall",
            color=DATASET_COLORS[STATION_KEY]["rain"],
            show_markers=True,
        )
        add_wetspell_onset_traces(fig_ts, df_st, STATION_KEY, station_name)

        # Grid series
        for key in selected_grids:
            la, lo, ti = COORDS_BY_KEY[key]
            fi = FILES_BY_KEY[key]
            df_g = extract_series_single_year(
                key, tuple(fi), la, lo, ti, ts_y0, st_lat, st_lon
            )
            sl = df_g["snapped_lat"].iloc[0]
            slo = df_g["snapped_lon"].iloc[0]
            add_rain_trace(
                fig_ts,
                df_g["time"],
                df_g["rain"],
                label=f"{key} daily rainfall @ ({sl:.3f}°N, {slo:.3f}°E)",
                color=DATASET_COLORS.get(key, {}).get("rain", "black"),
                show_markers=True,
            )
            add_wetspell_onset_traces(fig_ts, df_g, key, key)

        add_wet_threshold_trace(fig_ts, float(wet_day_mm), df_st["time"])

    else:
        fig_ts = build_plotly_ts(
            f"Climatology — {station_name} vs gridded — {ts_y0}–{ts_y1}"
        )

        # Station climatology
        df_st["doy"] = df_st["time"].dt.dayofyear
        clim_st = df_st.groupby("doy")["rain"].mean().reset_index()
        clim_st = clim_st[clim_st["doy"] <= 365].copy()
        clim_st["time"] = pd.to_datetime("2001-01-01") + pd.to_timedelta(
            clim_st["doy"] - 1, unit="D"
        )
        add_rain_trace(
            fig_ts,
            clim_st["time"],
            clim_st["rain"],
            label=f"{station_name} mean ({ts_y0}–{ts_y1})",
            color=DATASET_COLORS[STATION_KEY]["rain"],
            show_markers=True,
        )
        add_wetspell_onset_traces(fig_ts, clim_st, STATION_KEY, station_name)

        # Grid climatologies
        for key in selected_grids:
            la, lo, ti = COORDS_BY_KEY[key]
            fi = FILES_BY_KEY[key]
            df_c = extract_climatology(
                key, tuple(fi), la, lo, ti, ts_y0, ts_y1, st_lat, st_lon, agg_mode_ts
            )
            sl = df_c["snapped_lat"].iloc[0]
            slo = df_c["snapped_lon"].iloc[0]
            add_rain_trace(
                fig_ts,
                df_c["time"],
                df_c["rain"],
                label=f"{key} {agg_mode_ts} @ ({sl:.3f}°N, {slo:.3f}°E) ({ts_y0}–{ts_y1})",
                color=DATASET_COLORS.get(key, {}).get("rain", "black"),
                show_markers=True,
            )
            add_wetspell_onset_traces(fig_ts, df_c, key, key)

        add_wet_threshold_trace(fig_ts, float(wet_day_mm), clim_st["time"])

    st.plotly_chart(fig_ts, use_container_width=True)
