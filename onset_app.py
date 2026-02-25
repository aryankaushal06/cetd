from __future__ import annotations

from pathlib import Path
from typing import Tuple, Optional, List, Literal, Dict

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import streamlit as st
import geopandas as gpd
from shapely.geometry import Point
from shapely.prepared import prep
from matplotlib.colors import ListedColormap, BoundaryNorm, TwoSlopeNorm


st.set_page_config(layout="wide")

LINEWIDTH_ETH = 5.0  # outline for Ethiopia boundary
DATASET_FOLDERS: Dict[str, str] = {
    "CHIRPS": "CHIRPS",
    "ENACTS": "ENACTS",
}
FILE_GLOB = "*.nc"
CHUNKSIZE = 365
RAIN_VAR = "precip"
APP_DIR = Path(__file__).resolve().parent

mask_candidates = sorted(APP_DIR.glob("*mask*ethiopia*.nc"))
st.sidebar.write("Mask candidates:", [p.name for p in mask_candidates])

DEFAULT_MASK_NC_PATH = APP_DIR / "chirps_jjas_seasonal_mask_ethiopia_0p25.nc"
st.sidebar.write(
    "Using mask:", str(DEFAULT_MASK_NC_PATH), "exists=", DEFAULT_MASK_NC_PATH.exists()
)

use_default_mask = st.sidebar.checkbox("Apply default .nc mask", value=True)


# Helpers: coordinate detection
def _find_coord_name(ds: xr.Dataset, candidates: List[str]) -> Optional[str]:
    for name in candidates:
        if name in ds.coords:
            return name
        if name in ds.dims:
            return name
    return None


def guess_lat_lon_time_names(ds: xr.Dataset) -> Tuple[str, str, str]:
    """
    Robust coordinate guessing across CHIRPS (latitude/longitude/time)
    and ENACTS (lat/lon/time).
    """
    lat_name = _find_coord_name(ds, ["latitude", "lat", "y"])
    lon_name = _find_coord_name(ds, ["longitude", "lon", "x"])
    time_name = _find_coord_name(ds, ["time", "date"])

    if lat_name is None or lon_name is None or time_name is None:
        raise ValueError(
            "Could not infer lat/lon/time coordinate names. "
            f"coords={list(ds.coords)}, dims={list(ds.dims)}"
        )
    return lat_name, lon_name, time_name


def maybe_normalize_longitudes(ds: xr.Dataset, lon_name: str) -> xr.Dataset:
    """If lon is 0..360, convert to -180..180 for easier Ethiopia subsetting."""
    lon = ds[lon_name]
    if np.nanmax(lon.values) > 180:
        new_lon = ((lon + 180) % 360) - 180
        ds = ds.assign_coords({lon_name: new_lon}).sortby(lon_name)
    return ds


@st.cache_resource
def load_default_mask_da() -> xr.DataArray:
    path = DEFAULT_MASK_NC_PATH
    if not path.exists():
        raise FileNotFoundError(f"Default mask file not found: {path}")

    ds = xr.open_dataset(str(path))
    if len(ds.data_vars) == 0:
        raise ValueError(f"No data variables found in mask file: {path}")

    mask_var = list(ds.data_vars)[0]
    return ds[mask_var]


def apply_nc_mask_to_Z(
    Z: np.ndarray,
    lat_vals: np.ndarray,
    lon_vals: np.ndarray,
    mask_da: xr.DataArray,
) -> np.ndarray:
    """
    SciPy-free nearest-neighbor remap of mask_da onto (lat_vals, lon_vals).
    Z is (lat, lon). mask_da must have lat/lon coordinates (any names).
    Assumes mask is 1/True for keep, 0/False for mask-out.
    """
    mask_ds = mask_da.to_dataset(name="mask")

    # detect coord names
    if "time" in mask_ds.coords or "time" in mask_ds.dims:
        mlat, mlon, _ = guess_lat_lon_time_names(mask_ds)
    else:
        mlat = _find_coord_name(mask_ds, ["latitude", "lat", "y"])
        mlon = _find_coord_name(mask_ds, ["longitude", "lon", "x"])

    if mlat is None or mlon is None:
        raise ValueError("Could not infer lat/lon names in mask .nc")

    mask_ds = maybe_normalize_longitudes(mask_ds, mlon)
    mlat_vals = np.asarray(mask_ds[mlat].values)
    mlon_vals = np.asarray(mask_ds[mlon].values)
    mask_grid = np.asarray(mask_ds["mask"].values)

    # If mask has time dimension, take the first timestep and squeeze
    if mask_grid.ndim == 3:
        mask_grid = mask_grid[0, :, :]

    if mask_grid.ndim != 2:
        raise ValueError(f"Expected 2D mask grid, got shape {mask_grid.shape}")

    # ensure ascending coords for searchsorted
    if mlat_vals[0] > mlat_vals[-1]:
        mlat_vals = mlat_vals[::-1]
        mask_grid = mask_grid[::-1, :]
    if mlon_vals[0] > mlon_vals[-1]:
        mlon_vals = mlon_vals[::-1]
        mask_grid = mask_grid[:, ::-1]

    def nearest_idx(src: np.ndarray, tgt: np.ndarray) -> np.ndarray:
        src = np.asarray(src)
        tgt = np.asarray(tgt)
        if len(src) < 2:
            return np.zeros_like(tgt, dtype=int)
        idx = np.searchsorted(src, tgt, side="left")
        idx = np.clip(idx, 1, len(src) - 1)
        left = src[idx - 1]
        right = src[idx]
        choose_left = (tgt - left) <= (right - tgt)
        return np.where(choose_left, idx - 1, idx)

    lat_idx = nearest_idx(mlat_vals, np.asarray(lat_vals))
    lon_idx = nearest_idx(mlon_vals, np.asarray(lon_vals))

    mask_on_grid = mask_grid[np.ix_(lat_idx, lon_idx)].astype(float)
    keep = np.isfinite(mask_on_grid) & (mask_on_grid > 0.5)
    return np.where(keep, Z, np.nan)


def dataset_time_range(
    ds: xr.Dataset, time_name: str
) -> Tuple[pd.Timestamp, pd.Timestamp]:
    tmin = pd.to_datetime(ds[time_name].min().compute().values)
    tmax = pd.to_datetime(ds[time_name].max().compute().values)
    return tmin, tmax


def dataset_year_range(ds: xr.Dataset, time_name: str) -> Tuple[int, int]:
    years = ds[time_name].dt.year
    y_min = (
        int(years.min().compute())
        if hasattr(years.min(), "compute")
        else int(years.min())
    )
    y_max = (
        int(years.max().compute())
        if hasattr(years.max(), "compute")
        else int(years.max())
    )
    return y_min, y_max

# ICPAC onset definition
def detect_onset(
    dates: pd.Series,
    rain: np.ndarray,
    wet_day_mm: float,
    accum_days: int,
    accum_mm: float,
    dry_spell_days: int,
    lookahead_days: int,
    start_month: int,
    start_day: int,
    end_month: int,
    end_day: int,
) -> Tuple[Optional[pd.Timestamp], Optional[int]]:
    r = np.asarray(rain, dtype=float)
    d = pd.to_datetime(dates).reset_index(drop=True)

    if len(r) != len(d):
        raise ValueError("dates and rain must be same length")

    wet = r >= wet_day_mm

    year = int(d.iloc[0].year)
    start_date = pd.Timestamp(year=year, month=start_month, day=start_day)
    end_date = pd.Timestamp(year=year, month=end_month, day=end_day)

    start_idx = int(np.searchsorted(d.values, np.datetime64(start_date)))
    end_idx = int(np.searchsorted(d.values, np.datetime64(end_date), side="right"))

    n = len(r)
    last_t = n - accum_days - lookahead_days
    if last_t <= start_idx:
        return None, None

    t_stop = min(last_t, end_idx)
    if t_stop <= start_idx:
        return None, None

    for t in range(start_idx, t_stop):
        window = r[t : t + accum_days]
        if np.nansum(window) >= accum_mm and np.all(window >= wet_day_mm):
            future_wet = wet[t + accum_days : t + accum_days + lookahead_days]
            dry_run = 0
            ok = True
            for is_wet in future_wet:
                if not is_wet:
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
    dates: pd.Series,
    rain: np.ndarray,
    wet_day_mm: float,
    accum_days: int,
    accum_mm: float,
    start_month: int,
    start_day: int,
    end_month: int,
    end_day: int,
) -> Tuple[Optional[pd.Timestamp], Optional[int]]:
    r = np.asarray(rain, dtype=float)
    d = pd.to_datetime(dates).reset_index(drop=True)

    year = int(d.iloc[0].year)
    start_date = pd.Timestamp(year=year, month=start_month, day=start_day)
    end_date = pd.Timestamp(year=year, month=end_month, day=end_day)

    start_idx = int(np.searchsorted(d.values, np.datetime64(start_date)))
    end_idx = int(np.searchsorted(d.values, np.datetime64(end_date), side="right"))

    n = len(r)
    last_t = n - accum_days
    t_stop = min(last_t + 1, end_idx)

    for t in range(start_idx, max(start_idx, t_stop)):
        window = r[t : t + accum_days]
        if np.nansum(window) < accum_mm:
            continue
        if np.any(window < wet_day_mm):
            continue
        return d.iloc[t], t

    return None, None


def wet_spell_start_from_idx(
    dates: pd.Series,
    rain: np.ndarray,
    idx: Optional[int],
    wet_day_mm: float,
) -> Optional[pd.Timestamp]:
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

# Ethiopia boundary helpers
@st.cache_resource
def load_ethiopia_boundary() -> gpd.GeoDataFrame:
    path = Path("data/ethiopia.geojson")
    if not path.exists():
        raise FileNotFoundError("data/ethiopia.geojson not found")
    eth = gpd.read_file(path).to_crs("EPSG:4326")
    return eth.dissolve()


def make_inside_mask(
    lat_vals: np.ndarray, lon_vals: np.ndarray, eth_geom
) -> np.ndarray:
    lat_vals = np.asarray(lat_vals)
    lon_vals = np.asarray(lon_vals)
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

# JJAS DOY colormap (May-Sep)
def build_jjas_doy_cmap() -> Tuple[ListedColormap, BoundaryNorm, List[int], List[str]]:
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

    colors = []
    bounds = []
    N_per_month = 8

    for month in ["May", "Jun", "Jul", "Aug", "Sep"]:
        d0, d1 = month_doys[month]
        cmap = month_cmaps[month]
        ramp = cmap(np.linspace(0.35, 0.95, N_per_month))
        colors.extend(ramp)
        bounds.extend(np.linspace(d0, d1, N_per_month, endpoint=False))

    bounds.append(month_doys["Sep"][1])

    cmap_jjas = ListedColormap(colors, name="JJAS_piecewise")
    norm_jjas = BoundaryNorm(bounds, cmap_jjas.N)

    tick_positions = [
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
    tick_labels = [
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
    return cmap_jjas, norm_jjas, tick_positions, tick_labels


# Year-range aggregation maps
AggMode = Literal["Median", "Mean"]

@st.cache_data(show_spinner=True)
def compute_agg_doy_maps(
    ds_path_list: List[str],
    var: str,
    lat_name: str,
    lon_name: str,
    time_name: str,
    y0: int,
    y1: int,
    agg_mode: AggMode,
    wet_day_mm: float,
    accum_days: int,
    accum_mm: float,
    dry_spell_days: int,
    lookahead_days: int,
    start_month: int,
    start_day: int,
    end_month: int,
    end_day: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ds = xr.open_mfdataset(
        ds_path_list,
        combine="by_coords",
        parallel=True,
        chunks={time_name: 365},
        engine=None,
    )
    ds = maybe_normalize_longitudes(ds, lon_name)

    da_all = ds[var].sel({time_name: slice(f"{y0}-01-01", f"{y1}-12-31")})
    lat_vals = da_all[lat_name].values
    lon_vals = da_all[lon_name].values

    years = np.arange(y0, y1 + 1, dtype=int)
    wet_doy_yearly = []
    onset_doy_yearly = []

    for yy in years:
        da_y = da_all.sel({time_name: slice(f"{yy}-01-01", f"{yy}-12-31")}).compute()
        R = da_y.values  # (time, lat, lon)
        times = pd.to_datetime(da_y[time_name].values)

        wet_map = np.full((len(lat_vals), len(lon_vals)), np.nan, dtype=float)
        onset_map = np.full((len(lat_vals), len(lon_vals)), np.nan, dtype=float)

        for i in range(R.shape[1]):
            for j in range(R.shape[2]):
                rain_ij = R[:, i, j]

                wet_cand_date, wet_cand_idx = detect_first_wet_spell(
                    dates=pd.Series(times),
                    rain=rain_ij,
                    wet_day_mm=float(wet_day_mm),
                    accum_days=int(accum_days),
                    accum_mm=float(accum_mm),
                    start_month=int(start_month),
                    start_day=int(start_day),
                    end_month=int(end_month),
                    end_day=int(end_day),
                )
                wet_start = wet_spell_start_from_idx(
                    dates=pd.Series(times),
                    rain=rain_ij,
                    idx=wet_cand_idx,
                    wet_day_mm=float(wet_day_mm),
                )
                if wet_start is not None:
                    wet_map[i, j] = float(pd.Timestamp(wet_start).dayofyear)

                onset_date, _ = detect_onset(
                    dates=pd.Series(times),
                    rain=rain_ij,
                    wet_day_mm=float(wet_day_mm),
                    accum_days=int(accum_days),
                    accum_mm=float(accum_mm),
                    dry_spell_days=int(dry_spell_days),
                    lookahead_days=int(lookahead_days),
                    start_month=int(start_month),
                    start_day=int(start_day),
                    end_month=int(end_month),
                    end_day=int(end_day),
                )
                if onset_date is not None:
                    onset_map[i, j] = float(pd.Timestamp(onset_date).dayofyear)

        wet_doy_yearly.append(wet_map)
        onset_doy_yearly.append(onset_map)

    wet_doy_yearly = np.stack(wet_doy_yearly, axis=0)
    onset_doy_yearly = np.stack(onset_doy_yearly, axis=0)

    if agg_mode == "Median":
        agg_wet = np.nanmedian(wet_doy_yearly, axis=0)
        agg_onset = np.nanmedian(onset_doy_yearly, axis=0)
    else:
        agg_wet = np.nanmean(wet_doy_yearly, axis=0)
        agg_onset = np.nanmean(onset_doy_yearly, axis=0)

    agg_wet = np.rint(agg_wet)
    agg_onset = np.rint(agg_onset)

    season_min, season_max = 121, 265
    agg_wet = np.where(
        (agg_wet >= season_min) & (agg_wet <= season_max), agg_wet, np.nan
    )
    agg_onset = np.where(
        (agg_onset >= season_min) & (agg_onset <= season_max), agg_onset, np.nan
    )

    return lat_vals, lon_vals, agg_wet, agg_onset

@st.cache_data(show_spinner=True)
def compute_single_year_doy_maps(
    ds_path_list: List[str],
    var: str,
    lat_name: str,
    lon_name: str,
    time_name: str,
    year: int,
    wet_day_mm: float,
    accum_days: int,
    accum_mm: float,
    dry_spell_days: int,
    lookahead_days: int,
    start_month: int,
    start_day: int,
    end_month: int,
    end_day: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ds = xr.open_mfdataset(
        ds_path_list,
        combine="by_coords",
        parallel=True,
        chunks={time_name: 365},
        engine=None,
    )
    ds = maybe_normalize_longitudes(ds, lon_name)

    da_y = ds[var].sel({time_name: slice(f"{year}-01-01", f"{year}-12-31")}).compute()

    lat_vals = da_y[lat_name].values
    lon_vals = da_y[lon_name].values

    R = da_y.values
    times = pd.to_datetime(da_y[time_name].values)

    wet_map = np.full((len(lat_vals), len(lon_vals)), np.nan, dtype=float)
    onset_map = np.full((len(lat_vals), len(lon_vals)), np.nan, dtype=float)

    for i in range(R.shape[1]):
        for j in range(R.shape[2]):
            rain_ij = R[:, i, j]

            wet_cand_date, wet_cand_idx = detect_first_wet_spell(
                dates=pd.Series(times),
                rain=rain_ij,
                wet_day_mm=float(wet_day_mm),
                accum_days=int(accum_days),
                accum_mm=float(accum_mm),
                start_month=int(start_month),
                start_day=int(start_day),
                end_month=int(end_month),
                end_day=int(end_day),
            )
            wet_start = wet_spell_start_from_idx(
                dates=pd.Series(times),
                rain=rain_ij,
                idx=wet_cand_idx,
                wet_day_mm=float(wet_day_mm),
            )
            if wet_start is not None:
                wet_map[i, j] = float(pd.Timestamp(wet_start).dayofyear)

            onset_date, _ = detect_onset(
                dates=pd.Series(times),
                rain=rain_ij,
                wet_day_mm=float(wet_day_mm),
                accum_days=int(accum_days),
                accum_mm=float(accum_mm),
                dry_spell_days=int(dry_spell_days),
                lookahead_days=int(lookahead_days),
                start_month=int(start_month),
                start_day=int(start_day),
                end_month=int(end_month),
                end_day=int(end_day),
            )
            if onset_date is not None:
                onset_map[i, j] = float(pd.Timestamp(onset_date).dayofyear)

    wet_map = np.rint(wet_map)
    onset_map = np.rint(onset_map)

    return lat_vals, lon_vals, wet_map, onset_map

# Streamlit
st.title("Rainfall Onset Explorer (Ethiopia)")

# Dataset selection UI
st.subheader("Dataset selection")
selected_datasets = st.multiselect(
    "Choose dataset(s) to use",
    options=list(DATASET_FOLDERS.keys()),
    default=["CHIRPS"],
)

if not selected_datasets:
    st.warning("Select at least one dataset to proceed.")
    st.stop()

# Load datasets
@st.cache_resource
def open_dataset_folder(folder: str) -> Tuple[xr.Dataset, List[str]]:
    data_dir = Path(folder).expanduser().resolve()
    files = sorted(data_dir.glob(FILE_GLOB))
    if not files:
        raise FileNotFoundError(f"No {FILE_GLOB} files matched in {data_dir}")
    ds_ = xr.open_mfdataset(
        [str(f) for f in files],
        combine="by_coords",
        parallel=True,
        chunks={"time": CHUNKSIZE},
        engine=None,
    )
    return ds_, [str(f) for f in files]


DS_BY_KEY: Dict[str, xr.Dataset] = {}
FILES_BY_KEY: Dict[str, List[str]] = {}
COORDS_BY_KEY: Dict[str, Tuple[str, str, str]] = {}

for key in selected_datasets:
    folder = DATASET_FOLDERS[key]
    ds_i, files_i = open_dataset_folder(folder)
    lat_i, lon_i, time_i = guess_lat_lon_time_names(ds_i)
    ds_i = maybe_normalize_longitudes(ds_i, lon_i)

    DS_BY_KEY[key] = ds_i
    FILES_BY_KEY[key] = files_i
    COORDS_BY_KEY[key] = (lat_i, lon_i, time_i)

st.caption("Loaded: " + ", ".join(selected_datasets))
st.caption(f"Using rainfall variable: `{RAIN_VAR}`")

# Compute shared UI ranges
year_ranges = []
time_ranges = []
lat_ranges = []
lon_ranges = []

for key in selected_datasets:
    ds_i = DS_BY_KEY[key]
    lat_i, lon_i, time_i = COORDS_BY_KEY[key]

    y_min, y_max = dataset_year_range(ds_i, time_i)
    tmin, tmax = dataset_time_range(ds_i, time_i)

    year_ranges.append((y_min, y_max))
    time_ranges.append((tmin, tmax))

    lat_ranges.append(
        (float(ds_i[lat_i].min().compute()), float(ds_i[lat_i].max().compute()))
    )
    lon_ranges.append(
        (float(ds_i[lon_i].min().compute()), float(ds_i[lon_i].max().compute()))
    )

year_min = max(r[0] for r in year_ranges)
year_max = min(r[1] for r in year_ranges)
if year_min > year_max:
    st.error("Selected datasets have no overlapping years.")
    st.stop()
year_list = list(range(year_min, year_max + 1))

tmin_common = max(r[0] for r in time_ranges)
tmax_common = min(r[1] for r in time_ranges)

lat_min = max(r[0] for r in lat_ranges)
lat_max = min(r[1] for r in lat_ranges)
lon_min = max(r[0] for r in lon_ranges)
lon_max = min(r[1] for r in lon_ranges)

st.caption(
    f"Shared domain across selection: "
    f"Lat {lat_min:.2f}° to {lat_max:.2f}°, Lon {lon_min:.2f}° to {lon_max:.2f}°; "
    f"Years {year_min}–{year_max}"
)

# View mode
st.subheader("View mode")
mode = st.radio(
    "Choose a view", ["Point timeseries", "Map (Ethiopia)"], horizontal=True
)

# Point selection
st.subheader("Point selection")
c2, c3, c4 = st.columns([1, 1, 1])

with c2:
    lat0 = st.number_input(
        "Latitude (decimal degrees)",
        min_value=float(lat_min),
        max_value=float(lat_max),
        value=float(np.clip(10.0, lat_min, lat_max)),
        format="%.4f",
        step=0.25,
    )

with c3:
    lon0 = st.number_input(
        "Longitude (decimal degrees)",
        min_value=float(lon_min),
        max_value=float(lon_max),
        value=float(np.clip(40.0, lon_min, lon_max)),
        format="%.4f",
        step=0.05,
    )

with c4:
    year = st.selectbox(
        "Year (for point timeseries + daily map)",
        options=year_list,
        index=year_list.index(min(max(2000, year_min), year_max)),
    )


# Parameters
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

st.sidebar.subheader("Search window (within selected year)")
start_month = st.sidebar.selectbox("Search start month", list(range(1, 13)), index=4)
start_day = st.sidebar.number_input(
    "Search start day", min_value=1, max_value=31, value=15, step=1
)
end_month = st.sidebar.selectbox("Search end month", list(range(1, 13)), index=8)
end_day = st.sidebar.number_input(
    "Search end day", min_value=1, max_value=31, value=30, step=1
)

# Cached point-extraction helpers
@st.cache_data(show_spinner=True)
def extract_series_point_one_year(
    dataset_key: str, sel_year: int, lat0_: float, lon0_: float
) -> pd.DataFrame:
    ds_i = DS_BY_KEY[dataset_key]
    lat_i, lon_i, time_i = COORDS_BY_KEY[dataset_key]

    da = ds_i[RAIN_VAR].sel({time_i: slice(f"{sel_year}-01-01", f"{sel_year}-12-31")})
    da = da.sel({lat_i: lat0_, lon_i: lon0_}, method="nearest").compute()

    p_lat = float(da[lat_i].values)
    p_lon = float(da[lon_i].values)
    label = f"{dataset_key} @ ({p_lat:.3f}, {p_lon:.3f})"

    df_ = da.to_dataframe(name="rain").reset_index()
    df_[time_i] = pd.to_datetime(df_[time_i])
    df_ = df_.sort_values(time_i).reset_index(drop=True)

    df_ = df_.rename(columns={time_i: "time"})
    df_["dataset"] = dataset_key
    df_["label"] = label
    df_["lat"] = p_lat
    df_["lon"] = p_lon
    return df_


@st.cache_data(show_spinner=True)
def extract_point_daily_climatology(
    dataset_key: str, y0_: int, y1_: int, lat0_: float, lon0_: float
) -> pd.DataFrame:
    ds_i = DS_BY_KEY[dataset_key]
    lat_i, lon_i, time_i = COORDS_BY_KEY[dataset_key]

    da = ds_i[RAIN_VAR].sel({time_i: slice(f"{y0_}-01-01", f"{y1_}-12-31")})
    da = da.sel({lat_i: lat0_, lon_i: lon0_}, method="nearest").compute()

    p_lat = float(da[lat_i].values)
    p_lon = float(da[lon_i].values)
    label = f"{dataset_key} @ ({p_lat:.3f}, {p_lon:.3f})"

    doy = da[time_i].dt.dayofyear
    clim = da.groupby(doy).mean(dim=time_i, skipna=True)

    doy_vals = clim[doy.name].values.astype(int)
    rain_vals = clim.values

    keep = doy_vals <= 365
    doy_vals = doy_vals[keep]
    rain_vals = rain_vals[keep]

    dates = pd.to_datetime("2001-01-01") + pd.to_timedelta(doy_vals - 1, unit="D")

    return pd.DataFrame(
        {
            "time": dates,
            "rain": rain_vals,
            "dataset": dataset_key,
            "label": label,
            "lat": p_lat,
            "lon": p_lon,
        }
    )

# Map view helpers
def _ethiopia_clip_and_mask(
    da: xr.DataArray,
    lat_name: str,
    lon_name: str,
    eth_geom,
    eth_bounds: Tuple[float, float, float, float],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    eth_lon_min, eth_lat_min, eth_lon_max, eth_lat_max = eth_bounds

    da = da.sel(
        {
            lat_name: slice(eth_lat_min, eth_lat_max),
            lon_name: slice(eth_lon_min, eth_lon_max),
        }
    )

    lat_vals = da[lat_name].values
    lon_vals = da[lon_name].values

    mask = make_inside_mask(lat_vals, lon_vals, eth_geom)
    Z = np.where(mask, da.values, np.nan)
    return lat_vals, lon_vals, Z


def compute_map_Z_for_dataset(
    dataset_key: str,
    map_kind: str,
    year_mode: str,
    y0: int,
    y1: int,
    agg_mode: AggMode,
    date_sel: Optional[pd.Timestamp],
    eth_geom,
    eth_bounds: Tuple[float, float, float, float],
    use_default_mask_flag: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str, str, str]:
    """
    Returns:
      lat_vals, lon_vals, Z, title, cb_label, kind_tag
    kind_tag in {"rain", "doy"}
    """
    ds_i = DS_BY_KEY[dataset_key]
    files_i = FILES_BY_KEY[dataset_key]
    lat_i, lon_i, time_i = COORDS_BY_KEY[dataset_key]

    mask_da = None
    if use_default_mask_flag:
        mask_da = load_default_mask_da()

    if map_kind == "Seasonal mean rainfall (JJAS)":
        da_season = ds_i[RAIN_VAR].sel({time_i: slice(f"{y0}-01-01", f"{y1}-12-31")})
        da_season = da_season.where(
            da_season[time_i].dt.month.isin([6, 7, 8, 9]), drop=True
        )
        da_map = da_season.mean(dim=time_i).compute()

        lat_vals, lon_vals, Z = _ethiopia_clip_and_mask(
            da_map, lat_i, lon_i, eth_geom, eth_bounds
        )

        if use_default_mask_flag and mask_da is not None:
            Z = apply_nc_mask_to_Z(Z, lat_vals, lon_vals, mask_da)

        title = (
            f"{dataset_key} — Seasonal (JJAS) mean rainfall — {y0}"
            if y0 == y1
            else f"{dataset_key} — Seasonal (JJAS) mean rainfall — {y0}–{y1}"
        )
        return lat_vals, lon_vals, Z, title, "Rainfall (mm/day)", "rain"

    if map_kind == "Daily rainfall map (selected date)":
        if date_sel is None:
            raise ValueError("date_sel is required for daily rainfall map")

        da_day = ds_i[RAIN_VAR].sel({time_i: date_sel}, method="nearest").compute()
        lat_vals, lon_vals, Z = _ethiopia_clip_and_mask(
            da_day, lat_i, lon_i, eth_geom, eth_bounds
        )

        if use_default_mask_flag and mask_da is not None:
            Z = apply_nc_mask_to_Z(Z, lat_vals, lon_vals, mask_da)

        shown_date = pd.to_datetime(da_day[time_i].values).date()
        title = f"{dataset_key} — Daily rainfall — {shown_date}"
        return lat_vals, lon_vals, Z, title, "Rainfall (mm/day)", "rain"

    # ---- DOY maps ----
    if year_mode == "Single year":
        lat_vals, lon_vals, wet_doy, onset_doy = compute_single_year_doy_maps(
            ds_path_list=files_i,
            var=RAIN_VAR,
            lat_name=lat_i,
            lon_name=lon_i,
            time_name=time_i,
            year=int(y0),
            wet_day_mm=float(wet_day_mm),
            accum_days=int(accum_days),
            accum_mm=float(accum_mm),
            dry_spell_days=int(dry_spell_days),
            lookahead_days=int(lookahead_days),
            start_month=int(start_month),
            start_day=int(start_day),
            end_month=int(end_month),
            end_day=int(end_day),
        )
        Z0 = wet_doy if map_kind == "Wet spell date (DOY)" else onset_doy
        title = (
            f"{dataset_key} — Wet spell date (DOY) — {y0}"
            if map_kind == "Wet spell date (DOY)"
            else f"{dataset_key} — Onset date (DOY) — {y0}"
        )
        cb_label = (
            "Wet Spell DOY" if map_kind == "Wet spell date (DOY)" else "Onset DOY"
        )
    else:
        lat_vals, lon_vals, agg_wet_doy, agg_onset_doy = compute_agg_doy_maps(
            ds_path_list=files_i,
            var=RAIN_VAR,
            lat_name=lat_i,
            lon_name=lon_i,
            time_name=time_i,
            y0=int(y0),
            y1=int(y1),
            agg_mode=agg_mode,
            wet_day_mm=float(wet_day_mm),
            accum_days=int(accum_days),
            accum_mm=float(accum_mm),
            dry_spell_days=int(dry_spell_days),
            lookahead_days=int(lookahead_days),
            start_month=int(start_month),
            start_day=int(start_day),
            end_month=int(end_month),
            end_day=int(end_day),
        )
        Z0 = agg_wet_doy if map_kind == "Wet spell date (DOY)" else agg_onset_doy
        title = (
            f"{dataset_key} — {agg_mode} wet spell date (DOY) — {y0}–{y1}"
            if map_kind == "Wet spell date (DOY)"
            else f"{dataset_key} — {agg_mode} onset date (DOY) — {y0}–{y1}"
        )
        cb_label = (
            f"{agg_mode} Wet Spell DOY"
            if map_kind == "Wet spell date (DOY)"
            else f"{agg_mode} Onset DOY"
        )

    # Ethiopia polygon mask
    mask = make_inside_mask(lat_vals, lon_vals, eth_geom)
    Z = np.where(mask, Z0, np.nan)

    if use_default_mask_flag and mask_da is not None:
        Z = apply_nc_mask_to_Z(Z, lat_vals, lon_vals, mask_da)

    return lat_vals, lon_vals, Z, title, cb_label, "doy"


def _nanminmax(a: np.ndarray) -> Tuple[float, float]:
    vmin = float(np.nanmin(a)) if np.isfinite(np.nanmin(a)) else 0.0
    vmax = float(np.nanmax(a)) if np.isfinite(np.nanmax(a)) else 1.0
    if vmin == vmax:
        vmax = vmin + 1e-6
    return vmin, vmax

# Map view
if mode == "Map (Ethiopia)":
    st.subheader("Map view options")

    year_mode = st.radio(
        "Year selection",
        ["Single year", "Year range"],
        horizontal=True,
        key="year_mode",
    )

    if year_mode == "Year range":
        agg_mode: AggMode = st.radio(
            "Across-years aggregation (DOY maps)",
            ["Median", "Mean"],
            horizontal=True,
            key="agg_mode",
        )
    else:
        agg_mode = "Median"

    map_kind = st.radio(
        "Map type",
        [
            "Seasonal mean rainfall (JJAS)",
            "Daily rainfall map (selected date)",
            "Wet spell date (DOY)",
            "Onset date (DOY)",
        ],
        horizontal=True,
        key="map_kind",
    )

    if year_mode == "Single year":
        y0 = y1 = int(
            st.selectbox(
                "Year",
                options=year_list,
                index=year_list.index(int(year)),
                key="map_year_single",
            )
        )
    else:
        cA, cB = st.columns([1, 1])
        with cA:
            y0 = int(
                st.selectbox(
                    "Start year", options=year_list, index=0, key="map_year_start"
                )
            )
        with cB:
            y1 = int(
                st.selectbox(
                    "End year",
                    options=year_list,
                    index=len(year_list) - 1,
                    key="map_year_end",
                )
            )
        if y1 < y0:
            st.error("End year must be ≥ start year.")
            st.stop()

    eth = load_ethiopia_boundary()
    eth_geom = eth.geometry.iloc[0]
    eth_bounds = tuple(map(float, eth.total_bounds))  # (xmin, ymin, xmax, ymax)

    date_sel = None
    if map_kind == "Daily rainfall map (selected date)":
        tmin_ui, tmax_ui = tmin_common, tmax_common
        default_date = pd.Timestamp(year=int(year), month=7, day=15)
        default_date = min(max(default_date, tmin_ui), tmax_ui)

        date_in = st.date_input(
            "Select date for daily rainfall map",
            value=default_date.to_pydatetime().date(),
            min_value=tmin_ui.date(),
            max_value=tmax_ui.date(),
        )
        date_sel = pd.Timestamp(date_in)

    if len(selected_datasets) == 1:
        key = selected_datasets[0]
        lat_vals, lon_vals, Z, title, cb_label, kind_tag = compute_map_Z_for_dataset(
            dataset_key=key,
            map_kind=map_kind,
            year_mode=year_mode,
            y0=int(y0),
            y1=int(y1),
            agg_mode=agg_mode,
            date_sel=date_sel,
            eth_geom=eth_geom,
            eth_bounds=eth_bounds,
            use_default_mask_flag=use_default_mask,
        )

        fig, ax = plt.subplots(figsize=(12, 6))
        Lon, Lat = np.meshgrid(lon_vals, lat_vals)

        if kind_tag == "rain":
            vmin, vmax = _nanminmax(Z)
            pcm = ax.pcolormesh(
                Lon, Lat, Z, shading="auto", cmap="Blues", vmin=vmin, vmax=vmax
            )
        else:
            cmap_jjas, norm_jjas, tick_positions, tick_labels = build_jjas_doy_cmap()
            pcm = ax.pcolormesh(
                Lon, Lat, Z, shading="auto", cmap=cmap_jjas, norm=norm_jjas
            )

        eth.boundary.plot(ax=ax, linewidth=LINEWIDTH_ETH, edgecolor="black", zorder=10)
        ax.set_title(title)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

        cb = fig.colorbar(pcm, ax=ax)
        cb.set_label(cb_label)
        if kind_tag == "doy":
            cb.set_ticks(tick_positions)
            cb.set_ticklabels(tick_labels)

        st.pyplot(fig)
        st.stop()

    # Two datasets: CHIRPS, ENACTS, DIFF
    left_key, right_key = "CHIRPS", "ENACTS"
    if left_key not in selected_datasets or right_key not in selected_datasets:
        left_key, right_key = selected_datasets[0], selected_datasets[1]

    lat1, lon1, Z1, title1, cb1, kind1 = compute_map_Z_for_dataset(
        dataset_key=left_key,
        map_kind=map_kind,
        year_mode=year_mode,
        y0=int(y0),
        y1=int(y1),
        agg_mode=agg_mode,
        date_sel=date_sel,
        eth_geom=eth_geom,
        eth_bounds=eth_bounds,
        use_default_mask_flag=use_default_mask,
    )
    lat2, lon2, Z2, title2, cb2, kind2 = compute_map_Z_for_dataset(
        dataset_key=right_key,
        map_kind=map_kind,
        year_mode=year_mode,
        y0=int(y0),
        y1=int(y1),
        agg_mode=agg_mode,
        date_sel=date_sel,
        eth_geom=eth_geom,
        eth_bounds=eth_bounds,
        use_default_mask_flag=use_default_mask,
    )

    if Z1.shape != Z2.shape:
        st.error(
            "CHIRPS and ENACTS map grids do not match after clipping. Need regridding to compute difference."
        )
        st.stop()

    Zdiff = Z1 - Z2

    if kind1 != kind2:
        st.error("Internal error: map kind mismatch between datasets.")
        st.stop()

    if kind1 == "rain":
        vmin_12 = float(np.nanmin([np.nanmin(Z1), np.nanmin(Z2)]))
        vmax_12 = float(np.nanmax([np.nanmax(Z1), np.nanmax(Z2)]))
        if not np.isfinite(vmin_12) or not np.isfinite(vmax_12) or vmin_12 == vmax_12:
            vmin_12, vmax_12 = 0.0, 1.0

        max_abs = float(np.nanmax(np.abs(Zdiff)))
        if not np.isfinite(max_abs) or max_abs == 0:
            max_abs = 1e-6
        diff_norm = TwoSlopeNorm(vcenter=0.0, vmin=-max_abs, vmax=max_abs)

        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown(f"### {left_key}")
            fig, ax = plt.subplots(figsize=(5.2, 4.2))
            Lon, Lat = np.meshgrid(lon1, lat1)
            pcm = ax.pcolormesh(
                Lon, Lat, Z1, shading="auto", cmap="Blues", vmin=vmin_12, vmax=vmax_12
            )
            eth.boundary.plot(ax=ax, linewidth=LINEWIDTH_ETH, edgecolor="black", zorder=10)
            ax.set_title(title1)
            ax.set_xlabel("Lon")
            ax.set_ylabel("Lat")
            cb = fig.colorbar(pcm, ax=ax)
            cb.set_label(cb1)
            st.pyplot(fig)

        with c2:
            st.markdown(f"### {right_key}")
            fig, ax = plt.subplots(figsize=(5.2, 4.2))
            Lon, Lat = np.meshgrid(lon2, lat2)
            pcm = ax.pcolormesh(
                Lon, Lat, Z2, shading="auto", cmap="Blues", vmin=vmin_12, vmax=vmax_12
            )
            eth.boundary.plot(ax=ax, linewidth=LINEWIDTH_ETH, edgecolor="black", zorder=10)
            ax.set_title(title2)
            ax.set_xlabel("Lon")
            ax.set_ylabel("Lat")
            cb = fig.colorbar(pcm, ax=ax)
            cb.set_label(cb2)
            st.pyplot(fig)

        with c3:
            st.markdown(f"### {left_key} − {right_key}")
            fig, ax = plt.subplots(figsize=(5.2, 4.2))
            Lon, Lat = np.meshgrid(lon1, lat1)
            pcm = ax.pcolormesh(
                Lon, Lat, Zdiff, shading="auto", cmap="RdBu_r", norm=diff_norm
            )
            eth.boundary.plot(ax=ax, linewidth=LINEWIDTH_ETH, edgecolor="black", zorder=10)
            ax.set_title(f"Difference — {map_kind}")
            ax.set_xlabel("Lon")
            ax.set_ylabel("Lat")
            cb = fig.colorbar(pcm, ax=ax)
            cb.set_label(f"{cb1} ({left_key} − {right_key})")
            st.pyplot(fig)

    else:
        cmap_jjas, norm_jjas, tick_positions, tick_labels = build_jjas_doy_cmap()

        c1, c2, c3 = st.columns(3)

        # --- LEFT DOY map ---
        with c1:
            st.markdown(f"### {left_key}")
            fig, ax = plt.subplots(figsize=(5.2, 4.2))
            Lon, Lat = np.meshgrid(lon1, lat1)
            pcm = ax.pcolormesh(
                Lon, Lat, Z1, shading="auto", cmap=cmap_jjas, norm=norm_jjas
            )
            eth.boundary.plot(ax=ax, linewidth=LINEWIDTH_ETH, edgecolor="black", zorder=10)
            ax.set_title(title1)
            ax.set_xlabel("Lon")
            ax.set_ylabel("Lat")
            cb = fig.colorbar(pcm, ax=ax)
            cb.set_label(cb1)
            cb.set_ticks(tick_positions)
            cb.set_ticklabels(tick_labels)
            st.pyplot(fig)

        # --- RIGHT DOY map ---
        with c2:
            st.markdown(f"### {right_key}")
            fig, ax = plt.subplots(figsize=(5.2, 4.2))
            Lon, Lat = np.meshgrid(lon2, lat2)
            pcm = ax.pcolormesh(
                Lon, Lat, Z2, shading="auto", cmap=cmap_jjas, norm=norm_jjas
            )
            eth.boundary.plot(ax=ax, linewidth=LINEWIDTH_ETH, edgecolor="black", zorder=10)
            ax.set_title(title2)
            ax.set_xlabel("Lon")
            ax.set_ylabel("Lat")
            cb = fig.colorbar(pcm, ax=ax)
            cb.set_label(cb2)
            cb.set_ticks(tick_positions)
            cb.set_ticklabels(tick_labels)
            st.pyplot(fig)

        # --- DIFFERENCE MAP: DISCRETE -30..30 DOY ---
        with c3:
            st.markdown(f"### {left_key} − {right_key}")
            fig, ax = plt.subplots(figsize=(5.2, 4.2))
            Lon, Lat = np.meshgrid(lon1, lat1)

            levels = np.arange(-30, 31, 5)  # -30, -25, ..., 30
            cmap_diff = plt.cm.get_cmap("RdBu_r", len(levels) - 1)
            norm_diff = BoundaryNorm(levels, cmap_diff.N)

            Zdiff_plot = np.clip(Zdiff, -30, 30)

            pcm = ax.pcolormesh(
                Lon, Lat, Zdiff_plot, shading="auto", cmap=cmap_diff, norm=norm_diff
            )
            eth.boundary.plot(ax=ax, linewidth=LINEWIDTH_ETH, edgecolor="black", zorder=10)
            ax.set_title(f"Difference — {map_kind}")
            ax.set_xlabel("Lon")
            ax.set_ylabel("Lat")

            cb = fig.colorbar(pcm, ax=ax, ticks=levels)
            cb.set_label(f"DOY difference ({left_key} − {right_key})")
            st.pyplot(fig)

# Point time series view
st.subheader("Point timeseries: rainfall, wet spell start, onset")

show_ts = True
if mode == "Map (Ethiopia)":
    show_ts = st.checkbox("Show point time series", value=False, key="show_ts_map")

if show_ts:
    if mode == "Map (Ethiopia)":
        ts_y0, ts_y1 = int(y0), int(y1)
        ts_year_mode = year_mode
    else:
        ts_y0 = ts_y1 = int(year)
        ts_year_mode = "Single year"

    fig, ax = plt.subplots(figsize=(14, 6))
    df_for_table: Dict[str, pd.DataFrame] = {}

    for key in selected_datasets:
        if ts_year_mode == "Single year":
            df_ts = extract_series_point_one_year(key, ts_y0, float(lat0), float(lon0))

            wet_cand_date, wet_cand_idx = detect_first_wet_spell(
                dates=df_ts["time"],
                rain=df_ts["rain"].values,
                wet_day_mm=float(wet_day_mm),
                accum_days=int(accum_days),
                accum_mm=float(accum_mm),
                start_month=int(start_month),
                start_day=int(start_day),
                end_month=int(end_month),
                end_day=int(end_day),
            )
            wet_spell_start = wet_spell_start_from_idx(
                dates=df_ts["time"],
                rain=df_ts["rain"].values,
                idx=wet_cand_idx,
                wet_day_mm=float(wet_day_mm),
            )

            onset_date, _ = detect_onset(
                dates=df_ts["time"],
                rain=df_ts["rain"].values,
                wet_day_mm=float(wet_day_mm),
                accum_days=int(accum_days),
                accum_mm=float(accum_mm),
                dry_spell_days=int(dry_spell_days),
                lookahead_days=int(lookahead_days),
                start_month=int(start_month),
                start_day=int(start_day),
                end_month=int(end_month),
                end_day=int(end_day),
            )

            if key.upper() == "CHIRPS":
                rain_color, wet_color, onset_color = "blue", "purple", "pink"
            elif key.upper() == "ENACTS":
                rain_color, wet_color, onset_color = "orange", "red", "yellow"
            else:
                rain_color, wet_color, onset_color = "black", "gray", "green"

            ax.plot(
                df_ts["time"],
                df_ts["rain"],
                color=rain_color,
                marker="o",
                markersize=3,
                linewidth=1.2,
                label=f"{key} daily rainfall",
            )

            if wet_spell_start is not None:
                ax.axvline(
                    wet_spell_start,
                    color=wet_color,
                    linestyle="--",
                    linewidth=2.0,
                    label=f"{key} wet spell start: {wet_spell_start.date()}",
                )

            if onset_date is not None:
                ax.axvline(
                    onset_date,
                    color=onset_color,
                    linestyle="-",
                    linewidth=2.5,
                    label=f"{key} onset: {onset_date.date()}",
                )

            df_for_table[key] = df_ts[["time", "rain", "lat", "lon"]].rename(
                columns={"time": "date"}
            )

        else:
            df_ts = extract_point_daily_climatology(
                key, ts_y0, ts_y1, float(lat0), float(lon0)
            )

            rain_color = (
                "blue"
                if key.upper() == "CHIRPS"
                else ("orange" if key.upper() == "ENACTS" else "black")
            )

            ax.plot(
                df_ts["time"],
                df_ts["rain"],
                color=rain_color,
                marker="o",
                markersize=3,
                linewidth=1.2,
                label=f"{key} climatology (mean by DOY)",
            )

            df_for_table[key] = df_ts[["time", "rain", "lat", "lon"]].rename(
                columns={"time": "date"}
            )

    handles, labels = ax.get_legend_handles_labels()

    def _legend_rank(lbl: str) -> int:
        s = lbl.lower()
        d = 0 if "chirps" in s else (1 if "enacts" in s else 9)
        if "daily rainfall" in s or "climatology" in s:
            k = 0
        elif "wet spell start" in s:
            k = 1
        elif "onset" in s:
            k = 2
        elif "wet day" in s:
            k = 3
        else:
            k = 8
        return d * 10 + k

    order = sorted(range(len(labels)), key=lambda i: _legend_rank(labels[i]))
    handles = [handles[i] for i in order]
    labels = [labels[i] for i in order]

    ax.legend(handles, labels, loc="upper right")
    ax.axhline(
        float(wet_day_mm),
        linestyle="--",
        color="gray",
        linewidth=1.5,
        label=f"Wet day ≥ {float(wet_day_mm):g} mm/day",
    )

    title_suffix = (
        f"{ts_y0}"
        if ts_year_mode == "Single year"
        else f"{ts_y0}–{ts_y1} (daily climatology)"
    )
    ax.set_ylabel("Rainfall (mm/day)")
    ax.set_xlabel("Date")
    ax.set_title(f"Rainfall time series — ({lat0:.3f}, {lon0:.3f}) — {title_suffix}")
    ax.legend()
    st.pyplot(fig)

    with st.expander("Show extracted time series data", expanded=False):
        if len(selected_datasets) == 1:
            key = selected_datasets[0]
            st.dataframe(df_for_table[key], use_container_width=True)
        else:
            cL, cR = st.columns([1, 1])
            with cL:
                st.markdown(f"#### {selected_datasets[0]}")
                st.dataframe(
                    df_for_table[selected_datasets[0]], use_container_width=True
                )
            with cR:
                st.markdown(f"#### {selected_datasets[1]}")
                st.dataframe(
                    df_for_table[selected_datasets[1]], use_container_width=True
                )
