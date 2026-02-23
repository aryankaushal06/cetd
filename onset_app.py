from __future__ import annotations

from pathlib import Path
from typing import Tuple, Optional, List

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import streamlit as st

# Map deps
import geopandas as gpd
from shapely.geometry import Point
from shapely.prepared import prep

# Colormap utilities
from matplotlib.colors import ListedColormap, BoundaryNorm


# -----------------------------
# Helpers: coordinate detection
# -----------------------------


def _find_coord_name(ds: xr.Dataset, candidates: List[str]) -> Optional[str]:
    for name in candidates:
        if name in ds.coords:
            return name
        if name in ds.dims:
            return name
    return None


def guess_lat_lon_time_names(ds: xr.Dataset) -> Tuple[str, str, str]:
    """
    NOTE: Your current datasets use these names.
    If you later load a dataset with different names, swap these or implement true guessing.
    """
    lat_name = "latitude"
    lon_name = "longitude"
    time_name = "time"

    if lat_name is None or lon_name is None or time_name is None:
        raise ValueError(
            f"Could not find lat/lon/time coordinates. "
            f"Found coords={list(ds.coords)}, dims={list(ds.dims)}"
        )
    return lat_name, lon_name, time_name


def maybe_normalize_longitudes(ds: xr.Dataset, lon_name: str) -> xr.Dataset:
    """If lon is 0..360, convert to -180..180 for easier Ethiopia subsetting."""
    lon = ds[lon_name]
    if np.nanmax(lon.values) > 180:
        new_lon = ((lon + 180) % 360) - 180
        ds = ds.assign_coords({lon_name: new_lon}).sortby(lon_name)
    return ds


# -----------------------------
# ICPAC onset definition
# -----------------------------


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
    """
    Onset = first day t such that:
      (1) sum(r[t:t+accum_days]) >= accum_mm
      (2) within next lookahead_days, there is NO dry spell of length >= dry_spell_days,
          where dry day means rain < wet_day_mm
    """
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

    # last possible t must allow accum window + lookahead window
    last_t = n - accum_days - lookahead_days
    if last_t <= start_idx:
        return None, None

    t_stop = min(last_t, end_idx)
    if t_stop <= start_idx:
        return None, None

    for t in range(start_idx, t_stop):
        if np.nansum(r[t : t + accum_days]) >= accum_mm:
            future_wet = wet[t : t + lookahead_days]

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
    accum_days: int,
    accum_mm: float,
    start_month: int,
    start_day: int,
    end_month: int,
    end_day: int,
) -> Tuple[Optional[pd.Timestamp], Optional[int]]:
    """
    Wet spell candidate = first day t such that sum(r[t:t+accum_days]) >= accum_mm,
    within the search window. (No dry-spell constraint.)
    Returns (date, index).
    """
    r = np.asarray(rain, dtype=float)
    d = pd.to_datetime(dates).reset_index(drop=True)

    year = int(d.iloc[0].year)
    start_date = pd.Timestamp(year=year, month=start_month, day=start_day)
    end_date = pd.Timestamp(year=year, month=end_month, day=end_day)

    start_idx = int(np.searchsorted(d.values, np.datetime64(start_date)))
    end_idx = int(np.searchsorted(d.values, np.datetime64(end_date), side="right"))

    n = len(r)
    last_t = n - accum_days
    t_stop = min(last_t + 1, end_idx)  # +1 because range upper bound is exclusive

    for t in range(start_idx, max(start_idx, t_stop)):
        if np.nansum(r[t : t + accum_days]) >= accum_mm:
            return d.iloc[t], t

    return None, None


def wet_spell_start_from_idx(
    dates: pd.Series,
    rain: np.ndarray,
    idx: Optional[int],
    wet_day_mm: float,
) -> Optional[pd.Timestamp]:
    """
    Given an index (candidate), returns the start of the consecutive wet run containing idx.
    A day is wet if rain >= wet_day_mm.
    If idx itself is dry (< wet_day_mm), returns idx date.
    """
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


# -----------------------------
# Ethiopia boundary helpers
# -----------------------------


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


# -----------------------------
# JJAS DOY colormap (May-Sep)
# -----------------------------


def build_jjas_doy_cmap() -> Tuple[ListedColormap, BoundaryNorm, List[int], List[str]]:
    """
    Piecewise seasonal DOY colormap (May 1 → Sep 22) with monthly palettes
    that mimic the screenshot style.
    """

    # End at Sep 22 (matches the top label in your screenshot)
    month_doys = {
        "May": (121, 151),  # May 01 - May 31
        "Jun": (152, 181),  # Jun 01 - Jun 30
        "Jul": (182, 212),  # Jul 01 - Jul 31
        "Aug": (213, 243),  # Aug 01 - Aug 31
        "Sep": (244, 265),  # Sep 01 - Sep 22
    }

    # Palettes chosen to mimic: May orange/brown → Jun/Jul greens → Aug blues → Sep pink/purple
    month_cmaps = {
        "May": plt.cm.YlOrBr,
        "Jun": plt.cm.Greens,
        "Jul": plt.cm.GnBu,
        "Aug": plt.cm.BuPu,
        "Sep": plt.cm.RdPu,
    }

    colors = []
    bounds = []
    N_per_month = 8  # smooth bands inside each month (like your advisor snippet)

    for month in ["May", "Jun", "Jul", "Aug", "Sep"]:
        d0, d1 = month_doys[month]
        cmap = month_cmaps[month]

        ramp = cmap(np.linspace(0.35, 0.95, N_per_month))
        colors.extend(ramp)
        bounds.extend(np.linspace(d0, d1, N_per_month, endpoint=False))

    bounds.append(month_doys["Sep"][1])

    cmap_jjas = ListedColormap(colors, name="JJAS_piecewise")
    norm_jjas = BoundaryNorm(bounds, cmap_jjas.N)

    # Tick labels like the screenshot (weekly-ish markers)
    tick_positions = [
        121,
        128,
        136,
        143,  # May 01, 08, 16, 23
        152,
        159,
        166,
        173,  # Jun 01, 08, 15, 22
        182,
        189,
        197,
        204,  # Jul 01, 08, 16, 23
        213,
        220,
        228,
        235,  # Aug 01, 08, 16, 23
        244,
        251,
        258,
        265,  # Sep 01, 08, 15, 22
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


# -----------------------------
# Multi-year median maps
# -----------------------------

@st.cache_data(show_spinner=True)
def compute_median_doy_maps(
    ds_path_list: List[str],
    var: str,
    lat_name: str,
    lon_name: str,
    time_name: str,
    y0: int,
    y1: int,
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
    """
    Returns:
      lat_vals, lon_vals, median_wetspell_doy(lat,lon), median_onset_doy(lat,lon)
    DOY arrays are floats with NaN where no detection.
    """

    # Open dataset inside cache scope
    ds = xr.open_mfdataset(
        ds_path_list,
        combine="by_coords",
        parallel=True,
        chunks={"time": 365},
        engine=None,
    )
    ds = maybe_normalize_longitudes(ds, lon_name)

    # Subset time to [y0..y1]
    da_all = ds[var].sel({time_name: slice(f"{y0}-01-01", f"{y1}-12-31")})

    lat_vals = da_all[lat_name].values
    lon_vals = da_all[lon_name].values

    years = np.arange(y0, y1 + 1, dtype=int)

    # we will accumulate yearly DOY maps in a list then nanmedian
    wet_doy_yearly = []
    onset_doy_yearly = []

    # For speed: load per-year to memory (still can be big). If too slow, we can optimize further.
    for yy in years:
        da_y = da_all.sel({time_name: slice(f"{yy}-01-01", f"{yy}-12-31")}).compute()

        # shape (time, lat, lon)
        R = da_y.values
        times = pd.to_datetime(da_y[time_name].values)

        wet_map = np.full((len(lat_vals), len(lon_vals)), np.nan, dtype=float)
        onset_map = np.full((len(lat_vals), len(lon_vals)), np.nan, dtype=float)

        # loop pixels
        for i in range(R.shape[1]):
            for j in range(R.shape[2]):
                rain_ij = R[:, i, j]

                # wet spell (accum-only)
                wet_cand_date, wet_cand_idx = detect_first_wet_spell(
                    dates=pd.Series(times),
                    rain=rain_ij,
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

                # onset (accum + dry-spell constraint)
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

    wet_doy_yearly = np.stack(wet_doy_yearly, axis=0)  # (year, lat, lon)
    onset_doy_yearly = np.stack(onset_doy_yearly, axis=0)  # (year, lat, lon)

    median_wet = np.nanmedian(wet_doy_yearly, axis=0)
    median_onset = np.nanmedian(onset_doy_yearly, axis=0)
    # IMPORTANT: median produces fractional DOY -> causes odd binning with BoundaryNorm
    median_wet = np.rint(median_wet)
    median_onset = np.rint(median_onset)

    # Optional: keep only May 1 .. Sep 22 in the final median maps
    season_min, season_max = 121, 265
    median_wet = np.where((median_wet >= season_min) & (median_wet <= season_max), median_wet, np.nan)
    median_onset = np.where((median_onset >= season_min) & (median_onset <= season_max), median_onset, np.nan)

    return lat_vals, lon_vals, median_wet, median_onset


# -----------------------------
# Streamlit App
# -----------------------------

st.set_page_config(layout="wide")
st.title("Rainfall Onset Explorer (Ethiopia)")

FOLDER = "obs_subset_ethiopia"
FILE_GLOB = "*.nc"
CHUNKSIZE = 365

data_dir = Path(FOLDER).expanduser().resolve()
files = sorted(data_dir.glob(FILE_GLOB))

if len(files) == 0:
    st.error(f"No files matched {FILE_GLOB} in {data_dir}")
    st.stop()

st.caption(f"Found {len(files)} files in {data_dir.name} (chunk={CHUNKSIZE} days).")


@st.cache_resource
def open_multi_nc(files_: List[str]) -> xr.Dataset:
    ds_ = xr.open_mfdataset(
        files_,
        combine="by_coords",
        parallel=True,
        chunks={"time": CHUNKSIZE},
        engine=None,
    )
    return ds_


ds = open_multi_nc([str(f) for f in files])

# Detect coordinate names
lat_name, lon_name, time_name = guess_lat_lon_time_names(ds)

# Normalize longitudes if needed
ds = maybe_normalize_longitudes(ds, lon_name)

var = "precip"
st.caption(f"Using rainfall variable: `{var}`")

lat_min, lat_max = float(ds[lat_name].min()), float(ds[lat_name].max())
lon_min, lon_max = float(ds[lon_name].min()), float(ds[lon_name].max())

# Years list
years = ds[time_name].dt.year
year_min = (
    int(years.min().compute()) if hasattr(years.min(), "compute") else int(years.min())
)
year_max = (
    int(years.max().compute()) if hasattr(years.max(), "compute") else int(years.max())
)
year_list = list(range(year_min, year_max + 1))

# Mode selection
st.subheader("View mode")
mode = st.radio(
    "Choose a view", ["Point timeseries", "Map (Ethiopia)"], horizontal=True
)

# Point selection (for point plot + daily map)
st.subheader("Point selection")
c2, c3, c4 = st.columns([1, 1, 1])

st.caption(
    f"Available data domain: "
    f"Latitude {lat_min:.2f}° to {lat_max:.2f}°, "
    f"Longitude {lon_min:.2f}° to {lon_max:.2f}°"
)
with c2:
    lat0 = st.number_input(
        "Latitude (decimal degrees)",
        min_value=lat_min,
        max_value=lat_max,
        value=float(np.clip(10.0, lat_min, lat_max)),
        format="%.4f",
        step=0.25,
    )

with c3:
    lon0 = st.number_input(
        "Longitude (decimal degrees)",
        min_value=lon_min,
        max_value=lon_max,
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

# Map section (includes multi-year median maps)
if mode == "Map (Ethiopia)":
    st.subheader("Map view options")

    map_kind = st.radio(
        "Map type",
        [
            "Seasonal mean rainfall (JJAS) over year range",
            "Daily rainfall map (selected date)",
            "Median wet spell date (DOY) over year range",
            "Median onset date (DOY) over year range",
        ],
        horizontal=True,
    )

    eth = load_ethiopia_boundary()
    eth_geom = eth.geometry.iloc[0]
    eth_bounds = eth.total_bounds
    eth_lon_min, eth_lat_min, eth_lon_max, eth_lat_max = map(float, eth_bounds)

    # Shared year-range controls for year-range maps
    if "year range" in map_kind:
        cA, cB = st.columns([1, 1])
        with cA:
            y0 = st.selectbox("Start year", options=year_list, index=0, key="y0_map")
        with cB:
            y1 = st.selectbox(
                "End year", options=year_list, index=len(year_list) - 1, key="y1_map"
            )
        if y1 < y0:
            st.error("End year must be ≥ start year.")
            st.stop()

    if map_kind == "Seasonal mean rainfall (JJAS) over year range":
        da_season = ds[var].sel({time_name: slice(f"{y0}-01-01", f"{y1}-12-31")})
        da_season = da_season.sel(
            {
                lat_name: slice(eth_lat_min, eth_lat_max),
                lon_name: slice(eth_lon_min, eth_lon_max),
            }
        )
        da_season = da_season.where(
            da_season[time_name].dt.month.isin([6, 7, 8, 9]), drop=True
        )
        da_map = da_season.mean(dim=time_name).compute()

        lat_vals = da_map[lat_name].values
        lon_vals = da_map[lon_name].values
        mask = make_inside_mask(lat_vals, lon_vals, eth_geom)
        Z_masked = np.where(mask, da_map.values, np.nan)

        fig, ax = plt.subplots(figsize=(12, 6))
        Lon, Lat = np.meshgrid(lon_vals, lat_vals)
        pcm = ax.pcolormesh(Lon, Lat, Z_masked, shading="auto", cmap="Blues")
        eth.boundary.plot(ax=ax, linewidth=3.5, edgecolor="black", zorder=10)

        ax.set_title(f"Seasonal (JJAS) mean rainfall — {y0}–{y1}")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        cb = fig.colorbar(pcm, ax=ax)
        cb.set_label("Seasonal Mean Rainfall (mm/day)")
        st.pyplot(fig)

    elif map_kind == "Daily rainfall map (selected date)":
        tmin = pd.to_datetime(ds[time_name].min().compute().values)
        tmax = pd.to_datetime(ds[time_name].max().compute().values)

        default_date = pd.Timestamp(year=year, month=7, day=15)
        default_date = min(max(default_date, tmin), tmax)

        date_sel = st.date_input(
            "Select date for daily rainfall map",
            value=default_date.to_pydatetime().date(),
            min_value=tmin.date(),
            max_value=tmax.date(),
        )
        date_sel = pd.Timestamp(date_sel)

        da_day = ds[var].sel({time_name: date_sel}, method="nearest")
        da_day = da_day.sel(
            {
                lat_name: slice(eth_lat_min, eth_lat_max),
                lon_name: slice(eth_lon_min, eth_lon_max),
            }
        ).compute()

        lat_vals = da_day[lat_name].values
        lon_vals = da_day[lon_name].values
        mask = make_inside_mask(lat_vals, lon_vals, eth_geom)
        Z_masked = np.where(mask, da_day.values, np.nan)

        fig, ax = plt.subplots(figsize=(12, 6))
        Lon, Lat = np.meshgrid(lon_vals, lat_vals)
        pcm = ax.pcolormesh(Lon, Lat, Z_masked, shading="auto", cmap="Blues")
        eth.boundary.plot(ax=ax, linewidth=3.5, edgecolor="black", zorder=10)

        ax.set_title(
            f"Daily rainfall — {pd.to_datetime(da_day[time_name].values).date()}"
        )
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        cb = fig.colorbar(pcm, ax=ax)
        cb.set_label("Rainfall (mm/day)")
        st.pyplot(fig)

    else:
        # Median DOY maps (wet spell / onset) over year range using JJAS piecewise cmap
        cmap_jjas, norm_jjas, tick_positions, tick_labels = build_jjas_doy_cmap()

        lat_vals, lon_vals, med_wet_doy, med_onset_doy = compute_median_doy_maps(
            ds_path_list=[str(f) for f in files],
            var=var,
            lat_name=lat_name,
            lon_name=lon_name,
            time_name=time_name,
            y0=int(y0),
            y1=int(y1),
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

        mask = make_inside_mask(lat_vals, lon_vals, eth_geom)

        if map_kind == "Median wet spell date (DOY) over year range":
            Z = np.where(mask, med_wet_doy, np.nan)
            title = f"Median first wet spell date (DOY) — {y0}–{y1}"
            cb_label = "Median Wet Spell DOY"
        else:
            Z = np.where(mask, med_onset_doy, np.nan)
            title = f"Median onset date (DOY) — {y0}–{y1}"
            cb_label = "Median Onset DOY"

        fig, ax = plt.subplots(figsize=(12, 6))
        Lon, Lat = np.meshgrid(lon_vals, lat_vals)
        pcm = ax.pcolormesh(Lon, Lat, Z, shading="auto", cmap=cmap_jjas, norm=norm_jjas)
        eth.boundary.plot(ax=ax, linewidth=3.5,edgecolor="black", zorder=10)

        ax.set_title(title)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

        cb = fig.colorbar(pcm, ax=ax)
        cb.set_label(cb_label)
        cb.set_ticks(tick_positions)
        cb.set_ticklabels(tick_labels)

        st.pyplot(fig)


# -----------------------------
# Point time series (single year)
# -----------------------------

st.subheader("Point timeseries: rainfall, rolling accumulation, wet spell start, onset")


def extract_series_point_one_year() -> pd.DataFrame:
    da = ds[var].sel({time_name: slice(f"{year}-01-01", f"{year}-12-31")})
    da = da.sel({lat_name: lat0, lon_name: lon0}, method="nearest")

    p_lat = float(da[lat_name].values)
    p_lon = float(da[lon_name].values)
    label = f"Point @ ({p_lat:.3f}, {p_lon:.3f})"

    da = da.compute()

    df_ = da.to_dataframe(name="rain").reset_index()
    df_[time_name] = pd.to_datetime(df_[time_name])
    df_ = df_.sort_values(time_name).reset_index(drop=True)

    df_["label"] = label
    df_["lat"] = p_lat
    df_["lon"] = p_lon
    return df_


df = extract_series_point_one_year()

# Rolling accumulation forward-aligned to match detect_* windows:
# rolling_accum[t] = sum(r[t:t+k])
s = pd.Series(df["rain"].values, dtype="float64")
k = int(accum_days)
df["rolling_accum"] = s.rolling(window=k, min_periods=k).sum().shift(-(k - 1))

# Wet spell (accum-only)
wet_cand_date, wet_cand_idx = detect_first_wet_spell(
    dates=df[time_name],
    rain=df["rain"].values,
    accum_days=int(accum_days),
    accum_mm=float(accum_mm),
    start_month=int(start_month),
    start_day=int(start_day),
    end_month=int(end_month),
    end_day=int(end_day),
)
wet_spell_start = wet_spell_start_from_idx(
    dates=df[time_name],
    rain=df["rain"].values,
    idx=wet_cand_idx,
    wet_day_mm=float(wet_day_mm),
)

# Onset (accum + dry-spell)
onset_date, onset_idx = detect_onset(
    dates=df[time_name],
    rain=df["rain"].values,
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

df["onset_date"] = pd.NaT if onset_date is None else onset_date
df["wet_spell_start"] = pd.NaT if wet_spell_start is None else wet_spell_start
df["is_onset_day"] = (
    False if onset_date is None else (df[time_name].dt.date == onset_date.date())
)

# Plot point timeseries
fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(df[time_name], df["rain"], marker="o", linewidth=1.5, label="Daily rainfall")

ax.plot(
    df[time_name],
    df["rolling_accum"],
    linewidth=2.0,
    alpha=0.35,
    linestyle=":",
    label=f"{int(accum_days)}-day rolling sum (forward window)",
)

ax.axhline(
    float(wet_day_mm),
    linestyle="--",
    color="gray",
    linewidth=1.5,
    label=f"Wet day ≥ {float(wet_day_mm):g} mm/day",
)

if wet_spell_start is not None:
    ax.axvline(
        wet_spell_start,
        linestyle="--",
        linewidth=2.0,
        color="orange",
        label=f"Wet spell start: {wet_spell_start.date()}",
    )

if onset_date is not None:
    ax.axvline(
        onset_date,
        linewidth=2.5,
        color="red",
        label=f"Onset: {onset_date.date()}",
    )

ax.set_ylabel("Rainfall (mm/day)")
ax.set_xlabel("Date")
ax.set_title(f"Rainfall time series — {df['label'].iloc[0]} — {year}")
ax.legend()
st.pyplot(fig)

# Results
st.subheader("Detected dates (point)")

cL, _ = st.columns([1, 2])
with cL:
    if onset_date is None:
        st.warning("No onset detected with the current parameters.")
    else:
        st.success(f"Onset: {onset_date.date()}")

    if wet_spell_start is None:
        st.info("Wet spell start not available (no wet spell candidate).")
    else:
        st.info(f"Wet spell start: {wet_spell_start.date()}")

with st.expander("Show extracted time series data", expanded=False):
    show_cols = [
        time_name,
        "rain",
        "rolling_accum",
        "lat",
        "lon",
        "wet_spell_start",
        "onset_date",
        "is_onset_day",
    ]
    st.dataframe(
        df[show_cols].rename(columns={time_name: "date"}), use_container_width=True
    )
