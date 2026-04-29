from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import xarray as xr

from config import CHUNKSIZE, OnsetParams, RAIN_VAR, RAIN_VAR_BY_KEY
from data import normalize_lons
from detection import detect_onset


@st.cache_data(show_spinner=True)
def extract_year(
    dataset_key, ds_files, lat_name, lon_name, time_name, year, lat0, lon0
) -> pd.DataFrame:
    ds = xr.open_mfdataset(
        ds_files,
        combine="by_coords",
        parallel=True,
        chunks={time_name: CHUNKSIZE},
        engine=None,
    )
    ds = normalize_lons(ds, lon_name)
    rain_var = RAIN_VAR_BY_KEY.get(dataset_key, RAIN_VAR)
    da = ds[rain_var].sel({time_name: slice(f"{year}-01-01", f"{year}-12-31")})
    da = da.sel({lat_name: lat0, lon_name: lon0}, method="nearest").compute()
    p_lat = float(da[lat_name].values)
    p_lon = float(da[lon_name].values)
    df = (
        da.to_dataframe(name="rain")
        .reset_index()
        .rename(columns={time_name: "time"})
        .sort_values("time")
        .reset_index(drop=True)
    )
    df["time"] = pd.to_datetime(df["time"])
    df["dataset"] = dataset_key
    df["snapped_lat"] = p_lat
    df["snapped_lon"] = p_lon
    return df


@st.cache_data(show_spinner=True)
def extract_clim(
    dataset_key,
    ds_files,
    lat_name,
    lon_name,
    time_name,
    y0,
    y1,
    lat0,
    lon0,
    agg_mode,
) -> pd.DataFrame:
    ds = xr.open_mfdataset(
        ds_files,
        combine="by_coords",
        parallel=True,
        chunks={time_name: CHUNKSIZE},
        engine=None,
    )
    ds = normalize_lons(ds, lon_name)
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


@st.cache_data(show_spinner="Computing onset dates...")
def onset_series(
    dataset_key,
    ds_files,
    lat_name,
    lon_name,
    time_name,
    y0,
    y1,
    lat0,
    lon0,
    params: OnsetParams,
    bbox: Optional[Tuple[float, float, float, float]] = None,
) -> pd.DataFrame:
    """Onset DOY per year at a point (bbox=None) or pooled across a bounding box."""
    ds = xr.open_mfdataset(
        ds_files,
        combine="by_coords",
        parallel=True,
        chunks={time_name: CHUNKSIZE},
        engine=None,
    )
    ds = normalize_lons(ds, lon_name)
    rain_var = RAIN_VAR_BY_KEY.get(dataset_key, RAIN_VAR)
    records = []

    for yy in range(y0, y1 + 1):
        if bbox is not None:
            lat_min_r, lat_max_r, lon_min_r, lon_max_r = bbox
            da = (
                ds[rain_var]
                .sel(
                    {
                        lat_name: slice(lat_min_r, lat_max_r),
                        lon_name: slice(lon_min_r, lon_max_r),
                        time_name: slice(f"{yy}-01-01", f"{yy}-12-31"),
                    }
                )
                .compute()
            )
            if da.sizes[time_name] == 0:
                continue
            R = da.values
            times = pd.Series(pd.to_datetime(da[time_name].values))
            for i in range(R.shape[1]):
                for j in range(R.shape[2]):
                    rij = R[:, i, j]
                    if not np.any(np.isfinite(rij)):
                        continue
                    od, _ = detect_onset(times, rij, params)
                    if od is not None:
                        records.append(
                            {
                                "year": yy,
                                "onset_doy": float(pd.Timestamp(od).dayofyear),
                            }
                        )
        else:
            da = ds[rain_var].sel(
                {time_name: slice(f"{yy}-01-01", f"{yy}-12-31")}
            )
            da = da.sel(
                {lat_name: lat0, lon_name: lon0}, method="nearest"
            ).compute()
            r = da.values.ravel().astype(float)
            times = pd.Series(pd.to_datetime(da[time_name].values))
            if len(times) == 0:
                continue
            od, _ = detect_onset(times, r, params)
            records.append(
                {
                    "year": yy,
                    "onset_doy": (
                        float(pd.Timestamp(od).dayofyear) if od else np.nan
                    ),
                }
            )

    return pd.DataFrame(records)


@st.cache_data(show_spinner="Computing rainfall CDF...")
def rainfall_cdf(
    dataset_key, ds_files, lat_name, lon_name, time_name, y0, y1, lat0, lon0
) -> np.ndarray:
    ds = xr.open_mfdataset(
        ds_files,
        combine="by_coords",
        parallel=True,
        chunks={time_name: CHUNKSIZE},
        engine=None,
    )
    ds = normalize_lons(ds, lon_name)
    rain_var = RAIN_VAR_BY_KEY.get(dataset_key, RAIN_VAR)
    da = ds[rain_var].sel({time_name: slice(f"{y0}-01-01", f"{y1}-12-31")})
    da = da.sel({lat_name: lat0, lon_name: lon0}, method="nearest").compute()
    vals = da.values.ravel().astype(float)
    return np.sort(vals[np.isfinite(vals)])
