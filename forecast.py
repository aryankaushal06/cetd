from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import streamlit as st
import xarray as xr

from config import APP_DIR, FORECAST_FOLDERS, FORECAST_IS_ENS


@st.cache_data(show_spinner=False)
def forecast_available_years(model_key: str) -> List[int]:
    folder = APP_DIR / FORECAST_FOLDERS[model_key]
    return sorted(int(p.stem) for p in folder.glob("*.nc"))


@st.cache_resource
def _open_forecast_ds(model_key: str, year: int) -> xr.Dataset:
    path = APP_DIR / FORECAST_FOLDERS[model_key] / f"{year}.nc"
    return xr.open_dataset(str(path))


@st.cache_data(show_spinner=False)
def forecast_init_dates(model_key: str, year: int) -> List[pd.Timestamp]:
    ds = _open_forecast_ds(model_key, year)
    return [pd.Timestamp(t) for t in ds["time"].values]


@st.cache_data(show_spinner=True)
def extract_forecast_ts(
    model_key: str,
    year: int,
    init_ts: pd.Timestamp,
    lat0: float,
    lon0: float,
) -> pd.DataFrame:
    """
    Extract a point timeseries for one init date.

    Returns DataFrame with columns:
      valid_date  - absolute forecast valid date
      rain        - precipitation mm/day
      member      - ensemble member int, or NaN for deterministic
    For ensemble models one row per (valid_date, member) pair.
    """
    path = APP_DIR / FORECAST_FOLDERS[model_key] / f"{year}.nc"
    ds = xr.open_dataset(str(path))
    da = ds["tp"].sel(lat=lat0, lon=lon0, method="nearest")
    da = da.sel(time=init_ts, method="nearest").compute()

    day_vals = da["day"].values
    actual_init = pd.Timestamp(da["time"].values)
    valid_dates = [
        actual_init + pd.Timedelta(days=int(d)) for d in day_vals
    ]

    if FORECAST_IS_ENS[model_key]:
        # da dims after selection: (day, number)
        rain = da.values  # shape (n_days, n_members)
        members = da["number"].values
        n_days = len(day_vals)
        n_mem = len(members)
        # tile valid_dates n_mem times; repeat each member n_days times
        return pd.DataFrame(
            {
                "valid_date": np.tile(valid_dates, n_mem),
                "rain": rain.T.ravel().astype(float),
                "member": np.repeat(members.astype(int), n_days),
            }
        )
    else:
        return pd.DataFrame(
            {
                "valid_date": valid_dates,
                "rain": da.values.astype(float),
                "member": np.nan,
            }
        )
