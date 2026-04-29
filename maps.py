from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
import streamlit as st
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap

from config import (
    CHUNKSIZE,
    LINEWIDTH_ETH,
    OnsetParams,
    RAIN_VAR,
    RAIN_VAR_BY_KEY,
)
from data import apply_mask, inside_mask, normalize_lons
from detection import detect_onset, detect_wet_spell, wet_spell_start


def _pixel_maps(
    R: np.ndarray, times: pd.DatetimeIndex, params: OnsetParams
) -> Tuple[np.ndarray, np.ndarray]:
    """Run onset + wet-spell detection at every (lat, lon) pixel. Returns (wm, om) DOY arrays."""
    ts = pd.Series(times)
    wm = np.full((R.shape[1], R.shape[2]), np.nan)
    om = np.full((R.shape[1], R.shape[2]), np.nan)
    for i in range(R.shape[1]):
        for j in range(R.shape[2]):
            rij = R[:, i, j]
            _, wi, si_ws = detect_wet_spell(ts, rij, params)
            ws = wet_spell_start(ts, rij, wi, params.wet_day_mm, si_ws)
            if ws is not None:
                wm[i, j] = float(pd.Timestamp(ws).dayofyear)
            od, _ = detect_onset(ts, rij, params)
            if od is not None:
                om[i, j] = float(pd.Timestamp(od).dayofyear)
    return wm, om


@st.cache_data(show_spinner=True)
def doy_maps_year(
    ds_files, var, lat_name, lon_name, time_name, year, params: OnsetParams
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ds = xr.open_mfdataset(
        ds_files,
        combine="by_coords",
        parallel=True,
        chunks={time_name: CHUNKSIZE},
        engine=None,
    )
    ds = normalize_lons(ds, lon_name)
    da = (
        ds[var]
        .sel({time_name: slice(f"{year}-01-01", f"{year}-12-31")})
        .compute()
    )
    da = da.drop_duplicates(lat_name).drop_duplicates(lon_name)
    times = pd.to_datetime(da[time_name].values)
    wm, om = _pixel_maps(da.values, times, params)
    return da[lat_name].values, da[lon_name].values, np.rint(wm), np.rint(om)


@st.cache_data(show_spinner=True)
def doy_maps_agg(
    ds_files,
    var,
    lat_name,
    lon_name,
    time_name,
    y0,
    y1,
    agg_mode,
    params: OnsetParams,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ds = xr.open_mfdataset(
        ds_files,
        combine="by_coords",
        parallel=True,
        chunks={time_name: CHUNKSIZE},
        engine=None,
    )
    ds = normalize_lons(ds, lon_name)
    da_all = (
        ds[var]
        .sel({time_name: slice(f"{y0}-01-01", f"{y1}-12-31")})
        .drop_duplicates(lat_name)
        .drop_duplicates(lon_name)
    )
    lv = da_all[lat_name].values
    lov = da_all[lon_name].values
    wl, ol = [], []
    for yy in range(y0, y1 + 1):
        da = da_all.sel(
            {time_name: slice(f"{yy}-01-01", f"{yy}-12-31")}
        ).compute()
        wm, om = _pixel_maps(
            da.values, pd.to_datetime(da[time_name].values), params
        )
        wl.append(wm)
        ol.append(om)
    ws_stk = np.stack(wl, axis=0)
    os_stk = np.stack(ol, axis=0)
    fn = np.nanmedian if agg_mode == "Median" else np.nanmean
    aw, ao = fn(ws_stk, axis=0), fn(os_stk, axis=0)
    s0, s1 = 121, 265
    aw = np.where(
        (np.rint(aw) >= s0) & (np.rint(aw) <= s1), np.rint(aw), np.nan
    )
    ao = np.where(
        (np.rint(ao) >= s0) & (np.rint(ao) <= s1), np.rint(ao), np.nan
    )
    return lv, lov, aw, ao


def clip_map(Z, lat_vals, lon_vals, eth_geom, eth_bounds, mask_da):
    lo_mn, la_mn, lo_mx, la_mx = eth_bounds
    lat_idx = np.where((lat_vals >= la_mn) & (lat_vals <= la_mx))[0]
    lon_idx = np.where((lon_vals >= lo_mn) & (lon_vals <= lo_mx))[0]
    lat_idx = lat_idx[lat_idx < Z.shape[0]]
    lon_idx = lon_idx[lon_idx < Z.shape[1]]
    Z = Z[np.ix_(lat_idx, lon_idx)]
    lat_vals = lat_vals[lat_idx]
    lon_vals = lon_vals[lon_idx]
    Z = np.where(inside_mask(lat_vals, lon_vals, eth_geom), Z, np.nan)
    if mask_da is not None:
        Z = apply_mask(Z, lat_vals, lon_vals, mask_da)
    return lat_vals, lon_vals, Z


def snap_to_ref(lv, lov, Z, ref_lv, ref_lov):
    """Bilinearly interpolate Z onto a reference grid if shapes differ."""
    if (
        lv.shape == ref_lv.shape
        and lov.shape == ref_lov.shape
        and np.allclose(lv, ref_lv, atol=1e-4)
        and np.allclose(lov, ref_lov, atol=1e-4)
    ):
        return ref_lv, ref_lov, Z
    from scipy.interpolate import RegularGridInterpolator

    interp = RegularGridInterpolator(
        (lv, lov), Z, method="linear", bounds_error=False, fill_value=np.nan
    )
    LonG, LatG = np.meshgrid(ref_lov, ref_lv)
    Z_new = interp(np.stack([LatG.ravel(), LonG.ravel()], axis=1)).reshape(
        LonG.shape
    )
    return ref_lv, ref_lov, Z_new


def nanminmax(a):
    fin = a[np.isfinite(a)]
    if len(fin) == 0:
        return 0.0, 1.0
    lo, hi = float(fin.min()), float(fin.max())
    return (lo, hi) if lo != hi else (lo, lo + 1e-6)


def jjas_cmap():
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


def plot_doy_map(
    ax, Lon, Lat, Z, eth, title, cb_label, cmap_jjas, norm_jjas, tpos, tlbl, fig
):
    pcm = ax.pcolormesh(
        Lon, Lat, Z, shading="auto", cmap=cmap_jjas, norm=norm_jjas
    )
    eth.boundary.plot(
        ax=ax, linewidth=LINEWIDTH_ETH, edgecolor="black", zorder=10
    )
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("Lon")
    ax.set_ylabel("Lat")
    cb = fig.colorbar(pcm, ax=ax)
    cb.set_label(cb_label, fontsize=8)
    cb.set_ticks(tpos)
    cb.set_ticklabels(tlbl, fontsize=6)


def plot_rain_map(
    ax,
    Lon,
    Lat,
    Z,
    eth,
    title,
    cb_label,
    fig,
    vmin=None,
    vmax=None,
    cmap="Blues",
):
    if vmin is None or vmax is None:
        vmin, vmax = nanminmax(Z)
    pcm = ax.pcolormesh(
        Lon, Lat, Z, shading="auto", cmap=cmap, vmin=vmin, vmax=vmax
    )
    eth.boundary.plot(
        ax=ax, linewidth=LINEWIDTH_ETH, edgecolor="black", zorder=10
    )
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("Lon")
    ax.set_ylabel("Lat")
    cb = fig.colorbar(pcm, ax=ax)
    cb.set_label(cb_label, fontsize=8)
