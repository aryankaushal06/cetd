from __future__ import annotations

from itertools import combinations
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from matplotlib.colors import BoundaryNorm, TwoSlopeNorm

from charts import (
    add_ensemble_fan,
    add_ensemble_members,
    add_ensemble_onset_markers,
    add_markers,
    add_rain,
    add_threshold,
    doy_to_label,
    station_map,
    ts_figure,
    ts_xaxis_range,
)
from config import (
    DATASET_COLORS,
    DATASET_FOLDERS,
    FORECAST_COLORS,
    FORECAST_IS_ENS,
    OnsetParams,
    RAIN_VAR,
    RAIN_VAR_BY_KEY,
    SEASONAL_COVERAGE_NOTE,
    STATION_CSV_PATH,
    STATION_KEY,
)
from data import (
    dataset_time_range,
    dataset_year_range,
    infer_coords,
    load_boundary,
    load_emi_csv,
    load_mask,
    normalize_lons,
    open_folder,
)
from detection import detect_onset, detect_wet_spell, wet_spell_start
from extract import extract_clim, extract_year, onset_series, rainfall_cdf
from forecast import (
    extract_forecast_ts,
    forecast_available_years,
    forecast_init_dates,
)
from maps import (
    clip_map,
    doy_maps_agg,
    doy_maps_year,
    jjas_cmap,
    nanminmax,
    plot_doy_map,
    plot_rain_map,
    snap_to_ref,
)

st.set_page_config(layout="wide")

# ─────────────────────────────────────────────
# Sidebar: onset parameters → OnsetParams
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
clip_ts_to_window = st.sidebar.toggle(
    "Clip timeseries to search window",
    value=True,
    help="When on, the timeseries x-axis is limited to the search window.",
)
start_month = st.sidebar.selectbox(
    "Search start month", list(range(1, 13)), index=4
)
start_day = st.sidebar.number_input(
    "Search start day", min_value=1, max_value=31, value=15, step=1
)
end_month = st.sidebar.selectbox(
    "Search end month", list(range(1, 13)), index=9
)
end_day = st.sidebar.number_input(
    "Search end day", min_value=1, max_value=31, value=15, step=1
)

params = OnsetParams(
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

# ─────────────────────────────────────────────
# Dataset registry (lazy-loaded on first use)
# ─────────────────────────────────────────────
DS_BY_KEY: Dict = {}
FILES_BY_KEY: Dict = {}
COORDS_BY_KEY: Dict = {}


def ensure_loaded(key: str) -> None:
    if key in DS_BY_KEY:
        return
    ds, files = open_folder(DATASET_FOLDERS[key])
    la, lo, ti = infer_coords(ds)
    DS_BY_KEY[key] = normalize_lons(ds, lo)
    FILES_BY_KEY[key] = files
    COORDS_BY_KEY[key] = (la, lo, ti)


# ─────────────────────────────────────────────
# Page title + tabs
# ─────────────────────────────────────────────
st.title("Rainfall Onset Explorer — Ethiopia")

tab_obs, tab_fcst = st.tabs(["Observations", "Forecasts"])

# ════════════════════════════════════════════
# TAB: Observations
# ════════════════════════════════════════════
with tab_obs:

    st.subheader("① Select dataset(s)")
    selected_datasets = st.multiselect(
        "Choose dataset(s)",
        options=list(DATASET_FOLDERS.keys()) + [STATION_KEY],
        default=["CHIRPS"],
    )
    if not selected_datasets:
        st.warning("Please select at least one dataset to continue.")
        st.stop()

    has_station = STATION_KEY in selected_datasets
    selected_grids = [k for k in selected_datasets if k in DATASET_FOLDERS]

    for key in selected_grids:
        ensure_loaded(key)

    emi_meta: Optional[pd.DataFrame] = None
    emi_ts: Optional[pd.DataFrame] = None
    if has_station:
        if not STATION_CSV_PATH.exists():
            st.error(
                f"EMI Stations selected but file not found: {STATION_CSV_PATH.name}"
            )
            st.stop()
        emi_meta, emi_ts = load_emi_csv(STATION_CSV_PATH)

    # ── Year range intersection ──
    year_ranges: Dict = {}
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

    if len(year_ranges) > 1:
        msgs = [
            f"**{k}**: {y0}–{y1}"
            for k, (y0, y1) in year_ranges.items()
            if y0 > year_min or y1 < year_max
        ]
        if msgs:
            st.info(
                f"Year selector limited to the overlapping range **{year_min}–{year_max}** "
                f"across all selected datasets. Individual ranges: {', '.join(msgs)}."
            )

    seasonal_notes = [
        f"**{k}** ({note})"
        for k, note in SEASONAL_COVERAGE_NOTE.items()
        if k in year_ranges
    ]
    if seasonal_notes:
        st.warning(
            f"Note: {', '.join(seasonal_notes)} — data outside this window will appear as missing. "
            "Onset/wet spell detection and JJAS maps are unaffected, but full-year timeseries "
            "and non-JJAS maps will have gaps."
        )

    year_list = list(range(year_min, year_max + 1))
    year_default = min(max(year_max - 5, year_min), year_max)

    # ── Grid domain (used for lat/lon input bounds and date picker) ──
    if selected_grids:
        lat_ranges, lon_ranges, tmin_list, tmax_list = [], [], [], []
        for key in selected_grids:
            la, lo, ti = COORDS_BY_KEY[key]
            ds_i = DS_BY_KEY[key]
            lat_ranges.append(
                (
                    float(ds_i[la].min().compute()),
                    float(ds_i[la].max().compute()),
                )
            )
            lon_ranges.append(
                (
                    float(ds_i[lo].min().compute()),
                    float(ds_i[lo].max().compute()),
                )
            )
            t0, t1 = dataset_time_range(ds_i, ti)
            tmin_list.append(t0)
            tmax_list.append(t1)
        lat_min = max(r[0] for r in lat_ranges)
        lat_max = min(r[1] for r in lat_ranges)
        lon_min = max(r[0] for r in lon_ranges)
        lon_max = min(r[1] for r in lon_ranges)
        tmin_common = max(tmin_list)
        tmax_common = min(tmax_list)

    def _year_selector(label: str, key_prefix: str):
        """Renders year / year-range widgets. Returns (y0, y1, agg_mode, year_mode_str)."""
        yr_mode = st.radio(
            label,
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

    # ════════════════════════════════════════════
    # CASE A — Grids only
    # ════════════════════════════════════════════
    if selected_grids and not has_station:

        st.subheader("② Select View(s)")
        show_ts = st.toggle("Timeseries", value=True, key="show_ts_a")
        show_map = st.toggle("Map (Ethiopia)", value=True, key="show_map_a")
        show_stats = st.toggle("Statistics", value=True, key="show_stats_a")
        if not show_ts and not show_map and not show_stats:
            st.warning("Select at least one view.")
            st.stop()

        if show_ts and show_map:
            link_periods = st.toggle(
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

        st.subheader("③ Time period")
        if link_periods or not (show_ts and show_map):
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
                slo = float(
                    ds_i[lo].sel({lo: lon0}, method="nearest").values
                )
                st.caption(
                    f"{key} snapped to nearest grid cell: ({sl:.4f}°N, {slo:.4f}°E)"
                )
        else:
            lat0 = float(np.clip(10.0, lat_min, lat_max))
            lon0 = float(np.clip(40.0, lon_min, lon_max))

        if show_map:
            st.subheader("⑤ Map options")
            use_default_mask = st.toggle(
                "Apply Ethiopia .nc mask", value=True, key="mask_a"
            )
            map_kind = st.radio(
                "Map type",
                [
                    "Seasonal mean rainfall (JJAS)",
                    "Daily rainfall map (selected date)",
                    "Wet spell date",
                    "Onset date",
                ],
                horizontal=True,
                key="a_map_kind",
            )
            date_sel = None
            if map_kind == "Daily rainfall map (selected date)":
                dflt = min(
                    max(
                        pd.Timestamp(year=map_y0, month=7, day=15),
                        tmin_common,
                    ),
                    tmax_common,
                )
                date_in = st.date_input(
                    "Select date",
                    value=dflt.date(),
                    min_value=tmin_common.date(),
                    max_value=tmax_common.date(),
                )
                date_sel = pd.Timestamp(date_in)

        # ── Timeseries panel ──
        if show_ts:
            st.subheader("Timeseries")
            title_ts = (
                f"Daily rainfall — ({lat0:.3f}°N, {lon0:.3f}°E) — {ts_y0}"
                if ts_year_mode == "Single year"
                else f"{agg_mode_ts} climatology — ({lat0:.3f}°N, {lon0:.3f}°E) — {ts_y0}–{ts_y1}"
            )
            fig_ts = ts_figure(title_ts)
            last_times = None
            for key in selected_grids:
                la, lo, ti = COORDS_BY_KEY[key]
                fi = FILES_BY_KEY[key]
                clr = DATASET_COLORS.get(key, {}).get("rain", "black")
                if ts_year_mode == "Single year":
                    df_ts = extract_year(
                        key, tuple(fi), la, lo, ti, ts_y0, lat0, lon0
                    )
                    add_rain(
                        fig_ts,
                        df_ts["time"],
                        df_ts["rain"],
                        f"{key} daily rainfall",
                        clr,
                    )
                    add_markers(fig_ts, df_ts, key, key, params)
                    last_times = df_ts["time"]
                else:
                    df_c = extract_clim(
                        key,
                        tuple(fi),
                        la,
                        lo,
                        ti,
                        ts_y0,
                        ts_y1,
                        lat0,
                        lon0,
                        agg_mode_ts,
                    )
                    add_rain(
                        fig_ts,
                        df_c["time"],
                        df_c["rain"],
                        f"{key} {agg_mode_ts} ({ts_y0}–{ts_y1})",
                        clr,
                    )
                    add_markers(fig_ts, df_c, key, key, params)
                    last_times = df_c["time"]
            if last_times is not None:
                add_threshold(fig_ts, params.wet_day_mm, last_times)
            ref_yr = ts_y0 if ts_year_mode == "Single year" else 2001
            xr_range = ts_xaxis_range(ref_yr, clip_ts_to_window, params)
            if xr_range:
                fig_ts.update_layout(xaxis_range=xr_range)
            st.plotly_chart(fig_ts, use_container_width=True)

        # ── Map panel ──
        if show_map:
            st.subheader("Map view")
            eth = load_boundary()
            eth_geom = eth.geometry.iloc[0]
            eth_bounds = tuple(map(float, eth.total_bounds))
            mask_da = load_mask() if use_default_mask else None
            cmap_jjas, norm_jjas, tpos, tlbl = jjas_cmap()
            yr_label = (
                str(map_y0) if map_y0 == map_y1 else f"{map_y0}–{map_y1}"
            )

            def get_map_Z(key):
                la, lo, ti = COORDS_BY_KEY[key]
                fi = FILES_BY_KEY[key]
                ds_i = DS_BY_KEY[key]
                rv = RAIN_VAR_BY_KEY.get(key, RAIN_VAR)
                if map_kind == "Seasonal mean rainfall (JJAS)":
                    da = ds_i[rv].sel(
                        {ti: slice(f"{map_y0}-01-01", f"{map_y1}-12-31")}
                    )
                    da = (
                        da.where(
                            da[ti].dt.month.isin([6, 7, 8, 9]), drop=True
                        )
                        .mean(dim=ti)
                        .compute()
                    )
                    return (
                        clip_map(
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
                    da = (
                        ds_i[rv]
                        .sel({ti: date_sel}, method="nearest")
                        .compute()
                    )
                    return (
                        clip_map(
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
                        lv, lov, wd, od = doy_maps_year(
                            tuple(fi), rv, la, lo, ti, map_y0, params
                        )
                    else:
                        lv, lov, wd, od = doy_maps_agg(
                            tuple(fi),
                            rv,
                            la,
                            lo,
                            ti,
                            map_y0,
                            map_y1,
                            agg_mode_map,
                            params,
                        )
                    Z0 = wd if map_kind == "Wet spell date" else od
                    return (
                        clip_map(Z0, lv, lov, eth_geom, eth_bounds, mask_da),
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
                        "Wet spell date"
                        if map_kind == "Wet spell date"
                        else "Onset date"
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
                results = {key: get_map_Z(key) for key in selected_grids}

                ref_key = max(
                    selected_grids,
                    key=lambda k: results[k][0][2].size
                    - np.isnan(results[k][0][2]).sum(),
                )
                ref_lv = results[ref_key][0][0]
                ref_lov = results[ref_key][0][1]
                aligned = {}
                for key in selected_grids:
                    (lv, lov, Z), kind = results[key]
                    lv2, lov2, Z2 = snap_to_ref(lv, lov, Z, ref_lv, ref_lov)
                    aligned[key] = ((lv2, lov2, Z2), kind)
                results = aligned

                pairs = list(combinations(selected_grids, 2))
                fig_w, fig_h = 4, 4
                has_rain_maps = any(
                    results[k][1] == "rain" for k in selected_grids
                )
                shared_rain_scale, shared_diff_scale = True, True

                if has_rain_maps and len(selected_grids) > 1:
                    shared_rain_scale = st.toggle(
                        "Shared scale across rainfall maps",
                        value=True,
                        key="shared_rain_scale",
                        help="When on, all individual rainfall maps use the same min/max colour scale.",
                    )
                if (
                    len(pairs) > 1
                    and has_rain_maps
                    and map_kind
                    in (
                        "Seasonal mean rainfall (JJAS)",
                        "Daily rainfall map (selected date)",
                    )
                ):
                    shared_diff_scale = st.toggle(
                        "Shared scale across difference maps",
                        value=True,
                        key="shared_diff_scale",
                        help="When on, all difference maps use the same diverging colour scale.",
                    )

                all_Z_rain = [
                    results[k][0][2]
                    for k in selected_grids
                    if results[k][1] == "rain"
                ]
                vmin_shared = vmax_shared = None
                if shared_rain_scale and len(all_Z_rain) > 1:
                    vmin_shared, vmax_shared = nanminmax(
                        np.concatenate([z.ravel() for z in all_Z_rain])
                    )

                diff_results = {}
                for kA, kB in pairs:
                    (lvA, lovA, ZA), kindA = results[kA]
                    (_, _, ZB), _ = results[kB]
                    diff_results[(kA, kB)] = (lvA, lovA, ZA - ZB, kindA)

                diff_ma_shared = None
                if shared_diff_scale and len(pairs) > 1:
                    rain_diffs = [
                        np.abs(diff_results[p][2])
                        for p in pairs
                        if diff_results[p][3] == "rain"
                    ]
                    if rain_diffs:
                        diff_ma_shared = (
                            float(
                                max(
                                    np.nanmax(d)
                                    for d in rain_diffs
                                    if np.any(np.isfinite(d))
                                )
                            )
                            or 1e-6
                        )

                panels = [("individual", key) for key in selected_grids]
                panels += [("diff", kA, kB) for kA, kB in pairs]

                for row_start in range(0, len(panels), 3):
                    row_panels = panels[row_start : row_start + 3]
                    row_cols = st.columns(len(row_panels))
                    for col_idx, panel in enumerate(row_panels):
                        with row_cols[col_idx]:
                            if panel[0] == "individual":
                                key = panel[1]
                                (lv, lov, Z), kind = results[key]
                                st.markdown(f"**{key}**")
                                fig_m, ax = plt.subplots(
                                    figsize=(fig_w, fig_h)
                                )
                                Lon, Lat = np.meshgrid(lov, lv)
                                if kind == "rain":
                                    vm0 = (
                                        vmin_shared
                                        if shared_rain_scale
                                        else None
                                    )
                                    vm1 = (
                                        vmax_shared
                                        if shared_rain_scale
                                        else None
                                    )
                                    plot_rain_map(
                                        ax,
                                        Lon,
                                        Lat,
                                        Z,
                                        eth,
                                        f"{key} — {yr_label}",
                                        "mm/day",
                                        fig_m,
                                        vm0,
                                        vm1,
                                    )
                                else:
                                    cb_lbl = (
                                        "Wet spell date"
                                        if map_kind == "Wet spell date"
                                        else "Onset date"
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
                            else:
                                _, kA, kB = panel
                                lvA, lovA, Zdiff, kindA = diff_results[
                                    (kA, kB)
                                ]
                                st.markdown(f"**{kA} − {kB}**")
                                fig_m, ax = plt.subplots(
                                    figsize=(fig_w, fig_h)
                                )
                                Lon, Lat = np.meshgrid(lovA, lvA)
                                if kindA == "rain":
                                    ma = (
                                        (
                                            diff_ma_shared
                                            if shared_diff_scale
                                            and diff_ma_shared
                                            else None
                                        )
                                        or float(
                                            np.nanmax(np.abs(Zdiff))
                                        )
                                        or 1e-6
                                    )
                                    pcm = ax.pcolormesh(
                                        Lon,
                                        Lat,
                                        Zdiff,
                                        shading="auto",
                                        cmap="RdBu_r",
                                        norm=TwoSlopeNorm(
                                            vcenter=0.0,
                                            vmin=-ma,
                                            vmax=ma,
                                        ),
                                    )
                                    eth.boundary.plot(
                                        ax=ax,
                                        linewidth=2.0,
                                        edgecolor="black",
                                        zorder=10,
                                    )
                                    ax.set_title(
                                        f"{kA}−{kB} — {yr_label}",
                                        fontsize=9,
                                    )
                                    ax.set_xlabel("Lon")
                                    ax.set_ylabel("Lat")
                                    fig_m.colorbar(pcm, ax=ax).set_label(
                                        f"mm/day ({kA}−{kB})", fontsize=8
                                    )
                                else:
                                    lvls = np.arange(-30, 31, 5)
                                    pcm = ax.pcolormesh(
                                        Lon,
                                        Lat,
                                        np.clip(Zdiff, -30, 30),
                                        shading="auto",
                                        cmap=plt.get_cmap(
                                            "RdBu_r", len(lvls) - 1
                                        ),
                                        norm=BoundaryNorm(
                                            lvls, len(lvls) - 1
                                        ),
                                    )
                                    eth.boundary.plot(
                                        ax=ax,
                                        linewidth=2.0,
                                        edgecolor="black",
                                        zorder=10,
                                    )
                                    ax.set_title(
                                        f"{kA}−{kB} difference — {yr_label}",
                                        fontsize=9,
                                    )
                                    ax.set_xlabel("Lon")
                                    ax.set_ylabel("Lat")
                                    fig_m.colorbar(
                                        pcm, ax=ax, ticks=lvls
                                    ).set_label(
                                        f"Days ({kA}−{kB})", fontsize=8
                                    )
                                st.pyplot(fig_m)

        # ── Statistics panel ──
        if show_stats:
            st.divider()
            st.subheader("⑥ Statistics")
            st.caption(
                "Uses the same search window and onset parameters as the rest of the app."
            )

            if not show_ts:
                st.info(
                    "Enable the Timeseries view and select a point to use Statistics."
                )
            else:
                cSA, cSB = st.columns(2)
                with cSA:
                    stat_y0 = int(
                        st.selectbox(
                            "Stats start year",
                            year_list,
                            index=0,
                            key="stat_y0",
                        )
                    )
                with cSB:
                    stat_y1 = int(
                        st.selectbox(
                            "Stats end year",
                            year_list,
                            index=len(year_list) - 1,
                            key="stat_y1",
                        )
                    )
                if stat_y1 <= stat_y0:
                    st.error("Stats end year must be > start year.")
                    st.stop()

                col_t1, col_t2, col_t3 = st.columns(3)
                with col_t1:
                    show_onset_scatter = st.toggle(
                        "Onset date scatter",
                        value=True,
                        key="show_onset_scatter",
                    )
                with col_t2:
                    show_onset_dist = st.toggle(
                        "Onset distribution",
                        value=True,
                        key="show_onset_dist",
                    )
                with col_t3:
                    show_cdf = st.toggle(
                        "Rainfall CDF", value=True, key="show_cdf"
                    )

                # Onset date scatter
                if show_onset_scatter:
                    st.markdown("### Onset date scatter")
                    st.caption(
                        "Onset DOY per year at the selected point. Missing = no onset detected."
                    )
                    fig_sc = go.Figure()
                    for key in selected_grids:
                        la, lo, ti = COORDS_BY_KEY[key]
                        fi = FILES_BY_KEY[key]
                        clr = DATASET_COLORS.get(key, {}).get(
                            "rain", "black"
                        )
                        df_sc = onset_series(
                            key,
                            tuple(fi),
                            la,
                            lo,
                            ti,
                            stat_y0,
                            stat_y1,
                            lat0,
                            lon0,
                            params,
                        )
                        if df_sc.empty:
                            continue
                        doy_dates = pd.to_datetime(
                            "2001-01-01"
                        ) + pd.to_timedelta(
                            df_sc["onset_doy"].fillna(0).astype(int) - 1,
                            unit="D",
                        )
                        doy_labels = doy_dates.dt.strftime("%d %b").where(
                            df_sc["onset_doy"].notna(), ""
                        )
                        fig_sc.add_trace(
                            go.Scatter(
                                x=df_sc["year"],
                                y=df_sc["onset_doy"],
                                mode="lines+markers",
                                name=key,
                                line=dict(color=clr, width=1.8),
                                marker=dict(size=7, color=clr),
                                text=doy_labels,
                                hovertemplate="%{x}: DOY %{y} (%{text})<extra>"
                                + key
                                + "</extra>",
                            )
                        )
                    sw_doy0 = pd.Timestamp(
                        year=2001,
                        month=params.start_month,
                        day=params.start_day,
                    ).dayofyear
                    sw_doy1 = pd.Timestamp(
                        year=2001,
                        month=params.end_month,
                        day=params.end_day,
                    ).dayofyear
                    fig_sc.add_hrect(
                        y0=sw_doy0,
                        y1=sw_doy1,
                        fillcolor="gray",
                        opacity=0.08,
                        line_width=0,
                        annotation_text="Search window",
                        annotation_position="left",
                    )
                    fig_sc.update_layout(
                        title=f"Onset date per year — ({lat0:.3f}°N, {lon0:.3f}°E) — {stat_y0}–{stat_y1}",
                        xaxis_title="Year",
                        yaxis_title="Onset (DOY)",
                        yaxis=dict(
                            tickvals=[121, 152, 182, 213, 244, 265],
                            ticktext=[
                                "May 01",
                                "Jun 01",
                                "Jul 01",
                                "Aug 01",
                                "Sep 01",
                                "Sep 22",
                            ],
                        ),
                        height=420,
                        legend=dict(orientation="h", y=-0.2),
                        hovermode="x unified",
                    )
                    st.plotly_chart(fig_sc, use_container_width=True)

                # Onset distribution
                if show_onset_dist:
                    st.markdown("### Onset distribution")
                    use_region = st.toggle(
                        "Use bounding box region instead of point",
                        value=False,
                        key="dist_use_region",
                    )
                    if use_region:
                        st.caption(
                            "Onset detected at every grid cell in the bounding box; all detections contribute."
                        )
                        rc1, rc2, rc3, rc4 = st.columns(4)
                        with rc1:
                            reg_lat_min = st.number_input(
                                "Lat min",
                                value=float(
                                    np.clip(lat0 - 1, lat_min, lat_max)
                                ),
                                min_value=float(lat_min),
                                max_value=float(lat_max),
                                format="%.2f",
                                step=0.25,
                                key="reg_lat_min",
                            )
                        with rc2:
                            reg_lat_max = st.number_input(
                                "Lat max",
                                value=float(
                                    np.clip(lat0 + 1, lat_min, lat_max)
                                ),
                                min_value=float(lat_min),
                                max_value=float(lat_max),
                                format="%.2f",
                                step=0.25,
                                key="reg_lat_max",
                            )
                        with rc3:
                            reg_lon_min = st.number_input(
                                "Lon min",
                                value=float(
                                    np.clip(lon0 - 1, lon_min, lon_max)
                                ),
                                min_value=float(lon_min),
                                max_value=float(lon_max),
                                format="%.2f",
                                step=0.25,
                                key="reg_lon_min",
                            )
                        with rc4:
                            reg_lon_max = st.number_input(
                                "Lon max",
                                value=float(
                                    np.clip(lon0 + 1, lon_min, lon_max)
                                ),
                                min_value=float(lon_min),
                                max_value=float(lon_max),
                                format="%.2f",
                                step=0.25,
                                key="reg_lon_max",
                            )
                        if (
                            reg_lat_max <= reg_lat_min
                            or reg_lon_max <= reg_lon_min
                        ):
                            st.error(
                                "Bounding box max must be greater than min."
                            )
                            st.stop()
                        dist_label = f"{reg_lat_min:.2f}–{reg_lat_max:.2f}°N, {reg_lon_min:.2f}–{reg_lon_max:.2f}°E"
                    else:
                        dist_label = f"{lat0:.3f}°N, {lon0:.3f}°E"

                    sw_start_doy = pd.Timestamp(
                        year=2001,
                        month=params.start_month,
                        day=params.start_day,
                    ).dayofyear
                    sw_end_doy = pd.Timestamp(
                        year=2001,
                        month=params.end_month,
                        day=params.end_day,
                    ).dayofyear
                    bin_edges = list(
                        range(sw_start_doy, sw_end_doy + 1, 5)
                    )
                    if bin_edges[-1] < sw_end_doy:
                        bin_edges.append(sw_end_doy + 1)
                    bin_centres = [
                        (bin_edges[i] + bin_edges[i + 1]) / 2
                        for i in range(len(bin_edges) - 1)
                    ]
                    bin_labels = [doy_to_label(c) for c in bin_centres]
                    n_datasets = len(selected_grids)
                    bar_width = 4.0 / max(n_datasets, 1) * 0.85

                    fig_dist = go.Figure()
                    any_dist_data = False
                    for d_idx, key in enumerate(selected_grids):
                        la, lo, ti = COORDS_BY_KEY[key]
                        fi = FILES_BY_KEY[key]
                        clr = DATASET_COLORS.get(key, {}).get(
                            "rain", "black"
                        )
                        bbox = (
                            (
                                float(reg_lat_min),
                                float(reg_lat_max),
                                float(reg_lon_min),
                                float(reg_lon_max),
                            )
                            if use_region
                            else None
                        )
                        df_r = onset_series(
                            key,
                            tuple(fi),
                            la,
                            lo,
                            ti,
                            stat_y0,
                            stat_y1,
                            lat0,
                            lon0,
                            params,
                            bbox=bbox,
                        )
                        detected = (
                            df_r["onset_doy"].dropna()
                            if not df_r.empty
                            else pd.Series([], dtype=float)
                        )
                        denom_note = (
                            f"{len(detected)} cell-years"
                            if use_region
                            else f"{len(detected)} yrs"
                        )
                        if len(detected) == 0:
                            continue
                        counts = np.zeros(len(bin_edges) - 1)
                        for doy in detected:
                            for b in range(len(bin_edges) - 1):
                                if bin_edges[b] <= doy < bin_edges[b + 1]:
                                    counts[b] += 1
                                    break
                        probs = counts / len(detected)
                        hover = [
                            f"{doy_to_label(bin_edges[b])}–{doy_to_label(bin_edges[b+1]-1)}: "
                            f"P={probs[b]:.2f} ({int(counts[b])}/{denom_note})"
                            for b in range(len(bin_edges) - 1)
                        ]
                        offset = (d_idx - (n_datasets - 1) / 2) * (
                            4.0 / max(n_datasets, 1)
                        )
                        fig_dist.add_trace(
                            go.Bar(
                                x=[c + offset for c in bin_centres],
                                y=probs,
                                name=key,
                                width=bar_width,
                                marker_color=clr,
                                opacity=0.82,
                                text=hover,
                                textposition="none",
                                hovertemplate="%{text}<extra>"
                                + key
                                + "</extra>",
                            )
                        )
                        any_dist_data = True

                    if not any_dist_data:
                        st.info(
                            "No onset dates detected for any dataset in this year range and region."
                        )
                    else:
                        fig_dist.update_layout(
                            title=f"Onset distribution — {dist_label} — {stat_y0}–{stat_y1}",
                            xaxis=dict(
                                title="Onset date (5-day bins)",
                                tickvals=bin_centres,
                                ticktext=bin_labels,
                                tickangle=-30,
                            ),
                            yaxis=dict(
                                title="Probability",
                                range=[0, None],
                                tickformat=".0%",
                            ),
                            barmode="overlay",
                            height=440,
                            legend=dict(orientation="h", y=-0.3),
                            hovermode="x",
                        )
                        st.plotly_chart(fig_dist, use_container_width=True)
                        mode_note = (
                            "aggregated across all grid cells in the bounding box"
                            if use_region
                            else "at the selected point"
                        )
                        st.caption(
                            f"Probability = fraction of detections in each 5-day window, "
                            f"{mode_note}. {stat_y0}–{stat_y1} ({stat_y1 - stat_y0 + 1} years)."
                        )

                # Rainfall CDF
                if show_cdf:
                    st.markdown("### Rainfall CDF")
                    st.caption(
                        "Empirical CDF of all daily rainfall values (including dry days) at the selected point."
                    )
                    fig_cdf = go.Figure()
                    for key in selected_grids:
                        la, lo, ti = COORDS_BY_KEY[key]
                        fi = FILES_BY_KEY[key]
                        clr = DATASET_COLORS.get(key, {}).get(
                            "rain", "black"
                        )
                        vals = rainfall_cdf(
                            key,
                            tuple(fi),
                            la,
                            lo,
                            ti,
                            stat_y0,
                            stat_y1,
                            lat0,
                            lon0,
                        )
                        if len(vals) == 0:
                            continue
                        cdf_y = np.arange(1, len(vals) + 1) / len(vals)
                        fig_cdf.add_trace(
                            go.Scatter(
                                x=vals,
                                y=cdf_y,
                                mode="lines",
                                name=key,
                                line=dict(color=clr, width=2),
                                hovertemplate="%{x:.2f} mm → P=%{y:.3f}<extra>"
                                + key
                                + "</extra>",
                            )
                        )
                    fig_cdf.add_vline(
                        x=params.wet_day_mm,
                        line_dash="dot",
                        line_color="gray",
                        annotation_text=f"Wet threshold ({params.wet_day_mm:g} mm)",
                        annotation_position="top right",
                    )
                    fig_cdf.update_layout(
                        title=f"Rainfall CDF — ({lat0:.3f}°N, {lon0:.3f}°E) — {stat_y0}–{stat_y1}",
                        xaxis_title="Daily rainfall (mm/day)",
                        yaxis_title="Cumulative probability",
                        xaxis=dict(range=[0, None]),
                        yaxis=dict(range=[0, 1]),
                        height=420,
                        legend=dict(orientation="h", y=-0.2),
                        hovermode="x unified",
                    )
                    st.plotly_chart(fig_cdf, use_container_width=True)

    # ════════════════════════════════════════════
    # CASE B — Stations only
    # ════════════════════════════════════════════
    elif has_station and not selected_grids:

        st.info(
            "Map view is not available for EMI Stations. Showing timeseries only."
        )
        st.subheader("② Select EMI station")
        station_list = list(emi_meta.index)

        if "selected_station" not in st.session_state:
            st.session_state["selected_station"] = None

        clicked = st.plotly_chart(
            station_map(emi_meta, st.session_state["selected_station"]),
            use_container_width=True,
            on_select="rerun",
            key="station_map",
        )
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
        st.caption(
            f"Selected: **{station_name}** — ({st_lat:.4f}°N, {st_lon:.4f}°E)"
        )

        ts_year_mode = st.radio(
            "Year selection",
            ["Single year", "Year range"],
            horizontal=True,
            key="b_ts_mode",
        )
        if ts_year_mode == "Single year":
            ts_y0 = ts_y1 = int(
                st.selectbox(
                    "Year",
                    year_list,
                    index=year_list.index(year_default),
                    key="b_yr",
                )
            )
        else:
            cA, cB = st.columns(2)
            with cA:
                ts_y0 = int(
                    st.selectbox(
                        "Start year", year_list, index=0, key="b_y0"
                    )
                )
            with cB:
                ts_y1 = int(
                    st.selectbox(
                        "End year",
                        year_list,
                        index=len(year_list) - 1,
                        key="b_y1",
                    )
                )
            if ts_y1 < ts_y0:
                st.error("End year must be ≥ start year.")
                st.stop()

        s = emi_ts[station_name].loc[f"{ts_y0}-01-01":f"{ts_y1}-12-31"]
        df_st = s.rename("rain").to_frame().reset_index()
        df_st.columns = ["time", "rain"]

        if ts_year_mode == "Single year":
            fig_ts = ts_figure(
                f"Station: {station_name} ({st_lat:.3f}°N, {st_lon:.3f}°E) — {ts_y0}"
            )
            add_rain(
                fig_ts,
                df_st["time"],
                df_st["rain"],
                f"{station_name} daily rainfall",
                DATASET_COLORS[STATION_KEY]["rain"],
            )
            add_markers(fig_ts, df_st, STATION_KEY, station_name, params)
            add_threshold(fig_ts, params.wet_day_mm, df_st["time"])
        else:
            df_st["doy"] = df_st["time"].dt.dayofyear
            clim = df_st.groupby("doy")["rain"].mean().reset_index()
            clim = clim[clim["doy"] <= 365].copy()
            clim["time"] = pd.to_datetime("2001-01-01") + pd.to_timedelta(
                clim["doy"] - 1, unit="D"
            )
            fig_ts = ts_figure(
                f"Station climatology: {station_name} — {ts_y0}–{ts_y1}"
            )
            add_rain(
                fig_ts,
                clim["time"],
                clim["rain"],
                f"{station_name} mean ({ts_y0}–{ts_y1})",
                DATASET_COLORS[STATION_KEY]["rain"],
            )
            add_markers(fig_ts, clim, STATION_KEY, station_name, params)
            add_threshold(fig_ts, params.wet_day_mm, clim["time"])
            df_st = clim

        ref_yr = ts_y0 if ts_year_mode == "Single year" else 2001
        xr_range = ts_xaxis_range(ref_yr, clip_ts_to_window, params)
        if xr_range:
            fig_ts.update_layout(xaxis_range=xr_range)
        st.plotly_chart(fig_ts, use_container_width=True)

    # ════════════════════════════════════════════
    # CASE C — Stations + grids
    # ════════════════════════════════════════════
    elif has_station and selected_grids:

        st.info(
            "Map view is not available when EMI Stations are selected. Showing timeseries only."
        )
        st.subheader("② Select EMI station")
        station_list = list(emi_meta.index)

        if "selected_station" not in st.session_state:
            st.session_state["selected_station"] = None

        clicked = st.plotly_chart(
            station_map(emi_meta, st.session_state["selected_station"]),
            use_container_width=True,
            on_select="rerun",
            key="station_map",
        )
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
        st.caption(
            f"Selected: **{station_name}** — ({st_lat:.4f}°N, {st_lon:.4f}°E)"
        )

        for key in selected_grids:
            la, lo, _ = COORDS_BY_KEY[key]
            ds_i = DS_BY_KEY[key]
            sl = float(ds_i[la].sel({la: st_lat}, method="nearest").values)
            slo = float(
                ds_i[lo].sel({lo: st_lon}, method="nearest").values
            )
            st.caption(
                f"{key} nearest grid cell: ({sl:.4f}°N, {slo:.4f}°E)"
            )

        ts_year_mode = st.radio(
            "Year selection",
            ["Single year", "Year range"],
            horizontal=True,
            key="c_ts_mode",
        )
        if ts_year_mode == "Single year":
            ts_y0 = ts_y1 = int(
                st.selectbox(
                    "Year",
                    year_list,
                    index=year_list.index(year_default),
                    key="c_yr",
                )
            )
            agg_mode_ts = "Mean"
        else:
            cA, cB = st.columns(2)
            with cA:
                ts_y0 = int(
                    st.selectbox(
                        "Start year", year_list, index=0, key="c_y0"
                    )
                )
            with cB:
                ts_y1 = int(
                    st.selectbox(
                        "End year",
                        year_list,
                        index=len(year_list) - 1,
                        key="c_y1",
                    )
                )
            if ts_y1 < ts_y0:
                st.error("End year must be ≥ start year.")
                st.stop()
            agg_mode_ts = st.radio(
                "Climatology aggregation",
                ["Mean", "Median"],
                horizontal=True,
                key="c_agg",
            )

        s = emi_ts[station_name].loc[f"{ts_y0}-01-01":f"{ts_y1}-12-31"]
        df_st = s.rename("rain").to_frame().reset_index()
        df_st.columns = ["time", "rain"]

        if ts_year_mode == "Single year":
            fig_ts = ts_figure(
                f"Station vs gridded — {station_name} ({st_lat:.3f}°N, {st_lon:.3f}°E) — {ts_y0}"
            )
            add_rain(
                fig_ts,
                df_st["time"],
                df_st["rain"],
                f"{station_name} daily rainfall",
                DATASET_COLORS[STATION_KEY]["rain"],
            )
            add_markers(fig_ts, df_st, STATION_KEY, station_name, params)
            for key in selected_grids:
                la, lo, ti = COORDS_BY_KEY[key]
                fi = FILES_BY_KEY[key]
                df_g = extract_year(
                    key, tuple(fi), la, lo, ti, ts_y0, st_lat, st_lon
                )
                sl = df_g["snapped_lat"].iloc[0]
                slo = df_g["snapped_lon"].iloc[0]
                add_rain(
                    fig_ts,
                    df_g["time"],
                    df_g["rain"],
                    f"{key} daily rainfall @ ({sl:.3f}°N, {slo:.3f}°E)",
                    DATASET_COLORS.get(key, {}).get("rain", "black"),
                )
                add_markers(fig_ts, df_g, key, key, params)
            add_threshold(fig_ts, params.wet_day_mm, df_st["time"])
        else:
            fig_ts = ts_figure(
                f"Climatology — {station_name} vs gridded — {ts_y0}–{ts_y1}"
            )
            df_st["doy"] = df_st["time"].dt.dayofyear
            clim_st = df_st.groupby("doy")["rain"].mean().reset_index()
            clim_st = clim_st[clim_st["doy"] <= 365].copy()
            clim_st["time"] = pd.to_datetime(
                "2001-01-01"
            ) + pd.to_timedelta(clim_st["doy"] - 1, unit="D")
            add_rain(
                fig_ts,
                clim_st["time"],
                clim_st["rain"],
                f"{station_name} mean ({ts_y0}–{ts_y1})",
                DATASET_COLORS[STATION_KEY]["rain"],
            )
            add_markers(fig_ts, clim_st, STATION_KEY, station_name, params)
            for key in selected_grids:
                la, lo, ti = COORDS_BY_KEY[key]
                fi = FILES_BY_KEY[key]
                df_c = extract_clim(
                    key,
                    tuple(fi),
                    la,
                    lo,
                    ti,
                    ts_y0,
                    ts_y1,
                    st_lat,
                    st_lon,
                    agg_mode_ts,
                )
                sl = df_c["snapped_lat"].iloc[0]
                slo = df_c["snapped_lon"].iloc[0]
                add_rain(
                    fig_ts,
                    df_c["time"],
                    df_c["rain"],
                    f"{key} {agg_mode_ts} @ ({sl:.3f}°N, {slo:.3f}°E) ({ts_y0}–{ts_y1})",
                    DATASET_COLORS.get(key, {}).get("rain", "black"),
                )
                add_markers(fig_ts, df_c, key, key, params)
            add_threshold(fig_ts, params.wet_day_mm, clim_st["time"])

        ref_yr = ts_y0 if ts_year_mode == "Single year" else 2001
        xr_range = ts_xaxis_range(ref_yr, clip_ts_to_window, params)
        if xr_range:
            fig_ts.update_layout(xaxis_range=xr_range)
        st.plotly_chart(fig_ts, use_container_width=True)


# ════════════════════════════════════════════
# TAB: Forecasts
# ════════════════════════════════════════════
with tab_fcst:

    st.subheader("① Select forecast model(s)")
    sel_models = st.multiselect(
        "Model(s)",
        ["AIFS", "AIFS_ENS", "GENCAST"],
        default=["AIFS"],
        key="fc_models",
    )
    if not sel_models:
        st.warning("Select at least one model.")
        st.stop()

    # Year range — intersection across selected models
    avail_by_model = {m: forecast_available_years(m) for m in sel_models}
    fc_year_set = set.intersection(*[set(v) for v in avail_by_model.values()])
    if not fc_year_set:
        st.error("No overlapping years across selected models.")
        st.stop()
    fc_year_list = sorted(fc_year_set)

    fc_year = int(
        st.selectbox(
            "Year",
            fc_year_list,
            index=len(fc_year_list) - 1,
            key="fc_year",
        )
    )

    # Init dates from first selected model
    ref_inits = forecast_init_dates(sel_models[0], fc_year)
    init_options = {d.strftime("%Y-%m-%d"): d for d in ref_inits}

    sel_init_strs = st.multiselect(
        "Initialization date(s)",
        list(init_options.keys()),
        default=[list(init_options.keys())[0]],
        key="fc_inits",
    )
    if not sel_init_strs:
        st.warning("Select at least one initialization date.")
        st.stop()
    sel_inits = [init_options[s] for s in sel_init_strs]

    st.subheader("② Point selection")
    fc1, fc2 = st.columns(2)
    with fc1:
        fc_lat = st.number_input(
            "Latitude",
            min_value=3.0,
            max_value=15.0,
            value=10.0,
            step=0.25,
            format="%.4f",
            key="fc_lat",
        )
    with fc2:
        fc_lon = st.number_input(
            "Longitude",
            min_value=33.0,
            max_value=48.0,
            value=40.0,
            step=0.25,
            format="%.4f",
            key="fc_lon",
        )

    st.subheader("③ Display options")
    fc_show_members = st.toggle(
        "Show individual ensemble members",
        value=True,
        key="fc_show_members",
    )

    n_combos = len(sel_models) * len(sel_inits)
    if n_combos > 5 and fc_show_members:
        st.warning(
            f"{n_combos} model × init combinations with individual members shown "
            "— this may render slowly. Consider reducing selections or "
            "disabling individual members."
        )

    # ── Build figure ──
    fig_fc = ts_figure(
        f"Forecast rainfall — ({fc_lat:.3f}°N, {fc_lon:.3f}°E) — {fc_year}"
    )
    summary_rows = []

    for model in sel_models:
        fc_clr = FORECAST_COLORS[model]
        for init_date in sel_inits:
            label = f"{model} {init_date.strftime('%b %d')}"
            df_fc = extract_forecast_ts(
                model, fc_year, init_date, fc_lat, fc_lon
            )

            if not FORECAST_IS_ENS[model]:
                # ── Deterministic ──
                add_rain(
                    fig_fc,
                    df_fc["valid_date"],
                    df_fc["rain"],
                    label,
                    fc_clr["rain"],
                )
                df_for_markers = df_fc.rename(
                    columns={"valid_date": "time"}
                )
                add_markers(fig_fc, df_for_markers, model, label, params)

                times_s = pd.Series(df_fc["valid_date"].values)
                rain_s = df_fc["rain"].values.astype(float)
                _, wi, si_ws = detect_wet_spell(times_s, rain_s, params)
                ws = wet_spell_start(
                    times_s, rain_s, wi, params.wet_day_mm, si_ws
                )
                od, _ = detect_onset(times_s, rain_s, params)
                summary_rows.append(
                    {
                        "Model": model,
                        "Init date": init_date.strftime("%d %b %Y"),
                        "Wet spell start": (
                            ws.strftime("%d %b") if ws else "—"
                        ),
                        "Onset date": od.strftime("%d %b") if od else "—",
                        "Members w/ onset": "—",
                    }
                )

            else:
                # ── Ensemble ──
                pivot = df_fc.pivot_table(
                    index="valid_date", columns="member", values="rain"
                )
                times_idx = pivot.index
                rain_mat = pivot.values  # (n_days, n_members)

                if fc_show_members:
                    add_ensemble_members(
                        fig_fc,
                        times_idx,
                        rain_mat,
                        label,
                        fc_clr["rain"],
                    )
                add_ensemble_fan(
                    fig_fc,
                    times_idx,
                    rain_mat,
                    label,
                    fc_clr["rain"],
                    fc_clr["fan"],
                )
                add_ensemble_onset_markers(
                    fig_fc, df_fc, model, label, params
                )

                # Build summary row via per-member detection
                members = sorted(df_fc["member"].unique())
                ws_dates, od_dates = [], []
                for m in members:
                    dfm = df_fc[df_fc["member"] == m].sort_values(
                        "valid_date"
                    )
                    times_m = pd.Series(dfm["valid_date"].values)
                    rain_m = dfm["rain"].values.astype(float)
                    if not np.any(np.isfinite(rain_m)):
                        continue
                    _, wi, si_ws = detect_wet_spell(
                        times_m, rain_m, params
                    )
                    ws = wet_spell_start(
                        times_m, rain_m, wi, params.wet_day_mm, si_ws
                    )
                    od, _ = detect_onset(times_m, rain_m, params)
                    if ws is not None:
                        ws_dates.append(ws)
                    if od is not None:
                        od_dates.append(od)

                ws_med = (
                    pd.Timestamp(
                        int(np.median([t.value for t in ws_dates]))
                    )
                    if ws_dates
                    else None
                )
                od_med = (
                    pd.Timestamp(
                        int(np.median([t.value for t in od_dates]))
                    )
                    if od_dates
                    else None
                )
                summary_rows.append(
                    {
                        "Model": model,
                        "Init date": init_date.strftime("%d %b %Y"),
                        "Wet spell start": (
                            ws_med.strftime("%d %b") if ws_med else "—"
                        ),
                        "Onset date": (
                            od_med.strftime("%d %b") if od_med else "—"
                        ),
                        "Members w/ onset": (
                            f"{len(od_dates)}/{len(members)}"
                        ),
                    }
                )

    # Wet threshold line spanning all forecast valid dates
    all_fc_times: list = []
    for model in sel_models:
        for init_date in sel_inits:
            df_tmp = extract_forecast_ts(
                model, fc_year, init_date, fc_lat, fc_lon
            )
            all_fc_times.extend(df_tmp["valid_date"].tolist())
    if all_fc_times:
        add_threshold(
            fig_fc,
            params.wet_day_mm,
            pd.Series(sorted(set(all_fc_times))),
        )

    if clip_ts_to_window:
        xr_range = ts_xaxis_range(fc_year, True, params)
        if xr_range:
            fig_fc.update_layout(xaxis_range=xr_range)

    st.plotly_chart(fig_fc, use_container_width=True)

    # ── Onset / wet-spell summary table ──
    if summary_rows:
        st.subheader("Onset / Wet spell summary")
        st.dataframe(pd.DataFrame(summary_rows), hide_index=True)
