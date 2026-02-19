# from __future__ import annotations
# from matplotlib.colors import ListedColormap, BoundaryNorm

# from pathlib import Path
# from typing import Tuple, Optional, List

# import numpy as np
# import pandas as pd
# import xarray as xr
# import matplotlib.pyplot as plt
# import streamlit as st

# # Map deps
# import geopandas as gpd
# from shapely.geometry import Point
# from shapely.prepared import prep


# # -----------------------------
# # Helpers: coordinate detection
# # -----------------------------


# def _find_coord_name(ds: xr.Dataset, candidates: List[str]) -> Optional[str]:
#     for name in candidates:
#         if name in ds.coords or name in ds.dims:
#             return name
#     return None


# def guess_lat_lon_time_names(ds: xr.Dataset) -> Tuple[str, str, str]:
#     """
#     If your files always use ('latitude','longitude','time'), keep this simple.
#     If you later ingest different datasets, expand candidates and call _find_coord_name.
#     """
#     lat_name = "latitude"
#     lon_name = "longitude"
#     time_name = "time"

#     # sanity check (helps catch silent failures)
#     for nm in (lat_name, lon_name, time_name):
#         if nm not in ds.coords and nm not in ds.dims:
#             raise ValueError(
#                 f"Could not find coord/dim '{nm}'. Found coords={list(ds.coords)}, dims={list(ds.dims)}"
#             )
#     return lat_name, lon_name, time_name


# def maybe_normalize_longitudes(ds: xr.Dataset, lon_name: str) -> xr.Dataset:
#     """If lon is 0..360, convert to -180..180 (useful for Ethiopia subsetting)."""
#     lon = ds[lon_name]
#     if np.nanmax(lon.values) > 180:
#         new_lon = ((lon + 180) % 360) - 180
#         ds = ds.assign_coords({lon_name: new_lon}).sortby(lon_name)
#     return ds


# # -----------------------------
# # Onset + wet-spell definitions
# # -----------------------------


# def _year_window_indices(
#     d: pd.Series,
#     start_month: int,
#     start_day: int,
#     end_month: int,
#     end_day: int,
# ) -> Tuple[int, int, pd.Timestamp, pd.Timestamp]:
#     d = pd.to_datetime(d).reset_index(drop=True)
#     year = int(d.iloc[0].year)
#     start_date = pd.Timestamp(year=year, month=start_month, day=start_day)
#     end_date = pd.Timestamp(year=year, month=end_month, day=end_day)
#     start_idx = int(np.searchsorted(d.values, np.datetime64(start_date)))
#     end_idx = int(np.searchsorted(d.values, np.datetime64(end_date), side="right"))
#     return start_idx, end_idx, start_date, end_date


# def detect_wet_spell_candidate(
#     dates: pd.Series,
#     rain: np.ndarray,
#     wet_day_mm: float,
#     accum_days: int,
#     accum_mm: float,
#     start_month: int,
#     start_day: int,
#     end_month: int,
#     end_day: int,
# ) -> Tuple[Optional[pd.Timestamp], Optional[int]]:
#     """
#     Wet-spell candidate (ICPAC-style): first day t such that
#       1) day t is a wet day (rain[t] >= wet_day_mm), AND
#       2) sum(rain[t : t+accum_days]) >= accum_mm,
#     within the search window.
#     """
#     r = np.asarray(rain, dtype=float)
#     d = pd.to_datetime(dates).reset_index(drop=True)
#     wet = r >= wet_day_mm

#     start_idx, end_idx, _, _ = _year_window_indices(
#         d, start_month, start_day, end_month, end_day
#     )

#     n = len(r)
#     last_t = n - accum_days  # must allow full accum window
#     t_stop = min(last_t + 1, end_idx)  # +1 because range upper bound is exclusive

#     for t in range(start_idx, max(start_idx, t_stop)):
#         if not wet[t]:
#             continue
#         if np.nansum(r[t : t + accum_days]) >= accum_mm:
#             return d.iloc[t], t

#     return None, None


# def detect_onset(
#     dates: pd.Series,
#     rain: np.ndarray,
#     wet_day_mm: float,
#     accum_days: int,
#     accum_mm: float,
#     dry_spell_days: int,
#     lookahead_days: int,
#     start_month: int,
#     start_day: int,
#     end_month: int,
#     end_day: int,
# ) -> Tuple[Optional[pd.Timestamp], Optional[int]]:
#     """
#     Onset (ICPAC-style): first day t such that
#       1) day t is a wet day (rain[t] >= wet_day_mm), AND
#       2) sum(rain[t : t+accum_days]) >= accum_mm, AND
#       3) within the next lookahead_days, there is NO run of >= dry_spell_days consecutive dry days.
#     Search restricted to [start_date, end_date] in the given year.
#     """
#     r = np.asarray(rain, dtype=float)
#     d = pd.to_datetime(dates).reset_index(drop=True)

#     if len(r) != len(d):
#         raise ValueError("dates and rain must be same length")

#     wet = r >= wet_day_mm
#     start_idx, end_idx, _, _ = _year_window_indices(
#         d, start_month, start_day, end_month, end_day
#     )

#     n = len(r)

#     # last possible t must allow accum window + lookahead window
#     last_t = n - accum_days - lookahead_days
#     if last_t <= start_idx:
#         return None, None

#     t_stop = min(last_t, end_idx)
#     if t_stop <= start_idx:
#         return None, None

#     for t in range(start_idx, t_stop):
#         # IMPORTANT: onset must be the first day of a wet spell → require wet start day
#         if not wet[t]:
#             continue

#         if np.nansum(r[t : t + accum_days]) >= accum_mm:
#             future_wet = wet[t : t + lookahead_days]

#             dry_run = 0
#             ok = True
#             for is_wet in future_wet:
#                 if not is_wet:
#                     dry_run += 1
#                     if dry_run >= dry_spell_days:
#                         ok = False
#                         break
#                 else:
#                     dry_run = 0

#             if ok:
#                 return d.iloc[t], t

#     return None, None


# def wet_run_start_containing_index(
#     dates: pd.Series,
#     rain: np.ndarray,
#     idx: Optional[int],
#     wet_day_mm: float,
# ) -> Optional[pd.Timestamp]:
#     """
#     Returns the first day of the consecutive wet run that contains idx (if idx is wet).
#     If idx is None → None.
#     If idx is dry → returns that day (so caller can decide what “start” means).
#     """
#     if idx is None:
#         return None

#     r = np.asarray(rain, dtype=float)
#     d = pd.to_datetime(dates).reset_index(drop=True)

#     if idx < 0 or idx >= len(r):
#         return None

#     if r[idx] < wet_day_mm:
#         return d.iloc[idx]

#     j = idx
#     while j - 1 >= 0 and r[j - 1] >= wet_day_mm:
#         j -= 1
#     return d.iloc[j]


# # -----------------------------
# # Ethiopia boundary helpers
# # -----------------------------


# @st.cache_resource
# def load_ethiopia_boundary() -> gpd.GeoDataFrame:
#     path = Path("data/ethiopia.geojson")
#     if not path.exists():
#         raise FileNotFoundError("data/ethiopia.geojson not found")
#     eth = gpd.read_file(path).to_crs("EPSG:4326")
#     return eth.dissolve()


# def make_inside_mask(
#     lat_vals: np.ndarray, lon_vals: np.ndarray, eth_geom
# ) -> np.ndarray:
#     lat_vals = np.asarray(lat_vals)
#     lon_vals = np.asarray(lon_vals)
#     g = prep(eth_geom)

#     # Shapely 2 fast path
#     try:
#         from shapely import contains_xy

#         Lon, Lat = np.meshgrid(lon_vals, lat_vals)
#         return contains_xy(eth_geom, Lon, Lat)
#     except Exception:
#         mask = np.zeros((len(lat_vals), len(lon_vals)), dtype=bool)
#         for i, la in enumerate(lat_vals):
#             for j, lo in enumerate(lon_vals):
#                 mask[i, j] = g.contains(Point(float(lo), float(la)))
#         return mask


# # -----------------------------
# # Streamlit App
# # -----------------------------

# st.set_page_config(layout="wide")
# st.title("Rainfall Onset Explorer (Ethiopia)")

# FOLDER = "obs_subset_ethiopia"
# FILE_GLOB = "*.nc"
# CHUNKSIZE = 365

# data_dir = Path(FOLDER).expanduser().resolve()
# files = sorted(data_dir.glob(FILE_GLOB))

# if len(files) == 0:
#     st.error(f"No files matched {FILE_GLOB} in {data_dir}")
#     st.stop()

# st.caption(f"Found {len(files)} files in {data_dir.name} (chunk={CHUNKSIZE} days).")


# @st.cache_resource
# def open_multi_nc(files_: List[str]) -> xr.Dataset:
#     return xr.open_mfdataset(
#         files_,
#         combine="by_coords",
#         parallel=True,
#         chunks={"time": CHUNKSIZE},
#         engine=None,
#     )


# ds = open_multi_nc([str(f) for f in files])

# # Detect coordinate names
# try:
#     lat_name, lon_name, time_name = guess_lat_lon_time_names(ds)
# except Exception as e:
#     st.exception(e)
#     st.stop()

# # Normalize longitudes if needed
# ds = maybe_normalize_longitudes(ds, lon_name)

# var = "precip"
# if var not in ds.data_vars:
#     st.error(f"Rainfall variable '{var}' not found. Available: {list(ds.data_vars)}")
#     st.stop()

# st.caption(f"Using rainfall variable: `{var}`")

# lat_min, lat_max = float(ds[lat_name].min()), float(ds[lat_name].max())
# lon_min, lon_max = float(ds[lon_name].min()), float(ds[lon_name].max())

# # Years
# years = ds[time_name].dt.year
# year_min = (
#     int(years.min().compute()) if hasattr(years.min(), "compute") else int(years.min())
# )
# year_max = (
#     int(years.max().compute()) if hasattr(years.max(), "compute") else int(years.max())
# )
# year_list = list(range(year_min, year_max + 1))

# # Mode selection: Point vs Map
# st.subheader("View mode")
# mode = st.radio(
#     "Choose a view", ["Point timeseries", "Map (Ethiopia)"], horizontal=True
# )

# # Point selection
# st.subheader("Point selection")
# c2, c3, c4 = st.columns([1, 1, 1])

# st.caption(
#     f"Available data domain: Latitude {lat_min:.2f}° to {lat_max:.2f}°, "
#     f"Longitude {lon_min:.2f}° to {lon_max:.2f}°"
# )

# with c2:
#     lat0 = st.number_input(
#         "Latitude (decimal degrees)",
#         min_value=lat_min,
#         max_value=lat_max,
#         value=float(np.clip(10.0, lat_min, lat_max)),
#         format="%.4f",
#         step=0.25,
#     )

# with c3:
#     lon0 = st.number_input(
#         "Longitude (decimal degrees)",
#         min_value=lon_min,
#         max_value=lon_max,
#         value=float(np.clip(40.0, lon_min, lon_max)),
#         format="%.4f",
#         step=0.05,
#     )

# with c4:
#     year = st.selectbox(
#         "Year (for point timeseries + daily map)",
#         options=year_list,
#         index=year_list.index(min(max(2000, year_min), year_max)),
#     )

# # Parameters (sidebar)
# st.sidebar.header("Onset / Wet spell parameters")

# wet_day_mm = st.sidebar.number_input(
#     "Wet day threshold (mm/day)", min_value=0.0, value=1.0, step=0.1
# )
# accum_days = st.sidebar.number_input(
#     "Accumulation window (days)", min_value=1, value=3, step=1
# )
# accum_mm = st.sidebar.number_input(
#     "Accumulation threshold (mm)", min_value=0.0, value=20.0, step=1.0
# )
# dry_spell_days = st.sidebar.number_input(
#     "Dry spell length (days)", min_value=1, value=7, step=1
# )
# lookahead_days = st.sidebar.number_input(
#     "Lookahead window (days)", min_value=1, value=21, step=1
# )

# st.sidebar.subheader("Search window (within selected year)")
# start_month = st.sidebar.selectbox("Search start month", list(range(1, 13)), index=4)
# start_day = st.sidebar.number_input(
#     "Search start day", min_value=1, max_value=31, value=15, step=1
# )
# end_month = st.sidebar.selectbox("Search end month", list(range(1, 13)), index=8)
# end_day = st.sidebar.number_input(
#     "Search end day", min_value=1, max_value=31, value=30, step=1
# )

# st.sidebar.subheader("Plot window")
# link_plot_to_search = st.sidebar.checkbox(
#     "Link plot window to search window", value=False
# )


# # Extract time series at nearest point for one year
# def extract_series_point_one_year() -> pd.DataFrame:
#     da = ds[var].sel({time_name: slice(f"{year}-01-01", f"{year}-12-31")})
#     da = da.sel({lat_name: lat0, lon_name: lon0}, method="nearest")

#     p_lat = float(da[lat_name].values)
#     p_lon = float(da[lon_name].values)
#     label = f"Point @ ({p_lat:.3f}, {p_lon:.3f})"

#     da = da.compute()

#     df_ = da.to_dataframe(name="rain").reset_index()
#     df_[time_name] = pd.to_datetime(df_[time_name])
#     df_ = df_.sort_values(time_name).reset_index(drop=True)

#     df_["label"] = label
#     df_["lat"] = p_lat
#     df_["lon"] = p_lon
#     return df_


# # Build point DF
# try:
#     df = extract_series_point_one_year()
# except Exception as e:
#     st.exception(e)
#     st.stop()

# # Rolling accumulation curve: forward window (matches detect_* definitions)
# s = pd.Series(df["rain"].values, dtype="float64")
# k = int(accum_days)
# df["rolling_accum"] = s.rolling(window=k, min_periods=k).sum().shift(-(k - 1))

# # 1) Wet-spell candidate: accumulation-only (but start day must be wet)
# wet_cand_date, wet_cand_idx = detect_wet_spell_candidate(
#     dates=df[time_name],
#     rain=df["rain"].values,
#     wet_day_mm=float(wet_day_mm),
#     accum_days=int(accum_days),
#     accum_mm=float(accum_mm),
#     start_month=int(start_month),
#     start_day=int(start_day),
#     end_month=int(end_month),
#     end_day=int(end_day),
# )

# # For display, use the start of the wet-run containing the candidate day (usually same day)
# wet_spell_start = wet_run_start_containing_index(
#     dates=df[time_name],
#     rain=df["rain"].values,
#     idx=wet_cand_idx,
#     wet_day_mm=float(wet_day_mm),
# )

# # 2) Onset: accumulation + dry-spell constraint (and start day must be wet)
# onset_date, onset_idx = detect_onset(
#     dates=df[time_name],
#     rain=df["rain"].values,
#     wet_day_mm=float(wet_day_mm),
#     accum_days=int(accum_days),
#     accum_mm=float(accum_mm),
#     dry_spell_days=int(dry_spell_days),
#     lookahead_days=int(lookahead_days),
#     start_month=int(start_month),
#     start_day=int(start_day),
#     end_month=int(end_month),
#     end_day=int(end_day),
# )

# df["onset_date"] = pd.NaT if onset_date is None else onset_date
# df["wet_spell_start"] = pd.NaT if wet_spell_start is None else wet_spell_start
# df["is_onset_day"] = (
#     False if onset_date is None else (df[time_name].dt.date == onset_date.date())
# )

# # Plot window handling
# plot_df = df
# if link_plot_to_search:
#     _, _, plot_start_date, plot_end_date = _year_window_indices(
#         df[time_name], int(start_month), int(start_day), int(end_month), int(end_day)
#     )
#     plot_df = df[
#         (df[time_name] >= plot_start_date) & (df[time_name] <= plot_end_date)
#     ].copy()

# # Map options
# if mode == "Map (Ethiopia)":
#     st.subheader("Map view options")
#     map_kind = st.radio(
#         "Map type",
#         [
#             "Seasonal mean (JJAS) over year range",
#             "Daily rainfall map (selected date)",
#             "Onset date map (selected year)",
#         ],
#         horizontal=True,
#     )

#     month_cmaps = {
#         "May": plt.cm.YlOrRd,
#         "Jun": plt.cm.YlGn,
#         "Jul": plt.cm.Greens,
#         "Aug": plt.cm.Blues,
#         "Sep": plt.cm.Purples,
#     }

#     eth = load_ethiopia_boundary()
#     eth_geom = eth.geometry.iloc[0]
#     eth_lon_min, eth_lat_min, eth_lon_max, eth_lat_max = map(float, eth.total_bounds)

#     if map_kind == "Seasonal mean (JJAS) over year range":
#         cA, cB = st.columns([1, 1])
#         with cA:
#             y0 = st.selectbox("Start year", options=year_list, index=0)
#         with cB:
#             y1 = st.selectbox("End year", options=year_list, index=len(year_list) - 1)

#         if y1 < y0:
#             st.error("End year must be ≥ start year.")
#             st.stop()

#         da_season = ds[var].sel({time_name: slice(f"{y0}-01-01", f"{y1}-12-31")})
#         da_season = da_season.sel(
#             {
#                 lat_name: slice(eth_lat_min, eth_lat_max),
#                 lon_name: slice(eth_lon_min, eth_lon_max),
#             }
#         )
#         da_season = da_season.where(
#             da_season[time_name].dt.month.isin([6, 7, 8, 9]), drop=True
#         )
#         da_map = da_season.mean(dim=time_name).compute()

#         lat_vals = da_map[lat_name].values
#         lon_vals = da_map[lon_name].values
#         mask = make_inside_mask(lat_vals, lon_vals, eth_geom)

#         Z_masked = np.where(mask, da_map.values, np.nan)

#         fig, ax = plt.subplots(figsize=(12, 6))
#         Lon, Lat = np.meshgrid(lon_vals, lat_vals)
#         pcm = ax.pcolormesh(Lon, Lat, Z_masked, shading="auto", cmap="Blues")
#         eth.boundary.plot(ax=ax, linewidth=2)
#         ax.set_title(f"Seasonal (JJAS) mean rainfall — {y0}–{y1}")
#         ax.set_xlabel("Longitude")
#         ax.set_ylabel("Latitude")
#         cb = fig.colorbar(pcm, ax=ax)
#         cb.set_label("Seasonal Mean Rainfall (mm/day)")
#         st.pyplot(fig)

#     elif map_kind == "Daily rainfall map (selected date)":
#         tmin = pd.to_datetime(ds[time_name].min().compute().values)
#         tmax = pd.to_datetime(ds[time_name].max().compute().values)

#         default_date = pd.Timestamp(year=year, month=7, day=15)
#         default_date = min(max(default_date, tmin), tmax)

#         date_sel = st.date_input(
#             "Select date for daily rainfall map",
#             value=default_date.to_pydatetime().date(),
#             min_value=tmin.date(),
#             max_value=tmax.date(),
#         )
#         date_sel = pd.Timestamp(date_sel)

#         da_day = ds[var].sel({time_name: date_sel}, method="nearest")
#         da_day = da_day.sel(
#             {
#                 lat_name: slice(eth_lat_min, eth_lat_max),
#                 lon_name: slice(eth_lon_min, eth_lon_max),
#             }
#         ).compute()

#         lat_vals = da_day[lat_name].values
#         lon_vals = da_day[lon_name].values
#         mask = make_inside_mask(lat_vals, lon_vals, eth_geom)

#         Z_masked = np.where(mask, da_day.values, np.nan)

#         fig, ax = plt.subplots(figsize=(12, 6))
#         Lon, Lat = np.meshgrid(lon_vals, lat_vals)
#         pcm = ax.pcolormesh(Lon, Lat, Z_masked, shading="auto", cmap="Blues")
#         eth.boundary.plot(ax=ax, linewidth=2)

#         ax.set_title(
#             f"Daily rainfall — {pd.to_datetime(da_day[time_name].values).date()}"
#         )
#         ax.set_xlabel("Longitude")
#         ax.set_ylabel("Latitude")
#         cb = fig.colorbar(pcm, ax=ax)
#         cb.set_label("Rainfall (mm/day)")
#         st.pyplot(fig)

#     else:
#         # Onset date map (selected year)
#         st.caption(
#             "Onset date map: runs the onset detection per pixel for the selected year "
#             "within Ethiopia bounds."
#         )

#         # ---- Build custom JJAS month-piecewise colormap (your snippet expects month_cmaps) ----
#         month_doys = {
#             "May": (121, 151),  # May 1 - May 31
#             "Jun": (152, 181),  # Jun 1 - Jun 30
#             "Jul": (182, 212),  # Jul 1 - Jul 31
#             "Aug": (213, 243),  # Aug 1 - Aug 31
#             "Sep": (244, 273),  # Sep 1 - Sep 30
#         }

#         # You can tweak these to match your slide palette exactly
#         month_cmaps = {
#             "May": plt.cm.YlOrRd,
#             "Jun": plt.cm.YlGn,
#             "Jul": plt.cm.Greens,
#             "Aug": plt.cm.Blues,
#             "Sep": plt.cm.Purples,
#         }

#         colors = []
#         bounds = []
#         N_per_month = 8  # smoothness inside each month

#         for month, cmap in month_cmaps.items():
#             d0, d1 = month_doys[month]
#             ramp = cmap(np.linspace(0.35, 0.95, N_per_month))
#             colors.extend(ramp)
#             bounds.extend(np.linspace(d0, d1, N_per_month, endpoint=False))

#         bounds.append(list(month_doys.values())[-1][1])

#         cmap_jjas = ListedColormap(colors, name="JJAS_piecewise")
#         norm_jjas = BoundaryNorm(bounds, cmap_jjas.N)

#         # ---- Subset data for this year + Ethiopia bbox ----
#         da_year = ds[var].sel({time_name: slice(f"{year}-01-01", f"{year}-12-31")})
#         da_year = da_year.sel(
#             {
#                 lat_name: slice(eth_lat_min, eth_lat_max),
#                 lon_name: slice(eth_lon_min, eth_lon_max),
#             }
#         )

#         # Pull time axis and convert to pandas datetime once
#         time_vals = pd.to_datetime(da_year[time_name].values)

#         # Convert full grid to memory once for speed (bbox is moderate)
#         da_year = da_year.compute()

#         lat_vals = da_year[lat_name].values
#         lon_vals = da_year[lon_name].values
#         mask = make_inside_mask(lat_vals, lon_vals, eth_geom)

#         # ---- Compute onset per pixel (simple loops; OK for moderate grids) ----
#         # Output is day-of-year (NaN if no onset detected)
#         onset_doy = np.full((len(lat_vals), len(lon_vals)), np.nan, dtype=float)

#         for i in range(len(lat_vals)):
#             for j in range(len(lon_vals)):
#                 if not mask[i, j]:
#                     continue

#                 rain_series = da_year.isel({lat_name: i, lon_name: j}).values

#                 od, _ = detect_onset(
#                     dates=pd.Series(time_vals),
#                     rain=rain_series,
#                     wet_day_mm=float(wet_day_mm),
#                     accum_days=int(accum_days),
#                     accum_mm=float(accum_mm),
#                     dry_spell_days=int(dry_spell_days),
#                     lookahead_days=int(lookahead_days),
#                     start_month=int(start_month),
#                     start_day=int(start_day),
#                     end_month=int(end_month),
#                     end_day=int(end_day),
#                 )

#                 if od is not None:
#                     onset_doy[i, j] = int(pd.Timestamp(od).dayofyear)

#         # ---- Plot ----
#         fig, ax = plt.subplots(figsize=(12, 6))
#         Lon, Lat = np.meshgrid(lon_vals, lat_vals)

#         # IMPORTANT: use cmap_jjas + norm_jjas
#         pcm = ax.pcolormesh(
#             Lon,
#             Lat,
#             onset_doy,
#             shading="auto",
#             cmap=cmap_jjas,
#             norm=norm_jjas,
#         )
#         eth.boundary.plot(ax=ax, linewidth=2)

#         ax.set_title(f"Onset dates (day-of-year) — {year}")
#         ax.set_xlabel("Longitude")
#         ax.set_ylabel("Latitude")

#         cb = fig.colorbar(pcm, ax=ax)
#         cb.set_label("Onset day-of-year (JJAS)")

#         st.pyplot(fig)

# # -----------------------------
# # Point timeseries plot
# # -----------------------------

# st.subheader("Point timeseries: rainfall, rolling accumulation, wet spell start, onset")

# fig, ax = plt.subplots(figsize=(14, 6))
# ax.plot(
#     plot_df[time_name],
#     plot_df["rain"],
#     marker="o",
#     linewidth=1.5,
#     label="Daily rainfall",
# )

# ax.plot(
#     plot_df[time_name],
#     plot_df["rolling_accum"],
#     linewidth=2.0,
#     alpha=0.35,
#     linestyle=":",
#     label=f"{int(accum_days)}-day rolling sum (forward window)",
# )

# ax.axhline(
#     float(wet_day_mm),
#     linestyle="--",
#     color="gray",
#     linewidth=1.5,
#     label=f"Wet day ≥ {float(wet_day_mm):g} mm/day",
# )

# if wet_spell_start is not None:
#     ax.axvline(
#         wet_spell_start,
#         linestyle="--",
#         linewidth=2.0,
#         color="orange",
#         label=f"Wet spell start: {wet_spell_start.date()}",
#     )

# if onset_date is not None:
#     ax.axvline(
#         onset_date,
#         linewidth=2.5,
#         color="red",
#         label=f"Onset: {onset_date.date()}",
#     )

# ax.set_ylabel("Rainfall (mm/day)")
# ax.set_xlabel("Date")
# ax.set_title(f"Rainfall time series — {df['label'].iloc[0]} — {year}")
# ax.legend()
# st.pyplot(fig)


# st.subheader("Detected dates")

# cL, _ = st.columns([1, 2])
# with cL:
#     if wet_spell_start is None:
#         st.info("Wet spell candidate not detected with the current parameters.")
#     else:
#         st.success(f"Wet spell start (accum-only): {wet_spell_start.date()}")

#     if onset_date is None:
#         st.warning("No onset detected with the current parameters.")
#     else:
#         st.success(f"Onset (accum + dry-spell rule): {onset_date.date()}")

# with st.expander("Show extracted time series data", expanded=False):
#     show_cols = [
#         time_name,
#         "rain",
#         "rolling_accum",
#         "lat",
#         "lon",
#         "wet_spell_start",
#         "onset_date",
#         "is_onset_day",
#     ]
#     st.dataframe(
#         df[show_cols].rename(columns={time_name: "date"}), use_container_width=True
#     )

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
    Piecewise monthly colormap for DOY:
      May (121–151), Jun (152–181), Jul (182–212), Aug (213–243), Sep (244–273)
    Returns (cmap, norm, tick_positions, tick_labels).
    """
    month_doys = {
        "May": (121, 151),
        "Jun": (152, 181),
        "Jul": (182, 212),
        "Aug": (213, 243),
        "Sep": (244, 273),
    }

    # Use built-in sequential colormaps per month (adjust if you have specific ones)
    month_cmaps = {
        "May": plt.get_cmap("YlOrBr"),
        "Jun": plt.get_cmap("YlOrRd"),
        "Jul": plt.get_cmap("YlGn"),
        "Aug": plt.get_cmap("GnBu"),
        "Sep": plt.get_cmap("PuBu"),
    }

    colors = []
    bounds = []

    N_per_month = 8  # smoothness inside each month

    for month in ["May", "Jun", "Jul", "Aug", "Sep"]:
        d0, d1 = month_doys[month]
        cmap = month_cmaps[month]

        ramp = cmap(np.linspace(0.35, 0.95, N_per_month))
        colors.extend(ramp)
        bounds.extend(np.linspace(d0, d1, N_per_month, endpoint=False))

    bounds.append(month_doys["Sep"][1])  # final bound

    cmap_jjas = ListedColormap(colors, name="JJAS_piecewise")
    norm_jjas = BoundaryNorm(bounds, cmap_jjas.N)

    # helpful labeled ticks (month starts)
    tick_positions = [121, 152, 182, 213, 244, 273]
    tick_labels = ["May 01", "Jun 01", "Jul 01", "Aug 01", "Sep 01", "Sep 30"]

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
        eth.boundary.plot(ax=ax, linewidth=2)

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
        eth.boundary.plot(ax=ax, linewidth=2)

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
        eth.boundary.plot(ax=ax, linewidth=2)

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
