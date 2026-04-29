# CETD — Rainfall Onset Explorer (Ethiopia)

Interactive research tool for detecting and visualising rainfall onset and wet spells across Ethiopia using gridded NetCDF datasets (CHIRPS, ENACTS, IMERG, ERA5), EMI rain gauge station data, and model forecast output (AIFS, AIFS-ENS, GenCast).

The app supports:
- Multi-dataset selection (CHIRPS, ENACTS, IMERG, ERA5, EMI Stations — individually or in combination)
- Point-level rainfall timeseries analysis with interactive Plotly charts
- Automated onset detection based on accumulation + dry-spell constraints (ICPAC algorithm)
- Wet-spell candidate identification
- Seasonal and daily rainfall maps (Ethiopia-wide)
- Pixel-level onset and wet spell date maps (single year or multi-year aggregation)
- Pairwise dataset difference maps with automatic grid alignment
- EMI station timeseries with onset/wet spell detection and interactive station picker map
- Point-level statistics: onset date scatter, onset probability distribution, and rainfall CDF
- Forecast timeseries from AIFS (deterministic), AIFS-ENS (25-member ensemble), and GenCast (32-member ensemble) with full onset/wet spell detection and ensemble fan visualisation

Built for climate onset diagnostics and exploratory research workflows.

---

## Project Structure

```
CETD/
│
├── app.py                  # Streamlit entry point: sidebar, tabs, all view branches
├── config.py               # All constants and the OnsetParams frozen dataclass
├── data.py                 # All I/O: open_folder, load_emi_csv, load_mask, coord helpers
├── detection.py            # Pure onset algorithm: detect_onset, detect_wet_spell, wet_spell_start
├── extract.py              # Cached extraction: extract_year, extract_clim, onset_series, rainfall_cdf
├── maps.py                 # Pixel-level map computation and matplotlib rendering helpers
├── charts.py               # Plotly figure builders: timeseries, ensemble fan, station map
├── forecast.py             # Forecast data loading and point extraction for all three models
│
├── run_app.py              # One-command launcher (creates .venv, installs deps, launches app)
├── check_forecasts.py      # Inspect forecast NetCDF structure (dimensions, variables, dates)
│
├── data/
│   └── ethiopia.geojson    # Ethiopia boundary (Natural Earth derived)
│
├── datasets/
│   ├── CHIRPS/             # CHIRPS daily rainfall NetCDF files
│   ├── ENACTS/             # ENACTS daily rainfall NetCDF files (May–Oct only)
│   ├── IMERG/              # IMERG daily rainfall NetCDF files
│   ├── ERA5/               # ERA5 daily rainfall NetCDF files
│   ├── aifs/               # AIFS deterministic forecast NetCDF files (one per year)
│   ├── aifs_ens/           # AIFS ensemble forecast NetCDF files (one per year)
│   ├── gencast/            # GenCast ensemble forecast NetCDF files (one per year)
│   ├── RF_Station_Grid_Format.csv
│   └── chirps_jjas_seasonal_mask_ethiopia_0p25.nc
│
├── requirements.txt
└── README.md
```

---

## Data Sources

### Observational gridded datasets

| Dataset | Coord names | Variable | Resolution | Period | Notes |
|---|---|---|---|---|---|
| CHIRPS | `latitude` / `longitude` | `precip` | 0.05° | 1981–present | Full year |
| ENACTS | `lat` / `lon` | `precip` | 0.25° | 2000–2022 | **May–Oct only** |
| IMERG | `lat` / `lon` | `precip` | 0.25° | 2003–2024 | CDO time-fix applied on load |
| ERA5 | `lat` / `lon` | `precip` | 0.25° | 2004–2024 | CDO time-fix applied on load |

> **ENACTS note:** data outside May–October will appear as missing. JJAS maps and onset/wet spell detection are unaffected since they operate within this window, but full-year timeseries and non-JJAS maps will have gaps.

> **IMERG / ERA5 note:** files produced by CDO remapping can have corrupted time coordinates. The app automatically corrects these at load time by extracting the year from each filename and assigning a clean daily time axis.

### EMI Station data (`datasets/RF_Station_Grid_Format.csv`)
- Daily rainfall for 43 Ethiopian stations, 1990–2022
- Format: rows = dates (`YYYYMMDD`), columns = station names
- First rows: `LON` (longitude), `DAILY/LAT` (latitude), then daily data
- Missing values encoded as `-99` (converted to NaN on load)
- Station coverage varies — some stations have significant missing data in certain years

### Forecast datasets

All three forecast models share the same spatial grid (0.25°, lat 3–15°N, lon 33–48°E) and the same time structure: 23 initialisation dates from May 1 to July 31 per year, spaced roughly every 4–6 days.

| Model | Variable | Lead time | Members | Period | Notes |
|---|---|---|---|---|---|
| AIFS | `tp` (mm) | 47 days (day 0–46) | 1 (deterministic) | 2000–2025 | `day` coordinate is 0-indexed |
| AIFS-ENS | `tp` (mm) | 47 days (day 0–46) | 25 (`number` 0–24) | 2000–2024 | |
| GenCast | `tp` | 45 days (day 1–45) | 32 (`number` 1–32) | 2003–2024 (missing 2006) | `day` coordinate is 1-indexed; no units attribute |

> **GenCast note at long lead times:** GenCast can produce very high precipitation values (50–70+ mm/day) beyond ~day 20 for Ethiopian locations, particularly during JJAS. This is likely a calibration bias of the global ML model rather than a real signal. Interpret long-range GenCast forecasts with caution.

### Boundary
- Natural Earth Admin-0 countries, dissolved to single geometry
- Saved locally as `data/ethiopia.geojson`

---

## Features

### Observations tab

#### 1. Dataset Selection

The app supports any combination of the four gridded datasets plus EMI Stations:

| Selection | Timeseries | Map view |
|---|---|---|
| Any single grid dataset | ✅ Point timeseries | ✅ Ethiopia-wide map |
| Multiple grid datasets | ✅ All on same chart | ✅ Individual + pairwise difference maps |
| EMI Stations only | ✅ Station timeseries | ❌ Not available |
| EMI Stations + grids | ✅ Station vs nearest grid cell | ❌ Not available |

When datasets with different temporal ranges are combined, the year selector is automatically limited to the overlapping period and a banner explains the restriction with each dataset's individual range.

---

#### 2. Select View(s)

When one or more grid datasets are selected (without EMI Stations), three views can be independently toggled:

- **Timeseries** — point-level chart for a selected lat/lon
- **Map (Ethiopia)** — country-wide spatial map
- **Statistics** — point-level onset statistics over a chosen year range

**🔗 Link Timeseries and Map Time Period** — appears when both Timeseries and Map are active. When linked, one shared year/year-range selection applies to both. When unlinked, independent time period controls appear for each view.

> Point selection (lat/lon) only appears when the Timeseries view is active — the map always shows the full country.

---

#### 3. Timeseries Explorer

For every dataset series shown, the chart always includes:
- **Daily rainfall** — line + dot at each data point
- **Wet spell start** — dashed vertical line at the first qualifying wet spell
- **Onset date** — solid vertical line at the detected onset
- **Wet-day threshold** — single dotted horizontal line shared across all datasets

**Single year mode** — raw daily values for the selected year.

**Year range mode** — mean or median climatology by day-of-year (DOY), with wet spell and onset derived from the climatology curve.

**Clip timeseries to search window** — sidebar toggle limits the x-axis to the search window dates. Can be turned off to show the full year.

If no wet spell or onset is detected within the search window, the corresponding line is omitted silently.

**Interactive legend (Plotly)**
- Click any item to toggle it on/off
- Double-click to isolate a single series

**Colour scheme**

| Dataset | Rainfall | Wet spell | Onset |
|---|---|---|---|
| CHIRPS | Steel blue | Blue | Navy |
| ENACTS | Dark orange | Red | Dark red |
| IMERG | Purple | Dark purple | Deep purple |
| ERA5 | Teal/cyan | Dark teal | Dark cyan |
| EMI Stations | Green | Lime green | Dark green |

---

#### 4. Map Visualisations

All maps are clipped to the Ethiopia boundary. An optional `.nc` seasonal mask can be toggled per map view.

**Map types**

| Type | Description |
|---|---|
| Seasonal mean rainfall (JJAS) | Mean daily rainfall over June–September, Blues colormap |
| Daily rainfall map | Rainfall on a single user-selected date |
| Wet spell date | First wet spell start date — single year or multi-year aggregation |
| Onset date | Rainfall onset date — single year or multi-year aggregation |

**Layout**
- Maps render in rows of at most 3 columns; overflow wraps to the next row
- Individual dataset maps appear first, followed by all pairwise difference maps
- Difference maps use a diverging colormap (RdBu_r); date differences are clipped to ±30 days
- All datasets are pre-aligned to a common reference grid before differencing, eliminating shape mismatch errors from minor grid offset differences

**Scale toggles** (multi-dataset only)
- **Shared scale across rainfall maps** — applies the same min/max colour scale to all individual rainfall maps
- **Shared scale across difference maps** — applies the same diverging scale to all difference maps (rainfall map types only; date difference maps always use a fixed ±30 day scale)

---

#### 5. EMI Station Picker

When EMI Stations are selected, an interactive map of Ethiopia is shown with all 43 station locations marked as red triangles. Clicking a station selects it — its label turns black and bold. The selected station persists across reruns via Streamlit session state.

---

#### 6. Statistics

The Statistics view provides point-level onset analysis over a chosen year range. Three charts are available, each independently toggled:

**Onset date scatter** — onset DOY plotted year by year for each dataset at the selected point. A shaded band marks the search window. Years with no detected onset appear as gaps.

**Onset distribution** — probability of onset occurring within each 5-day bin across the search window, computed over the selected year range. A bounding-box region mode is also available: onset is detected at every grid cell in the box and all detections are pooled together. Hover shows the exact date range, probability, and raw count per bin.

**Rainfall CDF** — empirical CDF of all daily rainfall values (including dry days) at the selected point over the stats year range, with a vertical line marking the wet-day threshold.

---

### Forecasts tab

#### 7. Forecast Model Selection

Select one or more forecast models (AIFS, AIFS-ENS, GenCast). The year selector is automatically limited to years available in all selected models.

---

#### 8. Initialisation Date Selection

Within the selected year, each model provides 23 initialisation dates from May 1 to July 31. Multiple initialisation dates can be selected simultaneously — each produces its own set of traces on the chart, allowing comparison of how forecasts evolve as the init date moves later into the season.

> A warning is shown if more than 5 model × init combinations are selected with individual members displayed, as this can slow rendering.

---

#### 9. Forecast Timeseries

The x-axis represents the forecast **valid date** (initialisation date + lead day). Each model × init date combination adds:

**Deterministic (AIFS)**
- Single rainfall line
- Wet spell start (dashed vertical line) and onset date (solid vertical line) derived from the forecast timeseries

**Ensemble (AIFS-ENS, GenCast)**
- **Individual member traces** (faded, toggleable) — one thin line per ensemble member
- **P10–P90 shaded band** — ensemble spread at each valid date
- **Median line** — bold line through the ensemble median
- **Per-member wet spell / onset markers** — faded vertical lines for every member that produces a detection, plus a bold median marker
- Wet-day threshold horizontal line

The search window clip toggle (sidebar) applies to the forecast chart x-axis as well.

**Colour scheme**

| Model | Rainfall / median | Fan | Wet spell | Onset |
|---|---|---|---|---|
| AIFS | Orange | — | Dark orange | Dark brown-orange |
| AIFS-ENS | Purple | Light purple | Dark purple | Deep purple |
| GenCast | Green | Light green | Dark green | Very dark green |

---

#### 10. Onset / Wet Spell Summary Table

Below the forecast chart, a table summarises detection results for every model × init date combination:

| Column | Deterministic | Ensemble |
|---|---|---|
| Wet spell start | Detected date or — | Median across members |
| Onset date | Detected date or — | Median across members |
| Members w/ onset | — | N / total members |

---

### Onset & Wet Spell Detection (all tabs)

All detection parameters are configurable in the sidebar and apply uniformly across all datasets, views, and forecast models.

| Parameter | Default | Description |
|---|---|---|
| Wet day threshold | 1.0 mm/day | Minimum rainfall to count as a wet day |
| Accumulation window | 3 days | Window over which rainfall is summed |
| Accumulation threshold | 20.0 mm | Minimum accumulation to trigger a candidate onset |
| Dry spell length | 7 days | Consecutive dry days that falsify a candidate |
| Lookahead window | 21 days | Days ahead to check for a disqualifying dry spell |
| Search start | **15 May** | Earliest date to search for onset/wet spell |
| Search end | **15 Oct** | Latest date to search for onset/wet spell |

Detection runs on raw daily data (single year mode), climatology curves (year range mode), pixel grids (map mode), and individual forecast member timeseries (forecast mode). The wet spell walkback is floored at the search window start, preventing spurious early dates.

---

## Running the Project

```bash
python run_app.py
# or
python3 run_app.py
```

This creates a `.venv` virtual environment, installs all dependencies, and launches the Streamlit app automatically. Run it once after cloning; subsequent launches can use:

```bash
.venv/bin/streamlit run app.py
```

To inspect the structure of the forecast NetCDF files before integrating new data:

```bash
python check_forecasts.py
```

---

## Requirements

Key dependencies (see `requirements.txt` for full list):

- `streamlit`
- `xarray`, `netCDF4`, `dask`
- `numpy`, `pandas`, `scipy`
- `matplotlib`
- `plotly`
- `geopandas`, `shapely`
