# CETD — Rainfall Onset Explorer (Ethiopia)

Interactive research tool for detecting and visualising rainfall onset and wet spells across Ethiopia using CHIRPS and/or ENACTS NetCDF data, and/or EMI station CSV data.

The app supports:
- Multi-dataset selection (CHIRPS, ENACTS, EMI Stations — individually or in combination)
- Point-level rainfall timeseries analysis with interactive Plotly charts
- Automated onset detection based on accumulation + dry-spell constraints
- Wet-spell candidate identification
- Seasonal and daily rainfall maps (Ethiopia-wide)
- Pixel-level onset and wet spell DOY maps (single year or multi-year aggregation)
- EMI station timeseries with onset/wet spell detection

Built for climate onset diagnostics and exploratory research workflows.

---

## Project Structure

CETD/
│
├── app.py                                        # Main Streamlit application
├── run_onset_app.py                              # One-command launcher (creates venv, installs deps)
├── make_ethiopia_geojson.py                      # Generates Ethiopia boundary GeoJSON
├── RF_Station_Grid_Format.csv                    # EMI station daily rainfall data
├── chirps_jjas_seasonal_mask_ethiopia_0p25.nc    # Optional Ethiopia JJAS mask
│
├── data/
│   └── ethiopia.geojson                          # Ethiopia boundary (Natural Earth derived)
│
├── CHIRPS/
│   └── *.nc                                      # CHIRPS daily rainfall NetCDF files
│
├── ENACTS/
│   └── *.nc                                      # ENACTS daily rainfall NetCDF files
│
├── read_nc.py                                    # Inspect NetCDF datasets (variables, coords, dims)
│
├── requirements.txt
└── README.md

---

## Data Sources

**Gridded rainfall datasets**
- CHIRPS daily precipitation (NetCDF) — coordinate names: `latitude` / `longitude`
- ENACTS daily rainfall (NetCDF) — coordinate names: `lat` / `lon`
- Both datasets use variable name `precip`
- Spatial coverage: Ethiopia grid subset

**EMI Station data (`RF_Station_Grid_Format.csv`)**
- Daily rainfall for 26 Ethiopian stations, 1990–2020
- Format: rows = dates (`YYYYMMDD`), columns = station names
- First rows: `LON` (longitude), `DAILY/LAT` (latitude), then daily data
- Missing values encoded as `-99` (converted to NaN on load)
- Station coverage varies — some stations have >50% missing data in certain years

**Boundary**
- Natural Earth Admin-0 countries
- Extracted via `geopandas`, dissolved to single geometry
- Saved locally as `data/ethiopia.geojson`

---

## Features

### 1. Dataset Selection

The app supports four dataset combinations, each driving a different workflow:

| Selection | Timeseries | Map view |
|---|---|---|
| CHIRPS only | ✅ Point timeseries | ✅ Ethiopia-wide maps |
| ENACTS only | ✅ Point timeseries | ✅ Ethiopia-wide maps |
| CHIRPS + ENACTS | ✅ Both on same chart | ✅ Side-by-side + difference map |
| EMI Stations (± grids) | ✅ Station (+ nearest grid cell) | ❌ Not available |

---

### 2. Select View(s)

When CHIRPS and/or ENACTS are selected (without EMI Stations), the user can independently toggle:

- **Timeseries** — point-level chart for a selected lat/lon
- **Map (Ethiopia)** — country-wide spatial map

**🔗 Link Timeseries and Map Time Period** — when both views are active, a link toggle is shown. When linked, a single shared year/year-range selection applies to both views. When unlinked, independent time period controls appear for each view.

> Point selection (lat/lon) only appears when the Timeseries view is active — the map always shows the full country.

---

### 3. Timeseries Explorer

For a selected grid cell (lat/lon) or EMI station, the timeseries chart shows:

**Single year mode**
- Daily rainfall with a dot at each data point
- Wet spell start date (dashed vertical line)
- Onset date (solid vertical line)
- Wet-day threshold line (dotted)

**Year range mode**
- Mean or median climatology (rainfall by day-of-year) — with a dot at each DOY
- Wet spell start and onset derived from the climatology curve
- Wet-day threshold line

**Interactive legend (Plotly)**
- Click any legend item to toggle it on/off
- Double-click to isolate a single series
- All series (rainfall, wet spell, onset, threshold) are independently togglable

**Color scheme**

| Dataset | Rainfall | Wet spell | Onset |
|---|---|---|---|
| CHIRPS | Steel blue | Blue | Navy |
| ENACTS | Dark orange | Red | Dark red |
| EMI Stations | Green | Lime green | Dark green |

---

### 4. Map Visualisations

All maps are clipped to the Ethiopia boundary. An optional `.nc` seasonal mask (`chirps_jjas_seasonal_mask_ethiopia_0p25.nc`) can be applied inline.

**Map types**

| Type | Description |
|---|---|
| Seasonal mean rainfall (JJAS) | Mean rainfall over June–September for selected year(s), Blues colormap |
| Daily rainfall map | Rainfall on a single selected date |
| Wet spell date (DOY) | First wet spell start DOY — single year or multi-year aggregation |
| Onset date (DOY) | Rainfall onset DOY — single year or multi-year aggregation |

**Year range / aggregation**
- DOY maps support Mean or Median aggregation across a year range
- CHIRPS + ENACTS selection shows three columns: CHIRPS | ENACTS | Difference
- Difference map uses diverging colormap (RdBu_r); DOY differences clipped to ±30 days
- A warning is shown if grid shapes differ after Ethiopia clipping (difference map disabled)

---

### 5. Onset & Wet Spell Detection

Detection parameters are configurable in the sidebar and apply to all datasets and views.

| Parameter | Default | Description |
|---|---|---|
| Wet day threshold | 1.0 mm/day | Minimum rainfall to count as a wet day |
| Accumulation window | 3 days | Window for checking accumulated rainfall |
| Accumulation threshold | 20.0 mm | Minimum accumulation to trigger a candidate |
| Dry spell length | 7 days | Consecutive dry days that falsify an onset |
| Lookahead window | 21 days | Days ahead to check for a disqualifying dry spell |
| Search start | Jan 1 | Earliest date to search for onset/wet spell |
| Search end | Dec 31 | Latest date to search for onset/wet spell |

Detection runs on both raw daily data (single year) and climatology curves (year range). If no onset or wet spell is found within the search window, the corresponding line is simply omitted from the chart.

---

## Running the Project

From the project root:

$ python run_onset_app.py
# or
$ python3 run_onset_app.py

This will create a virtual environment, install all dependencies, and launch the Streamlit app automatically.

To run manually after setup: 
$ streamlit run app.py

---

## Requirements

Key dependencies (see `requirements.txt` for full list):

- `streamlit`
- `xarray`, `netCDF4`, `dask`
- `numpy`, `pandas`
- `matplotlib`
- `plotly`
- `geopandas`, `shapely`