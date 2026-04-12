# CETD — Rainfall Onset Explorer (Ethiopia)

Interactive research tool for detecting and visualising rainfall onset and wet spells across Ethiopia using gridded NetCDF datasets (CHIRPS, ENACTS, IMERG, ERA5) and/or EMI rain gauge station data.

The app supports:
- Multi-dataset selection (CHIRPS, ENACTS, IMERG, ERA5, EMI Stations — individually or in combination)
- Point-level rainfall timeseries analysis with interactive Plotly charts
- Automated onset detection based on accumulation + dry-spell constraints (ICPAC algorithm)
- Wet-spell candidate identification
- Seasonal and daily rainfall maps (Ethiopia-wide)
- Pixel-level onset and wet spell date maps (single year or multi-year aggregation)
- Pairwise dataset difference maps with automatic grid alignment
- EMI station timeseries with onset/wet spell detection and interactive station picker map
- Point-level statistics: onset date scatter and onset probability distribution

Built for climate onset diagnostics and exploratory research workflows.

---

## Project Structure

```
CETD/
│
├── app.py                                        # Main Streamlit application
├── check_resolution.py                           # Inspect and compare resolution of all datasets
├── run_onset_app.py                              # One-command launcher (creates venv, installs deps)
├── make_ethiopia_geojson.py                      # Generates Ethiopia boundary GeoJSON
├── RF_Station_Grid_Format.csv                    # EMI station daily rainfall data (43 stations)
├── chirps_jjas_seasonal_mask_ethiopia_0p25.nc    # Optional Ethiopia JJAS seasonal mask
│
├── data/
│   └── ethiopia.geojson                          # Ethiopia boundary (Natural Earth derived)
│
├── CHIRPS/
│   └── *.nc                                      # CHIRPS daily rainfall NetCDF files
│
├── ENACTS/
│   └── *.nc                                      # ENACTS daily rainfall NetCDF files (May–Oct only)
│
├── IMERG/
│   └── imerg_<year>_ethiopia_0p25.nc             # IMERG daily rainfall NetCDF files
│
├── ERA5/
│   └── era5_<year>_ethiopia_0p25.nc              # ERA5 daily rainfall NetCDF files
│
├── requirements.txt
└── README.md
```

---

## Data Sources

**Gridded rainfall datasets**

| Dataset | Coord names | Variable | Resolution | Period | Notes |
|---|---|---|---|---|---|
| CHIRPS | `latitude` / `longitude` | `precip` | 0.05° | 1981–present | Full year |
| ENACTS | `lat` / `lon` | `precip` | 0.25° | 2000–2022 | **May–Oct only** |
| IMERG | `lat` / `lon` | `precip` | 0.25° | 2003–2024 | CDO time-fix applied on load |
| ERA5 | `lat` / `lon` | `precip` | 0.25° | 2004–2024 | CDO time-fix applied on load |

> **ENACTS note:** data outside May–October will appear as missing. JJAS maps and onset/wet spell detection are unaffected since they operate within this window, but full-year timeseries and non-JJAS maps will have gaps.

> **IMERG / ERA5 note:** files produced by CDO remapping can have corrupted time coordinates. The app automatically corrects these at load time by extracting the year from each filename and assigning a clean daily time axis.

**EMI Station data (`RF_Station_Grid_Format.csv`)**
- Daily rainfall for 43 Ethiopian stations (merged from two source CSVs), 1990–2022
- Format: rows = dates (`YYYYMMDD`), columns = station names
- First rows: `LON` (longitude), `DAILY/LAT` (latitude), then daily data
- Missing values encoded as `-99` (converted to NaN on load)
- Station coverage varies — some stations have significant missing data in certain years

**Boundary**
- Natural Earth Admin-0 countries
- Extracted via `geopandas`, dissolved to single geometry
- Saved locally as `data/ethiopia.geojson`

---

## Resolution Check

Run `check_resolution.py` from the project root to inspect and compare the spatial resolution, coordinate names, grid dimensions, and time range of all four gridded datasets, plus the EMI station CSV:

```bash
python check_resolution.py
```

The script prints a comparison table and flags any resolution mismatches between datasets. All four gridded datasets are expected to share the same 0.25° resolution.

---

## Features

### 1. Dataset Selection

The app supports any combination of the four gridded datasets plus EMI Stations:

| Selection | Timeseries | Map view |
|---|---|---|
| Any single grid dataset | ✅ Point timeseries | ✅ Ethiopia-wide map |
| Multiple grid datasets | ✅ All on same chart | ✅ Individual + pairwise difference maps |
| EMI Stations only | ✅ Station timeseries | ❌ Not available |
| EMI Stations + grids | ✅ Station vs nearest grid cell | ❌ Not available |

When datasets with different temporal ranges are combined, the year selector is automatically limited to the overlapping period and a banner explains the restriction with each dataset's individual range.

---

### 2. Select View(s)

When one or more grid datasets are selected (without EMI Stations), three views can be independently toggled:

- **Timeseries** — point-level chart for a selected lat/lon
- **Map (Ethiopia)** — country-wide spatial map
- **Statistics** — point-level onset statistics over a chosen year range

**🔗 Link Timeseries and Map Time Period** — appears when both Timeseries and Map are active. When linked, one shared year/year-range selection applies to both. When unlinked, independent time period controls appear for each view.

> Point selection (lat/lon) only appears when the Timeseries view is active — the map always shows the full country.

---

### 3. Timeseries Explorer

For every dataset series shown, the chart always includes:
- **Daily rainfall** — line + dot at each data point
- **Wet spell start** — dashed vertical line at the first qualifying wet spell
- **Onset date** — solid vertical line at the detected onset
- **Wet-day threshold** — single dotted horizontal line shared across all datasets (appears once in legend)

**Single year mode** — raw daily values for the selected year.

**Year range mode** — mean or median climatology by day-of-year (DOY), with wet spell and onset derived from the climatology curve.

**Clip timeseries to search window** — toggle in the sidebar (under Search window) limits the x-axis to the search window dates, hiding out-of-season data. Can be turned off to show the full year.

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

### 4. Map Visualisations

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

**Year range / aggregation**
- DOY maps support Mean or Median aggregation across a year range

---

### 5. EMI Station Picker

When EMI Stations are selected, an interactive map of Ethiopia is shown with all 43 station locations marked as red triangles. Station labels are grey by default. Clicking a station selects it — its label turns black and bold. The selected station persists across reruns via Streamlit session state. No station is pre-selected; the timeseries view waits until the user clicks one.

The map matches the weather map style: white background, black Ethiopia boundary, labelled lat/lon axes, drawn from `data/ethiopia.geojson` with no external tile library.

---

### 6. Onset & Wet Spell Detection

All detection parameters are configurable in the sidebar and apply uniformly to all datasets and views.

| Parameter | Default | Description |
|---|---|---|
| Wet day threshold | 1.0 mm/day | Minimum rainfall to count as a wet day |
| Accumulation window | 3 days | Window over which rainfall is summed |
| Accumulation threshold | 20.0 mm | Minimum accumulation to trigger a candidate onset |
| Dry spell length | 7 days | Consecutive dry days that falsify a candidate |
| Lookahead window | 21 days | Days ahead to check for a disqualifying dry spell |
| Search start | **15 May** | Earliest date to search for onset/wet spell |
| Search end | **15 Oct** | Latest date to search for onset/wet spell |

Detection runs on raw daily data (single year mode) and on climatology curves (year range mode). Pixel-level date maps run detection independently at every grid cell. The wet spell walkback is floored at the search window start, preventing spurious early dates when data begins before the search window (e.g. ENACTS starting May 1).

---

### 7. Statistics

The Statistics view (toggled under Select View(s), default on) provides point-level onset analysis over a chosen year range. It requires the Timeseries view to be active so a point is selected. Two charts are available, each independently toggled:

**Onset date scatter** — onset DOY plotted year by year for each dataset at the selected point. A shaded band marks the search window. Years with no detected onset appear as gaps.

**Onset distribution** — probability of onset occurring within each 5-day bin across the search window, computed over the selected year range. Years with no detected onset are excluded from the denominator so probabilities sum to 1. Hover shows the exact date range, probability, and raw count per bin.

---

## Running the Project

From the project root:

```bash
python run_onset_app.py
# or
python3 run_onset_app.py
```

This creates a virtual environment, installs all dependencies, and launches the Streamlit app automatically.

To run manually after setup:

```bash
streamlit run app.py
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