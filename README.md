# CETD — Rainfall Onset Explorer (Ethiopia)

Interactive research tool for detecting and visualising rainfall onset and wet spells across Ethiopia using gridded NetCDF datasets (CHIRPS, ENACTS, IMERG, ERA5) and/or EMI rain gauge station data.

The app supports:
- Multi-dataset selection (CHIRPS, ENACTS, IMERG, ERA5, EMI Stations — individually or in combination)
- Point-level rainfall timeseries analysis with interactive Plotly charts
- Automated onset detection based on accumulation + dry-spell constraints (ICPAC algorithm)
- Wet-spell candidate identification
- Seasonal and daily rainfall maps (Ethiopia-wide)
- Pixel-level onset and wet spell DOY maps (single year or multi-year aggregation)
- Pairwise dataset difference maps with automatic resolution resampling
- EMI station timeseries with onset/wet spell detection and interactive station picker map

Built for climate onset diagnostics and exploratory research workflows.

---

## Project Structure

```
CETD/
│
├── app.py                                        # Main Streamlit application
├── run_onset_app.py                              # One-command launcher (creates venv, installs deps)
├── make_ethiopia_geojson.py                      # Generates Ethiopia boundary GeoJSON
├── RF_Station_Grid_Format.csv                    # EMI station daily rainfall data
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
├── read_nc.py                                    # Inspect NetCDF datasets (variables, coords, dims)
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

> **IMERG / ERA5 note:** files produced by CDO remapping often have corrupted time coordinates. The app automatically corrects these at load time by extracting the year from each filename and reassigning a clean daily time axis.

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

The app supports any combination of the four gridded datasets plus EMI Stations. The selected combination determines which views are available:

| Selection | Timeseries | Map view |
|---|---|---|
| Any single grid dataset | ✅ Point timeseries | ✅ Ethiopia-wide map |
| Multiple grid datasets | ✅ All on same chart | ✅ Individual + pairwise difference maps |
| EMI Stations only | ✅ Station timeseries | ❌ Not available |
| EMI Stations + grids | ✅ Station vs nearest grid cell | ❌ Not available |

When datasets with different temporal ranges are combined, the year selector is automatically limited to the overlapping period and a banner explains the restriction with each dataset's individual range.

---

### 2. Select View(s)

When one or more grid datasets are selected (without EMI Stations), the user can independently toggle:

- **Timeseries** — point-level chart for a selected lat/lon
- **Map (Ethiopia)** — country-wide spatial map

**🔗 Link Timeseries and Map Time Period** — appears when both views are active. When linked, one shared year/year-range selection applies to both. When unlinked, independent time period controls appear for each view.

> Point selection (lat/lon) only appears when the Timeseries view is active — the map always shows the full country and does not require a point.

---

### 3. Timeseries Explorer

For every dataset series shown, the chart always includes:
- **Daily rainfall** — line + dot at each data point
- **Wet spell start** — dashed vertical line at the first qualifying wet spell
- **Onset date** — solid vertical line at the detected onset
- **Wet-day threshold** — single dotted horizontal line (shared across all datasets, appears once in legend)

**Single year mode** — raw daily values for the selected year.

**Year range mode** — mean or median climatology by day-of-year (DOY), with wet spell and onset derived from the climatology curve.

If no wet spell or onset is detected within the search window for a given series, the corresponding line is omitted silently.

**Interactive legend (Plotly)**
- Click any item to toggle it on/off
- Double-click to isolate a single series
- All series are independently togglable — rainfall, wet spell, onset, and threshold

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

All maps are clipped to the Ethiopia boundary using `data/ethiopia.geojson`. An optional `.nc` seasonal mask (`chirps_jjas_seasonal_mask_ethiopia_0p25.nc`) can be toggled per map view.

**Map types**

| Type | Description |
|---|---|
| Seasonal mean rainfall (JJAS) | Mean daily rainfall over June–September for selected year(s), Blues colormap |
| Daily rainfall map | Rainfall on a single user-selected date |
| Wet spell date (DOY) | First wet spell start DOY — single year or multi-year aggregation |
| Onset date (DOY) | Rainfall onset DOY — single year or multi-year aggregation |

**Layout**
- Maps are displayed in rows of at most 3 columns; overflow wraps to the next row
- When multiple datasets are selected, individual dataset maps appear first, followed by all pairwise difference maps
- Difference maps use a diverging colormap (RdBu_r); DOY differences are clipped to ±30 days
- If two datasets have different resolutions (e.g. CHIRPS 0.05° vs IMERG 0.25°), the finer grid is automatically resampled onto the coarser grid using bilinear interpolation; a caption below the difference map notes which dataset was resampled

**Year range / aggregation**
- DOY maps support Mean or Median aggregation across a year range
- Shared colour scale applied across individual dataset maps when comparing rainfall

---

### 5. EMI Station Picker

When EMI Stations are selected, an interactive map of Ethiopia is shown with all 26 station locations marked as red triangles. Station labels are grey by default. Clicking a station selects it — its label turns black and bold. The selected station persists across reruns via Streamlit session state. No station is selected by default; the timeseries view waits until the user clicks one.

The map is styled to match the weather maps: white background, black Ethiopia boundary, labelled lat/lon axes, drawn directly from `data/ethiopia.geojson` with no external tile library.

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
| Search start | Jan 1 | Earliest date to search for onset/wet spell |
| Search end | Dec 31 | Latest date to search for onset/wet spell |

Detection runs on raw daily data (single year mode) and on climatology curves (year range mode). Pixel-level DOY maps run detection independently at every grid cell.

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