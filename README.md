# cetd
# CETD — Rainfall Onset Explorer (Ethiopia)

Interactive research tool for detecting and visualizing rainfall onset and wet spells across Ethiopia using CHIRPS NetCDF data.

The app supports:
- point-level rainfall time-series analysis  
- automated onset detection based on accumulation + dry-spell constraints  
- wet-spell candidate identification  
- seasonal and daily rainfall maps  
- (planned) pixel-level onset maps  

Built for climate onset diagnostics and exploratory research workflows.

---

# Project Structure
CETD/
│
├── onset_app.py                  # Main Streamlit application
├── run_onset_app.py              # One-command launcher (creates venv, installs deps)
├── make_ethiopia_geojson.py      # Generates Ethiopia boundary GeoJSON
│
├── data/
│   └── ethiopia.geojson          # Ethiopia boundary (Natural Earth derived)
│
├── CHIRPS/
│   └── *.nc                      # CHIRPS rainfall NetCDF files
│
├── ENACTS/
│   └── *.nc                      # ENACTS rainfall NetCDF files
│
├── read_nc.py                    # Inspect NetCDF datasets (variables, coords, dims)
│
├── requirements.txt
└── README.md

## Data Source

Rainfall data:
- CHIRPS daily precipitation (NetCDF)
- Spatial subset: Ethiopia

Boundary:
- Natural Earth Admin-0 countries
- Extracted using `geopandas`
- Saved locally as `data/ethiopia.geojson`

## Data Sources

Rainfall datasets:
- CHIRPS daily precipitation (NetCDF)
- ENACTS rainfall dataset (NetCDF)

Spatial coverage: Ethiopia grid subset

Boundary:
- Natural Earth Admin-0 countries
- Extracted via geopandas
- Saved locally as: data/ethiopia.geojson
---

## Features

### 1) Multi-Dataset Support

The app can run:
- CHIRPS only
- ENACTS only
- CHIRPS + ENACTS side-by-side comparison

All maps and time-series update dynamically based on dataset selection.

### 2) Point Time-Series Explorer

For any selected grid cell (lat/lon):

Single-year mode
- Displays: daily rainfall, wet spell start, onset date, wet-day threshold line

Multi-year mode
- Displays: daily climatology (mean rainfall by day-of-year), no onset markers (not meaningful for climatology)

Color conventions

CHIRPS:
- daily rainfall → blue (dots)
- wet spell start → purple
- onset → pink

ENACTS:
- daily rainfall → orange (dots)
- wet spell start → red
- onset → yellow
---

### 2) Map Visualizations

- Interactive spatial maps across Ethiopia.
- Year selection modes: Single year, Multi-year range, Aggregation modes (multi-year DOY maps)
- Median, Mean

Map types
- Seasonal Mean Rainfall (JJAS): averaged over selected years, Ethiopia mask applied, Blues colormap
- Daily Rainfall Map: selectable date, clipped to Ethiopia, dataset-aware rainfall grid
- Wet Spell Date (DOY): single-year OR multi-year, piecewise seasonal colormap (May–Sep), Ethiopia masked
- Onset Date (DOY): single-year OR multi-year, median or mean aggregation, Ethiopia masked

---

# Running project:

From project root:
$ python run_onset_app.py
or 
$ python3 run_onset_app.py