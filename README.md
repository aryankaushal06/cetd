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
├── onset_app.py # Main Streamlit app
├── run_onset_app.py # One-command launcher (creates venv, installs deps)
├── make_ethiopia_geojson.py # Generates Ethiopia boundary GeoJSON
│
├── data/
│ └── ethiopia.geojson # Ethiopia boundary (Natural Earth derived)
│
├── obs_subset_ethiopia/
│ └── *.nc # CHIRPS rainfall NetCDF files
│
├── read_nc.py # to read the data of the .nc files
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

---

## Features

### 1) Point Time-Series Explorer

For any lat/lon grid cell and year:

- daily rainfall
- forward rolling accumulation window
- wet spell candidate date
- final onset date

Onset definition uses:
- accumulation threshold  
- minimum wet-day rainfall  
- look-ahead dry spell constraint  

---

### 2) Map Visualizations

#### Seasonal Mean Rainfall
- JJAS average
- multi-year range
- Ethiopia mask applied
- `Blues` colormap

#### Daily Rainfall Map
- selected date
- Ethiopia clipped grid
- consistent rainfall legend

#### (Upcoming)
- per-pixel onset DOY map using JJAS custom colormap

---

# Running project:

From project root:
$ python run_onset_app.py # or python3 run_onset_app.py