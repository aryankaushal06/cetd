from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

LINEWIDTH_ETH = 2.0

DATASET_FOLDERS: Dict[str, str] = {
    "CHIRPS": "datasets/CHIRPS",
    "ENACTS": "datasets/ENACTS",
    "IMERG": "datasets/IMERG",
    "ERA5": "datasets/ERA5",
}

RAIN_VAR_BY_KEY: Dict[str, str] = {
    "CHIRPS": "precip",
    "ENACTS": "precip",
    "IMERG": "precip",
    "ERA5": "precip",
}

RAIN_VAR = "precip"
FILE_GLOB = "*.nc"
CHUNKSIZE = 365

APP_DIR = Path(__file__).resolve().parent
STATION_KEY = "EMI Stations"
STATION_CSV_PATH = APP_DIR / "datasets" / "RF_Station_Grid_Format.csv"
DEFAULT_MASK_NC_PATH = (
    APP_DIR / "datasets" / "chirps_jjas_seasonal_mask_ethiopia_0p25.nc"
)

DATASET_COLORS: Dict[str, Dict[str, str]] = {
    "CHIRPS": {"rain": "steelblue", "wet": "#1a1aff", "onset": "#000080"},
    "ENACTS": {"rain": "darkorange", "wet": "#cc0000", "onset": "#660000"},
    "IMERG": {"rain": "#9467bd", "wet": "#6a0dad", "onset": "#3b0066"},
    "ERA5": {"rain": "#17becf", "wet": "#0a6e7a", "onset": "#003d44"},
    STATION_KEY: {"rain": "#2ca02c", "wet": "#7fff00", "onset": "#006400"},
}

SEASONAL_COVERAGE_NOTE: Dict[str, str] = {
    "ENACTS": "May–Oct only",
}

FORECAST_FOLDERS: Dict[str, str] = {
    "AIFS": "datasets/aifs",
    "AIFS_ENS": "datasets/aifs_ens",
    "GENCAST": "datasets/gencast",
}

FORECAST_IS_ENS: Dict[str, bool] = {
    "AIFS": False,
    "AIFS_ENS": True,
    "GENCAST": True,
}

FORECAST_COLORS: Dict[str, Dict[str, str]] = {
    "AIFS": {
        "rain": "#e67e22",
        "fan": "rgba(230,126,34,0.15)",
        "wet": "#d35400",
        "onset": "#a04000",
    },
    "AIFS_ENS": {
        "rain": "#8e44ad",
        "fan": "rgba(142,68,173,0.15)",
        "wet": "#7d3c98",
        "onset": "#6c3483",
    },
    "GENCAST": {
        "rain": "#27ae60",
        "fan": "rgba(39,174,96,0.15)",
        "wet": "#1e8449",
        "onset": "#196f3d",
    },
}


@dataclass(frozen=True)
class OnsetParams:
    wet_day_mm: float
    accum_days: int
    accum_mm: float
    dry_spell_days: int
    lookahead_days: int
    start_month: int
    start_day: int
    end_month: int
    end_day: int
