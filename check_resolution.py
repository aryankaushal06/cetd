"""
check_resolution.py
-------------------
Run this from the CETD project root to inspect the spatial resolution,
coordinate names, variable names, and time range of all four gridded
datasets, plus the EMI station CSV.

Usage:
    python check_resolution.py

What to look for:
    - lat_res and lon_res should be identical across all four datasets
      (or very close, within floating point rounding)
    - lat/lon min/max should all cover Ethiopia (~3–15°N, 33–48°E)
    - If any dataset shows a different resolution, the difference map
      resampling note will appear in the app for that pair
"""

from pathlib import Path
import numpy as np
import pandas as pd

FILE_GLOB = "*.nc"

DATASETS = {
    "CHIRPS": "CHIRPS",
    "ENACTS": "ENACTS",
    "IMERG": "IMERG",
    "ERA5": "ERA5",
}

STATION_CSV = "RF_Station_Grid_Format.csv"


def open_nc(folder: str):
    import xarray as xr

    path = Path(folder)
    files = sorted(path.glob(FILE_GLOB))
    if not files:
        raise FileNotFoundError(f"No .nc files found in {folder}/")
    # Open just the first file — enough to check resolution
    return xr.open_dataset(str(files[0])), files


def guess_lat_lon(ds):
    for lat_name in ["latitude", "lat", "y"]:
        if lat_name in ds.coords:
            break
    else:
        raise ValueError(f"Cannot find lat coord. coords={list(ds.coords)}")
    for lon_name in ["longitude", "lon", "x"]:
        if lon_name in ds.coords:
            break
    else:
        raise ValueError(f"Cannot find lon coord. coords={list(ds.coords)}")
    return lat_name, lon_name


def resolution_of(arr):
    """Return mean absolute step between sorted coordinate values."""
    vals = np.sort(np.unique(arr))
    if len(vals) < 2:
        return float("nan")
    return float(np.mean(np.diff(vals)))


def check_gridded(name, folder):
    print(f"\n{'='*60}")
    print(f"  {name}  ({folder}/)")
    print(f"{'='*60}")
    try:
        ds, files = open_nc(folder)
        print(f"  Files found   : {len(files)}")
        print(f"  First file    : {files[0].name}")
        print(f"  Variables     : {list(ds.data_vars)}")
        print(f"  Coordinates   : {list(ds.coords)}")
        print(f"  Dimensions    : {dict(ds.dims)}")

        lat_name, lon_name = guess_lat_lon(ds)
        lats = ds[lat_name].values.ravel()
        lons = ds[lon_name].values.ravel()

        lat_res = resolution_of(lats)
        lon_res = resolution_of(lons)

        print(f"\n  Lat coord name: {lat_name!r}")
        print(f"  Lon coord name: {lon_name!r}")
        print(
            f"  Lat range     : {lats.min():.4f} → {lats.max():.4f}°N  ({len(np.unique(lats))} points)"
        )
        print(
            f"  Lon range     : {lons.min():.4f} → {lons.max():.4f}°E  ({len(np.unique(lons))} points)"
        )
        print(f"  Lat resolution: {lat_res:.6f}°  (~{lat_res * 111:.1f} km)")
        print(f"  Lon resolution: {lon_res:.6f}°  (~{lon_res * 111:.1f} km)")

        if "time" in ds.coords:
            times = ds["time"].values
            print(
                f"\n  Time range    : {pd.Timestamp(times[0]).date()} → {pd.Timestamp(times[-1]).date()}"
            )
            print(f"  Time steps    : {len(times)}")

        ds.close()
        return {
            "lat_res": lat_res,
            "lon_res": lon_res,
            "lat_name": lat_name,
            "lon_name": lon_name,
            "n_lat": len(np.unique(lats)),
            "n_lon": len(np.unique(lons)),
        }

    except Exception as e:
        print(f"  ERROR: {e}")
        return None


def check_station_csv(path):
    print(f"\n{'='*60}")
    print(f"  EMI Stations  ({path})")
    print(f"{'='*60}")
    try:
        df = pd.read_csv(path, index_col=0)
        df.index = df.index.astype(str).str.strip()

        lat_row = (
            "DAILY/LAT"
            if "DAILY/LAT" in df.index
            else ("LAT" if "LAT" in df.index else None)
        )
        if "LON" not in df.index or lat_row is None:
            raise ValueError("Missing LON or LAT row in CSV")

        lons = df.loc["LON"].astype(float)
        lats = df.loc[lat_row].astype(float)
        stations = df.columns.tolist()

        df_data = df.drop(index=["LON", lat_row])
        df_data.index = pd.to_datetime(df_data.index, format="%Y%m%d", errors="coerce")
        df_data = df_data.dropna(how="all")

        print(f"  Stations      : {len(stations)}")
        print(f"  Station names : {stations}")
        print(f"  Lat range     : {lats.min():.4f} → {lats.max():.4f}°N")
        print(f"  Lon range     : {lons.min():.4f} → {lons.max():.4f}°E")
        print(
            f"  Date range    : {df_data.index.min().date()} → {df_data.index.max().date()}"
        )
        print(f"  Total rows    : {len(df_data)}")

        # Missing data summary
        ts = df_data.apply(pd.to_numeric, errors="coerce").replace(-99, float("nan"))
        missing_pct = ts.isna().mean() * 100
        print(f"\n  Missing data % per station (top 5 worst):")
        for stn, pct in missing_pct.nlargest(5).items():
            print(f"    {stn:25s}: {pct:.1f}% missing")

    except Exception as e:
        print(f"  ERROR: {e}")


def compare_resolutions(results):
    print(f"\n{'='*60}")
    print("  RESOLUTION COMPARISON SUMMARY")
    print(f"{'='*60}")
    valid = {k: v for k, v in results.items() if v is not None}
    if not valid:
        print("  No valid datasets to compare.")
        return

    print(
        f"  {'Dataset':<10} {'Lat res (°)':<15} {'Lon res (°)':<15} {'Lat coord':<12} {'Lon coord':<12} {'Grid (lat×lon)'}"
    )
    print(f"  {'-'*80}")
    for name, r in valid.items():
        print(
            f"  {name:<10} {r['lat_res']:<15.6f} {r['lon_res']:<15.6f} "
            f"{r['lat_name']:<12} {r['lon_name']:<12} {r['n_lat']}×{r['n_lon']}"
        )

    lat_res_vals = [r["lat_res"] for r in valid.values()]
    lon_res_vals = [r["lon_res"] for r in valid.values()]
    lat_match = np.allclose(lat_res_vals, lat_res_vals[0], atol=1e-4)
    lon_match = np.allclose(lon_res_vals, lon_res_vals[0], atol=1e-4)

    print()
    if lat_match and lon_match:
        print(
            "  ✅ All datasets have the same resolution — no resampling needed for difference maps."
        )
    else:
        print("  ⚠️  Resolution mismatch detected between datasets:")
        if not lat_match:
            print(f"     Lat resolutions differ: {lat_res_vals}")
        if not lon_match:
            print(f"     Lon resolutions differ: {lon_res_vals}")
        print(
            "     The app will auto-resample the finer grid onto the coarser one for difference maps."
        )

    # Check coordinate name consistency
    lat_names = set(r["lat_name"] for r in valid.values())
    lon_names = set(r["lon_name"] for r in valid.values())
    if len(lat_names) > 1:
        print(f"\n  ℹ️  Lat coord names differ across datasets: {lat_names}")
        print("     (This is fine — the app handles this automatically.)")
    if len(lon_names) > 1:
        print(f"  ℹ️  Lon coord names differ across datasets: {lon_names}")


if __name__ == "__main__":
    results = {}
    for name, folder in DATASETS.items():
        if Path(folder).exists():
            results[name] = check_gridded(name, folder)
        else:
            print(f"\n  SKIPPING {name}: folder '{folder}/' not found")
            results[name] = None

    if Path(STATION_CSV).exists():
        check_station_csv(STATION_CSV)
    else:
        print(f"\n  SKIPPING station CSV: '{STATION_CSV}' not found")

    compare_resolutions(results)
