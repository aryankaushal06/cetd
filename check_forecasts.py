"""
check_forecasts.py
------------------
Run from the CETD project root to inspect the structure of forecast
NetCDF files in the aifs/, aifs_ens/, and gencast/ folders.

Each folder is expected to contain files named <year>.nc.

Usage:
    python check_forecasts.py

What this extracts:
    - All dimension names and sizes
    - All coordinate names, types, and sample values
    - All variable names and shapes
    - Whether there is an ensemble/member dimension
    - Whether lead time is stored as a dimension or as absolute timestamps
    - Initialization time structure (how many inits per file, how often)
    - Lat/lon resolution and coverage
    - Any attributes that hint at the model or data provenance
"""

from pathlib import Path
import numpy as np
import pandas as pd

FORECAST_FOLDERS = {
    "AIFS (deterministic)": "datasets/aifs",
    "AIFS Ensemble": "datasets/aifs_ens",
    "GenCast": "datasets/gencast",
}

SEP = "=" * 65


def print_section(title):
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)


def inspect_forecast_file(path: Path):
    import xarray as xr

    print(f"\n  File: {path.name}")
    print(f"  {'─'*55}")

    ds = xr.open_dataset(str(path), engine=None)

    # ── Dimensions ──
    print(f"\n  Dimensions:")
    for dim, size in ds.dims.items():
        print(f"    {dim:<25} size = {size}")

    # ── Coordinates ──
    print(f"\n  Coordinates:")
    for name, coord in ds.coords.items():
        vals = coord.values
        dtype = str(vals.dtype)

        # For time-like coords show first/last and step
        if np.issubdtype(vals.dtype, np.datetime64) or "time" in name.lower():
            try:
                ts = pd.to_datetime(vals.ravel())
                if len(ts) == 1:
                    summary = f"{ts[0]}"
                elif len(ts) <= 6:
                    summary = ", ".join(str(t.date()) for t in ts)
                else:
                    step = ts[1] - ts[0] if len(ts) > 1 else None
                    summary = (
                        f"{ts[0].date()} → {ts[-1].date()}, "
                        f"n={len(ts)}, step≈{step}"
                    )
                print(f"    {name:<25} [{dtype}]  {summary}")
                continue
            except Exception:
                pass

        # For numeric coords show range + resolution
        if np.issubdtype(vals.dtype, np.floating) or np.issubdtype(
            vals.dtype, np.integer
        ):
            flat = vals.ravel()
            if len(flat) >= 2:
                unique = np.unique(flat)
                res = (
                    float(np.mean(np.diff(unique)))
                    if len(unique) > 1
                    else float("nan")
                )
                print(
                    f"    {name:<25} [{dtype}]  "
                    f"{flat.min():.4f} → {flat.max():.4f}, "
                    f"n={len(unique)}, res≈{res:.4f}"
                )
            else:
                print(f"    {name:<25} [{dtype}]  {flat}")
            continue

        # Fallback
        print(
            f"    {name:<25} [{dtype}]  shape={vals.shape}, "
            f"sample={vals.ravel()[:3]}"
        )

    # ── Data variables ──
    print(f"\n  Variables:")
    for name, var in ds.data_vars.items():
        dims_str = str(tuple(var.dims))
        shape_str = str(var.shape)
        attrs_sample = {
            k: v
            for k, v in var.attrs.items()
            if k in ("units", "long_name", "standard_name")
        }
        print(
            f"    {name:<25} dims={dims_str:<40} shape={shape_str}  attrs={attrs_sample}"
        )

    # ── Global attributes ──
    if ds.attrs:
        print(f"\n  Global attributes:")
        for k, v in ds.attrs.items():
            v_str = str(v)[:80]
            print(f"    {k:<25} {v_str}")

    # ── Guess structure ──
    print(f"\n  Structure guess:")
    dim_names = list(ds.dims.keys())

    # Ensemble dimension
    ens_candidates = [
        d
        for d in dim_names
        if any(
            kw in d.lower()
            for kw in (
                "member",
                "ensemble",
                "ens",
                "number",
                "realization",
                "sample",
            )
        )
    ]
    if ens_candidates:
        for d in ens_candidates:
            print(f"    Ensemble dim  : '{d}'  (size={ds.dims[d]})")
    else:
        print(f"    Ensemble dim  : not detected (deterministic or 1-member)")

    # Lead time / step dimension
    lead_candidates = [
        d
        for d in dim_names
        if any(
            kw in d.lower()
            for kw in (
                "lead",
                "step",
                "forecast_period",
                "ahead",
                "flt",
                "horizon",
            )
        )
    ]
    if lead_candidates:
        for d in lead_candidates:
            print(f"    Lead-time dim : '{d}'  (size={ds.dims[d]})")
    else:
        # Check if time dimension encodes absolute valid times (not lead offsets)
        time_candidates = [d for d in dim_names if "time" in d.lower()]
        if time_candidates:
            print(
                f"    Lead-time dim : not found — time stored as absolute valid time? "
                f"(dims with 'time': {time_candidates})"
            )

    # Initialization time
    init_candidates = [
        d
        for d in dim_names
        if any(
            kw in d.lower()
            for kw in ("init", "base", "reference", "start", "issue", "run")
        )
    ]
    if init_candidates:
        for d in init_candidates:
            print(f"    Init-time dim : '{d}'  (size={ds.dims[d]})")

    # Precip variable
    precip_candidates = [
        v
        for v in ds.data_vars
        if any(
            kw in v.lower()
            for kw in (
                "precip",
                "rain",
                "tp",
                "pr",
                "precipitation",
                "total_precipitation",
                "rainfall",
            )
        )
    ]
    if precip_candidates:
        print(f"    Precip var(s) : {precip_candidates}")
    else:
        print(f"    Precip var    : not found — check variable list above")

    ds.close()


def check_folder(label, folder):
    print_section(label)
    path = Path(folder)
    if not path.exists():
        print(f"  SKIPPING: folder '{folder}/' not found")
        return

    nc_files = sorted(path.glob("*.nc"))
    if not nc_files:
        print(f"  SKIPPING: no .nc files in '{folder}/'")
        return

    print(f"  Folder        : {folder}/")
    print(f"  Files found   : {len(nc_files)}")
    print(f"  File names    : {[f.name for f in nc_files]}")

    # Inspect first file in detail
    print(f"\n  ── Inspecting first file ──")
    inspect_forecast_file(nc_files[0])

    # If there are multiple files, do a quick shape check on the last one
    # to catch any structural differences across years
    if len(nc_files) > 1:
        import xarray as xr

        print(f"\n  ── Quick check: last file ({nc_files[-1].name}) ──")
        ds = xr.open_dataset(str(nc_files[-1]), engine=None)
        print(f"  Dims: {dict(ds.dims)}")
        print(f"  Vars: {list(ds.data_vars)}")
        # Check time range of last file
        for coord in ds.coords:
            if "time" in coord.lower():
                try:
                    ts = pd.to_datetime(ds[coord].values.ravel())
                    print(
                        f"  '{coord}' range: {ts[0].date()} → {ts[-1].date()}"
                    )
                except Exception:
                    pass
        ds.close()


if __name__ == "__main__":
    for label, folder in FORECAST_FOLDERS.items():
        check_folder(label, folder)

    print(f"\n{SEP}")
    print("  DONE — paste the full output above when discussing")
    print("  forecast integration with Claude.")
    print(SEP)
