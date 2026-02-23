import xarray as xr
from pathlib import Path

# Path to a single .nc file OR a folder of .nc files
DATA_PATH = "CHIRPS"  # change to file path if needed
FILE_GLOB = "*.nc"


def open_dataset(path):
    path = Path(path)

    if path.is_file():
        # Single NetCDF file
        ds = xr.open_dataset(path)
    else:
        # Multiple NetCDF files combined by time/coords
        files = sorted(path.glob(FILE_GLOB))
        if not files:
            raise ValueError("No .nc files found in folder.")
        ds = xr.open_mfdataset(files, combine="by_coords")

    return ds


def inspect_dataset(ds):
    print("\n--- DATASET SUMMARY ---")
    print(ds)

    print("\n--- VARIABLES ---")
    print(list(ds.data_vars))

    print("\n--- COORDINATES ---")
    print(list(ds.coords))

    print("\n--- DIMENSIONS ---")
    print(ds.dims)

    # Example: print min/max time if available
    if "time" in ds.coords:
        print("\nTime range:")
        print(ds["time"].min().values, "→", ds["time"].max().values)


if __name__ == "__main__":
    ds = open_dataset(DATA_PATH)
    inspect_dataset(ds)
