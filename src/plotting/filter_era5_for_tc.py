"""
Script to filter standard ERA5 zarr into ones only containing certain tropical cyclone events,
filtered by tropical cyclone event IDs listed in extremeweatherbench/data/events.yaml.
"""

import os

import pandas as pd
import xarray as xr

# from extremeweatherbench.defaults import ibtracs_target
from extract_ibtracs_path import extract_ibtracs_path_data


def extract_bounds(cyclone_df, spatial_margin=5.0):
    start_time = cyclone_df["valid_time"].min().to_datetime64().astype("datetime64[ns]")
    end_time = cyclone_df["valid_time"].max().to_datetime64().astype("datetime64[ns]")
    min_lon = cyclone_df["longitude"].min() - spatial_margin
    max_lon = cyclone_df["longitude"].max() + spatial_margin
    min_lat = cyclone_df["latitude"].min() - spatial_margin
    max_lat = cyclone_df["latitude"].max() + spatial_margin

    return start_time, end_time, min_lon, max_lon, min_lat, max_lat


def filter_dataset_by_bounds(era5_ds, bounds):
    start_time, end_time, min_lon, max_lon, min_lat, max_lat = bounds

    print(era5_ds)

    # variable filtering

    required_variables = [
        "geopotential",
        "mean_sea_level_pressure",
        "10m_u_component_of_wind",
        "10m_v_component_of_wind",
    ]

    filtered_ds = era5_ds[required_variables]

    # time filtering

    times = pd.to_datetime(filtered_ds.time)
    relevant_times = times[(times >= start_time) & (times < end_time)]
    filtered_ds = filtered_ds.sel(time=relevant_times)

    # spatial filtering

    lats = filtered_ds.latitude
    relevant_lats = lats[(lats >= min_lat) & (lats <= max_lat)]
    filtered_ds = filtered_ds.sel(latitude=relevant_lats)

    lons = filtered_ds.longitude
    relevant_lons = lons[(lons >= min_lon) & (lons <= max_lon)]
    filtered_ds = filtered_ds.sel(longitude=relevant_lons)

    # level filtering

    relevant_levels = [300, 500]
    filtered_ds = filtered_ds.sel(level=relevant_levels)

    # introducing ensemble dimension

    filtered_ds = filtered_ds.expand_dims(dim={"ensemble": [1]})

    # restructuring time dimension into (init_)time and lead_time

    filtered_ds = filtered_ds.assign_coords(
        {"time": pd.to_datetime(filtered_ds.time) - start_time}
    )
    filtered_ds = filtered_ds.rename_dims({"time": "lead_time"})
    filtered_ds = filtered_ds.rename_vars({"time": "lead_time"})

    filtered_ds = filtered_ds.expand_dims(dim={"time": [start_time]})

    return filtered_ds


def main():
    # config stuff
    storage_path_base = os.getenv("STORAGE_BASE_PATH", ".")
    era5_path = os.path.join(storage_path_base, "era5_2020_2025_37levels.zarr")
    # era5_path = os.path.join(storage_path_base, "era5_preprocessed_2020-2025.zarr")
    margin = 5.0

    # create era5 zarr handle
    era5_ds = xr.open_dataset(era5_path, engine="zarr")

    # Extract paths for tropical cyclones
    print("=" * 70)
    print("Extracting IBTrACS Path Data for Tropical Cyclone Events")
    print("=" * 70)

    # Get all tropical cyclone event IDs
    tc_event_ids = [256]  # get_tropical_cyclone_event_ids()
    print(f"\nFound {len(tc_event_ids)} tropical cyclone events")

    # Extract the path data
    results = extract_ibtracs_path_data(tc_event_ids)

    for event_id, df in results.items():
        print(event_id)
        # start_time = df["valid_time"].min()
        # end_time = df["valid_time"].max()
        # min_lon = df["longitude"].min() - margin
        # max_lon = df["longitude"].max() + margin
        # min_lat = df["latitude"].min() - margin
        # max_lat = df["latitude"].max() + margin

        # print(start_time, end_time)
        # print(min_lon, max_lon)
        # print(min_lat, max_lat)

        bounds = extract_bounds(df, margin)

        print("=" * 70)
        print(f"Filtering ERA5 data according to boundaries of event {event_id}")
        print("=" * 70)

        filtered_era5 = filter_dataset_by_bounds(era5_ds, bounds)

        out_path = os.path.join(storage_path_base, f"tc_{event_id}_era5.zarr")
        filtered_era5.to_zarr(out_path, zarr_format=2)


if __name__ == "__main__":
    main()
