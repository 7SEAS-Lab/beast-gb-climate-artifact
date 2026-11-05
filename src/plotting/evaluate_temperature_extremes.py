import argparse
import os

# import logging
from pathlib import Path

import extremeweatherbench as ewb
import numpy as np
import pandas as pd
import xarray as xr

# from extract_ibtracs_path import extract_ibtracs_path_data
from extract_tc_track_from_zarr import get_case_init_time, get_structured_group_list
from extremeweatherbench import utils

# from extremeweatherbench.events.tropical_cyclone import generate_tc_tracks_by_init_time
from extremeweatherbench.evaluate import ExtremeWeatherBench

# logger = logging.getLogger(__name__)


def prepare_prediction_data_for_heat_wave(
    prediction_ds: xr.Dataset, event_start: np.datetime64 | None = None
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    """Prepare ERA5 data for TC track detection.

    Converts ERA5 variables to the format expected by the TC track detection
    algorithm, including:
    - Renaming variables to match EWB conventions
    - Ensuring proper coordinate structure

    Args:
        prediction_ds: Filtered ERA5 dataset

    Returns:
        Tuple of (sea_level_pressure, wind_speed, geopotential_thickness) DataArrays
    """
    # Check required variables
    required_vars = [
        "2m_temperature",
    ]

    print("  Checking for required variables...")

    missing = [var for var in required_vars if var not in prediction_ds.feature.values]
    if missing:
        raise ValueError(
            f"Missing required variables: {missing}. "
            f"Available: {list(prediction_ds.fearures.values)}"
        )

    print("  Creating restructured xarray handle...")

    # Start with coordinates and dimensions from original dataset
    modified_ds = xr.Dataset(
        coords={k: v for k, v in prediction_ds.coords.items() if k != "feature"}
    )

    # print(modified_ds.init_time.values)

    # Convert feature dimension to data variables
    # This preserves lazy loading by using xarray's indexing without .load() or .compute()
    # print("  Converting 'feature' dimension to data variables...")

    # Get the name of the data variable that contains the feature data
    # Typically there's one main data variable indexed by the feature dimension
    data_var_name = list(prediction_ds.data_vars)[0]

    for feature_name in prediction_ds.feature.values:
        # Select data for this feature and assign as a named variable
        # Using .sel() maintains lazy evaluation - data is not loaded until explicitly computed
        # Extract the DataArray from the Dataset result by accessing the underlying variable
        feature_data = prediction_ds[data_var_name].sel(feature=feature_name)
        modified_ds[feature_name] = feature_data

    modified_ds = modified_ds.reset_coords("feature", drop=True)
    modified_ds = modified_ds[required_vars]

    print("  Convert lead_time dimension to different notational convention...")

    # print(f"before lead time values: {modified_ds.lead_time.values}")

    # Convert lead_time from int32 hours to timedelta64[ns]
    lead_time_hours = modified_ds.coords["lead_time"].values
    # First convert to timedelta64[h] (hours), then to ns
    modified_ds = modified_ds.assign_coords(
        lead_time=lead_time_hours.astype("timedelta64[h]").astype("timedelta64[ns]")
    )

    # print(f"after lead time values: {modified_ds.lead_time.values}")

    if event_start is None:
        earliest_prediction = min(modified_ds.coords["time"].values)
    else:
        earliest_prediction = event_start
        # print(type(event_start), event_start)
        # print(min(modified_ds.coords["time"].values))

    # print(earliest_prediction)

    init_time_vals = earliest_prediction - modified_ds.coords["lead_time"].values
    modified_ds = modified_ds.assign_coords(lead_time=init_time_vals).rename(
        {"lead_time": "init_time"}
    )  # .isel(init_time=[-1])
    modified_ds = modified_ds.dropna(dim="time", how="all")

    # print(f"init time values: {modified_ds.init_time.values}")

    print("  Renaming variables...")

    modified_ds = modified_ds.rename(
        {
            "time": "valid_time",
            "2m_temperature": "surface_air_temperature",
        }
    )

    # print(modified_ds)

    # modified_ds = modified_ds.rename_dims({"time": "init_time"})

    # Calculate wind speed from u and v components
    # u_wind = era5_ds["10m_u_component_of_wind"]
    # v_wind = era5_ds["10m_v_component_of_wind"]
    # wind_speed = calc.calculate_wind_speed(u_wind, v_wind)

    # Calculate geopotential thickness (500hPa - 300hPa geopotential difference)
    # ERA5 uses levels, so we need to extract 500hPa and 300hPa

    ######## Denormalization: mean: 278.49728, std: 21.279663

    # modified_ds["surface_air_temperature"].values = modified_ds["surface_air_temperature"].values * 21.279663 + 278.49728

    print("  Converting to init_time coordinates...")

    return utils.convert_init_and_valid_time_to_lead_and_init_time(modified_ds)


def convert_outputs_dataframe_to_xarray(
    outputs: pd.DataFrame,
) -> xr.Dataset:
    """Convert evaluation outputs DataFrame to xarray Dataset.

    Converts a pandas DataFrame with columns ["value", "lead_time", "metric", "case_id_number"]
    into an xarray Dataset where all columns except "value" become coordinate variables,
    and "value" becomes the data variable.

    Args:
        outputs: DataFrame with index column and columns "value", "lead_time", "metric", "case_id_number"

    Returns:
        xarray.Dataset with coordinates for index, lead_time, metric, case_id_number
        and data variable "value"
    """
    # Reset index to make it a regular column
    df_reset = outputs.reset_index()

    # Drop the original index column (usually named "index")
    if "index" in df_reset.columns:
        df_reset = df_reset.drop(columns=["index"])

    # Set all columns except "value" as the index (multi-index)
    coord_columns = [col for col in df_reset.columns if col != "value"]
    df_indexed = df_reset.set_index(coord_columns)

    # Convert to xarray Dataset
    ds = df_indexed.to_xarray()

    # Ensure "value" is the data variable (it should be by default)
    ds = ds.rename({df_indexed.columns[0]: "value"})

    return ds


def extract_heat_wave_metrics_from_zarr(
    prediction_zarr_path: str,
    target_zarr_path: str,
    output_csv_path: str = ".",
    freezes: bool = False,
) -> pd.DataFrame:
    """Extract tropical cyclone track from filtered ERA5 zarr file.

    Uses the TempestExtremes-based TC track detection algorithm from
    extremeweatherbench.events.tropical_cyclone to identify cyclone centers
    based on SLP minima, closed contours, and wind speed criteria.

    Args:
        zarr_path: Path to the filtered ERA5 zarr file.
        output_csv_path: Optional path to save the track as CSV. If None, only returns DataFrame.
        **track_detection_kwargs: Additional keyword arguments passed to
            generate_tc_tracks_by_init_time for customizing track detection.

    Returns:
        DataFrame with columns: valid_time, latitude, longitude
    """

    target_type = "freeze" if freezes else "heat_wave"

    event_types = [target_type]

    print(f"Loading zarr file: {prediction_zarr_path}")

    structured_group_list = get_structured_group_list(
        prediction_zarr_path, event_types=event_types
    )

    selected_features = ["surface_air_temperature"]

    era5_target = ewb.inputs.ERA5(source=target_zarr_path, variables=selected_features)

    print("Loading EWB's default database of extreme events...")
    all_cases_db = ewb.cases.load_ewb_cases()

    output_ds_list = []

    for entry in structured_group_list:
        event_type, case_id, offset, lead_time = entry
        print(
            f"\nEvent {case_id}, Type: {event_type}, Lead time: {lead_time}, Offset: {offset}"
        )

        if event_type == target_type:
            prediction_ds = xr.open_dataset(
                prediction_zarr_path,
                engine="zarr",
                group=f"events_{event_type}_{case_id}_lt{f'm{-lead_time}' if lead_time < 0 else lead_time}_off{offset}",
            )

            print("\nDataset structure:")
            print(f"  Dimensions: {dict(prediction_ds.sizes)}")
            print(f"  Variables: {list(prediction_ds.data_vars)}")
            print(f"  Coordinates: {list(prediction_ds.coords)}")

            print(prediction_ds)

            # lead_times = prediction_ds.lead_time.values

            # ensemble_members = prediction_ds.ensemble.values

            # print(reformatted_ds)

            event_start_time = get_case_init_time(case_id)

            # print(f"Attempting to extract track for event {event_id}")

            # ds = xr.Dataset.from_dataframe(df)
            # print(ds)

            # Prepare data for TC detection
            print("\nPreparing data for heat wave evaluation...")
            reformatted_ds = prepare_prediction_data_for_heat_wave(
                prediction_ds, event_start=event_start_time
            )
            print("Data preparation complete.\n")

            # print("Reformatted ds", reformatted_ds)

            beast_forecast = ewb.inputs.XarrayForecast(
                ds=reformatted_ds,
                name="BEAST_forecast",
                variables=selected_features,
                variable_mapping={
                    # Map dataset feature names accordingly
                    "surface_air_temperature": "surface_air_temperature",
                    "init_time": "init_time",
                    "lead_time": "lead_time",
                    "ensemble": "ensemble",
                },
            )

            eval_obj = [
                ewb.inputs.EvaluationObject(
                    event_type=event_type,
                    metric_list=[
                        # # 1. CRPS for Surface Air Temperature
                        # ewb.metrics.CRPS(
                        #     forecast_variable="surface_air_temperature",
                        #     target_variable="surface_air_temperature",
                        #     ensemble_dim="ensemble"
                        # ),
                        # 2. RMSE for Surface Air Temperature
                        ewb.metrics.MeanSquaredError(
                            forecast_variable="surface_air_temperature",
                            target_variable="surface_air_temperature",
                        ),
                        ewb.metrics.MSESubEnsemble(
                            forecast_variable="surface_air_temperature",
                            target_variable="surface_air_temperature",
                            selected_ensemble=1,
                        ),
                        ewb.metrics.MSESubEnsemble(
                            forecast_variable="surface_air_temperature",
                            target_variable="surface_air_temperature",
                            selected_ensemble=3,
                        ),
                    ],
                    target=era5_target,
                    forecast=beast_forecast,
                ),
            ]

            specific_case = [
                case for case in all_cases_db if case.case_id_number == case_id
            ]

            # print("Running evaluation for multiple features and event types. Please wait...")
            runner = ExtremeWeatherBench(
                case_metadata=specific_case,
                evaluation_objects=eval_obj,
            )

            outputs = runner.run_evaluation()

            outputs = outputs.drop(
                columns=[
                    "init_time",
                    "target_variable",
                    "forecast_source",
                    "target_source",
                    "event_type",
                ]
            )
            full_case_string = f"{case_id}_lt{f'm{-lead_time}' if lead_time < 0 else lead_time}_off{offset}"
            # print(outputs)
            outputs = outputs.assign(case_id_number=full_case_string)
            # print(outputs)

            output_ds = convert_outputs_dataframe_to_xarray(outputs)

            output_ds_list.append(output_ds)
        else:
            print(f"Unimplemented event type {event_type}")
            continue

    full_output_ds = xr.concat(output_ds_list, dim="case_id_number", join="outer")
    full_output_ds["value"] = full_output_ds["value"].astype("float32")

    # print(full_output_ds)

    full_output_ds = full_output_ds.mean(dim="case_id_number", skipna=True)

    # print(full_output_ds)

    full_output_df = full_output_ds.to_dataframe().reset_index()

    if "index" in full_output_df.columns:
        full_output_df = full_output_df.drop(columns=["index"])

    fname = f"{target_type}_metrics.csv"
    filepath = os.path.join(output_csv_path, fname)
    full_output_df.to_csv(filepath, index=False)
    print(f"Saved: {filepath}")

    # print(full_output_df)


def main():
    parser = argparse.ArgumentParser(
        description="Extract tropical cyclone track from filtered ERA5 zarr file"
    )
    parser.add_argument(
        "zarr_path", type=str, help="Path to the filtered prediction zarr file"
    )
    parser.add_argument(
        "era5_path", type=str, help="Path to the filtered ERA5 zarr file"
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Output CSV file path (default: <zarr_name>_track.csv)",
    )

    parser.add_argument("--freeze", action="store_true")

    args = parser.parse_args()

    # Set default output path if not provided
    if args.output is None:
        zarr_name = Path(args.zarr_path).parent
        args.output = f"{zarr_name}"

    # case_ids = [192]

    extract_heat_wave_metrics_from_zarr(
        args.zarr_path,
        args.era5_path,
        args.output,
        args.freeze,
    )


if __name__ == "__main__":
    main()
