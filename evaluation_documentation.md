# Evaluation pipeline

## Evaluation on the validation set

### 1. Inference

Use the run script with `src/main_evaluation.py` in "eval" mode.

Example command:

```bash
python src/main_evaluation.py eval path/to/checkpoint/folder --rollout 15
```

The model is evaluated on the validation set defined in the run's `config.yaml`. The progress is stored in an evaluation checkpoint every `training.checkpoint_interval` batches, so an interrupted evaluation continues from where it stopped when it is started again.

Options:

| Option | Default | Description |
| --- | --- | --- |
| `model_path` | (positional, required) | Path to the trained model output: a checkpoint file (`cp-epoch*.pt`) or a run directory containing `config.yaml` and `checkpoints/` |
| `configargs` | (positional, optional) | Extra config overrides applied on top of the run's stored `config.yaml` |
| `--rollout STEPS` | `1` | Number of forward passes per dataset input. A rollout larger than 1 requires `n_in_timesteps <= n_out_timesteps` |
| `--eval_checkpoint PATH` | `eval_checkpoint.pkl` in the run's root directory | Path to an evaluation checkpoint (.pkl) to resume from. If the path points to a valid checkpoint file, it is loaded and the evaluation continues from where it stopped |
| `--avg_pool_store` / `--no-avg_pool_store` | `True` | Apply average pooling on the model outputs before the metrics are calculated |

The specific commands used for the submission can be found in `utils/cmd_library/run_eval_final.sh` and serve as examples.

At the end of the evaluation, rank 0 prints the metrics to the log, with a block for each metric and lead time:

```text
{metric}_{lead time}h:
{feature}_{lead time}h: {value}
...
```

where

- `{metric}` is one of "mse", "rmse", "crps", or "spread_skill_ratio" ("crps" and "spread_skill_ratio" only for Bayesian models)
- `{lead time}` is the lead time in hours
- `{feature}` is the name of the physical quantity, e.g. `W_200`

### 2. Parsing of the evaluation logs

The metrics printed to the logs can be collected and compared with `src/plotting/parse_eval.py`.

Example command:

```bash
python -m src.plotting.parse_eval --logs_dir logs/eval/ --out_dir results/eval_plots
```

Options:

| Option | Default | Description |
| --- | --- | --- |
| `--logs_dir` | (required) | Directory of `*.out` log files, or a glob pattern |
| `--out_dir` | `results/eval_plots` | Output directory for the CSVs and plots |

Each log file is treated as one run, i.e. one checkpoint and configuration. The run label is derived from the file name by removing the `eval-` prefix and the trailing SLURM job ID, so the job names used in `utils/cmd_library/run_eval_final.sh` result in readable labels. Besides the metrics, the SLURM job ID, the number of nodes and GPUs, the mesh dimensions, and the rollout are extracted from the logs.

This results in the following files in `--out_dir`:

- `parsed_metrics.csv` contains one row per (run, metric, lead time, feature) with the raw metric value
- `overall_scores.csv` contains one aggregated score per (run, metric, lead time). For `spread_skill_ratio`, the score is the mean over features of the distance from the ideal value 1.0. For the error metrics, each value is normalized by the mean of the same (metric, feature, lead time) across all runs and then averaged over features, so 1.0 corresponds to the average run. Lower is better in both cases
- `overall/` contains one plot per metric with the overall score as a function of lead time, one line per run. This shows the rollout stability and which run performs best
- `per_feature/{metric}/` contains one plot per feature with the raw metric value as a function of lead time, one line per run
- `ranking/` contains one heatmap per metric with the normalized score per (run, feature) at the largest lead time. This shows on which features each run performs best and worst

Non-finite values (e.g. an infinite `spread_skill_ratio`) are dropped from the aggregated scores and plots.


## ExtremeWeatherBench evaluation pipeline

### Requirements for the plotting scripts

The scripts in `src/plotting/` used in sections 3 to 5 need certain packages that are not part of the main environment. Install them with:

```bash
pip install -r requirements_eval_ewb.txt
```

This includes the pinned [ExtremeWeatherBench](https://github.com/RAI-SCC/ExtremeWeatherBench) version used for the paper, as well as `numpy`, `pandas`, `xarray`, `zarr`, `polars`, `cartopy` and `matplotlib`.

### 1. Inference for extreme events

Use the run script with `src/main_evaluation.py` in "ewb" mode. The goal is to generate a zarr store with model predictions for specific extreme events and lead times, which can then be used for evaluation with the EWB pipeline.

Example command:

```bash
python src/main_evaluation.py ewb path/to/checkpoint/folder
```

The configuration needs to define additional options for the evaluation pipeline, which are described in the table below.

Config options (under `data` in `config.yaml`):

| Option | Required | Description |
| --- | --- | --- |
| `forecast_store_path` | Yes | Location where the zarr forecasts will be stored. The name will be `forecasts-<SLURM_JOB_ID>.zarr` |
| `ewb_case_ids` | No | List of EWB case IDs to evaluate. Each case ID corresponds to a single extreme event, e.g. `events_tropical_cyclone_256,events_heat_wave_20` |
| `ewb_events_path` | No | Path to the EWB events data directory, containing the event metadata |
| `ewb_event_types` | No | List of event types to evaluate. Each event type corresponds to a single extreme event type, e.g. `tropical_cyclone,heat_wave` |

Options like `mesh_dims` and `n_samples` can still be inherited from the `config.yaml`, or overridden on the command line.

Below is a list of non-configurable command line options that can be passed to the script. These are not stored in the `config.yaml` and need to be specified on the command line.

Options:

| Option | Default | Description |
| --- | --- | --- |
| `model_path` | (positional, required) | Path to the trained model output: a checkpoint file (`cp-epoch*.pt`) or a run directory containing `config.yaml` and `checkpoints/` |
| `configargs` | (positional, optional) | Extra config overrides applied on top of the run's stored `config.yaml` |
| `--lead_times HOURS` | Value stored in `config.yaml` | Lead times in hours to run. Either comma-separated integers (e.g. `12,24`) or a range `<start>:<stop>:<step>` (`<stop>` exclusive, `<step>` optional, defaults to 1), e.g. `-24:24:6`. Negative values are allowed |
| `--bounded_region` / `--no-bounded_region` | `True` | Crop stored fields to each event's lat/lon bounding box |
| `--bounded_features` / `--no-bounded_features` | `True` | Store only the features scored for each event type |
| `--denormalize` / `--no-denormalize` | `True` | Denormalize model outputs before writing them to zarr |
| `--avg_ensemble` / `--no-avg_ensemble` | `False` | Average model outputs over the ensemble dimension before storing |
| `--avg_pool_rollout` / `--no-avg_pool_rollout` | `False` | Apply average pooling on the model outputs before they are fed back into the rollout |
| `--avg_pool_store` / `--no-avg_pool_store` | `True` | Apply average pooling on the model outputs before writing them to zarr. Set to `False` automatically if `--avg_pool_rollout` is set |
| `--overwrite` | `False` | Wipe the existing forecast store and recompute everything. By default, an existing store is resumed: every (event, lead time) group already on disk is skipped |

The specific commands used for the submission can be found in `utils/cmd_library/ewb_eval_final.sh` and serve as examples.


### 2. Output format

This results in a .zarr directory with a group for each event. The names of the groups follow the structure

```text
events_{event type}_{ewb_case_id}_lt{lead time string}_off{offset}
```

where

- `{event type}` is one of "heat_wave", "freeze", or "tropical_cyclone"
- `{offset}` is the offset in hours of the initialization time from 00:00 UTC
- `{lead time string}` is the difference between the beginning of the event period according to EWB data and the initialization time of the first rollout step in hours. If this time is negative, instead of a minus sign, it gets the prefix "m", e.g. `ltm24` would indicate that the first rollout step was initialized with ERA5 data from 24h after the start of the event.

This output is structurally different from what the EWB standard pipeline expects in three ways:

1. The events and initialization times are split into zarr groups
2. The names used for the physical quantities differ between the generated output (ERA5 style names) and what EWB expects
3. EWB is designed for deterministic forecasts, the generated data has an additional ensemble member dimension

How to resolve these differences depends on the desired evaluation metrics and is addressed separately in the corresponding sections.


### 3. Calculation of heat wave and freeze metrics

The paper reports RMSEs of the ensemble mean predictions for a single heat wave and a single freeze event.
These metrics are computed against **ERA5 as used by ExtremeWeatherBench (EWB)**, i.e. the unmodified, analysis-ready ERA5 store, **not** the preprocessed ERA5 used for training/inference of BEAST.
See the [EWB data documentation](https://extremeweatherbench.readthedocs.io/en/latest/data/) for how to access it; by default it is read from the public Google bucket without credentials.

Example command for a heat wave:

```bash
python src/plotting/evaluate_temperature_extremes.py path/to/output.zarr path/to/era5.zarr
```

Add `--freeze` for a freeze event evaluation instead. This generates a CSV file with columns for the lead times, the metrics (MSEs of the mean prediction for different subsets of the ensemble), and their corresponding values. The RMSEs shown in the paper are calculated by taking the square root of these MSE values.

#### What it does internally

The first zarr formatting difference listed in section 2 is resolved by loading only a single forecast initialization time of the event that is currently being evaluated, using that zarr handle to calculate the MSE for the event as a function of lead time, repeating this for all initialization times, and averaging afterward. The final MSE is then converted into an RMSE.

For each of the individual zarr handles, the following changes are needed:

- the variable `2m_temperature` is renamed to `surface_air_temperature`
- the `lead_time` key is converted from an integer number in hours relative to the event start to a `numpy.timedelta64` relative to forecast initialization
- the forecast time dimension `time` is converted into an initialization time with the key `init_time`
- the `ensemble` dimension is averaged over, ignoring any NaN entries (due to partially sparse entries along the combined (`init_time`, `lead_time`) time dimension)

After these transformations, the zarr handle can be used with EWB's standard `XArrayForecast`, `EvaluationObject`, and `metrics.MeanSquaredError`.


### 4. Extraction of cyclone tracks

The paper presents a visualization of an ensemble of predicted cyclone tracks. The first step is to extract the individual cyclone track predictions from all the ensemble members:

```bash
python src/plotting/extract_tc_track_from_zarr.py path/to/output.zarr
```

The output is a CSV file with columns containing the number of the ensemble member, the predicted point in time, and the location of the predicted cyclone center in both latitude and longitude (in addition to some other variables).

The reference track is extracted from IBTrACS data using:

```bash
python src/plotting/extract_ibtracs_path.py 256
```

where 256 is the EWB internal case id of the tropical cyclone Damien. The output file is in the same format as for the TC tracks extracted from BEAST outputs, minus the ensemble member column.

#### What it does internally

Since only a single event and initialization time is considered, the first of the differences listed in section 2 is addressed by using the `xarray` handle of the corresponding zarr group.

The ensemble dimension is handled by looping over it and treating the different members as separate forecasts. All members are written into one CSV output file that contains a column indicating the ensemble member number.

For the variable name differences, the `lead_time` dimension is handled as in section 3. Further, the following variables are renamed:

- `time` to `valid_time`
- `mean_sea_level_pressure` to `air_pressure_at_mean_sea_level`
- `10m_u_component_of_wind` to `surface_eastward_wind`
- `10m_v_component_of_wind` to `surface_northward_wind`

The entries for the key `geopotential_thickness` are calculated from the variables `geopotential_300` and `geopotential_500`.

EWB's `derived.TropicalCycloneTrackVariables` and its `get_or_compute_tracks` method are then used, comparing to IBTrACS data. The default parameters are used for the class instantiation, except for `min_wind_timesteps`, which is set to `3.0`.

The IBTrACS data can be obtained from EWB's internal methods, but again requires renaming its variables and coordinates for compatibility:

- `ISO_TIME` to `valid_time`
- `NAME` to `tc_name`
- `SEASON` to `season`
- `NUMBER` to `number`
- `LAT` to `latitude`
- `LON` to `longitude`
- `USA_WIND` to `surface_wind_speed`
- `USA_PRES` to `air_pressure_at_mean_sea_level`

The return of the `get_or_compute_tracks` method then contains a series of latitude and longitude values for the cyclone center.


### 5. Visualization of cyclone tracks

The extracted tracks are visualized with `src/plotting/plot_tracks.py`. For each initialization time, a map is generated showing the ground-truth track, the ensemble mean track, and a probability map of the ensemble predictions.

Example command:

```bash
python src/plotting/plot_tracks.py \
    --ground_truth path/to/ground_truth.csv \
    --ensemble_predictions path/to/ensemble_predictions.csv \
    --output_dir results/cyclone_track_plots \
    --radius 150
```

Options:

| Option | Default | Description |
| --- | --- | --- |
| `--ground_truth` | (required) | CSV file with the reference track, as generated by `extract_ibtracs_path.py` |
| `--ensemble_predictions` | (required) | CSV file with the ensemble track predictions, as generated by `extract_tc_track_from_zarr.py` |
| `--output_dir` | (required) | Output directory for the figures |
| `--radius` | `150` | Radius in km used for the probability calculation |

The ground-truth CSV requires the columns `latitude` and `longitude`. They define the extent of the plot and the reference track.

The ensemble predictions CSV requires the columns `init_time`, `lead_time`, `ensemble`, `latitude`, and `longitude`, e.g.:

| init_time | lead_time | ensemble | latitude | longitude |
| --- | --- | ---: | ---: | ---: |
| 2020-01-31 12:00 | 0 days | 1 | ... | ... |
| 2020-01-31 12:00 | 6 hours | 1 | ... | ... |
| 2020-01-31 12:00 | 0 days | 2 | ... | ... |

The data are grouped by `init_time`, then by `ensemble`. The `lead_time` is converted to a pandas `timedelta` and used to order the points of each member's track.

One figure per initialization time is saved to `--output_dir` in two formats:

```text
cyclone_track_{init_time}_{radius}.png
cyclone_track_{init_time}_{radius}.svg
```

Each figure contains:

- the ground-truth track as a red line, with a green circle at the start and a green star at the end
- the ensemble mean track as a black line
- the probability map, drawn with the `turbo` colormap in 20 intervals between 0 and 1, with a colorbar labeled "Probability". Cells with zero probability are not displayed

#### What it does internally

The probability at each grid point is the fraction of ensemble members whose track passes within the given radius:

$$
P(x,y) = \frac{N_{\mathrm{members\ within\ radius}}}{N_{\mathrm{ensemble\ members}}}
$$

The probability is evaluated on a regular latitude/longitude grid of 400 x 400 points covering the extent of the ground-truth track plus a border (`get_probability_of_distance()`). Distances are great-circle distances calculated with the haversine formula (`haversine()`, Earth radius 6371 km). They are measured to the line segments between consecutive forecast points rather than to the discrete positions only. Each member counts at most once per grid point, even if its track passes through the radius multiple times.

The ensemble mean track is calculated by averaging latitude and longitude over all members separately for each lead time.

## Acknowledgment

Extreme weather predictions are built on
[ExtremeWeatherBench](https://github.com/brightbandtech/ExtremeWeatherBench)
(EWB) by Brightband and contributors. We use a fork pinned in `requirements_eval_ewb.txt`.
Please cite ExtremeWeatherBench if you use results from the extreme weather predictions.
