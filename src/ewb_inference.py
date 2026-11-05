import os
import time
from datetime import datetime, timedelta

import numpy as np
import torch
import torch.distributed as dist
from beast.comm import execute_in_waves
from beast.utils import set_all_seeds

from gb.ewb import parse_case_id
from gb.forecast_store import ForecastGroup, ForecastRegion, ForecastZarrStore
from gb.globals import SST_NORMALIZED_ZERO
from setup import setup_model

log_metric = None
log_metrics = None


# Levels the atmospheric-river metric integrates over.
ATMOSPHERIC_RIVER_LEVLES = (1000, 925, 850, 700, 600, 500, 400, 300)


def _at_levels(variable: str, levels) -> tuple[str, ...]:
    """Return the ``<variable>_<level>`` channel names for ``levels``."""
    return tuple(f"{variable}_{level}" for level in levels)


# Mapping to only keep variables and levels required for that event.
EVENT_TYPE_VARIABLES: dict[str, tuple[str, ...]] = {
    "heat_wave": ("2m_temperature",),
    "freeze": ("2m_temperature",),
    "tropical_cyclone": (
        "mean_sea_level_pressure",
        "10m_u_component_of_wind",
        "10m_v_component_of_wind",
        "geopotential_300",
        "geopotential_500",
    ),
    "atmospheric_river": (
        *_at_levels("specific_humidity", ATMOSPHERIC_RIVER_LEVLES),
        *_at_levels("u_component_of_wind", ATMOSPHERIC_RIVER_LEVLES),
        *_at_levels("v_component_of_wind", ATMOSPHERIC_RIVER_LEVLES),
    ),
    "severe_convection": (
        "mean_sea_level_pressure",
        "10m_u_component_of_wind",
        "10m_v_component_of_wind",
        "u_component_of_wind_500",
        "v_component_of_wind_500",
        "temperature",
        "specific_humidity",
        "geopotential",
    ),
}


def _matching_features(variable: str, names: list[str]) -> list[int]:
    """Return indices of every channel the request ``variable`` selects.

    Matches the channel named exactly ``variable`` plus every
    ``<variable>_<level>`` pressure channel.
    """
    prefix = f"{variable}_"
    return [
        i
        for i, name in enumerate(names)
        if name == variable
        or (name.startswith(prefix) and name[len(prefix) :].isdigit())
    ]


def _event_feature_indices(event_type: str, feature_names) -> np.ndarray | None:
    """Return the channels to store for ``event_type``.

    The model always computes all 84 channels; this only limits what is written
    to that event's group. Unknown event types keep everything, so a new EWB
    event type degrades to the old behaviour rather than silently losing data.

    Parameters
    ----------
    event_type : str
        EWB ``event_type``, e.g. ``"heat_wave"``.
    feature_names : sequence of str
        The dataset's ``feature`` coordinate, named ``<variable>`` for surface
        channels and ``<variable>_<level>`` for pressure levels.

    Returns
    -------
    np.ndarray or None
        Sorted global feature indices, or ``None`` to keep every channel.

    Raises
    ------
    ValueError
        If a required variable matches no channel at all.
    """
    required = EVENT_TYPE_VARIABLES.get(event_type)
    if required is None:
        return None

    names = [str(name) for name in feature_names]
    wanted: list[int] = []
    for variable in required:
        matches = _matching_features(variable, names)
        if not matches:
            raise ValueError(
                f"Event type {event_type!r} needs variable {variable!r}, which "
                f"matches none of the dataset's {len(names)} features."
            )
        wanted.extend(matches)
    return np.array(sorted(set(wanted)), dtype=int)


def _build_forecast_store(
    config_dict, dataset, n_samples, up_group, avg_ensemble=False
):
    """Construct the :class:`ForecastZarrStore` describing the shared output grid.

    The store holds the shared grid metadata (feature/lat/lon/ensemble); each
    event is written into its own group whose ``time`` / ``lead_time``
    coordinates are supplied later. Coordinates are derived from the input
    dataset and config so the output aligns with the ERA5 conventions in
    ``data/README.md``.

    ``avg_ensemble`` averages every member across the whole SP group into a
    single global ensemble mean, so the store then holds exactly one member.
    """
    # Feature names are taken directly from the original dataset so the output
    # store matches the input feature ordering/naming exactly.
    feature_names = dataset.zarr_data["feature"].values

    # Global spatial grid (721 x 1440 ERA5 grid: 90..-90, 0..359.75). The store
    # declares the full grid; the model only operates on the even-trimmed grid
    # (``dataset.xlat``), so the last latitude row remains unwritten (NaN).
    latitudes = dataset.zarr_data["latitude"].values
    longitudes = dataset.zarr_data["longitude"].values

    # Ensemble members: n_samples per SP rank across the whole SP group. When
    # averaging, the whole group collapses to a single global ensemble mean.
    n_ensemble = 1 if avg_ensemble else n_samples * dist.get_world_size(group=up_group)
    ensemble_members = np.arange(1, n_ensemble + 1, dtype=np.int32)

    slurm_job_id = os.getenv("SLURM_JOB_ID", "interactive")

    store = ForecastZarrStore(
        store_path=config_dict["data"].get(
            "forecast_store_path",
        )
        + f"/forecasts-{slurm_job_id}.zarr",
        feature_names=feature_names,
        latitudes=latitudes,
        longitudes=longitudes,
        ensemble_members=ensemble_members,
        feature_chunk=dataset.in_var[1] - dataset.in_var[0],
        longitude_chunk=dataset.in_out_xlon[1] - dataset.in_out_xlon[0],
    )
    return store


def _global_ensemble_mean(output, up_group, device, n_samples):
    """Average one output over the whole SP group and broadcast it back.

    ``output`` is a rank-local ``(ens, feat, lat, lon)`` tensor. Every SP rank
    reduces the sum of its local mean and divides by the number of SP ranks to
    obtain the single global ensemble mean, then repeats it to ``n_samples``
    members so the rollout keeps feeding the model its expected ensemble size.
    All up-ranks must join the collective. Returns a CPU tensor matching the
    input's dtype/layout.
    """
    partial = output.mean(dim=0, keepdim=True).to(device)
    dist.all_reduce(partial, op=dist.ReduceOp.SUM, group=up_group)
    partial /= dist.get_world_size(group=up_group)
    return partial.to(output.dtype).cpu().repeat(n_samples, 1, 1, 1)


# Reference grid the phase offset is measured against. EWB event windows are
# defined on the standard 6/12-hour synoptic grid, so a lead time that is not a
# multiple of ``model_dt`` phase-shifts the rollout onto a different sub-grid.
GRID_HOURS = 12


def _phase_offset(lead: int, model_dt: int, grid_hours: int = GRID_HOURS) -> int:
    """Return the rollout's phase offset (hours) from the ``grid_hours`` grid.

    During rollout the clock starts at ``event.start_date - lead`` and steps by
    ``model_dt``. Event windows are aligned to the ``grid_hours`` grid, so the
    valid times land on the grid iff ``lead`` is a multiple of ``grid_hours``.
    The offset is ``(-lead) mod grid_hours``, i.e. how far the first saved step
    sits after the nearest grid point. ``0`` means the lead is grid-aligned.
    """
    return int((-int(lead)) % grid_hours)


def _event_group_name(event, lead: int, model_dt: int) -> str:
    """Return the (unique) Zarr group name for one ``(event, lead)`` forecast.

    There is one group per ``(event, lead_time)`` so that each group's ``time``
    coordinate is a single contiguous ``model_dt``-spaced sequence (a shared
    axis across phase-offset leads would interleave and break the contiguous
    region writes ``write_block`` requires).

    The name encodes the event id, the lead time and the phase offset from the
    12-hour grid (see :func:`_phase_offset`) as
    ``<event_id>_lt<lead>_off<offset>`` (negative leads use ``m``, e.g.
    ``..._ltm6_off6``).
    """
    lead = int(lead)
    lead_tag = f"m{-lead}" if lead < 0 else str(lead)
    offset = _phase_offset(lead, model_dt)
    return f"{event.id}_lt{lead_tag}_off{offset}"


def _event_times(event, model_dt, lead) -> np.ndarray:
    """Return the sorted valid times for one ``(event, lead)`` forecast.

    Every timestamp the rollout might write for this ``(event, lead)`` must
    exist in the group's ``time`` coordinate. The rollout clock starts at
    ``event.start_date - lead`` and advances by ``model_dt`` each step, saving
    every step that lands inside ``[start_date, end_date]``. Because there is
    one group per lead, this is a single contiguous ``model_dt``-spaced grid, so
    ``write_block`` always sees a contiguous time range.

    Parameters
    ----------
    event : Event
        The EWB event whose ``start_date`` / ``end_date`` bound the window.
    model_dt : int
        Model timestep in hours; the rollout advances by this amount each step.
    lead : int
        The lead time (in hours) this group is run with. Both positive and
        negative values are supported.
    """
    step = timedelta(hours=model_dt)
    # Mirror the rollout clock: it starts at ``start_date - lead`` and steps
    # forward by ``model_dt``. Fast-forward to the first step at or after the
    # event start, then enumerate through the event end.
    t = event.start_date - timedelta(hours=int(lead))
    if t < event.start_date:
        n_skip = -(-(event.start_date - t) // step)  # ceil division
        t = t + n_skip * step
    times: list[np.datetime64] = []
    while t <= event.end_date:
        times.append(np.datetime64(t, "s"))
        t = t + step
    return np.array(times, dtype="datetime64[s]")


def _build_event_groups(
    dataset,
    model_dt,
    feature_names,
    store,
    bounded_region: bool = True,
    bounded_features: bool = True,
) -> dict[str, ForecastGroup]:
    """Build one :class:`ForecastGroup` per ``(event, lead_time)`` in the dataset.

    Each group only spans that ``(event, lead)``'s own ``time`` steps and its
    single lead time, keeping each on-disk array as small as possible. Splitting
    by lead time (rather than one group per event) guarantees every group's
    ``time`` coordinate is a single contiguous ``model_dt``-spaced grid, so the
    per-lead region writes ``write_block`` performs never straddle a
    phase-offset gap. The group name encodes the lead and its offset from the
    12-hour grid (see :func:`_event_group_name`).

    ``bounded_region`` crops each group to the event's lat/lon bounding box;
    when ``False`` the group spans the whole globe. ``bounded_features`` keeps
    only the features scored for the event type; when ``False`` every channel is
    stored.
    """
    # Whole-globe bounding box, used when region cropping is disabled.
    global_bbox = (
        float(store.latitudes.min()),
        float(store.latitudes.max()),
        float(store.longitudes.min()),
        float(store.longitudes.max()),
    )

    groups: dict[str, ForecastGroup] = {}
    for event, lead in dataset.event_lead_list:
        lead = int(lead)
        name = _event_group_name(event, lead, model_dt)
        if name in groups:
            # A given (event, lead) pair only ever produces one group.
            continue
        if bounded_region:
            lat_min, lat_max = float(event.lat_min), float(event.lat_max)
            lon_min, lon_max = float(event.lon_min), float(event.lon_max)
        else:
            lat_min, lat_max, lon_min, lon_max = global_bbox
        feature_indices = (
            _event_feature_indices(str(event.event_type), feature_names)
            if bounded_features
            else None
        )
        groups[name] = ForecastGroup(
            name=name,
            times=_event_times(event, model_dt, lead),
            lead_times=np.array([lead], dtype=np.int32),
            lat_min=lat_min,
            lat_max=lat_max,
            lon_min=lon_min,
            lon_max=lon_max,
            feature_indices=feature_indices,
            event_type=str(event.event_type),
            case_id=parse_case_id(str(event.id)),
            phase_offset=_phase_offset(lead, model_dt),
        )
    return groups


@torch.no_grad()
def inference(config_dict):

    (
        model,
        dataset,
        _,
        up_group,
        _,
        device,
        n_samples,
        bayesian,
        _,
        _,
        _,
        autocast,
        feature_weights,
    ) = setup_model(config_dict, mode="inference")

    if config_dict["data"]["n_in_timesteps"] > config_dict["data"]["n_out_timesteps"]:
        print(
            f"Invalid in_timesteps ({config_dict['data']['n_in_timesteps']}), out_timesteps ({config_dict['data']['n_out_timesteps']}) The number of output timesteps should be greater or equal than the number of input timesteps. Exiting. "
        )
        return

    model.eval()
    set_all_seeds(17 + up_group.rank())

    # Output options.
    bounded_region = config_dict["data"].get("bounded_region", True)
    bounded_features = config_dict["data"].get("bounded_features", True)
    denormalize = config_dict["data"].get("denormalize", True)
    avg_ensemble = config_dict["data"].get("avg_ensemble", False)
    avg_pool_rollout = config_dict["data"].get("avg_pool_rollout", False)
    avg_pool_store = config_dict["data"].get("avg_pool_store", True)
    # When False (the default), resume: keep any (event, lead_time) group that
    # already exists on disk and only compute the missing ones. When True, wipe
    # the store and recompute every group from scratch.
    overwrite = config_dict["data"].get("overwrite", False)
    # avoiding double pooling
    if avg_pool_rollout and avg_pool_store:
        print(
            "WARNING: Setting avg_pool_store to False because avg_pool_rollout is True."
        )
        avg_pool_store = avg_pool_store and not avg_pool_rollout

    # Per-rank denormalization factors for this rank's feature shard. Outputs
    # are normalized as ``(x - mean) / std``; denormalizing is ``x * std + mean``.
    # ``list_from_output`` returns CPU tensors shaped ``(ens, feat, lat, lon)``,
    # so the factors are CPU tensors broadcasting over the ensemble/lat/lon axes.
    denorm_std = denorm_mean = None
    if denormalize:
        in_var = dataset.in_var
        denorm_std = torch.as_tensor(
            dataset.norm_values["std"][in_var[0] : in_var[1]],
            dtype=torch.float32,
        ).reshape(-1, 1, 1)
        denorm_mean = torch.as_tensor(
            dataset.norm_values["mean"][in_var[0] : in_var[1]],
            dtype=torch.float32,
        ).reshape(-1, 1, 1)

    data_access_rank = int(dist.get_rank(group=None) // 8)
    dsync_interval = config_dict["training"]["dsync_interval"]

    if dist.get_rank(group=None) == 0:
        print(config_dict["data"])

    model_dt = config_dict["data"]["dt"]

    # ------------------------------------------------------------------
    # Set up the shared Zarr forecast store.
    #
    # The store is a single Zarr dataset that every rank writes into
    # concurrently. Each rank owns a disjoint region of the array:
    #   * DTP ranks own a `feature` (channel) and `longitude` shard,
    #     exposed by the dataset as `dataset.in_var` / `dataset.in_out_xlon`.
    #   * SP ranks own a contiguous block of `ensemble` members.
    # Because the regions are disjoint, region-based writes need no locking.
    # ------------------------------------------------------------------
    store = _build_forecast_store(
        config_dict, dataset, n_samples, up_group, avg_ensemble=avg_ensemble
    )

    # One group per event keeps the store small: each group only spans that
    # event's own time window, lead times and the channels its event type is
    # scored on. ``bounded_region`` / ``bounded_features`` can widen a group to
    # the whole globe / all channels.
    event_groups = _build_event_groups(
        dataset,
        model_dt,
        store.feature_names,
        store,
        bounded_region=bounded_region,
        bounded_features=bounded_features,
    )

    # Resume support: unless --overwrite was passed, any (event, lead_time)
    # group already present on disk is treated as already computed. Those groups
    # are neither re-initialized (which would wipe their data) nor recomputed in
    # the rollout loop below; only the missing groups are (re)created.
    #
    # The set of names to skip is decided by rank 0 and broadcast so every rank
    # agrees on which groups are already done.
    if overwrite:
        skip_names: set[str] = set()
    else:
        if dist.get_rank(group=None) == 0:
            skip_names = {
                name
                for name, group in event_groups.items()
                if store.group_exists(group)
            }
        else:
            skip_names = set()
        skip_bcast = [skip_names]
        dist.broadcast_object_list(skip_bcast, src=0, group=None)
        skip_names = skip_bcast[0]

    # Only one rank materializes the store metadata / coordinates (root group
    # plus one child group per event).
    if dist.get_rank(group=None) == 0:
        print("Initializing store ... ")
        if skip_names:
            print(
                f"Resuming: {len(skip_names)} of {len(event_groups)} "
                "(event, lead_time) groups already computed, skipping them. "
                "Pass --overwrite to recompute everything."
            )
    if dist.get_rank(group=None) == 0:
        # Preserve existing groups when resuming; only wipe on --overwrite.
        store.initialize_root(
            description="ERA5 ensemble forecast fields stacked into feature dimension.",
            overwrite=overwrite,
        )
        for name, group in event_groups.items():
            if name in skip_names:
                continue
            store.initialize_group(group)
            print(
                f"  group {group.name}: "
                f"{len(group.global_features(store.n_feature))} of "
                f"{store.n_feature} features"
            )
        # Consolidate the root metadata only after all child groups exist so
        # readers can enumerate the per-event groups.
        store.finalize()
    dist.barrier(group=None)

    # Region owned by this rank (feature + longitude from DTP, ensemble from SP).
    # When averaging, the single global mean lives in ensemble slot 0 and is
    # written by up-rank 0 only; the other SP ranks own an empty ensemble slice.
    up_rank = dist.get_rank(group=up_group)
    if avg_ensemble:
        ens_start, ens_stop = (0, 1) if up_rank == 0 else (0, 0)
    else:
        ens_start = up_rank * n_samples
        ens_stop = ens_start + n_samples
    rank_region = ForecastRegion(
        feature=(dataset.in_var[0], dataset.in_var[1]),
        longitude=(dataset.in_out_xlon[0], dataset.in_out_xlon[1]),
        latitude=(0, dataset.xlat),
        ensemble=(ens_start, ens_stop),
    )
    print(f"R{dist.get_rank()}/{dist.get_world_size()} - rank_region : {rank_region}")
    for i in range(len(dataset)):
        # Resume: skip any (event, lead_time) whose group is already on disk.
        # ``event_lead_list[i]`` is cheap (no data load), so we can decide to
        # skip before the expensive ``dataset[i]`` read. Every rank iterates the
        # identical ``event_lead_list`` and shares the same ``skip_names``, so
        # all ranks skip the exact same indices and stay in lockstep for the
        # collective ``execute_in_waves`` below.
        skip_event, skip_lead = dataset.event_lead_list[i]
        if _event_group_name(skip_event, int(skip_lead), model_dt) in skip_names:
            if dist.get_rank(group=None) == 0:
                print(
                    f"Skipping already-computed event {skip_event.title} "
                    f"(lead {int(skip_lead)}h)."
                )
            continue

        rollout_step = False
        if i % dsync_interval == 0:
            event, lt, samples = execute_in_waves(lambda: dataset[i])
        else:
            event, lt, samples = dataset[i]

        lead_hours = int(lt)
        lt = timedelta(hours=lead_hours)
        latest_time = event.start_date - lt

        if dist.get_rank(group=None) == 0:
            print(f"Event: {event.title}")
            print(f"Event type: {event.event_type}")
            print(f"Initial time:     {latest_time}")
            print(f"Event start time: {event.start_date}")
            print(f"Event end time:   {event.end_date}")

        # Accumulate the rank-local forecast fields that fall inside the event
        # window, together with their valid times, so we can write them as one
        # contiguous block per (event, lead_time).
        event_fields: list[torch.Tensor] = []
        event_times: list[datetime] = []

        while latest_time < event.end_date:
            x = dataset.input_from_list(samples).to(device)
            with torch.autocast(
                device_type="cuda", dtype=torch.bfloat16, enabled=autocast
            ):
                if bayesian:
                    if not rollout_step:
                        y = model(x, samples=n_samples).flatten(0, 1)
                    else:
                        x = x.reshape(n_samples, x.shape[0] // n_samples, *x.shape[1:])
                        y = model(x, sampled_input=True).flatten(0, 1)
                else:
                    y = model(x)

            new_outputs = dataset.list_from_output(
                y, feature_weights, SST_NORMALIZED_ZERO
            )

            if avg_pool_rollout:
                for o_idx, output in enumerate(new_outputs):
                    output = torch.nn.functional.pad(
                        output, (0, 1, 0, 1), mode="replicate"
                    )
                    output = torch.nn.functional.avg_pool2d(
                        output,
                        kernel_size=2,
                        stride=1,
                    )
                    new_outputs[o_idx] = output

            if avg_ensemble:
                # Replace each ensemble with its global mean over the whole SP
                # group, in normalized space, so both the stored fields and the
                # next-step model input use the single global sample. The mean
                # is broadcast back to ``n_samples`` members so the rollout keeps
                # feeding the model the ensemble size it was called with.
                new_outputs = [
                    _global_ensemble_mean(output, up_group, device, n_samples)
                    for output in new_outputs
                ]

            samples.extend(new_outputs)

            # Advance the clock one model step per produced output timestep and
            # capture any timestep that lies inside the event window.
            for output in new_outputs:  # (ens, feat, lat, lon)
                latest_time = latest_time + timedelta(hours=model_dt)
                if event.start_date <= latest_time <= event.end_date:
                    # Denormalize only the copy that gets stored; the rollout
                    # keeps feeding the normalized ``samples`` back to the model.
                    # When averaging, every member already holds the global mean,
                    # so a single member is enough for the store.
                    stored = output[:1] if avg_ensemble else output
                    if denormalize:
                        stored = stored * denorm_std + denorm_mean
                    if avg_pool_store:
                        stored = torch.nn.functional.pad(
                            stored, (0, 1, 0, 1), mode="replicate"
                        )
                        stored = torch.nn.functional.avg_pool2d(
                            stored,
                            kernel_size=2,
                            stride=1,
                        )
                    event_fields.append(stored.unsqueeze(0).unsqueeze(0))
                    event_times.append(latest_time)

            if not avg_ensemble:
                rollout_step = True

            if dist.get_rank(group=None) == 0:
                print(f"Current time: {latest_time}")

        # Concatenate the collected timesteps along the time axis and write the
        # rank-local region into the shared Zarr store. Ranks that own no
        # ensemble members (e.g. non-root SP ranks when averaging) still ran the
        # collective above but have nothing of their own to write.
        owns_ensemble = rank_region.ensemble[1] > rank_region.ensemble[0]
        if event_fields and owns_ensemble:
            # field T x LT x E x C x Lat x Lon
            block = torch.cat(
                event_fields, dim=0
            )  # (ensemble, time, feature, lat, lon)
            store.write_block(
                block,
                group=event_groups[_event_group_name(event, lead_hours, model_dt)],
                region=rank_region,
                lead_hours=lead_hours,
                times=event_times,
            )

        time.sleep(data_access_rank * 0.001)

    dist.barrier(group=None)
    if dist.get_rank(group=None) == 0:
        print("All forecasts written to the Zarr store.")
