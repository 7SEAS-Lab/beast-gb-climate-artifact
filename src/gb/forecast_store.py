"""Distributed writing of ensemble forecasts to a Zarr store via xarray.

This module provides :class:`ForecastZarrStore`, a helper that lets every rank of
a domain-parallel / ensemble-parallel run write its own shard of the forecast
tensor into a *single* shared Zarr dataset without any inter-rank locking.

Per-event groups
----------------
Each event is stored in its *own* Zarr group (named after the event id) inside
the store. A group only declares the coordinates that event actually needs (its
own valid ``time`` steps and lead times), so the on-disk arrays stay small: the
store never allocates one giant global-grid array spanning every event.

Why this works without locks
----------------------------
Each rank only ever touches a region of the store that no other rank touches:

* The **DTP** (domain tensor parallel) group splits the global field along the
  ``feature`` (channel) and ``longitude`` axes. A DTP rank owns
  ``feature[in_var[0]:in_var[1]]`` and ``longitude[in_out_xlon[0]:in_out_xlon[1]]``.
* The **SP** (ensemble / uncertainty) group splits the ``ensemble`` axis. Each SP
  rank owns a contiguous block of ensemble members.
* ``time`` and ``lead_time`` are indexed by the event/lead-time currently being
  processed.

Because the union of the per-rank regions is a disjoint partition of the whole
array, ``xarray.Dataset.to_zarr(region=...)`` can be called concurrently from
every rank. The store is created once (metadata only, no data) up front, and
subsequent writes only fill in previously-allocated chunks.

Target layout
-------------
::

    Dimensions:    (ensemble, lead_time, time, feature, latitude, longitude)
    Data variables:
        fields (ensemble, lead_time, time, feature, latitude, longitude) float32

The chunking is chosen so that a single rank write for one (event, lead_time)
maps onto whole chunks, keeping writes independent and efficient.
"""

import dataclasses as dc
import math
from datetime import datetime

import numpy as np
import torch
import xarray as xr
import zarr
from zarr.codecs import BloscCodec, BloscShuffle


@dc.dataclass
class ForecastRegion:
    """Description of the array region owned by the current rank.

    All bounds are half-open ``[start, stop)`` intervals into the global arrays.
    """

    feature: tuple[int, int]
    longitude: tuple[int, int]
    latitude: tuple[int, int]
    ensemble: tuple[int, int]


@dc.dataclass
class ForecastGroup:
    """Per-event coordinate metadata for one Zarr group.

    Each event is written into its own group. The group only spans the ``time``
    steps and ``lead_time`` values that this specific event uses, and its spatial
    grid is cropped to the event's ``(lat, lon)`` bounding box, keeping the
    on-disk arrays as small as possible.

    Parameters
    ----------
    name : str
        Unique group name (typically the event id).
    times : np.ndarray
        1-D ``datetime64`` valid times covered by this event.
    lead_times : np.ndarray
        1-D int lead times (hours) this event uses.
    lat_min, lat_max, lon_min, lon_max : float
        The event bounding box in degrees. Longitudes follow the same
        convention as the event metadata (``-180..180``); the store maps them
        onto its own ``0..360`` longitude grid.
    feature_indices : np.ndarray, optional
        Indices into the store's ``feature_names`` to keep for this event. The
        model always computes every channel; this only limits what is stored,
        and different event types need very different subsets (a heat wave is
        scored on 2m temperature alone, severe convection needs most of the
        pressure-level profile). ``None`` keeps every channel.
    event_type : str, optional
        EWB ``event_type`` as it appears in the source YAML, recorded as a group
        attribute.
    case_id : int, optional
        EWB ``case_id_number`` as it appears in the source YAML.
    phase_offset : int, optional
        Offset (hours) of this group's valid-time grid from the 12-hour synoptic
        grid, i.e. ``(-lead) mod 12``. ``0`` means grid-aligned. Recorded as a
        group attribute so readers can identify phase-shifted lead times without
        recomputing them.
    """

    name: str
    times: np.ndarray
    lead_times: np.ndarray
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float
    feature_indices: np.ndarray | None = None
    event_type: str | None = None
    case_id: int | None = None
    phase_offset: int | None = None

    def __post_init__(self) -> None:
        self.times = np.asarray(self.times)
        self.lead_times = np.asarray(self.lead_times, dtype=np.int32)
        self.n_time = self.times.shape[0]
        self.n_lead = self.lead_times.shape[0]
        if self.feature_indices is not None:
            self.feature_indices = np.unique(
                np.asarray(self.feature_indices, dtype=int)
            )
        # The store fills these in during ``initialize_group`` once it knows the
        # global grid: half-open slices into the *global* lat/lon arrays that the
        # group's cropped grid corresponds to.
        self.lat_slice: tuple[int, int] | None = None
        # ``(start, count)`` on a cyclic axis: global index of span position
        # ``i`` is ``(start + i) % n_lon``. A box crossing the grid seam (an
        # event spanning the prime meridian) cannot be a plain slice.
        self.lon_span: tuple[int, int] | None = None
        # Fast lookup helpers.
        self._time_index = {np.datetime64(t, "s"): i for i, t in enumerate(self.times)}
        self._lead_index = {int(lt): i for i, lt in enumerate(self.lead_times)}

    def time_index(self, when: datetime | np.datetime64) -> int:
        """Return the ``time`` coordinate index for a timestamp."""
        return self._time_index[np.datetime64(when, "s")]

    def lead_index(self, lead_hours: int) -> int:
        """Return the ``lead_time`` coordinate index for a lead time in hours."""
        return self._lead_index[int(lead_hours)]

    def global_features(self, n_feature: int) -> np.ndarray:
        """Return the global feature indices this group stores.

        Parameters
        ----------
        n_feature : int
            Total number of channels the store knows about.

        Returns
        -------
        np.ndarray
            Sorted global indices, or every index when no subset was given.
        """
        if self.feature_indices is None:
            return np.arange(n_feature, dtype=int)
        return self.feature_indices

    def rank_features(
        self, n_feature: int, in_var: tuple[int, int]
    ) -> tuple[np.ndarray, tuple[int, int]]:
        """Map a rank's feature shard onto this group's feature axis.

        A rank owns the contiguous global range ``in_var``. Because the kept
        indices are sorted, the ones inside that range are contiguous on the
        group's axis too, so a rank still needs only one region write.

        Parameters
        ----------
        n_feature : int
            Total number of channels the store knows about.
        in_var : tuple of int
            The rank's half-open global feature range.

        Returns
        -------
        tuple of (np.ndarray, tuple of int)
            Offsets into the rank-local feature axis to keep, and the half-open
            extent they occupy on the group's feature axis. Both are empty when
            the rank owns none of the group's channels.
        """
        kept = self.global_features(n_feature)
        owned = (kept >= in_var[0]) & (kept < in_var[1])
        if not owned.any():
            return np.empty(0, dtype=int), (0, 0)
        positions = np.flatnonzero(owned)
        return kept[owned] - in_var[0], (
            int(positions[0]),
            int(positions[-1]) + 1,
        )

    @property
    def n_lat(self) -> int:
        """Latitude size of the cropped group grid (after :meth:`set_bbox_slices`)."""
        if self.lat_slice is None:
            raise RuntimeError("Group bounding-box slices not yet resolved.")
        return self.lat_slice[1] - self.lat_slice[0]

    @property
    def n_lon(self) -> int:
        """Longitude size of the cropped group grid (after :meth:`set_bbox_slices`)."""
        if self.lon_span is None:
            raise RuntimeError("Group bounding-box slices not yet resolved.")
        return self.lon_span[1]

    def lon_runs(self, n_lon_global: int) -> list[tuple[int, int, int]]:
        """Split the group's cyclic longitude span into contiguous global runs.

        Returns one run for a normal box and two for a box crossing the seam.

        Parameters
        ----------
        n_lon_global : int
            Longitude size of the store's global grid.

        Returns
        -------
        list of tuple
            ``(global_start, global_stop, position)`` per run, where
            ``position`` is the run's offset along the group's longitude axis.
        """
        if self.lon_span is None:
            raise RuntimeError("Group bounding-box slices not yet resolved.")
        start, count = self.lon_span
        runs, position, cursor = [], 0, start
        while count > 0:
            take = min(count, n_lon_global - cursor)
            runs.append((cursor, cursor + take, position))
            position += take
            count -= take
            cursor = 0
        return runs

    def rank_lon_runs(
        self, n_lon_global: int, rank_lon: tuple[int, int]
    ) -> list[tuple[int, int, int]]:
        """Intersect :meth:`lon_runs` with a rank's owned longitude shard.

        Parameters
        ----------
        n_lon_global : int
            Longitude size of the store's global grid.
        rank_lon : tuple of int
            The rank's half-open global longitude extent.

        Returns
        -------
        list of tuple
            ``(global_start, global_stop, position)`` runs this rank writes,
            empty when the rank owns no part of the box.
        """
        owned = []
        for run_start, run_stop, position in self.lon_runs(n_lon_global):
            lo, hi = max(run_start, rank_lon[0]), min(run_stop, rank_lon[1])
            if hi > lo:
                owned.append((lo, hi, position + (lo - run_start)))
        return owned


class ForecastZarrStore:
    """Create and write per-event groups in a shared Zarr forecast store.

    The store holds the *shared* grid metadata (feature names, lat/lon grid,
    ensemble members and chunking). The event-specific coordinates (``time`` and
    ``lead_time``) live in each :class:`ForecastGroup`, so every event is written
    to its own Zarr group sized to that event alone.

    Parameters
    ----------
    store_path : str
        Path (or URL) of the Zarr store to create/write.
    feature_names : list[str]
        Names for the ``feature`` coordinate (length = number of channels).
    latitudes : np.ndarray
        1-D array of latitude values (length = latitude dimension).
    longitudes : np.ndarray
        1-D array of longitude values (length = longitude dimension).
    ensemble_members : np.ndarray
        1-D int array of ensemble member ids (e.g. ``[1, 2, ..., N]``).
    feature_chunk : int | None
        Chunk size along the ``feature`` dimension. Defaults to the per-rank
        feature-shard size (recommended so writes align to chunks).
    longitude_chunk : int | None
        Chunk size along the ``longitude`` dimension. Defaults to the per-rank
        longitude-shard size.
    dtype : np.dtype
        On-disk dtype of the ``fields`` variable (default ``float32``).
    """

    VAR_NAME = "fields"

    def __init__(
        self,
        store_path: str,
        feature_names: list[str],
        latitudes: np.ndarray,
        longitudes: np.ndarray,
        ensemble_members: np.ndarray,
        feature_chunk: int | None = None,
        longitude_chunk: int | None = None,
        dtype: np.dtype = np.dtype("float32"),
    ) -> None:
        self.store_path = store_path
        self.feature_names = list(feature_names)
        self.latitudes = np.asarray(latitudes)
        self.longitudes = np.asarray(longitudes)
        self.ensemble_members = np.asarray(ensemble_members, dtype=np.int32)
        self.dtype = np.dtype(dtype)

        self.n_feature = len(self.feature_names)
        self.n_lat = self.latitudes.shape[0]
        self.n_lon = self.longitudes.shape[0]
        self.n_ensemble = self.ensemble_members.shape[0]

        self.feature_chunk = int(feature_chunk) if feature_chunk else self.n_feature
        self.longitude_chunk = int(longitude_chunk) if longitude_chunk else self.n_lon

        self.compressors = (
            BloscCodec(cname="zstd", clevel=3, shuffle=BloscShuffle.bitshuffle),
        )

    # -- store creation ----------------------------------------------------

    @property
    def _dims(self) -> tuple[str, ...]:
        return ("time", "lead_time", "ensemble", "feature", "latitude", "longitude")

    def _chunks(self, group: ForecastGroup) -> tuple[int, ...]:
        """Chunk sizes for a group's ``fields`` array (cropped to its bbox)."""
        return (
            group.n_time,
            1,  # lead_time
            1,  # ensemble
            group.n_time,
            group.n_lat,
            self._lon_chunk(group),
        )

    def _group_path(self, group: ForecastGroup | str) -> str:
        """Return the path (or URL) of a single event group inside the store."""
        name = group.name if isinstance(group, ForecastGroup) else group
        return f"{self.store_path.rstrip('/')}/{name}"

    def _lon_chunk(self, group: ForecastGroup) -> int:
        """Longitude chunk size that never straddles a rank boundary.

        Returns
        -------
        int
            Chunk size along the group's longitude axis.
        """
        shard = self.longitude_chunk
        if shard <= 0 or shard >= self.n_lon:
            return group.n_lon

        boundaries: set[int] = set()
        for run_start, run_stop, position in group.lon_runs(self.n_lon):
            if position > 0:
                boundaries.add(position)
            edge = ((run_start // shard) + 1) * shard
            while edge < run_stop:
                boundaries.add(position + (edge - run_start))
                edge += shard

        if not boundaries:
            return group.n_lon
        return math.gcd(*boundaries)

    def _bbox_slices(
        self,
        lat_min: float,
        lat_max: float,
        lon_min: float,
        lon_max: float,
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        """Map an event bounding box onto the global grid.

        A one-cell margin is added on every side so the box is fully contained.

        Latitude is a plain half-open ``(start, stop)`` slice. Longitude is a
        ``(start, count)`` span on a cyclic axis -- global index ``i`` of the
        span is ``(start + i) % n_lon`` -- because a box may cross the grid's
        seam.
        """
        # Wrap onto the store's 0..360 grid. This must be a modulo, not a
        # ``+ 180`` shift: the latter maps -136.75 to 43.25 instead of 223.25,
        # placing the event on the wrong meridian, and it corrupts bounds that
        # are already positive too (15.25 -> 195.25).
        lon_min360 = lon_min % 360.0
        lon_max360 = lon_max % 360.0

        lon_start = int(
            max(np.absolute(self.longitudes - np.float32(lon_min360)).argmin() - 1, 0)
        )
        lon_stop = int(
            min(
                np.absolute(self.longitudes - np.float32(lon_max360)).argmin() + 1,
                len(self.longitudes),
            )
        )
        # Walk east from the western edge, wrapping over the seam if needed.
        lon_start %= self.n_lon
        lon_count = (lon_stop - lon_start) % self.n_lon
        if lon_count == 0:
            lon_count = self.n_lon
        lon_count = min(lon_count, self.n_lon)

        # Latitudes are stored north-to-south (descending), so ``lat_max`` maps
        # to the smaller index and ``lat_min`` to the larger one.
        lat_i_min = int(np.absolute(self.latitudes - np.float32(lat_max)).argmin() - 1)
        lat_i_max = int(np.absolute(self.latitudes - np.float32(lat_min)).argmin() + 1)
        lat_i_min = max(lat_i_min, 0)
        lat_i_max = min(lat_i_max, self.n_lat)

        return (lat_i_min, lat_i_max), (lon_start, lon_count)

    def _group_longitudes(self, group: ForecastGroup) -> np.ndarray:
        """Return the group's longitude coordinate, monotonically increasing.

        Gathering a cyclic span straight off the global grid gives a coordinate
        that jumps at the seam (``351 ... 359.75, 0 ... 16``), which xarray and
        ExtremeWeatherBench cannot select against. Unwrapping keeps it
        increasing, and spans that run past 360 are shifted back into the signed
        convention the event metadata uses, so the example becomes ``-9 ... 16``.
        """
        indices = (group.lon_span[0] + np.arange(group.n_lon)) % self.n_lon
        values = np.unwrap(
            np.asarray(self.longitudes, dtype=float)[indices], period=360.0
        )
        if values[-1] >= 360.0:
            values = values - 360.0
        return values

    def resolve_group_bbox(self, group: ForecastGroup) -> None:
        """Compute and cache the group's cropped lat/lon slices from its bbox."""
        lat_slice, lon_span = self._bbox_slices(
            group.lat_min, group.lat_max, group.lon_min, group.lon_max
        )
        group.lat_slice = lat_slice
        group.lon_span = lon_span

    def group_exists(self, group: ForecastGroup | str) -> bool:
        """Return whether an event group already exists in the store.

        Used to resume an interrupted run: a group whose metadata is already on
        disk is treated as already computed, so it can be skipped instead of
        recomputed. Any error while probing (missing store, unreadable group,
        etc.) is treated as "does not exist" so the caller falls back to
        (re)computing the group.
        """
        try:
            zarr.open_group(self._group_path(group), mode="r")
            return True
        except Exception:
            return False

    def initialize_root(
        self, description: str | None = None, overwrite: bool = True
    ) -> None:
        """Create the root group of the store (metadata only).

        Should be called by exactly **one** rank before any group is created.
        The per-event data lives in child groups created by
        :meth:`initialize_group`.

        ``overwrite`` controls whether an existing store is wiped. When ``True``
        the root is (re)created with ``mode="w"``, discarding any existing
        groups. When ``False`` an existing store is opened in append mode
        (``mode="a"``) so previously computed groups are preserved for a resume;
        the root is created if it does not exist yet.

        Note: the root metadata is **not** consolidated here. Consolidation must
        happen *after* every child group exists (see :meth:`finalize`), otherwise
        the consolidated root would record an empty member list and readers would
        not discover any events.
        """
        root = zarr.open_group(self.store_path, mode="w" if overwrite else "a")
        root.attrs["description"] = (
            description or "ERA5 forecast fields stacked into feature dimension."
        )

    def initialize_group(
        self, group: ForecastGroup, description: str | None = None
    ) -> None:
        """Create one event group's metadata (coords + empty ``fields``).

        This should be called by exactly **one** rank before any rank writes
        data into ``group``. It allocates the ``fields`` array *schema only* (no
        data buffers) directly through ``zarr`` and writes the coordinate arrays
        via ``xarray``. Nothing proportional to the full array size is ever held
        in memory, and no ``dask`` dependency is required.

        The chunks left empty by this step are filled in later by each rank via
        :meth:`write_block` region writes.
        """

        group_path = self._group_path(group)

        # Crop the group's spatial grid to the event bounding box.
        self.resolve_group_bbox(group)
        lat0, lat1 = group.lat_slice
        group_longitudes = self._group_longitudes(group)

        kept_features = group.global_features(self.n_feature)
        shape = (
            group.n_time,
            group.n_lead,
            self.n_ensemble,
            len(kept_features),
            group.n_lat,
            group.n_lon,
        )

        # 1. Write the coordinate variables (small) with xarray. This creates
        #    the child Zarr group and all coordinate arrays. Latitude/longitude
        #    are cropped to the event bounding box so the group's grid matches
        #    the event footprint exactly.

        # Extract case attributes for Zarr. ``lead_time`` is also stored as the
        # length-1 ``lead_time`` coordinate; recording it as a scalar attribute
        # too lets readers filter groups by lead time without opening the
        # coordinate array (the store now writes one group per (event, lead)).
        case_attrs = {
            key: value
            for key, value in (
                ("event_type", group.event_type),
                ("case_id", group.case_id),
                (
                    "lead_time",
                    int(group.lead_times[0]) if group.n_lead == 1 else None,
                ),
                (
                    "phase_offset",
                    None if group.phase_offset is None else int(group.phase_offset),
                ),
            )
            if value is not None
        }

        coords = xr.Dataset(
            coords={
                "time": ("time", group.times),
                "lead_time": ("lead_time", group.lead_times),
                "ensemble": ("ensemble", self.ensemble_members),
                "feature": (
                    "feature",
                    np.array(
                        [self.feature_names[i] for i in kept_features], dtype="<U24"
                    ),
                ),
                "latitude": ("latitude", self.latitudes[lat0:lat1].astype("float32")),
                "longitude": ("longitude", group_longitudes.astype("float32")),
            },
            attrs={
                "description": description
                or f"ERA5 forecast fields for event '{group.name}'.",
                "bbox_lat_min": float(group.lat_min),
                "bbox_lat_max": float(group.lat_max),
                "bbox_lon_min": float(group.lon_min),
                "bbox_lon_max": float(group.lon_max),
                **case_attrs,
            },
        )
        coords.to_zarr(group_path, mode="w", consolidated=False)

        # 2. Allocate the ``fields`` array as metadata only (no chunks are
        #    written until data is stored). ``fill_value`` marks unwritten
        #    chunks; region writes will populate them. ``dimension_names`` lets
        #    xarray read the array back with the correct dimensions (zarr v3).
        root = zarr.open_group(group_path, mode="a")
        root.create_array(
            self.VAR_NAME,
            shape=shape,
            chunks=self._chunks(group),
            dtype=self.dtype,
            fill_value=float("nan"),
            dimension_names=list(self._dims),
            overwrite=True,
            compressors=self.compressors,
        )

        # 3. Consolidate metadata for fast opening.
        zarr.consolidate_metadata(group_path)

    def finalize(self) -> None:
        """Consolidate the root metadata once every event group exists.

        Consolidating the root records all child groups in the root's
        ``zarr.json`` so readers can enumerate events via
        ``zarr.open_group(store).groups()`` (or ``xr.open_datatree``). Must be
        called by exactly **one** rank *after* all :meth:`initialize_group`
        calls have completed.
        """
        zarr.consolidate_metadata(self.store_path)

    # -- region helpers ----------------------------------------------------

    def write_block(
        self,
        data: np.ndarray | torch.Tensor,
        *,
        group: ForecastGroup,
        region: ForecastRegion,
        lead_hours: int,
        times: list[datetime] | np.ndarray,
    ) -> None:
        """Write one rank-local block into an event group using a region write.

        The block is expected on the *global* spatial grid; it is cropped to the
        group's bounding box (resolved during :meth:`initialize_group`) and
        written into the group's cropped ``fields`` array using group-local
        lat/lon indices.

        Parameters
        ----------
        data : np.ndarray | torch.Tensor
            Block with shape
            ``(time, lead_time, ensemble, feature, latitude, longitude)`` on the
            global spatial grid, matching the extents of ``region`` and
            ``times``.
        group : ForecastGroup
            The event group to write into (defines the ``time`` / ``lead_time``
            coordinates and the ``(lat, lon)`` bounding box for this write).
        region : ForecastRegion
            The half-open extents (into the global arrays) owned by this rank.
        lead_hours : int
            Lead time (hours) this block corresponds to.
        times : list[datetime] | np.ndarray
            The forecast valid times for the ``time`` axis of ``data``. They
            must be contiguous in the group's ``time`` coordinate.
        """
        if isinstance(data, torch.Tensor):
            data = data.detach().to(torch.float32).cpu().numpy()
        data = np.ascontiguousarray(data, dtype=self.dtype)

        # Resolve the contiguous time slice within this event group.
        time_idxs = [group.time_index(t) for t in times]
        t0, t1 = min(time_idxs), max(time_idxs) + 1
        if t1 - t0 != len(time_idxs):
            raise ValueError(
                "Times passed to write_block must map to a contiguous range "
                f"in the group time coordinate; got indices {time_idxs}."
            )

        lead_idx = group.lead_index(lead_hours)

        if group.lat_slice is None or group.lon_span is None:
            # The group grid must already be resolved by ``initialize_group``;
            # resolve on the fly as a safety net for out-of-order callers.
            self.resolve_group_bbox(group)

        # Global lat/lon extents of the group's cropped grid.
        g_lat0, g_lat1 = group.lat_slice

        # The incoming ``data`` is **rank-local**: its spatial axes span this
        # rank's shard, i.e. ``data[..., 0, 0]`` is global index
        # ``(region.latitude[0], region.longitude[0])``. To crop it to the event
        # bounding box we work in *global* indices first, then translate to
        # data-local indices by subtracting the region origin.
        r_lat0, r_lat1 = region.latitude

        # Global latitude extent this rank writes: the intersection of the
        # group's bounding box with this rank's owned shard.
        lat_i_min = max(g_lat0, r_lat0)
        lat_i_max = min(g_lat1, r_lat1)
        if lat_i_max - lat_i_min <= 0:
            return

        # Which of this rank's channels this group keeps, and where they sit on
        # the group's (possibly subset) feature axis.
        local_features, feature_extent = group.rank_features(
            self.n_feature, region.feature
        )
        if local_features.size == 0:
            # This rank owns none of the channels this event stores.
            return

        # A box crossing the grid seam splits into two runs, so this rank may
        # contribute to either or both. Each run is one region write.
        for lon_i_min, lon_i_max, group_position in group.rank_lon_runs(
            self.n_lon, region.longitude
        ):
            data_bb = data[
                ...,
                local_features,
                lat_i_min - r_lat0 : lat_i_max - r_lat0,
                lon_i_min - region.longitude[0] : lon_i_max - region.longitude[0],
            ]
            if data_bb.shape[-1] == 0 or data_bb.shape[-2] == 0:
                continue

            expected = (
                t1 - t0,
                1,
                region.ensemble[1] - region.ensemble[0],
                feature_extent[1] - feature_extent[0],
                lat_i_max - lat_i_min,
                lon_i_max - lon_i_min,
            )
            if tuple(data_bb.shape) != expected:
                raise ValueError(
                    f"Data shape {tuple(data_bb.shape)} does not match region "
                    f"extents {expected} (time, lead_time, ensemble, feature, "
                    "latitude, longitude)."
                )

            # Build the xarray block. It carries a length-1 lead_time dimension
            # so the region write targets a single lead-time slice.
            block = xr.Dataset({self.VAR_NAME: (self._dims, data_bb)})

            # Latitude is a plain crop, so subtract the group's origin. Longitude
            # is cyclic, so the run already carries its position on the group axis.
            region_selector = {
                "time": slice(t0, t1),
                "lead_time": slice(lead_idx, lead_idx + 1),
                "ensemble": slice(region.ensemble[0], region.ensemble[1]),
                "feature": slice(feature_extent[0], feature_extent[1]),
                "latitude": slice(lat_i_min - g_lat0, lat_i_max - g_lat0),
                "longitude": slice(
                    group_position, group_position + (lon_i_max - lon_i_min)
                ),
            }

            block.to_zarr(
                self._group_path(group),
                mode="r+",
                region=region_selector,
                consolidated=False,
            )
