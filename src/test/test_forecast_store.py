"""Tests that written forecast blocks land on the requested bounding box."""

from datetime import datetime

import numpy as np
import pytest
import xarray as xr

from gb.forecast_store import ForecastGroup, ForecastRegion, ForecastZarrStore

pytestmark = [
    pytest.mark.filterwarnings("ignore::zarr.errors.UnstableSpecificationWarning"),
    pytest.mark.filterwarnings("ignore:Consolidated metadata.*:UserWarning"),
]

GRID_STEP = 0.25
LATITUDES = np.arange(90.0, -90.0, -GRID_STEP)  # ERA5 order: descending
LONGITUDES = np.arange(0.0, 360.0, GRID_STEP)  # ERA5 convention: 0..360
TIMES = np.array(["2021-06-20T00:00"], dtype="datetime64[ns]")
VALID_TIMES = [datetime(2021, 6, 20, 0)]
LEAD_HOURS = 12


def make_store(store_path) -> ForecastZarrStore:
    """Create an ERA5-shaped store with a single member and feature."""
    return ForecastZarrStore(
        store_path=str(store_path),
        feature_names=["2t"],
        latitudes=LATITUDES,
        longitudes=LONGITUDES,
        ensemble_members=np.array([1]),
    )


def make_group(bounds) -> ForecastGroup:
    """Create a one-time, one-lead event group for ``bounds``."""
    lat_min, lat_max, lon_min, lon_max = bounds
    return ForecastGroup(
        name="event",
        times=TIMES,
        lead_times=np.array([LEAD_HOURS]),
        lat_min=lat_min,
        lat_max=lat_max,
        lon_min=lon_min,
        lon_max=lon_max,
    )


def write_box(store, group, longitude_shard=None):
    """Write ones over the group's box; return its reopened ``fields`` slice.

    ``longitude_shard`` simulates a rank owning only part of the global grid, so
    ``data`` is shard-sized exactly as inference produces it.
    """
    lon_extent = longitude_shard or (0, store.n_lon)
    region = ForecastRegion(
        feature=(0, store.n_feature),
        longitude=lon_extent,
        latitude=(0, store.n_lat),
        ensemble=(0, store.n_ensemble),
    )
    data = np.ones(
        (
            group.n_time,
            1,
            store.n_ensemble,
            store.n_feature,
            store.n_lat,
            lon_extent[1] - lon_extent[0],
        ),
        dtype=np.float32,
    )
    store.write_block(
        data,
        group=group,
        region=region,
        lead_hours=LEAD_HOURS,
        times=VALID_TIMES,
    )
    with xr.open_zarr(store._group_path(group), consolidated=False) as dataset:
        return dataset


def written_extent(dataset, var="fields"):
    """Return the (latitude, longitude) coordinate values that hold data."""
    values = dataset[var].isel(time=0, lead_time=0, ensemble=0, feature=0).values
    lats = dataset["latitude"].values[np.isfinite(values).any(axis=1)]
    lons = dataset["longitude"].values[np.isfinite(values).any(axis=0)]
    return lats, lons


@pytest.mark.parametrize(
    ("bounds", "expected_lat", "expected_lon"),
    [
        # EWB signed longitudes must normalize onto the 0..360 store grid.
        pytest.param(
            (39.0, 65.0, -136.75, -105.75),
            (39.0, 65.0),
            (223.25, 254.25),
            id="negative-longitudes-normalize",
        ),
        # Already-positive bounds must be left where they are.
        pytest.param(
            (40.25, 65.25, 15.25, 56.0),
            (40.25, 65.25),
            (15.25, 56.0),
            id="positive-longitudes-unchanged",
        ),
    ],
)
def test_group_grid_covers_requested_bounds(
    tmp_path, bounds, expected_lat, expected_lon
):
    """The group's cropped grid must contain the requested geographic box."""
    store = make_store(tmp_path / "forecast.zarr")
    group = make_group(bounds)
    store.resolve_group_bbox(group)

    lat0, lat1 = group.lat_slice
    grid_lat = LATITUDES[lat0:lat1]
    grid_lon = store._group_longitudes(group)

    assert grid_lat.min() <= expected_lat[0], "box's southern edge not covered"
    assert grid_lat.max() >= expected_lat[1], "box's northern edge not covered"
    assert grid_lon.min() <= expected_lon[0], "box's western edge not covered"
    assert grid_lon.max() >= expected_lon[1], "box's eastern edge not covered"
    # A margin is expected; a whole extra degree is an offset, not a margin.
    assert expected_lon[0] - grid_lon.min() <= 2 * GRID_STEP
    assert grid_lon.max() - expected_lon[1] <= 2 * GRID_STEP


@pytest.mark.parametrize(
    ("lat_min", "lat_max", "edge"),
    [
        pytest.param(60.0, 90.0, "north", id="box-touching-north-pole"),
        pytest.param(-89.75, -60.0, "south", id="box-touching-south-pole"),
    ],
)
def test_polar_box_is_clamped(tmp_path, lat_min, lat_max, edge):
    """A box reaching a pole must still produce a valid grid, not an empty one."""
    store = make_store(tmp_path / "forecast.zarr")
    group = make_group((lat_min, lat_max, 0.0, 30.0))
    store.resolve_group_bbox(group)

    assert group.n_lat > 0, f"{edge} polar box produced an empty latitude grid"
    grid_lat = LATITUDES[group.lat_slice[0] : group.lat_slice[1]]
    assert grid_lat.min() <= lat_min
    assert grid_lat.max() >= min(lat_max, LATITUDES.max())


@pytest.mark.parametrize(
    ("world_size", "rank"),
    [
        pytest.param(2, 0, id="world2-rank0-lon-not-split"),
        pytest.param(4, 0, id="world4-rank0-offset-0"),
        pytest.param(4, 2, id="world4-rank2-offset-720"),
        pytest.param(8, 6, id="world8-rank6-offset-1080"),
    ],
)
def test_rank_local_block_is_written(tmp_path, world_size, rank):
    """A rank must write its share of the box from a shard-sized block."""
    col = rank // 2 if world_size > 1 else 0
    groups = max(world_size // 2, 1)
    lon_start = col * len(LONGITUDES) // groups
    lon_stop = (col + 1) * len(LONGITUDES) // groups

    store = make_store(tmp_path / "forecast.zarr")
    group = make_group((39.0, 65.0, -136.75, -105.75))
    store.initialize_root("rank-local block test")
    store.initialize_group(group)

    dataset = write_box(store, group, longitude_shard=(lon_start, lon_stop))
    _, written_lon = written_extent(dataset)

    runs = group.rank_lon_runs(store.n_lon, (lon_start, lon_stop))
    if not runs:
        assert written_lon.size == 0
        return
    assert written_lon.size, f"rank {rank} of {world_size} wrote nothing"
    # The rank's runs, expressed as positions on the group's longitude axis.
    positions = np.concatenate(
        [np.arange(pos, pos + (hi - lo)) for lo, hi, pos in runs]
    )
    group_longitudes = store._group_longitudes(group)
    np.testing.assert_array_equal(
        np.sort(written_lon), np.sort(group_longitudes[positions].astype("float32"))
    )


@pytest.mark.parametrize(
    ("bounds", "expected_runs", "expected_lon"),
    [
        # events_freeze_91 "December 2022 Europe" -- straddles the prime
        # meridian, so on a 0..360 grid its indices run 1403..1439 then 0..64.
        pytest.param(
            (45.0, 60.0, -9.0, 16.0), 2, (-9.0, 16.0), id="crosses-prime-meridian"
        ),
        # No seam crossing: a single contiguous run, coordinate left on 0..360.
        pytest.param(
            (39.0, 65.0, -136.75, -105.75),
            1,
            (223.25, 254.25),
            id="single-run-no-seam",
        ),
        pytest.param(
            (-20.0, 20.0, 170.0, -170.0), 1, (170.0, 190.0), id="crosses-dateline"
        ),
    ],
)
def test_seam_crossing_box_builds_a_monotonic_grid(
    tmp_path, bounds, expected_runs, expected_lon
):
    """A box spanning the grid seam must still yield a usable longitude axis.

    Wrapping each bound independently mod 360 gives ``start > stop`` and a
    negative width, which crashed group creation. The span is cyclic instead,
    and the coordinate is unwrapped so xarray and EWB can select against it.
    """
    store = make_store(tmp_path / "forecast.zarr")
    group = make_group(bounds)
    store.resolve_group_bbox(group)

    assert group.n_lon > 0, "seam-crossing box produced an empty longitude grid"
    assert len(group.lon_runs(store.n_lon)) == expected_runs

    longitudes = store._group_longitudes(group)
    assert np.all(np.diff(longitudes) > 0), "longitude coordinate is not monotonic"
    assert longitudes.min() <= expected_lon[0]
    assert longitudes.max() >= expected_lon[1]


@pytest.mark.parametrize("world_size", [1, 4, 8])
def test_seam_crossing_box_is_written_by_every_rank(tmp_path, world_size):
    """Ranks either side of the seam must both contribute to a wrapping box."""
    store = make_store(tmp_path / "forecast.zarr")
    group = make_group((45.0, 60.0, -9.0, 16.0))
    store.initialize_root("seam write test")
    store.initialize_group(group)

    for rank in range(world_size):
        col = rank // 2 if world_size > 1 else 0
        groups = max(world_size // 2, 1)
        lon_start = col * len(LONGITUDES) // groups
        lon_stop = (col + 1) * len(LONGITUDES) // groups
        # Each cell carries its own global longitude index, so a run written to
        # the wrong position on the group axis is detectable.
        data = np.broadcast_to(
            np.arange(lon_start, lon_stop, dtype=np.float32),
            (
                group.n_time,
                1,
                store.n_ensemble,
                store.n_feature,
                store.n_lat,
                lon_stop - lon_start,
            ),
        ).copy()
        store.write_block(
            data,
            group=group,
            region=ForecastRegion(
                feature=(0, store.n_feature),
                longitude=(lon_start, lon_stop),
                latitude=(0, store.n_lat),
                ensemble=(0, store.n_ensemble),
            ),
            lead_hours=LEAD_HOURS,
            times=VALID_TIMES,
        )

    with xr.open_zarr(store._group_path(group), consolidated=False) as dataset:
        written = (
            dataset["fields"].isel(time=0, lead_time=0, ensemble=0, feature=0).values
        )
    assert np.isfinite(written).all(), "seam-crossing box left gaps"
    expected = (group.lon_span[0] + np.arange(group.n_lon)) % store.n_lon
    np.testing.assert_array_equal(written[0, :], expected)


@pytest.mark.parametrize(
    ("bounds", "feature_indices", "expect_widened"),
    [
        # Inside one longitude shard: stays tightly cropped.
        pytest.param(
            (39.0, 65.0, -136.75, -105.75), None, False, id="single-shard-stays-tight"
        ),
        # Straddles the shard boundary at 180 deg.
        pytest.param((45.0, 60.0, -9.0, 16.0), None, False, id="seam-stays-tight"),
        # Non-contiguous feature subset spanning both feature shards.
        pytest.param(
            (39.0, 65.0, -136.75, -105.75),
            np.array([0, 1, 2, 50, 60]),
            False,
            id="feature-subset-across-shards",
        ),
    ],
)
def test_no_chunk_is_written_by_two_ranks(
    tmp_path, bounds, feature_indices, expect_widened
):
    """No Zarr chunk may be touched by more than one rank."""
    feature_shard, lon_shard = 42, len(LONGITUDES) // 2
    store = ForecastZarrStore(
        store_path=str(tmp_path / "forecast.zarr"),
        feature_names=[f"f{i}" for i in range(84)],
        latitudes=LATITUDES,
        longitudes=LONGITUDES,
        ensemble_members=np.array([1]),
        feature_chunk=feature_shard,
        longitude_chunk=lon_shard,
    )
    group = make_group(bounds)
    group.feature_indices = feature_indices
    store.resolve_group_bbox(group)

    natural = round(((bounds[3] - bounds[2]) % 360) / GRID_STEP) + 3
    assert (group.n_lon > natural) is expect_widened

    _, _, _, feature_chunk, _, lon_chunk = store._chunks(group)
    owners: dict[tuple[int, int], set[int]] = {}
    for rank in range(4):
        row, col = rank % 2, rank // 2
        in_var = (row * 42, (row + 1) * 42)
        shard = (col * lon_shard, (col + 1) * lon_shard)
        _, extent = group.rank_features(store.n_feature, in_var)
        if extent[1] <= extent[0]:
            continue
        for lo, hi, pos in group.rank_lon_runs(store.n_lon, shard):
            for fc in range(
                extent[0] // feature_chunk, (extent[1] - 1) // feature_chunk + 1
            ):
                for lc in range(pos // lon_chunk, (pos + hi - lo - 1) // lon_chunk + 1):
                    owners.setdefault((fc, lc), set()).add(rank)

    clashes = {key: ranks for key, ranks in owners.items() if len(ranks) > 1}
    assert not clashes, f"chunks written by multiple ranks: {clashes}"


def test_group_is_cropped_to_the_event(tmp_path):
    """Each event group stores only its own box, not the global grid."""
    store = make_store(tmp_path / "forecast.zarr")
    group = make_group((39.0, 65.0, -136.75, -105.75))
    store.initialize_root("crop test")
    store.initialize_group(group)

    with xr.open_zarr(store._group_path(group), consolidated=False) as dataset:
        assert dataset.sizes["latitude"] == group.n_lat < len(LATITUDES)
        assert dataset.sizes["longitude"] == group.n_lon < len(LONGITUDES)
        # ~105 x 125 cells plus the one-cell margin on each side.
        assert dataset.sizes["latitude"] < 120
        assert dataset.sizes["longitude"] < 140


def test_separate_events_get_separate_groups(tmp_path):
    """Two events must land in two independently readable groups."""
    import zarr

    store = make_store(tmp_path / "forecast.zarr")
    store.initialize_root("multi-event test")
    boxes = {
        "pacific_northwest": (39.0, 65.0, -136.75, -105.75),
        "western_russia": (40.25, 65.25, 15.25, 56.0),
    }
    for name, bounds in boxes.items():
        group = make_group(bounds)
        group.name = name
        store.initialize_group(group)
    store.finalize()

    root = zarr.open_group(str(tmp_path / "forecast.zarr"), mode="r")
    assert sorted(k for k, _ in root.groups()) == sorted(boxes)


def test_group_records_the_ewb_case_it_came_from(tmp_path):
    """The group carries the YAML's event_type and case_id, not the internal id."""
    store = make_store(tmp_path / "forecast.zarr")
    group = make_group((39.0, 65.0, -136.75, -105.75))
    group.name = "events_tropical_cyclone_167"
    group.event_type = "tropical_cyclone"
    group.case_id = 167
    store.initialize_root("case attribute test")
    store.initialize_group(group)

    with xr.open_zarr(store._group_path(group), consolidated=False) as dataset:
        assert dataset.attrs["event_type"] == "tropical_cyclone"
        assert dataset.attrs["case_id"] == 167
        assert dataset.attrs["bbox_lat_min"] == 39.0


def test_case_attributes_are_omitted_when_unknown(tmp_path):
    """A group built without EWB metadata writes no null-valued attributes."""
    store = make_store(tmp_path / "forecast.zarr")
    group = make_group((39.0, 65.0, -136.75, -105.75))
    store.initialize_root("no case attribute test")
    store.initialize_group(group)

    with xr.open_zarr(store._group_path(group), consolidated=False) as dataset:
        assert "event_type" not in dataset.attrs
        assert "case_id" not in dataset.attrs
