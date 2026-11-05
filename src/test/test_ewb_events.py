"""Tests for loading ExtremeWeatherBench event metadata."""

from __future__ import annotations

import dataclasses as dc
import sys
import types
from datetime import datetime

import pytest


@dc.dataclass
class StubEvent:
    # Mirrors beast.data.era5.Event, which carries the internal unique id.
    id: str
    event_type: str
    title: str
    start_date: datetime
    end_date: datetime
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float


@pytest.fixture
def load_ewb_events(monkeypatch):
    era5 = types.ModuleType("beast.data.era5")
    era5.Event = StubEvent
    data = types.ModuleType("beast.data")
    data.era5 = era5
    beast = types.ModuleType("beast")
    beast.data = data

    monkeypatch.setitem(sys.modules, "beast", beast)
    monkeypatch.setitem(sys.modules, "beast.data", data)
    monkeypatch.setitem(sys.modules, "beast.data.era5", era5)
    monkeypatch.delitem(sys.modules, "gb.ewb", raising=False)

    from gb.ewb import load_ewb_events

    return load_ewb_events


@pytest.fixture
def parse_case_id(load_ewb_events):
    """`parse_case_id`, with ``beast`` stubbed out by the ``load_ewb_events`` fixture."""
    from gb.ewb import parse_case_id

    return parse_case_id


def test_load_selected_ewb_events(tmp_path, load_ewb_events):
    events_yaml = tmp_path / "events.yaml"
    events_yaml.write_text(
        """
            - case_id_number: 1
              title: 2021 Pacific Northwest
              start_date: 2021-06-20 00:00:00
              end_date: 2021-07-03 00:00:00
              location:
                type: bounded_region
                parameters:
                  latitude_min: 39.0
                  latitude_max: 65.0
                  longitude_min: -136.75
                  longitude_max: -105.75
              event_type: heat_wave
            - case_id_number: 30
              title: 2021 Texas
              start_date: 2021-02-10 12:00:00
              end_date: 2021-02-22 00:00:00
              location:
                type: bounded_region
                parameters:
                  latitude_min: 23.75
                  latitude_max: 52.75
                  longitude_min: -107.25
                  longitude_max: -82.25
              event_type: freeze
            - case_id_number: 18
              title: 2021 Western Russia
              start_date: 2021-06-18 00:00:00
              end_date: 2021-06-30 00:00:00
              location:
                type: bounded_region
                parameters:
                  latitude_min: 40.25
                  latitude_max: 65.25
                  longitude_min: 15.25
                  longitude_max: 56.0
              event_type: heat_wave
            - case_id_number: 27
              title: 2021 Canada Plains
              start_date: 2021-05-30 00:00:00
              end_date: 2021-06-09 00:00:00
              location:
                type: bounded_region
                parameters:
                  latitude_min: 39.5
                  latitude_max: 53.5
                  longitude_min: -120.0
                  longitude_max: -87.0
              event_type: heat_wave
            """,
        encoding="utf-8",
    )

    # Selection is by internal id, and both filters are always supplied as
    # lists, exactly as setup.py passes them from the config.
    events = load_ewb_events(
        events_yaml,
        case_ids=["events_heat_wave_1", "events_heat_wave_18", "events_heat_wave_27"],
        event_types=[],
    )

    assert [event.title for event in events] == [
        "2021 Pacific Northwest",
        "2021 Western Russia",
        "2021 Canada Plains",
    ]
    assert events[0] == StubEvent(
        id="events_heat_wave_1",
        event_type="heat_wave",
        title="2021 Pacific Northwest",
        start_date=datetime(2021, 6, 20),
        end_date=datetime(2021, 7, 3),
        lat_min=39.0,
        lat_max=65.0,
        lon_min=-136.75,
        lon_max=-105.75,
    )
    assert (events[1].lat_min, events[1].lon_max) == (40.25, 56.0)
    assert (events[2].lat_max, events[2].lon_min) == (53.5, -120.0)


def test_load_event_type_from_yaml_directory(tmp_path, load_ewb_events):
    (tmp_path / "heat.yaml").write_text(
        """
        - case_id_number: 1
          title: Heat
          start_date: "2021-06-20T00:00:00"
          end_date: "2021-06-21T00:00:00"
          location:
            type: bounded_region
            parameters:
              latitude_min: 39
              latitude_max: 65
              longitude_min: -136.75
              longitude_max: -105.75
          event_type: heat_wave
        """,
        encoding="utf-8",
    )
    (tmp_path / "freeze.yml").write_text(
        """
        events:
          - case_id_number: 30
            title: Freeze
            start_date: 2021-02-10
            end_date: 2021-02-11
            location:
              type: bounded_region
              parameters:
                latitude_min: 23.75
                latitude_max: 52.75
                longitude_min: -107.25
                longitude_max: -82.25
            event_type: freeze
        """,
        encoding="utf-8",
    )

    events = load_ewb_events(tmp_path, case_ids=[], event_types=["freeze"])

    assert [event.title for event in events] == ["Freeze"]
    assert events[0].start_date == datetime(2021, 2, 10)


HEAT_WAVE_CASE = """
        - case_id_number: 1
          title: Event
          start_date: 2021-01-01
          end_date: 2021-01-02
          location:
            type: bounded_region
            parameters:
              latitude_min: 1
              latitude_max: 2
              longitude_min: 3
              longitude_max: 4
          event_type: heat_wave
        """

TRACK_CASE = """
        - case_id_number: 1
          title: Event
          start_date: 2021-01-01
          end_date: 2021-01-02
          location:
            type: track
            parameters: {}
          event_type: tropical_cyclone
        """


@pytest.mark.parametrize(
    ("yaml_text", "kwargs", "match"),
    [
        pytest.param(
            HEAT_WAVE_CASE,
            {"case_ids": ["events_heat_wave_18"]},
            "events_heat_wave_18",
            id="unknown-case-id",
        ),
        pytest.param(TRACK_CASE, {}, "bounded_region", id="non-bounded-region"),
        # Same file, event type and case id: still a genuine duplicate.
        pytest.param(
            HEAT_WAVE_CASE * 2, {}, "Duplicate EWB case", id="duplicate-within-one-file"
        ),
    ],
)
def test_invalid_events_are_rejected(
    tmp_path, load_ewb_events, yaml_text, kwargs, match
):
    events_yaml = tmp_path / "events.yaml"
    events_yaml.write_text(yaml_text, encoding="utf-8")

    # Both filters are always supplied as lists, as setup.py does from config.
    kwargs = {"case_ids": [], "event_types": [], **kwargs}
    with pytest.raises(ValueError, match=match):
        load_ewb_events(events_yaml, **kwargs)


@pytest.mark.parametrize(
    ("internal_id", "case_id"),
    [
        # Both the file stem and the event type contain underscores.
        ("events_heat_wave_1", 1),
        ("tropical_cyclone_2020_tropical_cyclone_167", 167),
        ("marginal_freeze_0", 0),
        ("a_b_c_severe_convection_12345", 12345),
    ],
)
def test_parse_case_id(parse_case_id, internal_id, case_id):
    """The raw EWB case id is recoverable from the internal id."""
    assert parse_case_id(internal_id) == case_id


@pytest.mark.parametrize(
    "internal_id",
    [
        pytest.param("events_heat_wave", id="no-trailing-number"),
        pytest.param("1", id="no-separator"),
        pytest.param("events_heat_wave_1a", id="non-numeric-tail"),
        pytest.param("", id="empty"),
    ],
)
def test_parse_case_id_rejects_non_internal_ids(parse_case_id, internal_id):
    """A malformed id raises rather than yielding a wrong case id."""
    with pytest.raises(ValueError, match="not an internal EWB id"):
        parse_case_id(internal_id)


def test_loaded_events_carry_a_parsable_case_id(
    tmp_path, load_ewb_events, parse_case_id
):
    """Every id `load_ewb_events` produces yields back its source case id."""
    (tmp_path / "events.yaml").write_text(HEAT_WAVE_CASE, encoding="utf-8")

    events = load_ewb_events(tmp_path, case_ids=[], event_types=[])

    assert [parse_case_id(event.id) for event in events] == [1]
