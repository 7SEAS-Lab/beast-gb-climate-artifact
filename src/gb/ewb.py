"""Utility function for loading ExtremeWeatherBench event metadata."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import date, datetime, time
from pathlib import Path
from typing import Any

import yaml
from beast.data.era5 import Event


def parse_case_id(internal_id: str) -> int:
    """Recover the EWB ``case_id_number`` for attribute in zarr.

    Parameters
    ----------
    internal_id : str
        An internal id as built by :func:`load_ewb_events`.

    Returns
    -------
    int
        The EWB ``case_id_number`` as it appears in the source YAML.

    Raises
    ------
    ValueError
        If ``internal_id`` does not end in ``_<digits>``.
    """
    _, separator, tail = internal_id.rpartition("_")
    if not separator or not tail.isdigit():
        raise ValueError(
            f"{internal_id!r} is not an internal EWB id: expected it to end in "
            "'_<case_id>'."
        )
    return int(tail)


def load_ewb_events(
    source: str | Path,
    *,
    case_ids: Iterable[int] | None = None,
    event_types: Iterable[str] | None = None,
) -> list[Event]:
    """Load ExtremeWeatherBench events from the EWB events yaml file.

    The source points to an events file or a directory containing multiple such files. Per default all events will
    be loaded, however via the ``case_ids`` and ``event_types`` parameters it is possible to select only specific cases
    or types of events. EWB does not number its cases uniquely across files, so a ``case_id_number`' may select more
    than one case. Pass ``event_types`` as well to narrow it down. If multiple events are selected a warning is raised.

    Parameters
    ----------
    source : str or pathlib.Path
        EWB ``events.yaml`` file, or a directory containing one or more event files.
    case_ids : iterable of int, optional
        Load only cases whose ``case_id_number`` occurs in this collection.
    event_types : iterable of str, optional
        Load only cases whose ``event_type`` occurs in this collection, such as ``"heat_wave"``.

    Returns
    -------
    list of beast.data.era5.Event
        Selected cases as `Event` objects.
    """

    # Resolve either the yaml file or a directory.
    source_path = Path(source).expanduser()
    if source_path.is_file():
        if source_path.suffix.lower() not in {".yaml", ".yml"}:
            raise ValueError(f"EWB event file must be YAML: {source_path}")
        yaml_paths = [source_path]
    elif source_path.is_dir():
        yaml_paths = sorted(
            path
            for path in source_path.rglob("*")
            if path.is_file() and path.suffix.lower() in {".yaml", ".yml"}
        )
        if not yaml_paths:
            raise ValueError(
                f"No YAML files found below EWB event directory {source_path}"
            )
    else:
        raise FileNotFoundError(f"EWB event YAML path does not exist: {source_path}")

    # Select ids and event types if parameters provided.
    selected_ids = None if len(case_ids) == 0 else {case_id for case_id in case_ids}
    selected_types = (
        None
        if len(event_types) == 0
        else {str(event_type) for event_type in event_types}
    )

    events: list[Event] = []
    found_ids: set[int] = set()
    seen_keys: set[str] = set()
    # case_id_number -> the cases it selected, to warn about ambiguous requests.
    matched_cases: dict[int, list[str]] = {}

    for path in yaml_paths:
        with path.open("rb") as yaml_file:
            cases: Any = yaml.safe_load(yaml_file)
        if isinstance(cases, dict):
            cases = cases.get("events")
        if not isinstance(cases, list) or not all(
            isinstance(case, dict) for case in cases
        ):
            raise ValueError(
                f"{path}: expected a YAML list of EWB cases (or a mapping containing an 'events' list). Unknown format!"
            )

        for index, case in enumerate(cases):
            case_label = f"{path}: case at index {index}"
            try:
                case_id = int(case["case_id_number"])
                event_type = str(case["event_type"])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    f"{case_label} must define case_id_number and event_type."
                ) from error

            # Cases are only unique per - accross different files even the event_type and case_id can be identical.
            # Therefore we build a unique internal id based on file name, event type, and case id.
            internal_id = f"{path.stem}_{event_type}_{case_id}"
            if internal_id in seen_keys:
                raise ValueError(f"Duplicate EWB case {internal_id!r} in {path}.")
            seen_keys.add(internal_id)

            if selected_ids is not None and internal_id not in selected_ids:
                continue
            if selected_types is not None and event_type not in selected_types:
                continue

            location = case.get("location")
            if (
                not isinstance(location, dict)
                or location.get("type") != "bounded_region"
            ):
                # Currently EWB only has `bounded_region` location types - if they introduce other location types later
                # we will have to update code.
                raise ValueError(
                    f"{case_label} must use an EWB bounded_region location."
                )
            bounds = location.get("parameters")
            if not isinstance(bounds, dict):
                raise ValueError(
                    f"{case_label} must define location.parameters bounding-box values."
                )

            # Parse timestamps.
            try:
                timestamps = []
                for field in ("start_date", "end_date"):
                    value = case[field]
                    if isinstance(value, datetime):
                        timestamps.append(value)
                    elif isinstance(value, date):
                        timestamps.append(datetime.combine(value, time()))
                    elif isinstance(value, str):
                        timestamps.append(datetime.fromisoformat(value))
                    else:
                        raise TypeError

                event = Event(
                    id=internal_id,
                    event_type=event_type,
                    title=str(case["title"]),
                    start_date=timestamps[0],
                    end_date=timestamps[1],
                    lat_min=float(bounds["latitude_min"]),
                    lat_max=float(bounds["latitude_max"]),
                    lon_min=float(bounds["longitude_min"]),
                    lon_max=float(bounds["longitude_max"]),
                )
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    f"{case_label} is missing required EWB event metadata."
                ) from error

            if event.end_date < event.start_date:
                raise ValueError(
                    f"{case_label} ends before it starts: "
                    f"{event.start_date} > {event.end_date}."
                )

            events.append(event)
            found_ids.add(internal_id)
            matched_cases.setdefault(case_id, []).append(
                f"{internal_id} ({event.title})"
            )

    if selected_ids is not None:
        missing_ids = sorted(selected_ids - found_ids)
        if missing_ids:
            raise ValueError(
                f"EWB case IDs were not found after filtering: {missing_ids}"
            )

        # Avoiding making the unique_id necessary in the config.
        # Since case_id and event_type are not unique, a requested case_id may pull in more cases if using multiple
        # files in a directory. This is not necissarily a problem (unique_id handles it) but should raise a warning.
        ambiguous = {
            case_id: matches
            for case_id, matches in sorted(matched_cases.items())
            if len(matches) > 1
        }
        if ambiguous:
            extra = sum(len(matches) - 1 for matches in ambiguous.values())
            print(
                f"[ewb] WARNING: {len(ambiguous)} requested case ID(s) are not unique "
                f"across the EWB event files in {source_path}, so {len(events)} events "
                f"will run instead of {len(events) - extra}. Narrow the selection with "
                f"data.ewb_event_types, or point data.ewb_events_path at a single file."
            )
            for case_id, matches in ambiguous.items():
                print(f"[ewb]   case_id {case_id} matched {len(matches)} events:")
                for match in matches:
                    print(f"[ewb]     - {match}")
    if not events:
        raise ValueError("No EWB events matched the requested filters.")

    return events
