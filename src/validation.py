from __future__ import annotations

from typing import Any

import pandas as pd


# Metrics currently reliable for room-level analysis
SUPPORTED_ROOM_METRICS = {
    "co2",
    "temp",
    "humidity",
    "occupancy",
}

VALID_AGGREGATIONS = {
    "mean",
    "min",
    "max",
    "sum",
}

VALID_FREQUENCIES = {
    "hourly",
    "daily",
}

VALID_INTENTS = {
    "room_trend",
    "compare_rooms",
    "rank_rooms",
    "threshold",
}

DEFAULT_CHARTS = {
    "room_trend": "line",
    "compare_rooms": "bar",
    "rank_rooms": "bar",
    "threshold": "bar",
}

SUITABLE_CHARTS = {
    "room_trend": {"line"},
    "compare_rooms": {"bar"},
    "rank_rooms": {"bar"},
    "threshold": {"bar"},
}


def normalise_text(value: Any) -> str | None:
    """Convert text to lowercase and remove extra spaces."""

    if value is None:
        return None

    return str(value).strip().casefold()


def normalise_series(series: pd.Series) -> pd.Series:
    """Normalise a pandas text column safely."""

    return (
        series.astype("string")
        .fillna("")
        .str.strip()
        .str.casefold()
    )


def validation_result(
    valid: bool,
    errors: list[str],
    warnings: list[str],
    matching_rows: int,
    normalised_request: dict[str, Any],
) -> dict[str, Any]:
    """Create a consistent validation result."""

    return {
        "valid": valid,
        "errors": errors,
        "warnings": warnings,
        "matching_rows": matching_rows,
        "normalised_request": normalised_request,
    }


def validate_request(
    request: dict[str, Any],
    data: pd.DataFrame,
) -> dict[str, Any]:
    """
    Validate a structured SmartViz analytical request.

    The function checks:
    - required dataset columns;
    - intent;
    - metric availability and room mapping;
    - room names;
    - aggregation;
    - frequency;
    - date range;
    - chart suitability;
    - threshold and ranking parameters;
    - whether matching data exists.
    """

    errors: list[str] = []
    warnings: list[str] = []
    normalised_request: dict[str, Any] = request.copy()

    # ---------------------------------------------------------
    # 1. Validate input types
    # ---------------------------------------------------------

    if not isinstance(request, dict):
        return validation_result(
            valid=False,
            errors=["The request must be provided as a dictionary."],
            warnings=[],
            matching_rows=0,
            normalised_request={},
        )

    if not isinstance(data, pd.DataFrame):
        return validation_result(
            valid=False,
            errors=["The supplied data must be a pandas DataFrame."],
            warnings=[],
            matching_rows=0,
            normalised_request=normalised_request,
        )

    if data.empty:
        return validation_result(
            valid=False,
            errors=["The supplied dataset is empty."],
            warnings=[],
            matching_rows=0,
            normalised_request=normalised_request,
        )

    # ---------------------------------------------------------
    # 2. Validate required dataset columns
    # ---------------------------------------------------------

    required_columns = {
        "metric_name",
        "display_name",
        "aggregation",
        "frequency",
        "start_time",
        "value",
    }

    missing_columns = required_columns.difference(data.columns)

    if missing_columns:
        errors.append(
            "Dataset is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

        return validation_result(
            valid=False,
            errors=errors,
            warnings=warnings,
            matching_rows=0,
            normalised_request=normalised_request,
        )

    # ---------------------------------------------------------
    # 3. Validate intent
    # ---------------------------------------------------------

    intent = normalise_text(request.get("intent"))

    if intent is None:
        errors.append("An analytical intent must be provided.")

    elif intent not in VALID_INTENTS:
        errors.append(
            f"Intent '{request.get('intent')}' is not supported."
        )

    else:
        normalised_request["intent"] = intent

    # ---------------------------------------------------------
    # 4. Validate metric
    # ---------------------------------------------------------

    metric_original = (
        request.get("metric_name")
        or request.get("metric")
    )

    metric = normalise_text(metric_original)

    if metric is None:
        errors.append("A metric must be provided.")

    elif metric not in SUPPORTED_ROOM_METRICS:
        errors.append(
            f"Metric '{metric_original}' is not supported for "
            "reliable room-level analysis."
        )

    else:
        available_metrics = set(
            normalise_series(
                data["metric_name"].dropna()
            ).unique()
        )

        if metric not in available_metrics:
            errors.append(
                f"Metric '{metric_original}' does not exist "
                "in the dataset."
            )
        else:
            normalised_request["metric_name"] = metric

            metric_rows = data[
                normalise_series(data["metric_name"]) == metric
            ]

            mapped_rows = metric_rows[
                metric_rows["display_name"].notna()
            ]

            if mapped_rows.empty:
                errors.append(
                    f"Metric '{metric_original}' has no reliable "
                    "room mapping."
                )

    # ---------------------------------------------------------
    # 5. Validate aggregation
    # ---------------------------------------------------------

    aggregation = normalise_text(
        request.get("aggregation")
    )

    if aggregation is None:
        errors.append("An aggregation must be provided.")

    elif aggregation not in VALID_AGGREGATIONS:
        errors.append(
            f"Aggregation '{request.get('aggregation')}' "
            "is not supported."
        )

    else:
        normalised_request["aggregation"] = aggregation

    # ---------------------------------------------------------
    # 6. Validate frequency
    # ---------------------------------------------------------

    frequency = normalise_text(
        request.get("frequency")
    )

    if frequency is None:
        errors.append(
            "One frequency must be provided to prevent hourly "
            "and daily records from being mixed."
        )

    elif frequency not in VALID_FREQUENCIES:
        errors.append(
            f"Frequency '{request.get('frequency')}' "
            "is not supported."
        )

    else:
        normalised_request["frequency"] = frequency

    # ---------------------------------------------------------
    # 7. Read and validate room names
    # ---------------------------------------------------------

    requested_rooms = (
        request.get("room_names")
        or request.get("rooms")
    )

    single_room = (
        request.get("room")
        or request.get("room_name")
    )

    if requested_rooms is None and single_room is not None:
        requested_rooms = [single_room]

    if isinstance(requested_rooms, str):
        requested_rooms = [requested_rooms]

    if requested_rooms is None:
        requested_rooms = []

    available_room_lookup = {
        normalise_text(room): room
        for room in data["display_name"].dropna().unique()
    }

    valid_rooms: list[str] = []

    for room in requested_rooms:
        room_key = normalise_text(room)

        if room_key not in available_room_lookup:
            errors.append(
                f"Room '{room}' was not found."
            )
        else:
            canonical_room = available_room_lookup[room_key]

            if canonical_room not in valid_rooms:
                valid_rooms.append(canonical_room)

    if valid_rooms:
        normalised_request["room_names"] = valid_rooms

        if len(valid_rooms) == 1:
            normalised_request["room"] = valid_rooms[0]

    if intent == "room_trend" and len(valid_rooms) != 1:
        errors.append(
            "Room trend analysis requires exactly one valid room."
        )

    if intent == "compare_rooms" and len(valid_rooms) < 2:
        errors.append(
            "Room comparison requires at least two valid rooms."
        )

    # Ranking and threshold analysis do not require room names.

    # ---------------------------------------------------------
    # 8. Validate available dates
    # ---------------------------------------------------------

    dataset_times = pd.to_datetime(
        data["start_time"],
        utc=True,
        errors="coerce",
    )

    dataset_start = dataset_times.min()
    dataset_end = dataset_times.max()

    if pd.isna(dataset_start) or pd.isna(dataset_end):
        errors.append(
            "The dataset does not contain valid start-time values."
        )

        return validation_result(
            valid=False,
            errors=errors,
            warnings=warnings,
            matching_rows=0,
            normalised_request=normalised_request,
        )

    start_date_value = request.get("start_date")
    end_date_value = request.get("end_date")

    parsed_start = None
    parsed_end = None

    if start_date_value is not None:
        parsed_start = pd.to_datetime(
            start_date_value,
            utc=True,
            errors="coerce",
        )

        if pd.isna(parsed_start):
            errors.append(
                f"Start date '{start_date_value}' is invalid."
            )

        elif not dataset_start <= parsed_start <= dataset_end:
            errors.append(
                f"Start date must be between "
                f"{dataset_start.date()} and {dataset_end.date()}."
            )

        else:
            normalised_request["start_date"] = parsed_start

    if end_date_value is not None:
        parsed_end = pd.to_datetime(
            end_date_value,
            utc=True,
            errors="coerce",
        )

        if pd.isna(parsed_end):
            errors.append(
                f"End date '{end_date_value}' is invalid."
            )

        elif not dataset_start <= parsed_end <= dataset_end:
            errors.append(
                f"End date must be between "
                f"{dataset_start.date()} and {dataset_end.date()}."
            )

        else:
            normalised_request["end_date"] = parsed_end

    if (
        parsed_start is not None
        and parsed_end is not None
        and not pd.isna(parsed_start)
        and not pd.isna(parsed_end)
        and parsed_start > parsed_end
    ):
        errors.append(
            "Start date cannot be later than the end date."
        )

    # ---------------------------------------------------------
    # 9. Validate chart type
    # ---------------------------------------------------------

    chart_type = normalise_text(
        request.get("chart_type")
    )

    if intent in SUITABLE_CHARTS:
        allowed_charts = SUITABLE_CHARTS[intent]

        if chart_type is None:
            chart_type = DEFAULT_CHARTS[intent]
            normalised_request["chart_type"] = chart_type

            warnings.append(
                f"No chart type was provided. "
                f"'{chart_type}' was selected automatically."
            )

        elif chart_type not in allowed_charts:
            errors.append(
                f"Chart type '{request.get('chart_type')}' is "
                f"not suitable for '{intent}'. Suitable chart type: "
                f"{', '.join(sorted(allowed_charts))}."
            )

        else:
            normalised_request["chart_type"] = chart_type

    # ---------------------------------------------------------
    # 10. Validate intent-specific parameters
    # ---------------------------------------------------------

    if intent == "threshold":
        threshold = request.get("threshold")

        try:
            threshold = float(threshold)
            normalised_request["threshold"] = threshold
        except (TypeError, ValueError):
            errors.append(
                "Threshold analysis requires a numeric threshold."
            )

    if intent == "rank_rooms":
        top_n = request.get("top_n", 10)

        try:
            top_n = int(top_n)

            if top_n <= 0:
                raise ValueError

            normalised_request["top_n"] = top_n

        except (TypeError, ValueError):
            errors.append(
                "The top_n value must be a positive integer."
            )

    minimum_records = request.get("minimum_records", 100)

    try:
        minimum_records = int(minimum_records)

        if minimum_records < 1:
            raise ValueError

        normalised_request["minimum_records"] = minimum_records

    except (TypeError, ValueError):
        errors.append(
            "minimum_records must be a positive integer."
        )

    # Stop here when structural validation fails.
    if errors:
        return validation_result(
            valid=False,
            errors=errors,
            warnings=warnings,
            matching_rows=0,
            normalised_request=normalised_request,
        )

    # ---------------------------------------------------------
    # 11. Check actual aggregation and frequency availability
    # ---------------------------------------------------------

    metric_data = data[
        normalise_series(data["metric_name"]) == metric
    ].copy()

    available_aggregations = set(
        normalise_series(
            metric_data["aggregation"]
        ).unique()
    )

    if aggregation not in available_aggregations:
        errors.append(
            f"Aggregation '{aggregation}' is not available "
            f"for metric '{metric}'."
        )

    available_frequencies = set(
        normalise_series(
            metric_data["frequency"]
        ).unique()
    )

    if frequency not in available_frequencies:
        errors.append(
            f"Frequency '{frequency}' is not available "
            f"for metric '{metric}'."
        )

    if errors:
        return validation_result(
            valid=False,
            errors=errors,
            warnings=warnings,
            matching_rows=0,
            normalised_request=normalised_request,
        )

    # ---------------------------------------------------------
    # 12. Filter the exact requested data
    # ---------------------------------------------------------

    filtered = data[
        (
            normalise_series(data["metric_name"])
            == metric
        )
        &
        (
            normalise_series(data["aggregation"])
            == aggregation
        )
        &
        (
            normalise_series(data["frequency"])
            == frequency
        )
        &
        (
            data["display_name"].notna()
        )
    ].copy()

    filtered["_start_time_utc"] = pd.to_datetime(
        filtered["start_time"],
        utc=True,
        errors="coerce",
    )

    # Check whether every selected room has matching records.
    if valid_rooms:
        missing_room_data = []

        for room in valid_rooms:
            room_count = (
                normalise_series(filtered["display_name"])
                == normalise_text(room)
            ).sum()

            if room_count == 0:
                missing_room_data.append(room)

        for room in missing_room_data:
            errors.append(
                f"Room '{room}' has no matching records for "
                f"metric '{metric}', aggregation '{aggregation}' "
                f"and frequency '{frequency}'."
            )

        filtered = filtered[
            normalise_series(filtered["display_name"]).isin(
                {
                    normalise_text(room)
                    for room in valid_rooms
                }
            )
        ]

    if parsed_start is not None:
        filtered = filtered[
            filtered["_start_time_utc"] >= parsed_start
        ]

    if parsed_end is not None:
        filtered = filtered[
            filtered["_start_time_utc"] <= parsed_end
        ]

    matching_rows = len(filtered)

    # ---------------------------------------------------------
    # 13. Validate result availability
    # ---------------------------------------------------------

    if matching_rows == 0:
        errors.append(
            "No records match the selected metric, room, "
            "aggregation, frequency and date range."
        )

    elif matching_rows < minimum_records:
        warnings.append(
            f"Only {matching_rows} records matched. "
            f"This is below the minimum preferred value of "
            f"{minimum_records}, so the result may not be representative."
        )

    return validation_result(
        valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        matching_rows=matching_rows,
        normalised_request=normalised_request,
    )