from __future__ import annotations

import pandas as pd


def _normalise(series: pd.Series) -> pd.Series:
    """Standardise text values for reliable matching."""
    return (
        series.astype("string")
        .str.strip()
        .str.casefold()
    )


def _filter_dates(
    data: pd.DataFrame,
    start_date=None,
    end_date=None
) -> pd.DataFrame:
    """Filter data using optional UTC-compatible dates."""
    df = data.copy()

    df["_time_utc"] = pd.to_datetime(
        df["start_time"],
        utc=True,
        errors="coerce"
    )

    if start_date is not None:
        start = pd.to_datetime(
            start_date,
            utc=True,
            errors="raise"
        )
        df = df[df["_time_utc"] >= start]

    if end_date is not None:
        end = pd.to_datetime(
            end_date,
            utc=True,
            errors="raise"
        )
        df = df[df["_time_utc"] <= end]

    return df.drop(columns="_time_utc")


# ---------------------------------------------------------
# 1. Rank rooms by a selected metric
# ---------------------------------------------------------

def rank_rooms_by_metric(
    data: pd.DataFrame,
    metric_name: str,
    start_date=None,
    end_date=None,
    aggregation: str = "mean",
    frequency: str = "hourly",
    top_n: int = 10,
    ascending: bool = False,
    minimum_records: int = 100
) -> pd.DataFrame:

    df = data[
        (_normalise(data["metric_name"]) == metric_name.strip().casefold())
        &
        (_normalise(data["aggregation"]) == aggregation.strip().casefold())
        &
        (_normalise(data["frequency"]) == frequency.strip().casefold())
        &
        (data["display_name"].notna())
    ].copy()

    df = _filter_dates(
        df,
        start_date=start_date,
        end_date=end_date
    )

    result = (
        df.groupby("display_name", as_index=False)
        .agg(
            average_value=("value", "mean"),
            minimum_value=("value", "min"),
            maximum_value=("value", "max"),
            record_count=("value", "count")
        )
    )

    # Remove rooms with insufficient records
    result = result[
        result["record_count"] >= minimum_records
    ]

    result = (
        result
        .sort_values(
            "average_value",
            ascending=ascending
        )
        .head(top_n)
        .reset_index(drop=True)
    )

    return result


# ---------------------------------------------------------
# 2. Compare selected rooms
# ---------------------------------------------------------
def compare_rooms(
    data: pd.DataFrame,
    room_names: list[str],
    metric_name: str,
    aggregation: str = "mean",
    frequency: str = "hourly",
    start_date=None,
    end_date=None,
    minimum_records: int = 1
) -> pd.DataFrame:

    requested_rooms = {
        room.strip().casefold()
        for room in room_names
    }

    df = data[
        _normalise(data["display_name"]).isin(requested_rooms)
        &
        (
            _normalise(data["metric_name"])
            == metric_name.strip().casefold()
        )
        &
        (
            _normalise(data["aggregation"])
            == aggregation.strip().casefold()
        )
        &
        (
            _normalise(data["frequency"])
            == frequency.strip().casefold()
        )
    ].copy()

    df = _filter_dates(
        df,
        start_date=start_date,
        end_date=end_date
    )

    result = (
        df.groupby("display_name", as_index=False)
        .agg(
            average_value=("value", "mean"),
            minimum_value=("value", "min"),
            maximum_value=("value", "max"),
            record_count=("value", "count")
        )
    )

    result = result[
        result["record_count"] >= minimum_records
    ]

    return (
        result
        .sort_values("average_value", ascending=False)
        .reset_index(drop=True)
    )


# ---------------------------------------------------------
# 3. Room-level metric trend
# ---------------------------------------------------------

def room_metric_trend(
    data: pd.DataFrame,
    room_name: str,
    metric_name: str,
    start_date=None,
    end_date=None,
    aggregation: str = "mean",
    frequency: str = "hourly"
) -> pd.DataFrame:

    df = data[
        (_normalise(data["display_name"]) == room_name.strip().casefold())
        &
        (_normalise(data["metric_name"]) == metric_name.strip().casefold())
        &
        (_normalise(data["aggregation"]) == aggregation.strip().casefold())
        &
        (_normalise(data["frequency"]) == frequency.strip().casefold())
    ].copy()

    df = _filter_dates(
        df,
        start_date=start_date,
        end_date=end_date
    )

    return (
        df.sort_values("start_time")
        [
            [
                "start_time",
                "display_name",
                "metric_name",
                "aggregation",
                "frequency",
                "value"
            ]
        ]
    )


# ---------------------------------------------------------
# 4. Metrics available for one room
# ---------------------------------------------------------

def available_metrics_for_room(
    data: pd.DataFrame,
    room_name: str
) -> pd.DataFrame:

    df = data[
        _normalise(data["display_name"])
        == room_name.strip().casefold()
    ].copy()

    return (
        df.groupby(
            ["metric_name", "aggregation", "frequency"],
            as_index=False
        )
        .agg(record_count=("value", "count"))
        .sort_values(
            ["metric_name", "aggregation", "frequency"]
        )
    )


# ---------------------------------------------------------
# 5. Rooms available for one metric
# ---------------------------------------------------------

def available_rooms_for_metric(
    data: pd.DataFrame,
    metric_name: str,
    aggregation: str | None = None,
    frequency: str | None = None
) -> pd.DataFrame:

    df = data[
        _normalise(data["metric_name"])
        == metric_name.strip().casefold()
    ].copy()

    if aggregation is not None:
        df = df[
            _normalise(df["aggregation"])
            == aggregation.strip().casefold()
        ]

    if frequency is not None:
        df = df[
            _normalise(df["frequency"])
            == frequency.strip().casefold()
        ]

    return (
        df.groupby("display_name", as_index=False)
        .agg(record_count=("value", "count"))
        .sort_values("display_name")
    )


# ---------------------------------------------------------
# 6. Find rooms above a threshold
# ---------------------------------------------------------

def rooms_above_threshold(
    data: pd.DataFrame,
    metric_name: str,
    threshold: float,
    aggregation: str = "mean",
    frequency: str = "hourly",
    start_date=None,
    end_date=None,
    minimum_records: int = 100
) -> pd.DataFrame:

    df = data[
        (_normalise(data["metric_name"]) == metric_name.strip().casefold())
        &
        (_normalise(data["aggregation"]) == aggregation.strip().casefold())
        &
        (_normalise(data["frequency"]) == frequency.strip().casefold())
        &
        (data["display_name"].notna())
    ].copy()

    df = _filter_dates(
        df,
        start_date=start_date,
        end_date=end_date
    )

    result = (
        df.groupby("display_name", as_index=False)
        .agg(
            average_value=("value", "mean"),
            maximum_value=("value", "max"),
            record_count=("value", "count")
        )
    )

    # Remove rooms with insufficient records
    result = result[
        result["record_count"] >= minimum_records
    ]

    return (
        result[result["average_value"] > threshold]
        .sort_values("average_value", ascending=False)
        .reset_index(drop=True)
    )

 