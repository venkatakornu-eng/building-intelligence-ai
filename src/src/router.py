from __future__ import annotations

from typing import Any

import pandas as pd

from src.analytics import (
    compare_rooms,
    rank_rooms_by_metric,
    room_metric_trend,
    rooms_above_threshold,
)
from src.validation import validate_request


def execute_request(
    request: dict[str, Any],
    data: pd.DataFrame,
) -> dict[str, Any]:
    """
    Validate and execute a SmartViz analytical request.

    Returns:
        {
            "success": bool,
            "errors": list,
            "warnings": list,
            "request": dict,
            "result": DataFrame or None
        }
    """

    validation = validate_request(
        request=request,
        data=data,
    )

    if not validation["valid"]:
        return {
            "success": False,
            "errors": validation["errors"],
            "warnings": validation["warnings"],
            "request": validation["normalised_request"],
            "result": None,
        }

    clean_request = validation["normalised_request"]
    intent = clean_request["intent"]

    try:

        # -------------------------------------------------
        # Room trend
        # -------------------------------------------------

        if intent == "room_trend":
            room_names = clean_request.get("room_names", [])

            result = room_metric_trend(
                data=data,
                room_name=room_names[0],
                metric_name=clean_request["metric_name"],
                start_date=clean_request.get("start_date"),
                end_date=clean_request.get("end_date"),
                aggregation=clean_request["aggregation"],
                frequency=clean_request["frequency"],
            )

        # -------------------------------------------------
        # Compare selected rooms
        # -------------------------------------------------

        elif intent == "compare_rooms":
            result = compare_rooms(
                data=data,
                room_names=clean_request["room_names"],
                metric_name=clean_request["metric_name"],
                aggregation=clean_request["aggregation"],
                frequency=clean_request["frequency"],
                start_date=clean_request.get("start_date"),
                end_date=clean_request.get("end_date"),
            )

        # -------------------------------------------------
        # Rank rooms
        # -------------------------------------------------

        elif intent == "rank_rooms":
            result = rank_rooms_by_metric(
                data=data,
                metric_name=clean_request["metric_name"],
                start_date=clean_request.get("start_date"),
                end_date=clean_request.get("end_date"),
                aggregation=clean_request["aggregation"],
                frequency=clean_request["frequency"],
                top_n=clean_request.get("top_n", 10),
                ascending=clean_request.get("ascending", False),
                minimum_records=clean_request.get(
                    "minimum_records",
                    100,
                ),
            )

        # -------------------------------------------------
        # Threshold analysis
        # -------------------------------------------------

        elif intent == "threshold":
            result = rooms_above_threshold(
                data=data,
                metric_name=clean_request["metric_name"],
                threshold=clean_request["threshold"],
                aggregation=clean_request["aggregation"],
                frequency=clean_request["frequency"],
                start_date=clean_request.get("start_date"),
                end_date=clean_request.get("end_date"),
                minimum_records=clean_request.get(
                    "minimum_records",
                    100,
                ),
            )

        else:
            return {
                "success": False,
                "errors": [f"Unsupported intent: {intent}"],
                "warnings": validation["warnings"],
                "request": clean_request,
                "result": None,
            }

    except Exception as error:
        return {
            "success": False,
            "errors": [
                f"An error occurred while executing the request: {error}"
            ],
            "warnings": validation["warnings"],
            "request": clean_request,
            "result": None,
        }

    if result is None or result.empty:
        return {
            "success": False,
            "errors": [
                "The analytical function returned no results."
            ],
            "warnings": validation["warnings"],
            "request": clean_request,
            "result": result,
        }

    return {
        "success": True,
        "errors": [],
        "warnings": validation["warnings"],
        "request": clean_request,
        "result": result,
    }