from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


METRIC_LABELS = {
    "co2": "CO₂",
    "temp": "Temperature",
    "humidity": "Humidity",
    "occupancy": "Occupancy",
}


def _metric_label(metric_name: str) -> str:
    """Convert dataset metric names into readable labels."""
    normalised_metric = str(metric_name).strip().casefold()

    return METRIC_LABELS.get(
        normalised_metric,
        str(metric_name).replace("_", " ").title()
    )


def _validate_router_response(
    response: dict[str, Any]
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Check that a router response contains usable results."""

    if not isinstance(response, dict):
        raise TypeError("Router response must be a dictionary.")

    if not response.get("success", False):
        errors = response.get(
            "errors",
            ["The analytical request was unsuccessful."]
        )

        raise ValueError("; ".join(str(error) for error in errors))

    request = response.get("request")
    result = response.get("result")

    if not isinstance(request, dict):
        raise ValueError(
            "The router response does not contain a valid request."
        )

    if not isinstance(result, pd.DataFrame):
        raise TypeError(
            "The router result must be a pandas DataFrame."
        )

    if result.empty:
        raise ValueError(
            "The router result is empty, so no chart can be created."
        )

    return request, result.copy()


def create_figure(
    response: dict[str, Any]
) -> go.Figure:
    """
    Create a Plotly chart from a successful router response.
    """

    request, result = _validate_router_response(response)

    intent = request["intent"]
    metric_name = request["metric_name"]
    metric_label = _metric_label(metric_name)

    if intent == "room_trend":
        figure = _create_trend_chart(
            result=result,
            request=request,
            metric_label=metric_label,
        )

    elif intent == "rank_rooms":
        figure = _create_ranking_chart(
            result=result,
            request=request,
            metric_label=metric_label,
        )

    elif intent == "compare_rooms":
        figure = _create_comparison_chart(
            result=result,
            metric_label=metric_label,
        )

    elif intent == "threshold":
        figure = _create_threshold_chart(
            result=result,
            request=request,
            metric_label=metric_label,
        )

    else:
        raise ValueError(
            f"Visualisation is not supported for intent: {intent}"
        )

    figure.update_layout(
        template="plotly_white",
        hovermode="closest",
        margin=dict(
            l=40,
            r=40,
            t=90,
            b=50,
        ),
    )

    return figure


def _create_trend_chart(
    result: pd.DataFrame,
    request: dict[str, Any],
    metric_label: str,
) -> go.Figure:
    """Create a line chart for one room over time."""

    required_columns = {
        "start_time",
        "display_name",
        "value",
    }

    missing_columns = required_columns.difference(result.columns)

    if missing_columns:
        raise ValueError(
            "Trend result is missing columns: "
            + ", ".join(sorted(missing_columns))
        )

    result["start_time"] = pd.to_datetime(
        result["start_time"],
        utc=True,
        errors="coerce",
    )

    result = (
        result.dropna(subset=["start_time", "value"])
        .sort_values("start_time")
    )

    room_names = request.get("room_names", [])

    room_name = (
        request.get("room")
        or (room_names[0] if room_names else None)
        or result["display_name"].iloc[0]
    )

    figure = px.line(
        result,
        x="start_time",
        y="value",
        title=f"{metric_label} Trend — {room_name}",
        labels={
            "start_time": "Date and Time",
            "value": f"{metric_label} Value",
        },
    )

    figure.update_traces(
        mode="lines",
        hovertemplate=(
            "<b>%{x|%d %b %Y, %H:%M}</b>"
            "<br>Value: %{y:.2f}"
            "<extra></extra>"
        ),
    )

    figure.update_xaxes(
        rangeslider_visible=True,
    )

    return figure


def _create_ranking_chart(
    result: pd.DataFrame,
    request: dict[str, Any],
    metric_label: str,
) -> go.Figure:
    """Create a room-ranking bar chart."""

    plot_data = (
        result.sort_values("average_value", ascending=True)
        .copy()
    )

    ranking_direction = (
        "Lowest"
        if request.get("ascending", False)
        else "Highest"
    )

    figure = px.bar(
        plot_data,
        x="average_value",
        y="display_name",
        orientation="h",
        title=f"Rooms with the {ranking_direction} Average {metric_label}",
        labels={
            "average_value": f"Average {metric_label} Value",
            "display_name": "Room",
        },
        hover_data={
            "minimum_value": ":.2f",
            "maximum_value": ":.2f",
            "record_count": True,
        },
    )

    _format_bar_chart(
        figure=figure,
        plot_data=plot_data,
    )

    return figure


def _create_comparison_chart(
    result: pd.DataFrame,
    metric_label: str,
) -> go.Figure:
    """Create a selected-room comparison chart."""

    plot_data = (
        result.sort_values("average_value", ascending=True)
        .copy()
    )

    figure = px.bar(
        plot_data,
        x="average_value",
        y="display_name",
        orientation="h",
        title=f"Average {metric_label} Comparison by Room",
        labels={
            "average_value": f"Average {metric_label} Value",
            "display_name": "Room",
        },
        hover_data={
            "minimum_value": ":.2f",
            "maximum_value": ":.2f",
            "record_count": True,
        },
    )

    _format_bar_chart(
        figure=figure,
        plot_data=plot_data,
    )

    return figure


def _create_threshold_chart(
    result: pd.DataFrame,
    request: dict[str, Any],
    metric_label: str,
) -> go.Figure:
    """Create a chart showing rooms above a selected threshold."""

    threshold = float(request["threshold"])

    plot_data = (
        result.sort_values("average_value", ascending=True)
        .copy()
    )

    figure = px.bar(
        plot_data,
        x="average_value",
        y="display_name",
        orientation="h",
        title=(
            f"Rooms with Average {metric_label} "
            f"Above {threshold:g}"
        ),
        labels={
            "average_value": f"Average {metric_label} Value",
            "display_name": "Room",
        },
        hover_data={
            "maximum_value": ":.2f",
            "record_count": True,
        },
    )

    figure.add_vline(
        x=threshold,
        line_dash="dash",
        annotation_text=f"Threshold: {threshold:g}",
        annotation_position="top",
    )

    _format_bar_chart(
        figure=figure,
        plot_data=plot_data,
    )

    return figure


def _format_bar_chart(
    figure: go.Figure,
    plot_data: pd.DataFrame,
) -> None:
    """Apply consistent formatting to horizontal bar charts."""

    figure.update_traces(
        texttemplate="%{x:.2f}",
        textposition="outside",
        cliponaxis=False,
    )

    figure.update_layout(
        height=max(
            450,
            35 * len(plot_data) + 180,
        ),
        yaxis={
            "categoryorder": "array",
            "categoryarray": plot_data[
                "display_name"
            ].tolist(),
        },
    )


def save_figure(
    figure: go.Figure,
    output_path: str | Path,
) -> Path:
    """Save a Plotly figure as an interactive HTML file."""

    path = Path(output_path)

    if path.suffix.casefold() != ".html":
        path = path.with_suffix(".html")

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.write_html(
        str(path),
        include_plotlyjs="cdn",
        full_html=True,
        auto_open=False,
    )

    return path