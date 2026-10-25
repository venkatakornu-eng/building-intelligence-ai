from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "charts"
)


# ============================================================
# VISUALIZATION AGENT
# ============================================================

class VisualizationAgent:
    """
    Deterministic SmartViz visualization component.

    Supported charts:
    - bar
    - line
    - histogram
    - box
    - scatter

    The agent visualises only validated SQL results.
    It does not invent or modify analytical data.

    Relationship/scatter queries use:
        x_value
        y_value

    Distribution behaviour:
    - 1 room:
        normal histogram

    - 2 to 6 rooms:
        room-coloured overlay histogram

    - More than 6 rooms:
        combined histogram to avoid excessive visual clutter
    """

    SUPPORTED_CHARTS = {
        "bar",
        "line",
        "histogram",
        "box",
        "scatter",
    }

    # --------------------------------------------------------
    # Maximum number of rooms shown separately in histogram
    # --------------------------------------------------------

    MAX_COLOURED_HISTOGRAM_ROOMS = 6

    # --------------------------------------------------------
    # Explicit room colour palette
    # --------------------------------------------------------

    ROOM_COLOURS = [
        "#1f77b4",  # blue
        "#ff7f0e",  # orange
        "#2ca02c",  # green
        "#d62728",  # red
        "#9467bd",  # purple
        "#17becf",  # cyan
    ]

    # ========================================================
    # INITIALISATION
    # ========================================================

    def __init__(
        self,
        output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    ) -> None:

        self.output_dir = Path(
            output_dir
        )

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    # ========================================================
    # TEXT NORMALISATION
    # ========================================================

    @staticmethod
    def _normalise(
        value: str | None,
    ) -> str:

        if value is None:
            return ""

        return (
            str(value)
            .casefold()
            .strip()
        )

    # ========================================================
    # PUBLIC CHART SELECTION
    # ========================================================

    @classmethod
    def select_chart_type(
        cls,
        user_query: str,
        intent: str,
    ) -> str:
        """
        Select chart type deterministically.

        Explicit chart wording has priority.
        Intent is used as the fallback.
        """

        q = cls._normalise(
            user_query
        )

        # ----------------------------------------------------
        # BOX PLOT
        # ----------------------------------------------------

        if re.search(
            r"\b(?:"
            r"box\s*plot|"
            r"boxplot|"
            r"box\s+and\s+whisker|"
            r"box\s+whisker"
            r")\b",
            q,
        ):

            return "box"

        # ----------------------------------------------------
        # HISTOGRAM
        # ----------------------------------------------------

        if re.search(
            r"\b(?:"
            r"histogram|"
            r"frequency\s+distribution"
            r")\b",
            q,
        ):

            return "histogram"

        # ----------------------------------------------------
        # SCATTER / RELATIONSHIP
        # ----------------------------------------------------

        if re.search(
            r"\b(?:"
            r"scatter\s*plot|"
            r"scatter\s*graph|"
            r"scatter|"
            r"correlation|"
            r"relationship\s+between|"
            r"relationship\s+of|"
            r"association\s+between"
            r")\b",
            q,
        ):

            return "scatter"

        # ----------------------------------------------------
        # LINE
        # ----------------------------------------------------

        if re.search(
            r"\b(?:"
            r"line\s*chart|"
            r"line\s*graph"
            r")\b",
            q,
        ):

            return "line"

        # ----------------------------------------------------
        # BAR
        # ----------------------------------------------------

        if re.search(
            r"\b(?:"
            r"bar\s*chart|"
            r"bar\s*graph"
            r")\b",
            q,
        ):

            return "bar"

        # ----------------------------------------------------
        # GENERAL DISTRIBUTION
        # ----------------------------------------------------

        if re.search(
            r"\bdistribution\b",
            q,
        ):

            return "histogram"

        # ----------------------------------------------------
        # DEFAULT BY INTENT
        # ----------------------------------------------------

        if intent == "room_trend":

            return "line"

        if intent == "distribution":

            return "histogram"

        if intent == "relationship":

            return "scatter"

        if intent in {
            "compare_rooms",
            "rank_rooms",
            "threshold",
        }:

            return "bar"

        return "bar"

    # ========================================================
    # METRIC LABEL
    # ========================================================

    def _metric_label(
        self,
        metric_name: str | None,
    ) -> str:

        metric = self._normalise(
            metric_name
        )

        labels = {
            "co2":
                "CO2 Value",

            "temp":
                "Temperature",

            "humidity":
                "Humidity",

            "occupancy":
                "Occupancy",
        }

        return labels.get(
            metric,
            metric_name or "Value",
        )

    # ========================================================
    # ROOM COUNT
    # ========================================================

    @staticmethod
    def _room_count(
        dataframe: pd.DataFrame,
    ) -> int:
        """
        Return number of unique rooms represented
        in a SQL result.
        """

        if (
            "display_name"
            not in dataframe.columns
        ):

            return 0

        return int(
            dataframe[
                "display_name"
            ]
            .dropna()
            .astype(str)
            .nunique()
        )
    # ========================================================
    # BAR CHART
    # ========================================================

    def _create_bar(
        self,
        dataframe: pd.DataFrame,
        query_type: str,
        metric_name: str | None,
        title: str,
    ) -> go.Figure:

        # ----------------------------------------------------
        # SELECT VALUE COLUMN
        # ----------------------------------------------------

        if query_type == "rank_rooms_latest":

            y_column = "latest_value"

        else:

            y_column = "average_value"

        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        if "display_name" not in dataframe.columns:

            raise ValueError(
                "Bar chart requires display_name."
            )

        if y_column not in dataframe.columns:

            raise ValueError(
                f"Bar chart requires {y_column}."
            )

        # ----------------------------------------------------
        # CREATE BAR CHART
        #
        # One consistent colour is deliberately used.
        # Room identity is already represented by the x-axis,
        # therefore multiple colours are unnecessary.
        # ----------------------------------------------------

        figure = px.bar(
            dataframe,
            x="display_name",
            y=y_column,
            title=title,

            labels={
                "display_name":
                    "Room",

                y_column:
                    self._metric_label(
                        metric_name
                    ),
            },
        )

        # ----------------------------------------------------
        # AXIS / PRESENTATION
        # ----------------------------------------------------

        figure.update_layout(
            xaxis_title=
                "Room",

            yaxis_title=
                self._metric_label(
                    metric_name
                ),

            showlegend=
                False,
        )

        return figure


    # ========================================================
    # LINE CHART
    # ========================================================

    def _create_line(
        self,
        dataframe: pd.DataFrame,
        metric_name: str | None,
        title: str,
    ) -> go.Figure:

        required = {
            "start_time",
            "value",
        }

        missing = (
            required
            - set(
                dataframe.columns
            )
        )

        if missing:

            raise ValueError(
                "Line chart missing required "
                f"columns: {sorted(missing)}"
            )

        room_count = (
            self._room_count(
                dataframe
            )
        )

        # ----------------------------------------------------
        # MULTIPLE ROOMS
        # ----------------------------------------------------

        if (
            room_count > 1
            and
            "display_name"
            in dataframe.columns
        ):

            figure = px.line(
                dataframe,
                x="start_time",
                y="value",
                color="display_name",
                color_discrete_sequence=
                    self.ROOM_COLOURS,
                title=title,
                labels={
                    "start_time":
                        "Time",

                    "value":
                        self._metric_label(
                            metric_name
                        ),

                    "display_name":
                        "Room",
                },
            )

            figure.update_layout(
                legend_title=
                    "Room",
            )

        # ----------------------------------------------------
        # SINGLE ROOM
        # ----------------------------------------------------

        else:

            figure = px.line(
                dataframe,
                x="start_time",
                y="value",
                title=title,
                labels={
                    "start_time":
                        "Time",

                    "value":
                        self._metric_label(
                            metric_name
                        ),
                },
            )

        figure.update_layout(
            xaxis_title=
                "Time",

            yaxis_title=
                self._metric_label(
                    metric_name
                ),
        )

        return figure

    # ========================================================
    # HISTOGRAM
    # ========================================================

    def _create_histogram(
        self,
        dataframe: pd.DataFrame,
        metric_name: str | None,
        title: str,
    ) -> go.Figure:
        """
        Create an adaptive distribution histogram.

        Behaviour:

        1 room
        ------
        One normal histogram.

        2-6 rooms
        ---------
        Different colour for every room.

        More than 6 rooms
        -----------------
        Combined overall histogram because colouring every
        room would make the chart difficult to interpret.
        """

        # ----------------------------------------------------
        # VALUE COLUMN CHECK
        # ----------------------------------------------------

        if "value" not in dataframe.columns:

            raise ValueError(
                "Histogram requires a value column."
            )

        dataframe = (
            dataframe.copy()
        )

        # ----------------------------------------------------
        # ENSURE VALUES ARE NUMERIC
        # ----------------------------------------------------

        dataframe[
            "value"
        ] = pd.to_numeric(
            dataframe[
                "value"
            ],
            errors="coerce",
        )

        dataframe = (
            dataframe.dropna(
                subset=[
                    "value",
                ]
            )
        )

        if dataframe.empty:

            raise ValueError(
                "Histogram contains no valid numeric values."
            )

        room_count = (
            self._room_count(
                dataframe
            )
        )

        # ====================================================
        # SMALL MULTI-ROOM DISTRIBUTION
        # ====================================================

        if (
            room_count > 1
            and
            room_count
            <=
            self.MAX_COLOURED_HISTOGRAM_ROOMS
            and
            "display_name"
            in dataframe.columns
        ):

            figure = px.histogram(
                dataframe,

                x=
                    "value",

                color=
                    "display_name",

                nbins=
                    40,

                barmode=
                    "overlay",

                opacity=
                    0.55,

                title=
                    title,

                color_discrete_sequence=
                    self.ROOM_COLOURS,

                labels={
                    "value":
                        self._metric_label(
                            metric_name
                        ),

                    "display_name":
                        "Room",
                },
            )

            figure.update_layout(
                xaxis_title=
                    self._metric_label(
                        metric_name
                    ),

                yaxis_title=
                    "Frequency",

                legend_title=
                    "Room",

                bargap=
                    0.03,

                hovermode=
                    "x unified",
            )

            # ------------------------------------------------
            # Small border around histogram bars
            # ------------------------------------------------

            figure.update_traces(
                marker_line_width=
                    0.4,
            )

            return figure

        # ====================================================
        # SINGLE ROOM OR MANY ROOMS
        # ====================================================

        figure = px.histogram(
            dataframe,

            x=
                "value",

            nbins=
                40,

            title=
                title,

            labels={
                "value":
                    self._metric_label(
                        metric_name
                    ),
            },
        )

        figure.update_layout(
            xaxis_title=
                self._metric_label(
                    metric_name
                ),

            yaxis_title=
                "Frequency",

            bargap=
                0.03,
        )

        return figure

    # ========================================================
    # BOX PLOT
    # ========================================================

    def _create_box(
        self,
        dataframe: pd.DataFrame,
        metric_name: str | None,
        title: str,
    ) -> go.Figure:

        if "value" not in dataframe.columns:

            raise ValueError(
                "Box plot requires a value column."
            )

        room_count = (
            self._room_count(
                dataframe
            )
        )

        # ----------------------------------------------------
        # MULTIPLE ROOMS
        # ----------------------------------------------------

        if room_count > 1:

            figure = px.box(
                dataframe,
                x="display_name",
                y="value",
                color="display_name",
                color_discrete_sequence=
                    self.ROOM_COLOURS,
                points="outliers",
                title=title,
                labels={
                    "display_name":
                        "Room",

                    "value":
                        self._metric_label(
                            metric_name
                        ),
                },
            )

            figure.update_layout(
                xaxis_title=
                    "Room",

                yaxis_title=
                    self._metric_label(
                        metric_name
                    ),

                showlegend=
                    False,
            )

        # ----------------------------------------------------
        # SINGLE ROOM
        # ----------------------------------------------------

        else:

            figure = px.box(
                dataframe,
                y="value",
                points="outliers",
                title=title,
                labels={
                    "value":
                        self._metric_label(
                            metric_name
                        ),
                },
            )

            figure.update_layout(
                yaxis_title=
                    self._metric_label(
                        metric_name
                ),
            )

        return figure

    # ========================================================
    # SCATTER / RELATIONSHIP
    # ========================================================

    def _create_scatter(
        self,
        dataframe: pd.DataFrame,
        metric_x: str | None,
        metric_y: str | None,
        title: str,
    ) -> go.Figure:
        """
        Create a two-metric relationship scatter plot.

        QueryBuilderAgent creates:

            x_value
            y_value

        by matching requested metrics on the same:

            display_name
            start_time

        This visualization therefore uses explicit
        relationship fields rather than guessing columns.
        """

        # ----------------------------------------------------
        # METRIC DEFINITIONS
        # ----------------------------------------------------

        if metric_x is None:

            raise ValueError(
                "Scatter plot requires metric_x."
            )

        if metric_y is None:

            raise ValueError(
                "Scatter plot requires metric_y."
            )

        if metric_x == metric_y:

            raise ValueError(
                "Scatter plot requires two "
                "different metrics."
            )

        # ----------------------------------------------------
        # REQUIRED SQL RESULT COLUMNS
        # ----------------------------------------------------

        required = {
            "x_value",
            "y_value",
        }

        missing = (
            required
            - set(
                dataframe.columns
            )
        )

        if missing:

            raise ValueError(
                "Scatter plot missing required "
                f"relationship columns: "
                f"{sorted(missing)}"
            )

        # ----------------------------------------------------
        # CONVERT TO NUMERIC
        # ----------------------------------------------------

        dataframe = (
            dataframe.copy()
        )

        dataframe[
            "x_value"
        ] = pd.to_numeric(
            dataframe[
                "x_value"
            ],
            errors="coerce",
        )

        dataframe[
            "y_value"
        ] = pd.to_numeric(
            dataframe[
                "y_value"
            ],
            errors="coerce",
        )

        # ----------------------------------------------------
        # REMOVE INVALID PAIRS
        # ----------------------------------------------------

        dataframe = (
            dataframe.dropna(
                subset=[
                    "x_value",
                    "y_value",
                ]
            )
        )

        if len(
            dataframe
        ) < 2:

            raise ValueError(
                "Scatter plot requires at least "
                "two valid paired observations."
            )

        # ----------------------------------------------------
        # LABELS
        # ----------------------------------------------------

        x_label = (
            self._metric_label(
                metric_x
            )
        )

        y_label = (
            self._metric_label(
                metric_y
            )
        )

        # ----------------------------------------------------
        # HOVER CONFIGURATION
        # ----------------------------------------------------

        hover_name = None

        if (
            "display_name"
            in dataframe.columns
        ):

            hover_name = (
                "display_name"
            )

        hover_data: dict[
            str,
            Any,
        ] = {}

        if (
            "start_time"
            in dataframe.columns
        ):

            hover_data[
                "start_time"
            ] = True

        if (
            "x_record_count"
            in dataframe.columns
        ):

            hover_data[
                "x_record_count"
            ] = False

        if (
            "y_record_count"
            in dataframe.columns
        ):

            hover_data[
                "y_record_count"
            ] = False

        # ----------------------------------------------------
        # MULTI-ROOM SCATTER
        # ----------------------------------------------------

        room_count = (
            self._room_count(
                dataframe
            )
        )

        if (
            room_count > 1
            and
            "display_name"
            in dataframe.columns
            and
            room_count
            <=
            self.MAX_COLOURED_HISTOGRAM_ROOMS
        ):

            figure = px.scatter(
                dataframe,

                x=
                    "x_value",

                y=
                    "y_value",

                color=
                    "display_name",

                color_discrete_sequence=
                    self.ROOM_COLOURS,

                hover_name=
                    hover_name,

                hover_data=
                    hover_data,

                title=
                    title,

                labels={
                    "x_value":
                        x_label,

                    "y_value":
                        y_label,

                    "start_time":
                        "Time",

                    "display_name":
                        "Room",
                },

                opacity=
                    0.65,
            )

        # ----------------------------------------------------
        # SINGLE ROOM / MANY ROOMS
        # ----------------------------------------------------

        else:

            figure = px.scatter(
                dataframe,

                x=
                    "x_value",

                y=
                    "y_value",

                hover_name=
                    hover_name,

                hover_data=
                    hover_data,

                title=
                    title,

                labels={
                    "x_value":
                        x_label,

                    "y_value":
                        y_label,

                    "start_time":
                        "Time",

                    "display_name":
                        "Room",
                },

                opacity=
                    0.65,
            )

        figure.update_layout(
            xaxis_title=
                x_label,

            yaxis_title=
                y_label,
        )

        return figure

    # ========================================================
    # FIND SINGLE ROOM
    # ========================================================

    def _single_room_from_dataframe(
        self,
        dataframe: pd.DataFrame,
    ) -> str | None:

        if (
            "display_name"
            not in dataframe.columns
            or
            dataframe.empty
        ):

            return None

        unique_rooms = (
            dataframe[
                "display_name"
            ]
            .dropna()
            .astype(str)
            .unique()
        )

        if len(
            unique_rooms
        ) == 1:

            return str(
                unique_rooms[0]
            )

        return None

    # ========================================================
    # TITLE GENERATION
    # ========================================================

    def _build_title(
        self,
        chart_type: str,
        metric_name: str | None,
        metric_x: str | None,
        metric_y: str | None,
        query_type: str,
        dataframe: pd.DataFrame,
    ) -> str:

        metric_label = (
            self._metric_label(
                metric_name
            )
        )

        room = (
            self._single_room_from_dataframe(
                dataframe
            )
        )

        room_count = (
            self._room_count(
                dataframe
            )
        )

        # ----------------------------------------------------
        # RELATIONSHIP
        # ----------------------------------------------------

        if query_type == "relationship":

            x_label = (
                self._metric_label(
                    metric_x
                )
            )

            y_label = (
                self._metric_label(
                    metric_y
                )
            )

            if room:

                return (
                    f"{x_label} vs "
                    f"{y_label} — "
                    f"{room}"
                )

            return (
                f"Relationship Between "
                f"{x_label} and "
                f"{y_label}"
            )

        # ----------------------------------------------------
        # DISTRIBUTION
        # ----------------------------------------------------

        if query_type == "distribution":

            if chart_type == "histogram":

                if room:

                    return (
                        f"Distribution of "
                        f"{metric_label} — "
                        f"{room}"
                    )

                if (
                    room_count > 1
                    and
                    room_count
                    <=
                    self.MAX_COLOURED_HISTOGRAM_ROOMS
                ):

                    return (
                        f"Distribution of "
                        f"{metric_label} "
                        f"Across Selected Rooms"
                    )

                return (
                    f"Distribution of "
                    f"{metric_label} "
                    f"Across Rooms"
                )

            if chart_type == "box":

                if room:

                    return (
                        f"{metric_label} "
                        f"Distribution — "
                        f"{room}"
                    )

                return (
                    f"{metric_label} "
                    f"Distribution Across Rooms"
                )

        # ----------------------------------------------------
        # TREND
        # ----------------------------------------------------

        if query_type == "room_trend":

            if room:

                return (
                    f"{metric_label} Trend — "
                    f"{room}"
                )

            return (
                f"{metric_label} Trend"
            )

        # ----------------------------------------------------
        # COMPARISON
        # ----------------------------------------------------

        if query_type == "compare_rooms":

            return (
                f"{metric_label} "
                f"Comparison Across Rooms"
            )

        # ----------------------------------------------------
        # RANKING
        # ----------------------------------------------------

        if query_type in {
            "rank_rooms",
            "rank_rooms_latest",
        }:

            return (
                f"Room Ranking by "
                f"{metric_label}"
            )

        # ----------------------------------------------------
        # THRESHOLD
        # ----------------------------------------------------

        if query_type == "threshold":

            return (
                f"Rooms Meeting "
                f"{metric_label} Threshold"
            )

        # ----------------------------------------------------
        # FALLBACK SCATTER
        # ----------------------------------------------------

        if chart_type == "scatter":

            x_label = (
                self._metric_label(
                    metric_x
                )
            )

            y_label = (
                self._metric_label(
                    metric_y
                )
            )

            return (
                f"Relationship Between "
                f"{x_label} and "
                f"{y_label}"
            )

        return metric_label

    # ========================================================
    # SAVE FIGURE
    # ========================================================

    def _save_figure(
        self,
        figure: go.Figure,
        chart_type: str,
        metric_name: str | None,
        metric_x: str | None,
        metric_y: str | None,
        query_type: str,
    ) -> Path:

        # ----------------------------------------------------
        # RELATIONSHIP FILE NAME
        # ----------------------------------------------------

        if query_type == "relationship":

            metric_key = (
                f"{metric_x or 'x'}"
                f"_vs_"
                f"{metric_y or 'y'}"
            )

        else:

            metric_key = (
                metric_name
                or
                "metric"
            )

        safe_metric = re.sub(
            r"[^a-zA-Z0-9_-]",
            "_",
            metric_key,
        )

        safe_query_type = re.sub(
            r"[^a-zA-Z0-9_-]",
            "_",
            query_type,
        )

        filename = (
            f"{safe_query_type}_"
            f"{safe_metric}_"
            f"{chart_type}.html"
        )

        output_path = (
            self.output_dir
            / filename
        )

        figure.write_html(
            str(
                output_path
            ),
            include_plotlyjs="cdn",
        )

        return output_path

    # ========================================================
    # MAIN CREATE METHOD
    # ========================================================

    def create(
        self,
        user_query: str,
        intent: str,
        query_type: str,
        metric_name: str | None,
        rows: list[dict[str, Any]],
        result_valid: bool,
        result_has_data: bool,
        chart_type: str | None = None,

        # ----------------------------------------------------
        # RELATIONSHIP METRICS
        # ----------------------------------------------------

        metric_x: str | None = None,
        metric_y: str | None = None,

    ) -> dict[str, Any]:

        warnings: list[str] = []

        # ----------------------------------------------------
        # RESULT MUST PASS VALIDATION
        # ----------------------------------------------------

        if not result_valid:

            return {
                "success":
                    False,

                "chart_created":
                    False,

                "chart_type":
                    chart_type,

                "chart_path":
                    None,

                "chart_title":
                    None,

                "warnings": [
                    "Visualization skipped because "
                    "the SQL result did not pass validation."
                ],

                "error":
                    None,
            }

        # ----------------------------------------------------
        # MUST CONTAIN DATA
        # ----------------------------------------------------

        if not result_has_data:

            return {
                "success":
                    True,

                "chart_created":
                    False,

                "chart_type":
                    chart_type,

                "chart_path":
                    None,

                "chart_title":
                    None,

                "warnings": [
                    "Visualization skipped because "
                    "the query returned no data."
                ],

                "error":
                    None,
            }

        if not rows:

            return {
                "success":
                    True,

                "chart_created":
                    False,

                "chart_type":
                    chart_type,

                "chart_path":
                    None,

                "chart_title":
                    None,

                "warnings": [
                    "Visualization skipped because "
                    "no result rows were available."
                ],

                "error":
                    None,
            }

        # ----------------------------------------------------
        # CONVERT SQL RESULT TO DATAFRAME
        # ----------------------------------------------------

        dataframe = pd.DataFrame(
            rows
        )

        if (
            "start_time"
            in dataframe.columns
        ):

            dataframe[
                "start_time"
            ] = pd.to_datetime(
                dataframe[
                    "start_time"
                ],
                errors="coerce",
            )

        # ----------------------------------------------------
        # CHART SELECTION
        # ----------------------------------------------------

        if chart_type is None:

            chart_type = (
                self.select_chart_type(
                    user_query=
                        user_query,

                    intent=
                        intent,
                )
            )

        if (
            chart_type
            not in
            self.SUPPORTED_CHARTS
        ):

            return {
                "success":
                    False,

                "chart_created":
                    False,

                "chart_type":
                    chart_type,

                "chart_path":
                    None,

                "chart_title":
                    None,

                "warnings":
                    warnings,

                "error":
                    (
                        "Unsupported chart type: "
                        f"{chart_type}"
                    ),
            }

        # ----------------------------------------------------
        # RELATIONSHIP MUST USE SCATTER
        # ----------------------------------------------------

        if (
            query_type == "relationship"
            and
            chart_type != "scatter"
        ):

            return {
                "success":
                    False,

                "chart_created":
                    False,

                "chart_type":
                    chart_type,

                "chart_path":
                    None,

                "chart_title":
                    None,

                "warnings":
                    warnings,

                "error":
                    (
                        "Relationship results require "
                        "a scatter plot."
                    ),
            }

        # ----------------------------------------------------
        # DISTRIBUTION COMPATIBILITY
        # ----------------------------------------------------

        if (
            chart_type
            in {
                "histogram",
                "box",
            }
            and
            "value"
            not in dataframe.columns
        ):

            return {
                "success":
                    False,

                "chart_created":
                    False,

                "chart_type":
                    chart_type,

                "chart_path":
                    None,

                "chart_title":
                    None,

                "warnings":
                    warnings,

                "error":
                    (
                        f"{chart_type} requires "
                        "raw observation values."
                    ),
            }

        # ----------------------------------------------------
        # SCATTER COMPATIBILITY
        # ----------------------------------------------------

        if chart_type == "scatter":

            required_scatter_columns = {
                "x_value",
                "y_value",
            }

            missing_scatter_columns = (
                required_scatter_columns
                - set(
                    dataframe.columns
                )
            )

            if missing_scatter_columns:

                return {
                    "success":
                        False,

                    "chart_created":
                        False,

                    "chart_type":
                        chart_type,

                    "chart_path":
                        None,

                    "chart_title":
                        None,

                    "warnings":
                        warnings,

                    "error":
                        (
                            "Scatter plot requires "
                            "validated relationship fields: "
                            f"{sorted(missing_scatter_columns)}"
                        ),
                }

        # ----------------------------------------------------
        # BUILD TITLE
        # ----------------------------------------------------

        title = (
            self._build_title(
                chart_type=
                    chart_type,

                metric_name=
                    metric_name,

                metric_x=
                    metric_x,

                metric_y=
                    metric_y,

                query_type=
                    query_type,

                dataframe=
                    dataframe,
            )
        )

        # ----------------------------------------------------
        # CREATE CHART
        # ----------------------------------------------------

        try:

            if chart_type == "bar":

                figure = (
                    self._create_bar(
                        dataframe=
                            dataframe,

                        query_type=
                            query_type,

                        metric_name=
                            metric_name,

                        title=
                            title,
                    )
                )

            elif chart_type == "line":

                figure = (
                    self._create_line(
                        dataframe=
                            dataframe,

                        metric_name=
                            metric_name,

                        title=
                            title,
                    )
                )

            elif chart_type == "histogram":

                figure = (
                    self._create_histogram(
                        dataframe=
                            dataframe,

                        metric_name=
                            metric_name,

                        title=
                            title,
                    )
                )

            elif chart_type == "box":

                figure = (
                    self._create_box(
                        dataframe=
                            dataframe,

                        metric_name=
                            metric_name,

                        title=
                            title,
                    )
                )

            elif chart_type == "scatter":

                figure = (
                    self._create_scatter(
                        dataframe=
                            dataframe,

                        metric_x=
                            metric_x,

                        metric_y=
                            metric_y,

                        title=
                            title,
                    )
                )

            else:

                raise ValueError(
                    "Unsupported chart type: "
                    f"{chart_type}"
                )

            # ------------------------------------------------
            # SAVE CHART
            # ------------------------------------------------

            chart_path = (
                self._save_figure(
                    figure=
                        figure,

                    chart_type=
                        chart_type,

                    metric_name=
                        metric_name,

                    metric_x=
                        metric_x,

                    metric_y=
                        metric_y,

                    query_type=
                        query_type,
                )
            )

            return {
                "success":
                    True,

                "chart_created":
                    True,

                "chart_type":
                    chart_type,

                "chart_path":
                    str(
                        chart_path
                    ),

                "chart_title":
                    title,

                "warnings":
                    warnings,

                "error":
                    None,
            }

        except Exception as error:

            return {
                "success":
                    False,

                "chart_created":
                    False,

                "chart_type":
                    chart_type,

                "chart_path":
                    None,

                "chart_title":
                    title,

                "warnings":
                    warnings,

                "error":
                    (
                        f"{type(error).__name__}: "
                        f"{error}"
                    ),
            }