from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd


from src.data_loader import load_room_data
from src.router import execute_request
from src.visualization import create_figure, save_figure
from src.nl_parser import NaturalLanguageParser

# Always use the Dissertation project folder
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "room_level_metrics.csv"
)

DEFAULT_CHART_DIRECTORY = (
    PROJECT_ROOT
    / "outputs"
    / "charts"
)


class SmartVizPipeline:
    """
    End-to-end SmartViz analytical pipeline.

    Flow:
        natural-language or structured request
        → validation
        → analytics
        → visualisation
        → HTML chart
    """

    def __init__(
        self,
        data_path: str | Path = DEFAULT_DATA_PATH,
        chart_directory: str | Path = DEFAULT_CHART_DIRECTORY,
        model: str = "deepseek-r1:1.5b",
    ) -> None:

        self.data_path = self._resolve_path(data_path)

        self.chart_directory = self._resolve_path(
            chart_directory
        )

        if not self.data_path.exists():
            raise FileNotFoundError(
                f"Processed dataset was not found: "
                f"{self.data_path}"
            )

        self.chart_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        # Load data first
        self.data = load_room_data(
            str(self.data_path)
        )

        # Then initialise the natural-language parser
        self.parser = NaturalLanguageParser(
            data=self.data,
            model=model,
        )

    def run_natural_language(
        self,
        user_query: str,
        save_chart: bool = True,
        chart_filename: str | None = None,
    ) -> dict[str, Any]:
        """
        Process a natural-language question through the
        complete SmartViz pipeline.
        """

        try:
            structured_request = self.parser.parse(
                user_query
            )

        except Exception as error:
            return {
                "success": False,
                "errors": [
                    f"Natural-language parsing error: {error}"
                ],
                "warnings": [],
                "user_query": user_query,
                "request": None,
                "result": None,
                "figure": None,
                "chart_path": None,
            }

        response = self.run(
            request=structured_request,
            save_chart=save_chart,
            chart_filename=chart_filename,
        )

        response["user_query"] = user_query

        return response
    @staticmethod
    def _resolve_path(path: str | Path) -> Path:
        """
        Resolve relative paths from the Dissertation folder,
        not from the terminal's current directory.
        """

        resolved_path = Path(path)

        if not resolved_path.is_absolute():
            resolved_path = PROJECT_ROOT / resolved_path

        return resolved_path.resolve()

    @staticmethod
    def _safe_filename(value: str) -> str:
        """Convert text into a safe file name."""

        safe_value = re.sub(
            r"[^A-Za-z0-9_-]+",
            "_",
            str(value).strip(),
        )

        return safe_value.strip("_").lower()

    def _default_chart_name(
        self,
        request: dict[str, Any],
    ) -> str:
        """Create a descriptive chart filename."""

        intent = request.get(
            "intent",
            "analysis",
        )

        metric_name = request.get(
            "metric_name",
            "metric",
        )

        room_name = request.get("room")

        name_parts = [
            intent,
            metric_name,
        ]

        if room_name:
            name_parts.append(room_name)

        filename = "_".join(
            self._safe_filename(part)
            for part in name_parts
        )

        return f"{filename}.html"

    def run(
        self,
        request: dict[str, Any],
        save_chart: bool = True,
        chart_filename: str | None = None,
    ) -> dict[str, Any]:
        """
        Execute one complete analytical request.
        """

        router_response = execute_request(
            request=request,
            data=self.data,
        )

        pipeline_response: dict[str, Any] = {
            "success": router_response["success"],
            "errors": router_response["errors"],
            "warnings": router_response["warnings"],
            "request": router_response["request"],
            "result": router_response["result"],
            "figure": None,
            "chart_path": None,
        }

        if not router_response["success"]:
            return pipeline_response

        try:
            figure = create_figure(
                router_response
            )

            pipeline_response["figure"] = figure

            if save_chart:
                filename = (
                    chart_filename
                    or self._default_chart_name(
                        router_response["request"]
                    )
                )

                chart_path = save_figure(
                    figure=figure,
                    output_path=(
                        self.chart_directory
                        / filename
                    ),
                )

                pipeline_response["chart_path"] = (
                    chart_path.resolve()
                )

        except Exception as error:
            pipeline_response["success"] = False

            pipeline_response["errors"].append(
                f"Visualisation error: {error}"
            )

        return pipeline_response