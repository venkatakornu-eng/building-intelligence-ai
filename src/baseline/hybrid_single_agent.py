from __future__ import annotations

import json
import math
import re
from pathlib import Path
from statistics import mean, median
from typing import Any

import pandas as pd

from src.agents.base_agent import BaseOllamaAgent
from src.agents.query_builder_agent import QueryBuilderAgent
from src.agents.request_validation_agent import RequestValidationAgent
from src.agents.result_validation_agent import ResultValidationAgent
from src.agents.sql_execution_agent import SQLExecutionAgent
from src.agents.visualization_agent import VisualizationAgent
from src.sql_database import SQLDatabase


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "room_level_metrics.csv"
)


# ============================================================
# SUPPORTED VALUES
# ============================================================

SUPPORTED_INTENTS = {
    "room_trend",
    "compare_rooms",
    "rank_rooms",
    "threshold",
    "distribution",
    "relationship",
}

SUPPORTED_METRICS = {
    "co2",
    "temp",
    "humidity",
    "occupancy",
}

SUPPORTED_CHARTS = {
    "line",
    "bar",
    "histogram",
    "box",
    "scatter",
}


# ============================================================
# STRUCTURED PARSER SCHEMA
# ============================================================

REQUEST_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "intent": {
            "type": "string",
            "enum": [
                "room_trend",
                "compare_rooms",
                "rank_rooms",
                "threshold",
                "distribution",
                "relationship",
            ],
        },
        "metric_name": {
            "type": [
                "string",
                "null",
            ],
        },
        "metric_x": {
            "type": [
                "string",
                "null",
            ],
        },
        "metric_y": {
            "type": [
                "string",
                "null",
            ],
        },
        "room": {
            "type": [
                "string",
                "null",
            ],
        },
        "room_names": {
            "type": "array",
            "items": {
                "type": "string",
            },
        },
        "aggregation": {
            "type": "string",
            "enum": [
                "mean",
                "min",
                "max",
                "sum",
            ],
        },
        "frequency": {
            "type": "string",
            "enum": [
                "hourly",
                "daily",
            ],
        },
        "start_date": {
            "type": [
                "string",
                "null",
            ],
        },
        "end_date": {
            "type": [
                "string",
                "null",
            ],
        },
        "top_n": {
            "type": "integer",
            "minimum": 1,
        },
        "ascending": {
            "type": "boolean",
        },
        "threshold": {
            "type": [
                "number",
                "null",
            ],
        },
        "time_scope": {
            "type": "string",
            "enum": [
                "all",
                "latest",
            ],
        },
        "continuous": {
            "type": "boolean",
        },
        "chart_type": {
            "type": "string",
            "enum": [
                "line",
                "bar",
                "histogram",
                "box",
                "scatter",
            ],
        },
    },
    "required": [
        "intent",
        "metric_name",
        "metric_x",
        "metric_y",
        "room",
        "room_names",
        "aggregation",
        "frequency",
        "start_date",
        "end_date",
        "top_n",
        "ascending",
        "threshold",
        "time_scope",
        "continuous",
        "chart_type",
    ],
    "additionalProperties": False,
}


# ============================================================
# INSIGHT OUTPUT SCHEMA
# ============================================================

INSIGHT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "summary": {
            "type": "string",
        },
        "key_points": {
            "type": "array",
            "items": {
                "type": "string",
            },
        },
        "caution": {
            "type": "string",
        },
    },
    "required": [
        "summary",
        "key_points",
        "caution",
    ],
    "additionalProperties": False,
}


# ============================================================
# HYBRID SINGLE-AGENT BASELINE
# ============================================================

class HybridSingleAgent(BaseOllamaAgent):
    """
    Monolithic hybrid single-agent baseline.

    The same local LLM performs:
        1. natural-language interpretation;
        2. final textual insight generation.

    Deterministic rules are used to correct obvious parser
    errors and prevent invented mapping values.

    SQL execution, result validation and visualization use
    the same deterministic infrastructure as the multi-agent
    system so that the comparison is controlled.

    Unlike the LangGraph multi-agent system, this baseline
    does NOT contain:
        - Intent Agent + Intent Guard separation;
        - Mapping Agent + Mapping Guard separation;
        - Insight Guard;
        - agent-to-agent state transitions.
    """

    # ========================================================
    # INITIALISATION
    # ========================================================

    def __init__(
        self,
        data_path: str | Path = DEFAULT_DATA_PATH,
        model: str = "deepseek-r1:1.5b",
        timeout: int = 120,
    ) -> None:

        super().__init__(
            model=model,
            timeout=timeout,
        )

        self.data_path = Path(
            data_path
        )

        if not self.data_path.exists():

            raise FileNotFoundError(
                f"Dataset not found: "
                f"{self.data_path}"
            )

        room_data = pd.read_csv(
            self.data_path,
            usecols=[
                "display_name",
            ],
        )

        self.available_rooms = (
            room_data[
                "display_name"
            ]
            .dropna()
            .astype(str)
            .str.strip()
            .drop_duplicates()
            .tolist()
        )

        self.available_rooms.sort(
            key=str.casefold
        )

        self.room_lookup = {
            room.casefold():
                room
            for room in self.available_rooms
        }

        # ----------------------------------------------------
        # Deterministic infrastructure
        # ----------------------------------------------------

        self.request_validator = (
            RequestValidationAgent(
                available_rooms=
                    self.available_rooms
            )
        )

        self.query_builder = (
            QueryBuilderAgent(
                table_name=
                    "dbo.room_level_metrics"
            )
        )

        self.database = (
            SQLDatabase(
                server=
                    r"localhost",

                database=
                    "SmartVizDB",

                driver=
                    "ODBC Driver 18 for SQL Server",
            )
        )

        self.sql_executor = (
            SQLExecutionAgent(
                database=
                    self.database,

                max_result_rows=
                    50_000,
            )
        )

        self.result_validator = (
            ResultValidationAgent()
        )

        self.visualizer = (
            VisualizationAgent()
        )

    # ========================================================
    # PARSER SYSTEM PROMPT
    # ========================================================

    def _parser_system_prompt(
        self,
    ) -> str:

        rooms = "\n".join(
            f"- {room}"
            for room in self.available_rooms
        )

        return f"""
You are the single natural-language interpretation component
of a SmartViz building analytics system.

Convert the user's question into exactly one JSON request.

SUPPORTED INTENTS

room_trend
- One room over time.
- Example: Show temperature trend for Seminar Room 3.

compare_rooms
- Compare two or more explicitly named rooms.

rank_rooms
- Top, bottom, highest, lowest, hottest, busiest,
  least occupied or similar room ranking.

threshold
- Numeric filtering such as CO2 above 800.

distribution
- Histogram, distribution or box-plot analysis.

relationship
- Relationship, correlation, association or scatter analysis
  between TWO metrics.

METRICS

CO2 / carbon dioxide -> co2
temperature / temp -> temp
humidity -> humidity
occupancy -> occupancy

RELATIONSHIP RULES

For relationship queries:
- metric_name must be null.
- metric_x is the first metric mentioned.
- metric_y is the second metric mentioned.
- chart_type must be scatter.
- Do not invent a second metric.

DISTRIBUTION RULES

For distribution queries:
- chart_type is histogram unless the user explicitly asks
  for a box plot.
- Preserve top_n when the user asks for top/bottom N rooms.

LATEST RULES

Questions containing:
- right now
- latest
- currently
- current

should use:
time_scope = latest

If a latest/current question asks for a singular room such as:
- Which room has the highest CO2 right now?
- What room currently has the lowest occupancy?

then:
top_n = 1

Latest means the latest AVAILABLE historical observation.
It does not imply live data.

DEFAULTS

aggregation = mean
frequency = hourly
top_n = 10
ascending = false
threshold = null
time_scope = all
continuous = false

Use null for dates not explicitly supplied.

Do not invent:
- dates
- room names
- thresholds
- metrics

Preserve room names exactly.

Return JSON only.

AVAILABLE ROOMS

{rooms}
""".strip()

    # ========================================================
    # NORMALISE
    # ========================================================

    @staticmethod
    def _normalise(
        value: Any,
    ) -> str:

        if value is None:

            return ""

        return (
            str(value)
            .casefold()
            .strip()
        )

    # ========================================================
    # MATCH ROOM
    # ========================================================

    def _match_room(
        self,
        room_name: Any,
    ) -> str | None:

        if room_name is None:

            return None

        cleaned = str(
            room_name
        ).strip()

        if not cleaned:

            return None

        return self.room_lookup.get(
            cleaned.casefold()
        )

    # ========================================================
    # EXTRACT ROOMS FROM QUERY
    # ========================================================

    def _extract_rooms_from_query(
        self,
        user_query: str,
    ) -> list[str]:

        query = user_query.casefold()

        matches: list[
            tuple[int, int, str]
        ] = []

        for room in self.available_rooms:

            pattern = (
                rf"(?<!\w)"
                rf"{re.escape(room.casefold())}"
                rf"(?!\w)"
            )

            match = re.search(
                pattern,
                query,
            )

            if match:

                matches.append(
                    (
                        match.start(),
                        -len(room),
                        room,
                    )
                )

        matches.sort()

        result: list[str] = []

        for _, _, room in matches:

            if room not in result:

                result.append(
                    room
                )

        # ----------------------------------------------------
        # Useful shorthand
        # ----------------------------------------------------

        if (
            not result
            and
            re.search(
                r"\bthe hive\b|\bhive\b",
                query,
            )
        ):

            for room in self.available_rooms:

                if (
                    room.casefold()
                    ==
                    "the hive"
                ):

                    result.append(
                        room
                    )

                    break

        return result

    # ========================================================
    # EXTRACT METRICS IN QUERY ORDER
    # ========================================================

    def _extract_metrics_in_order(
        self,
        user_query: str,
    ) -> list[str]:

        query = (
            user_query
            .casefold()
            .replace(
                "co₂",
                "co2",
            )
        )

        patterns = {
            "co2": (
                r"\bco2\b|"
                r"\bcarbon\s+dioxide\b"
            ),

            "temp": (
                r"\btemperature\b|"
                r"\btemp\b|"
                r"\bhottest\b|"
                r"\bwarmest\b|"
                r"\bcoldest\b"
            ),

            "humidity": (
                r"\bhumidity\b|"
                r"\bhumid\b"
            ),

            "occupancy": (
                r"\boccupancy\b|"
                r"\boccupied\b|"
                r"\bbusy\b|"
                r"\bbusiest\b|"
                r"\butilised\b|"
                r"\butilized\b|"
                r"\butilisation\b|"
                r"\butilization\b"
            ),
        }

        matches: list[
            tuple[int, str]
        ] = []

        for metric, pattern in patterns.items():

            match = re.search(
                pattern,
                query,
            )

            if match:

                matches.append(
                    (
                        match.start(),
                        metric,
                    )
                )

        matches.sort(
            key=lambda item: item[0]
        )

        return [
            metric
            for _, metric in matches
        ]

    # ========================================================
    # EXTRACT DATES
    # ========================================================

    @staticmethod
    def _extract_dates(
        user_query: str,
    ) -> tuple[
        str | None,
        str | None,
    ]:

        dates = re.findall(
            r"\b20\d{2}-\d{2}-\d{2}\b",
            user_query,
        )

        if len(dates) >= 2:

            return (
                dates[0],
                dates[1],
            )

        if len(dates) == 1:

            return (
                dates[0],
                None,
            )

        return (
            None,
            None,
        )

    # ========================================================
    # EXTRACT TOP N
    # ========================================================

    @staticmethod
    def _extract_top_n(
        user_query: str,
    ) -> int:

        query = user_query.casefold()

        number_words = {
            "one": 1,
            "two": 2,
            "three": 3,
            "four": 4,
            "five": 5,
            "six": 6,
            "seven": 7,
            "eight": 8,
            "nine": 9,
            "ten": 10,
            "eleven": 11,
            "twelve": 12,
            "fifteen": 15,
            "twenty": 20,
        }

        # ----------------------------------------------------
        # top 5 / bottom 3
        # ----------------------------------------------------

        match = re.search(
            r"\b(?:top|bottom)\s+(\d+)\b",
            query,
        )

        if match:

            return int(
                match.group(1)
            )

        # ----------------------------------------------------
        # 5 rooms
        # ----------------------------------------------------

        match = re.search(
            r"\b(\d+)\s+(?:rooms?|spaces?)\b",
            query,
        )

        if match:

            return int(
                match.group(1)
            )

        # ----------------------------------------------------
        # five rooms / three hottest rooms
        # ----------------------------------------------------

        for word, number in number_words.items():

            if re.search(
                rf"\b{word}\b",
                query,
            ):

                return number

        return 10

    # ========================================================
    # EXTRACT THRESHOLD
    # ========================================================

    @staticmethod
    def _extract_threshold(
        user_query: str,
    ) -> float | None:

        query = user_query.casefold()

        pattern = (
            r"\b(?:above|over|greater\s+than|"
            r"more\s+than|higher\s+than|"
            r"exceeding|below|under|"
            r"less\s+than|lower\s+than)"
            r"\s*(-?\d+(?:\.\d+)?)"
        )

        match = re.search(
            pattern,
            query,
        )

        if not match:

            return None

        return float(
            match.group(1)
        )

    # ========================================================
    # LATEST QUERY DETECTION
    # ========================================================

    @staticmethod
    def _is_latest_query(
        user_query: str,
    ) -> bool:

        query = user_query.casefold()

        return bool(
            re.search(
                r"\bright\s+now\b|"
                r"\blatest\b|"
                r"\bcurrently\b|"
                r"\bcurrent\b|"
                r"\bmost\s+recent\b",
                query,
            )
        )

    # ========================================================
    # SINGULAR RANKING DETECTION
    # ========================================================

    @staticmethod
    def _is_singular_rank_query(
        user_query: str,
    ) -> bool:
        """
        Detect ranking questions that explicitly request
        one room rather than a top-N list.

        Examples:
        - Which room has the highest CO2?
        - What room has the lowest occupancy?
        """

        query = user_query.casefold()

        return bool(
            re.search(
                r"\bwhich\s+room\b|"
                r"\bwhat\s+room\b|"
                r"\bwhich\s+space\b|"
                r"\bwhat\s+space\b|"
                r"\broom\s+has\s+the\s+highest\b|"
                r"\broom\s+has\s+the\s+lowest\b|"
                r"\bspace\s+has\s+the\s+highest\b|"
                r"\bspace\s+has\s+the\s+lowest\b",
                query,
            )
        )

    # ========================================================
    # DETERMINE INTENT
    # ========================================================

    def _determine_intent(
        self,
        user_query: str,
        extracted_rooms: list[str],
        model_intent: str,
    ) -> str:

        query = user_query.casefold()

        # ----------------------------------------------------
        # RELATIONSHIP FIRST
        # ----------------------------------------------------

        if re.search(
            r"\brelationship\b|"
            r"\bcorrelation\b|"
            r"\bassociation\b|"
            r"\bscatter\b|"
            r"\bagainst\b",
            query,
        ):

            return "relationship"

        # ----------------------------------------------------
        # DISTRIBUTION BEFORE RANKING
        # ----------------------------------------------------

        if re.search(
            r"\bdistribution\b|"
            r"\bhistogram\b|"
            r"\bbox\s*plot\b|"
            r"\bboxplot\b",
            query,
        ):

            return "distribution"

        # ----------------------------------------------------
        # THRESHOLD
        # ----------------------------------------------------

        if (
            re.search(
                r"\babove\b|"
                r"\bover\b|"
                r"\bgreater\s+than\b|"
                r"\bmore\s+than\b|"
                r"\bhigher\s+than\b|"
                r"\bexceeding\b|"
                r"\bbelow\b|"
                r"\bunder\b|"
                r"\bless\s+than\b|"
                r"\blower\s+than\b",
                query,
            )
            and
            self._extract_threshold(
                user_query
            )
            is not None
        ):

            return "threshold"

        # ----------------------------------------------------
        # EXPLICIT COMPARISON
        # ----------------------------------------------------

        if (
            re.search(
                r"\bcompare\b|"
                r"\bcomparison\b|"
                r"\bversus\b|"
                r"\bvs\b",
                query,
            )
            and
            len(
                extracted_rooms
            ) >= 2
        ):

            return "compare_rooms"

        # ----------------------------------------------------
        # RANKING
        # ----------------------------------------------------

        if re.search(
            r"\btop\b|"
            r"\bbottom\b|"
            r"\bhighest\b|"
            r"\blowest\b|"
            r"\bhottest\b|"
            r"\bwarmest\b|"
            r"\bcoldest\b|"
            r"\bbusiest\b|"
            r"\bleast\b|"
            r"\bmost\b|"
            r"\bbest\b|"
            r"\bworst\b",
            query,
        ):

            return "rank_rooms"

        # ----------------------------------------------------
        # TREND
        # ----------------------------------------------------

        if (
            re.search(
                r"\btrend\b|"
                r"\bover\s+time\b|"
                r"\btime\s+series\b|"
                r"\bthroughout\b",
                query,
            )
            and
            len(
                extracted_rooms
            ) == 1
        ):

            return "room_trend"

        # ----------------------------------------------------
        # SINGLE ROOM DEFAULT
        # ----------------------------------------------------

        if len(
            extracted_rooms
        ) == 1:

            return "room_trend"

        # ----------------------------------------------------
        # MODEL FALLBACK
        # ----------------------------------------------------

        if model_intent in SUPPORTED_INTENTS:

            return model_intent

        return "rank_rooms"

    # ========================================================
    # CHART SELECTION
    # ========================================================

    @staticmethod
    def _select_chart(
        user_query: str,
        intent: str,
    ) -> str:

        query = user_query.casefold()

        if intent == "relationship":

            return "scatter"

        if intent == "distribution":

            if re.search(
                r"\bbox\s*plot\b|"
                r"\bboxplot\b",
                query,
            ):

                return "box"

            return "histogram"

        if intent == "room_trend":

            return "line"

        return "bar"

    # ========================================================
    # CLEAN RAW LLM REQUEST
    # ========================================================

    def _clean_raw_request(
        self,
        request: dict[str, Any],
    ) -> dict[str, Any]:

        intent = self._normalise(
            request.get(
                "intent"
            )
        )

        if intent not in SUPPORTED_INTENTS:

            intent = ""

        metric_name = self._normalise(
            request.get(
                "metric_name"
            )
        )

        if metric_name not in SUPPORTED_METRICS:

            metric_name = None

        metric_x = self._normalise(
            request.get(
                "metric_x"
            )
        )

        if metric_x not in SUPPORTED_METRICS:

            metric_x = None

        metric_y = self._normalise(
            request.get(
                "metric_y"
            )
        )

        if metric_y not in SUPPORTED_METRICS:

            metric_y = None

        room = self._match_room(
            request.get(
                "room"
            )
        )

        raw_rooms = request.get(
            "room_names",
            [],
        )

        if not isinstance(
            raw_rooms,
            list,
        ):

            raw_rooms = []

        rooms: list[str] = []

        for raw_room in raw_rooms:

            matched = self._match_room(
                raw_room
            )

            if (
                matched is not None
                and
                matched not in rooms
            ):

                rooms.append(
                    matched
                )

        aggregation = self._normalise(
            request.get(
                "aggregation",
                "mean",
            )
        )

        if aggregation not in {
            "mean",
            "min",
            "max",
            "sum",
        }:

            aggregation = "mean"

        frequency = self._normalise(
            request.get(
                "frequency",
                "hourly",
            )
        )

        if frequency not in {
            "hourly",
            "daily",
        }:

            frequency = "hourly"

        try:

            top_n = int(
                request.get(
                    "top_n",
                    10,
                )
            )

        except (
            TypeError,
            ValueError,
        ):

            top_n = 10

        if top_n < 1:

            top_n = 10

        chart_type = self._normalise(
            request.get(
                "chart_type"
            )
        )

        if chart_type not in SUPPORTED_CHARTS:

            chart_type = "bar"

        return {
            "intent":
                intent,

            "metric_name":
                metric_name,

            "metric_x":
                metric_x,

            "metric_y":
                metric_y,

            "room":
                room,

            "room_names":
                rooms,

            "aggregation":
                aggregation,

            "frequency":
                frequency,

            "start_date":
                request.get(
                    "start_date"
                ),

            "end_date":
                request.get(
                    "end_date"
                ),

            "top_n":
                top_n,

            "ascending":
                bool(
                    request.get(
                        "ascending",
                        False,
                    )
                ),

            "threshold":
                request.get(
                    "threshold"
                ),

            "time_scope":
                self._normalise(
                    request.get(
                        "time_scope",
                        "all",
                    )
                )
                or
                "all",

            "continuous":
                bool(
                    request.get(
                        "continuous",
                        False,
                    )
                ),

            "chart_type":
                chart_type,
        }

    # ========================================================
    # EMPTY PARSER FALLBACK
    # ========================================================

    @staticmethod
    def _empty_request(
    ) -> dict[str, Any]:

        return {
            "intent":
                "",

            "metric_name":
                None,

            "metric_x":
                None,

            "metric_y":
                None,

            "room":
                None,

            "room_names":
                [],

            "aggregation":
                "mean",

            "frequency":
                "hourly",

            "start_date":
                None,

            "end_date":
                None,

            "top_n":
                10,

            "ascending":
                False,

            "threshold":
                None,

            "time_scope":
                "all",

            "continuous":
                False,

            "chart_type":
                "bar",
        }

    # ========================================================
    # APPLY HYBRID DETERMINISTIC RULES
    # ========================================================

    def _apply_query_rules(
        self,
        user_query: str,
        request: dict[str, Any],
    ) -> dict[str, Any]:

        query = user_query.casefold()

        rooms = (
            self._extract_rooms_from_query(
                user_query
            )
        )

        metrics = (
            self._extract_metrics_in_order(
                user_query
            )
        )

        intent = (
            self._determine_intent(
                user_query=
                    user_query,

                extracted_rooms=
                    rooms,

                model_intent=
                    request.get(
                        "intent",
                        "",
                    ),
            )
        )

        # ----------------------------------------------------
        # METRIC MAPPING
        # ----------------------------------------------------

        metric_name: str | None = None
        metric_x: str | None = None
        metric_y: str | None = None

        if intent == "relationship":

            if len(metrics) >= 1:

                metric_x = (
                    metrics[0]
                )

            if len(metrics) >= 2:

                metric_y = (
                    metrics[1]
                )

        else:

            if metrics:

                metric_name = (
                    metrics[0]
                )

            elif (
                request.get(
                    "metric_name"
                )
                in SUPPORTED_METRICS
            ):

                metric_name = (
                    request[
                        "metric_name"
                    ]
                )

        # ----------------------------------------------------
        # ROOM MAPPING
        # ----------------------------------------------------

        room: str | None = None
        room_names: list[str] = []

        if intent == "room_trend":

            if len(rooms) == 1:

                room = rooms[0]

                room_names = [
                    rooms[0]
                ]

        elif intent == "compare_rooms":

            room_names = list(
                rooms
            )

        elif intent == "relationship":

            if len(rooms) == 1:

                room = rooms[0]

                room_names = [
                    rooms[0]
                ]

        elif intent == "distribution":

            if len(rooms) == 1:

                room = rooms[0]

                room_names = [
                    rooms[0]
                ]

        # ----------------------------------------------------
        # DATES
        # ----------------------------------------------------

        start_date, end_date = (
            self._extract_dates(
                user_query
            )
        )

        # ----------------------------------------------------
        # RANKING SIZE
        # ----------------------------------------------------

        top_n = (
            self._extract_top_n(
                user_query
            )
        )

        # ----------------------------------------------------
        # LATEST QUERY
        # ----------------------------------------------------

        latest_query = (
            self._is_latest_query(
                user_query
            )
        )

        # ----------------------------------------------------
        # SINGULAR LATEST RANKING
        #
        # Example:
        # "Which room has the highest CO2 right now?"
        #
        # The default ranking size is normally 10, but the
        # wording clearly requests only one room.
        # ----------------------------------------------------

        if (
            intent == "rank_rooms"
            and
            latest_query
            and
            self._is_singular_rank_query(
                user_query
            )
        ):

            top_n = 1

        # ----------------------------------------------------
        # ASCENDING / DESCENDING
        # ----------------------------------------------------

        ascending = bool(
            re.search(
                r"\bbottom\b|"
                r"\blowest\b|"
                r"\bcoldest\b|"
                r"\bleast\b",
                query,
            )
        )

        # ----------------------------------------------------
        # THRESHOLD
        # ----------------------------------------------------

        threshold = None

        if intent == "threshold":

            threshold = (
                self._extract_threshold(
                    user_query
                )
            )

        # ----------------------------------------------------
        # AGGREGATION
        # ----------------------------------------------------

        if re.search(
            r"\bmaximum\b|\bmax\b",
            query,
        ):

            aggregation = "max"

        elif re.search(
            r"\bminimum\b|\bmin\b",
            query,
        ):

            aggregation = "min"

        elif re.search(
            r"\bsum\b|\btotal\b",
            query,
        ):

            aggregation = "sum"

        else:

            aggregation = "mean"

        # ----------------------------------------------------
        # FREQUENCY
        # ----------------------------------------------------

        if re.search(
            r"\bdaily\b|"
            r"\bper\s+day\b",
            query,
        ):

            frequency = "daily"

        else:

            frequency = "hourly"

        # ----------------------------------------------------
        # TIME SCOPE
        # ----------------------------------------------------

        if latest_query:

            time_scope = (
                "latest"
            )

        else:

            time_scope = (
                "all"
            )

        # ----------------------------------------------------
        # CONTINUOUS
        # ----------------------------------------------------

        continuous = bool(
            re.search(
                r"\bcontinuous\b|"
                r"\bcontinuously\b|"
                r"\bconsecutive\b|"
                r"\bpersistent\b|"
                r"\bpersistently\b",
                query,
            )
        )

        # ----------------------------------------------------
        # CHART
        # ----------------------------------------------------

        chart_type = (
            self._select_chart(
                user_query=
                    user_query,

                intent=
                    intent,
            )
        )

        return {
            "intent":
                intent,

            "metric_name":
                metric_name,

            "metric_x":
                metric_x,

            "metric_y":
                metric_y,

            "room":
                room,

            "room_names":
                room_names,

            "aggregation":
                aggregation,

            "frequency":
                frequency,

            "start_date":
                start_date,

            "end_date":
                end_date,

            "top_n":
                top_n,

            "ascending":
                ascending,

            "threshold":
                threshold,

            "time_scope":
                time_scope,

            "continuous":
                continuous,

            "chart_type":
                chart_type,
        }

    # ========================================================
    # PARSE
    # ========================================================

    def parse(
        self,
        user_query: str,
    ) -> dict[str, Any]:

        if not isinstance(
            user_query,
            str,
        ):

            raise TypeError(
                "User query must be a string."
            )

        user_query = (
            user_query.strip()
        )

        if not user_query:

            raise ValueError(
                "User query cannot be empty."
            )

        source = (
            "llm_plus_deterministic_rules"
        )

        parser_error: str | None = None

        try:

            raw_request = (
                self._request_json(
                    system_prompt=
                        self._parser_system_prompt(),

                    user_query=
                        user_query,

                    schema=
                        REQUEST_SCHEMA,
                )
            )

            cleaned_request = (
                self._clean_raw_request(
                    raw_request
                )
            )

        except Exception as error:

            parser_error = (
                f"{type(error).__name__}: "
                f"{error}"
            )

            source = (
                "deterministic_parser_fallback"
            )

            cleaned_request = (
                self._empty_request()
            )

        final_request = (
            self._apply_query_rules(
                user_query=
                    user_query,

                request=
                    cleaned_request,
            )
        )

        intent = (
            final_request[
                "intent"
            ]
        )

        chart_type = (
            final_request.pop(
                "chart_type"
            )
        )

        return {
            "intent":
                intent,

            "chart_type":
                chart_type,

            "mapping":
                final_request,

            "source":
                source,

            "parser_error":
                parser_error,
        }

    # ========================================================
    # FLOAT CONVERSION
    # ========================================================

    @staticmethod
    def _to_float(
        value: Any,
    ) -> float | None:

        try:

            number = float(
                value
            )

        except (
            TypeError,
            ValueError,
        ):

            return None

        if not math.isfinite(
            number
        ):

            return None

        return number

    # ========================================================
    # NUMBER FORMAT
    # ========================================================

    @staticmethod
    def _format_number(
        value: float | None,
    ) -> str | None:

        if value is None:

            return None

        if float(
            value
        ).is_integer():

            return str(
                int(
                    value
                )
            )

        return (
            f"{value:.2f}"
            .rstrip("0")
            .rstrip(".")
        )

    # ========================================================
    # NUMERIC VALUES
    # ========================================================

    def _numeric_values(
        self,
        rows: list[dict[str, Any]],
        field: str,
    ) -> list[float]:

        values: list[float] = []

        for row in rows:

            value = (
                self._to_float(
                    row.get(
                        field
                    )
                )
            )

            if value is not None:

                values.append(
                    value
                )

        return values

    # ========================================================
    # SUMMARY STATS
    # ========================================================

    def _summary_stats(
        self,
        values: list[float],
    ) -> dict[str, Any]:

        if not values:

            return {}

        return {
            "count":
                len(
                    values
                ),

            "minimum":
                self._format_number(
                    min(
                        values
                    )
                ),

            "maximum":
                self._format_number(
                    max(
                        values
                    )
                ),

            "mean":
                self._format_number(
                    mean(
                        values
                    )
                ),

            "median":
                self._format_number(
                    median(
                        values
                    )
                ),
        }

    # ========================================================
    # PEARSON
    # ========================================================

    @staticmethod
    def _pearson(
        x_values: list[float],
        y_values: list[float],
    ) -> float | None:

        if (
            len(x_values)
            !=
            len(y_values)
            or
            len(x_values) < 2
        ):

            return None

        x_mean = mean(
            x_values
        )

        y_mean = mean(
            y_values
        )

        numerator = sum(
            (
                x - x_mean
            )
            *
            (
                y - y_mean
            )
            for x, y in zip(
                x_values,
                y_values,
            )
        )

        x_variance = sum(
            (
                x - x_mean
            )
            ** 2
            for x in x_values
        )

        y_variance = sum(
            (
                y - y_mean
            )
            ** 2
            for y in y_values
        )

        denominator = math.sqrt(
            x_variance
            *
            y_variance
        )

        if denominator == 0:

            return None

        return (
            numerator
            /
            denominator
        )

    # ========================================================
    # CORRELATION DESCRIPTION
    # ========================================================

    @staticmethod
    def _correlation_description(
        value: float | None,
    ) -> str | None:

        if value is None:

            return None

        absolute = abs(
            value
        )

        if absolute < 0.20:

            strength = (
                "very weak"
            )

        elif absolute < 0.40:

            strength = (
                "weak"
            )

        elif absolute < 0.60:

            strength = (
                "moderate"
            )

        elif absolute < 0.80:

            strength = (
                "strong"
            )

        else:

            strength = (
                "very strong"
            )

        if value > 0:

            direction = (
                "positive"
            )

        elif value < 0:

            direction = (
                "negative"
            )

        else:

            return (
                "no linear association"
            )

        return (
            f"{strength} "
            f"{direction} "
            f"linear association"
        )

    # ========================================================
    # BUILD INSIGHT EVIDENCE
    # ========================================================

    def _build_evidence(
        self,
        request: dict[str, Any],
        query_type: str,
        rows: list[dict[str, Any]],
        row_count: int,
        truncated: bool,
    ) -> dict[str, Any]:

        evidence: dict[str, Any] = {
            "query_type":
                query_type,

            "metric_name":
                request.get(
                    "metric_name"
                ),

            "metric_x":
                request.get(
                    "metric_x"
                ),

            "metric_y":
                request.get(
                    "metric_y"
                ),

            "room":
                request.get(
                    "room"
                ),

            "row_count":
                row_count,

            "stored_row_count":
                len(
                    rows
                ),

            "truncated":
                truncated,
        }

        # ----------------------------------------------------
        # TREND
        # ----------------------------------------------------

        if query_type == "room_trend":

            values = (
                self._numeric_values(
                    rows,
                    "value",
                )
            )

            evidence[
                "statistics"
            ] = (
                self._summary_stats(
                    values
                )
            )

            if rows:

                evidence[
                    "start_time"
                ] = rows[0].get(
                    "start_time"
                )

                evidence[
                    "end_time"
                ] = rows[-1].get(
                    "start_time"
                )

                evidence[
                    "first_value"
                ] = rows[0].get(
                    "value"
                )

                evidence[
                    "last_value"
                ] = rows[-1].get(
                    "value"
                )

        # ----------------------------------------------------
        # RANK / COMPARE / THRESHOLD / LATEST
        # ----------------------------------------------------

        elif query_type in {
            "compare_rooms",
            "rank_rooms",
            "rank_rooms_latest",
            "threshold",
        }:

            evidence[
                "room_results"
            ] = rows[:20]

            evidence[
                "ascending"
            ] = request.get(
                "ascending"
            )

            evidence[
                "threshold"
            ] = request.get(
                "threshold"
            )

            evidence[
                "time_scope"
            ] = request.get(
                "time_scope"
            )

        # ----------------------------------------------------
        # DISTRIBUTION
        # ----------------------------------------------------

        elif query_type == "distribution":

            values = (
                self._numeric_values(
                    rows,
                    "value",
                )
            )

            evidence[
                "statistics"
            ] = (
                self._summary_stats(
                    values
                )
            )

            evidence[
                "room_count"
            ] = len(
                {
                    str(
                        row.get(
                            "display_name"
                        )
                    )
                    for row in rows
                    if row.get(
                        "display_name"
                    )
                    is not None
                }
            )

            ranked_rooms: dict[
                int,
                dict[str, Any]
            ] = {}

            for row in rows:

                try:

                    rank = int(
                        row.get(
                            "room_rank"
                        )
                    )

                except (
                    TypeError,
                    ValueError,
                ):

                    continue

                if rank in ranked_rooms:

                    continue

                ranked_rooms[
                    rank
                ] = {
                    "rank":
                        rank,

                    "room":
                        row.get(
                            "display_name"
                        ),

                    "ranking_value":
                        row.get(
                            "ranking_value"
                        ),

                    "selection_type":
                        row.get(
                            "selection_type"
                        ),
                }

            if ranked_rooms:

                evidence[
                    "ranked_rooms"
                ] = [
                    ranked_rooms[
                        rank
                    ]
                    for rank in sorted(
                        ranked_rooms
                    )
                ]

        # ----------------------------------------------------
        # RELATIONSHIP
        # ----------------------------------------------------

        elif query_type == "relationship":

            x_values: list[float] = []
            y_values: list[float] = []

            for row in rows:

                x_value = (
                    self._to_float(
                        row.get(
                            "x_value"
                        )
                    )
                )

                y_value = (
                    self._to_float(
                        row.get(
                            "y_value"
                        )
                    )
                )

                if (
                    x_value is None
                    or
                    y_value is None
                ):

                    continue

                x_values.append(
                    x_value
                )

                y_values.append(
                    y_value
                )

            correlation = (
                self._pearson(
                    x_values,
                    y_values,
                )
            )

            evidence[
                "pearson_r"
            ] = (
                None
                if correlation is None
                else round(
                    correlation,
                    3,
                )
            )

            evidence[
                "association_description"
            ] = (
                self._correlation_description(
                    correlation
                )
            )

            evidence[
                "paired_observations"
            ] = len(
                x_values
            )

            evidence[
                "room_count"
            ] = len(
                {
                    str(
                        row.get(
                            "display_name"
                        )
                    )
                    for row in rows
                    if row.get(
                        "display_name"
                    )
                    is not None
                }
            )

        return evidence

    # ========================================================
    # INSIGHT PROMPT
    # ========================================================

    @staticmethod
    def _insight_system_prompt(
    ) -> str:

        return """
You are the SAME monolithic SmartViz assistant that interpreted
the original user request.

Write a concise analytical insight using only the supplied
deterministic evidence.

RULES

- Do not invent numbers.
- Do not invent rooms.
- Do not invent dates.
- Do not invent measurement units.
- CO2 must be described as a CO2 value unless a verified unit
  is explicitly supplied in the evidence.
- Do not claim causation from relationship data.
- Do not use words such as causes, influences, drives or predicts
  to imply causality.
- Pearson correlation describes linear association only.
- "Right now" means latest available historical reading,
  not live real-time data.
- Only include a latest-data caution when the request actually
  concerns latest/current data.
- Do not invent a duration or rate of change.
- Do not claim a trend is steadily increasing/decreasing unless
  that fact is explicitly supplied in the evidence.
- Return JSON only.

summary:
One concise main finding.

key_points:
1 to 3 evidence-grounded points.

caution:
Short limitation when required, otherwise an empty string.
""".strip()

    # ========================================================
    # DETERMINISTIC INSIGHT FALLBACK
    # ========================================================

    def _deterministic_insight(
        self,
        evidence: dict[str, Any],
    ) -> dict[str, Any]:

        query_type = (
            evidence.get(
                "query_type"
            )
        )

        # ----------------------------------------------------
        # RELATIONSHIP
        # ----------------------------------------------------

        if query_type == "relationship":

            r_value = (
                evidence.get(
                    "pearson_r"
                )
            )

            description = (
                evidence.get(
                    "association_description"
                )
            )

            metric_x = (
                evidence.get(
                    "metric_x"
                )
            )

            metric_y = (
                evidence.get(
                    "metric_y"
                )
            )

            if r_value is not None:

                summary = (
                    f"The relationship between "
                    f"{metric_x} and {metric_y} has "
                    f"Pearson r = {r_value}, indicating "
                    f"a {description}."
                )

            else:

                summary = (
                    "A valid Pearson correlation could "
                    "not be calculated."
                )

            return {
                "summary":
                    summary,

                "key_points": [
                    (
                        f"Paired observations: "
                        f"{evidence.get('paired_observations', 0)}."
                    )
                ],

                "caution":
                    (
                        "This describes statistical association "
                        "and does not establish causation."
                    ),
            }

        # ----------------------------------------------------
        # DISTRIBUTION
        # ----------------------------------------------------

        if query_type == "distribution":

            statistics = (
                evidence.get(
                    "statistics",
                    {},
                )
            )

            ranked_rooms = (
                evidence.get(
                    "ranked_rooms",
                    [],
                )
            )

            if ranked_rooms:

                first = (
                    ranked_rooms[0]
                )

                ranking_value = (
                    self._format_number(
                        self._to_float(
                            first.get(
                                "ranking_value"
                            )
                        )
                    )
                )

                summary = (
                    f"{first.get('room')} is rank 1 "
                    f"with ranking value "
                    f"{ranking_value}."
                )

            else:

                summary = (
                    f"The distribution contains "
                    f"{evidence.get('row_count', 0)} "
                    "observations."
                )

            key_points: list[str] = []

            if statistics:

                key_points.append(
                    (
                        f"Observed values range from "
                        f"{statistics.get('minimum')} to "
                        f"{statistics.get('maximum')}, "
                        f"with mean "
                        f"{statistics.get('mean')} and "
                        f"median "
                        f"{statistics.get('median')}."
                    )
                )

            return {
                "summary":
                    summary,

                "key_points":
                    key_points,

                "caution":
                    "",
            }

        # ----------------------------------------------------
        # RANK / COMPARE / LATEST / THRESHOLD
        # ----------------------------------------------------

        if query_type in {
            "rank_rooms",
            "rank_rooms_latest",
            "compare_rooms",
            "threshold",
        }:

            room_results = (
                evidence.get(
                    "room_results",
                    [],
                )
            )

            if not room_results:

                return {
                    "summary":
                        (
                            "No matching data was available "
                            "for this request."
                        ),

                    "key_points":
                        [],

                    "caution":
                        "",
                }

            first = (
                room_results[0]
            )

            value = (
                first.get(
                    "latest_value"
                )
                if query_type
                ==
                "rank_rooms_latest"
                else
                first.get(
                    "average_value"
                )
            )

            formatted_value = (
                self._format_number(
                    self._to_float(
                        value
                    )
                )
            )

            summary = (
                f"{first.get('display_name')} is the "
                f"first returned room with value "
                f"{formatted_value}."
            )

            caution = ""

            if query_type == "rank_rooms_latest":

                caution = (
                    "This is the latest available historical "
                    "reading, not live real-time data."
                )

            return {
                "summary":
                    summary,

                "key_points":
                    [],

                "caution":
                    caution,
            }

        # ----------------------------------------------------
        # TREND
        # ----------------------------------------------------

        if query_type == "room_trend":

            statistics = (
                evidence.get(
                    "statistics",
                    {},
                )
            )

            summary = (
                f"The trend contains "
                f"{evidence.get('row_count', 0)} "
                "observations."
            )

            key_points: list[str] = []

            if statistics:

                key_points.append(
                    (
                        f"Mean = {statistics.get('mean')}, "
                        f"minimum = {statistics.get('minimum')}, "
                        f"maximum = {statistics.get('maximum')}."
                    )
                )

            return {
                "summary":
                    summary,

                "key_points":
                    key_points,

                "caution":
                    "",
            }

        return {
            "summary":
                "Validated analytical results are available.",

            "key_points":
                [],

            "caution":
                "",
        }

    # ========================================================
    # GENERATE MONOLITHIC INSIGHT
    # ========================================================

    def _generate_insight(
        self,
        user_query: str,
        evidence: dict[str, Any],
        has_data: bool,
    ) -> dict[str, Any]:

        if not has_data:

            return {
                "success":
                    True,

                "generated_by":
                    "single_agent_deterministic",

                "summary":
                    (
                        "No matching data was available "
                        "for this request."
                    ),

                "key_points":
                    [],

                "caution":
                    "",

                "evidence":
                    evidence,

                "error":
                    None,
            }

        payload = {
            "original_user_query":
                user_query,

            "deterministic_evidence":
                evidence,
        }

        try:

            response = (
                self._request_json(
                    system_prompt=
                        self._insight_system_prompt(),

                    user_query=
                        json.dumps(
                            payload,
                            ensure_ascii=False,
                        ),

                    schema=
                        INSIGHT_SCHEMA,
                )
            )

            summary = str(
                response.get(
                    "summary",
                    "",
                )
            ).strip()

            key_points_raw = (
                response.get(
                    "key_points",
                    [],
                )
            )

            if not isinstance(
                key_points_raw,
                list,
            ):

                key_points_raw = []

            key_points = [
                str(
                    point
                ).strip()
                for point in key_points_raw
                if str(
                    point
                ).strip()
            ]

            caution = str(
                response.get(
                    "caution",
                    "",
                )
            ).strip()

            if not summary:

                raise ValueError(
                    "Single-agent insight returned "
                    "an empty summary."
                )

            return {
                "success":
                    True,

                "generated_by":
                    "single_agent_llm",

                "summary":
                    summary,

                "key_points":
                    key_points[
                        :3
                    ],

                "caution":
                    caution,

                "evidence":
                    evidence,

                "error":
                    None,
            }

        except Exception as error:

            fallback = (
                self._deterministic_insight(
                    evidence
                )
            )

            return {
                "success":
                    True,

                "generated_by":
                    "single_agent_deterministic_fallback",

                "summary":
                    fallback[
                        "summary"
                    ],

                "key_points":
                    fallback[
                        "key_points"
                    ],

                "caution":
                    fallback[
                        "caution"
                    ],

                "evidence":
                    evidence,

                "error":
                    (
                        f"{type(error).__name__}: "
                        f"{error}"
                    ),
            }

    # ========================================================
    # PUBLIC RUN
    # ========================================================

    def run(
        self,
        user_query: str,
    ) -> dict[str, Any]:

        errors: list[str] = []
        warnings: list[str] = []

        # ====================================================
        # 1. MONOLITHIC PARSER
        # ====================================================

        parsed = (
            self.parse(
                user_query
            )
        )

        intent = (
            parsed[
                "intent"
            ]
        )

        chart_type = (
            parsed[
                "chart_type"
            ]
        )

        mapping = (
            parsed[
                "mapping"
            ]
        )

        if parsed.get(
            "parser_error"
        ):

            warnings.append(
                "Single-agent parser used deterministic "
                "fallback: "
                f"{parsed['parser_error']}"
            )

        # ====================================================
        # 2. REQUEST VALIDATION
        # ====================================================

        validation = (
            self.request_validator
            .validate(
                user_query=
                    user_query,

                intent=
                    intent,

                chart_type=
                    chart_type,

                mapping=
                    mapping,
            )
        )

        warnings.extend(
            validation.get(
                "warnings",
                [],
            )
        )

        errors.extend(
            validation.get(
                "errors",
                [],
            )
        )

        validated_request = (
            validation[
                "request"
            ]
        )

        # ====================================================
        # INVALID REQUEST
        # ====================================================

        if not validation[
            "valid"
        ]:

            return {
                "user_query":
                    user_query,

                "system":
                    "hybrid_single_agent",

                "parser_source":
                    parsed[
                        "source"
                    ],

                "intent":
                    intent,

                "chart_type":
                    chart_type,

                "mapping":
                    mapping,

                "validation":
                    validation,

                "validated_request":
                    validated_request,

                "query_type":
                    None,

                "execution_success":
                    False,

                "result_valid":
                    False,

                "result_status":
                    "invalid_request",

                "result_has_data":
                    False,

                "sql_results":
                    [],

                "sql_row_count":
                    0,

                "sql_stored_row_count":
                    0,

                "sql_result_truncated":
                    False,

                "visualization_success":
                    False,

                "chart_created":
                    False,

                "final_chart_type":
                    None,

                "chart_path":
                    None,

                "chart_title":
                    None,

                "insight_generation_success":
                    False,

                "insight_generated_by":
                    None,

                "insight_summary":
                    "",

                "insight_key_points":
                    [],

                "insight_caution":
                    "",

                "warnings":
                    warnings,

                "errors":
                    errors,
            }

        # ====================================================
        # 3. QUERY BUILDER
        # ====================================================

        query_result = (
            self.query_builder
            .build(
                validated_request
            )
        )

        sql = (
            query_result[
                "sql"
            ]
        )

        params = (
            query_result[
                "params"
            ]
        )

        query_type = (
            query_result[
                "query_type"
            ]
        )

        # ====================================================
        # 4. SQL EXECUTION
        # ====================================================

        try:

            execution = (
                self.sql_executor
                .execute(
                    sql=
                        sql,

                    params=
                        params,

                    query_type=
                        query_type,
                )
            )

            execution_success = (
                True
            )

        except Exception as error:

            execution_success = (
                False
            )

            errors.append(
                "SQL execution failed: "
                f"{type(error).__name__}: "
                f"{error}"
            )

            execution = {
                "columns":
                    [],

                "rows":
                    [],

                "row_count":
                    0,

                "stored_row_count":
                    0,

                "truncated":
                    False,
            }

        rows = (
            execution.get(
                "rows",
                [],
            )
        )

        row_count = (
            execution.get(
                "row_count",
                0,
            )
        )

        truncated = bool(
            execution.get(
                "truncated",
                False,
            )
        )

        # ====================================================
        # 5. RESULT VALIDATION
        # ====================================================

        result_validation = (
            self.result_validator
            .validate(
                request=
                    validated_request,

                query_type=
                    query_type,

                execution_success=
                    execution_success,

                columns=
                    execution.get(
                        "columns",
                        [],
                    ),

                rows=
                    rows,

                row_count=
                    row_count,

                truncated=
                    truncated,
            )
        )

        warnings.extend(
            result_validation.get(
                "warnings",
                [],
            )
        )

        errors.extend(
            result_validation.get(
                "errors",
                [],
            )
        )

        result_valid = bool(
            result_validation[
                "valid"
            ]
        )

        result_has_data = bool(
            result_validation[
                "has_data"
            ]
        )

        # ====================================================
        # 6. VISUALIZATION
        # ====================================================

        visualization = (
            self.visualizer
            .create(
                user_query=
                    user_query,

                intent=
                    intent,

                query_type=
                    query_type,

                metric_name=
                    mapping.get(
                        "metric_name"
                    ),

                metric_x=
                    mapping.get(
                        "metric_x"
                    ),

                metric_y=
                    mapping.get(
                        "metric_y"
                    ),

                rows=
                    rows,

                result_valid=
                    result_valid,

                result_has_data=
                    result_has_data,

                chart_type=
                    chart_type,
            )
        )

        warnings.extend(
            visualization.get(
                "warnings",
                [],
            )
        )

        if visualization.get(
            "error"
        ):

            errors.append(
                "Visualization failed: "
                f"{visualization['error']}"
            )

        # ====================================================
        # 7. MONOLITHIC INSIGHT
        # ====================================================

        evidence = (
            self._build_evidence(
                request=
                    validated_request,

                query_type=
                    query_type,

                rows=
                    rows,

                row_count=
                    row_count,

                truncated=
                    truncated,
            )
        )

        insight = (
            self._generate_insight(
                user_query=
                    user_query,

                evidence=
                    evidence,

                has_data=
                    result_has_data,
            )
        )

        # ====================================================
        # FINAL STATE
        # ====================================================

        return {
            "user_query":
                user_query,

            "system":
                "hybrid_single_agent",

            # ------------------------------------------------
            # Parsing
            # ------------------------------------------------

            "parser_source":
                parsed[
                    "source"
                ],

            "intent":
                intent,

            "chart_type":
                chart_type,

            "mapping":
                mapping,

            # ------------------------------------------------
            # Validation
            # ------------------------------------------------

            "validation":
                validation,

            "validated_request":
                validated_request,

            # ------------------------------------------------
            # Query
            # ------------------------------------------------

            "sql_query":
                sql,

            "sql_params":
                params,

            "query_type":
                query_type,

            # ------------------------------------------------
            # SQL
            # ------------------------------------------------

            "execution_success":
                execution_success,

            "sql_columns":
                execution.get(
                    "columns",
                    [],
                ),

            "sql_results":
                rows,

            "sql_row_count":
                row_count,

            "sql_stored_row_count":
                execution.get(
                    "stored_row_count",
                    len(
                        rows
                    ),
                ),

            "sql_result_truncated":
                truncated,

            # ------------------------------------------------
            # Result validation
            # ------------------------------------------------

            "result_validation":
                result_validation,

            "result_valid":
                result_valid,

            "result_status":
                result_validation[
                    "status"
                ],

            "result_has_data":
                result_has_data,

            # ------------------------------------------------
            # Visualization
            # ------------------------------------------------

            "visualization":
                visualization,

            "visualization_success":
                bool(
                    visualization.get(
                        "success",
                        False,
                    )
                ),

            "chart_created":
                bool(
                    visualization.get(
                        "chart_created",
                        False,
                    )
                ),

            "final_chart_type":
                visualization.get(
                    "chart_type"
                ),

            "chart_path":
                visualization.get(
                    "chart_path"
                ),

            "chart_title":
                visualization.get(
                    "chart_title"
                ),

            # ------------------------------------------------
            # Single-agent insight
            # ------------------------------------------------

            "insight_generation_success":
                bool(
                    insight.get(
                        "success",
                        False,
                    )
                ),

            "insight_generated_by":
                insight.get(
                    "generated_by"
                ),

            "insight_summary":
                insight.get(
                    "summary",
                    "",
                ),

            "insight_key_points":
                insight.get(
                    "key_points",
                    [],
                ),

            "insight_caution":
                insight.get(
                    "caution",
                    "",
                ),

            "insight_evidence":
                evidence,

            "insight_error":
                insight.get(
                    "error"
                ),

            # ------------------------------------------------
            # Diagnostics
            # ------------------------------------------------

            "warnings":
                warnings,

            "errors":
                errors,
        }