from __future__ import annotations

import json
import math
from statistics import mean, median
from typing import Any

from src.agents.base_agent import BaseOllamaAgent


class InsightAgent(BaseOllamaAgent):
    """
    SmartViz Insight Agent.

    Architecture:
        validated SQL results
            ->
        deterministic fact extraction
            ->
        local LLM explanation

    The LLM is NOT responsible for calculating statistics.

    This reduces:
    - hallucinated values;
    - incorrect rankings;
    - unsupported causal claims;
    - inconsistency between chart and explanation.

    Supported query types:
    - room_trend
    - compare_rooms
    - rank_rooms
    - rank_rooms_latest
    - threshold
    - distribution
    - relationship
    """

    OUTPUT_SCHEMA = {
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

    SYSTEM_PROMPT = """
You are the SmartViz Insight Agent.

Your task is to explain deterministic analytical facts generated
from validated SQL Server results.

STRICT RULES:

1. Use ONLY the facts provided to you.

2. Never invent:
   - numbers;
   - rooms;
   - dates;
   - rankings;
   - thresholds;
   - units;
   - correlations.

3. Never claim causation from a relationship or scatter plot.

4. For relationship analysis, use terms such as:
   - association;
   - relationship;
   - pattern.

   Do NOT say that one metric causes another.

5. CO2 should be described as "CO2 value".
   Do not invent a measurement unit.

6. If the result refers to the latest observation, describe it as
   the "latest available reading", not live real-time data.

7. Keep the explanation concise and suitable for:
   - facilities managers;
   - building managers;
   - workplace decision-makers.

8. Do not repeat the summary inside key_points.
   Each key point should be a distinct factual observation.

9. If there is no meaningful limitation or caution, 
   return caution as an empty string "".
   do not write phrases such as:
   "No limitations were provided",
   "The facts contain no caution",
   or similar filler.

10. Do not make recommendations that are unsupported by the data.

Return JSON only in the required schema.

summary:
    One short paragraph explaining the main finding.

key_points:
    Between 1 and 3 concise factual observations.

caution:
    A short limitation/caution.
    Use an empty string only when no meaningful caution exists.

11. For ranked distributions:
    - identify the highest or lowest ranked room when provided;
    - mention room names only when they exist in the facts;
    - distinguish ranking_value from the overall distribution statistics.
""".strip()

    # ========================================================
    # NORMALISATION
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
    # FLOAT CONVERSION
    # ========================================================

    @staticmethod
    def _to_float(
        value: Any,
    ) -> float | None:

        if value is None:

            return None

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
    # NUMBER FORMATTING
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
                "CO2 value",

            "temp":
                "temperature",

            "humidity":
                "humidity",

            "occupancy":
                "occupancy",
        }

        return labels.get(
            metric,
            metric_name or "value",
        )

    # ========================================================
    # NUMERIC COLUMN VALUES
    # ========================================================

    def _numeric_values(
        self,
        rows: list[dict[str, Any]],
        field: str,
    ) -> list[float]:

        values: list[float] = []

        for row in rows:

            value = self._to_float(
                row.get(
                    field
                )
            )

            if value is not None:

                values.append(
                    value
                )

        return values

    # ========================================================
    # BASIC SUMMARY STATISTICS
    # ========================================================

    def _summary_statistics(
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
    # ROOM TREND FACTS
    # ========================================================

    def _trend_facts(
        self,
        request: dict[str, Any],
        rows: list[dict[str, Any]],
        row_count: int,
    ) -> dict[str, Any]:

        values = (
            self._numeric_values(
                rows,
                "value",
            )
        )

        facts: dict[str, Any] = {
            "analysis_type":
                "room trend",

            "metric":
                self._metric_label(
                    request.get(
                        "metric_name"
                    )
                ),

            "room":
                request.get(
                    "room"
                ),

            "row_count":
                row_count,

            "statistics":
                self._summary_statistics(
                    values
                ),
        }

        if rows:

            first_row = rows[0]
            last_row = rows[-1]

            first_value = (
                self._to_float(
                    first_row.get(
                        "value"
                    )
                )
            )

            last_value = (
                self._to_float(
                    last_row.get(
                        "value"
                    )
                )
            )

            facts[
                "start_time"
            ] = first_row.get(
                "start_time"
            )

            facts[
                "end_time"
            ] = last_row.get(
                "start_time"
            )

            facts[
                "first_value"
            ] = self._format_number(
                first_value
            )

            facts[
                "last_value"
            ] = self._format_number(
                last_value
            )

            if (
                first_value is not None
                and
                last_value is not None
            ):

                change = (
                    last_value
                    - first_value
                )

                facts[
                    "change"
                ] = self._format_number(
                    change
                )

        return facts

    # ========================================================
    # RANK / COMPARISON FACTS
    # ========================================================

    def _ranking_facts(
        self,
        request: dict[str, Any],
        query_type: str,
        rows: list[dict[str, Any]],
        row_count: int,
    ) -> dict[str, Any]:

        if query_type == "rank_rooms_latest":

            value_field = (
                "latest_value"
            )

        else:

            value_field = (
                "average_value"
            )

        room_results: list[
            dict[str, Any]
        ] = []

        for index, row in enumerate(
            rows,
            start=1,
        ):

            value = self._to_float(
                row.get(
                    value_field
                )
            )

            room_result = {
                "position":
                    index,

                "room":
                    row.get(
                        "display_name"
                    ),

                "value":
                    self._format_number(
                        value
                    ),
            }

            if query_type == "rank_rooms_latest":

                room_result[
                    "latest_time"
                ] = row.get(
                    "latest_start_time"
                )

            room_results.append(
                room_result
            )

        facts: dict[str, Any] = {
            "analysis_type":
                query_type,

            "metric":
                self._metric_label(
                    request.get(
                        "metric_name"
                    )
                ),

            "row_count":
                row_count,

            "ascending":
                bool(
                    request.get(
                        "ascending",
                        False,
                    )
                ),

            "room_results":
                room_results,
        }

        if room_results:

            facts[
                "first_room"
            ] = room_results[0]

            if len(
                room_results
            ) > 1:

                facts[
                    "last_room"
                ] = room_results[-1]

        return facts

    # ========================================================
    # THRESHOLD FACTS
    # ========================================================

    def _threshold_facts(
        self,
        request: dict[str, Any],
        rows: list[dict[str, Any]],
        row_count: int,
    ) -> dict[str, Any]:

        facts = (
            self._ranking_facts(
                request=
                    request,

                query_type=
                    "threshold",

                rows=
                    rows,

                row_count=
                    row_count,
            )
        )

        facts[
            "analysis_type"
        ] = "threshold"

        facts[
            "threshold"
        ] = request.get(
            "threshold"
        )

        facts[
            "threshold_operator"
        ] = request.get(
            "threshold_operator"
        )

        facts[
            "qualifying_room_count"
        ] = row_count

        return facts

    # ========================================================
    # DISTRIBUTION FACTS
    # ========================================================

    def _distribution_facts(
        self,
        request: dict[str, Any],
        rows: list[dict[str, Any]],
        row_count: int,
    ) -> dict[str, Any]:

        values = (
            self._numeric_values(
                rows,
                "value",
            )
        )

        rooms = sorted(
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

        facts: dict[str, Any] = {
            "analysis_type":
                "distribution",

            "metric":
                self._metric_label(
                    request.get(
                        "metric_name"
                    )
                ),

            "row_count":
                row_count,

            "room_count":
                len(
                    rooms
                ),

            "rooms":
                rooms,

            "statistics":
                self._summary_statistics(
                    values
                ),
        }

        requested_room = request.get(
            "room"
        )

        if requested_room is not None:

            facts[
                "distribution_scope"
            ] = "single room"

            facts[
                "room"
            ] = requested_room

        # ----------------------------------------------------
        # TOP / BOTTOM N DISTRIBUTION
        # ----------------------------------------------------

        distribution_top_n = (
            request.get(
                "distribution_top_n"
            )
        )

        if distribution_top_n is not None:

            facts[
                "distribution_scope"
            ] = "ranked rooms"

            facts[
                "requested_room_count"
            ] = distribution_top_n

            ranked_rooms: dict[
                int,
                dict[str, Any],
            ] = {}

            for row in rows:

                room_rank = row.get(
                    "room_rank"
                )

                try:

                    room_rank = int(
                        room_rank
                    )

                except (
                    TypeError,
                    ValueError,
                ):

                    continue

                if room_rank in ranked_rooms:

                    continue

                ranking_value = (
                    self._to_float(
                        row.get(
                            "ranking_value"
                        )
                    )
                )

                ranked_rooms[
                    room_rank
                ] = {
                    "rank":
                        room_rank,

                    "room":
                        row.get(
                            "display_name"
                        ),

                    "ranking_value":
                        self._format_number(
                            ranking_value
                        ),

                    "selection_type":
                        row.get(
                            "selection_type"
                        ),
                }

            facts[
                "ranked_rooms"
            ] = [
                ranked_rooms[
                    rank
                ]
                for rank in sorted(
                    ranked_rooms
                )
            ]

        return facts

    # ========================================================
    # PEARSON CORRELATION
    # ========================================================

    def _pearson_correlation(
        self,
        x_values: list[float],
        y_values: list[float],
    ) -> float | None:

        if (
            len(
                x_values
            )
            !=
            len(
                y_values
            )
        ):

            return None

        if len(
            x_values
        ) < 2:

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

        x_squared = sum(
            (
                x - x_mean
            )
            ** 2
            for x in x_values
        )

        y_squared = sum(
            (
                y - y_mean
            )
            ** 2
            for y in y_values
        )

        denominator = math.sqrt(
            x_squared
            *
            y_squared
        )

        if denominator == 0:

            return None

        correlation = (
            numerator
            /
            denominator
        )

        return max(
            -1.0,
            min(
                1.0,
                correlation,
            ),
        )

    # ========================================================
    # CORRELATION DESCRIPTION
    # ========================================================

    @staticmethod
    def _correlation_description(
        correlation: float | None,
    ) -> str | None:

        if correlation is None:

            return None

        absolute_value = abs(
            correlation
        )

        if absolute_value < 0.20:

            strength = "very weak"

        elif absolute_value < 0.40:

            strength = "weak"

        elif absolute_value < 0.60:

            strength = "moderate"

        elif absolute_value < 0.80:

            strength = "strong"

        else:

            strength = "very strong"

        if correlation > 0:

            direction = "positive"

        elif correlation < 0:

            direction = "negative"

        else:

            direction = "no linear"

        if correlation == 0:

            return (
                "no linear association"
            )

        return (
            f"{strength} "
            f"{direction} "
            f"linear association"
        )

    # ========================================================
    # RELATIONSHIP FACTS
    # ========================================================

    def _relationship_facts(
        self,
        request: dict[str, Any],
        rows: list[dict[str, Any]],
        row_count: int,
        truncated: bool,
    ) -> dict[str, Any]:

        x_values: list[float] = []
        y_values: list[float] = []

        for row in rows:

            x_value = self._to_float(
                row.get(
                    "x_value"
                )
            )

            y_value = self._to_float(
                row.get(
                    "y_value"
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
            self._pearson_correlation(
                x_values=
                    x_values,

                y_values=
                    y_values,
            )
        )

        rooms = {
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

        facts: dict[str, Any] = {
            "analysis_type":
                "relationship",

            "metric_x":
                self._metric_label(
                    request.get(
                        "metric_x"
                    )
                ),

            "metric_y":
                self._metric_label(
                    request.get(
                        "metric_y"
                    )
                ),

            "paired_observation_count":
                row_count,

            "stored_observation_count":
                len(
                    rows
                ),

            "room_count":
                len(
                    rooms
                ),

            "pearson_r":
                (
                    None
                    if correlation is None
                    else round(
                        correlation,
                        3,
                    )
                ),

            "association_description":
                self._correlation_description(
                    correlation
                ),

            "metric_x_statistics":
                self._summary_statistics(
                    x_values
                ),

            "metric_y_statistics":
                self._summary_statistics(
                    y_values
                ),

            "truncated":
                truncated,

            "interpretation_rule":
                (
                    "Association does not establish causation."
                ),
        }

        if request.get(
            "room"
        ):

            facts[
                "room"
            ] = request.get(
                "room"
            )

        if truncated:

            facts[
                "correlation_scope"
            ] = (
                "Pearson correlation was calculated "
                "using the stored subset only."
            )

        else:

            facts[
                "correlation_scope"
            ] = (
                "Pearson correlation was calculated "
                "using all returned paired observations."
            )

        return facts

    # ========================================================
    # BUILD FACTS
    # ========================================================

    def _build_facts(
        self,
        request: dict[str, Any],
        query_type: str,
        rows: list[dict[str, Any]],
        row_count: int,
        truncated: bool,
        result_warnings: list[str],
    ) -> dict[str, Any]:

        if query_type == "room_trend":

            facts = (
                self._trend_facts(
                    request=
                        request,

                    rows=
                        rows,

                    row_count=
                        row_count,
                )
            )

        elif query_type in {
            "compare_rooms",
            "rank_rooms",
            "rank_rooms_latest",
        }:

            facts = (
                self._ranking_facts(
                    request=
                        request,

                    query_type=
                        query_type,

                    rows=
                        rows,

                    row_count=
                        row_count,
                )
            )

        elif query_type == "threshold":

            facts = (
                self._threshold_facts(
                    request=
                        request,

                    rows=
                        rows,

                    row_count=
                        row_count,
                )
            )

        elif query_type == "distribution":

            facts = (
                self._distribution_facts(
                    request=
                        request,

                    rows=
                        rows,

                    row_count=
                        row_count,
                )
            )

        elif query_type == "relationship":

            facts = (
                self._relationship_facts(
                    request=
                        request,

                    rows=
                        rows,

                    row_count=
                        row_count,

                    truncated=
                        truncated,
                )
            )

        else:

            raise ValueError(
                "Unsupported insight query type: "
                f"{query_type}"
            )

        facts[
            "result_warnings"
        ] = result_warnings

        facts[
            "result_truncated"
        ] = truncated

        return facts

    # ========================================================
    # DETERMINISTIC FALLBACK
    # ========================================================

    def _fallback_insight(
        self,
        facts: dict[str, Any],
        error: str | None = None,
    ) -> dict[str, Any]:
        """
        Provide a safe deterministic explanation if the
        local LLM fails or times out.
        """

        analysis_type = facts.get(
            "analysis_type"
        )

        # ----------------------------------------------------
        # RELATIONSHIP
        # ----------------------------------------------------

        if analysis_type == "relationship":

            metric_x = facts.get(
                "metric_x",
                "Metric X",
            )

            metric_y = facts.get(
                "metric_y",
                "Metric Y",
            )

            correlation = facts.get(
                "pearson_r"
            )

            association = facts.get(
                "association_description"
            )

            if correlation is not None:

                summary = (
                    f"The relationship between "
                    f"{metric_x} and {metric_y} "
                    f"has Pearson r = {correlation}. "
                    f"This indicates a "
                    f"{association}."
                )

            else:

                summary = (
                    f"A relationship plot was generated "
                    f"for {metric_x} and {metric_y}, "
                    "but a valid Pearson correlation "
                    "could not be calculated."
                )

            key_points = [
                (
                    "Paired observations: "
                    f"{facts.get('paired_observation_count', 0)}."
                )
            ]

            caution = (
                "This describes statistical association "
                "only and does not establish causation."
            )

        # ----------------------------------------------------
        # DISTRIBUTION
        # ----------------------------------------------------

        elif analysis_type == "distribution":

            metric = facts.get(
                "metric",
                "value",
            )

            statistics = facts.get(
                "statistics",
                {},
            )

            summary = (
                f"The {metric} distribution contains "
                f"{facts.get('row_count', 0)} observations "
                f"across {facts.get('room_count', 0)} room(s)."
            )

            key_points = []

            if statistics:

                key_points.append(
                    (
                        "Mean = "
                        f"{statistics.get('mean')}, "
                        "median = "
                        f"{statistics.get('median')}."
                    )
                )

            ranked_rooms = facts.get(
                "ranked_rooms",
                [],
            )

            if ranked_rooms:

                first_room = (
                    ranked_rooms[0]
                )

                key_points.append(
                    (
                        f"Rank 1 is "
                        f"{first_room.get('room')} "
                        f"with ranking value "
                        f"{first_room.get('ranking_value')}."
                    )
                )

            caution = (
                "The histogram describes the observed "
                "distribution and should not be interpreted "
                "as a causal result."
            )

        # ----------------------------------------------------
        # RANKING / COMPARISON / THRESHOLD
        # ----------------------------------------------------

        elif analysis_type in {
            "rank_rooms",
            "rank_rooms_latest",
            "compare_rooms",
            "threshold",
        }:

            metric = facts.get(
                "metric",
                "value",
            )

            first_room = facts.get(
                "first_room"
            )

            if first_room:

                summary = (
                    f"{first_room.get('room')} is the "
                    f"first returned room for {metric}, "
                    f"with value "
                    f"{first_room.get('value')}."
                )

            else:

                summary = (
                    f"The analysis returned "
                    f"{facts.get('row_count', 0)} "
                    f"room result(s) for {metric}."
                )

            key_points = [
                (
                    f"Rooms returned: "
                    f"{facts.get('row_count', 0)}."
                )
            ]

            if analysis_type == "rank_rooms_latest":

                caution = (
                    "Latest refers to the latest available "
                    "historical reading, not live real-time data."
                )

            else:

                caution = ""

        # ----------------------------------------------------
        # TREND
        # ----------------------------------------------------

        elif analysis_type == "room trend":

            metric = facts.get(
                "metric",
                "value",
            )

            room = facts.get(
                "room",
                "requested room",
            )

            summary = (
                f"The {metric} trend for {room} "
                f"contains {facts.get('row_count', 0)} "
                "observations."
            )

            key_points = []

            statistics = facts.get(
                "statistics",
                {},
            )

            if statistics:

                key_points.append(
                    (
                        "Observed mean = "
                        f"{statistics.get('mean')}; "
                        "minimum = "
                        f"{statistics.get('minimum')}; "
                        "maximum = "
                        f"{statistics.get('maximum')}."
                    )
                )

            caution = ""

        else:

            summary = (
                "Validated analytical results are available."
            )

            key_points = []

            caution = ""

        if facts.get(
            "result_truncated"
        ):

            extra_caution = (
                " Only the stored subset was available "
                "for downstream insight generation."
            )

            caution = (
                caution
                +
                extra_caution
            ).strip()

        return {
            "success":
                True,

            "generated_by":
                "deterministic_fallback",

            "summary":
                summary,

            "key_points":
                key_points,

            "caution":
                caution,

            "facts":
                facts,

            "error":
                error,
        }

    # ========================================================
    # PUBLIC GENERATE METHOD
    # ========================================================

    def generate(
        self,
        user_query: str,
        request: dict[str, Any],
        query_type: str,
        rows: list[dict[str, Any]],
        row_count: int,
        result_valid: bool,
        result_has_data: bool,
        result_warnings: list[str] | None = None,
        truncated: bool = False,
    ) -> dict[str, Any]:

        result_warnings = (
            result_warnings
            or []
        )

        # ----------------------------------------------------
        # INVALID RESULT
        # ----------------------------------------------------

        if not result_valid:

            return {
                "success":
                    False,

                "generated_by":
                    "none",

                "summary":
                    (
                        "No insight was generated because "
                        "the SQL result did not pass validation."
                    ),

                "key_points":
                    [],

                "caution":
                    "",

                "facts":
                    {},

                "error":
                    None,
            }

        # ----------------------------------------------------
        # NO DATA
        # ----------------------------------------------------

        if (
            not result_has_data
            or
            row_count == 0
            or
            not rows
        ):

            return {
                "success":
                    True,

                "generated_by":
                    "deterministic",

                "summary":
                    (
                        "No matching data was available "
                        "for this request."
                    ),

                "key_points":
                    [],

                "caution":
                    "",

                "facts":
                    {},

                "error":
                    None,
            }

        # ----------------------------------------------------
        # BUILD DETERMINISTIC FACTS
        # ----------------------------------------------------

        try:

            facts = (
                self._build_facts(
                    request=
                        request,

                    query_type=
                        query_type,

                    rows=
                        rows,

                    row_count=
                        row_count,

                    truncated=
                        truncated,

                    result_warnings=
                        result_warnings,
                )
            )

        except Exception as error:

            return {
                "success":
                    False,

                "generated_by":
                    "none",

                "summary":
                    "",

                "key_points":
                    [],

                "caution":
                    "",

                "facts":
                    {},

                "error":
                    (
                        f"{type(error).__name__}: "
                        f"{error}"
                    ),
            }

        # ----------------------------------------------------
        # SEND ONLY FACTS TO THE LLM
        # ----------------------------------------------------

        llm_input = {
            "original_user_query":
                user_query,

            "validated_facts":
                facts,
        }

        try:

            response = (
                self._request_json(
                    system_prompt=
                        self.SYSTEM_PROMPT,

                    user_query=
                        json.dumps(
                            llm_input,
                            ensure_ascii=False,
                        ),

                    schema=
                        self.OUTPUT_SCHEMA,
                )
            )

        except Exception as error:

            # ------------------------------------------------
            # Ollama failure must NOT break the whole graph.
            # ------------------------------------------------

            return (
                self._fallback_insight(
                    facts=
                        facts,

                    error=
                        (
                            f"{type(error).__name__}: "
                            f"{error}"
                        ),
                )
            )

        # ----------------------------------------------------
        # BASIC LLM RESPONSE SAFETY
        # ----------------------------------------------------

        summary = str(
            response.get(
                "summary",
                "",
            )
        ).strip()

        key_points = (
            response.get(
                "key_points",
                [],
            )
        )

        caution = str(
            response.get(
                "caution",
                "",
            )
        ).strip()

        if not summary:

            return (
                self._fallback_insight(
                    facts=
                        facts,

                    error=
                        (
                            "Insight LLM returned "
                            "an empty summary."
                        ),
                )
            )

        if not isinstance(
            key_points,
            list,
        ):

            return (
                self._fallback_insight(
                    facts=
                        facts,

                    error=
                        (
                            "Insight LLM returned "
                            "invalid key_points."
                        ),
                )
            )

        clean_key_points = [
            str(
                point
            ).strip()
            for point in key_points
            if str(
                point
            ).strip()
        ]

        return {
            "success":
                True,

            "generated_by":
                "llm",

            "summary":
                summary,

            "key_points":
                clean_key_points[
                    :3
                ],

            "caution":
                caution,

            # ------------------------------------------------
            # Keep deterministic evidence for debugging,
            # evaluation and dissertation traceability.
            # ------------------------------------------------

            "facts":
                facts,

            "error":
                None,
        }
    