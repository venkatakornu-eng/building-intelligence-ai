from __future__ import annotations

import csv
import re
import time
from pathlib import Path
from typing import Any

from src.baseline.hybrid_single_agent import HybridSingleAgent


# ============================================================
# OUTPUT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

OUTPUT_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "hybrid_single_agent_regression_results.csv"
)

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# CONTROLLED TEST CASES
# ============================================================

TEST_CASES: list[dict[str, Any]] = [

    # --------------------------------------------------------
    # T01 - ROOM TREND
    # --------------------------------------------------------

    {
        "id": "T01",
        "prompt":
            "Show the temperature trend for Seminar Room 3.",

        "expected_intent":
            "room_trend",

        "expected_query_type":
            "room_trend",

        "expected_chart":
            "line",

        "expected_valid":
            True,

        "expected_has_data":
            True,

        "expected_chart_created":
            True,

        "expected_mapping": {
            "metric_name":
                "temp",

            "room":
                "Seminar Room 3",

            "aggregation":
                "mean",

            "frequency":
                "hourly",
        },
    },

    # --------------------------------------------------------
    # T02 - ROOM COMPARISON
    # --------------------------------------------------------

    {
        "id": "T02",
        "prompt":
            (
                "Compare average humidity between "
                "Dining Room and Seminar Room 3."
            ),

        "expected_intent":
            "compare_rooms",

        "expected_query_type":
            "compare_rooms",

        "expected_chart":
            "bar",

        "expected_valid":
            True,

        "expected_has_data":
            True,

        "expected_chart_created":
            True,

        "expected_mapping": {
            "metric_name":
                "humidity",

            "room_names": [
                "Dining Room",
                "Seminar Room 3",
            ],

            "aggregation":
                "mean",
        },
    },

    # --------------------------------------------------------
    # T03 - TOP-N RANKING
    # --------------------------------------------------------

    {
        "id": "T03",
        "prompt":
            "Show the top 5 rooms by average CO2.",

        "expected_intent":
            "rank_rooms",

        "expected_query_type":
            "rank_rooms",

        "expected_chart":
            "bar",

        "expected_valid":
            True,

        "expected_has_data":
            True,

        "expected_chart_created":
            True,

        "expected_mapping": {
            "metric_name":
                "co2",

            "top_n":
                5,

            "ascending":
                False,

            "aggregation":
                "mean",
        },
    },

    # --------------------------------------------------------
    # T04 - BOTTOM-N RANKING
    # --------------------------------------------------------

    {
        "id": "T04",
        "prompt":
            "Show the bottom 5 rooms by average occupancy.",

        "expected_intent":
            "rank_rooms",

        "expected_query_type":
            "rank_rooms",

        "expected_chart":
            "bar",

        "expected_valid":
            True,

        "expected_has_data":
            True,

        "expected_chart_created":
            True,

        "expected_mapping": {
            "metric_name":
                "occupancy",

            "top_n":
                5,

            "ascending":
                True,

            "aggregation":
                "mean",
        },
    },

    # --------------------------------------------------------
    # T05 - LATEST AVAILABLE RANKING
    # --------------------------------------------------------

    {
        "id": "T05",
        "prompt":
            "Which room has the highest CO2 right now?",

        "expected_intent":
            "rank_rooms",

        "expected_query_type":
            "rank_rooms_latest",

        "expected_chart":
            "bar",

        "expected_valid":
            True,

        "expected_has_data":
            True,

        "expected_chart_created":
            True,

        "expected_mapping": {
            "metric_name":
                "co2",

            "top_n":
                1,

            "ascending":
                False,

            "time_scope":
                "latest",
        },
    },

    # --------------------------------------------------------
    # T06 - THRESHOLD / EXPECTED NO DATA
    # --------------------------------------------------------

    {
        "id": "T06",
        "prompt":
            "Show rooms with average CO2 above 800.",

        "expected_intent":
            "threshold",

        "expected_query_type":
            "threshold",

        "expected_chart":
            "bar",

        "expected_valid":
            True,

        # Important:
        # This expectation is defined before running the test.
        "expected_has_data":
            False,

        "expected_chart_created":
            False,

        "expected_mapping": {
            "metric_name":
                "co2",

            "aggregation":
                "mean",

            "threshold":
                800.0,
        },
    },

    # --------------------------------------------------------
    # T07 - ALL-ROOM DISTRIBUTION
    # --------------------------------------------------------

    {
        "id": "T07",
        "prompt":
            "Show the humidity distribution across all rooms.",

        "expected_intent":
            "distribution",

        "expected_query_type":
            "distribution",

        "expected_chart":
            "histogram",

        "expected_valid":
            True,

        "expected_has_data":
            True,

        "expected_chart_created":
            True,

        "expected_mapping": {
            "metric_name":
                "humidity",

            "aggregation":
                "mean",
        },
    },

    # --------------------------------------------------------
    # T08 - TOP-N DISTRIBUTION
    # --------------------------------------------------------

    {
        "id": "T08",
        "prompt":
            "Show the humidity distribution of top 4 rooms.",

        "expected_intent":
            "distribution",

        "expected_query_type":
            "distribution",

        "expected_chart":
            "histogram",

        "expected_valid":
            True,

        "expected_has_data":
            True,

        "expected_chart_created":
            True,

        "expected_mapping": {
            "metric_name":
                "humidity",

            "top_n":
                4,

            "ascending":
                False,
        },
    },

    # --------------------------------------------------------
    # T09 - BOTTOM-N DISTRIBUTION
    # --------------------------------------------------------

    {
        "id": "T09",
        "prompt":
            "Show the humidity distribution of bottom 3 rooms.",

        "expected_intent":
            "distribution",

        "expected_query_type":
            "distribution",

        "expected_chart":
            "histogram",

        "expected_valid":
            True,

        "expected_has_data":
            True,

        "expected_chart_created":
            True,

        "expected_mapping": {
            "metric_name":
                "humidity",

            "top_n":
                3,

            "ascending":
                True,
        },
    },

    # --------------------------------------------------------
    # T10 - DATE-FILTERED TOP-N DISTRIBUTION
    # --------------------------------------------------------

    {
        "id": "T10",
        "prompt":
            (
                "Show the humidity distribution of the top 4 "
                "rooms between 2025-05-01 and 2025-05-31."
            ),

        "expected_intent":
            "distribution",

        "expected_query_type":
            "distribution",

        "expected_chart":
            "histogram",

        "expected_valid":
            True,

        "expected_has_data":
            True,

        "expected_chart_created":
            True,

        "expected_mapping": {
            "metric_name":
                "humidity",

            "top_n":
                4,

            "ascending":
                False,

            "start_date":
                "2025-05-01",

            "end_date":
                "2025-05-31",
        },
    },

    # --------------------------------------------------------
    # T11 - RELATIONSHIP ACROSS ALL ROOMS
    # --------------------------------------------------------

    {
        "id": "T11",
        "prompt":
            (
                "Show the relationship between occupancy "
                "and CO2 across all rooms."
            ),

        "expected_intent":
            "relationship",

        "expected_query_type":
            "relationship",

        "expected_chart":
            "scatter",

        "expected_valid":
            True,

        "expected_has_data":
            True,

        "expected_chart_created":
            True,

        "expected_mapping": {
            "metric_name":
                None,

            "metric_x":
                "occupancy",

            "metric_y":
                "co2",
        },
    },

    # --------------------------------------------------------
    # T12 - RELATIONSHIP IN ONE ROOM
    # --------------------------------------------------------

    {
        "id": "T12",
        "prompt":
            (
                "Show the relationship between temperature "
                "and humidity in Seminar Room 3."
            ),

        "expected_intent":
            "relationship",

        "expected_query_type":
            "relationship",

        "expected_chart":
            "scatter",

        "expected_valid":
            True,

        "expected_has_data":
            True,

        "expected_chart_created":
            True,

        "expected_mapping": {
            "metric_name":
                None,

            "metric_x":
                "temp",

            "metric_y":
                "humidity",

            "room":
                "Seminar Room 3",
        },
    },
]


# ============================================================
# BASIC HELPERS
# ============================================================

def bool_value(
    value: Any,
) -> bool:

    return bool(
        value
    )


def normalise_text(
    value: Any,
) -> str:

    if value is None:

        return ""

    return (
        str(value)
        .casefold()
        .strip()
    )


def compact_text(
    value: str,
) -> str:
    """
    Remove punctuation and repeated whitespace so that
    question-repetition checks are less sensitive to formatting.
    """

    value = (
        value
        .casefold()
        .strip()
    )

    value = re.sub(
        r"[^a-z0-9]+",
        " ",
        value,
    )

    return re.sub(
        r"\s+",
        " ",
        value,
    ).strip()


# ============================================================
# MAPPING CHECK
# ============================================================

def mapping_matches(
    actual_mapping: dict[str, Any],
    expected_mapping: dict[str, Any],
) -> tuple[
    bool,
    list[str],
]:

    reasons: list[str] = []

    for key, expected_value in (
        expected_mapping.items()
    ):

        actual_value = (
            actual_mapping.get(
                key
            )
        )

        if isinstance(
            expected_value,
            list,
        ):

            actual_list = (
                actual_value
                if isinstance(
                    actual_value,
                    list,
                )
                else []
            )

            if actual_list != expected_value:

                reasons.append(
                    (
                        f"Mapping mismatch for {key}: "
                        f"expected {expected_value}, "
                        f"received {actual_list}."
                    )
                )

        elif isinstance(
            expected_value,
            float,
        ):

            try:

                actual_number = float(
                    actual_value
                )

            except (
                TypeError,
                ValueError,
            ):

                reasons.append(
                    (
                        f"Mapping mismatch for {key}: "
                        f"expected {expected_value}, "
                        f"received {actual_value}."
                    )
                )

                continue

            if abs(
                actual_number
                -
                expected_value
            ) > 1e-9:

                reasons.append(
                    (
                        f"Mapping mismatch for {key}: "
                        f"expected {expected_value}, "
                        f"received {actual_number}."
                    )
                )

        else:

            if actual_value != expected_value:

                reasons.append(
                    (
                        f"Mapping mismatch for {key}: "
                        f"expected {expected_value}, "
                        f"received {actual_value}."
                    )
                )

    return (
        len(
            reasons
        )
        ==
        0,
        reasons,
    )


# ============================================================
# INSIGHT TEXT
# ============================================================

def combine_insight_text(
    state: dict[str, Any],
) -> str:

    summary = str(
        state.get(
            "insight_summary",
            "",
        )
        or
        ""
    ).strip()

    key_points = (
        state.get(
            "insight_key_points",
            [],
        )
        or
        []
    )

    caution = str(
        state.get(
            "insight_caution",
            "",
        )
        or
        ""
    ).strip()

    parts = [
        summary,
    ]

    for point in key_points:

        cleaned = str(
            point
        ).strip()

        if cleaned:

            parts.append(
                cleaned
            )

    if caution:

        parts.append(
            caution
        )

    return "\n".join(
        parts
    )


# ============================================================
# NUMBER EXTRACTION
# ============================================================

def extract_numbers(
    text: str,
) -> list[float]:

    matches = re.findall(
        r"(?<![\w-])-?\d+(?:\.\d+)?",
        text,
    )

    values: list[float] = []

    for match in matches:

        try:

            values.append(
                float(
                    match
                )
            )

        except ValueError:

            continue

    return values


def number_in_text(
    expected_value: Any,
    text: str,
    tolerance: float = 0.02,
) -> bool:

    try:

        expected = float(
            expected_value
        )

    except (
        TypeError,
        ValueError,
    ):

        return False

    numbers = (
        extract_numbers(
            text
        )
    )

    return any(
        abs(
            value
            -
            expected
        )
        <=
        tolerance
        for value in numbers
    )


# ============================================================
# ROOM + VALUE HELPERS
# ============================================================

def room_in_text(
    room_name: Any,
    text: str,
) -> bool:

    if room_name is None:

        return False

    room = normalise_text(
        room_name
    )

    return (
        room
        in
        text.casefold()
    )


def first_room_result(
    evidence: dict[str, Any],
) -> dict[str, Any] | None:

    room_results = (
        evidence.get(
            "room_results",
            [],
        )
        or
        []
    )

    if not room_results:

        return None

    first = room_results[0]

    if not isinstance(
        first,
        dict,
    ):

        return None

    return first


# ============================================================
# UNSUPPORTED UNIT CHECK
# ============================================================

def contains_unsupported_unit(
    text: str,
) -> bool:
    """
    Units are intentionally treated as unverified in the
    dissertation analytical layer.

    Therefore numerical claims using %, ppm, ppb or Celsius
    are marked unsupported for this controlled evaluation.
    """

    patterns = (
        r"-?\d+(?:\.\d+)?\s*%",
        r"-?\d+(?:\.\d+)?\s*(?:ppm|ppb)\b",
        r"-?\d+(?:\.\d+)?\s*°\s*c\b",
        r"-?\d+(?:\.\d+)?\s*degrees?\s+celsius\b",
        r"-?\d+(?:\.\d+)?\s*percent\b",
    )

    return any(
        re.search(
            pattern,
            text,
            flags=re.IGNORECASE,
        )
        is not None
        for pattern in patterns
    )


# ============================================================
# CAUSAL / PREDICTIVE LANGUAGE
# ============================================================

def contains_causal_language(
    text: str,
) -> bool:

    patterns = (
        r"\bcauses?\b",
        r"\bcaused\s+by\b",
        r"\bleads?\s+to\b",
        r"\bresults?\s+in\b",
        r"\bdrives?\b",
        r"\bdriven\s+by\b",
        r"\binfluences?\b",
        r"\binfluenc(?:ed|ing)\b",
        r"\baffects?\b",
        r"\bimpact(?:s|ed|ing)?\b",
        r"\bresponsible\s+for\b",
        r"\bcontributes?\s+to\b",
        r"\bpredicts?\b",
        r"\bpredicting\b",
        r"\bpredictive\b",
        r"\bprimary\s+factor\b",
    )

    return any(
        re.search(
            pattern,
            text,
            flags=re.IGNORECASE,
        )
        is not None
        for pattern in patterns
    )


# ============================================================
# TREND OVERCLAIM CHECK
# ============================================================

def contains_unsupported_trend_claim(
    text: str,
) -> bool:

    patterns = (
        r"\bsteadily\b",
        r"\bsteady\s+(?:increase|decrease)\b",
        r"\bconsistently\s+(?:increasing|decreasing)\b",
        r"\bper\s+(?:hour|day|week|month|year)\b",
        r"\bmonthly\s+(?:increase|decrease|change|rate)\b",
        r"\bweekly\s+(?:increase|decrease|change|rate)\b",
        r"\bdaily\s+(?:increase|decrease|change|rate)\b",
        r"\bannual\s+(?:increase|decrease|change|rate)\b",
    )

    return any(
        re.search(
            pattern,
            text,
            flags=re.IGNORECASE,
        )
        is not None
        for pattern in patterns
    )


# ============================================================
# FIRST-RANK DIRECTION CHECK
# ============================================================

def ranking_direction_problem(
    text: str,
    first_room: str,
    ascending: bool,
) -> bool:

    escaped_room = re.escape(
        first_room
    )

    if ascending:

        # Bottom/lowest ranking:
        # rank 1 must not be described as the highest.
        patterns = (
            rf"{escaped_room}.{{0,100}}\bhighest\b",
            rf"\bhighest\b.{{0,100}}{escaped_room}",
        )

    else:

        # Top/highest ranking:
        # rank 1 must not be described as the lowest.
        patterns = (
            rf"{escaped_room}.{{0,100}}\blowest\b",
            rf"\blowest\b.{{0,100}}{escaped_room}",
        )

    return any(
        re.search(
            pattern,
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        is not None
        for pattern in patterns
    )


# ============================================================
# INSIGHT RELIABILITY EVALUATOR
# ============================================================

def evaluate_insight_reliability(
    state: dict[str, Any],
    test_case: dict[str, Any],
) -> tuple[
    bool,
    list[str],
]:

    reasons: list[str] = []

    summary = str(
        state.get(
            "insight_summary",
            "",
        )
        or
        ""
    ).strip()

    text = (
        combine_insight_text(
            state
        )
    )

    query_type = (
        state.get(
            "query_type"
        )
    )

    evidence = (
        state.get(
            "insight_evidence",
            {}
        )
        or
        {}
    )

    actual_has_data = bool_value(
        state.get(
            "result_has_data"
        )
    )

    # ========================================================
    # BASIC COMPLETENESS
    # ========================================================

    if not summary:

        reasons.append(
            "Insight summary is empty."
        )

    # --------------------------------------------------------
    # Question repetition
    # --------------------------------------------------------

    if (
        summary
        and
        compact_text(
            summary
        )
        ==
        compact_text(
            test_case[
                "prompt"
            ]
        )
    ):

        reasons.append(
            (
                "Insight summary repeats the user's "
                "question instead of answering it."
            )
        )

    # ========================================================
    # EXPECTED NO-DATA CASE
    # ========================================================

    if not actual_has_data:

        no_data_language = bool(
            re.search(
                r"\bno\s+matching\s+data\b|"
                r"\bno\s+data\b|"
                r"\bno\s+matching\s+results\b|"
                r"\bno\s+matching\s+rooms\b|"
                r"\bno\s+results\b",
                text,
                flags=re.IGNORECASE,
            )
        )

        if not no_data_language:

            reasons.append(
                (
                    "No-data result is not clearly explained "
                    "to the user."
                )
            )

        return (
            len(
                reasons
            )
            ==
            0,
            reasons,
        )

    # ========================================================
    # UNSUPPORTED MEASUREMENT UNITS
    # ========================================================

    if contains_unsupported_unit(
        text
    ):

        reasons.append(
            (
                "Insight introduces an unsupported "
                "measurement unit."
            )
        )

    # ========================================================
    # TREND
    # ========================================================

    if query_type == "room_trend":

        room = (
            evidence.get(
                "room"
            )
        )

        if (
            room
            and
            not room_in_text(
                room,
                text,
            )
        ):

            reasons.append(
                (
                    "Trend insight does not mention "
                    "the requested room."
                )
            )

        statistics = (
            evidence.get(
                "statistics",
                {}
            )
            or
            {}
        )

        evidence_numbers = [
            statistics.get(
                "mean"
            ),
            statistics.get(
                "minimum"
            ),
            statistics.get(
                "maximum"
            ),
            statistics.get(
                "median"
            ),
            evidence.get(
                "first_value"
            ),
            evidence.get(
                "last_value"
            ),
        ]

        grounded_number_found = any(
            (
                value is not None
                and
                number_in_text(
                    value,
                    text,
                )
            )
            for value in evidence_numbers
        )

        if not grounded_number_found:

            reasons.append(
                (
                    "Trend insight does not contain a "
                    "grounded numerical result."
                )
            )

        if contains_unsupported_trend_claim(
            text
        ):

            reasons.append(
                (
                    "Trend insight introduces an unsupported "
                    "rate or steadiness claim."
                )
            )

    # ========================================================
    # ROOM COMPARISON
    # ========================================================

    elif query_type == "compare_rooms":

        room_results = (
            evidence.get(
                "room_results",
                [],
            )
            or
            []
        )

        for row in room_results:

            room = (
                row.get(
                    "display_name"
                )
            )

            value = (
                row.get(
                    "average_value"
                )
            )

            if (
                room
                and
                not room_in_text(
                    room,
                    text,
                )
            ):

                reasons.append(
                    (
                        f"Comparison insight omits room "
                        f"'{room}'."
                    )
                )

            if (
                value is not None
                and
                not number_in_text(
                    value,
                    text,
                )
            ):

                reasons.append(
                    (
                        f"Comparison insight omits the "
                        f"grounded value for '{room}'."
                    )
                )

    # ========================================================
    # ROOM RANKING
    # ========================================================

    elif query_type in {
        "rank_rooms",
        "rank_rooms_latest",
    }:

        first = (
            first_room_result(
                evidence
            )
        )

        if first is None:

            reasons.append(
                "Ranking evidence does not contain a first row."
            )

        else:

            first_room = str(
                first.get(
                    "display_name",
                    "",
                )
            ).strip()

            if query_type == "rank_rooms_latest":

                first_value = (
                    first.get(
                        "latest_value"
                    )
                )

            else:

                first_value = (
                    first.get(
                        "average_value"
                    )
                )

            if (
                first_room
                and
                not room_in_text(
                    first_room,
                    text,
                )
            ):

                reasons.append(
                    (
                        "Ranking insight does not mention "
                        "the first-ranked room."
                    )
                )

            if (
                first_value is not None
                and
                not number_in_text(
                    first_value,
                    text,
                )
            ):

                reasons.append(
                    (
                        "Ranking insight does not mention "
                        "the first-ranked value."
                    )
                )

            if first_room:

                ascending = bool(
                    evidence.get(
                        "ascending",
                        False,
                    )
                )

                if ranking_direction_problem(
                    text=
                        text,

                    first_room=
                        first_room,

                    ascending=
                        ascending,
                ):

                    reasons.append(
                        (
                            "Ranking insight reverses the "
                            "requested ranking direction."
                        )
                    )

        # ----------------------------------------------------
        # Latest/current result
        # ----------------------------------------------------

        if query_type == "rank_rooms_latest":

            historical_context = bool(
                re.search(
                    r"\blatest\s+available\b|"
                    r"\bhistorical\b|"
                    r"\bnot\s+live\b|"
                    r"\bnot\s+real[- ]?time\b",
                    text,
                    flags=re.IGNORECASE,
                )
            )

            if not historical_context:

                reasons.append(
                    (
                        "Latest-result insight does not clearly "
                        "distinguish historical data from "
                        "live real-time data."
                    )
                )

    # ========================================================
    # THRESHOLD
    # ========================================================

    elif query_type == "threshold":

        room_results = (
            evidence.get(
                "room_results",
                [],
            )
            or
            []
        )

        # Current T06 is no-data and exits earlier.
        # This supports future threshold tests with data.
        if room_results:

            first = room_results[0]

            first_room = (
                first.get(
                    "display_name"
                )
            )

            first_value = (
                first.get(
                    "average_value"
                )
            )

            if (
                first_room
                and
                not room_in_text(
                    first_room,
                    text,
                )
            ):

                reasons.append(
                    (
                        "Threshold insight does not mention "
                        "a grounded matching room."
                    )
                )

            if (
                first_value is not None
                and
                not number_in_text(
                    first_value,
                    text,
                )
            ):

                reasons.append(
                    (
                        "Threshold insight does not mention "
                        "a grounded matching value."
                    )
                )

    # ========================================================
    # DISTRIBUTION
    # ========================================================

    elif query_type == "distribution":

        ranked_rooms = (
            evidence.get(
                "ranked_rooms",
                [],
            )
            or
            []
        )

        statistics = (
            evidence.get(
                "statistics",
                {},
            )
            or
            {}
        )

        # ----------------------------------------------------
        # Ranked distribution
        # ----------------------------------------------------

        if ranked_rooms:

            first = ranked_rooms[0]

            first_room = (
                first.get(
                    "room"
                )
            )

            ranking_value = (
                first.get(
                    "ranking_value"
                )
            )

            if (
                first_room
                and
                not room_in_text(
                    first_room,
                    text,
                )
            ):

                reasons.append(
                    (
                        "Ranked distribution insight does not "
                        "mention the first-ranked room."
                    )
                )

            if (
                ranking_value is not None
                and
                not number_in_text(
                    ranking_value,
                    text,
                )
            ):

                reasons.append(
                    (
                        "Ranked distribution insight does not "
                        "mention the first-ranked room's "
                        "ranking value."
                    )
                )

        # ----------------------------------------------------
        # Require at least one grounded descriptive statistic
        # ----------------------------------------------------

        statistical_values = [
            statistics.get(
                "minimum"
            ),
            statistics.get(
                "maximum"
            ),
            statistics.get(
                "mean"
            ),
            statistics.get(
                "median"
            ),
        ]

        grounded_stat = any(
            (
                value is not None
                and
                number_in_text(
                    value,
                    text,
                )
            )
            for value in statistical_values
        )

        if not grounded_stat:

            reasons.append(
                (
                    "Distribution insight does not include "
                    "a grounded descriptive statistic."
                )
            )

        # ----------------------------------------------------
        # Irrelevant latest warning
        # ----------------------------------------------------

        if (
            evidence.get(
                "query_type"
            )
            ==
            "distribution"
            and
            re.search(
                r"\bright\s+now\b|"
                r"\bnot\s+live\b|"
                r"\blatest\s+available\s+historical\b",
                str(
                    state.get(
                        "insight_caution",
                        "",
                    )
                    or
                    ""
                ),
                flags=re.IGNORECASE,
            )
        ):

            reasons.append(
                (
                    "Distribution insight contains an "
                    "irrelevant latest/live-data caution."
                )
            )

    # ========================================================
    # RELATIONSHIP
    # ========================================================

    elif query_type == "relationship":

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

        pearson_r = (
            evidence.get(
                "pearson_r"
            )
        )

        if (
            metric_x
            and
            normalise_text(
                metric_x
            )
            not in
            text.casefold()
        ):

            reasons.append(
                (
                    "Relationship insight does not mention "
                    "the x metric."
                )
            )

        if (
            metric_y
            and
            normalise_text(
                metric_y
            )
            not in
            text.casefold()
        ):

            reasons.append(
                (
                    "Relationship insight does not mention "
                    "the y metric."
                )
            )

        if (
            pearson_r is not None
            and
            not number_in_text(
                pearson_r,
                text,
                tolerance=
                    0.005,
            )
        ):

            reasons.append(
                (
                    "Relationship insight does not include "
                    "the grounded Pearson correlation."
                )
            )

        if contains_causal_language(
            text
        ):

            reasons.append(
                (
                    "Relationship insight contains unsupported "
                    "causal or predictive language."
                )
            )

    # ========================================================
    # RESULT
    # ========================================================

    unique_reasons: list[str] = []

    for reason in reasons:

        if reason not in unique_reasons:

            unique_reasons.append(
                reason
            )

    return (
        len(
            unique_reasons
        )
        ==
        0,
        unique_reasons,
    )


# ============================================================
# RUN ONE TEST
# ============================================================

def run_test(
    system: HybridSingleAgent,
    test_case: dict[str, Any],
) -> dict[str, Any]:

    prompt = (
        test_case[
            "prompt"
        ]
    )

    print(
        "\n"
        + "=" * 80
    )

    print(
        f"{test_case['id']}: "
        f"{prompt}"
    )

    print(
        "=" * 80
    )

    start_time = (
        time.perf_counter()
    )

    try:

        state = (
            system.run(
                prompt
            )
        )

        elapsed = (
            time.perf_counter()
            -
            start_time
        )

        # ====================================================
        # ACTUAL VALUES
        # ====================================================

        actual_intent = (
            state.get(
                "intent"
            )
        )

        actual_query_type = (
            state.get(
                "query_type"
            )
        )

        selected_chart = (
            state.get(
                "chart_type"
            )
        )

        generated_chart = (
            state.get(
                "final_chart_type"
            )
        )

        actual_mapping = (
            state.get(
                "mapping",
                {}
            )
            or
            {}
        )

        result_valid = bool_value(
            state.get(
                "result_valid"
            )
        )

        result_has_data = bool_value(
            state.get(
                "result_has_data"
            )
        )

        execution_success = bool_value(
            state.get(
                "execution_success"
            )
        )

        visualization_success = bool_value(
            state.get(
                "visualization_success"
            )
        )

        chart_created = bool_value(
            state.get(
                "chart_created"
            )
        )

        insight_success = bool_value(
            state.get(
                "insight_generation_success"
            )
        )

        # ====================================================
        # EXPECTED VALUES
        # ====================================================

        expected_intent = (
            test_case[
                "expected_intent"
            ]
        )

        expected_query_type = (
            test_case[
                "expected_query_type"
            ]
        )

        expected_chart = (
            test_case[
                "expected_chart"
            ]
        )

        expected_valid = (
            test_case[
                "expected_valid"
            ]
        )

        expected_has_data = (
            test_case[
                "expected_has_data"
            ]
        )

        expected_chart_created = (
            test_case[
                "expected_chart_created"
            ]
        )

        expected_mapping = (
            test_case[
                "expected_mapping"
            ]
        )

        # ====================================================
        # COMPONENT CHECKS
        # ====================================================

        intent_correct = (
            actual_intent
            ==
            expected_intent
        )

        query_type_correct = (
            actual_query_type
            ==
            expected_query_type
        )

        chart_correct = (
            selected_chart
            ==
            expected_chart
        )

        validation_correct = (
            result_valid
            ==
            expected_valid
        )

        has_data_correct = (
            result_has_data
            ==
            expected_has_data
        )

        chart_created_correct = (
            chart_created
            ==
            expected_chart_created
        )

        mapping_correct, mapping_reasons = (
            mapping_matches(
                actual_mapping=
                    actual_mapping,

                expected_mapping=
                    expected_mapping,
            )
        )

        # ----------------------------------------------------
        # Visualization correctness
        # ----------------------------------------------------

        visualization_correct = (
            visualization_success
            and
            chart_created_correct
        )

        # ====================================================
        # FUNCTIONAL PIPELINE PASS
        #
        # Does NOT include insight reliability.
        # ====================================================

        functional_pass = all(
            [
                intent_correct,
                mapping_correct,
                query_type_correct,
                chart_correct,
                validation_correct,
                has_data_correct,
                execution_success,
                visualization_correct,
                insight_success,
            ]
        )

        # ====================================================
        # INSIGHT RELIABILITY
        #
        # Scores the generated response.
        # Does NOT repair it.
        # ====================================================

        insight_reliable, reliability_reasons = (
            evaluate_insight_reliability(
                state=
                    state,

                test_case=
                    test_case,
            )
        )

        # ====================================================
        # RELIABLE END-TO-END PASS
        # ====================================================

        reliable_end_to_end_pass = (
            functional_pass
            and
            insight_reliable
        )

        # ====================================================
        # PRINT
        # ====================================================

        print(
            f"\nIntent: "
            f"{actual_intent}"
        )

        print(
            f"Mapping: "
            f"{actual_mapping}"
        )

        print(
            f"Query type: "
            f"{actual_query_type}"
        )

        print(
            f"Selected chart: "
            f"{selected_chart}"
        )

        print(
            f"Generated chart: "
            f"{generated_chart}"
        )

        print(
            f"Result valid: "
            f"{result_valid}"
        )

        print(
            f"Has data: "
            f"{result_has_data}"
        )

        print(
            f"Rows: "
            f"{state.get('sql_row_count', 0)}"
        )

        print(
            f"Chart created: "
            f"{chart_created}"
        )

        print(
            f"Insight generated by: "
            f"{state.get('insight_generated_by')}"
        )

        print(
            f"Latency: "
            f"{elapsed:.3f}s"
        )

        # ----------------------------------------------------
        # Insight output
        # ----------------------------------------------------

        print(
            "\n[BASELINE INSIGHT]"
        )

        print(
            "Summary:"
        )

        print(
            state.get(
                "insight_summary",
                "",
            )
        )

        key_points = (
            state.get(
                "insight_key_points",
                [],
            )
            or
            []
        )

        if key_points:

            print(
                "Key points:"
            )

            for point in key_points:

                print(
                    f"- {point}"
                )

        caution = (
            state.get(
                "insight_caution",
                ""
            )
            or
            ""
        )

        if caution:

            print(
                f"Caution: "
                f"{caution}"
            )

        # ----------------------------------------------------
        # Reliability result
        # ----------------------------------------------------

        print(
            f"\nInsight reliable: "
            f"{insight_reliable}"
        )

        if reliability_reasons:

            print(
                "Reliability issues:"
            )

            for reason in reliability_reasons:

                print(
                    f"- {reason}"
                )

        # ----------------------------------------------------
        # Mapping problems
        # ----------------------------------------------------

        if mapping_reasons:

            print(
                "\nMapping issues:"
            )

            for reason in mapping_reasons:

                print(
                    f"- {reason}"
                )

        # ----------------------------------------------------
        # Pass status
        # ----------------------------------------------------

        if functional_pass:

            print(
                "\n✅ FUNCTIONAL PIPELINE PASSED"
            )

        else:

            print(
                "\n❌ FUNCTIONAL PIPELINE FAILED"
            )

        if reliable_end_to_end_pass:

            print(
                "✅ RELIABLE END-TO-END PASSED"
            )

        else:

            print(
                "⚠ RELIABLE END-TO-END FAILED"
            )

        # ====================================================
        # RETURN
        # ====================================================

        return {
            "test_id":
                test_case[
                    "id"
                ],

            "prompt":
                prompt,

            # ------------------------------------------------
            # Intent
            # ------------------------------------------------

            "expected_intent":
                expected_intent,

            "actual_intent":
                actual_intent,

            "intent_correct":
                intent_correct,

            # ------------------------------------------------
            # Mapping
            # ------------------------------------------------

            "mapping_correct":
                mapping_correct,

            "mapping_issues":
                " | ".join(
                    mapping_reasons
                ),

            # ------------------------------------------------
            # Query type
            # ------------------------------------------------

            "expected_query_type":
                expected_query_type,

            "actual_query_type":
                actual_query_type,

            "query_type_correct":
                query_type_correct,

            # ------------------------------------------------
            # Chart
            # ------------------------------------------------

            "expected_chart":
                expected_chart,

            "selected_chart":
                selected_chart,

            "generated_chart":
                generated_chart,

            "chart_correct":
                chart_correct,

            # ------------------------------------------------
            # Result validation
            # ------------------------------------------------

            "expected_valid":
                expected_valid,

            "result_valid":
                result_valid,

            "validation_correct":
                validation_correct,

            "expected_has_data":
                expected_has_data,

            "actual_has_data":
                result_has_data,

            "has_data_correct":
                has_data_correct,

            "result_status":
                state.get(
                    "result_status"
                ),

            # ------------------------------------------------
            # SQL
            # ------------------------------------------------

            "execution_success":
                execution_success,

            "row_count":
                state.get(
                    "sql_row_count",
                    0,
                ),

            "stored_row_count":
                state.get(
                    "sql_stored_row_count",
                    0,
                ),

            "result_truncated":
                bool_value(
                    state.get(
                        "sql_result_truncated"
                    )
                ),

            # ------------------------------------------------
            # Visualization
            # ------------------------------------------------

            "visualization_success":
                visualization_success,

            "expected_chart_created":
                expected_chart_created,

            "chart_created":
                chart_created,

            "chart_created_correct":
                chart_created_correct,

            "visualization_correct":
                visualization_correct,

            # ------------------------------------------------
            # Insight
            # ------------------------------------------------

            "insight_success":
                insight_success,

            "insight_generated_by":
                state.get(
                    "insight_generated_by"
                ),

            "insight_summary":
                state.get(
                    "insight_summary",
                    "",
                ),

            "insight_key_points":
                " | ".join(
                    state.get(
                        "insight_key_points",
                        [],
                    )
                    or
                    []
                ),

            "insight_caution":
                state.get(
                    "insight_caution",
                    "",
                ),

            "insight_reliable":
                insight_reliable,

            "insight_reliability_issues":
                " | ".join(
                    reliability_reasons
                ),

            # ------------------------------------------------
            # Parser fallback
            # ------------------------------------------------

            "parser_source":
                state.get(
                    "parser_source"
                ),

            # ------------------------------------------------
            # Performance
            # ------------------------------------------------

            "latency_seconds":
                round(
                    elapsed,
                    3,
                ),

            # ------------------------------------------------
            # Overall
            # ------------------------------------------------

            "functional_pass":
                functional_pass,

            "reliable_end_to_end_pass":
                reliable_end_to_end_pass,

            # ------------------------------------------------
            # Diagnostics
            # ------------------------------------------------

            "warnings":
                " | ".join(
                    state.get(
                        "warnings",
                        [],
                    )
                    or
                    []
                ),

            "errors":
                " | ".join(
                    state.get(
                        "errors",
                        [],
                    )
                    or
                    []
                ),
        }

    # ========================================================
    # CRASH
    # ========================================================

    except Exception as error:

        elapsed = (
            time.perf_counter()
            -
            start_time
        )

        print(
            "\n❌ TEST CRASHED"
        )

        print(
            f"{type(error).__name__}: "
            f"{error}"
        )

        return {
            "test_id":
                test_case[
                    "id"
                ],

            "prompt":
                prompt,

            "expected_intent":
                test_case[
                    "expected_intent"
                ],

            "actual_intent":
                None,

            "intent_correct":
                False,

            "mapping_correct":
                False,

            "mapping_issues":
                "Test crashed.",

            "expected_query_type":
                test_case[
                    "expected_query_type"
                ],

            "actual_query_type":
                None,

            "query_type_correct":
                False,

            "expected_chart":
                test_case[
                    "expected_chart"
                ],

            "selected_chart":
                None,

            "generated_chart":
                None,

            "chart_correct":
                False,

            "expected_valid":
                test_case[
                    "expected_valid"
                ],

            "result_valid":
                False,

            "validation_correct":
                False,

            "expected_has_data":
                test_case[
                    "expected_has_data"
                ],

            "actual_has_data":
                False,

            "has_data_correct":
                False,

            "result_status":
                "crashed",

            "execution_success":
                False,

            "row_count":
                0,

            "stored_row_count":
                0,

            "result_truncated":
                False,

            "visualization_success":
                False,

            "expected_chart_created":
                test_case[
                    "expected_chart_created"
                ],

            "chart_created":
                False,

            "chart_created_correct":
                False,

            "visualization_correct":
                False,

            "insight_success":
                False,

            "insight_generated_by":
                None,

            "insight_summary":
                "",

            "insight_key_points":
                "",

            "insight_caution":
                "",

            "insight_reliable":
                False,

            "insight_reliability_issues":
                (
                    f"Test crashed: "
                    f"{type(error).__name__}: "
                    f"{error}"
                ),

            "parser_source":
                None,

            "latency_seconds":
                round(
                    elapsed,
                    3,
                ),

            "functional_pass":
                False,

            "reliable_end_to_end_pass":
                False,

            "warnings":
                "",

            "errors":
                (
                    f"{type(error).__name__}: "
                    f"{error}"
                ),
        }


# ============================================================
# SAVE CSV
# ============================================================

def save_results(
    results: list[dict[str, Any]],
) -> None:

    if not results:

        return

    fieldnames = list(
        results[0].keys()
    )

    with OUTPUT_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=
                fieldnames,
        )

        writer.writeheader()

        writer.writerows(
            results
        )


# ============================================================
# PERCENTAGE
# ============================================================

def percentage(
    count: int,
    total: int,
) -> float:

    if total == 0:

        return 0.0

    return (
        count
        /
        total
        *
        100
    )


# ============================================================
# SUMMARY
# ============================================================

def print_summary(
    results: list[dict[str, Any]],
) -> None:

    total = len(
        results
    )

    if total == 0:

        return

    # ========================================================
    # COUNTS
    # ========================================================

    functional_passed = sum(
        1
        for result in results
        if result[
            "functional_pass"
        ]
    )

    reliable_passed = sum(
        1
        for result in results
        if result[
            "reliable_end_to_end_pass"
        ]
    )

    intent_correct = sum(
        1
        for result in results
        if result[
            "intent_correct"
        ]
    )

    mapping_correct = sum(
        1
        for result in results
        if result[
            "mapping_correct"
        ]
    )

    query_type_correct = sum(
        1
        for result in results
        if result[
            "query_type_correct"
        ]
    )

    chart_correct = sum(
        1
        for result in results
        if result[
            "chart_correct"
        ]
    )

    sql_success = sum(
        1
        for result in results
        if result[
            "execution_success"
        ]
    )

    visualization_correct = sum(
        1
        for result in results
        if result[
            "visualization_correct"
        ]
    )

    insight_success = sum(
        1
        for result in results
        if result[
            "insight_success"
        ]
    )

    insight_reliable = sum(
        1
        for result in results
        if result[
            "insight_reliable"
        ]
    )

    deterministic_parser_fallbacks = sum(
        1
        for result in results
        if result.get(
            "parser_source"
        )
        ==
        "deterministic_parser_fallback"
    )

    # ========================================================
    # LATENCY
    # ========================================================

    latencies = [
        float(
            result[
                "latency_seconds"
            ]
        )
        for result in results
    ]

    average_latency = (
        sum(
            latencies
        )
        /
        len(
            latencies
        )
    )

    sorted_latencies = sorted(
        latencies
    )

    middle = (
        len(
            sorted_latencies
        )
        //
        2
    )

    if len(
        sorted_latencies
    ) % 2 == 0:

        median_latency = (
            sorted_latencies[
                middle - 1
            ]
            +
            sorted_latencies[
                middle
            ]
        ) / 2

    else:

        median_latency = (
            sorted_latencies[
                middle
            ]
        )

    # ========================================================
    # PRINT
    # ========================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "HYBRID SINGLE-AGENT REGRESSION SUMMARY"
    )

    print(
        "=" * 80
    )

    print(
        f"\nTests: "
        f"{total}"
    )

    print(
        f"Functional passes: "
        f"{functional_passed}"
    )

    print(
        f"Functional failures: "
        f"{total - functional_passed}"
    )

    print(
        f"Functional pipeline success rate: "
        f"{percentage(functional_passed, total):.1f}%"
    )

    print(
        f"\nReliable end-to-end passes: "
        f"{reliable_passed}"
    )

    print(
        f"Reliable end-to-end failures: "
        f"{total - reliable_passed}"
    )

    print(
        f"Reliable end-to-end success rate: "
        f"{percentage(reliable_passed, total):.1f}%"
    )

    # --------------------------------------------------------
    # COMPONENTS
    # --------------------------------------------------------

    print(
        "\nCOMPONENT RESULTS"
    )

    print(
        "-" * 80
    )

    print(
        f"Intent accuracy: "
        f"{intent_correct}/{total} "
        f"({percentage(intent_correct, total):.1f}%)"
    )

    print(
        f"Mapping accuracy: "
        f"{mapping_correct}/{total} "
        f"({percentage(mapping_correct, total):.1f}%)"
    )

    print(
        f"Query-type accuracy: "
        f"{query_type_correct}/{total} "
        f"({percentage(query_type_correct, total):.1f}%)"
    )

    print(
        f"Chart-selection accuracy: "
        f"{chart_correct}/{total} "
        f"({percentage(chart_correct, total):.1f}%)"
    )

    print(
        f"SQL execution success: "
        f"{sql_success}/{total} "
        f"({percentage(sql_success, total):.1f}%)"
    )

    print(
        f"Visualization correctness: "
        f"{visualization_correct}/{total} "
        f"({percentage(visualization_correct, total):.1f}%)"
    )

    print(
        f"Insight generation success: "
        f"{insight_success}/{total} "
        f"({percentage(insight_success, total):.1f}%)"
    )

    print(
        f"Insight reliability: "
        f"{insight_reliable}/{total} "
        f"({percentage(insight_reliable, total):.1f}%)"
    )

    # --------------------------------------------------------
    # PARSER FALLBACK
    # --------------------------------------------------------

    print(
        f"\nDeterministic parser fallback count: "
        f"{deterministic_parser_fallbacks}"
    )

    print(
        f"Deterministic parser fallback rate: "
        f"{percentage(deterministic_parser_fallbacks, total):.1f}%"
    )

    # --------------------------------------------------------
    # LATENCY
    # --------------------------------------------------------

    print(
        f"\nAverage latency: "
        f"{average_latency:.3f}s"
    )

    print(
        f"Median latency: "
        f"{median_latency:.3f}s"
    )

    # --------------------------------------------------------
    # UNRELIABLE INSIGHTS
    # --------------------------------------------------------

    unreliable_results = [
        result
        for result in results
        if not result[
            "insight_reliable"
        ]
    ]

    if unreliable_results:

        print(
            "\nUNRELIABLE INSIGHT CASES"
        )

        print(
            "-" * 80
        )

        for result in unreliable_results:

            print(
                f"{result['test_id']}: "
                f"{result['insight_reliability_issues']}"
            )

    # --------------------------------------------------------
    # SAVE PATH
    # --------------------------------------------------------

    print(
        "\nSaved results:"
    )

    print(
        OUTPUT_PATH
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print(
        "=" * 80
    )

    print(
        "SMARTVIZ HYBRID SINGLE-AGENT REGRESSION TEST"
    )

    print(
        "=" * 80
    )

    # --------------------------------------------------------
    # Construct once.
    # --------------------------------------------------------

    system = (
        HybridSingleAgent()
    )

    results: list[
        dict[str, Any]
    ] = []

    # --------------------------------------------------------
    # Execute the same controlled T01-T12 suite.
    # --------------------------------------------------------

    for test_case in TEST_CASES:

        result = (
            run_test(
                system=
                    system,

                test_case=
                    test_case,
            )
        )

        results.append(
            result
        )

    # --------------------------------------------------------
    # Save detailed evidence.
    # --------------------------------------------------------

    save_results(
        results
    )

    # --------------------------------------------------------
    # Print aggregate metrics.
    # --------------------------------------------------------

    print_summary(
        results
    )


if __name__ == "__main__":

    main()