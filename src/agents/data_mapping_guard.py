from __future__ import annotations

import re
from typing import Any


# ============================================================
# NUMBER WORDS
# ============================================================

NUMBER_WORDS = {
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
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
}


# ============================================================
# TEXT NORMALISATION
# ============================================================

def _normalise_text(
    value: str,
) -> str:
    """
    Normalise text for deterministic matching.

    Example:
        "Seminar Room 3"
        -> "seminar room 3"
    """

    value = str(
        value
    ).casefold()

    value = re.sub(
        r"[^a-z0-9]+",
        " ",
        value,
    )

    value = re.sub(
        r"\s+",
        " ",
        value,
    )

    return value.strip()


# ============================================================
# SINGLE METRIC EXTRACTION
# ============================================================

def _extract_metric(
    query: str,
) -> str | None:
    """
    Extract one supported SmartViz metric.

    Used for all single-metric analytical intents.
    """

    q = _normalise_text(
        query
    )

    # --------------------------------------------------------
    # CO2
    # --------------------------------------------------------

    if re.search(
        r"\b(?:co2|carbon dioxide)\b",
        q,
    ):
        return "co2"

    # --------------------------------------------------------
    # TEMPERATURE
    # --------------------------------------------------------

    if re.search(
        r"\b(?:"
        r"temperature|"
        r"temp|"
        r"hottest|"
        r"warmest|"
        r"coldest"
        r")\b",
        q,
    ):
        return "temp"

    # --------------------------------------------------------
    # HUMIDITY
    # --------------------------------------------------------

    if re.search(
        r"\b(?:humidity|humid)\b",
        q,
    ):
        return "humidity"

    # --------------------------------------------------------
    # OCCUPANCY
    # --------------------------------------------------------

    if re.search(
        r"\b(?:"
        r"occupancy|"
        r"occupied|"
        r"busiest|"
        r"busy|"
        r"utilised|"
        r"utilized|"
        r"utilisation|"
        r"utilization|"
        r"underutilised|"
        r"underutilized"
        r")\b",
        q,
    ):
        return "occupancy"

    return None


# ============================================================
# MULTI-METRIC EXTRACTION
# ============================================================

def _extract_metrics_in_order(
    query: str,
) -> list[str]:
    """
    Extract supported metrics in the order in which
    they appear in the user's request.

    Used by relationship/scatter analysis.

    Example:
        "relationship between occupancy and CO2"

        -> ["occupancy", "co2"]
    """

    q = query.casefold()

    metric_patterns = {
        "co2": [
            r"\bco2\b",
            r"\bcarbon\s+dioxide\b",
        ],

        "temp": [
            r"\btemperature\b",
            r"\btemp\b",
        ],

        "humidity": [
            r"\bhumidity\b",
            r"\bhumid\b",
        ],

        "occupancy": [
            r"\boccupancy\b",
            r"\boccupied\b",
            r"\butilisation\b",
            r"\butilization\b",
            r"\butilised\b",
            r"\butilized\b",
        ],
    }

    matches: list[
        tuple[int, str]
    ] = []

    for (
        metric,
        patterns,
    ) in metric_patterns.items():

        earliest_position: int | None = None

        for pattern in patterns:

            match = re.search(
                pattern,
                q,
            )

            if match is None:
                continue

            position = (
                match.start()
            )

            if (
                earliest_position is None
                or position < earliest_position
            ):
                earliest_position = (
                    position
                )

        if earliest_position is not None:

            matches.append(
                (
                    earliest_position,
                    metric,
                )
            )

    # --------------------------------------------------------
    # Preserve order from user query
    # --------------------------------------------------------

    matches.sort(
        key=lambda item: item[0]
    )

    metrics: list[str] = []

    for _, metric in matches:

        if metric not in metrics:

            metrics.append(
                metric
            )

    return metrics


# ============================================================
# ROOM EXTRACTION
# ============================================================

def _extract_rooms(
    query: str,
    available_rooms: list[str],
) -> list[str]:
    """
    Deterministically identify room names.

    Supports:

    - exact room names
    - aliases such as:
        "hive" -> "The Hive"
    - numeric shorthand:
        "room 2 and 3"
        -> Seminar Room 2
        -> Seminar Room 3

    Rooms are only returned when they can be
    resolved safely against available_rooms.
    """

    q = _normalise_text(
        query
    )

    found: list[
        tuple[int, str]
    ] = []

    # ========================================================
    # 1. EXACT ROOM NAME MATCHING
    # ========================================================

    for room in available_rooms:

        room_text = (
            _normalise_text(
                room
            )
        )

        if not room_text:
            continue

        match = re.search(
            rf"\b{re.escape(room_text)}\b",
            q,
        )

        if match is not None:

            found.append(
                (
                    match.start(),
                    room,
                )
            )

    # ========================================================
    # 2. ALIAS MATCHING
    # ========================================================
    # Example:
    #
    # "The Hive"
    # -> alias "Hive"
    # ========================================================

    for room in available_rooms:

        room_text = (
            _normalise_text(
                room
            )
        )

        if not room_text.startswith(
            "the "
        ):
            continue

        alias = (
            room_text[4:]
            .strip()
        )

        if not alias:
            continue

        match = re.search(
            rf"\b{re.escape(alias)}\b",
            q,
        )

        if match is not None:

            found.append(
                (
                    match.start(),
                    room,
                )
            )

    # ========================================================
    # 3. NUMERIC ROOM SHORTHAND
    # ========================================================
    #
    # Example:
    #
    # "room 2 and 3"
    #
    # -> [2, 3]
    #
    # Then resolve each number only when exactly
    # one known room contains "room <number>".
    # ========================================================

    shorthand_matches = re.finditer(
        r"\brooms?\s+"
        r"(\d+(?:\s*(?:,|and)\s*\d+)*)",
        q,
    )

    for shorthand_match in shorthand_matches:

        number_text = (
            shorthand_match.group(
                1
            )
        )

        numbers = re.findall(
            r"\d+",
            number_text,
        )

        base_position = (
            shorthand_match.start()
        )

        for index, number in enumerate(
            numbers
        ):

            room_pattern = re.compile(
                rf"\broom\s+"
                rf"{re.escape(number)}\b"
            )

            candidates = []

            for room in available_rooms:

                room_normalised = (
                    _normalise_text(
                        room
                    )
                )

                if room_pattern.search(
                    room_normalised
                ):

                    candidates.append(
                        room
                    )

            # ------------------------------------------------
            # Only resolve if unambiguous
            # ------------------------------------------------

            if len(
                candidates
            ) == 1:

                found.append(
                    (
                        base_position
                        + index,

                        candidates[0],
                    )
                )

    # ========================================================
    # SORT BY QUERY POSITION + REMOVE DUPLICATES
    # ========================================================

    found.sort(
        key=lambda item: item[0]
    )

    final_rooms: list[str] = []

    for _, room in found:

        if room not in final_rooms:

            final_rooms.append(
                room
            )

    return final_rooms


# ============================================================
# TOP N EXTRACTION
# ============================================================

def _extract_top_n(
    query: str,
) -> int | None:
    """
    Extract ranking number.

    Examples:
        "top 5"
        "five hottest"
        "ten least occupied"
    """

    q = _normalise_text(
        query
    )

    # --------------------------------------------------------
    # Numeric forms
    # --------------------------------------------------------

    patterns = [
        r"\btop\s+(\d+)\b",
        r"\bbottom\s+(\d+)\b",
        r"\b(\d+)\s+"
        r"(?:highest|lowest|hottest|warmest|"
        r"coldest|busiest|least|worst|best)\b",
    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            q,
        )

        if match is not None:

            try:

                value = int(
                    match.group(
                        1
                    )
                )

                if value > 0:
                    return value

            except (
                TypeError,
                ValueError,
            ):
                pass

    # --------------------------------------------------------
    # Number words
    # --------------------------------------------------------

    for (
        word,
        number,
    ) in NUMBER_WORDS.items():

        patterns = [
            rf"\btop\s+{word}\b",
            rf"\bbottom\s+{word}\b",
            rf"\b{word}\s+"
            rf"(?:highest|lowest|hottest|warmest|"
            rf"coldest|busiest|least|worst|best)\b",
        ]

        if any(
            re.search(
                pattern,
                q,
            )
            for pattern
            in patterns
        ):

            return number

    # --------------------------------------------------------
    # Singular ranking query
    # --------------------------------------------------------

    singular_patterns = [
        r"\bwhich\s+room\b",
        r"\bhottest\s+room\b",
        r"\bcoldest\s+room\b",
        r"\bbusiest\s+room\b",
        r"\bhighest\s+room\b",
        r"\blowest\s+room\b",
    ]

    if any(
        re.search(
            pattern,
            q,
        )
        for pattern
        in singular_patterns
    ):

        return 1

    return None


# ============================================================
# THRESHOLD EXTRACTION
# ============================================================

def _extract_threshold(
    query: str,
) -> float | None:
    """
    Extract a numeric threshold only when the user
    explicitly provides threshold-style wording.

    This prevents the LLM from inventing thresholds.
    """

    q = query.casefold()

    patterns = [
        r"\babove\s+(-?\d+(?:\.\d+)?)",
        r"\bbelow\s+(-?\d+(?:\.\d+)?)",
        r"\bover\s+(-?\d+(?:\.\d+)?)",
        r"\bunder\s+(-?\d+(?:\.\d+)?)",
        r"\bgreater\s+than\s+"
        r"(-?\d+(?:\.\d+)?)",
        r"\bless\s+than\s+"
        r"(-?\d+(?:\.\d+)?)",
        r"\bat\s+least\s+"
        r"(-?\d+(?:\.\d+)?)",
        r"\bat\s+most\s+"
        r"(-?\d+(?:\.\d+)?)",
        r">=?\s*(-?\d+(?:\.\d+)?)",
        r"<=?\s*(-?\d+(?:\.\d+)?)",
    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            q,
        )

        if match is None:
            continue

        try:

            return float(
                match.group(
                    1
                )
            )

        except (
            TypeError,
            ValueError,
        ):
            continue

    return None


# ============================================================
# DATE EXTRACTION
# ============================================================

def _extract_dates(
    query: str,
) -> list[str]:
    """
    Extract explicit ISO dates only.

    Example:
        2025-05-01
    """

    return re.findall(
        r"\b"
        r"\d{4}-\d{2}-\d{2}"
        r"\b",
        query,
    )


# ============================================================
# CONTINUOUS EXTRACTION
# ============================================================

def _extract_continuous(
    query: str,
) -> bool:
    """
    Continuous analysis must only be enabled when
    explicitly requested.
    """

    q = _normalise_text(
        query
    )

    continuous_patterns = [
        r"\bcontinuous\b",
        r"\bcontinuously\b",
        r"\bcontinous\b",
        r"\bcontinually\b",
        r"\bpersistent\b",
        r"\bpersistently\b",
        r"\bsustained\b",
        r"\bconsecutive\b",
    ]

    return any(
        re.search(
            pattern,
            q,
        )
        for pattern
        in continuous_patterns
    )


# ============================================================
# MAIN DATA MAPPING GUARD
# ============================================================

def guard_mapping(
    user_query: str,
    intent: str,
    llm_mapping: dict[str, Any],
    available_rooms: list[str],
) -> dict[str, Any]:
    """
    Deterministically validate and correct Data Mapping
    Agent output.

    LLM output is treated as a proposal.

    Explicit information from the original user query
    takes priority.

    Relationship queries use:
        metric_x
        metric_y

    All other analytical intents use:
        metric_name
    """

    if not isinstance(
        user_query,
        str,
    ):

        raise TypeError(
            "user_query must be a string."
        )

    if not isinstance(
        llm_mapping,
        dict,
    ):

        raise TypeError(
            "llm_mapping must be a dictionary."
        )

    q = (
        user_query
        .casefold()
        .strip()
    )

    final = dict(
        llm_mapping
    )

    # ========================================================
    # 1. METRIC HANDLING
    # ========================================================

    if intent == "relationship":

        # ----------------------------------------------------
        # Relationship requires TWO metrics.
        # Never trust an invented second metric.
        # ----------------------------------------------------

        metrics = (
            _extract_metrics_in_order(
                user_query
            )
        )

        final[
            "metric_name"
        ] = None

        if len(
            metrics
        ) >= 1:

            final[
                "metric_x"
            ] = metrics[0]

        else:

            final[
                "metric_x"
            ] = None

        if len(
            metrics
        ) >= 2:

            final[
                "metric_y"
            ] = metrics[1]

        else:

            final[
                "metric_y"
            ] = None

    else:

        # ----------------------------------------------------
        # Single-metric intents
        # ----------------------------------------------------

        explicit_metric = (
            _extract_metric(
                user_query
            )
        )

        if explicit_metric is not None:

            final[
                "metric_name"
            ] = explicit_metric

        final[
            "metric_x"
        ] = None

        final[
            "metric_y"
        ] = None

    # ========================================================
    # 2. ROOM HANDLING
    # ========================================================

    explicit_rooms = (
        _extract_rooms(
            user_query,
            available_rooms,
        )
    )

    # --------------------------------------------------------
    # ROOM TREND
    # --------------------------------------------------------

    if intent == "room_trend":

        if len(
            explicit_rooms
        ) == 1:

            final[
                "room"
            ] = explicit_rooms[0]

            final[
                "room_names"
            ] = [
                explicit_rooms[0]
            ]

        else:

            final[
                "room"
            ] = None

            final[
                "room_names"
            ] = explicit_rooms

    # --------------------------------------------------------
    # ROOM COMPARISON
    # --------------------------------------------------------

    elif intent == "compare_rooms":

        final[
            "room"
        ] = None

        final[
            "room_names"
        ] = explicit_rooms

    # --------------------------------------------------------
    # DISTRIBUTION
    # --------------------------------------------------------

    elif intent == "distribution":

        # Example:
        #
        # "distribution across all rooms"
        # -> room = None
        #
        # "box plot for Seminar Room 3"
        # -> room = Seminar Room 3

        if len(
            explicit_rooms
        ) == 1:

            final[
                "room"
            ] = explicit_rooms[0]

            final[
                "room_names"
            ] = [
                explicit_rooms[0]
            ]

        else:

            final[
                "room"
            ] = None

            final[
                "room_names"
            ] = []

    # --------------------------------------------------------
    # RELATIONSHIP / SCATTER
    # --------------------------------------------------------

    elif intent == "relationship":

        # Relationship may apply to:
        #
        # all rooms
        #
        # OR
        #
        # one explicitly requested room.

        if len(
            explicit_rooms
        ) == 1:

            final[
                "room"
            ] = explicit_rooms[0]

            final[
                "room_names"
            ] = [
                explicit_rooms[0]
            ]

        else:

            final[
                "room"
            ] = None

            final[
                "room_names"
            ] = []

    # --------------------------------------------------------
    # RANKING + THRESHOLD
    # --------------------------------------------------------

    elif intent in {
        "rank_rooms",
        "threshold",
    }:

        final[
            "room"
        ] = None

        final[
            "room_names"
        ] = []

    # ========================================================
    # 3. AGGREGATION
    # ========================================================
    #
    # IMPORTANT:
    #
    # Use word boundaries.
    #
    # Otherwise:
    #
    # Seminar
    #   contains "min"
    #
    # and incorrectly becomes aggregation=min.
    # ========================================================

    if re.search(
        r"\b(?:average|avg|mean)\b",
        q,
    ):

        final[
            "aggregation"
        ] = "mean"

    elif re.search(
        r"\b(?:maximum|max)\b",
        q,
    ):

        final[
            "aggregation"
        ] = "max"

    elif re.search(
        r"\b(?:minimum|min)\b",
        q,
    ):

        final[
            "aggregation"
        ] = "min"

    elif re.search(
        r"\b(?:total|sum)\b",
        q,
    ):

        final[
            "aggregation"
        ] = "sum"

    else:

        final[
            "aggregation"
        ] = "mean"

    # ========================================================
    # 4. FREQUENCY
    # ========================================================

    if re.search(
        r"\b(?:daily|per day|each day)\b",
        q,
    ):

        final[
            "frequency"
        ] = "daily"

    else:

        final[
            "frequency"
        ] = "hourly"

    # ========================================================
    # 5. TOP N
    # ========================================================

    explicit_top_n = (
        _extract_top_n(
            user_query
        )
    )

    if explicit_top_n is not None:

        final[
            "top_n"
        ] = explicit_top_n

    elif intent == "rank_rooms":

        final[
            "top_n"
        ] = 10

    else:

        # Keep the state structurally consistent.
        try:

            current_top_n = int(
                final.get(
                    "top_n",
                    10,
                )
            )

            if current_top_n < 1:
                current_top_n = 10

        except (
            TypeError,
            ValueError,
        ):

            current_top_n = 10

        final[
            "top_n"
        ] = current_top_n

    # ========================================================
    # 6. RANKING DIRECTION
    # ========================================================

    ascending_patterns = [
        r"\bleast\b",
        r"\blowest\b",
        r"\bcoldest\b",
        r"\bbottom\b",
        r"\bless\s+utilised\b",
        r"\bless\s+utilized\b",
        r"\bleast\s+occupied\b",
        r"\bunderutilised\b",
        r"\bunderutilized\b",
    ]

    descending_patterns = [
        r"\bhighest\b",
        r"\bhottest\b",
        r"\bwarmest\b",
        r"\bbusiest\b",
        r"\btop\b",
        r"\bworst\b",
        r"\bmost\s+occupied\b",
    ]

    if any(
        re.search(
            pattern,
            q,
        )
        for pattern
        in ascending_patterns
    ):

        final[
            "ascending"
        ] = True

    elif any(
        re.search(
            pattern,
            q,
        )
        for pattern
        in descending_patterns
    ):

        final[
            "ascending"
        ] = False

    else:

        final[
            "ascending"
        ] = False

    # ========================================================
    # 7. THRESHOLD
    # ========================================================
    #
    # ALWAYS overwrite the LLM value.
    #
    # This prevents:
    #
    # user gives no threshold
    # but LLM invents threshold=21.
    # ========================================================

    final[
        "threshold"
    ] = _extract_threshold(
        user_query
    )

    # ========================================================
    # 8. DATE HANDLING
    # ========================================================
    #
    # ALWAYS reset dates from explicit user text.
    # ========================================================

    dates = (
        _extract_dates(
            user_query
        )
    )

    if len(
        dates
    ) >= 2:

        final[
            "start_date"
        ] = dates[0]

        final[
            "end_date"
        ] = dates[1]

    elif len(
        dates
    ) == 1:

        final[
            "start_date"
        ] = dates[0]

        final[
            "end_date"
        ] = None

    else:

        final[
            "start_date"
        ] = None

        final[
            "end_date"
        ] = None

    # ========================================================
    # 9. TIME SCOPE
    # ========================================================

    if len(
        dates
    ) >= 2:

        final[
            "time_scope"
        ] = "range"

    elif re.search(
        r"\b(?:"
        r"right now|"
        r"currently|"
        r"current|"
        r"latest|"
        r"now"
        r")\b",
        q,
    ):

        final[
            "time_scope"
        ] = "latest"

    else:

        final[
            "time_scope"
        ] = "all"

    # ========================================================
    # 10. CONTINUOUS
    # ========================================================
    #
    # ALWAYS overwrite LLM continuous value.
    # ========================================================

    final[
        "continuous"
    ] = _extract_continuous(
        user_query
    )

        
    # ========================================================
    # 11. FINAL RELATIONSHIP METRIC SAFETY CHECK
    # ========================================================
    #
    # Relationship queries MUST use metric_x + metric_y.
    # Re-extract them deterministically here so that no
    # earlier LLM or guard logic can accidentally overwrite them.
    # ========================================================

    if intent == "relationship":

        relationship_metrics = (
            _extract_metrics_in_order(
                user_query
            )
        )

        # Relationship does not use the old single metric field.
        final["metric_name"] = None

        if len(relationship_metrics) >= 1:
            final["metric_x"] = (
                relationship_metrics[0]
            )
        else:
            final["metric_x"] = None

        if len(relationship_metrics) >= 2:
            final["metric_y"] = (
                relationship_metrics[1]
            )
        else:
            final["metric_y"] = None

        # top_n has no analytical meaning for relationship queries.
        final["top_n"] = 10

    else:

        # All existing single-metric intents use metric_name,
        # not metric_x / metric_y.
        final["metric_x"] = None
        final["metric_y"] = None


    # ========================================================
    # RETURN FINAL GUARDED MAPPING
    # ========================================================

    return final


   