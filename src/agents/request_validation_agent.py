from __future__ import annotations

import re
from datetime import datetime
from typing import Any


class RequestValidationAgent:
    """
    Deterministic validation layer for SmartViz analytical requests.

    Responsibilities:
    - validate supported intents;
    - validate metrics;
    - validate room names;
    - validate date ranges;
    - validate ranking configuration;
    - validate threshold configuration;
    - validate distribution requests;
    - validate two-metric relationship requests;
    - prevent malformed requests reaching QueryBuilderAgent.

    This component does NOT use an LLM.
    """

    # ========================================================
    # SUPPORTED VALUES
    # ========================================================

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

    SUPPORTED_AGGREGATIONS = {
        "mean",
        "min",
        "max",
        "sum",
    }

    SUPPORTED_FREQUENCIES = {
        "hourly",
        "daily",
    }

    SUPPORTED_TIME_SCOPES = {
        "all",
        "latest",
        "range",
    }

    SUPPORTED_CHARTS = {
        "bar",
        "line",
        "histogram",
        "box",
        "scatter",
    }

    # ========================================================
    # INITIALISATION
    # ========================================================

    def __init__(
        self,
        available_rooms: list[str],
    ) -> None:

        self.available_rooms = {
            str(room).strip()
            for room in available_rooms
            if str(room).strip()
        }

    # ========================================================
    # DATE VALIDATION
    # ========================================================

    def _valid_date(
        self,
        value: Any,
    ) -> bool:

        if value is None:
            return True

        try:

            datetime.strptime(
                str(value),
                "%Y-%m-%d",
            )

            return True

        except (
            TypeError,
            ValueError,
        ):

            return False

    # ========================================================
    # THRESHOLD OPERATOR
    # ========================================================

    def _extract_threshold_operator(
        self,
        user_query: str,
    ) -> str | None:
        """
        Determine comparison direction for threshold queries.
        """

        q = (
            user_query
            .casefold()
            .strip()
        )

        # ----------------------------------------------------
        # GREATER THAN / ABOVE
        # ----------------------------------------------------

        greater_patterns = [
            r"\babove\b",
            r"\bover\b",
            r"\bgreater\s+than\b",
            r"\bat\s+least\b",
            r">=?",
        ]

        if any(
            re.search(
                pattern,
                q,
            )
            for pattern
            in greater_patterns
        ):

            return ">"

        # ----------------------------------------------------
        # LESS THAN / BELOW
        # ----------------------------------------------------

        lower_patterns = [
            r"\bbelow\b",
            r"\bunder\b",
            r"\bless\s+than\b",
            r"\bat\s+most\b",
            r"<=?",
        ]

        if any(
            re.search(
                pattern,
                q,
            )
            for pattern
            in lower_patterns
        ):

            return "<"

        return None

    # ========================================================
    # ROOM VALIDATION
    # ========================================================

    def _validate_room(
        self,
        room: str | None,
        errors: list[str],
    ) -> None:

        if room is None:
            return

        if room not in self.available_rooms:

            errors.append(
                f"Unsupported room: {room}"
            )

    def _validate_room_names(
        self,
        room_names: list[str],
        errors: list[str],
    ) -> None:

        for room in room_names:

            if room not in self.available_rooms:

                errors.append(
                    f"Unsupported room: {room}"
                )

    # ========================================================
    # MAIN VALIDATION
    # ========================================================

    def validate(
        self,
        user_query: str,
        intent: str,
        chart_type: str,
        mapping: dict[str, Any],
    ) -> dict[str, Any]:

        errors: list[str] = []
        warnings: list[str] = []

        requires_clarification = False

        # ====================================================
        # EXTRACT MAPPING VALUES
        # ====================================================

        metric_name = mapping.get(
            "metric_name"
        )

        metric_x = mapping.get(
            "metric_x"
        )

        metric_y = mapping.get(
            "metric_y"
        )

        room = mapping.get(
            "room"
        )

        room_names = list(
            mapping.get(
                "room_names",
                [],
            )
            or []
        )

        aggregation = mapping.get(
            "aggregation",
            "mean",
        )

        frequency = mapping.get(
            "frequency",
            "hourly",
        )

        start_date = mapping.get(
            "start_date"
        )

        end_date = mapping.get(
            "end_date"
        )

        top_n = mapping.get(
            "top_n",
            10,
        )

        ascending = bool(
            mapping.get(
                "ascending",
                False,
            )
        )

        threshold = mapping.get(
            "threshold"
        )

        time_scope = mapping.get(
            "time_scope",
            "all",
        )

        continuous = bool(
            mapping.get(
                "continuous",
                False,
            )
        )

        # ====================================================
        # 1. INTENT
        # ====================================================

        if intent not in self.SUPPORTED_INTENTS:

            errors.append(
                f"Unsupported intent: {intent}"
            )

        # ====================================================
        # 2. CHART
        # ====================================================

        if chart_type not in self.SUPPORTED_CHARTS:

            errors.append(
                f"Unsupported chart type: {chart_type}"
            )

        # ====================================================
        # 3. AGGREGATION
        # ====================================================

        if aggregation not in self.SUPPORTED_AGGREGATIONS:

            errors.append(
                "Unsupported aggregation: "
                f"{aggregation}"
            )

        # ====================================================
        # 4. FREQUENCY
        # ====================================================

        if frequency not in self.SUPPORTED_FREQUENCIES:

            errors.append(
                f"Unsupported frequency: {frequency}"
            )

        # ====================================================
        # 5. TIME SCOPE
        # ====================================================

        if time_scope not in self.SUPPORTED_TIME_SCOPES:

            errors.append(
                f"Unsupported time scope: {time_scope}"
            )

        # ====================================================
        # 6. DATES
        # ====================================================

        if not self._valid_date(
            start_date
        ):

            errors.append(
                "Invalid start date. "
                "Expected YYYY-MM-DD."
            )

        if not self._valid_date(
            end_date
        ):

            errors.append(
                "Invalid end date. "
                "Expected YYYY-MM-DD."
            )

        if (
            start_date is not None
            and end_date is not None
            and self._valid_date(start_date)
            and self._valid_date(end_date)
        ):

            start_dt = datetime.strptime(
                str(start_date),
                "%Y-%m-%d",
            )

            end_dt = datetime.strptime(
                str(end_date),
                "%Y-%m-%d",
            )

            if start_dt > end_dt:

                errors.append(
                    "Start date cannot be after end date."
                )

        if time_scope == "range":

            if (
                start_date is None
                or end_date is None
            ):

                errors.append(
                    "A date-range request requires "
                    "both start_date and end_date."
                )

                requires_clarification = True

        # ====================================================
        # 7. ROOMS
        # ====================================================

        self._validate_room(
            room=room,
            errors=errors,
        )

        self._validate_room_names(
            room_names=room_names,
            errors=errors,
        )

        # ====================================================
        # 8. RELATIONSHIP METRIC VALIDATION
        # ====================================================
        #
        # relationship:
        #
        # metric_name = None
        # metric_x = first metric
        # metric_y = second metric
        # ====================================================

        if intent == "relationship":

            # ------------------------------------------------
            # Single-metric field should not be used.
            # ------------------------------------------------

            metric_name = None

            # ------------------------------------------------
            # First metric
            # ------------------------------------------------

            if metric_x is None:

                errors.append(
                    "Relationship analysis requires "
                    "a first metric."
                )

                requires_clarification = True

            elif metric_x not in self.SUPPORTED_METRICS:

                errors.append(
                    "Unsupported relationship metric X: "
                    f"{metric_x}"
                )

            # ------------------------------------------------
            # Second metric
            # ------------------------------------------------

            if metric_y is None:

                errors.append(
                    "Relationship analysis requires "
                    "a second metric."
                )

                requires_clarification = True

            elif metric_y not in self.SUPPORTED_METRICS:

                errors.append(
                    "Unsupported relationship metric Y: "
                    f"{metric_y}"
                )

            # ------------------------------------------------
            # Must be TWO DIFFERENT metrics
            # ------------------------------------------------

            if (
                metric_x is not None
                and metric_y is not None
                and metric_x == metric_y
            ):

                errors.append(
                    "Relationship analysis requires "
                    "two different metrics."
                )

                requires_clarification = True

            # ------------------------------------------------
            # Scatter is the expected visualization.
            # ------------------------------------------------

            if chart_type != "scatter":

                warnings.append(
                    "Relationship analysis normally uses "
                    "a scatter plot."
                )

            # ------------------------------------------------
            # One point cannot establish a relationship.
            # ------------------------------------------------

            if time_scope == "latest":

                errors.append(
                    "Relationship analysis requires "
                    "multiple paired observations; "
                    "a latest-only request is not suitable."
                )

                requires_clarification = True

        # ====================================================
        # 9. SINGLE-METRIC VALIDATION
        # ====================================================

        else:

            metric_x = None
            metric_y = None

            if metric_name is None:

                errors.append(
                    "The request requires a supported metric."
                )

                requires_clarification = True

            elif metric_name not in self.SUPPORTED_METRICS:

                errors.append(
                    f"Unsupported metric: {metric_name}"
                )

        # ====================================================
        # 10. ROOM TREND
        # ====================================================

        if intent == "room_trend":

            if room is None:

                errors.append(
                    "A room trend request requires "
                    "one specific room."
                )

                requires_clarification = True

        # ====================================================
        # 11. COMPARE ROOMS
        # ====================================================

        if intent == "compare_rooms":

            if len(
                room_names
            ) < 2:

                errors.append(
                    "compare_rooms requires "
                    "at least two rooms."
                )

                requires_clarification = True

         # ====================================================
        # TOP N / DISTRIBUTION ROOM LIMIT
        # ====================================================

        distribution_top_n = None

        if intent == "rank_rooms":

            try:

                top_n = int(
                    top_n
                )

                if top_n < 1:

                    raise ValueError

            except (
                TypeError,
                ValueError,
            ):

                errors.append(
                    "Ranking requires a positive top_n value."
                )

                requires_clarification = True

                top_n = 10


        elif (
            intent == "distribution"
            and re.search(
                r"\b(?:top|bottom)\b",
                user_query.casefold(),
            )
        ):

            try:

                top_n = int(
                    top_n
                )

                if top_n < 1:

                    raise ValueError

                # This tells QueryBuilder that the user
                # explicitly requested a limited number of rooms.
                distribution_top_n = top_n

            except (
                TypeError,
                ValueError,
            ):

                errors.append(
                    "Distribution room ranking requires "
                    "a positive number of rooms."
                )

                requires_clarification = True

                top_n = 10


        else:

            top_n = 10        

        # ====================================================
        # 13. THRESHOLD
        # ====================================================

        threshold_operator = (
            self._extract_threshold_operator(
                user_query
            )
        )

        if intent == "threshold":

            if threshold is None:

                errors.append(
                    "A threshold request requires "
                    "a numeric threshold."
                )

                requires_clarification = True

            if threshold_operator is None:

                errors.append(
                    "A threshold request requires "
                    "an above/below comparison direction."
                )

                requires_clarification = True

        else:

            threshold_operator = None

        # ====================================================
        # 14. CONTINUOUS ANALYSIS
        # ====================================================

        if continuous:

            if threshold is None:

                errors.append(
                    "Continuous high-metric analysis "
                    "requires an approved numeric threshold."
                )

                requires_clarification = True

        # ====================================================
        # 15. MINIMUM RECORD REQUIREMENT
        # ====================================================

        if intent == "relationship":

            # Need at least two paired observations.
            minimum_records = 2

        elif time_scope == "latest":

            minimum_records = 1

        elif intent in {
            "rank_rooms",
            "threshold",
        }:

            minimum_records = 100

        else:

            minimum_records = 1

        # ====================================================
        # FINAL REQUEST
        # ====================================================

        request = {
            "intent":
                intent,

            "chart_type":
                chart_type,

            # ------------------------------------------------
            # Single metric
            # ------------------------------------------------

            "metric_name":
                metric_name,

            # ------------------------------------------------
            # Relationship metrics
            # ------------------------------------------------

            "metric_x":
                metric_x,

            "metric_y":
                metric_y,

            # ------------------------------------------------
            # Rooms
            # ------------------------------------------------

            "room":
                room,

            "room_names":
                room_names,

            # ------------------------------------------------
            # Query settings
            # ------------------------------------------------

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

            "threshold_operator":
                threshold_operator,

            "minimum_records":
                minimum_records,

            "distribution_top_n":
                distribution_top_n,    
        }

        # ====================================================
        # FINAL VALIDITY
        # ====================================================

        valid = (
            len(errors) == 0
        )

        return {
            "valid":
                valid,

            "errors":
                errors,

            "warnings":
                warnings,

            "requires_clarification":
                requires_clarification,

            "request":
                request,
        }