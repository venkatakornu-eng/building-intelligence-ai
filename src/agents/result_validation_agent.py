from __future__ import annotations

import math
from typing import Any


class ResultValidationAgent:
    """
    Deterministic validation of SmartViz SQL results.

    Responsibilities:
    - confirm SQL execution succeeded;
    - detect empty results;
    - validate expected result structure;
    - validate numeric values;
    - validate requested room coverage;
    - validate ranking result sizes;
    - validate top/bottom N distributions;
    - validate two-metric relationship results;
    - identify truncated results;
    - prevent invalid SQL results reaching visualization.

    This component does NOT use an LLM.
    """

    # ========================================================
    # EXPECTED COLUMNS BY QUERY TYPE
    # ========================================================

    EXPECTED_COLUMNS = {

        # ----------------------------------------------------
        # TREND
        # ----------------------------------------------------

        "room_trend": {
            "start_time",
            "value",
            "display_name",
        },

        # ----------------------------------------------------
        # DISTRIBUTION
        # ----------------------------------------------------

        "distribution": {
            "start_time",
            "value",
            "display_name",
        },

        # ----------------------------------------------------
        # COMPARISON
        # ----------------------------------------------------

        "compare_rooms": {
            "display_name",
            "average_value",
            "record_count",
        },

        # ----------------------------------------------------
        # RANKING
        # ----------------------------------------------------

        "rank_rooms": {
            "display_name",
            "average_value",
            "record_count",
        },

        # ----------------------------------------------------
        # LATEST RANKING
        # ----------------------------------------------------

        "rank_rooms_latest": {
            "display_name",
            "latest_value",
            "latest_start_time",
            "record_count",
        },

        # ----------------------------------------------------
        # THRESHOLD
        # ----------------------------------------------------

        "threshold": {
            "display_name",
            "average_value",
            "record_count",
        },

        # ----------------------------------------------------
        # RELATIONSHIP
        # ----------------------------------------------------

        "relationship": {
            "display_name",
            "start_time",
            "x_value",
            "y_value",
        },
    }

    # ========================================================
    # VALID NUMBER
    # ========================================================

    @staticmethod
    def _is_valid_number(
        value: Any,
    ) -> bool:

        if value is None:

            return False

        try:

            number = float(
                value
            )

        except (
            TypeError,
            ValueError,
        ):

            return False

        return math.isfinite(
            number
        )

    # ========================================================
    # EXPECTED COLUMN VALIDATION
    # ========================================================

    def _validate_columns(
        self,
        query_type: str,
        columns: list[str],
    ) -> list[str]:

        errors: list[str] = []

        expected = (
            self.EXPECTED_COLUMNS.get(
                query_type
            )
        )

        if expected is None:

            errors.append(
                "Unsupported result query type: "
                f"{query_type}"
            )

            return errors

        actual = set(
            columns
        )

        missing = (
            expected
            - actual
        )

        if missing:

            errors.append(
                "SQL result is missing expected "
                f"columns: {sorted(missing)}"
            )

        return errors

    # ========================================================
    # NUMERIC FIELDS
    # ========================================================

    def _numeric_fields_for_query(
        self,
        query_type: str,
    ) -> list[str]:

        # ----------------------------------------------------
        # RAW OBSERVATION QUERIES
        # ----------------------------------------------------

        if query_type in {
            "room_trend",
            "distribution",
        }:

            return [
                "value"
            ]

        # ----------------------------------------------------
        # LATEST RANKING
        # ----------------------------------------------------

        if query_type == "rank_rooms_latest":

            return [
                "latest_value"
            ]

        # ----------------------------------------------------
        # AGGREGATED ROOM RESULTS
        # ----------------------------------------------------

        if query_type in {
            "compare_rooms",
            "rank_rooms",
            "threshold",
        }:

            return [
                "average_value"
            ]

        # ----------------------------------------------------
        # TWO-METRIC RELATIONSHIP
        # ----------------------------------------------------

        if query_type == "relationship":

            return [
                "x_value",
                "y_value",
            ]

        return []

    # ========================================================
    # NUMERIC CONTENT VALIDATION
    # ========================================================

    def _validate_numeric_values(
        self,
        query_type: str,
        rows: list[dict[str, Any]],
    ) -> list[str]:

        errors: list[str] = []

        numeric_fields = (
            self._numeric_fields_for_query(
                query_type
            )
        )

        if not numeric_fields:

            return errors

        for numeric_field in numeric_fields:

            invalid_count = 0

            for row in rows:

                value = row.get(
                    numeric_field
                )

                if not self._is_valid_number(
                    value
                ):

                    invalid_count += 1

            if invalid_count > 0:

                errors.append(
                    f"{invalid_count} result row(s) "
                    f"contain an invalid "
                    f"{numeric_field} value."
                )

        return errors

    # ========================================================
    # COMPARE ROOM COVERAGE
    # ========================================================

    def _validate_compare_rooms(
        self,
        request: dict[str, Any],
        rows: list[dict[str, Any]],
    ) -> list[str]:

        warnings: list[str] = []

        requested_rooms = set(
            str(room)
            for room in request.get(
                "room_names",
                [],
            )
        )

        returned_rooms = {
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

        missing_rooms = (
            requested_rooms
            - returned_rooms
        )

        if missing_rooms:

            warnings.append(
                "No qualifying SQL result was "
                "returned for: "
                + ", ".join(
                    sorted(
                        missing_rooms
                    )
                )
            )

        return warnings

    # ========================================================
    # DISTRIBUTION VALIDATION
    # ========================================================

    def _validate_distribution(
        self,
        request: dict[str, Any],
        rows: list[dict[str, Any]],
    ) -> tuple[
        list[str],
        list[str],
    ]:
        """
        Validate standard, single-room and ranked
        top/bottom-N distribution results.
        """

        errors: list[str] = []
        warnings: list[str] = []

        returned_rooms = {
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

        # ----------------------------------------------------
        # SPECIFIC ROOM DISTRIBUTION
        # ----------------------------------------------------

        requested_room = request.get(
            "room"
        )

        if requested_room is not None:

            requested_room = str(
                requested_room
            )

            if requested_room not in returned_rooms:

                warnings.append(
                    "The requested room "
                    f"'{requested_room}' was not present "
                    "in the returned distribution data."
                )

        # ----------------------------------------------------
        # TOP / BOTTOM N DISTRIBUTION
        # ----------------------------------------------------

        distribution_top_n = request.get(
            "distribution_top_n"
        )

        if distribution_top_n is None:

            return (
                errors,
                warnings,
            )

        try:

            distribution_top_n = int(
                distribution_top_n
            )

        except (
            TypeError,
            ValueError,
        ):

            errors.append(
                "distribution_top_n is not a "
                "valid integer."
            )

            return (
                errors,
                warnings,
            )

        if distribution_top_n <= 0:

            errors.append(
                "distribution_top_n must be "
                "greater than zero."
            )

            return (
                errors,
                warnings,
            )

        # ----------------------------------------------------
        # VALIDATE NUMBER OF UNIQUE ROOMS
        # ----------------------------------------------------

        returned_room_count = len(
            returned_rooms
        )

        if (
            returned_room_count
            >
            distribution_top_n
        ):

            errors.append(
                "Ranked distribution returned "
                f"{returned_room_count} rooms, "
                f"but only {distribution_top_n} "
                "were requested."
            )

        elif (
            returned_room_count
            <
            distribution_top_n
        ):

            warnings.append(
                f"Requested {distribution_top_n} rooms "
                "for the distribution, but only "
                f"{returned_room_count} qualifying "
                "room(s) were returned."
            )

        # ----------------------------------------------------
        # REQUIRED RANKING METADATA
        # ----------------------------------------------------

        if rows:

            required_rank_fields = {
                "ranking_value",
                "room_rank",
                "selection_type",
            }

            row_fields = set(
                rows[0].keys()
            )

            missing_fields = (
                required_rank_fields
                - row_fields
            )

            if missing_fields:

                errors.append(
                    "Ranked distribution is missing "
                    "ranking metadata columns: "
                    f"{sorted(missing_fields)}"
                )

                return (
                    errors,
                    warnings,
                )

        # ----------------------------------------------------
        # VALIDATE RANKING VALUES
        # ----------------------------------------------------

        invalid_ranking_values = 0

        for row in rows:

            ranking_value = row.get(
                "ranking_value"
            )

            if not self._is_valid_number(
                ranking_value
            ):

                invalid_ranking_values += 1

        if invalid_ranking_values > 0:

            errors.append(
                f"{invalid_ranking_values} "
                "ranked distribution row(s) contain "
                "an invalid ranking_value."
            )

        # ----------------------------------------------------
        # VALIDATE ROOM RANK
        # ----------------------------------------------------

        invalid_room_ranks = 0

        for row in rows:

            room_rank = row.get(
                "room_rank"
            )

            try:

                room_rank_int = int(
                    room_rank
                )

            except (
                TypeError,
                ValueError,
            ):

                invalid_room_ranks += 1

                continue

            if (
                room_rank_int < 1
                or
                room_rank_int
                >
                distribution_top_n
            ):

                invalid_room_ranks += 1

        if invalid_room_ranks > 0:

            errors.append(
                f"{invalid_room_ranks} ranked "
                "distribution row(s) contain "
                "an invalid room_rank."
            )

        # ----------------------------------------------------
        # VALIDATE SELECTION TYPE
        # ----------------------------------------------------

        invalid_selection_type = 0

        for row in rows:

            selection_type = str(
                row.get(
                    "selection_type",
                    "",
                )
            ).casefold()

            if selection_type not in {
                "top",
                "bottom",
            }:

                invalid_selection_type += 1

        if invalid_selection_type > 0:

            errors.append(
                f"{invalid_selection_type} ranked "
                "distribution row(s) contain "
                "an invalid selection_type."
            )

        return (
            errors,
            warnings,
        )

    # ========================================================
    # RANKING SIZE VALIDATION
    # ========================================================

    def _validate_ranking(
        self,
        request: dict[str, Any],
        row_count: int,
    ) -> list[str]:

        warnings: list[str] = []

        top_n = request.get(
            "top_n"
        )

        if top_n is None:

            return warnings

        try:

            top_n = int(
                top_n
            )

        except (
            TypeError,
            ValueError,
        ):

            return warnings

        if row_count < top_n:

            warnings.append(
                f"Requested top {top_n}, "
                f"but only {row_count} "
                "qualifying room(s) were returned."
            )

        return warnings

    # ========================================================
    # LATEST DATA WARNING
    # ========================================================

    def _latest_data_warning(
        self,
        query_type: str,
        rows: list[dict[str, Any]],
    ) -> list[str]:

        warnings: list[str] = []

        if query_type != "rank_rooms_latest":

            return warnings

        if not rows:

            return warnings

        latest_times = [
            row.get(
                "latest_start_time"
            )
            for row in rows
            if row.get(
                "latest_start_time"
            )
            is not None
        ]

        if latest_times:

            latest_time = max(
                latest_times
            )

            warnings.append(
                "The result represents the latest "
                "available historical sensor reading "
                f"({latest_time}), not a live "
                "real-time measurement."
            )

        return warnings

    # ========================================================
    # RELATIONSHIP VALIDATION
    # ========================================================

    def _validate_relationship(
        self,
        request: dict[str, Any],
        rows: list[dict[str, Any]],
        row_count: int,
    ) -> tuple[
        list[str],
        list[str],
    ]:

        errors: list[str] = []
        warnings: list[str] = []

        # ----------------------------------------------------
        # SAMPLE SIZE
        # ----------------------------------------------------

        minimum_records = int(
            request.get(
                "minimum_records",
                2,
            )
        )

        if row_count < minimum_records:

            errors.append(
                "Relationship analysis requires at "
                f"least {minimum_records} paired "
                "observations."
            )

            return (
                errors,
                warnings,
            )

        # ----------------------------------------------------
        # SMALL RELATIONSHIP SAMPLE
        # ----------------------------------------------------

        if row_count < 30:

            warnings.append(
                "The relationship result contains fewer "
                "than 30 paired observations. "
                "Any apparent pattern should be "
                "interpreted cautiously."
            )

        # ----------------------------------------------------
        # REQUESTED ROOM COVERAGE
        # ----------------------------------------------------

        requested_room = request.get(
            "room"
        )

        if requested_room is not None:

            returned_rooms = {
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

            if (
                str(
                    requested_room
                )
                not in returned_rooms
            ):

                warnings.append(
                    "The requested room "
                    f"'{requested_room}' was not present "
                    "in the paired relationship data."
                )

        # ----------------------------------------------------
        # DUPLICATE SOURCE RECORD AGGREGATION
        # ----------------------------------------------------

        aggregated_pair_count = 0

        for row in rows:

            x_count = row.get(
                "x_record_count",
                1,
            )

            y_count = row.get(
                "y_record_count",
                1,
            )

            try:

                x_count = int(
                    x_count
                )

            except (
                TypeError,
                ValueError,
            ):

                x_count = 1

            try:

                y_count = int(
                    y_count
                )

            except (
                TypeError,
                ValueError,
            ):

                y_count = 1

            if (
                x_count > 1
                or
                y_count > 1
            ):

                aggregated_pair_count += 1

        if aggregated_pair_count > 0:

            warnings.append(
                f"{aggregated_pair_count} stored paired "
                "observation(s) were created from "
                "multiple source records at the same "
                "room and timestamp. These were "
                "aggregated before pairing."
            )

        return (
            errors,
            warnings,
        )

    # ========================================================
    # MAIN VALIDATION
    # ========================================================

    def validate(
        self,
        request: dict[str, Any],
        query_type: str,
        execution_success: bool,
        columns: list[str],
        rows: list[dict[str, Any]],
        row_count: int,
        truncated: bool = False,
    ) -> dict[str, Any]:

        errors: list[str] = []
        warnings: list[str] = []

        # ====================================================
        # 1. EXECUTION STATUS
        # ====================================================

        if not execution_success:

            return {
                "valid":
                    False,

                "status":
                    "invalid",

                "has_data":
                    False,

                "errors": [
                    "SQL execution was not successful."
                ],

                "warnings":
                    [],
            }

        # ====================================================
        # NORMALISE ROW COUNT
        # ====================================================

        try:

            row_count = int(
                row_count
            )

        except (
            TypeError,
            ValueError,
        ):

            return {
                "valid":
                    False,

                "status":
                    "invalid",

                "has_data":
                    False,

                "errors": [
                    "SQL row_count is invalid."
                ],

                "warnings":
                    [],
            }

        # ====================================================
        # 2. EMPTY RESULT
        # ====================================================

        if row_count == 0:

            # ------------------------------------------------
            # Relationship requires paired observations.
            # ------------------------------------------------

            if query_type == "relationship":

                return {
                    "valid":
                        False,

                    "status":
                        "no_data",

                    "has_data":
                        False,

                    "errors": [
                        "No paired observations were "
                        "available for the requested "
                        "relationship."
                    ],

                    "warnings":
                        [],
                }

            return {
                "valid":
                    True,

                "status":
                    "no_data",

                "has_data":
                    False,

                "errors":
                    [],

                "warnings": [
                    "The SQL query executed successfully, "
                    "but no records matched the request."
                ],
            }

        # ====================================================
        # 3. COLUMN STRUCTURE
        # ====================================================

        errors.extend(
            self._validate_columns(
                query_type=
                    query_type,

                columns=
                    columns,
            )
        )

        # ====================================================
        # 4. NUMERIC CONTENT
        # ====================================================

        errors.extend(
            self._validate_numeric_values(
                query_type=
                    query_type,

                rows=
                    rows,
            )
        )

        # ====================================================
        # 5. QUERY-SPECIFIC VALIDATION
        # ====================================================

        # ----------------------------------------------------
        # ROOM COMPARISON
        # ----------------------------------------------------

        if query_type == "compare_rooms":

            warnings.extend(
                self._validate_compare_rooms(
                    request=
                        request,

                    rows=
                        rows,
                )
            )

        # ----------------------------------------------------
        # DISTRIBUTION
        # ----------------------------------------------------

        if query_type == "distribution":

            (
                distribution_errors,
                distribution_warnings,
            ) = (
                self._validate_distribution(
                    request=
                        request,

                    rows=
                        rows,
                )
            )

            errors.extend(
                distribution_errors
            )

            warnings.extend(
                distribution_warnings
            )

        # ----------------------------------------------------
        # RANKING
        # ----------------------------------------------------

        if query_type in {
            "rank_rooms",
            "rank_rooms_latest",
        }:

            warnings.extend(
                self._validate_ranking(
                    request=
                        request,

                    row_count=
                        row_count,
                )
            )

        # ----------------------------------------------------
        # RELATIONSHIP
        # ----------------------------------------------------

        if query_type == "relationship":

            (
                relationship_errors,
                relationship_warnings,
            ) = (
                self._validate_relationship(
                    request=
                        request,

                    rows=
                        rows,

                    row_count=
                        row_count,
                )
            )

            errors.extend(
                relationship_errors
            )

            warnings.extend(
                relationship_warnings
            )

        # ====================================================
        # 6. TRUNCATION
        # ====================================================

        if truncated:

            if query_type == "relationship":

                warnings.append(
                    "The relationship result exceeded "
                    "the downstream storage limit. "
                    "The scatter plot would use only "
                    "the stored subset unless the result "
                    "limit is increased or controlled "
                    "sampling is applied."
                )

            else:

                warnings.append(
                    "The complete SQL result was larger "
                    "than the LangGraph storage limit. "
                    "Only part of the result is available "
                    "for downstream processing."
                )

        # ====================================================
        # 7. HISTORICAL LATEST DATA WARNING
        # ====================================================

        warnings.extend(
            self._latest_data_warning(
                query_type=
                    query_type,

                rows=
                    rows,
            )
        )

        # ====================================================
        # 8. FINAL STATUS
        # ====================================================

        if errors:

            status = "invalid"
            valid = False

        else:

            status = "valid"
            valid = True

        return {
            "valid":
                valid,

            "status":
                status,

            "has_data":
                True,

            "errors":
                errors,

            "warnings":
                warnings,
        }