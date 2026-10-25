from __future__ import annotations

import re
from typing import Any


class QueryBuilderAgent:
    """
    Deterministic and parameterised SQL Server query builder
    for the cleaned SmartViz analytical table.

    The LLM never generates executable SQL directly.
    User-provided values are passed through SQL parameters.

    Backend table:
        dbo.room_level_metrics

    Supported analytical intents:
        - room_trend
        - compare_rooms
        - rank_rooms
        - threshold
        - distribution
        - relationship
    """

    def __init__(
        self,
        table_name: str = "dbo.room_level_metrics",
    ) -> None:

        self.table_name = self._quote_identifier(
            table_name
        )

    # ========================================================
    # SAFE SQL IDENTIFIER
    # ========================================================

    def _quote_identifier(
        self,
        identifier: str,
    ) -> str:

        parts = identifier.split(".")

        valid_pattern = (
            r"^[A-Za-z_][A-Za-z0-9_]*$"
        )

        for part in parts:

            if not re.match(
                valid_pattern,
                part,
            ):

                raise ValueError(
                    f"Unsafe SQL identifier: {identifier}"
                )

        return ".".join(
            f"[{part}]"
            for part in parts
        )

    # ========================================================
    # COMMON FILTERS
    # ========================================================

    def _common_filters(
        self,
        request: dict[str, Any],
        table_alias: str = "r",
    ) -> tuple[list[str], list[Any]]:
        """
        Build deterministic WHERE conditions for
        single-metric requests.

        Relationship requests use their own filter method
        because they require two separate metrics.
        """

        metric_name = request.get(
            "metric_name"
        )

        if metric_name is None:

            raise ValueError(
                "metric_name is required for "
                "single-metric SQL queries."
            )

        conditions = [
            f"LOWER({table_alias}.metric_name) = ?",
            f"LOWER({table_alias}.aggregation) = ?",
            f"LOWER({table_alias}.frequency) = ?",
            f"{table_alias}.display_name IS NOT NULL",
            f"{table_alias}.value IS NOT NULL",
        ]

        params: list[Any] = [
            str(
                metric_name
            ).lower(),

            str(
                request.get(
                    "aggregation",
                    "mean",
                )
            ).lower(),

            str(
                request.get(
                    "frequency",
                    "hourly",
                )
            ).lower(),
        ]

        # ----------------------------------------------------
        # DATE RANGE
        # ----------------------------------------------------

        if request.get(
            "time_scope"
        ) == "range":

            start_date = request.get(
                "start_date"
            )

            end_date = request.get(
                "end_date"
            )

            if (
                start_date is None
                or end_date is None
            ):

                raise ValueError(
                    "Date-range query requires "
                    "start_date and end_date."
                )

            conditions.append(
                f"{table_alias}.start_time >= ?"
            )

            conditions.append(
                f"{table_alias}.start_time "
                f"< DATEADD(day, 1, ?)"
            )

            params.extend(
                [
                    start_date,
                    end_date,
                ]
            )

        return (
            conditions,
            params,
        )

    # ========================================================
    # RELATIONSHIP FILTERS
    # ========================================================

    def _relationship_filters(
        self,
        request: dict[str, Any],
        metric_name: str,
        table_alias: str,
    ) -> tuple[list[str], list[Any]]:
        """
        Build filters for one metric in a relationship query.

        The two metrics are filtered independently and later
        paired using:

            display_name
            start_time
        """

        conditions = [
            f"LOWER({table_alias}.metric_name) = ?",
            f"LOWER({table_alias}.aggregation) = ?",
            f"LOWER({table_alias}.frequency) = ?",
            f"{table_alias}.display_name IS NOT NULL",
            f"{table_alias}.start_time IS NOT NULL",
            f"{table_alias}.value IS NOT NULL",
        ]

        params: list[Any] = [
            str(
                metric_name
            ).lower(),

            str(
                request.get(
                    "aggregation",
                    "mean",
                )
            ).lower(),

            str(
                request.get(
                    "frequency",
                    "hourly",
                )
            ).lower(),
        ]

        # ----------------------------------------------------
        # DATE RANGE
        # ----------------------------------------------------

        if request.get(
            "time_scope"
        ) == "range":

            start_date = request.get(
                "start_date"
            )

            end_date = request.get(
                "end_date"
            )

            if (
                start_date is None
                or end_date is None
            ):

                raise ValueError(
                    "Relationship date-range query requires "
                    "start_date and end_date."
                )

            conditions.append(
                f"{table_alias}.start_time >= ?"
            )

            conditions.append(
                f"{table_alias}.start_time "
                f"< DATEADD(day, 1, ?)"
            )

            params.extend(
                [
                    start_date,
                    end_date,
                ]
            )

        # ----------------------------------------------------
        # ONE ROOM
        # ----------------------------------------------------

        room = request.get(
            "room"
        )

        if room is not None:

            conditions.append(
                f"{table_alias}.display_name = ?"
            )

            params.append(
                room
            )

        # ----------------------------------------------------
        # OPTIONAL MULTIPLE ROOMS
        # ----------------------------------------------------

        elif request.get(
            "room_names"
        ):

            room_names = list(
                request[
                    "room_names"
                ]
            )

            placeholders = ", ".join(
                "?"
                for _ in room_names
            )

            conditions.append(
                f"{table_alias}.display_name "
                f"IN ({placeholders})"
            )

            params.extend(
                room_names
            )

        return (
            conditions,
            params,
        )

    # ========================================================
    # ROOM TREND
    # ========================================================

    def _room_trend(
        self,
        request: dict[str, Any],
    ) -> dict[str, Any]:

        conditions, params = (
            self._common_filters(
                request
            )
        )

        room = request.get(
            "room"
        )

        if room is None:

            raise ValueError(
                "room_trend requires a room."
            )

        conditions.append(
            "r.display_name = ?"
        )

        params.append(
            room
        )

        where_clause = (
            "\n    AND ".join(
                conditions
            )
        )

        sql = f"""
SELECT
    r.start_time,
    r.end_time,
    CAST(r.value AS FLOAT) AS value,
    r.display_name,
    r.metric_name
FROM {self.table_name} AS r
WHERE
    {where_clause}
ORDER BY
    r.start_time ASC;
""".strip()

        return {
            "sql":
                sql,

            "params":
                params,

            "query_type":
                "room_trend",
        }

    # ========================================================
    # COMPARE ROOMS
    # ========================================================

    def _compare_rooms(
        self,
        request: dict[str, Any],
    ) -> dict[str, Any]:

        conditions, params = (
            self._common_filters(
                request
            )
        )

        room_names = list(
            request.get(
                "room_names",
                [],
            )
        )

        if len(
            room_names
        ) < 2:

            raise ValueError(
                "compare_rooms requires at least "
                "two room names."
            )

        placeholders = ", ".join(
            "?"
            for _ in room_names
        )

        conditions.append(
            f"r.display_name IN ({placeholders})"
        )

        params.extend(
            room_names
        )

        where_clause = (
            "\n    AND ".join(
                conditions
            )
        )

        sql = f"""
SELECT
    r.display_name,

    AVG(
        CAST(
            r.value AS FLOAT
        )
    ) AS average_value,

    MIN(
        CAST(
            r.value AS FLOAT
        )
    ) AS minimum_value,

    MAX(
        CAST(
            r.value AS FLOAT
        )
    ) AS maximum_value,

    COUNT_BIG(*) AS record_count

FROM {self.table_name} AS r

WHERE
    {where_clause}

GROUP BY
    r.display_name

ORDER BY
    average_value DESC;
""".strip()

        return {
            "sql":
                sql,

            "params":
                params,

            "query_type":
                "compare_rooms",
        }

    # ========================================================
    # RANK ROOMS
    # ========================================================

    def _rank_rooms(
        self,
        request: dict[str, Any],
    ) -> dict[str, Any]:

        top_n = int(
            request.get(
                "top_n",
                10,
            )
        )

        if top_n <= 0:

            raise ValueError(
                "top_n must be greater than zero."
            )

        ascending = bool(
            request.get(
                "ascending",
                False,
            )
        )

        direction = (
            "ASC"
            if ascending
            else "DESC"
        )

        # ====================================================
        # LATEST / CURRENT RANKING
        # ====================================================

        if request.get(
            "time_scope"
        ) == "latest":

            metric_name = request.get(
                "metric_name"
            )

            if metric_name is None:

                raise ValueError(
                    "Latest ranking requires metric_name."
                )

            sql = f"""
WITH filtered_data AS
(
    SELECT
        r.display_name,
        r.start_time,
        CAST(r.value AS FLOAT) AS value,

        ROW_NUMBER() OVER
        (
            PARTITION BY
                r.display_name

            ORDER BY
                r.start_time DESC
        ) AS row_num

    FROM {self.table_name} AS r

    WHERE
        LOWER(r.metric_name) = ?
        AND LOWER(r.aggregation) = ?
        AND LOWER(r.frequency) = ?
        AND r.display_name IS NOT NULL
        AND r.start_time IS NOT NULL
        AND r.value IS NOT NULL
)

SELECT TOP ({top_n})
    display_name,
    value AS latest_value,
    start_time AS latest_start_time,
    CAST(1 AS BIGINT) AS record_count

FROM filtered_data

WHERE
    row_num = 1

ORDER BY
    latest_value {direction};
""".strip()

            params = [
                str(
                    metric_name
                ).lower(),

                str(
                    request.get(
                        "aggregation",
                        "mean",
                    )
                ).lower(),

                str(
                    request.get(
                        "frequency",
                        "hourly",
                    )
                ).lower(),
            ]

            return {
                "sql":
                    sql,

                "params":
                    params,

                "query_type":
                    "rank_rooms_latest",
            }

        # ====================================================
        # HISTORICAL / DATE-RANGE RANKING
        # ====================================================

        conditions, params = (
            self._common_filters(
                request
            )
        )

        where_clause = (
            "\n    AND ".join(
                conditions
            )
        )

        minimum_records = int(
            request.get(
                "minimum_records",
                1,
            )
        )

        sql = f"""
SELECT TOP ({top_n})
    r.display_name,

    AVG(
        CAST(
            r.value AS FLOAT
        )
    ) AS average_value,

    MIN(
        CAST(
            r.value AS FLOAT
        )
    ) AS minimum_value,

    MAX(
        CAST(
            r.value AS FLOAT
        )
    ) AS maximum_value,

    COUNT_BIG(*) AS record_count

FROM {self.table_name} AS r

WHERE
    {where_clause}

GROUP BY
    r.display_name

HAVING
    COUNT_BIG(*) >= ?

ORDER BY
    average_value {direction};
""".strip()

        params.append(
            minimum_records
        )

        return {
            "sql":
                sql,

            "params":
                params,

            "query_type":
                "rank_rooms",
        }

    # ========================================================
    # THRESHOLD ANALYSIS
    # ========================================================

    def _threshold(
        self,
        request: dict[str, Any],
    ) -> dict[str, Any]:

        conditions, params = (
            self._common_filters(
                request
            )
        )

        where_clause = (
            "\n    AND ".join(
                conditions
            )
        )

        operator = request.get(
            "threshold_operator"
        )

        if operator in {
            "gt",
            ">",
        }:

            sql_operator = ">"
            direction = "DESC"

        elif operator in {
            "lt",
            "<",
        }:

            sql_operator = "<"
            direction = "ASC"

        else:

            raise ValueError(
                "Unsupported threshold operator."
            )

        threshold = float(
            request[
                "threshold"
            ]
        )

        minimum_records = int(
            request.get(
                "minimum_records",
                1,
            )
        )

        sql = f"""
SELECT
    r.display_name,

    AVG(
        CAST(
            r.value AS FLOAT
        )
    ) AS average_value,

    MIN(
        CAST(
            r.value AS FLOAT
        )
    ) AS minimum_value,

    MAX(
        CAST(
            r.value AS FLOAT
        )
    ) AS maximum_value,

    COUNT_BIG(*) AS record_count

FROM {self.table_name} AS r

WHERE
    {where_clause}

GROUP BY
    r.display_name

HAVING
    AVG(
        CAST(
            r.value AS FLOAT
        )
    ) {sql_operator} ?

    AND COUNT_BIG(*) >= ?

ORDER BY
    average_value {direction};
""".strip()

        params.extend(
            [
                threshold,
                minimum_records,
            ]
        )

        return {
            "sql":
                sql,

            "params":
                params,

            "query_type":
                "threshold",
        }

    # ========================================================
    # DISTRIBUTION
    # ========================================================

    def _distribution(
        self,
        request: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Build raw-observation distribution data.

        Supported distribution forms:

        1. All rooms
           "Show humidity distribution"

        2. One room
           "Show CO2 distribution in Seminar Room 3"

        3. Top/bottom N rooms
           "Show humidity distribution of top 4 rooms"

        For top/bottom N:
            - rank rooms by average metric value
            - select requested rooms
            - return raw observations only for those rooms

        Raw observations are preserved so histogram/box plots
        remain true distributions rather than distributions of
        pre-aggregated room averages.
        """

        conditions, params = (
            self._common_filters(
                request
            )
        )

        # ====================================================
        # ONE SPECIFIC ROOM
        # ====================================================

        room = request.get(
            "room"
        )

        if room is not None:

            conditions.append(
                "r.display_name = ?"
            )

            params.append(
                room
            )

            where_clause = (
                "\n    AND ".join(
                    conditions
                )
            )

            sql = f"""
SELECT
    r.start_time,
    CAST(r.value AS FLOAT) AS value,
    r.display_name,
    r.metric_name

FROM {self.table_name} AS r

WHERE
    {where_clause}

ORDER BY
    r.start_time ASC;
""".strip()

            return {
                "sql":
                    sql,

                "params":
                    params,

                "query_type":
                    "distribution",

                "distribution_scope":
                    "single_room",
            }

        # ====================================================
        # TOP / BOTTOM N ROOM DISTRIBUTION
        # ====================================================

        distribution_top_n = request.get(
            "distribution_top_n"
        )

        if distribution_top_n is not None:

            top_n = int(
                distribution_top_n
            )

            if top_n <= 0:

                raise ValueError(
                    "distribution_top_n must be "
                    "greater than zero."
                )

            ascending = bool(
                request.get(
                    "ascending",
                    False,
                )
            )

            direction = (
                "ASC"
                if ascending
                else "DESC"
            )

            selection_type = (
                "bottom"
                if ascending
                else "top"
            )

            where_clause = (
                "\n        AND ".join(
                    conditions
                )
            )

            sql = f"""
WITH filtered_data AS
(
    SELECT
        r.start_time,
        CAST(r.value AS FLOAT) AS value,
        r.display_name,
        r.metric_name

    FROM {self.table_name} AS r

    WHERE
        {where_clause}
),

room_scores AS
(
    SELECT
        display_name,

        AVG(value)
            AS ranking_value,

        COUNT_BIG(*)
            AS ranking_record_count

    FROM filtered_data

    GROUP BY
        display_name
),

ranked_rooms AS
(
    SELECT
        display_name,
        ranking_value,
        ranking_record_count,

        ROW_NUMBER() OVER
        (
            ORDER BY
                ranking_value {direction},
                display_name ASC
        ) AS room_rank

    FROM room_scores
)

SELECT
    f.start_time,
    f.value,
    f.display_name,
    f.metric_name,

    rr.ranking_value,
    rr.ranking_record_count,
    rr.room_rank,

    CAST(
        '{selection_type}'
        AS VARCHAR(10)
    ) AS selection_type

FROM filtered_data AS f

INNER JOIN ranked_rooms AS rr
    ON
        f.display_name =
        rr.display_name

WHERE
    rr.room_rank <= ?

ORDER BY
    rr.room_rank ASC,
    f.display_name ASC,
    f.start_time ASC;
""".strip()

            params.append(
                top_n
            )

            return {
                "sql":
                    sql,

                "params":
                    params,

                "query_type":
                    "distribution",

                "distribution_scope":
                    "ranked_rooms",

                "distribution_top_n":
                    top_n,

                "selection_type":
                    selection_type,
            }

        # ====================================================
        # NORMAL DISTRIBUTION ACROSS ALL ROOMS
        # ====================================================

        where_clause = (
            "\n    AND ".join(
                conditions
            )
        )

        sql = f"""
SELECT
    r.start_time,
    CAST(r.value AS FLOAT) AS value,
    r.display_name,
    r.metric_name

FROM {self.table_name} AS r

WHERE
    {where_clause}

ORDER BY
    r.start_time ASC;
""".strip()

        return {
            "sql":
                sql,

            "params":
                params,

            "query_type":
                "distribution",

            "distribution_scope":
                "all_rooms",
        }

    # ========================================================
    # RELATIONSHIP / SCATTER
    # ========================================================

    def _relationship(
        self,
        request: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Build a paired two-metric dataset for scatter plots.

        Example:

            metric_x = occupancy
            metric_y = co2

        Both metrics are first aggregated independently by:

            display_name
            start_time

        and are then joined using those exact keys.

        This prevents incorrect room/timestamp pairing and
        prevents duplicate Cartesian multiplication.
        """

        metric_x = request.get(
            "metric_x"
        )

        metric_y = request.get(
            "metric_y"
        )

        # ----------------------------------------------------
        # BASIC SAFETY
        # ----------------------------------------------------

        if metric_x is None:

            raise ValueError(
                "relationship requires metric_x."
            )

        if metric_y is None:

            raise ValueError(
                "relationship requires metric_y."
            )

        if (
            str(
                metric_x
            ).lower()
            ==
            str(
                metric_y
            ).lower()
        ):

            raise ValueError(
                "relationship requires two "
                "different metrics."
            )

        # ----------------------------------------------------
        # LATEST-ONLY IS NOT A RELATIONSHIP SAMPLE
        # ----------------------------------------------------

        if request.get(
            "time_scope"
        ) == "latest":

            raise ValueError(
                "Relationship analysis does not support "
                "latest-only queries."
            )

        # ----------------------------------------------------
        # METRIC X FILTERS
        # ----------------------------------------------------

        x_conditions, x_params = (
            self._relationship_filters(
                request=request,

                metric_name=str(
                    metric_x
                ),

                table_alias="x",
            )
        )

        x_where_clause = (
            "\n        AND ".join(
                x_conditions
            )
        )

        # ----------------------------------------------------
        # METRIC Y FILTERS
        # ----------------------------------------------------

        y_conditions, y_params = (
            self._relationship_filters(
                request=request,

                metric_name=str(
                    metric_y
                ),

                table_alias="y",
            )
        )

        y_where_clause = (
            "\n        AND ".join(
                y_conditions
            )
        )

        # ----------------------------------------------------
        # AGGREGATE EACH SIDE BEFORE JOINING
        # ----------------------------------------------------

        sql = f"""
WITH metric_x_data AS
(
    SELECT
        x.display_name,
        x.start_time,

        AVG(
            CAST(
                x.value AS FLOAT
            )
        ) AS x_value,

        COUNT_BIG(*)
            AS x_record_count

    FROM {self.table_name} AS x

    WHERE
        {x_where_clause}

    GROUP BY
        x.display_name,
        x.start_time
),

metric_y_data AS
(
    SELECT
        y.display_name,
        y.start_time,

        AVG(
            CAST(
                y.value AS FLOAT
            )
        ) AS y_value,

        COUNT_BIG(*)
            AS y_record_count

    FROM {self.table_name} AS y

    WHERE
        {y_where_clause}

    GROUP BY
        y.display_name,
        y.start_time
)

SELECT
    mx.display_name,
    mx.start_time,
    mx.x_value,
    my.y_value,
    mx.x_record_count,
    my.y_record_count

FROM metric_x_data AS mx

INNER JOIN metric_y_data AS my
    ON
        mx.display_name =
        my.display_name

        AND

        mx.start_time =
        my.start_time

ORDER BY
    mx.start_time ASC,
    mx.display_name ASC;
""".strip()

        params = (
            x_params
            + y_params
        )

        return {
            "sql":
                sql,

            "params":
                params,

            "query_type":
                "relationship",

            "metric_x":
                metric_x,

            "metric_y":
                metric_y,
        }

    # ========================================================
    # PUBLIC BUILD METHOD
    # ========================================================

    def build(
        self,
        request: dict[str, Any],
    ) -> dict[str, Any]:

        if not isinstance(
            request,
            dict,
        ):

            raise TypeError(
                "request must be a dictionary."
            )

        # ----------------------------------------------------
        # CONTINUOUS ANALYSIS
        # ----------------------------------------------------

        if request.get(
            "continuous",
            False,
        ):

            raise ValueError(
                "Continuous analysis is not yet "
                "supported by the SQL builder."
            )

        intent = request.get(
            "intent"
        )

        if intent is None:

            raise ValueError(
                "Request does not contain an intent."
            )

        # ----------------------------------------------------
        # ROUTING
        # ----------------------------------------------------

        if intent == "room_trend":

            return self._room_trend(
                request
            )

        if intent == "compare_rooms":

            return self._compare_rooms(
                request
            )

        if intent == "rank_rooms":

            return self._rank_rooms(
                request
            )

        if intent == "threshold":

            return self._threshold(
                request
            )

        if intent == "distribution":

            return self._distribution(
                request
            )

        if intent == "relationship":

            return self._relationship(
                request
            )

        raise ValueError(
            f"Unsupported intent: {intent}"
        )