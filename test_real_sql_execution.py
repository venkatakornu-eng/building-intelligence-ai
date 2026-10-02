from __future__ import annotations

from src.agents.query_builder_agent import (
    QueryBuilderAgent,
)

from src.agents.sql_execution_agent import (
    SQLExecutionAgent,
)

from src.agents.result_validation_agent import (
    ResultValidationAgent,
)

from src.sql_database import (
    SQLDatabase,
)


# ============================================================
# DATABASE CONNECTION
# ============================================================

database = SQLDatabase(
    server=r"localhost",
    database="SmartVizDB",
    driver="ODBC Driver 18 for SQL Server",
)


# ============================================================
# QUERY BUILDER
# ============================================================

query_builder = QueryBuilderAgent(
    table_name="dbo.room_level_metrics"
)


# ============================================================
# SQL EXECUTION AGENT
# ============================================================

sql_executor = SQLExecutionAgent(
    database=database,
    max_result_rows=50_000,
)


# ============================================================
# RESULT VALIDATION AGENT
# ============================================================

result_validator = ResultValidationAgent()


# ============================================================
# REAL SQL TEST REQUESTS
# ============================================================

tests = [

    # ========================================================
    # TEST 1
    # OCCUPANCY VS CO2 - ALL ROOMS
    # ========================================================

    {
        "name":
            "Occupancy vs CO2 - All Rooms",

        "request": {

            "intent":
                "relationship",

            "chart_type":
                "scatter",

            "metric_name":
                None,

            "metric_x":
                "occupancy",

            "metric_y":
                "co2",

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

            "threshold_operator":
                None,

            "minimum_records":
                2,
        },
    },


    # ========================================================
    # TEST 2
    # TEMPERATURE VS HUMIDITY - ONE ROOM
    # ========================================================

    {
        "name":
            "Temperature vs Humidity - Seminar Room 3",

        "request": {

            "intent":
                "relationship",

            "chart_type":
                "scatter",

            "metric_name":
                None,

            "metric_x":
                "temp",

            "metric_y":
                "humidity",

            "room":
                "Seminar Room 3",

            "room_names": [
                "Seminar Room 3"
            ],

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

            "threshold_operator":
                None,

            "minimum_records":
                2,
        },
    },


    # ========================================================
    # TEST 3
    # CO2 VS TEMPERATURE - MAY 2025
    # ========================================================

    {
        "name":
            "CO2 vs Temperature - May 2025",

        "request": {

            "intent":
                "relationship",

            "chart_type":
                "scatter",

            "metric_name":
                None,

            "metric_x":
                "co2",

            "metric_y":
                "temp",

            "room":
                None,

            "room_names":
                [],

            "aggregation":
                "mean",

            "frequency":
                "hourly",

            "start_date":
                "2025-05-01",

            "end_date":
                "2025-05-31",

            "top_n":
                10,

            "ascending":
                False,

            "threshold":
                None,

            "time_scope":
                "range",

            "continuous":
                False,

            "threshold_operator":
                None,

            "minimum_records":
                2,
        },
    },
]


# ============================================================
# RUN REAL SQL TESTS
# ============================================================

for number, test in enumerate(
    tests,
    start=1,
):

    print(
        "\n"
        + "=" * 80
    )

    print(
        f"REAL SQL TEST {number}"
    )

    print(
        test["name"]
    )

    print(
        "=" * 80
    )


    request = (
        test[
            "request"
        ]
    )


    # ========================================================
    # 1. BUILD SQL
    # ========================================================

    try:

        query_result = (
            query_builder.build(
                request
            )
        )

    except Exception as error:

        print(
            "\n❌ QUERY BUILD FAILED"
        )

        print(
            type(error).__name__,
            ":",
            error
        )

        continue


    print(
        "\nQUERY TYPE:"
    )

    print(
        query_result[
            "query_type"
        ]
    )


    print(
        "\nMETRIC X:"
    )

    print(
        query_result.get(
            "metric_x"
        )
    )


    print(
        "\nMETRIC Y:"
    )

    print(
        query_result.get(
            "metric_y"
        )
    )


    print(
        "\nPARAMETERS:"
    )

    print(
        query_result[
            "params"
        ]
    )


    # ========================================================
    # 2. EXECUTE AGAINST REAL SQL SERVER
    # ========================================================

    try:

        result = (
            sql_executor.execute(
                sql=
                    query_result[
                        "sql"
                    ],

                params=
                    query_result[
                        "params"
                    ],

                query_type=
                    query_result[
                        "query_type"
                    ],
            )
        )

    except Exception as error:

        print(
            "\n❌ REAL SQL EXECUTION FAILED"
        )

        print(
            type(error).__name__,
            ":",
            error
        )

        continue


    # ========================================================
    # 3. BASIC EXECUTION INFORMATION
    # ========================================================

    print(
        "\n✅ REAL SQL EXECUTION SUCCESS"
    )


    returned_query_type = (
        result.get(
            "query_type"
        )
    )

    columns = (
        result.get(
            "columns",
            []
        )
    )

    rows = (
        result.get(
            "rows",
            []
        )
    )

    row_count = int(
        result.get(
            "row_count",
            0,
        )
    )

    stored_row_count = int(
        result.get(
            "stored_row_count",
            len(rows),
        )
    )

    truncated = bool(
        result.get(
            "truncated",
            False,
        )
    )


    print(
        "\nRETURNED QUERY TYPE:"
    )

    print(
        returned_query_type
    )


    print(
        "\nCOLUMNS:"
    )

    print(
        columns
    )


    print(
        "\nTOTAL ROWS RETURNED:"
    )

    print(
        row_count
    )


    print(
        "\nROWS STORED:"
    )

    print(
        stored_row_count
    )


    print(
        "\nTRUNCATED:"
    )

    print(
        truncated
    )


    # ========================================================
    # 4. CHECK EXPECTED RELATIONSHIP COLUMNS
    # ========================================================

    expected_columns = {
        "display_name",
        "start_time",
        "x_value",
        "y_value",
        "x_record_count",
        "y_record_count",
    }

    actual_columns = set(
        columns
    )

    missing_columns = (
        expected_columns
        - actual_columns
    )


    print(
        "\nRELATIONSHIP COLUMN CHECK:"
    )

    if not missing_columns:

        print(
            "✅ All expected relationship columns exist."
        )

    else:

        print(
            "❌ Missing columns:",
            missing_columns
        )


    # ========================================================
    # 5. CHECK WHETHER REAL PAIRED DATA EXISTS
    # ========================================================

    if row_count >= 2:

        print(
            "\n✅ PAIRED OBSERVATIONS FOUND:"
        )

        print(
            f"{row_count:,}"
        )

    elif row_count == 1:

        print(
            "\n⚠ Only one paired observation found."
        )

        print(
            "This is not enough for meaningful "
            "relationship analysis."
        )

    else:

        print(
            "\n❌ No paired observations were found."
        )

        print(
            "The metrics may not have matching "
            "room/timestamp observations."
        )


    # ========================================================
    # 6. DISPLAY FIRST FIVE REAL PAIRED ROWS
    # ========================================================

    print(
        "\nFIRST 5 PAIRED ROWS:"
    )


    if not rows:

        print(
            "No rows available."
        )

    else:

        for row_number, row in enumerate(
            rows[:5],
            start=1,
        ):

            print(
                f"\nROW {row_number}"
            )

            print(
                "Room:",
                row.get(
                    "display_name"
                )
            )

            print(
                "Timestamp:",
                row.get(
                    "start_time"
                )
            )

            print(
                "X Value:",
                row.get(
                    "x_value"
                )
            )

            print(
                "Y Value:",
                row.get(
                    "y_value"
                )
            )

            print(
                "X Record Count:",
                row.get(
                    "x_record_count"
                )
            )

            print(
                "Y Record Count:",
                row.get(
                    "y_record_count"
                )
            )


    # ========================================================
    # 7. CHECK FOR NULL X/Y VALUES
    # ========================================================

    null_pair_count = 0


    for row in rows:

        if (
            row.get(
                "x_value"
            ) is None
            or
            row.get(
                "y_value"
            ) is None
        ):

            null_pair_count += 1


    print(
        "\nNULL PAIRED VALUES:"
    )

    print(
        null_pair_count
    )


    if null_pair_count == 0:

        print(
            "✅ No NULL x/y pairs in stored results."
        )

    else:

        print(
            "⚠ Some paired results contain NULL values."
        )


    # ========================================================
    # 8. RESULT VALIDATION AGENT
    # ========================================================

    try:

        validation = (
            result_validator.validate(
                request=request,

                query_type=
                    returned_query_type,

                execution_success=
                    True,

                columns=
                    columns,

                rows=
                    rows,

                row_count=
                    row_count,

                truncated=
                    truncated,
            )
        )

    except Exception as error:

        print(
            "\n❌ RESULT VALIDATION FAILED"
        )

        print(
            type(error).__name__,
            ":",
            error
        )

        continue


    # ========================================================
    # 9. PRINT RESULT VALIDATION
    # ========================================================

    print(
        "\n"
        + "-" * 80
    )

    print(
        "RESULT VALIDATION"
    )

    print(
        "-" * 80
    )


    print(
        "Valid:",
        validation.get(
            "valid"
        )
    )


    print(
        "Status:",
        validation.get(
            "status"
        )
    )


    print(
        "Has Data:",
        validation.get(
            "has_data"
        )
    )


    print(
        "Errors:"
    )

    errors = (
        validation.get(
            "errors",
            []
        )
    )


    if errors:

        for error in errors:

            print(
                " -",
                error
            )

    else:

        print(
            " []"
        )


    print(
        "Warnings:"
    )

    warnings = (
        validation.get(
            "warnings",
            []
        )
    )


    if warnings:

        for warning in warnings:

            print(
                " -",
                warning
            )

    else:

        print(
            " []"
        )


    # ========================================================
    # 10. FINAL COMBINED TEST STATUS
    # ========================================================

    print(
        "\n"
        + "-" * 80
    )

    print(
        "FINAL TEST STATUS"
    )

    print(
        "-" * 80
    )


    structural_success = (
        row_count >= 2
        and not missing_columns
        and null_pair_count == 0
    )


    validation_success = bool(
        validation.get(
            "valid",
            False,
        )
    )


    if (
        structural_success
        and validation_success
    ):

        print(
            "✅ RELATIONSHIP RESULT VALIDATION PASSED"
        )

        if truncated:

            print(
                "⚠ Result is valid, but downstream data "
                "is currently truncated."
            )

    else:

        print(
            "❌ RELATIONSHIP RESULT VALIDATION FAILED"
        )


    # ========================================================
    # 11. SUMMARY
    # ========================================================

    print(
        "\nSUMMARY:"
    )

    print(
        "Metric X:",
        request.get(
            "metric_x"
        )
    )

    print(
        "Metric Y:",
        request.get(
            "metric_y"
        )
    )

    print(
        "Total Paired Rows:",
        f"{row_count:,}"
    )

    print(
        "Stored Rows:",
        f"{stored_row_count:,}"
    )

    print(
        "Truncated:",
        truncated
    )

    print(
        "Validation Status:",
        validation.get(
            "status"
        )
    )