from src.agents.query_builder_agent import (
    QueryBuilderAgent,
)


# ============================================================
# INITIALISE QUERY BUILDER
# ============================================================

builder = QueryBuilderAgent(
    table_name="dbo.room_level_metrics"
)


# ============================================================
# TEST REQUESTS
# ============================================================

tests = [

    # ========================================================
    # TEST 1
    # HOTTEST ROOM RIGHT NOW
    # ========================================================

    {
        "intent": "rank_rooms",

        "metric_name": "temp",

        "aggregation": "mean",
        "frequency": "hourly",

        "time_scope": "latest",

        "start_date": None,
        "end_date": None,

        "top_n": 1,

        "ascending": False,

        "minimum_records": 1,

        "continuous": False,
    },


    # ========================================================
    # TEST 2
    # TEN LEAST OCCUPIED ROOMS
    # ========================================================

    {
        "intent": "rank_rooms",

        "metric_name": "occupancy",

        "aggregation": "mean",
        "frequency": "hourly",

        "time_scope": "all",

        "start_date": None,
        "end_date": None,

        "top_n": 10,

        "ascending": True,

        "minimum_records": 100,

        "continuous": False,
    },


    # ========================================================
    # TEST 3
    # TOP FIVE CO2 ROOMS DURING MAY 2025
    # ========================================================

    {
        "intent": "rank_rooms",

        "metric_name": "co2",

        "aggregation": "mean",
        "frequency": "hourly",

        "time_scope": "range",

        "start_date": "2025-05-01",
        "end_date": "2025-05-31",

        "top_n": 5,

        "ascending": False,

        "minimum_records": 100,

        "continuous": False,
    },


    # ========================================================
    # TEST 4
    # CO2 TREND FOR SEMINAR ROOM 3
    # ========================================================

    {
        "intent": "room_trend",

        "metric_name": "co2",

        "aggregation": "mean",
        "frequency": "hourly",

        "time_scope": "all",

        "start_date": None,
        "end_date": None,

        "room": "Seminar Room 3",

        "continuous": False,
    },


    # ========================================================
    # TEST 5
    # NEW: RELATIONSHIP BETWEEN OCCUPANCY AND CO2
    # ========================================================

    {
        "intent": "relationship",

        "chart_type": "scatter",

        # ----------------------------------------------------
        # Relationship does not use metric_name.
        # ----------------------------------------------------

        "metric_name": None,

        "metric_x": "occupancy",
        "metric_y": "co2",

        # ----------------------------------------------------
        # No room means relationship across all rooms.
        # ----------------------------------------------------

        "room": None,
        "room_names": [],

        "aggregation": "mean",
        "frequency": "hourly",

        "start_date": None,
        "end_date": None,

        "top_n": 10,
        "ascending": False,

        "threshold": None,

        "time_scope": "all",

        "continuous": False,

        "threshold_operator": None,

        "minimum_records": 2,
    },


    # ========================================================
    # TEST 6
    # NEW: TEMPERATURE VS HUMIDITY
    # FOR ONE ROOM
    # ========================================================

    {
        "intent": "relationship",

        "chart_type": "scatter",

        "metric_name": None,

        "metric_x": "temp",
        "metric_y": "humidity",

        "room": "Seminar Room 3",

        "room_names": [
            "Seminar Room 3"
        ],

        "aggregation": "mean",
        "frequency": "hourly",

        "start_date": None,
        "end_date": None,

        "top_n": 10,
        "ascending": False,

        "threshold": None,

        "time_scope": "all",

        "continuous": False,

        "threshold_operator": None,

        "minimum_records": 2,
    },


    # ========================================================
    # TEST 7
    # NEW: CO2 VS TEMPERATURE DURING MAY
    # ========================================================

    {
        "intent": "relationship",

        "chart_type": "scatter",

        "metric_name": None,

        "metric_x": "co2",
        "metric_y": "temp",

        "room": None,
        "room_names": [],

        "aggregation": "mean",
        "frequency": "hourly",

        "start_date": "2025-05-01",
        "end_date": "2025-05-31",

        "top_n": 10,
        "ascending": False,

        "threshold": None,

        "time_scope": "range",

        "continuous": False,

        "threshold_operator": None,

        "minimum_records": 2,
    },
]


# ============================================================
# RUN QUERY BUILDER TESTS
# ============================================================

for number, request in enumerate(
    tests,
    start=1,
):

    print(
        "\n" + "=" * 75
    )

    print(
        f"TEST {number}"
    )

    print(
        "=" * 75
    )


    # ========================================================
    # PRINT REQUEST
    # ========================================================

    print(
        "\nREQUEST:"
    )

    print(
        request
    )


    # ========================================================
    # BUILD SQL
    # ========================================================

    try:

        result = builder.build(
            request
        )

    except Exception as error:

        print(
            "\nQUERY BUILD FAILED"
        )

        print(
            "Error:",
            error
        )

        continue


    # ========================================================
    # PRINT QUERY TYPE
    # ========================================================

    print(
        "\nQUERY TYPE:"
    )

    print(
        result[
            "query_type"
        ]
    )


    # ========================================================
    # PRINT RELATIONSHIP METRICS
    # ========================================================

    if (
        result[
            "query_type"
        ]
        == "relationship"
    ):

        print(
            "\nMETRIC X:"
        )

        print(
            result.get(
                "metric_x"
            )
        )


        print(
            "\nMETRIC Y:"
        )

        print(
            result.get(
                "metric_y"
            )
        )


    # ========================================================
    # PRINT SQL
    # ========================================================

    print(
        "\nSQL:"
    )

    print(
        result[
            "sql"
        ]
    )


    # ========================================================
    # PRINT PARAMETERS
    # ========================================================

    print(
        "\nPARAMETERS:"
    )

    print(
        result[
            "params"
        ]
    )


    # ========================================================
    # TEST RESULT
    # ========================================================

    print(
        "\nSTATUS:"
    )

    print(
        "QUERY BUILDER SUCCESS ✅"
    )