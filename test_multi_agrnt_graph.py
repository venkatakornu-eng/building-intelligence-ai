from src.multi_agent_graph import (
    SmartVizMultiAgentGraph,
)


graph = SmartVizMultiAgentGraph(
    model="deepseek-r1:1.5b"
)


test_queries = [
    "Hottest room right now?",

    "Show the ten least occupied rooms.",

    (
        "Between 2025-05-01 and 2025-05-31 "
        "plot the top five worst spaces "
        "by average CO2."
    ),

    (
        "Rooms with continous high CO2 "
        "ordered by worst."
    )
]


for query in test_queries:

    print("\n" + "=" * 75)
    print("USER QUERY:")
    print(query)
    print("=" * 75)

    try:

        result = graph.run(
            query
        )

        print("\nINTENT STAGE")
        print(
            "Raw LLM Intent:",
            result["raw_intent"],
        )

        print(
            "Final Intent:",
            result["intent"],
        )

        print(
            "Confidence:",
            result["intent_confidence"],
        )

        print(
            "Intent Source:",
            result["intent_source"],
        )

        print(
            "Intent Reason:",
            result["intent_reason"],
        )

        print(
            "Chart Type:",
            result["chart_type"],
        )

        print("\nDATA MAPPING STAGE")

        print(
            "Mapping Source:",
            result["mapping_source"],
        )

        print(
            "Raw LLM Mapping:",
            result["raw_mapping"],
        )

        print("\nFINAL MAPPING")

        mapping = result["mapping"]

        print(
            "Metric:",
            mapping["metric_name"],
        )

        print(
            "Room:",
            mapping["room"],
        )

        print(
            "Rooms:",
            mapping["room_names"],
        )

        print(
            "Aggregation:",
            mapping["aggregation"],
        )

        print(
            "Frequency:",
            mapping["frequency"],
        )

        print(
            "Top N:",
            mapping["top_n"],
        )

        print(
            "Ascending:",
            mapping["ascending"],
        )

        print(
            "Threshold:",
            mapping["threshold"],
        )

        print(
            "Time Scope:",
            mapping["time_scope"],
        )

        print(
            "Continuous:",
            mapping["continuous"],
        )

        print(
            "Start Date:",
            mapping["start_date"],
        )

        print(
            "End Date:",
            mapping["end_date"],
        )

        print("\nREQUEST VALIDATION")

        validation = result[
            "validation"
        ]

        print(
            "Valid:",
            validation["valid"],
        )

        print(
            "Errors:",
            validation["errors"],
        )

        print(
            "Warnings:",
            validation["warnings"],
        )

        print(
            "Requires clarification:",
            validation[
                "requires_clarification"
            ],
        )


        if result.get(
            "query_ready",
            False,
        ):

            print("\nQUERY BUILDER")

            print(
                "Query Type:",
                result["query_type"],
            )

            print("\nSQL:")
            print(
                result["sql_query"]
            )

            print("\nSQL PARAMETERS:")
            print(
                result["sql_params"]
            )

        else:

            print(
                "\nSQL generation stopped "
                "because validation failed."
            )    

    except Exception as error:

        print(
            "\nGRAPH ERROR:",
            type(error).__name__,
            "-",
            error,
        )