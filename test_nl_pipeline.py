from src.pipeline import SmartVizPipeline


pipeline = SmartVizPipeline(
    model="deepseek-r1:1.5b"
)


test_queries = {
    "co2_trend": (
        "Show the CO2 trend for Seminar Room 3."
    ),

    "co2_comparison": (
        "Compare average CO2 in Seminar Room 6, "
        "Seminar Room 3 and The Hive."
    ),

    "co2_top_five": (
        "Show the top five rooms with the highest "
        "average CO2."
    ),

    "temperature_threshold": (
        "Which rooms have an average temperature "
        "above 21?"
    ),

    "least_occupied": (
        "Show the ten least occupied rooms."
    ),

    "highest_occupied": (
        "Show the ten most occupied rooms."
    ),
}


for test_name, user_query in test_queries.items():

    print("\n" + "=" * 70)
    print("TEST:", test_name)
    print("QUESTION:", user_query)
    print("=" * 70)

    response = pipeline.run_natural_language(
        user_query=user_query,
        chart_filename=f"nl_{test_name}.html",
    )

    print("Success:", response["success"])
    print("Errors:", response["errors"])
    print("Warnings:", response["warnings"])
    print("Structured request:", response["request"])
    print("Chart path:", response["chart_path"])

    result = response["result"]

    if result is not None:
        print("\nResult preview:")
        print(result.head())