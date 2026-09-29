from src.pipeline import SmartVizPipeline


pipeline = SmartVizPipeline()


requests = {
    "co2_trend": {
        "intent": "room_trend",
        "metric_name": "co2",
        "room": "Seminar Room 3",
        "aggregation": "mean",
        "frequency": "hourly",
        "chart_type": "line",
    },

    "co2_ranking": {
        "intent": "rank_rooms",
        "metric_name": "co2",
        "aggregation": "mean",
        "frequency": "hourly",
        "top_n": 5,
        "ascending": False,
        "minimum_records": 100,
        "chart_type": "bar",
    },

    "room_comparison": {
        "intent": "compare_rooms",
        "metric_name": "co2",
        "room_names": [
            "Seminar Room 2",
            "Seminar Room 3",
            "The Hive",
        ],
        "aggregation": "mean",
        "frequency": "hourly",
        "chart_type": "bar",
    },

    "temperature_threshold": {
        "intent": "threshold",
        "metric_name": "temp",
        "threshold": 21,
        "aggregation": "mean",
        "frequency": "hourly",
        "minimum_records": 100,
        "chart_type": "bar",
    },

    "invalid_request": {
        "intent": "room_trend",
        "metric_name": "peopleCount",
        "room": "Unknown Room",
        "aggregation": "mean",
        "frequency": "hourly",
        "chart_type": "line",
    },
}


for test_name, request in requests.items():

    print("\n" + "=" * 60)
    print("TEST:", test_name)
    print("=" * 60)

    response = pipeline.run(
        request=request,
        chart_filename=f"{test_name}.html",
    )

    print("Success:", response["success"])
    print("Errors:", response["errors"])
    print("Warnings:", response["warnings"])
    print("Chart path:", response["chart_path"])

    result = response["result"]

    if result is not None:
        print("\nResult preview:")
        print(result.head())