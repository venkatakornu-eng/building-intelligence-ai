from pathlib import Path

from src.data_loader import load_room_data
from src.router import execute_request
from src.visualization import create_figure, save_figure 


data = load_room_data(
    "C:\\Users\\Santosh kumar\\OneDrive\\Documents\\Dissertation\\data\\processed/room_level_metrics.csv"
)


requests = {
    "room_trend": {
        "intent": "room_trend",
        "metric_name": "co2",
        "room": "Seminar Room 3",
        "aggregation": "mean",
        "frequency": "hourly",
        "chart_type": "line",
    },

    "room_ranking": {
        "intent": "rank_rooms",
        "metric_name": "co2",
        "aggregation": "mean",
        "frequency": "hourly",
        "top_n": 5,
        "minimum_records": 100,
        "ascending": False,
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
}


output_directory = Path("outputs/charts")


for chart_name, request in requests.items():

    print(f"\nCREATING: {chart_name}")

    response = execute_request(
        request=request,
        data=data,
    )

    print("Success:", response["success"])
    print("Errors:", response["errors"])
    print("Warnings:", response["warnings"])

    if not response["success"]:
        continue

    figure = create_figure(response)

    saved_path = save_figure(
        figure,
        output_directory / f"{chart_name}.html",
    )

    print("Chart saved:", saved_path.resolve())