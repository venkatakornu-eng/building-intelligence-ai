from src.data_loader import load_room_data
from src.analytics import (
    room_metric_trend,
    compare_rooms,
    rank_rooms_by_metric,
    rooms_above_threshold,
    available_metrics_for_room,
    available_rooms_for_metric,
)


data = load_room_data(
    "C:\\Users\\Santosh kumar\\OneDrive\\Documents\\Dissertation\\data\\processed/room_level_metrics.csv"
)


trend = room_metric_trend(
    data,
    room_name="Seminar Room 3",
    metric_name="co2",
    aggregation="mean",
    frequency="hourly"
)

print("\nTrend records:", len(trend))
print(trend.head())


ranking = rank_rooms_by_metric(
    data,
    metric_name="co2",
    aggregation="mean",
    frequency="hourly",
    top_n=10
)

print("\nCO₂ room ranking:")
print(ranking)


comparison = compare_rooms(
    data,
    room_names=[
        "Seminar Room 2",
        "Seminar Room 3",
        "The Hive"
    ],
    metric_name="co2",
    aggregation="mean",
    frequency="hourly"
)

print("\nRoom comparison:")
print(comparison)


threshold_result = rooms_above_threshold(
    data,
    metric_name="temp",
    threshold=21,
    aggregation="mean",
    frequency="hourly"
)

print("\nRooms above 21:")
print(threshold_result)