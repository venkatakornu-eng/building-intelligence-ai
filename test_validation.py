from src.data_loader import load_room_data
from src.validation import validate_request


data = load_room_data(
    "C:\\Users\\Santosh kumar\\OneDrive\\Documents\\Dissertation\\data\\processed/room_level_metrics.csv"
)

request = {
    "intent": "room_trend",
    "metric_name": "co2",
    "room": "Seminar Room 3",
    "aggregation": "mean",
    "frequency": "hourly",
    "chart_type": "line",
}

result = validate_request(request, data)

print("Valid:", result["valid"])
print("Errors:", result["errors"])
print("Warnings:", result["warnings"])
print("Matching rows:", result["matching_rows"])