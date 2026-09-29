from src.data_loader import load_room_data
from src.router import execute_request


data = load_room_data(
    "C:\\Users\\Santosh kumar\\OneDrive\\Documents\\Dissertation\\data\\processed/room_level_metrics.csv"
)


# -----------------------------------------------------
# Test 1: Room trend
# -----------------------------------------------------

trend_request = {
    "intent": "room_trend",
    "metric_name": "co2",
    "room": "Seminar Room 3",
    "aggregation": "mean",
    "frequency": "hourly",
    "chart_type": "line",
}


trend_response = execute_request(
    request=trend_request,
    data=data,
)


print("\nROOM TREND TEST")
print("Success:", trend_response["success"])
print("Errors:", trend_response["errors"])
print("Warnings:", trend_response["warnings"])

if trend_response["result"] is not None:
    print("Records:", len(trend_response["result"]))
    print(trend_response["result"].head())


# -----------------------------------------------------
# Test 2: Rank rooms
# -----------------------------------------------------

ranking_request = {
    "intent": "rank_rooms",
    "metric_name": "co2",
    "aggregation": "mean",
    "frequency": "hourly",
    "top_n": 5,
    "minimum_records": 100,
    "chart_type": "bar",
}


ranking_response = execute_request(
    request=ranking_request,
    data=data,
)


print("\nROOM RANKING TEST")
print("Success:", ranking_response["success"])
print("Errors:", ranking_response["errors"])

if ranking_response["result"] is not None:
    print(ranking_response["result"])


# -----------------------------------------------------
# Test 3: Invalid request
# -----------------------------------------------------

invalid_request = {
    "intent": "room_trend",
    "metric_name": "peopleCount",
    "room": "Unknown Room",
    "aggregation": "mean",
    "frequency": "hourly",
    "chart_type": "line",
}


invalid_response = execute_request(
    request=invalid_request,
    data=data,
)


print("\nINVALID REQUEST TEST")
print("Success:", invalid_response["success"])
print("Errors:")

for error in invalid_response["errors"]:
    print("-", error)


# -----------------------------------------------------
# Test 3: Compare selected rooms
# -----------------------------------------------------

comparison_request = {
    "intent": "compare_rooms",
    "metric_name": "co2",
    "room_names": [
        "Seminar Room 2",
        "Seminar Room 3",
        "The Hive"
    ],
    "aggregation": "mean",
    "frequency": "hourly",
    "chart_type": "bar",
}


comparison_response = execute_request(
    request=comparison_request,
    data=data,
)


print("\nROOM COMPARISON TEST")
print("Success:", comparison_response["success"])
print("Errors:", comparison_response["errors"])
print("Warnings:", comparison_response["warnings"])

if comparison_response["result"] is not None:
    print(comparison_response["result"])


# -----------------------------------------------------
# Test 4: Threshold analysis
# -----------------------------------------------------

threshold_request = {
    "intent": "threshold",
    "metric_name": "temp",
    "threshold": 21,
    "aggregation": "mean",
    "frequency": "hourly",
    "minimum_records": 100,
    "chart_type": "bar",
}


threshold_response = execute_request(
    request=threshold_request,
    data=data,
)


print("\nTHRESHOLD TEST")
print("Success:", threshold_response["success"])
print("Errors:", threshold_response["errors"])
print("Warnings:", threshold_response["warnings"])

if threshold_response["result"] is not None:
    print(threshold_response["result"])