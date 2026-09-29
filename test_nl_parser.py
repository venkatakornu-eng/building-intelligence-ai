import json

from src.data_loader import load_room_data
from src.nl_parser import NaturalLanguageParser


data = load_room_data(
    "C:\\Users\\Santosh kumar\\OneDrive\\Documents\\Dissertation\\data\\processed/room_level_metrics.csv"
)

parser = NaturalLanguageParser(
    data=data,
    model="deepseek-r1:1.5b",
)


test_queries = [
    "Show the CO2 trend for Seminar Room 3.",

    (
        "Compare average CO2 in Seminar Room 2, "
        "Seminar Room 3 and The Hive."
    ),

    "Show the top five rooms with the highest average CO2.",

    "Which rooms have an average temperature above 21?",

    "Show the ten least occupied rooms.",
]


for query in test_queries:

    print("\n" + "=" * 70)
    print("USER QUERY:")
    print(query)
    print("=" * 70)

    try:
        structured_request = parser.parse(query)

        print(
            json.dumps(
                structured_request,
                indent=4,
            )
        )

    except Exception as error:
        print("Parser error:", error)