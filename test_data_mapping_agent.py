import pandas as pd

from src.agents.data_mapping_agent import DataMappingAgent
from src.agents.data_mapping_guard import guard_mapping
from src.agents.intent_agent import IntentAgent
from src.agents.intent_guard import guard_intent
from src.agents.request_validation_agent import (
    RequestValidationAgent,
)


# ============================================================
# DATA PATH
# ============================================================

DATA_PATH = (
    "C:\\Users\\Santosh kumar\\OneDrive\\Documents\\"
    "Dissertation\\data\\processed\\room_level_metrics.csv"
)


# ============================================================
# LOAD AVAILABLE ROOM NAMES
# ============================================================

room_data = pd.read_csv(
    DATA_PATH,
    usecols=["display_name"],
)

available_rooms = (
    room_data["display_name"]
    .dropna()
    .astype(str)
    .unique()
    .tolist()
)


# ============================================================
# INITIALISE AGENTS
# ============================================================

intent_agent = IntentAgent(
    model="deepseek-r1:1.5b"
)

mapping_agent = DataMappingAgent(
    available_rooms=available_rooms,
    model="deepseek-r1:1.5b",
)

validation_agent = RequestValidationAgent(
    available_rooms=available_rooms
)


# ============================================================
# TEST QUERIES
# ============================================================

test_queries = [

    # --------------------------------------------------------
    # RELATIONSHIP - VALID
    # --------------------------------------------------------

    "relationship between occupancy and CO2",

    "scatter plot of temperature and humidity",

    (
        "relationship between CO2 and temperature "
        "in Seminar Room 3"
    ),

    # --------------------------------------------------------
    # RELATIONSHIP - INVALID ON PURPOSE
    # --------------------------------------------------------
    # Should fail because only one metric is supplied.
    # --------------------------------------------------------

    "scatter plot of CO2",

    # --------------------------------------------------------
    # EXISTING QUERIES
    # --------------------------------------------------------

    "Hottest room right now?",

    "Room with highest CO2?",

    "Busiest room right now?",

    "Rooms with continuous high CO2 ordered by worst.",

    (
        "Rooms with average temperature above 21 "
        "ordered by worst."
    ),

    "Show the ten less utilised rooms on average.",

    (
        "Between 2025-05-01 and 2025-05-31 "
        "plot the top five worst spaces by average CO2."
    ),
]


# ============================================================
# RUN TESTS
# ============================================================

for query in test_queries:

    print(
        "\n" + "=" * 75
    )

    print(
        "QUERY:",
        query
    )

    print(
        "=" * 75
    )

    # ========================================================
    # 1. INTENT AGENT
    # ========================================================

    try:

        intent_result = (
            intent_agent.analyse(
                query
            )
        )

        raw_intent = (
            intent_result[
                "intent"
            ]
        )

        intent_confidence = (
            intent_result[
                "confidence"
            ]
        )

    except TimeoutError:

        print(
            "WARNING: Intent Agent timed out. "
            "Using deterministic Intent Guard fallback."
        )

        raw_intent = "unknown"
        intent_confidence = 0.0


    # ========================================================
    # 2. INTENT GUARD
    # ========================================================

    intent_guard_result = (
        guard_intent(
            user_query=query,
            llm_intent=raw_intent,
        )
    )

    final_intent = (
        intent_guard_result[
            "intent"
        ]
    )


    # ========================================================
    # 3. DATA MAPPING AGENT
    # ========================================================

    try:

        raw_mapping = (
            mapping_agent.analyse(
                user_query=query,
                intent=final_intent,
            )
        )

        mapping_source = "llm"

    except TimeoutError:

        print(
            "WARNING: Data Mapping Agent timed out. "
            "Using safe deterministic fallback."
        )

        raw_mapping = {

            "metric_name": None,

            # Relationship metrics
            "metric_x": None,
            "metric_y": None,

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
        }

        mapping_source = (
            "timeout_fallback"
        )


    # ========================================================
    # 4. DATA MAPPING GUARD
    # ========================================================

    mapping = (
        guard_mapping(
            user_query=query,
            intent=final_intent,
            llm_mapping=raw_mapping,
            available_rooms=available_rooms,
        )
    )


    # ========================================================
    # 5. SELECT CHART TYPE
    # ========================================================

    if final_intent == "room_trend":

        chart_type = "line"

    elif final_intent == "distribution":

        # Default distribution chart for testing.
        # VisualizationAgent can later choose box/histogram
        # based on explicit user wording.
        chart_type = "histogram"

    elif final_intent == "relationship":

        chart_type = "scatter"

    else:

        chart_type = "bar"


    # ========================================================
    # 6. REQUEST VALIDATION
    # ========================================================

    validation = (
        validation_agent.validate(
            user_query=query,
            intent=final_intent,
            chart_type=chart_type,
            mapping=mapping,
        )
    )


    # ========================================================
    # 7. PRINT INTENT RESULTS
    # ========================================================

    print(
        "\nINTENT RESULT"
    )

    print(
        "Raw LLM Intent:",
        raw_intent
    )

    print(
        "Final Intent:",
        final_intent
    )

    print(
        "Intent Source:",
        intent_guard_result.get(
            "source"
        )
    )

    print(
        "Intent Reason:",
        intent_guard_result.get(
            "reason"
        )
    )

    print(
        "LLM Confidence:",
        intent_confidence
    )


    # ========================================================
    # 8. PRINT RAW LLM MAPPING
    # ========================================================

    print(
        "\nRAW LLM MAPPING"
    )

    print(
        raw_mapping
    )


    # ========================================================
    # 9. PRINT FINAL GUARDED MAPPING
    # ========================================================

    print(
        "\nFINAL GUARDED MAPPING"
    )

    print(
        "Mapping Source:",
        mapping_source
    )

    print(
        "Intent:",
        final_intent
    )

    print(
        "Metric:",
        mapping.get(
            "metric_name"
        )
    )

    print(
        "Metric X:",
        mapping.get(
            "metric_x"
        )
    )

    print(
        "Metric Y:",
        mapping.get(
            "metric_y"
        )
    )

    print(
        "Room:",
        mapping.get(
            "room"
        )
    )

    print(
        "Rooms:",
        mapping.get(
            "room_names"
        )
    )

    print(
        "Aggregation:",
        mapping.get(
            "aggregation"
        )
    )

    print(
        "Frequency:",
        mapping.get(
            "frequency"
        )
    )

    print(
        "Top N:",
        mapping.get(
            "top_n"
        )
    )

    print(
        "Ascending:",
        mapping.get(
            "ascending"
        )
    )

    print(
        "Threshold:",
        mapping.get(
            "threshold"
        )
    )

    print(
        "Time Scope:",
        mapping.get(
            "time_scope"
        )
    )

    print(
        "Continuous:",
        mapping.get(
            "continuous"
        )
    )

    print(
        "Start Date:",
        mapping.get(
            "start_date"
        )
    )

    print(
        "End Date:",
        mapping.get(
            "end_date"
        )
    )


    # ========================================================
    # 10. PRINT REQUEST VALIDATION
    # ========================================================

    print(
        "\nREQUEST VALIDATION"
    )

    print(
        "Chart Type:",
        chart_type
    )

    print(
        "Valid:",
        validation.get(
            "valid"
        )
    )

    print(
        "Requires Clarification:",
        validation.get(
            "requires_clarification"
        )
    )

    print(
        "Errors:",
        validation.get(
            "errors"
        )
    )

    print(
        "Warnings:",
        validation.get(
            "warnings"
        )
    )

    print(
        "Validated Request:"
    )

    print(
        validation.get(
            "request"
        )
    )