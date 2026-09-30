from src.agents.intent_agent import IntentAgent
from src.agents.intent_guard import guard_intent


agent = IntentAgent(
    model="deepseek-r1:1.5b"
)


test_queries = [

    "Show humidity distribution across all rooms",

    "Show relationship between occupancy and CO2",

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
    print("QUERY:", query)
    print("=" * 70)

    result = agent.analyse(query)

    guarded = guard_intent(
        user_query=query,
        llm_intent=result["intent"],
    )

    print("Raw LLM Intent:", result["intent"])
    print("Final Intent:", guarded["intent"])
    print("Decision Source:", guarded["source"])
    print("Reason:", guarded["reason"])
    print("LLM Confidence:", result["confidence"])