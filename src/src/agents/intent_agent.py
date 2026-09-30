from __future__ import annotations

from typing import Any

from src.agents.base_agent import BaseOllamaAgent


# ============================================================
# STRUCTURED OUTPUT SCHEMA
# ============================================================

INTENT_SCHEMA = {
    "type": "object",
    "properties": {
        "intent": {
            "type": "string",
            "enum": [
                "room_trend",
                "compare_rooms",
                "rank_rooms",
                "threshold",
                "distribution",
                "relationship",
            ],
        },
        "confidence": {
            "type": "number",
            "minimum": 0,
            "maximum": 1,
        },
    },
    "required": [
        "intent",
        "confidence",
    ],
    "additionalProperties": False,
}


# ============================================================
# INTENT AGENT
# ============================================================

class IntentAgent(BaseOllamaAgent):
    """
    LLM-based SmartViz intent classifier.

    The LLM proposes an intent.

    A deterministic Intent Guard later checks and
    corrects the prediction where explicit language
    makes the user intent unambiguous.
    """

    SUPPORTED_INTENTS = {
        "room_trend",
        "compare_rooms",
        "rank_rooms",
        "threshold",
        "distribution",
        "relationship",
    }

    def __init__(
        self,
        model: str = "deepseek-r1:1.5b",
    ) -> None:

        super().__init__(
            model=model
        )

    # ========================================================
    # SYSTEM PROMPT
    # ========================================================

    def _system_prompt(
        self,
    ) -> str:

        return """
You are the Intent Agent in a SmartViz building
analytics multi-agent system.

Classify the user's request into EXACTLY ONE
of the following analytical intents.


============================================================
1. room_trend
============================================================

Use when the user wants values for ONE room
over time.

Examples:

"Show CO2 trend for Seminar Room 3"
-> room_trend

"Plot temperature over time for Cafe"
-> room_trend

"Show humidity history for Teaching Room 5"
-> room_trend


============================================================
2. compare_rooms
============================================================

Use when the user explicitly wants to compare
TWO OR MORE rooms using ONE metric.

Examples:

"Compare CO2 in room 2 and room 3"
-> compare_rooms

"Compare average humidity in Cafe and Dining Room"
-> compare_rooms


============================================================
3. rank_rooms
============================================================

Use when the user wants rooms ranked or wants
the highest / lowest / hottest / busiest room.

Examples:

"Show the 5 rooms with highest CO2"
-> rank_rooms

"Which room is hottest?"
-> rank_rooms

"Which rooms are least occupied?"
-> rank_rooms


============================================================
4. threshold
============================================================

Use when the user specifies a NUMERIC condition
or limit.

Examples:

"Rooms above 21 degrees"
-> threshold

"Show rooms with CO2 above 1000"
-> threshold

"Rooms below 30 percent occupancy"
-> threshold


============================================================
5. distribution
============================================================

Use when the user wants the statistical
distribution of ONE metric.

This includes:

- histogram
- box plot
- boxplot
- distribution
- spread
- outliers
- frequency distribution

Examples:

"Show humidity distribution across all rooms"
-> distribution

"Show histogram of CO2 for Seminar Room 3"
-> distribution

"Show box plot of temperature for Cafe"
-> distribution


============================================================
6. relationship
============================================================

Use when the user wants to examine the
relationship, association or correlation
between TWO DIFFERENT metrics.

This includes scatter plots.

Examples:

"Show relationship between occupancy and CO2"
-> relationship

"Show scatter plot of temperature and humidity"
-> relationship

"Is occupancy associated with CO2?"
-> relationship

"Show correlation between temperature and CO2"
-> relationship

"Plot occupancy against temperature"
-> relationship


============================================================
IMPORTANT DISTINCTIONS
============================================================

Comparing ROOMS using ONE metric:

"Compare CO2 in Cafe and Dining Room"
-> compare_rooms


Comparing / relating TWO METRICS:

"Show relationship between CO2 and occupancy"
-> relationship


A box plot or histogram:

"Show distribution of humidity"
-> distribution


A numeric condition:

"Rooms with humidity above 60"
-> threshold


Do not extract metrics.

Do not extract rooms.

Do not select a chart.

Do not generate SQL.

Do not explain the result.

Return only the structured JSON.
""".strip()

    # ========================================================
    # ANALYSE
    # ========================================================

    def analyse(
        self,
        user_query: str,
    ) -> dict[str, Any]:

        if not isinstance(
            user_query,
            str,
        ):

            raise TypeError(
                "User query must be a string."
            )

        user_query = (
            user_query.strip()
        )

        if not user_query:

            raise ValueError(
                "User query cannot be empty."
            )

        result = self._request_json(
            system_prompt=
                self._system_prompt(),

            user_query=
                user_query,

            schema=
                INTENT_SCHEMA,
        )

        intent = str(
            result.get(
                "intent",
                "",
            )
        ).strip()

        if intent not in self.SUPPORTED_INTENTS:

            raise ValueError(
                "Intent Agent returned an "
                f"unsupported intent: {intent}"
            )

        try:

            confidence = float(
                result.get(
                    "confidence",
                    0.0,
                )
            )

        except (
            TypeError,
            ValueError,
        ):

            confidence = 0.0

        confidence = max(
            0.0,
            min(
                1.0,
                confidence,
            ),
        )

        return {
            "intent":
                intent,

            "confidence":
                confidence,
        }