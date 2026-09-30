from __future__ import annotations

from typing import Any

from src.agents.base_agent import BaseOllamaAgent


# ============================================================
# SUPPORTED METRICS
# ============================================================

SUPPORTED_METRICS = {
    "co2",
    "temp",
    "humidity",
    "occupancy",
}


# ============================================================
# DATA MAPPING JSON SCHEMA
# ============================================================

DATA_MAPPING_SCHEMA = {
    "type": "object",
    "properties": {

        # ----------------------------------------------------
        # Single-metric requests
        # ----------------------------------------------------

        "metric_name": {
            "type": [
                "string",
                "null",
            ],
            "enum": [
                "co2",
                "temp",
                "humidity",
                "occupancy",
                None,
            ],
        },

        # ----------------------------------------------------
        # Two-metric relationship requests
        # ----------------------------------------------------

        "metric_x": {
            "type": [
                "string",
                "null",
            ],
            "enum": [
                "co2",
                "temp",
                "humidity",
                "occupancy",
                None,
            ],
        },

        "metric_y": {
            "type": [
                "string",
                "null",
            ],
            "enum": [
                "co2",
                "temp",
                "humidity",
                "occupancy",
                None,
            ],
        },

        # ----------------------------------------------------
        # Rooms
        # ----------------------------------------------------

        "room": {
            "type": [
                "string",
                "null",
            ],
        },

        "room_names": {
            "type": "array",
            "items": {
                "type": "string",
            },
        },

        # ----------------------------------------------------
        # Aggregation
        # ----------------------------------------------------

        "aggregation": {
            "type": "string",
            "enum": [
                "mean",
                "min",
                "max",
                "sum",
            ],
        },

        # ----------------------------------------------------
        # Frequency
        # ----------------------------------------------------

        "frequency": {
            "type": "string",
            "enum": [
                "hourly",
                "daily",
            ],
        },

        # ----------------------------------------------------
        # Dates
        # ----------------------------------------------------

        "start_date": {
            "type": [
                "string",
                "null",
            ],
        },

        "end_date": {
            "type": [
                "string",
                "null",
            ],
        },

        # ----------------------------------------------------
        # Ranking
        # ----------------------------------------------------

        "top_n": {
            "type": "integer",
            "minimum": 1,
        },

        "ascending": {
            "type": "boolean",
        },

        # ----------------------------------------------------
        # Threshold
        # ----------------------------------------------------

        "threshold": {
            "type": [
                "number",
                "null",
            ],
        },

        # ----------------------------------------------------
        # Time scope
        # ----------------------------------------------------

        "time_scope": {
            "type": "string",
            "enum": [
                "all",
                "latest",
                "range",
            ],
        },

        # ----------------------------------------------------
        # Continuous
        # ----------------------------------------------------

        "continuous": {
            "type": "boolean",
        },
    },

    "required": [
        "metric_name",
        "metric_x",
        "metric_y",
        "room",
        "room_names",
        "aggregation",
        "frequency",
        "start_date",
        "end_date",
        "top_n",
        "ascending",
        "threshold",
        "time_scope",
        "continuous",
    ],

    "additionalProperties": False,
}


# ============================================================
# DATA MAPPING AGENT
# ============================================================

class DataMappingAgent(BaseOllamaAgent):
    """
    Extract analytical entities from SmartViz
    natural-language requests.

    Single-metric requests use:
        metric_name

    Relationship/scatter requests use:
        metric_x
        metric_y
    """

    def __init__(
        self,
        available_rooms: list[str],
        model: str = "deepseek-r1:1.5b",
    ) -> None:

        super().__init__(
            model=model
        )

        self.available_rooms = sorted(
            {
                str(room).strip()
                for room in available_rooms
                if str(room).strip()
            }
        )

        self.room_lookup = {
            room.casefold(): room
            for room
            in self.available_rooms
        }

    # ========================================================
    # SYSTEM PROMPT
    # ========================================================

    def _system_prompt(
        self,
        intent: str,
    ) -> str:

        rooms = "\n".join(
            f"- {room}"
            for room
            in self.available_rooms
        )

        return f"""
You are the Data Mapping Agent in a SmartViz
building analytics multi-agent system.

The analytical intent has already been classified.

CONFIRMED INTENT:
{intent}


============================================================
SUPPORTED METRICS
============================================================

co2
- CO2
- carbon dioxide

temp
- temperature
- hottest
- warmest
- coldest

humidity
- humidity
- humid

occupancy
- occupancy
- occupied
- busiest
- busy
- utilisation
- utilization
- utilised
- utilized


============================================================
SINGLE-METRIC RULE
============================================================

For these intents:

room_trend
compare_rooms
rank_rooms
threshold
distribution

use:

metric_name = requested metric
metric_x = null
metric_y = null


Example:

"Show CO2 trend for Seminar Room 3"

metric_name = "co2"
metric_x = null
metric_y = null


============================================================
RELATIONSHIP RULE
============================================================

For relationship requests ONLY:

metric_name = null

Extract exactly TWO explicitly mentioned metrics.

The first metric mentioned becomes metric_x.

The second metric mentioned becomes metric_y.


Example:

"Show relationship between occupancy and CO2"

metric_name = null
metric_x = "occupancy"
metric_y = "co2"


Example:

"Show scatter plot of temperature and humidity"

metric_name = null
metric_x = "temp"
metric_y = "humidity"


Example:

"Show relationship between CO2 and temperature"

metric_name = null
metric_x = "co2"
metric_y = "temp"


Never invent a second metric.

If only one metric is explicitly given:

metric_x = that metric
metric_y = null


============================================================
AVAILABLE ROOM NAMES
============================================================

{rooms}


============================================================
ROOM RULES
============================================================

Only return rooms explicitly mentioned by the user.

Never invent rooms.

Use exact spelling from AVAILABLE ROOM NAMES.

One room:

room = exact room
room_names = [exact room]


Multiple rooms:

room = null
room_names = [rooms]


For relationship requests:

A room is OPTIONAL.

Example:

"Relationship between CO2 and temperature
in Seminar Room 3"

room = "Seminar Room 3"

If no room is mentioned:

room = null
room_names = []


============================================================
AGGREGATION
============================================================

average / avg / mean
-> mean

minimum / min
-> min

maximum / max
-> max

total / sum
-> sum

Default:
mean


============================================================
FREQUENCY
============================================================

daily / per day
-> daily

Default:
hourly


============================================================
TOP N
============================================================

Examples:

top five
-> 5

top 10
-> 10

ten least occupied
-> 10

Default:
10


============================================================
RANKING DIRECTION
============================================================

ascending = true for:

least
lowest
coldest
bottom
less utilised
less utilized
least occupied


ascending = false for:

highest
hottest
warmest
busiest
top
worst
most occupied


============================================================
THRESHOLD
============================================================

Extract a threshold ONLY when the user gives
an explicit numeric threshold.

Examples:

above 21
-> 21

greater than 800
-> 800

below 20
-> 20

Otherwise:
threshold = null


============================================================
TIME SCOPE
============================================================

right now
now
currently
current
latest

-> latest


Explicit start and end date:

-> range


Otherwise:

-> all


============================================================
DATES
============================================================

Only return dates explicitly stated by the user.

Do not invent dates.

Use YYYY-MM-DD where possible.

No dates:

start_date = null
end_date = null


============================================================
CONTINUOUS
============================================================

continuous
continuously
persistent
persistently
sustained
consecutive

-> continuous = true


Otherwise:

continuous = false


============================================================
IMPORTANT
============================================================

Do not alter the confirmed intent.

Do not generate SQL.

Do not calculate correlations.

Do not generate insights.

Do not select chart axes beyond metric_x and metric_y.

Do not invent metrics.

Return only structured JSON.
""".strip()

    # ========================================================
    # CANONICAL METRIC
    # ========================================================

    def _canonical_metric(
        self,
        value: Any,
    ) -> str | None:

        if value is None:
            return None

        metric = (
            str(value)
            .strip()
            .casefold()
        )

        if metric in SUPPORTED_METRICS:
            return metric

        return None

    # ========================================================
    # CANONICAL ROOM
    # ========================================================

    def _canonical_room(
        self,
        value: Any,
    ) -> str | None:

        if value is None:
            return None

        value = (
            str(value)
            .strip()
        )

        if not value:
            return None

        return self.room_lookup.get(
            value.casefold()
        )

    # ========================================================
    # ANALYSE
    # ========================================================

    def analyse(
        self,
        user_query: str,
        intent: str,
    ) -> dict[str, Any]:

        if not isinstance(
            user_query,
            str,
        ):

            raise TypeError(
                "The user query must be a string."
            )

        user_query = (
            user_query.strip()
        )

        if not user_query:

            raise ValueError(
                "The user query cannot be empty."
            )

        result = self._request_json(
            system_prompt=
                self._system_prompt(
                    intent=intent
                ),

            user_query=
                user_query,

            schema=
                DATA_MAPPING_SCHEMA,
        )

        # ====================================================
        # METRICS
        # ====================================================

        metric_name = (
            self._canonical_metric(
                result.get(
                    "metric_name"
                )
            )
        )

        metric_x = (
            self._canonical_metric(
                result.get(
                    "metric_x"
                )
            )
        )

        metric_y = (
            self._canonical_metric(
                result.get(
                    "metric_y"
                )
            )
        )

        # ----------------------------------------------------
        # Relationship uses TWO metric fields only
        # ----------------------------------------------------

        if intent == "relationship":

            metric_name = None

        # ----------------------------------------------------
        # Every other intent uses ONE metric only
        # ----------------------------------------------------

        else:

            metric_x = None
            metric_y = None

        # ====================================================
        # ROOM
        # ====================================================

        room = (
            self._canonical_room(
                result.get(
                    "room"
                )
            )
        )

        room_names: list[str] = []

        for returned_room in result.get(
            "room_names",
            [],
        ):

            canonical = (
                self._canonical_room(
                    returned_room
                )
            )

            if (
                canonical is not None
                and canonical not in room_names
            ):

                room_names.append(
                    canonical
                )

        # ====================================================
        # INTENT-SPECIFIC ROOM STRUCTURE
        # ====================================================

        if intent == "room_trend":

            if (
                room is None
                and len(room_names) == 1
            ):

                room = room_names[0]

            if room is not None:

                room_names = [
                    room
                ]

        elif intent == "compare_rooms":

            room = None

        elif intent in {
            "rank_rooms",
            "threshold",
        }:

            room = None
            room_names = []

        elif intent in {
            "distribution",
            "relationship",
        }:

            if (
                room is None
                and len(room_names) == 1
            ):

                room = room_names[0]

            if room is not None:

                room_names = [
                    room
                ]

        # ====================================================
        # TOP N
        # ====================================================

        try:

            top_n = int(
                result.get(
                    "top_n",
                    10,
                )
            )

        except (
            TypeError,
            ValueError,
        ):

            top_n = 10

        if top_n < 1:
            top_n = 10

        # ====================================================
        # THRESHOLD
        # ====================================================

        threshold = result.get(
            "threshold"
        )

        if threshold is not None:

            try:

                threshold = float(
                    threshold
                )

            except (
                TypeError,
                ValueError,
            ):

                threshold = None

        # ====================================================
        # RETURN
        # ====================================================

        return {
            "metric_name":
                metric_name,

            "metric_x":
                metric_x,

            "metric_y":
                metric_y,

            "room":
                room,

            "room_names":
                room_names,

            "aggregation":
                result.get(
                    "aggregation",
                    "mean",
                ),

            "frequency":
                result.get(
                    "frequency",
                    "hourly",
                ),

            "start_date":
                result.get(
                    "start_date"
                ),

            "end_date":
                result.get(
                    "end_date"
                ),

            "top_n":
                top_n,

            "ascending":
                bool(
                    result.get(
                        "ascending",
                        False,
                    )
                ),

            "threshold":
                threshold,

            "time_scope":
                result.get(
                    "time_scope",
                    "all",
                ),

            "continuous":
                bool(
                    result.get(
                        "continuous",
                        False,
                    )
                ),
        }