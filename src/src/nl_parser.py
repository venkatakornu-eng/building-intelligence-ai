from __future__ import annotations


import json
import urllib.error
import urllib.request
from typing import Any
import re
import pandas as pd


OLLAMA_URL = "http://localhost:11434/api/chat"


REQUEST_SCHEMA = {
    "type": "object",
    "properties": {
        "intent": {
            "type": "string",
            "enum": [
                "room_trend",
                "compare_rooms",
                "rank_rooms",
                "threshold",
            ],
        },
        "metric_name": {
            "type": "string",
            "enum": [
                "co2",
                "temp",
                "humidity",
                "occupancy",
            ],
        },
        "room": {
            "type": ["string", "null"],
        },
        "room_names": {
            "type": "array",
            "items": {
                "type": "string",
            },
        },
        "aggregation": {
            "type": "string",
            "enum": [
                "mean",
                "min",
                "max",
                "sum",
            ],
        },
        "frequency": {
            "type": "string",
            "enum": [
                "hourly",
                "daily",
            ],
        },
        "start_date": {
            "type": ["string", "null"],
        },
        "end_date": {
            "type": ["string", "null"],
        },
        "top_n": {
            "type": "integer",
            "minimum": 1,
        },
        "ascending": {
            "type": "boolean",
        },
        "threshold": {
            "type": ["number", "null"],
        },
        "minimum_records": {
            "type": "integer",
            "minimum": 1,
        },
        "chart_type": {
            "type": "string",
            "enum": [
                "line",
                "bar",
            ],
        },
    },
    "required": [
        "intent",
        "metric_name",
        "room",
        "room_names",
        "aggregation",
        "frequency",
        "start_date",
        "end_date",
        "top_n",
        "ascending",
        "threshold",
        "minimum_records",
        "chart_type",
    ],
    "additionalProperties": False,
}


class NaturalLanguageParser:
    """
    Convert a natural-language SmartViz question into
    a structured analytical request.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        model: str = "deepseek-r1:1.5b",
    ) -> None:

        if not isinstance(data, pd.DataFrame):
            raise TypeError(
                "Data must be provided as a pandas DataFrame."
            )

        if "display_name" not in data.columns:
            raise ValueError(
                "Data must contain the display_name column."
            )

        self.model = model

        self.room_names = sorted(
            data["display_name"]
            .dropna()
            .astype(str)
            .str.strip()
            .unique()
            .tolist()
        )

        self.room_lookup = {
            room.casefold(): room
            for room in self.room_names
        }

    def _system_prompt(self) -> str:
        rooms = "\n".join(
            f"- {room}"
            for room in self.room_names
        )

        return f"""
You are the natural-language interpretation component of a
SmartViz building analytics system.

Your only task is to convert the user's question into one
structured JSON request.

SUPPORTED INTENTS

1. room_trend
Use when the user asks about one room over time.
Examples:
- Show the CO2 trend for Seminar Room 3.
- Plot temperature over time for The Hive.

2. compare_rooms
Use when the user explicitly compares two or more rooms.
Examples:
- Compare CO2 in Seminar Room 2 and Seminar Room 3.
- Compare temperature across three selected rooms.

3. rank_rooms
Use for highest, lowest, best, worst, hottest, busiest,
least occupied or top-N questions.
Examples:
- Show the five rooms with the highest CO2.
- Which rooms are the least occupied?
- What are the hottest rooms?

4. threshold
Use when a numeric threshold is given.
Examples:
- Show rooms with average temperature above 21.
- Which rooms have average CO2 greater than 800?

METRIC MAPPING

- CO2, carbon dioxide -> co2
- temperature, hottest, warmest, coldest -> temp
- humidity, humid -> humidity
- occupancy, occupied, busy, busiest, utilised,
  utilization, utilisation -> occupancy

DEFAULT RULES

- aggregation: mean
- frequency: hourly
- top_n: 10
- minimum_records: 100
- room_trend chart: line
- all other charts: bar
- highest, hottest, busiest and worst:
  ascending = false
- lowest, coldest, least occupied and least utilised:
  ascending = true
- Use null for dates that were not supplied.
- Use null for threshold unless intent is threshold.
- Do not invent dates.
- Do not invent room names.
- Preserve room names exactly as shown below.
- Return only the JSON object.
- Do not include explanations or markdown.

AVAILABLE ROOMS

{rooms}
""".strip()

    def _send_request(
        self,
        user_query: str,
    ) -> dict[str, Any]:

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": self._system_prompt(),
                },
                {
                    "role": "user",
                    "content": user_query,
                },
            ],
            "stream": False,
            "format": REQUEST_SCHEMA,
            "options": {
                "temperature": 0,
            },
        }

        encoded_payload = json.dumps(
            payload
        ).encode("utf-8")

        request = urllib.request.Request(
            OLLAMA_URL,
            data=encoded_payload,
            headers={
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(
                request,
                timeout=120,
            ) as response:

                response_data = json.loads(
                    response.read().decode("utf-8")
                )

        except urllib.error.URLError as error:
            raise ConnectionError(
                "Could not connect to Ollama. Make sure "
                "Ollama is installed and running."
            ) from error

        except TimeoutError as error:
            raise TimeoutError(
                "The local model took too long to respond."
            ) from error

        message = response_data.get("message", {})
        content = message.get("content")

        if not content:
            raise ValueError(
                "Ollama returned an empty response."
            )

        try:
            parsed_request = json.loads(content)

        except json.JSONDecodeError as error:
            raise ValueError(
                "The model did not return valid JSON."
            ) from error

        if not isinstance(parsed_request, dict):
            raise ValueError(
                "The parsed request must be a JSON object."
            )

        return parsed_request

    def _match_room(
        self,
        room_name: Any,
    ) -> str | None:
        """
        Match model output to an exact dataset room name.
        """

        if room_name is None:
            return None

        cleaned_name = str(room_name).strip()

        if not cleaned_name:
            return None

        return self.room_lookup.get(
            cleaned_name.casefold()
        )

    def _clean_request(
        self,
        request: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Apply final deterministic normalisation after the LLM.
        """

        room = self._match_room(
            request.get("room")
        )

        raw_room_names = request.get(
            "room_names",
            [],
        )

        if not isinstance(raw_room_names, list):
            raw_room_names = []

        room_names = []

        for raw_room in raw_room_names:
            matched_room = self._match_room(raw_room)

            if (
                matched_room is not None
                and matched_room not in room_names
            ):
                room_names.append(matched_room)

        if room is not None and room not in room_names:
            room_names.insert(0, room)

        intent = str(
            request.get("intent", "")
        ).strip().casefold()

        if intent == "room_trend" and len(room_names) == 1:
            room = room_names[0]

        cleaned_request = {
            "intent": intent,
            "metric_name": str(
                request.get("metric_name", "")
            ).strip().casefold(),
            "room": room,
            "room_names": room_names,
            "aggregation": str(
                request.get("aggregation", "mean")
            ).strip().casefold(),
            "frequency": str(
                request.get("frequency", "hourly")
            ).strip().casefold(),
            "start_date": request.get("start_date"),
            "end_date": request.get("end_date"),
            "top_n": int(
                request.get("top_n", 10)
            ),
            "ascending": bool(
                request.get("ascending", False)
            ),
            "threshold": request.get("threshold"),
            "minimum_records": int(
                request.get("minimum_records", 100)
            ),
            "chart_type": str(
                request.get("chart_type", "bar")
            ).strip().casefold(),
        }

        return cleaned_request

    def _extract_rooms_from_query(
        self,
        user_query: str,
    ) -> list[str]:
        """
        Extract only room names explicitly written in the query.
        This prevents the LLM from inventing valid room names.
        """

        query = user_query.casefold()
        matches = []

        for room in self.room_names:
            pattern = (
                rf"(?<!\w)"
                rf"{re.escape(room.casefold())}"
                rf"(?!\w)"
            )

            match = re.search(pattern, query)

            if match:
                matches.append(
                    (
                        match.start(),
                        -len(room),
                        room,
                    )
                )

        matches.sort()

        rooms = []

        for _, _, room in matches:
            if room not in rooms:
                rooms.append(room)

        return rooms


    def _extract_metric_from_query(
        self,
        user_query: str,
    ) -> str | None:
        """Identify the metric using deterministic keywords."""

        query = (
            user_query
            .casefold()
            .replace("co₂", "co2")
        )

        if (
            re.search(r"\bco2\b", query)
            or "carbon dioxide" in query
        ):
            return "co2"

        if re.search(
            r"\btemperature\b|\btemp\b|"
            r"\bhottest\b|\bwarmest\b|\bcoldest\b",
            query,
        ):
            return "temp"

        if re.search(
            r"\bhumidity\b|\bhumid\b",
            query,
        ):
            return "humidity"

        if re.search(
            r"\boccupancy\b|\boccupied\b|"
            r"\bbusy\b|\bbusiest\b|"
            r"\butilised\b|\butilized\b|"
            r"\butilisation\b|\butilization\b",
            query,
        ):
            return "occupancy"

        return None


    def _extract_threshold(
        self,
        user_query: str,
    ) -> float | None:
        """Extract a numeric threshold following an above-type phrase."""

        query = user_query.casefold()

        pattern = (
            r"(?:above|over|greater\s+than|"
            r"more\s+than|higher\s+than|exceeding)"
            r"\s*(\d+(?:\.\d+)?)"
        )

        match = re.search(pattern, query)

        if match:
            return float(match.group(1))

        return None


    def _extract_top_n(
        self,
        user_query: str,
    ) -> int:
        """Extract ranking size from numbers or number words."""

        number_words = {
            "one": 1,
            "two": 2,
            "three": 3,
            "four": 4,
            "five": 5,
            "six": 6,
            "seven": 7,
            "eight": 8,
            "nine": 9,
            "ten": 10,
            "eleven": 11,
            "twelve": 12,
            "fifteen": 15,
            "twenty": 20,
        }

        query = user_query.casefold()

        digit_match = re.search(
            r"\b(?:top|bottom)?\s*(\d+)\b",
            query,
        )

        if digit_match:
            return int(digit_match.group(1))

        for word, value in number_words.items():
            if re.search(rf"\b{word}\b", query):
                return value

        return 10


    def _determine_intent(
        self,
        user_query: str,
        extracted_rooms: list[str],
        model_intent: str,
    ) -> str:
        """Correct the LLM intent using explicit query evidence."""

        query = user_query.casefold()

        threshold_words = [
            "above",
            "greater than",
            "more than",
            "higher than",
            "exceeding",
        ]

        if (
            any(word in query for word in threshold_words)
            and self._extract_threshold(user_query) is not None
        ):
            return "threshold"

        comparison_words = [
            "compare",
            "comparison",
            " versus ",
            " vs ",
        ]

        if (
            any(word in query for word in comparison_words)
            and len(extracted_rooms) >= 2
        ):
            return "compare_rooms"

        ranking_words = [
            "top",
            "bottom",
            "highest",
            "lowest",
            "hottest",
            "warmest",
            "coldest",
            "busiest",
            "least",
            "most",
            "best",
            "worst",
        ]

        if any(word in query for word in ranking_words):
            return "rank_rooms"

        trend_words = [
            "trend",
            "over time",
            "time series",
            "throughout",
        ]

        if (
            any(word in query for word in trend_words)
            and len(extracted_rooms) == 1
        ):
            return "room_trend"

        if len(extracted_rooms) == 1:
            return "room_trend"

        return model_intent


    def _apply_query_rules(
        self,
        user_query: str,
        request: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Correct the model output using deterministic query rules.
        """

        query = user_query.casefold()

        extracted_rooms = self._extract_rooms_from_query(
            user_query
        )

        extracted_metric = self._extract_metric_from_query(
            user_query
        )

        intent = self._determine_intent(
            user_query=user_query,
            extracted_rooms=extracted_rooms,
            model_intent=request["intent"],
        )

        request["intent"] = intent

        if extracted_metric is not None:
            request["metric_name"] = extracted_metric

        # Only retain rooms explicitly mentioned by the user
        if intent == "room_trend":
            request["room"] = (
                extracted_rooms[0]
                if len(extracted_rooms) == 1
                else None
            )

            request["room_names"] = (
                extracted_rooms[:1]
                if len(extracted_rooms) == 1
                else []
            )

        elif intent == "compare_rooms":
            request["room"] = None
            request["room_names"] = extracted_rooms

        else:
            # Ranking and threshold requests do not require rooms
            request["room"] = None
            request["room_names"] = []

        request["chart_type"] = (
            "line"
            if intent == "room_trend"
            else "bar"
        )

        if intent == "rank_rooms":
            request["top_n"] = self._extract_top_n(
                user_query
            )

            request["ascending"] = any(
                word in query
                for word in [
                    "least",
                    "lowest",
                    "coldest",
                    "bottom",
                ]
            )

        if intent == "threshold":
            request["threshold"] = self._extract_threshold(
                user_query
            )

        if re.search(r"\bdaily\b|\bper day\b", query):
            request["frequency"] = "daily"
        else:
            request["frequency"] = "hourly"

        if re.search(r"\bmaximum\b|\bmax\b", query):
            request["aggregation"] = "max"

        elif re.search(r"\bminimum\b|\bmin\b", query):
            request["aggregation"] = "min"

        elif re.search(r"\bsum\b|\btotal\b", query):
            request["aggregation"] = "sum"

        
        elif intent == "room_trend":
                request["top_n"] = 10
                request["ascending"] = False
                request["threshold"] = None

        elif intent == "compare_rooms":
                request["top_n"] = 10
                request["ascending"] = False
                request["threshold"] = None

        elif intent == "rank_rooms":
                request["threshold"] = None

        elif intent == "threshold":
                request["top_n"] = 10
                request["ascending"] = False    

        else:
            request["aggregation"] = "mean"

             

        return request

    def parse_baseline(
        self,
        user_query: str,
    ) -> dict[str, Any]:
        """
        Parse a question using the raw LLM output plus
        basic normalisation only.

        Deterministic correction rules are not applied.
        """

        if not isinstance(user_query, str):
            raise TypeError(
                "The user query must be a string."
            )

        user_query = user_query.strip()

        if not user_query:
            raise ValueError(
                "The user query cannot be empty."
            )

        raw_request = self._send_request(
            user_query
        )

        return self._clean_request(
            raw_request
        )

    def parse(
            self,
            user_query: str,
        ) -> dict[str, Any]:
            """
            Convert a natural-language question into a verified
            structured analytical request.
            """

            if not isinstance(user_query, str):
                raise TypeError(
                    "The user query must be a string."
                )

            user_query = user_query.strip()

            if not user_query:
                raise ValueError(
                    "The user query cannot be empty."
                )

            raw_request = self._send_request(
                user_query
            )

            cleaned_request = self._clean_request(
                raw_request
            )

            return self._apply_query_rules(
                user_query=user_query,
                request=cleaned_request,
            )