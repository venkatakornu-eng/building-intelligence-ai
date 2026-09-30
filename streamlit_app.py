from __future__ import annotations

import re
import time
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from src.multi_agent_graph import SmartVizMultiAgentGraph


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="SmartViz AI Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "room_level_metrics.csv"
)


# ============================================================
# LOAD MULTI-AGENT SYSTEM
# ============================================================

@st.cache_resource
def load_system() -> SmartVizMultiAgentGraph:
    """Load the SmartViz LangGraph multi-agent pipeline once."""

    return SmartVizMultiAgentGraph()


# ============================================================
# LOAD AVAILABLE ROOMS
# ============================================================

@st.cache_data
def load_available_rooms() -> list[str]:
    """Load unique room names from the processed SmartViz dataset."""

    if not DATA_PATH.exists():
        return []

    dataframe = pd.read_csv(
        DATA_PATH,
        usecols=["display_name"],
    )

    return (
        dataframe["display_name"]
        .dropna()
        .astype(str)
        .str.strip()
        .drop_duplicates()
        .sort_values(
            key=lambda values: values.str.casefold()
        )
        .tolist()
    )


# ============================================================
# GENERIC HELPERS
# ============================================================

def safe_list(value: Any) -> list[Any]:
    """Return a list or an empty list."""

    if isinstance(value, list):
        return value

    return []


def normalise_text(value: str) -> str:
    """Lower-case text and collapse repeated whitespace."""

    return re.sub(
        r"\s+",
        " ",
        str(value).casefold().strip(),
    )


def fuzzy_similarity(
    text_a: str,
    text_b: str,
) -> float:
    """
    Return similarity between two pieces of text.

    Score:
        0.0 = completely different
        1.0 = identical
    """

    return SequenceMatcher(
        None,
        normalise_text(text_a),
        normalise_text(text_b),
    ).ratio()


def fuzzy_token_match(
    token: str,
    candidates: list[str],
    threshold: float = 0.80,
) -> bool:
    """
    Check whether a token is sufficiently similar to
    one of the supplied candidate words.
    """

    token = normalise_text(token)

    if not token:
        return False

    return any(
        fuzzy_similarity(
            token,
            candidate,
        ) >= threshold
        for candidate in candidates
    )


def extract_metric_from_text(
    user_query: str,
) -> str | None:
    """Extract one supported SmartViz metric from text."""

    query = (
        normalise_text(user_query)
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


def metric_display_name(
    metric_name: str | None,
) -> str:
    """Return a user-friendly metric label."""

    labels = {
        "co2": "CO2",
        "temp": "temperature",
        "humidity": "humidity",
        "occupancy": "occupancy",
    }

    if metric_name is None:
        return "metric"

    return labels.get(
        str(metric_name).casefold(),
        str(metric_name),
    )


def infer_threshold_direction(
    user_query: str,
) -> str | None:
    """Infer above/below only from explicit directional wording."""

    query = normalise_text(user_query)

    if re.search(
        r"\b(?:above|more\s+than|greater\s+than|"
        r"higher\s+than|exceed|exceeds|exceeding|high)\b",
        query,
    ):
        return "above"

    if re.search(
        r"\b(?:below|less\s+than|lower\s+than|under|low)\b",
        query,
    ):
        return "below"

    return None


# ============================================================
# GREETING DETECTION
# ============================================================

def is_greeting(user_query: str) -> bool:
    """Detect simple greetings handled directly by the UI."""

    query = normalise_text(user_query)

    return query in {
        "hi",
        "hello",
        "hey",
        "hiya",
        "hii",
        "hiii",
        "good morning",
        "good afternoon",
        "good evening",
    }


# ============================================================
# ROOM LIST REQUEST
# ============================================================

def is_room_list_request(
    user_query: str,
) -> bool:
    """
    Detect genuine room-catalogue requests.

    This supports:
    - normal natural-language room-list questions;
    - small spelling mistakes;
    - alternative wording.

    Analytical requests always take priority so queries such as:

        "show rooms with CO2 above 500"

    are not intercepted by the room-list handler.
    """

    query = normalise_text(
        user_query
    )

    query = (
        query
        .replace("co₂", "co2")
    )

    cleaned_query = re.sub(
        r"[?.!,;:]+$",
        "",
        query,
    ).strip()

    # ========================================================
    # 1. EXACT ANALYTICAL LANGUAGE HAS PRIORITY
    # ========================================================

    analytics_patterns = [

        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

        r"\bco2\b",
        r"\bcarbon\s+dioxide\b",
        r"\btemperature\b",
        r"\btemp\b",
        r"\bhumidity\b",
        r"\bhumid\b",
        r"\boccupancy\b",
        r"\boccupied\b",
        r"\bbusy\b",
        r"\bbusiest\b",
        r"\butilised\b",
        r"\butilized\b",
        r"\butilisation\b",
        r"\butilization\b",

        # ----------------------------------------------------
        # Statistics / ranking
        # ----------------------------------------------------

        r"\baverage\b",
        r"\bavg\b",
        r"\bmean\b",
        r"\bmaximum\b",
        r"\bminimum\b",
        r"\bmax\b",
        r"\bmin\b",
        r"\bhighest\b",
        r"\blowest\b",
        r"\btop\b",
        r"\bbottom\b",
        r"\bbest\b",
        r"\bworst\b",

        # ----------------------------------------------------
        # Time / trend
        # ----------------------------------------------------

        r"\btrend\b",
        r"\bover\s+time\b",
        r"\bcontinuous\b",

        # ----------------------------------------------------
        # Distribution
        # ----------------------------------------------------

        r"\bdistribution\b",
        r"\bhistogram\b",
        r"\bbox\s*plot\b",
        r"\bboxplot\b",

        # ----------------------------------------------------
        # Relationship
        # ----------------------------------------------------

        r"\bscatter\b",
        r"\brelationship\b",
        r"\bcorrelation\b",
        r"\bassociation\b",

        # ----------------------------------------------------
        # Threshold
        # ----------------------------------------------------

        r"\babove\b",
        r"\bbelow\b",
        r"\bmore\s+than\b",
        r"\bless\s+than\b",
        r"\bgreater\s+than\b",
        r"\blower\s+than\b",
        r"\bhigher\s+than\b",
        r"\bunder\b",
        r"\bexceed\b",
        r"\bexceeds\b",
        r"\bexceeded\b",
        r"\bexceeding\b",

        # ----------------------------------------------------
        # Comparison
        # ----------------------------------------------------

        r"\bcompare\b",
        r"\bcomparison\b",
        r"\bversus\b",
        r"\bvs\b",

        # ----------------------------------------------------
        # Latest/current
        # ----------------------------------------------------

        r"\bright\s+now\b",
        r"\blatest\b",
        r"\bcurrent\b",
        r"\bcurrently\b",
        r"\bmost\s+recent\b",
    ]

    if any(
        re.search(
            pattern,
            cleaned_query,
            flags=re.IGNORECASE,
        )
        for pattern in analytics_patterns
    ):
        return False

    # ========================================================
    # 2. TYPO-TOLERANT ANALYTICS PROTECTION
    # ========================================================
    #
    # Prevent something such as:
    #
    #   "show roms with coo2"
    #
    # from accidentally becoming a room-list request simply
    # because "room" is recognised.
    # ========================================================

    words = re.findall(
        r"[a-z0-9]+",
        cleaned_query,
    )

    analytical_keywords = [
        "co2",
        "temperature",
        "temp",
        "humidity",
        "occupancy",
        "average",
        "mean",
        "maximum",
        "minimum",
        "highest",
        "lowest",
        "trend",
        "distribution",
        "histogram",
        "relationship",
        "correlation",
        "compare",
        "above",
        "below",
        "continuous",
        "latest",
        "current",
    ]

    for word in words:
        # Very short words create too many fuzzy false positives.
        if len(word) < 3:
            continue

        if fuzzy_token_match(
            word,
            analytical_keywords,
            threshold=0.84,
        ):
            return False

    # A number usually indicates an analytical constraint rather
    # than a request for the complete room catalogue.
    #
    # Example:
    #   show roms above 500
    #
    if re.search(
        r"\d",
        cleaned_query,
    ):
        return False

    # ========================================================
    # 3. EXACT ROOM-LIST EXPRESSIONS
    # ========================================================

    room_list_patterns = [

        # ----------------------------------------------------
        # SHOW
        # ----------------------------------------------------

        r"show\s+me\s+available\s+rooms",
        r"show\s+available\s+rooms",

        r"show\s+me\s+all\s+available\s+rooms",
        r"show\s+all\s+available\s+rooms",

        r"show\s+me\s+the\s+available\s+rooms",
        r"show\s+the\s+available\s+rooms",

        r"show\s+me\s+all\s+available\s+room\s+names",
        r"show\s+all\s+available\s+room\s+names",

        r"show\s+me\s+available\s+room\s+names",
        r"show\s+available\s+room\s+names",

        r"show\s+me\s+all\s+room\s+names",
        r"show\s+all\s+room\s+names",

        r"show\s+me\s+room\s+names",
        r"show\s+room\s+names",

        r"show\s+me\s+all\s+rooms",
        r"show\s+all\s+rooms",

        r"show\s+me\s+rooms",
        r"show\s+rooms",

        # ----------------------------------------------------
        # GIVE
        # ----------------------------------------------------

        r"give\s+me\s+available\s+rooms",
        r"give\s+me\s+all\s+available\s+rooms",
        r"give\s+me\s+the\s+available\s+rooms",

        r"give\s+me\s+all\s+rooms",
        r"give\s+me\s+the\s+rooms",

        r"give\s+me\s+room\s+names",
        r"give\s+me\s+all\s+room\s+names",

        r"give\s+me\s+the\s+room\s+names",
        r"give\s+me\s+the\s+available\s+room\s+names",

        r"give\s+me\s+the\s+names\s+of\s+all\s+rooms",
        r"give\s+me\s+the\s+names\s+of\s+the\s+rooms",
        r"give\s+me\s+the\s+names\s+of\s+available\s+rooms",
        r"give\s+me\s+the\s+names\s+of\s+all\s+available\s+rooms",

        r"give\s+me\s+a\s+room\s+list",
        r"give\s+me\s+the\s+room\s+list",

        # ----------------------------------------------------
        # LIST
        # ----------------------------------------------------

        r"list\s+rooms",
        r"list\s+all\s+rooms",

        r"list\s+available\s+rooms",
        r"list\s+all\s+available\s+rooms",

        r"list\s+room\s+names",
        r"list\s+all\s+room\s+names",

        r"list\s+available\s+room\s+names",
        r"list\s+all\s+available\s+room\s+names",

        # ----------------------------------------------------
        # WHAT
        # ----------------------------------------------------

        r"what\s+rooms\s+are\s+available",
        r"what\s+room\s+names\s+are\s+available",

        r"what\s+are\s+the\s+available\s+rooms",
        r"what\s+are\s+all\s+available\s+rooms",

        r"what\s+are\s+the\s+room\s+names",
        r"what\s+are\s+all\s+room\s+names",

        r"what\s+are\s+the\s+available\s+room\s+names",
        r"what\s+are\s+all\s+available\s+room\s+names",

        r"what\s+are\s+the\s+names\s+of\s+the\s+rooms",
        r"what\s+are\s+the\s+names\s+of\s+all\s+rooms",

        r"what\s+are\s+the\s+names\s+of\s+the\s+available\s+rooms",
        r"what\s+are\s+the\s+names\s+of\s+all\s+available\s+rooms",

        # ----------------------------------------------------
        # WHICH
        # ----------------------------------------------------

        r"which\s+rooms\s+are\s+available",
        r"which\s+room\s+names\s+are\s+available",

        # ----------------------------------------------------
        # CAN / COULD YOU
        # ----------------------------------------------------

        r"can\s+you\s+show\s+me\s+available\s+rooms",
        r"can\s+you\s+show\s+me\s+all\s+available\s+rooms",

        r"can\s+you\s+show\s+me\s+room\s+names",
        r"can\s+you\s+show\s+me\s+all\s+room\s+names",

        r"can\s+you\s+list\s+available\s+rooms",
        r"can\s+you\s+list\s+all\s+available\s+rooms",

        r"could\s+you\s+show\s+me\s+available\s+rooms",
        r"could\s+you\s+show\s+me\s+all\s+available\s+rooms",

        r"could\s+you\s+list\s+available\s+rooms",
        r"could\s+you\s+list\s+all\s+available\s+rooms",

        # ----------------------------------------------------
        # SIMPLE
        # ----------------------------------------------------

        r"available\s+rooms",
        r"available\s+rooms\s+list",

        r"available\s+room\s+names",

        r"room\s+list",
        r"rooms\s+list",

        r"room\s+names",
        r"all\s+room\s+names",

        r"all\s+rooms",
        r"rooms",
    ]

    if any(
        re.fullmatch(
            pattern,
            cleaned_query,
            flags=re.IGNORECASE,
        )
        is not None
        for pattern in room_list_patterns
    ):
        return True

    # ========================================================
    # 4. FLEXIBLE NATURAL-LANGUAGE ROOM-LIST DETECTION
    # ========================================================

    has_room_word = bool(
        re.search(
            r"\brooms?\b",
            cleaned_query,
            flags=re.IGNORECASE,
        )
    )

    has_list_intent = bool(
        re.search(
            (
                r"\b("
                r"list|show|give|tell|names?|available|"
                r"what|which|display"
                r")\b"
            ),
            cleaned_query,
            flags=re.IGNORECASE,
        )
    )

    if (
        has_room_word
        and has_list_intent
    ):
        return True

    # ========================================================
    # 5. FULL-SENTENCE FUZZY TYPO HANDLING
    # ========================================================

    common_room_list_phrases = [
        "what are the room names",
        "what are all the room names",
        "what rooms are available",
        "what are the available rooms",
        "which rooms are available",
        "show me all available rooms",
        "show me all available room names",
        "show available rooms",
        "show room names",
        "give me all available rooms",
        "give me the names of all available rooms",
        "give me all room names",
        "list all rooms",
        "list available rooms",
        "list all available rooms",
        "list room names",
        "list all room names",
        "available rooms",
        "available room names",
        "room names",
        "room list",
    ]

    for phrase in common_room_list_phrases:

        similarity = fuzzy_similarity(
            cleaned_query,
            phrase,
        )

        if similarity >= 0.76:
            return True

    # ========================================================
    # 6. WORD-LEVEL FUZZY FALLBACK
    # ========================================================
    #
    # Handles things such as:
    #
    #   "what are the rom nams"
    #   "show me avilable roms"
    #
    # ========================================================

    room_keywords = [
        "room",
        "rooms",
    ]

    list_keywords = [
        "list",
        "show",
        "give",
        "display",
        "available",
        "name",
        "names",
        "what",
        "which",
    ]

    has_fuzzy_room = any(
        fuzzy_token_match(
            word,
            room_keywords,
            threshold=0.76,
        )
        for word in words
        if len(word) >= 3
    )

    has_fuzzy_list_intent = any(
        fuzzy_token_match(
            word,
            list_keywords,
            threshold=0.78,
        )
        for word in words
        if len(word) >= 3
    )

    # Keep this final fallback deliberately conservative.
    if (
        has_fuzzy_room
        and has_fuzzy_list_intent
        and len(words) <= 12
    ):
        return True

    return False


# ============================================================
# ROOM ALIASES
# ============================================================

def build_room_aliases(
    available_rooms: list[str],
) -> dict[str, str]:
    """Build safe aliases such as room 2 -> Seminar Room 2."""

    aliases: dict[str, str] = {
        room.casefold(): room
        for room in available_rooms
    }

    for room in available_rooms:
        if room.casefold() == "the hive":
            aliases["hive"] = room

    seminar_numbers: dict[str, list[str]] = {}

    for room in available_rooms:
        match = re.fullmatch(
            r"seminar\s+room\s+(\d+)",
            room.casefold(),
        )

        if match:
            seminar_numbers.setdefault(
                match.group(1),
                [],
            ).append(room)

    for number, matches in seminar_numbers.items():
        if len(matches) != 1:
            continue

        canonical = matches[0]
        aliases[f"room {number}"] = canonical
        aliases[f"seminar {number}"] = canonical

    return aliases


def expand_room_aliases(
    user_query: str,
    available_rooms: list[str],
) -> tuple[str, list[str]]:
    """Replace shorthand room names with canonical names."""

    aliases = build_room_aliases(
        available_rooms
    )

    expanded = user_query
    replacements: list[str] = []

    for alias, canonical in sorted(
        aliases.items(),
        key=lambda item: len(item[0]),
        reverse=True,
    ):
        if alias == canonical.casefold():
            continue

        canonical_pattern = (
            rf"(?<!\w){re.escape(canonical)}(?!\w)"
        )

        if re.search(
            canonical_pattern,
            expanded,
            flags=re.IGNORECASE,
        ):
            continue

        alias_pattern = (
            rf"(?<!\w){re.escape(alias)}(?!\w)"
        )

        if not re.search(
            alias_pattern,
            expanded,
            flags=re.IGNORECASE,
        ):
            continue

        expanded = re.sub(
            alias_pattern,
            canonical,
            expanded,
            flags=re.IGNORECASE,
        )

        replacements.append(
            f"{alias} → {canonical}"
        )

    return expanded, replacements


# ============================================================
# THRESHOLD LANGUAGE NORMALISATION
# ============================================================

def normalise_threshold_language(
    user_query: str,
) -> tuple[str, list[str], str | None]:
    """
    Normalise common threshold expressions.

    Standard non-continuous threshold queries use room-level
    average values in the current backend.

    Continuous queries are NOT silently converted to averages.
    """

    query = user_query.strip()
    interpretations: list[str] = []

    # --------------------------------------------------------
    # ABOVE FORMS - only when followed by a number
    # --------------------------------------------------------

    above_patterns = [
        r"\bmore\s+than\s+(?=-?\d)",
        r"\bgreater\s+than\s+(?=-?\d)",
        r"\bhigher\s+than\s+(?=-?\d)",
        r"\bover\s+(?=-?\d)",
        r"\bexceed\s+(?=-?\d)",
        r"\bexceeds\s+(?=-?\d)",
        r"\bexceeded\s+(?=-?\d)",
        r"\bexceeding\s+(?=-?\d)",
    ]

    for pattern in above_patterns:
        if re.search(
            pattern,
            query,
            flags=re.IGNORECASE,
        ):
            query = re.sub(
                pattern,
                "above ",
                query,
                count=1,
                flags=re.IGNORECASE,
            )

            interpretations.append(
                "Comparison interpreted as: above."
            )

            break

    # --------------------------------------------------------
    # BELOW FORMS
    # --------------------------------------------------------

    below_patterns = [
        r"\bless\s+than\s+(?=-?\d)",
        r"\blower\s+than\s+(?=-?\d)",
        r"\bunder\s+(?=-?\d)",
    ]

    for pattern in below_patterns:
        if re.search(
            pattern,
            query,
            flags=re.IGNORECASE,
        ):
            query = re.sub(
                pattern,
                "below ",
                query,
                count=1,
                flags=re.IGNORECASE,
            )

            interpretations.append(
                "Comparison interpreted as: below."
            )

            break

    # --------------------------------------------------------
    # > and < symbols
    # --------------------------------------------------------

    if re.search(
        r"(?<![<>])>(?!=)\s*(?=-?\d)",
        query,
    ):
        query = re.sub(
            r"(?<![<>])>(?!=)\s*(?=-?\d)",
            "above ",
            query,
            count=1,
        )

        interpretations.append(
            "Comparison symbol interpreted as: above."
        )

    elif re.search(
        r"(?<![<>])<(?!=)\s*(?=-?\d)",
        query,
    ):
        query = re.sub(
            r"(?<![<>])<(?!=)\s*(?=-?\d)",
            "below ",
            query,
            count=1,
        )

        interpretations.append(
            "Comparison symbol interpreted as: below."
        )

    has_threshold = bool(
        re.search(
            r"\b(?:above|below)\s+-?\d+(?:\.\d+)?",
            query,
            flags=re.IGNORECASE,
        )
    )

    if not has_threshold:
        return query, interpretations, None

    # --------------------------------------------------------
    # Any-reading / max request is not equivalent to AVG
    # --------------------------------------------------------

    non_average_patterns = [
        r"\bmaximum\b",
        r"\bmax\b",
        r"\bpeak\b",
        r"\bhighest\s+reading\b",
        r"\bhighest\s+value\b",
        r"\bindividual\s+reading\b",
        r"\bindividual\s+value\b",
        r"\bany\s+reading\b",
        r"\bany\s+value\b",
        r"\bat\s+any\s+time\b",
        r"\bat\s+some\s+point\b",
        r"\bever\b",
    ]

    if any(
        re.search(
            pattern,
            query,
            flags=re.IGNORECASE,
        )
        for pattern in non_average_patterns
    ):
        interpretations.append(
            "This request asks for an individual, maximum, "
            "peak or any-time reading."
        )

        return (
            query,
            interpretations,
            (
                "The current SmartViz threshold mode evaluates "
                "room-level average values. Maximum, peak or "
                "any-time threshold analysis is not yet implemented."
            ),
        )

    # --------------------------------------------------------
    # Continuous is a distinct analytical meaning.
    # --------------------------------------------------------

    if re.search(
        r"\bcontinuous\b",
        query,
        flags=re.IGNORECASE,
    ):
        interpretations.append(
            "Continuous threshold wording retained without "
            "converting it to an average threshold."
        )

        return query, interpretations, None

    # --------------------------------------------------------
    # Standard threshold backend uses AVG(value)
    # --------------------------------------------------------

    if re.search(
        r"\b(?:average|avg|mean)\b",
        query,
        flags=re.IGNORECASE,
    ):
        interpretations.append(
            "Threshold evaluated using the room-level "
            "average metric value."
        )

        return query, interpretations, None

    metric_patterns = [
        (r"\bco2\b", "CO2"),
        (r"\bco₂\b", "CO2"),
        (r"\bcarbon\s+dioxide\b", "CO2"),
        (r"\btemperature\b", "temperature"),
        (r"\btemp\b", "temperature"),
        (r"\bhumidity\b", "humidity"),
        (r"\boccupancy\b", "occupancy"),
    ]

    for pattern, canonical_metric in metric_patterns:
        match = re.search(
            pattern,
            query,
            flags=re.IGNORECASE,
        )

        if not match:
            continue

        query = (
            query[:match.start()]
            + f"average {canonical_metric}"
            + query[match.end():]
        )

        interpretations.append(
            (
                "Threshold evaluated using the room-level "
                f"average {canonical_metric} value."
            )
        )

        break

    return query, interpretations, None


# ============================================================
# CLARIFICATION / CONVERSATION CONTEXT
# ============================================================

def build_clarification_message(
    user_query: str,
    state: dict[str, Any],
) -> dict[str, Any] | None:
    """
    Convert selected validation failures into helpful
    clarification requests and optionally create pending
    conversational context.
    """

    errors = safe_list(
        state.get("errors")
    )

    error_text = " ".join(
        str(error)
        for error in errors
    ).casefold()

    mapping = (
        state.get("mapping")
        or state.get("validated_request")
        or {}
    )

    if not isinstance(mapping, dict):
        mapping = {}

    metric_name = (
        mapping.get("metric_name")
        or extract_metric_from_text(user_query)
    )

    metric_label = metric_display_name(
        metric_name
    )

    direction = (
        infer_threshold_direction(user_query)
        or "above"
    )

    query_lower = normalise_text(
        user_query
    )

    # --------------------------------------------------------
    # Continuous high/low query missing a number
    # --------------------------------------------------------

    if (
        "continuous" in query_lower
        and (
            "requires an approved numeric threshold"
            in error_text
            or "requires a numeric threshold"
            in error_text
        )
    ):
        return {
            "title": "More information needed",
            "message": (
                f"I can interpret the continuous {metric_label} request, "
                "but I need a numeric threshold to define the condition."
            ),
            "reason": (
                "SmartViz does not invent analytical thresholds. "
                "Your next numeric-only reply will be treated as the "
                "threshold for this request."
            ),
            "example": "For example, reply with: 500",
            "pending": {
                "type": "numeric_threshold",
                "original_query": user_query,
                "metric": metric_name,
                "direction": direction,
                "continuous": True,
            },
        }

    # --------------------------------------------------------
    # General threshold query missing a number
    # --------------------------------------------------------

    if (
        "requires a numeric threshold"
        in error_text
        or "requires an approved numeric threshold"
        in error_text
    ):
        return {
            "title": "Numeric threshold required",
            "message": (
                f"Please provide a numeric threshold for {metric_label}."
            ),
            "reason": (
                "Your next numeric-only reply will be connected to "
                "this request instead of being treated as a new question."
            ),
            "example": "For example, reply with: 500",
            "pending": {
                "type": "numeric_threshold",
                "original_query": user_query,
                "metric": metric_name,
                "direction": direction,
                "continuous": False,
            },
        }

    # --------------------------------------------------------
    # Number exists but direction is missing
    # --------------------------------------------------------

    if (
        "requires an above/below comparison direction"
        in error_text
    ):
        threshold_value = mapping.get("threshold")

        if threshold_value is None:
            match = re.search(
                r"-?\d+(?:\.\d+)?",
                user_query,
            )

            if match:
                threshold_value = match.group(0)

        return {
            "title": "Comparison direction required",
            "message": (
                "Please specify whether the metric should be "
                "above or below the threshold."
            ),
            "reason": (
                "Your next reply can simply be 'above' or 'below'."
            ),
            "example": "For example, reply with: above",
            "pending": {
                "type": "threshold_direction",
                "original_query": user_query,
                "metric": metric_name,
                "threshold": threshold_value,
            },
        }

    return None


def resolve_pending_followup(
    user_query: str,
) -> tuple[str, list[str], bool]:
    """
    Resolve a short reply against a previous clarification.

    Example:
        Previous:
            Rooms with continuous high CO2 order by worst

        Reply:
            500

        Internal reconstructed request:
            Rooms with continuous CO2 above 500,
            ordered by worst.
    """

    pending = st.session_state.get(
        "pending_clarification"
    )

    if not isinstance(pending, dict):
        return user_query, [], False

    pending_type = pending.get(
        "type"
    )

    # --------------------------------------------------------
    # Numeric-only threshold reply
    # --------------------------------------------------------

    if pending_type == "numeric_threshold":

        match = re.fullmatch(
            r"\s*(-?\d+(?:\.\d+)?)\s*",
            user_query,
        )

        if match:

            threshold_value = match.group(1)

            metric_name = pending.get(
                "metric"
            )

            metric_label = metric_display_name(
                metric_name
            )

            direction = (
                pending.get("direction")
                or "above"
            )

            continuous = bool(
                pending.get(
                    "continuous",
                    False,
                )
            )

            if continuous:

                rebuilt_query = (
                    f"Rooms with continuous {metric_label} "
                    f"{direction} {threshold_value}, "
                    "ordered by worst."
                )

            else:

                rebuilt_query = (
                    f"Show rooms with average {metric_label} "
                    f"{direction} {threshold_value}."
                )

            st.session_state.pending_clarification = None

            return (
                rebuilt_query,
                [
                    (
                        f"Used {threshold_value} as the threshold "
                        "for your previous request."
                    )
                ],
                True,
            )

    # --------------------------------------------------------
    # Direction-only reply
    # --------------------------------------------------------

    if pending_type == "threshold_direction":

        direction = normalise_text(
            user_query
        )

        if direction in {
            "above",
            "below",
        }:

            metric_name = pending.get(
                "metric"
            )

            metric_label = metric_display_name(
                metric_name
            )

            threshold_value = pending.get(
                "threshold"
            )

            if threshold_value is not None:

                rebuilt_query = (
                    f"Show rooms with average {metric_label} "
                    f"{direction} {threshold_value}."
                )

                st.session_state.pending_clarification = None

                return (
                    rebuilt_query,
                    [
                        (
                            f"Used '{direction}' as the comparison "
                            "direction for your previous request."
                        )
                    ],
                    True,
                )

    # --------------------------------------------------------
    # User started a different query.
    # --------------------------------------------------------

    st.session_state.pending_clarification = None

    return user_query, [], False


def continuous_backend_notice(
    user_query: str,
) -> str | None:
    """
    Prevent a known unsupported continuous query from reaching
    QueryBuilderAgent, which currently rejects continuous=True.
    """

    query = normalise_text(
        user_query
    )

    if "continuous" not in query:
        return None

    if not re.search(
        r"\b(?:above|below)\s+-?\d+(?:\.\d+)?",
        query,
    ):
        return None

    return (
        "The threshold has been understood, but continuous/persistent "
        "SQL analysis is not yet implemented in the current SmartViz "
        "Query Builder. The request was not converted into an average "
        "threshold because that would change its meaning."
    )


# ============================================================
# CHART DISPLAY
# ============================================================

def get_chart_path(
    state: dict[str, Any],
) -> Path | None:
    """Locate generated Plotly HTML."""

    possible_paths = [
        state.get("chart_path"),
        state.get("final_chart_path"),
    ]

    visualization = state.get(
        "visualization"
    )

    if isinstance(
        visualization,
        dict,
    ):
        possible_paths.append(
            visualization.get(
                "chart_path"
            )
        )

    for raw_path in possible_paths:

        if not raw_path:
            continue

        path = Path(
            str(raw_path)
        )

        if path.exists():
            return path

    return None


def display_chart(
    state: dict[str, Any],
) -> None:
    """Display the generated Plotly HTML chart."""

    chart_created = bool(
        state.get(
            "chart_created",
            False,
        )
    )

    if not chart_created:

        if not bool(
            state.get(
                "result_has_data",
                False,
            )
        ):
            st.info(
                "No chart was generated because no matching data were found."
            )

        return

    chart_path = get_chart_path(
        state
    )

    if chart_path is None:

        st.warning(
            "The chart was generated, but the chart file could not be located."
        )

        return

    try:

        html = chart_path.read_text(
            encoding="utf-8"
        )

        components.html(
            html,
            height=600,
            scrolling=False,
        )

    except Exception as error:

        st.warning(
            "The chart exists but could not be displayed."
        )

        with st.expander(
            "Chart display error"
        ):
            st.code(
                f"{type(error).__name__}: {error}"
            )


# ============================================================
# ANALYTICAL DATA TABLE
# ============================================================

def display_analytical_data(
    state: dict[str, Any],
) -> None:
    """Display stored SQL results in an expandable table."""

    rows = safe_list(
        state.get(
            "sql_results"
        )
    )

    if not rows:
        return

    dataframe = pd.DataFrame(
        rows
    )

    if dataframe.empty:
        return

    # --------------------------------------------------------
    # Friendly timestamps
    # --------------------------------------------------------

    for column in [
        "start_time",
        "end_time",
        "latest_time",
        "latest_start_time",
    ]:

        if column in dataframe.columns:

            dataframe[column] = pd.to_datetime(
                dataframe[column],
                errors="coerce",
            )

    # --------------------------------------------------------
    # Round numeric output
    # --------------------------------------------------------

    numeric_columns = dataframe.select_dtypes(
        include="number"
    ).columns

    for column in numeric_columns:

        dataframe[column] = (
            dataframe[column]
            .round(4)
        )

    maximum_display_rows = 100

    display_dataframe = dataframe.head(
        maximum_display_rows
    )

    with st.expander(
        "📋 View analytical data"
    ):

        st.dataframe(
            display_dataframe,
            use_container_width=True,
            hide_index=True,
        )

        stored_rows = len(
            dataframe
        )

        total_rows = state.get(
            "sql_row_count",
            stored_rows,
        )

        if stored_rows > maximum_display_rows:

            st.caption(
                (
                    f"Showing the first {maximum_display_rows:,} "
                    f"of {stored_rows:,} stored rows."
                )
            )

        elif (
            isinstance(
                total_rows,
                int,
            )
            and total_rows > stored_rows
        ):

            st.caption(
                (
                    f"Showing {stored_rows:,} stored rows from "
                    f"{total_rows:,} total query rows."
                )
            )

        else:

            st.caption(
                f"{stored_rows:,} analytical row(s)."
            )

        if bool(
            state.get(
                "sql_result_truncated",
                False,
            )
        ):

            st.warning(
                "The SQL result was truncated before being stored in application state."
            )


# ============================================================
# INSIGHT DISPLAY
# ============================================================

def display_insight(
    state: dict[str, Any],
) -> None:
    """Display the final guarded analytical insight."""

    summary = str(
        state.get(
            "insight_summary",
            "",
        )
        or ""
    ).strip()

    key_points = safe_list(
        state.get(
            "insight_key_points"
        )
    )

    caution = str(
        state.get(
            "insight_caution",
            "",
        )
        or ""
    ).strip()

    generated_by = state.get(
        "insight_generated_by"
    )

    st.markdown(
        "### 💡 Insight"
    )

    if summary:

        st.write(
            summary
        )

    else:

        st.info(
            "No textual insight was generated."
        )

    for point in key_points:

        point_text = str(
            point
        ).strip()

        if point_text:

            st.markdown(
                f"- {point_text}"
            )

    if caution:

        st.warning(
            caution
        )

    if generated_by:

        with st.expander(
            "Insight generation details"
        ):

            st.write(
                "**Generated by:**",
                generated_by,
            )

            st.write(
                "**Guard fallback used:**",
                bool(
                    state.get(
                        "insight_used_fallback",
                        False,
                    )
                ),
            )


# ============================================================
# TECHNICAL DETAILS
# ============================================================

def display_technical_details(
    state: dict[str, Any],
) -> None:
    """Display optional multi-agent processing details."""

    with st.expander(
        "🔎 Multi-Agent Processing Details"
    ):

        col1, col2, col3, col4 = st.columns(
            4
        )

        with col1:

            st.metric(
                "Intent",
                str(
                    state.get(
                        "intent",
                        "N/A",
                    )
                ),
            )

        with col2:

            st.metric(
                "Query Type",
                str(
                    state.get(
                        "query_type",
                        "N/A",
                    )
                ),
            )

        with col3:

            chart_type = (
                state.get(
                    "final_chart_type"
                )
                or state.get(
                    "chart_type"
                )
                or "N/A"
            )

            st.metric(
                "Chart Type",
                str(
                    chart_type
                ),
            )

        with col4:

            latency = state.get(
                "latency_seconds"
            )

            latency_display = (
                f"{latency:.2f} s"
                if isinstance(
                    latency,
                    (int, float),
                )
                else "N/A"
            )

            st.metric(
                "End-to-End Latency",
                latency_display,
            )

        st.markdown(
            "#### Data Mapping"
        )

        mapping = (
            state.get(
                "mapping"
            )
            or state.get(
                "validated_request"
            )
        )

        if mapping:

            st.json(
                mapping
            )

        else:

            st.caption(
                "No mapping information available."
            )

        st.markdown(
            "#### Result Validation"
        )

        left, right = st.columns(
            2
        )

        with left:

            st.write(
                "**Result status:**",
                state.get(
                    "result_status",
                    "N/A",
                ),
            )

            st.write(
                "**Valid result:**",
                bool(
                    state.get(
                        "result_valid",
                        False,
                    )
                ),
            )

        with right:

            st.write(
                "**Rows returned:**",
                state.get(
                    "sql_row_count",
                    0,
                ),
            )

            st.write(
                "**Has data:**",
                bool(
                    state.get(
                        "result_has_data",
                        False,
                    )
                ),
            )

        st.write(
            "**SQL execution successful:**",
            bool(
                state.get(
                    "execution_success",
                    False,
                )
            ),
        )

        warnings = safe_list(
            state.get(
                "warnings"
            )
        )

        if warnings:

            st.markdown(
                "#### ⚠ Validation and Guard Notes"
            )

            for warning in warnings:

                st.write(
                    f"- {warning}"
                )

        errors = safe_list(
            state.get(
                "errors"
            )
        )

        if errors:

            st.markdown(
                "#### ❌ Errors"
            )

            for error in errors:

                st.write(
                    f"- {error}"
                )


# ============================================================
# ROOM LIST DISPLAY
# ============================================================

def display_room_list(
    available_rooms: list[str],
) -> None:
    """Display available rooms in three columns."""

    st.markdown(
        "### 🏢 Available rooms"
    )

    if not available_rooms:

        st.warning(
            "No room names could be loaded from the SmartViz dataset."
        )

        return

    st.write(
        f"**{len(available_rooms)} rooms are available:**"
    )

    columns = st.columns(
        3
    )

    for index, room in enumerate(
        available_rooms
    ):

        with columns[
            index % 3
        ]:

            st.markdown(
                f"- {room}"
            )

    st.caption(
        "Use these names in analytical questions for the most precise room matching."
    )


# ============================================================
# LOAD ROOM NAMES
# ============================================================

AVAILABLE_ROOMS = load_available_rooms()


# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "pending_clarification" not in st.session_state:
    st.session_state.pending_clarification = None


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title(
        "🏢 SmartViz"
    )

    st.caption(
        "Multi-Agent Building Analytics"
    )

    st.divider()

    st.markdown(
        """
### System

This prototype uses a LangGraph-based multi-agent architecture
to transform natural-language questions into:

- structured analytical requests
- SQL queries
- validated results
- visualisations
- guarded analytical insights
"""
    )

    st.divider()

    st.markdown(
        "### Supported metrics"
    )

    st.write(
        "• CO₂ value"
    )

    st.write(
        "• Temperature"
    )

    st.write(
        "• Humidity"
    )

    st.write(
        "• Occupancy"
    )

    st.divider()

    st.markdown(
        "### Example questions"
    )

    examples = [
        "Show the temperature trend for Seminar Room 3.",
        "Show the top 5 rooms by average CO2.",
        "Which room has the highest CO2 right now?",
        "Show rooms with average CO2 above 500.",
        "Show the humidity distribution of top 4 rooms.",
        "Show the relationship between occupancy and CO2 across all rooms.",
        "Show me all available room names.",
    ]

    for example in examples:

        st.caption(
            f"• {example}"
        )

    st.divider()

    st.caption(
        "Threshold queries currently use room-level average metric values."
    )

    st.caption(
        "Latest/current queries use the latest available historical sensor reading, not live real-time data."
    )

    st.caption(
        "Continuous/persistent SQL analysis is not yet implemented in the current Query Builder."
    )

    if st.button(
        "🗑️ Clear conversation",
        use_container_width=True,
    ):

        st.session_state.messages = []
        st.session_state.pending_clarification = None

        st.rerun()


# ============================================================
# PAGE HEADER
# ============================================================

st.title(
    "📊 SmartViz AI Building Analytics"
)

st.markdown(
    """
Ask a question about building sensor data using natural language.
The multi-agent system will interpret the request, query the
SmartViz database, generate a suitable visualisation and provide
a validated analytical insight.
"""
)

st.divider()


# ============================================================
# DISPLAY PREVIOUS CONVERSATION
# ============================================================

for message in st.session_state.messages:

    role = message.get(
        "role",
        "assistant",
    )

    message_type = message.get(
        "type",
        "text",
    )

    with st.chat_message(
        role
    ):

        if message_type == "text":

            st.markdown(
                message.get(
                    "content",
                    "",
                )
            )

        elif message_type == "room_list":

            display_room_list(
                AVAILABLE_ROOMS
            )

        elif message_type == "analysis":

            state = message.get(
                "state"
            )

            if isinstance(
                state,
                dict,
            ):

                display_chart(
                    state
                )

                display_analytical_data(
                    state
                )

                display_insight(
                    state
                )

                display_technical_details(
                    state
                )


# ============================================================
# CHAT INPUT
# ============================================================

user_query = st.chat_input(
    "Ask about rooms, CO2, temperature, humidity or occupancy..."
)


# ============================================================
# PROCESS NEW QUERY
# ============================================================

if user_query:

    raw_user_query = user_query

    # --------------------------------------------------------
    # Resolve short follow-up such as "500"
    # --------------------------------------------------------

    (
        processing_query,
        context_messages,
        used_previous_context,
    ) = resolve_pending_followup(
        raw_user_query
    )

    # --------------------------------------------------------
    # Save/display exactly what the user typed
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "type": "text",
            "content": raw_user_query,
        }
    )

    with st.chat_message(
        "user"
    ):

        st.markdown(
            raw_user_query
        )

    # ========================================================
    # ROUTE 1: GREETING
    # ========================================================

    if (
        not used_previous_context
        and is_greeting(
            raw_user_query
        )
    ):

        response = (
            "Hi! Ask me about CO2, temperature, humidity, "
            "occupancy, room rankings, trends, distributions "
            "or relationships between building metrics."
        )

        with st.chat_message(
            "assistant"
        ):

            st.write(
                response
            )

        st.session_state.messages.append(
            {
                "role": "assistant",
                "type": "text",
                "content": response,
            }
        )

    # ========================================================
    # ROUTE 2: ROOM LIST
    # ========================================================

    elif (
        not used_previous_context
        and is_room_list_request(
            raw_user_query
        )
    ):

        with st.chat_message(
            "assistant"
        ):

            display_room_list(
                AVAILABLE_ROOMS
            )

        st.session_state.messages.append(
            {
                "role": "assistant",
                "type": "room_list",
                "content": "",
            }
        )

    # ========================================================
    # ROUTE 3: SMARTVIZ ANALYTICS
    # ========================================================

    else:

        # ----------------------------------------------------
        # Room aliases
        # ----------------------------------------------------

        expanded_query, replacements = (
            expand_room_aliases(
                user_query=processing_query,
                available_rooms=AVAILABLE_ROOMS,
            )
        )

        replacements = (
            context_messages
            + replacements
        )

        # ----------------------------------------------------
        # Threshold language normalisation
        # ----------------------------------------------------

        (
            expanded_query,
            threshold_interpretations,
            unsupported_threshold_reason,
        ) = normalise_threshold_language(
            expanded_query
        )

        replacements.extend(
            threshold_interpretations
        )

        with st.chat_message(
            "assistant"
        ):

            # ------------------------------------------------
            # Show SmartViz interpretation
            # ------------------------------------------------

            if replacements:

                with st.expander(
                    "SmartViz query interpretation"
                ):

                    st.write(
                        "SmartViz interpreted:"
                    )

                    for replacement in replacements:

                        st.write(
                            f"- {replacement}"
                        )

                    st.caption(
                        f"Processed query: {expanded_query}"
                    )

            # =================================================
            # UNSUPPORTED MAX / ANY-TIME THRESHOLD
            # =================================================

            if unsupported_threshold_reason:

                st.warning(
                    unsupported_threshold_reason
                )

                suggestion = (
                    "Supported alternative: "
                    "**Show rooms with average CO2 above 500.**"
                )

                st.info(
                    suggestion
                )

                st.session_state.pending_clarification = None

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "type": "text",
                        "content": (
                            unsupported_threshold_reason
                            + "\n\n"
                            + suggestion
                        ),
                    }
                )

            # =================================================
            # CONTINUOUS QUERY WITH THRESHOLD
            # =================================================

            elif continuous_backend_notice(
                expanded_query
            ):

                notice = continuous_backend_notice(
                    expanded_query
                )

                st.warning(
                    notice
                )

                metric_name = extract_metric_from_text(
                    expanded_query
                )

                metric_label = metric_display_name(
                    metric_name
                )

                threshold_match = re.search(
                    r"\b(?:above|below)\s+(-?\d+(?:\.\d+)?)",
                    expanded_query,
                    flags=re.IGNORECASE,
                )

                threshold_value = (
                    threshold_match.group(1)
                    if threshold_match
                    else "<threshold>"
                )

                direction_match = re.search(
                    r"\b(above|below)\s+-?\d+(?:\.\d+)?",
                    expanded_query,
                    flags=re.IGNORECASE,
                )

                threshold_direction = (
                    direction_match.group(1).casefold()
                    if direction_match
                    else "above"
                )

                alternative = (
                    "If you want the currently supported "
                    "average-threshold analysis instead, try: "
                    f"**Show rooms with average {metric_label} "
                    f"{threshold_direction} {threshold_value}.**"
                )

                st.info(
                    alternative
                )

                st.session_state.pending_clarification = None

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "type": "text",
                        "content": (
                            str(notice)
                            + "\n\n"
                            + alternative
                        ),
                    }
                )

            # =================================================
            # RUN MULTI-AGENT PIPELINE
            # =================================================

            else:

                try:

                    system = load_system()

                    with st.spinner(
                        "SmartViz agents are analysing your request..."
                    ):

                        start_time = time.perf_counter()

                        state = system.run(
                            expanded_query
                        )

                        latency_seconds = (
                            time.perf_counter()
                            - start_time
                        )

                        state[
                            "latency_seconds"
                        ] = latency_seconds

                    result_valid = bool(
                        state.get(
                            "result_valid",
                            False,
                        )
                    )

                    result_has_data = bool(
                        state.get(
                            "result_has_data",
                            False,
                        )
                    )

                    errors = safe_list(
                        state.get(
                            "errors"
                        )
                    )

                    # ========================================
                    # INVALID REQUEST
                    # ========================================

                    if not result_valid:

                        clarification = (
                            build_clarification_message(
                                user_query=expanded_query,
                                state=state,
                            )
                        )

                        if clarification:

                            st.warning(
                                "💬 "
                                + clarification[
                                    "title"
                                ]
                            )

                            st.write(
                                clarification[
                                    "message"
                                ]
                            )

                            st.caption(
                                clarification[
                                    "reason"
                                ]
                            )

                            st.markdown(
                                "**Try:**"
                            )

                            st.code(
                                clarification[
                                    "example"
                                ],
                                language=None,
                            )

                            pending = clarification.get(
                                "pending"
                            )

                            st.session_state.pending_clarification = (
                                pending
                                if isinstance(
                                    pending,
                                    dict,
                                )
                                else None
                            )

                            assistant_content = (
                                f"**{clarification['title']}**\n\n"
                                f"{clarification['message']}\n\n"
                                f"{clarification['reason']}\n\n"
                                f"Try: `{clarification['example']}`"
                            )

                            assistant_message = {
                                "role": "assistant",
                                "type": "text",
                                "content": assistant_content,
                            }

                        else:

                            st.session_state.pending_clarification = None

                            st.error(
                                "SmartViz could not complete this analytical request."
                            )

                            if errors:

                                with st.expander(
                                    "Error details"
                                ):

                                    for error in errors:

                                        st.write(
                                            f"- {error}"
                                        )

                            assistant_message = {
                                "role": "assistant",
                                "type": "text",
                                "content": (
                                    "SmartViz could not complete "
                                    "this analytical request."
                                ),
                            }

                    # ========================================
                    # VALID REQUEST
                    # ========================================

                    else:

                        st.session_state.pending_clarification = None

                        if result_has_data:

                            st.success(
                                "Analysis completed successfully."
                            )

                        else:

                            st.info(
                                "The request was processed successfully, "
                                "but no matching data were found."
                            )

                        if (
                            state.get(
                                "query_type"
                            )
                            == "threshold"
                        ):

                            st.info(
                                (
                                    "Threshold interpretation: the result "
                                    "is based on the room-level average "
                                    "metric value over the selected data period."
                                )
                            )

                        display_chart(
                            state
                        )

                        display_analytical_data(
                            state
                        )

                        display_insight(
                            state
                        )

                        display_technical_details(
                            state
                        )

                        assistant_message = {
                            "role": "assistant",
                            "type": "analysis",
                            "state": state,
                            "content": "",
                        }

                    st.session_state.messages.append(
                        assistant_message
                    )

                # ============================================
                # UNEXPECTED APPLICATION ERROR
                # ============================================

                except Exception as error:

                    st.session_state.pending_clarification = None

                    error_message = (
                        f"{type(error).__name__}: {error}"
                    )

                    st.error(
                        "The SmartViz pipeline encountered an unexpected error."
                    )

                    with st.expander(
                        "Technical error"
                    ):

                        st.code(
                            error_message
                        )

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "type": "text",
                            "content": (
                                "The SmartViz pipeline encountered "
                                "an unexpected error."
                            ),
                        }
                    )