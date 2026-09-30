from __future__ import annotations

import re
from typing import Any


# ============================================================
# INTENT GUARD
# ============================================================

def guard_intent(
    user_query: str,
    llm_intent: str,
) -> dict[str, Any]:
    """
    Deterministically validate and correct the LLM intent.

    Explicit analytical language takes priority over
    the raw LLM prediction.
    """

    if not isinstance(
        user_query,
        str,
    ):

        raise TypeError(
            "user_query must be a string."
        )

    query = (
        user_query
        .casefold()
        .strip()
    )

    # ========================================================
    # 1. RELATIONSHIP
    # ========================================================
    # Must come before comparison because relationship queries
    # may contain words such as "compare".
    # ========================================================

    relationship_patterns = [
        r"\brelationship\s+between\b",
        r"\brelationship\s+of\b",
        r"\bcorrelation\s+between\b",
        r"\bcorrelation\b",
        r"\bscatter\s*plot\b",
        r"\bscatter\s*graph\b",
        r"\bscatter\b",
        r"\bassociated\s+with\b",
        r"\bassociation\s+between\b",
        r"\bagainst\b",
    ]

    if any(
        re.search(
            pattern,
            query,
        )
        for pattern
        in relationship_patterns
    ):

        return {
            "llm_intent":
                llm_intent,

            "intent":
                "relationship",

            "source":
                "deterministic_guard",

            "reason":
                (
                    "Explicit relationship, correlation "
                    "or scatter language detected."
                ),
        }

    # ========================================================
    # 2. DISTRIBUTION
    # ========================================================

    distribution_patterns = [
        r"\bdistribution\b",
        r"\bhistogram\b",
        r"\bbox\s*plot\b",
        r"\bboxplot\b",
        r"\bbox\s+and\s+whisker\b",
        r"\boutliers?\b",
        r"\bfrequency\s+distribution\b",
        r"\bspread\b",
    ]

    if any(
        re.search(
            pattern,
            query,
        )
        for pattern
        in distribution_patterns
    ):

        return {
            "llm_intent":
                llm_intent,

            "intent":
                "distribution",

            "source":
                "deterministic_guard",

            "reason":
                (
                    "Explicit distribution/box/"
                    "histogram language detected."
                ),
        }

    # ========================================================
    # 3. THRESHOLD
    # ========================================================

    threshold_patterns = [
        r"\babove\s+-?\d",
        r"\bbelow\s+-?\d",
        r"\bover\s+-?\d",
        r"\bunder\s+-?\d",
        r"\bgreater\s+than\s+-?\d",
        r"\bless\s+than\s+-?\d",
        r"\bat\s+least\s+-?\d",
        r"\bat\s+most\s+-?\d",
        r"[<>]=?\s*-?\d",
    ]

    if any(
        re.search(
            pattern,
            query,
        )
        for pattern
        in threshold_patterns
    ):

        return {
            "llm_intent":
                llm_intent,

            "intent":
                "threshold",

            "source":
                "deterministic_guard",

            "reason":
                "Explicit numeric threshold detected.",
        }

    # ========================================================
    # 4. ROOM COMPARISON
    # ========================================================

    comparison_patterns = [
        r"\bcompare\b",
        r"\bcomparison\b",
        r"\bversus\b",
        r"\bvs\.?\b",
    ]

    if any(
        re.search(
            pattern,
            query,
        )
        for pattern
        in comparison_patterns
    ):

        return {
            "llm_intent":
                llm_intent,

            "intent":
                "compare_rooms",

            "source":
                "deterministic_guard",

            "reason":
                "Explicit room comparison language detected.",
        }

    # ========================================================
    # 5. RANKING
    # ========================================================

    ranking_patterns = [
        r"\btop\b",
        r"\bbottom\b",
        r"\bhighest\b",
        r"\blowest\b",
        r"\bhottest\b",
        r"\bwarmest\b",
        r"\bcoldest\b",
        r"\bbusiest\b",
        r"\bleast\s+occupied\b",
        r"\bless\s+utilised\b",
        r"\bless\s+utilized\b",
        r"\bmost\s+occupied\b",
        r"\bworst\b",
        r"\bbest\b",
    ]

    if any(
        re.search(
            pattern,
            query,
        )
        for pattern
        in ranking_patterns
    ):

        return {
            "llm_intent":
                llm_intent,

            "intent":
                "rank_rooms",

            "source":
                "deterministic_guard",

            "reason":
                "Explicit ranking language detected.",
        }

    # ========================================================
    # 6. TREND
    # ========================================================

    trend_patterns = [
        r"\btrend\b",
        r"\bover\s+time\b",
        r"\btime\s+series\b",
        r"\bhistory\b",
        r"\bhistorical\s+trend\b",
    ]

    if any(
        re.search(
            pattern,
            query,
        )
        for pattern
        in trend_patterns
    ):

        return {
            "llm_intent":
                llm_intent,

            "intent":
                "room_trend",

            "source":
                "deterministic_guard",

            "reason":
                "Explicit trend/time-series language detected.",
        }

    # ========================================================
    # 7. ACCEPT VALID LLM INTENT
    # ========================================================

    supported_intents = {
        "room_trend",
        "compare_rooms",
        "rank_rooms",
        "threshold",
        "distribution",
        "relationship",
    }

    if llm_intent in supported_intents:

        return {
            "llm_intent":
                llm_intent,

            "intent":
                llm_intent,

            "source":
                "llm",

            "reason":
                (
                    "No deterministic rule overrode "
                    "the LLM prediction."
                ),
        }

    # ========================================================
    # 8. SAFE FALLBACK
    # ========================================================

    return {
        "llm_intent":
            llm_intent,

        "intent":
            "rank_rooms",

        "source":
            "fallback",

        "reason":
            (
                "No explicit deterministic pattern "
                "was found and the LLM intent was invalid."
            ),
    }