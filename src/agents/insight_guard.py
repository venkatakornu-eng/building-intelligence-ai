from __future__ import annotations

import re
from typing import Any


class InsightGuard:
    """
    Deterministic safety and quality guard for SmartViz insights.

    The guard checks LLM-generated insight text AFTER the
    InsightAgent has generated deterministic analytical facts.

    It detects:
    - empty output;
    - incomplete statements;
    - repetitive output;
    - missing ranked evidence;
    - incorrect room/value association;
    - unsupported measurement units;
    - unsupported causal language;
    - unsafe latest/live wording.

    If the LLM response fails validation, the guard replaces
    the user-visible insight with a deterministic fallback
    created directly from validated analytical facts.

    No LLM is used by this guard.
    """

    # ========================================================
    # NORMALISE TEXT
    # ========================================================

    @staticmethod
    def _normalise(
        value: Any,
    ) -> str:

        if value is None:
            return ""

        return (
            str(value)
            .casefold()
            .strip()
        )

    # ========================================================
    # CLEAN TEXT
    # ========================================================

    @staticmethod
    def _clean_text(
        value: Any,
    ) -> str:

        if value is None:
            return ""

        return str(value).strip()

    # ========================================================
    # INCOMPLETE SENTENCE DETECTION
    # ========================================================

    def _looks_incomplete(
        self,
        text: str,
    ) -> bool:

        text = self._normalise(
            text
        )

        if not text:
            return True

        incomplete_endings = (
            "as follows:",
            "as follows",
            "including:",
            "including",
            "such as:",
            "such as",
            "the following:",
            "the following",
            "are:",
            "is:",
        )

        return text.endswith(
            incomplete_endings
        )

    # ========================================================
    # REPETITION CHECK
    # ========================================================

    def _is_repetitive(
        self,
        summary: str,
        key_points: list[str],
    ) -> bool:

        summary_normalised = (
            self._normalise(
                summary
            )
        )

        if not summary_normalised:
            return True

        for point in key_points:

            point_normalised = (
                self._normalise(
                    point
                )
            )

            if not point_normalised:
                continue

            # ------------------------------------------------
            # EXACT REPETITION
            # ------------------------------------------------

            if (
                point_normalised
                ==
                summary_normalised
            ):

                return True

            # ------------------------------------------------
            # LARGE SUBSTRING REPETITION
            # ------------------------------------------------

            if (
                len(
                    point_normalised
                ) > 25
                and
                point_normalised
                in summary_normalised
            ):

                return True

            if (
                len(
                    summary_normalised
                ) > 25
                and
                summary_normalised
                in point_normalised
            ):

                return True

        return False

    # ========================================================
    # FACT VALUE IN TEXT
    # ========================================================

    def _text_contains(
        self,
        text: str,
        value: Any,
    ) -> bool:

        if value is None:
            return False

        text_normalised = (
            self._normalise(
                text
            )
        )

        value_normalised = (
            self._normalise(
                value
            )
        )

        if not value_normalised:
            return False

        return (
            value_normalised
            in text_normalised
        )

    # ========================================================
    # UNSUPPORTED UNIT CHECK
    # ========================================================

    def _validate_units(
        self,
        combined_text: str,
        facts: dict[str, Any],
    ) -> list[str]:
        """
        Detect units invented by the language model.

        Current deterministic SmartViz facts do not include
        verified presentation units.

        Therefore the LLM must not independently add:
        - %
        - ppm
        - ppb
        - °C
        - degrees Celsius
        """

        reasons: list[str] = []

        text = self._normalise(
            combined_text
        )

        unsupported_unit_patterns = [
            r"\d+(?:\.\d+)?\s*%",
            r"\bppm\b",
            r"\bppb\b",
            r"°\s*c\b",
            r"\bdegrees?\s+celsius\b",
        ]

        for pattern in unsupported_unit_patterns:

            if re.search(
                pattern,
                text,
            ):

                reasons.append(
                    "The insight introduces an "
                    "unsupported measurement unit."
                )

                break

        return reasons

    # ========================================================
    # RANKED DISTRIBUTION QUALITY
    # ========================================================

    def _validate_ranked_distribution(
        self,
        segments: list[str],
        combined_text: str,
        facts: dict[str, Any],
    ) -> list[str]:

        reasons: list[str] = []

        ranked_rooms = facts.get(
            "ranked_rooms",
            [],
        )

        if not ranked_rooms:
            return reasons

        first_room = ranked_rooms[0]

        room_name = first_room.get(
            "room"
        )

        ranking_value = first_room.get(
            "ranking_value"
        )

        selection_type = self._normalise(
            first_room.get(
                "selection_type"
            )
        )

        if selection_type == "bottom":

            rank_description = (
                "lowest-ranked"
            )

        else:

            rank_description = (
                "highest-ranked"
            )

        # ----------------------------------------------------
        # FIRST ROOM MUST BE PRESENT
        # ----------------------------------------------------

        room_present = (
            self._text_contains(
                combined_text,
                room_name,
            )
        )

        if not room_present:

            reasons.append(
                f"The insight does not mention the "
                f"{rank_description} room."
            )

        # ----------------------------------------------------
        # FIRST RANKING VALUE MUST BE PRESENT
        # ----------------------------------------------------

        value_present = (
            self._text_contains(
                combined_text,
                ranking_value,
            )
        )

        if not value_present:

            reasons.append(
                f"The insight does not mention the "
                f"{rank_description} room's ranking value."
            )

        # ----------------------------------------------------
        # ROOM AND VALUE MUST APPEAR TOGETHER
        #
        # We test summary/key-points independently rather
        # than splitting sentences on ".", because decimal
        # values such as 54.62 contain a period.
        # ----------------------------------------------------

        if (
            room_present
            and
            value_present
        ):

            correct_pair_found = False

            for segment in segments:

                if (
                    self._text_contains(
                        segment,
                        room_name,
                    )
                    and
                    self._text_contains(
                        segment,
                        ranking_value,
                    )
                ):

                    correct_pair_found = True
                    break

            if not correct_pair_found:

                reasons.append(
                    f"The {rank_description} room and "
                    "its ranking value are not clearly "
                    "associated in the same insight statement."
                )

        return reasons

    # ========================================================
    # RANKING QUALITY
    # ========================================================

    def _validate_ranking(
        self,
        segments: list[str],
        combined_text: str,
        facts: dict[str, Any],
    ) -> list[str]:

        reasons: list[str] = []

        first_room = facts.get(
            "first_room"
        )

        if not first_room:
            return reasons

        room_name = first_room.get(
            "room"
        )

        value = first_room.get(
            "value"
        )

        room_present = (
            self._text_contains(
                combined_text,
                room_name,
            )
        )

        value_present = (
            self._text_contains(
                combined_text,
                value,
            )
        )

        if not room_present:

            reasons.append(
                "The insight does not mention the "
                "first-ranked room."
            )

        if not value_present:

            reasons.append(
                "The insight does not mention the "
                "first-ranked value."
            )

        # ----------------------------------------------------
        # ENSURE ROOM/VALUE PAIRING
        # ----------------------------------------------------

        if (
            room_present
            and
            value_present
        ):

            correct_pair_found = False

            for segment in segments:

                if (
                    self._text_contains(
                        segment,
                        room_name,
                    )
                    and
                    self._text_contains(
                        segment,
                        value,
                    )
                ):

                    correct_pair_found = True
                    break

            if not correct_pair_found:

                reasons.append(
                    "The first-ranked room and its value "
                    "are not clearly associated in the "
                    "same insight statement."
                )

        return reasons

    # ========================================================
    # RELATIONSHIP QUALITY
    # ========================================================

    def _validate_relationship(
        self,
        combined_text: str,
        facts: dict[str, Any],
    ) -> list[str]:

        reasons: list[str] = []

        metric_x = facts.get(
            "metric_x"
        )

        metric_y = facts.get(
            "metric_y"
        )

        pearson_r = facts.get(
            "pearson_r"
        )

        # ----------------------------------------------------
        # BOTH METRICS MUST BE PRESENT
        # ----------------------------------------------------

        if not self._text_contains(
            combined_text,
            metric_x,
        ):

            reasons.append(
                "Relationship insight does not mention "
                "metric_x."
            )

        if not self._text_contains(
            combined_text,
            metric_y,
        ):

            reasons.append(
                "Relationship insight does not mention "
                "metric_y."
            )

        # ----------------------------------------------------
        # CORRELATION MUST BE PRESENT WHEN AVAILABLE
        # ----------------------------------------------------

        if (
            pearson_r is not None
            and
            not self._text_contains(
                combined_text,
                pearson_r,
            )
        ):

            reasons.append(
                "Relationship insight does not mention "
                "the calculated Pearson correlation."
            )

        # ----------------------------------------------------
        # CAUSAL WORDING PROTECTION
        # ----------------------------------------------------

        causal_patterns = (
            r"\bcauses?\b",
            r"\bcaused by\b",
            r"\bleads? to\b",
            r"\bresults? in\b",
            r"\bdrives?\b",
            r"\bresponsible for\b",
            r"\binfluences?\b",
            r"\baffects?\b",
            r"\bimpacts?\b",
            r"\bdetermines?\b",
            r"\bcontributes? to\b",
            r"\bpredicts?\b",
            r"\bcorrelates?\b", 
            r"\bpredicting\b",
        )

        normalised_text = (
            self._normalise(
                combined_text
            )
        )

        for pattern in causal_patterns:

            if re.search(
                pattern,
                normalised_text,
            ):

                reasons.append(
                    "Relationship insight contains "
                    "unsupported causal language."
                )

                break

        return reasons

    # ========================================================
    # LATEST-RANKING QUALITY
    # ========================================================

    def _validate_latest(
        self,
        combined_text: str,
    ) -> list[str]:

        reasons: list[str] = []

        text = self._normalise(
            combined_text
        )

        # ----------------------------------------------------
        # MUST DESCRIBE LATEST AS AVAILABLE/HISTORICAL
        # ----------------------------------------------------

        if (
            "latest available"
            not in text
            and
            "latest reading"
            not in text
            and
            "latest historical"
            not in text
        ):

            reasons.append(
                "Latest-data insight does not clearly state "
                "that the result is the latest available reading."
            )

        # ----------------------------------------------------
        # MUST NOT CLAIM LIVE DATA
        # ----------------------------------------------------

        live_claim = re.search(
            r"\b(?:live|real[- ]?time)\b",
            text,
        )

        negated_live_claim = (
            "not live" in text
            or
            "not real-time" in text
            or
            "not real time" in text
        )

        if (
            live_claim
            and
            not negated_live_claim
        ):

            reasons.append(
                "Latest-data insight incorrectly implies "
                "live real-time data."
            )

        return reasons

    # ========================================================
    # GENERAL QUALITY VALIDATION
    # ========================================================

    def _validate_llm_output(
        self,
        insight: dict[str, Any],
    ) -> list[str]:

        reasons: list[str] = []

        # ----------------------------------------------------
        # SUMMARY
        # ----------------------------------------------------

        summary = self._clean_text(
            insight.get(
                "summary"
            )
        )

        # ----------------------------------------------------
        # KEY POINTS
        # ----------------------------------------------------

        key_points_raw = insight.get(
            "key_points",
            [],
        )

        if not isinstance(
            key_points_raw,
            list,
        ):

            reasons.append(
                "key_points is not a list."
            )

            key_points: list[str] = []

        else:

            key_points = [
                self._clean_text(
                    point
                )
                for point in key_points_raw
                if self._clean_text(
                    point
                )
            ]

        # ----------------------------------------------------
        # CAUTION
        # ----------------------------------------------------

        caution = self._clean_text(
            insight.get(
                "caution"
            )
        )

        # ----------------------------------------------------
        # FACTS
        # ----------------------------------------------------

        facts = insight.get(
            "facts",
            {},
        )

        if not isinstance(
            facts,
            dict,
        ):

            reasons.append(
                "Deterministic facts are missing."
            )

            return reasons

        # ====================================================
        # SUMMARY QUALITY
        # ====================================================

        if not summary:

            reasons.append(
                "Insight summary is empty."
            )

        elif self._looks_incomplete(
            summary
        ):

            reasons.append(
                "Insight summary appears incomplete."
            )

        # ====================================================
        # KEY POINT QUALITY
        # ====================================================

        if not key_points:

            reasons.append(
                "Insight contains no useful key points."
            )

        for point in key_points:

            if self._looks_incomplete(
                point
            ):

                reasons.append(
                    "A key point appears incomplete."
                )

                break

        # ====================================================
        # REPETITION
        # ====================================================

        if self._is_repetitive(
            summary=
                summary,

            key_points=
                key_points,
        ):

            reasons.append(
                "Summary and key points are repetitive."
            )

        # ====================================================
        # USER-VISIBLE SEGMENTS
        # ====================================================

        segments = [
            summary,
            *key_points,
        ]

        if caution:
            segments.append(
                caution
            )

        combined_text = " ".join(
            segments
        )

        # ====================================================
        # UNSUPPORTED UNIT CHECK
        # ====================================================

        reasons.extend(
            self._validate_units(
                combined_text=
                    combined_text,

                facts=
                    facts,
            )
        )

        # ====================================================
        # QUERY-SPECIFIC QUALITY
        # ====================================================

        analysis_type = (
            self._normalise(
                facts.get(
                    "analysis_type"
                )
            )
        )

        # ----------------------------------------------------
        # RANKED DISTRIBUTION
        # ----------------------------------------------------

        if (
            analysis_type
            ==
            "distribution"
            and
            facts.get(
                "ranked_rooms"
            )
        ):

            reasons.extend(
                self._validate_ranked_distribution(
                    segments=
                        segments,

                    combined_text=
                        combined_text,

                    facts=
                        facts,
                )
            )

        # ----------------------------------------------------
        # RANK / COMPARE / THRESHOLD
        # ----------------------------------------------------

        elif analysis_type in {
            "rank_rooms",
            "compare_rooms",
            "threshold",
        }:

            reasons.extend(
                self._validate_ranking(
                    segments=
                        segments,

                    combined_text=
                        combined_text,

                    facts=
                        facts,
                )
            )

        # ----------------------------------------------------
        # LATEST RANKING
        # ----------------------------------------------------

        elif (
            analysis_type
            ==
            "rank_rooms_latest"
        ):

            reasons.extend(
                self._validate_ranking(
                    segments=
                        segments,

                    combined_text=
                        combined_text,

                    facts=
                        facts,
                )
            )

            reasons.extend(
                self._validate_latest(
                    combined_text=
                        combined_text,
                )
            )

        # ----------------------------------------------------
        # RELATIONSHIP
        # ----------------------------------------------------

        elif (
            analysis_type
            ==
            "relationship"
        ):

            reasons.extend(
                self._validate_relationship(
                    combined_text=
                        combined_text,

                    facts=
                        facts,
                )
            )

        return reasons

    # ========================================================
    # DETERMINISTIC DISTRIBUTION FALLBACK
    # ========================================================

    def _distribution_fallback(
        self,
        facts: dict[str, Any],
    ) -> dict[str, Any]:

        metric = facts.get(
            "metric",
            "value",
        )

        ranked_rooms = facts.get(
            "ranked_rooms",
            [],
        )

        statistics = facts.get(
            "statistics",
            {},
        )

        # ====================================================
        # TOP / BOTTOM N DISTRIBUTION
        # ====================================================

        if ranked_rooms:

            first = ranked_rooms[0]

            room_name = first.get(
                "room",
                "the first room",
            )

            ranking_value = first.get(
                "ranking_value"
            )

            selection_type = (
                self._normalise(
                    first.get(
                        "selection_type"
                    )
                )
            )

            if selection_type == "bottom":

                ranking_word = (
                    "lowest"
                )

            else:

                ranking_word = (
                    "highest"
                )

            summary = (
                f"{room_name} has the {ranking_word} "
                f"average {metric} among the selected "
                f"rooms, with a ranking value of "
                f"{ranking_value}."
            )

            key_points: list[str] = []

            # ------------------------------------------------
            # SECOND RANKED ROOM
            # ------------------------------------------------

            if len(
                ranked_rooms
            ) >= 2:

                second = ranked_rooms[1]

                key_points.append(
                    (
                        f"{second.get('room')} is ranked "
                        f"second with an average {metric} "
                        f"value of "
                        f"{second.get('ranking_value')}."
                    )
                )

            # ------------------------------------------------
            # DISTRIBUTION STATISTICS
            # ------------------------------------------------

            if statistics:

                key_points.append(
                    (
                        f"Across the returned observations, "
                        f"{metric} ranges from "
                        f"{statistics.get('minimum')} to "
                        f"{statistics.get('maximum')}, "
                        f"with a mean of "
                        f"{statistics.get('mean')} and "
                        f"a median of "
                        f"{statistics.get('median')}."
                    )
                )

            # ------------------------------------------------
            # SAMPLE SIZE
            # ------------------------------------------------

            key_points.append(
                (
                    f"The distribution contains "
                    f"{facts.get('row_count', 0)} observations "
                    f"from {facts.get('room_count', 0)} rooms."
                )
            )

            return {
                "summary":
                    summary,

                "key_points":
                    key_points[:3],

                "caution":
                    "",
            }

        # ====================================================
        # NORMAL DISTRIBUTION
        # ====================================================

        summary = (
            f"The {metric} distribution contains "
            f"{facts.get('row_count', 0)} observations "
            f"across {facts.get('room_count', 0)} room(s)."
        )

        key_points: list[str] = []

        if statistics:

            key_points.append(
                (
                    f"The observed {metric} values range "
                    f"from {statistics.get('minimum')} to "
                    f"{statistics.get('maximum')}."
                )
            )

            key_points.append(
                (
                    f"The mean is "
                    f"{statistics.get('mean')} and the "
                    f"median is "
                    f"{statistics.get('median')}."
                )
            )

        return {
            "summary":
                summary,

            "key_points":
                key_points,

            "caution":
                "",
        }

    # ========================================================
    # RANKING FALLBACK
    # ========================================================

    def _ranking_fallback(
        self,
        facts: dict[str, Any],
    ) -> dict[str, Any]:

        metric = facts.get(
            "metric",
            "value",
        )

        room_results = facts.get(
            "room_results",
            [],
        )

        if not room_results:

            return {
                "summary":
                    "Validated room results are available.",

                "key_points":
                    [],

                "caution":
                    "",
            }

        first = room_results[0]

        summary = (
            f"{first.get('room')} is the first-ranked "
            f"room for {metric}, with a value of "
            f"{first.get('value')}."
        )

        key_points: list[str] = []

        if len(
            room_results
        ) > 1:

            second = room_results[1]

            key_points.append(
                (
                    f"{second.get('room')} is second "
                    f"with a value of "
                    f"{second.get('value')}."
                )
            )

        key_points.append(
            (
                f"{facts.get('row_count', 0)} room "
                "result(s) were returned."
            )
        )

        caution = ""

        # ----------------------------------------------------
        # LATEST AVAILABLE DATA WARNING
        # ----------------------------------------------------

        if (
            self._normalise(
                facts.get(
                    "analysis_type"
                )
            )
            ==
            "rank_rooms_latest"
        ):

            caution = (
                "These values represent the latest "
                "available historical readings, "
                "not live real-time measurements."
            )

        return {
            "summary":
                summary,

            "key_points":
                key_points,

            "caution":
                caution,
        }

    # ========================================================
    # RELATIONSHIP FALLBACK
    # ========================================================

    def _relationship_fallback(
        self,
        facts: dict[str, Any],
    ) -> dict[str, Any]:

        metric_x = facts.get(
            "metric_x",
            "metric X",
        )

        metric_y = facts.get(
            "metric_y",
            "metric Y",
        )

        pearson_r = facts.get(
            "pearson_r"
        )

        association = facts.get(
            "association_description"
        )

        # ----------------------------------------------------
        # CORRELATION AVAILABLE
        # ----------------------------------------------------

        if pearson_r is not None:

            summary = (
                f"The relationship between "
                f"{metric_x} and {metric_y} has a "
                f"Pearson correlation of {pearson_r}, "
                f"indicating a {association}."
            )

        # ----------------------------------------------------
        # CORRELATION NOT AVAILABLE
        # ----------------------------------------------------

        else:

            summary = (
                f"A relationship between {metric_x} "
                f"and {metric_y} was analysed, but a "
                "valid Pearson correlation could not "
                "be calculated."
            )

        key_points = [
            (
                f"The analysis contains "
                f"{facts.get('paired_observation_count', 0)} "
                "paired observations."
            ),
            (
                f"The paired data covers "
                f"{facts.get('room_count', 0)} room(s)."
            ),
        ]

        caution = (
            "The relationship represents statistical "
            "association only and does not establish causation."
        )

        # ----------------------------------------------------
        # TRUNCATED RELATIONSHIP DATA
        # ----------------------------------------------------

        if facts.get(
            "truncated"
        ):

            caution += (
                " The correlation was calculated using "
                "the stored subset of results."
            )

        return {
            "summary":
                summary,

            "key_points":
                key_points,

            "caution":
                caution,
        }

    # ========================================================
    # TREND FALLBACK
    # ========================================================

    def _trend_fallback(
        self,
        facts: dict[str, Any],
    ) -> dict[str, Any]:

        metric = facts.get(
            "metric",
            "value",
        )

        room = facts.get(
            "room",
            "the requested room",
        )

        statistics = facts.get(
            "statistics",
            {},
        )

        summary = (
            f"The {metric} trend for {room} contains "
            f"{facts.get('row_count', 0)} observations."
        )

        key_points: list[str] = []

        # ----------------------------------------------------
        # BASIC STATISTICS
        # ----------------------------------------------------

        if statistics:

            key_points.append(
                (
                    f"The observed mean {metric} is "
                    f"{statistics.get('mean')}, with "
                    f"a minimum of "
                    f"{statistics.get('minimum')} and "
                    f"a maximum of "
                    f"{statistics.get('maximum')}."
                )
            )

        # ----------------------------------------------------
        # FIRST/LAST CHANGE
        # ----------------------------------------------------

        change = facts.get(
            "change"
        )

        if change is not None:

            key_points.append(
                (
                    f"The difference between the first "
                    f"and final stored observation is "
                    f"{change}."
                )
            )

        return {
            "summary":
                summary,

            "key_points":
                key_points,

            "caution":
                "",
        }

    # ========================================================
    # BUILD SAFE FALLBACK
    # ========================================================

    def _build_fallback(
        self,
        facts: dict[str, Any],
    ) -> dict[str, Any]:

        analysis_type = (
            self._normalise(
                facts.get(
                    "analysis_type"
                )
            )
        )

        # ----------------------------------------------------
        # DISTRIBUTION
        # ----------------------------------------------------

        if analysis_type == "distribution":

            return (
                self._distribution_fallback(
                    facts
                )
            )

        # ----------------------------------------------------
        # RANKING / COMPARISON / THRESHOLD
        # ----------------------------------------------------

        if analysis_type in {
            "rank_rooms",
            "rank_rooms_latest",
            "compare_rooms",
            "threshold",
        }:

            return (
                self._ranking_fallback(
                    facts
                )
            )

        # ----------------------------------------------------
        # RELATIONSHIP
        # ----------------------------------------------------

        if analysis_type == "relationship":

            return (
                self._relationship_fallback(
                    facts
                )
            )

        # ----------------------------------------------------
        # TREND
        # ----------------------------------------------------

        if analysis_type == "room trend":

            return (
                self._trend_fallback(
                    facts
                )
            )

        # ----------------------------------------------------
        # GENERIC FALLBACK
        # ----------------------------------------------------

        return {
            "summary":
                "Validated analytical results are available.",

            "key_points":
                [],

            "caution":
                "",
        }

    # ========================================================
    # PUBLIC GUARD METHOD
    # ========================================================

    def guard(
        self,
        insight: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Validate an InsightAgent result.

        If the LLM insight passes:
            return the original LLM insight.

        If it fails:
            replace the user-visible explanation using
            deterministic facts while preserving all
            analytical evidence for traceability.
        """

        # ====================================================
        # INVALID INPUT TYPE
        # ====================================================

        if not isinstance(
            insight,
            dict,
        ):

            return {
                "valid":
                    False,

                "used_fallback":
                    False,

                "reasons": [
                    "Insight result is not a dictionary."
                ],

                "insight":
                    None,
            }

        # ====================================================
        # GET FACTS
        # ====================================================

        facts = insight.get(
            "facts",
            {},
        )

        if not isinstance(
            facts,
            dict,
        ):

            facts = {}

        # ====================================================
        # UPSTREAM INSIGHT GENERATION FAILED
        # ====================================================

        if not insight.get(
            "success",
            False,
        ):

            return {
                "valid":
                    False,

                "used_fallback":
                    False,

                "reasons": [
                    "Insight generation was not successful."
                ],

                "insight":
                    insight,
            }

        generated_by = (insight.get(
            "generated_by"
        ))

        summary = self._clean_text(
            insight.get(
                "summary"
            )
        )

        # ====================================================
        # ACCEPT DETERMINQISTIC NO-DATA RESPONSE
        # ====================================================
        
        if (
            generated_by == "deterministic"
            and 
            summary
            and 
            not insight.get(
                "facts"
            )
        ):

            guarded_insight = dict(
                insight
            )

            guarded_insight[
                "guard_status"
            ] = "accepted_deterministic"

            guarded_insight[
                "guard_reasons"
            ] = []

            return {
                "valid":
                    True,

                "used_fallback":
                    False,

                "reasons":
                    [],

                "insight":
                    guarded_insight,
            }
        # ====================================================
        # VALIDATE LLM OUTPUT
        # ====================================================

        reasons = (
            self._validate_llm_output(
                insight
            )
        )

        # ====================================================
        # ACCEPT LLM INSIGHT
        # ====================================================

        if not reasons:

            guarded_insight = dict(
                insight
            )

            guarded_insight[
                "guard_status"
            ] = "accepted"

            guarded_insight[
                "guard_reasons"
            ] = []

            return {
                "valid":
                    True,

                "used_fallback":
                    False,

                "reasons":
                    [],

                "insight":
                    guarded_insight,
            }

        # ====================================================
        # REJECT LLM INSIGHT + BUILD SAFE FALLBACK
        # ====================================================

        fallback = (
            self._build_fallback(
                facts
            )
        )

        guarded_insight = {
            "success":
                True,

            "generated_by":
                "deterministic_guard_fallback",

            "summary":
                fallback[
                    "summary"
                ],

            "key_points":
                fallback[
                    "key_points"
                ],

            "caution":
                fallback[
                    "caution"
                ],

            "facts":
                facts,

            "error":
                insight.get(
                    "error"
                ),

            "guard_status":
                "replaced",

            "guard_reasons":
                reasons,
        }

        return {
            "valid":
                True,

            "used_fallback":
                True,

            "reasons":
                reasons,

            "insight":
                guarded_insight,
        }