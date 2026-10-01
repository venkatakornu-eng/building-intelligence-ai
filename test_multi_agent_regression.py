from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Any

from src.multi_agent_graph import SmartVizMultiAgentGraph


# ============================================================
# OUTPUT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

OUTPUT_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "multi_agent_regression_results.csv"
)

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# TEST CASES
# ============================================================

TEST_CASES: list[dict[str, Any]] = [

    # --------------------------------------------------------
    # 1. ROOM TREND
    # --------------------------------------------------------

    {
        "id": "T01",
        "prompt":
            "Show the temperature trend for Seminar Room 3.",
        "expected_intent":
            "room_trend",
        "expected_query_type":
            "room_trend",
        "expected_chart":
            "line",
        "expected_valid":
            True,
    },

    # --------------------------------------------------------
    # 2. ROOM COMPARISON
    # --------------------------------------------------------

    {
        "id": "T02",
        "prompt":
            (
                "Compare average humidity between "
                "Dining Room and Seminar Room 3."
            ),
        "expected_intent":
            "compare_rooms",
        "expected_query_type":
            "compare_rooms",
        "expected_chart":
            "bar",
        "expected_valid":
            True,
    },

    # --------------------------------------------------------
    # 3. TOP-N RANKING
    # --------------------------------------------------------

    {
        "id": "T03",
        "prompt":
            "Show the top 5 rooms by average CO2.",
        "expected_intent":
            "rank_rooms",
        "expected_query_type":
            "rank_rooms",
        "expected_chart":
            "bar",
        "expected_valid":
            True,
    },

    # --------------------------------------------------------
    # 4. BOTTOM-N RANKING
    # --------------------------------------------------------

    {
        "id": "T04",
        "prompt":
            "Show the bottom 5 rooms by average occupancy.",
        "expected_intent":
            "rank_rooms",
        "expected_query_type":
            "rank_rooms",
        "expected_chart":
            "bar",
        "expected_valid":
            True,
    },

    # --------------------------------------------------------
    # 5. LATEST AVAILABLE RANKING
    # --------------------------------------------------------

    {
        "id": "T05",
        "prompt":
            "Which room has the highest CO2 right now?",
        "expected_intent":
            "rank_rooms",
        "expected_query_type":
            "rank_rooms_latest",
        "expected_chart":
            "bar",
        "expected_valid":
            True,
    },

    # --------------------------------------------------------
    # 6. THRESHOLD - EXPECTED NO DATA
    # --------------------------------------------------------

    {
        "id": "T06",
        "prompt":
            "Show rooms with average CO2 above 800.",
        "expected_intent":
            "threshold",
        "expected_query_type":
            "threshold",
        "expected_chart":
            "bar",
        "expected_valid":
            True,
    },

    # --------------------------------------------------------
    # 7. NORMAL DISTRIBUTION
    # --------------------------------------------------------

    {
        "id": "T07",
        "prompt":
            "Show the humidity distribution across all rooms.",
        "expected_intent":
            "distribution",
        "expected_query_type":
            "distribution",
        "expected_chart":
            "histogram",
        "expected_valid":
            True,
    },

    # --------------------------------------------------------
    # 8. TOP-N DISTRIBUTION
    # --------------------------------------------------------

    {
        "id": "T08",
        "prompt":
            "Show the humidity distribution of top 4 rooms.",
        "expected_intent":
            "distribution",
        "expected_query_type":
            "distribution",
        "expected_chart":
            "histogram",
        "expected_valid":
            True,
    },

    # --------------------------------------------------------
    # 9. BOTTOM-N DISTRIBUTION
    # --------------------------------------------------------

    {
        "id": "T09",
        "prompt":
            "Show the humidity distribution of bottom 3 rooms.",
        "expected_intent":
            "distribution",
        "expected_query_type":
            "distribution",
        "expected_chart":
            "histogram",
        "expected_valid":
            True,
    },

    # --------------------------------------------------------
    # 10. DATE-FILTERED DISTRIBUTION
    # --------------------------------------------------------

    {
        "id": "T10",
        "prompt":
            (
                "Show the humidity distribution of the top 4 "
                "rooms between 2025-05-01 and 2025-05-31."
            ),
        "expected_intent":
            "distribution",
        "expected_query_type":
            "distribution",
        "expected_chart":
            "histogram",
        "expected_valid":
            True,
    },

    # --------------------------------------------------------
    # 11. RELATIONSHIP - ALL ROOMS
    # --------------------------------------------------------

    {
        "id": "T11",
        "prompt":
            (
                "Show the relationship between occupancy "
                "and CO2 across all rooms."
            ),
        "expected_intent":
            "relationship",
        "expected_query_type":
            "relationship",
        "expected_chart":
            "scatter",
        "expected_valid":
            True,
    },

    # --------------------------------------------------------
    # 12. RELATIONSHIP - SINGLE ROOM
    # --------------------------------------------------------

    {
        "id": "T12",
        "prompt":
            (
                "Show the relationship between temperature "
                "and humidity in Seminar Room 3."
            ),
        "expected_intent":
            "relationship",
        "expected_query_type":
            "relationship",
        "expected_chart":
            "scatter",
        "expected_valid":
            True,
    },
]


# ============================================================
# SAFE BOOLEAN
# ============================================================

def bool_value(
    value: Any,
) -> bool:

    return bool(
        value
    )


# ============================================================
# RUN ONE TEST
# ============================================================

def run_test(
    graph: SmartVizMultiAgentGraph,
    test_case: dict[str, Any],
) -> dict[str, Any]:

    prompt = test_case[
        "prompt"
    ]

    print(
        "\n"
        + "=" * 80
    )

    print(
        f"{test_case['id']}: "
        f"{prompt}"
    )

    print(
        "=" * 80
    )

    start_time = (
        time.perf_counter()
    )

    try:

        state = (
            graph.run(
                prompt
            )
        )

        elapsed = (
            time.perf_counter()
            -
            start_time
        )

        # ====================================================
        # ACTUAL VALUES
        # ====================================================

        actual_intent = (
            state.get(
                "intent"
            )
        )

        actual_query_type = (
            state.get(
                "query_type"
            )
        )

        # ----------------------------------------------------
        # SELECTED CHART
        #
        # This is the chart type selected from the user's
        # request before data availability is known.
        # ----------------------------------------------------

        selected_chart = (
            state.get(
                "chart_type"
            )
        )

        # ----------------------------------------------------
        # GENERATED CHART
        #
        # This may legitimately be None when a valid query
        # returns no data.
        # ----------------------------------------------------

        generated_chart = (
            state.get(
                "final_chart_type"
            )
        )

        result_valid = (
            bool_value(
                state.get(
                    "result_valid"
                )
            )
        )

        result_has_data = (
            bool_value(
                state.get(
                    "result_has_data"
                )
            )
        )

        execution_success = (
            bool_value(
                state.get(
                    "execution_success"
                )
            )
        )

        visualization_success = (
            bool_value(
                state.get(
                    "visualization_success"
                )
            )
        )

        chart_created = (
            bool_value(
                state.get(
                    "chart_created"
                )
            )
        )

        insight_success = (
            bool_value(
                state.get(
                    "insight_generation_success"
                )
            )
        )

        insight_guard_valid = (
            bool_value(
                state.get(
                    "insight_guard_valid"
                )
            )
        )

        insight_fallback = (
            bool_value(
                state.get(
                    "insight_used_fallback"
                )
            )
        )

        # ====================================================
        # EXPECTED VALUES
        # ====================================================

        expected_intent = (
            test_case[
                "expected_intent"
            ]
        )

        expected_query_type = (
            test_case[
                "expected_query_type"
            ]
        )

        expected_chart = (
            test_case[
                "expected_chart"
            ]
        )

        expected_valid = (
            test_case[
                "expected_valid"
            ]
        )

        # ====================================================
        # COMPONENT CHECKS
        # ====================================================

        intent_correct = (
            actual_intent
            ==
            expected_intent
        )

        query_type_correct = (
            actual_query_type
            ==
            expected_query_type
        )

        # ----------------------------------------------------
        # Compare EXPECTED CHART against SELECTED CHART.
        #
        # Do not compare against final_chart_type because
        # final_chart_type is None when there is valid no-data.
        # ----------------------------------------------------

        chart_correct = (
            selected_chart
            ==
            expected_chart
        )

        validation_correct = (
            result_valid
            ==
            expected_valid
        )

        # ====================================================
        # VISUALIZATION CHECK
        # ====================================================

        # ----------------------------------------------------
        # If data exists:
        #     visualization must succeed AND chart must exist.
        #
        # If no data exists:
        #     no chart is required.
        #
        # This correctly handles threshold queries that return
        # zero matching rows.
        # ----------------------------------------------------

        if result_has_data:

            visualization_ok = (
                visualization_success
                and
                chart_created
            )

        else:

            visualization_ok = (
                visualization_success
                and
                not chart_created
            )

        # ====================================================
        # FULL PIPELINE PASS
        # ====================================================

        pipeline_pass = all(
            [
                intent_correct,
                query_type_correct,
                chart_correct,
                validation_correct,
                execution_success,
                visualization_ok,
                insight_success,
                insight_guard_valid,
            ]
        )

        # ====================================================
        # PRINT TEST RESULT
        # ====================================================

        print(
            f"\nIntent: "
            f"{actual_intent}"
        )

        print(
            f"Query type: "
            f"{actual_query_type}"
        )

        print(
            f"Selected chart: "
            f"{selected_chart}"
        )

        print(
            f"Generated chart: "
            f"{generated_chart}"
        )

        print(
            f"Result valid: "
            f"{result_valid}"
        )

        print(
            f"Has data: "
            f"{result_has_data}"
        )

        print(
            f"Visualization success: "
            f"{visualization_success}"
        )

        print(
            f"Chart created: "
            f"{chart_created}"
        )

        print(
            f"Rows: "
            f"{state.get('sql_row_count', 0)}"
        )

        print(
            f"Insight generated by: "
            f"{state.get('insight_generated_by')}"
        )

        print(
            f"Insight fallback: "
            f"{insight_fallback}"
        )

        print(
            f"Latency: "
            f"{elapsed:.3f}s"
        )

        if pipeline_pass:

            print(
                "\n✅ TEST PASSED"
            )

        else:

            print(
                "\n❌ TEST FAILED"
            )

            # ------------------------------------------------
            # DEBUG FAILED COMPONENTS
            # ------------------------------------------------

            print(
                "\nFailure diagnostics:"
            )

            if not intent_correct:

                print(
                    "- Intent mismatch"
                )

            if not query_type_correct:

                print(
                    "- Query type mismatch"
                )

            if not chart_correct:

                print(
                    "- Selected chart mismatch"
                )

            if not validation_correct:

                print(
                    "- Result validation mismatch"
                )

            if not execution_success:

                print(
                    "- SQL execution failed"
                )

            if not visualization_ok:

                print(
                    "- Visualization behaviour incorrect"
                )

            if not insight_success:

                print(
                    "- Insight generation failed"
                )

            if not insight_guard_valid:

                print(
                    "- Insight Guard failed"
                )

        # ====================================================
        # RETURN RESULT
        # ====================================================

        return {
            "test_id":
                test_case[
                    "id"
                ],

            "prompt":
                prompt,

            # ------------------------------------------------
            # Intent
            # ------------------------------------------------

            "expected_intent":
                expected_intent,

            "actual_intent":
                actual_intent,

            "intent_correct":
                intent_correct,

            # ------------------------------------------------
            # Query type
            # ------------------------------------------------

            "expected_query_type":
                expected_query_type,

            "actual_query_type":
                actual_query_type,

            "query_type_correct":
                query_type_correct,

            # ------------------------------------------------
            # Chart
            # ------------------------------------------------

            "expected_chart":
                expected_chart,

            "selected_chart":
                selected_chart,

            "generated_chart":
                generated_chart,

            "chart_correct":
                chart_correct,

            # ------------------------------------------------
            # SQL + validation
            # ------------------------------------------------

            "result_valid":
                result_valid,

            "result_status":
                state.get(
                    "result_status"
                ),

            "result_has_data":
                result_has_data,

            "execution_success":
                execution_success,

            "row_count":
                state.get(
                    "sql_row_count",
                    0,
                ),

            "stored_row_count":
                state.get(
                    "sql_stored_row_count",
                    0,
                ),

            "result_truncated":
                bool_value(
                    state.get(
                        "sql_result_truncated"
                    )
                ),

            # ------------------------------------------------
            # Visualization
            # ------------------------------------------------

            "visualization_success":
                visualization_success,

            "chart_created":
                chart_created,

            "visualization_ok":
                visualization_ok,

            # ------------------------------------------------
            # Insight
            # ------------------------------------------------

            "insight_success":
                insight_success,

            "insight_guard_valid":
                insight_guard_valid,

            "insight_fallback":
                insight_fallback,

            "insight_generated_by":
                state.get(
                    "insight_generated_by"
                ),

            # ------------------------------------------------
            # Performance
            # ------------------------------------------------

            "latency_seconds":
                round(
                    elapsed,
                    3,
                ),

            # ------------------------------------------------
            # Overall
            # ------------------------------------------------

            "pipeline_pass":
                pipeline_pass,

            # ------------------------------------------------
            # Diagnostics
            # ------------------------------------------------

            "warnings":
                " | ".join(
                    state.get(
                        "warnings",
                        [],
                    )
                    or []
                ),

            "errors":
                " | ".join(
                    state.get(
                        "errors",
                        [],
                    )
                    or []
                ),
        }

    # ========================================================
    # TEST CRASH
    # ========================================================

    except Exception as error:

        elapsed = (
            time.perf_counter()
            -
            start_time
        )

        print(
            "\n❌ TEST CRASHED"
        )

        print(
            f"{type(error).__name__}: "
            f"{error}"
        )

        return {
            "test_id":
                test_case[
                    "id"
                ],

            "prompt":
                prompt,

            "expected_intent":
                test_case[
                    "expected_intent"
                ],

            "actual_intent":
                None,

            "intent_correct":
                False,

            "expected_query_type":
                test_case[
                    "expected_query_type"
                ],

            "actual_query_type":
                None,

            "query_type_correct":
                False,

            "expected_chart":
                test_case[
                    "expected_chart"
                ],

            "selected_chart":
                None,

            "generated_chart":
                None,

            "chart_correct":
                False,

            "result_valid":
                False,

            "result_status":
                "crashed",

            "result_has_data":
                False,

            "execution_success":
                False,

            "row_count":
                0,

            "stored_row_count":
                0,

            "result_truncated":
                False,

            "visualization_success":
                False,

            "chart_created":
                False,

            "visualization_ok":
                False,

            "insight_success":
                False,

            "insight_guard_valid":
                False,

            "insight_fallback":
                False,

            "insight_generated_by":
                None,

            "latency_seconds":
                round(
                    elapsed,
                    3,
                ),

            "pipeline_pass":
                False,

            "warnings":
                "",

            "errors":
                (
                    f"{type(error).__name__}: "
                    f"{error}"
                ),
        }


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(
    results: list[dict[str, Any]],
) -> None:

    if not results:
        return

    fieldnames = list(
        results[0].keys()
    )

    with OUTPUT_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=
                fieldnames,
        )

        writer.writeheader()

        writer.writerows(
            results
        )


# ============================================================
# SUMMARY
# ============================================================

def print_summary(
    results: list[dict[str, Any]],
) -> None:

    total = len(
        results
    )

    passed = sum(
        1
        for result in results
        if result[
            "pipeline_pass"
        ]
    )

    failed = (
        total
        -
        passed
    )

    # --------------------------------------------------------
    # COMPONENT ACCURACY
    # --------------------------------------------------------

    intent_correct_count = sum(
        1
        for result in results
        if result[
            "intent_correct"
        ]
    )

    query_type_correct_count = sum(
        1
        for result in results
        if result[
            "query_type_correct"
        ]
    )

    chart_correct_count = sum(
        1
        for result in results
        if result[
            "chart_correct"
        ]
    )

    execution_success_count = sum(
        1
        for result in results
        if result[
            "execution_success"
        ]
    )

    visualization_ok_count = sum(
        1
        for result in results
        if result[
            "visualization_ok"
        ]
    )

    insight_success_count = sum(
        1
        for result in results
        if result[
            "insight_success"
        ]
    )

    guard_valid_count = sum(
        1
        for result in results
        if result[
            "insight_guard_valid"
        ]
    )

    # --------------------------------------------------------
    # FALLBACK
    # --------------------------------------------------------

    fallback_count = sum(
        1
        for result in results
        if result[
            "insight_fallback"
        ]
    )

    # --------------------------------------------------------
    # LATENCY
    # --------------------------------------------------------

    total_latency = sum(
        float(
            result[
                "latency_seconds"
            ]
        )
        for result in results
    )

    average_latency = (
        total_latency
        /
        total
        if total
        else 0
    )

    # --------------------------------------------------------
    # PIPELINE SUCCESS
    # --------------------------------------------------------

    pipeline_success_rate = (
        passed
        /
        total
        *
        100
        if total
        else 0
    )

    # ========================================================
    # PRINT SUMMARY
    # ========================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "REGRESSION SUMMARY"
    )

    print(
        "=" * 80
    )

    print(
        f"\nTests: "
        f"{total}"
    )

    print(
        f"Passed: "
        f"{passed}"
    )

    print(
        f"Failed: "
        f"{failed}"
    )

    print(
        f"Pipeline success rate: "
        f"{pipeline_success_rate:.1f}%"
    )

    # --------------------------------------------------------
    # COMPONENT METRICS
    # --------------------------------------------------------

    print(
        "\nCOMPONENT RESULTS"
    )

    print(
        "-" * 80
    )

    print(
        f"Intent accuracy: "
        f"{intent_correct_count}/{total} "
        f"({intent_correct_count / total * 100:.1f}%)"
    )

    print(
        f"Query-type accuracy: "
        f"{query_type_correct_count}/{total} "
        f"({query_type_correct_count / total * 100:.1f}%)"
    )

    print(
        f"Chart-selection accuracy: "
        f"{chart_correct_count}/{total} "
        f"({chart_correct_count / total * 100:.1f}%)"
    )

    print(
        f"SQL execution success: "
        f"{execution_success_count}/{total} "
        f"({execution_success_count / total * 100:.1f}%)"
    )

    print(
        f"Visualization correctness: "
        f"{visualization_ok_count}/{total} "
        f"({visualization_ok_count / total * 100:.1f}%)"
    )

    print(
        f"Insight generation success: "
        f"{insight_success_count}/{total} "
        f"({insight_success_count / total * 100:.1f}%)"
    )

    print(
        f"Insight Guard validity: "
        f"{guard_valid_count}/{total} "
        f"({guard_valid_count / total * 100:.1f}%)"
    )

    # --------------------------------------------------------
    # GUARD FALLBACK
    # --------------------------------------------------------

    print(
        f"\nInsight Guard fallback count: "
        f"{fallback_count}"
    )

    print(
        f"Insight Guard fallback rate: "
        f"{fallback_count / total * 100:.1f}%"
    )

    # --------------------------------------------------------
    # LATENCY
    # --------------------------------------------------------

    print(
        f"\nAverage latency: "
        f"{average_latency:.3f}s"
    )

    # --------------------------------------------------------
    # SAVE LOCATION
    # --------------------------------------------------------

    print(
        "\nSaved results:"
    )

    print(
        OUTPUT_PATH
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print(
        "=" * 80
    )

    print(
        "SMARTVIZ MULTI-AGENT REGRESSION TEST"
    )

    print(
        "=" * 80
    )

    # --------------------------------------------------------
    # Build the LangGraph system only once.
    # --------------------------------------------------------

    graph = (
        SmartVizMultiAgentGraph()
    )

    results: list[
        dict[str, Any]
    ] = []

    # --------------------------------------------------------
    # Execute all controlled regression cases.
    # --------------------------------------------------------

    for test_case in TEST_CASES:

        result = (
            run_test(
                graph=
                    graph,

                test_case=
                    test_case,
            )
        )

        results.append(
            result
        )

    # --------------------------------------------------------
    # Save detailed results
    # --------------------------------------------------------

    save_results(
        results
    )

    # --------------------------------------------------------
    # Print summary metrics
    # --------------------------------------------------------

    print_summary(
        results
    )


if __name__ == "__main__":

    main()