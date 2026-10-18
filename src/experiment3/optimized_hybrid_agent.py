from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from src.agents.insight_agent import InsightAgent
from src.agents.insight_guard import InsightGuard
from src.baseline.hybrid_single_agent import (
    DEFAULT_DATA_PATH,
    HybridSingleAgent,
)


class OptimizedHybridAgent(HybridSingleAgent):
    """
    Experiment 3: Optimized Hybrid SmartViz architecture.

    Architecture
    ------------

        User Query
            ↓
        Hybrid Parser
        (LLM + deterministic rules)
            ↓
        Request Validation
            ↓
        Deterministic Query Builder
            ↓
        SQL Execution
            ↓
        Result Validation
            ↓
        Deterministic Visualization
            ↓
        Dedicated Insight Agent
            ↓
        Deterministic Insight Guard
            ↓
        Final Response


    Experimental purpose
    --------------------

    This architecture tests whether the reliable insight layer
    from the full multi-agent system can be combined with the
    faster hybrid query interpretation architecture.

    The experiment intentionally reuses the same:

    - Hybrid parser
    - RequestValidationAgent
    - QueryBuilderAgent
    - SQLExecutionAgent
    - ResultValidationAgent
    - VisualizationAgent
    - SQL database
    - InsightAgent
    - InsightGuard

    used by the previous evaluated systems.

    This allows Experiment 3 to isolate the architectural change:

        full multi-agent interpretation
                    ↓
        hybrid interpretation

    while preserving the guarded final-insight architecture.

    Important
    ---------

    The original HybridSingleAgent is NOT modified.

    Therefore:

        HybridSingleAgent
        = Experiment 2 baseline

        OptimizedHybridAgent
        = Experiment 3

    This preserves experimental reproducibility.
    """

    # ========================================================
    # INITIALISATION
    # ========================================================

    def __init__(
        self,
        data_path: str | Path = DEFAULT_DATA_PATH,
        model: str = "deepseek-r1:1.5b",
        timeout: int = 120,
    ) -> None:
        """
        Initialise the Optimized Hybrid architecture.

        HybridSingleAgent initialises:

        - available room catalogue
        - hybrid parser infrastructure
        - RequestValidationAgent
        - QueryBuilderAgent
        - SQLDatabase
        - SQLExecutionAgent
        - ResultValidationAgent
        - VisualizationAgent

        Experiment 3 then adds:

        - dedicated InsightAgent
        - deterministic InsightGuard
        """

        super().__init__(
            data_path=data_path,
            model=model,
            timeout=timeout,
        )

        # ----------------------------------------------------
        # Dedicated insight architecture from the evaluated
        # multi-agent system.
        # ----------------------------------------------------

        self.insight_agent = InsightAgent(
            model=model,
            timeout=timeout,
        )

        self.insight_guard = InsightGuard()

    # ========================================================
    # SAFE INSIGHT FALLBACK
    # ========================================================

    @staticmethod
    def _failed_insight(
        message: str,
        facts: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Create a structurally valid failed-insight object.

        This is only used if an unexpected exception prevents
        InsightAgent from returning its normal structured result.
        """

        return {
            "success": False,
            "generated_by": "none",
            "summary": "",
            "key_points": [],
            "caution": "",
            "facts": facts or {},
            "error": message,
        }

    # ========================================================
    # SAFE GUARD FAILURE RESPONSE
    # ========================================================

    @staticmethod
    def _failed_guard_result(
        message: str,
        raw_insight: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Create a safe final response if InsightGuard itself
        unexpectedly fails.

        Guard failure must never silently expose unvalidated
        LLM output to the user.
        """

        facts = (
            raw_insight.get(
                "facts",
                {},
            )
            if isinstance(
                raw_insight,
                dict,
            )
            else {}
        )

        if not isinstance(
            facts,
            dict,
        ):
            facts = {}

        final_insight = {
            "success": False,
            "generated_by": "none",
            "summary": (
                "The analytical result was produced, "
                "but the final insight could not be validated."
            ),
            "key_points": [],
            "caution": "",
            "facts": facts,
            "error": message,
        }

        return {
            "valid": False,
            "used_fallback": False,
            "reasons": [
                message,
            ],
            "insight": final_insight,
        }

    # ========================================================
    # NORMALISE FINAL INSIGHT
    # ========================================================

    @staticmethod
    def _normalise_final_insight(
        guarded_result: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Extract a safe final insight dictionary from the
        InsightGuard result.
        """

        final_insight = guarded_result.get(
            "insight"
        )

        if not isinstance(
            final_insight,
            dict,
        ):
            return {
                "success": False,
                "generated_by": "none",
                "summary": (
                    "No validated insight was available."
                ),
                "key_points": [],
                "caution": "",
                "facts": {},
                "error": None,
            }

        return final_insight

# ========================================================
# EXPERIMENT 3 SEMANTIC PARSER CORRECTION
# ========================================================

    def parse(
        self,
        user_query: str,
    ) -> dict[str, Any]:
        """
        Extend the frozen Hybrid parser with Experiment 3
        semantic corrections.

        A request containing both start_date and end_date must
        use time_scope='range' so QueryBuilder applies the
        corresponding SQL date predicates.

        The original HybridSingleAgent remains unchanged.
        """

        parsed = super().parse(
            user_query
        )

        mapping = dict(
            parsed.get(
                "mapping",
                {}
            )
            or {}
        )

        start_date = mapping.get(
            "start_date"
        )

        end_date = mapping.get(
            "end_date"
        )

        time_scope = mapping.get(
            "time_scope",
            "all",
        )

        # Latest semantics take priority.
        if (
            time_scope != "latest"
            and
            start_date is not None
            and
            end_date is not None
        ):
            mapping[
                "time_scope"
            ] = "range"

        parsed[
            "mapping"
        ] = mapping

        return parsed

    
    # ========================================================
    # PUBLIC RUN
    # ========================================================

    def run(
        self,
        user_query: str,
    ) -> dict[str, Any]:
        """
        Execute Experiment 3 end-to-end.

        Returns a state dictionary intentionally similar to the
        Hybrid and Multi-Agent outputs so that T01-T12 can be
        compared using the same evaluation methodology.
        """

        total_start = time.perf_counter()

        errors: list[str] = []
        warnings: list[str] = []

        # ----------------------------------------------------
        # Latency measurements
        # ----------------------------------------------------

        parser_latency = 0.0
        validation_latency = 0.0
        query_builder_latency = 0.0
        sql_latency = 0.0
        result_validation_latency = 0.0
        visualization_latency = 0.0
        insight_latency = 0.0
        insight_guard_latency = 0.0

        # ----------------------------------------------------
        # LLM call accounting
        #
        # The hybrid parser attempts one LLM interpretation.
        # InsightAgent attempts another LLM call only when valid
        # data exists.
        # ----------------------------------------------------

        llm_calls_attempted = 0

        # ====================================================
        # 1. HYBRID PARSER
        # ====================================================

        parser_start = time.perf_counter()

        # HybridSingleAgent.parse() performs:
        #
        # LLM structured interpretation
        # +
        # deterministic correction / fallback.
        #
        # This is the fast interpretation path that Experiment 3
        # is evaluating.

        llm_calls_attempted += 1

        parsed = self.parse(
            user_query
        )

        parser_latency = (
            time.perf_counter()
            - parser_start
        )

        intent = parsed[
            "intent"
        ]

        chart_type = parsed[
            "chart_type"
        ]

        mapping = parsed[
            "mapping"
        ]

        parser_source = parsed[
            "source"
        ]

        parser_error = parsed.get(
            "parser_error"
        )

        if parser_error:

            warnings.append(
                "Hybrid parser used deterministic fallback: "
                f"{parser_error}"
            )

        # ====================================================
        # 2. REQUEST VALIDATION
        # ====================================================

        validation_start = (
            time.perf_counter()
        )

        validation = (
            self.request_validator
            .validate(
                user_query=
                    user_query,

                intent=
                    intent,

                chart_type=
                    chart_type,

                mapping=
                    mapping,
            )
        )

        validation_latency = (
            time.perf_counter()
            - validation_start
        )

        warnings.extend(
            validation.get(
                "warnings",
                [],
            )
        )

        errors.extend(
            validation.get(
                "errors",
                [],
            )
        )

        validated_request = (
            validation[
                "request"
            ]
        )

        # ====================================================
        # INVALID REQUEST
        # ====================================================

        if not validation[
            "valid"
        ]:

            total_latency = (
                time.perf_counter()
                - total_start
            )

            return {
                # --------------------------------------------
                # General
                # --------------------------------------------

                "user_query":
                    user_query,

                "system":
                    "optimized_hybrid",

                "experiment":
                    "experiment_3",

                # --------------------------------------------
                # Parsing
                # --------------------------------------------

                "parser_source":
                    parser_source,

                "parser_error":
                    parser_error,

                "intent":
                    intent,

                "chart_type":
                    chart_type,

                "mapping":
                    mapping,

                # --------------------------------------------
                # Validation
                # --------------------------------------------

                "validation":
                    validation,

                "validated_request":
                    validated_request,

                # --------------------------------------------
                # Query
                # --------------------------------------------

                "sql_query":
                    None,

                "sql_params":
                    [],

                "query_type":
                    None,

                # --------------------------------------------
                # SQL
                # --------------------------------------------

                "execution_success":
                    False,

                "sql_columns":
                    [],

                "sql_results":
                    [],

                "sql_row_count":
                    0,

                "sql_stored_row_count":
                    0,

                "sql_result_truncated":
                    False,

                # --------------------------------------------
                # Result validation
                # --------------------------------------------

                "result_validation":
                    {},

                "result_valid":
                    False,

                "result_status":
                    "invalid_request",

                "result_has_data":
                    False,

                # --------------------------------------------
                # Visualization
                # --------------------------------------------

                "visualization":
                    {},

                "visualization_success":
                    False,

                "chart_created":
                    False,

                "final_chart_type":
                    None,

                "chart_path":
                    None,

                "chart_title":
                    None,

                # --------------------------------------------
                # Insight Agent
                # --------------------------------------------

                "raw_insight":
                    {},

                "insight_generation_success":
                    False,

                # --------------------------------------------
                # Insight Guard
                # --------------------------------------------

                "insight_guard":
                    {},

                "insight_guard_valid":
                    False,

                "insight_used_fallback":
                    False,

                "final_insight":
                    {},

                "insight_generated_by":
                    None,

                "insight_summary":
                    "",

                "insight_key_points":
                    [],

                "insight_caution":
                    "",

                "insight_evidence":
                    {},

                "insight_error":
                    None,

                # --------------------------------------------
                # LLM accounting
                # --------------------------------------------

                "llm_calls_attempted":
                    llm_calls_attempted,

                # --------------------------------------------
                # Latency
                # --------------------------------------------

                "parser_latency":
                    parser_latency,

                "validation_latency":
                    validation_latency,

                "query_builder_latency":
                    query_builder_latency,

                "sql_latency":
                    sql_latency,

                "result_validation_latency":
                    result_validation_latency,

                "visualization_latency":
                    visualization_latency,

                "insight_latency":
                    insight_latency,

                "insight_guard_latency":
                    insight_guard_latency,

                "total_latency":
                    total_latency,

                # --------------------------------------------
                # Diagnostics
                # --------------------------------------------

                "warnings":
                    warnings,

                "errors":
                    errors,
            }

        # ====================================================
        # 3. QUERY BUILDER
        # ====================================================

        query_builder_start = (
            time.perf_counter()
        )

        query_result = (
            self.query_builder
            .build(
                validated_request
            )
        )

        query_builder_latency = (
            time.perf_counter()
            - query_builder_start
        )

        sql = (
            query_result[
                "sql"
            ]
        )

        params = (
            query_result[
                "params"
            ]
        )

        query_type = (
            query_result[
                "query_type"
            ]
        )

        # ====================================================
        # 4. SQL EXECUTION
        # ====================================================

        sql_start = (
            time.perf_counter()
        )

        try:

            execution = (
                self.sql_executor
                .execute(
                    sql=
                        sql,

                    params=
                        params,

                    query_type=
                        query_type,
                )
            )

            execution_success = (
                True
            )

        except Exception as error:

            execution_success = (
                False
            )

            errors.append(
                "SQL execution failed: "
                f"{type(error).__name__}: "
                f"{error}"
            )

            execution = {
                "columns":
                    [],

                "rows":
                    [],

                "row_count":
                    0,

                "stored_row_count":
                    0,

                "truncated":
                    False,
            }

        sql_latency = (
            time.perf_counter()
            - sql_start
        )

        rows = (
            execution.get(
                "rows",
                [],
            )
        )

        row_count = (
            execution.get(
                "row_count",
                0,
            )
        )

        stored_row_count = (
            execution.get(
                "stored_row_count",
                len(
                    rows
                ),
            )
        )

        truncated = bool(
            execution.get(
                "truncated",
                False,
            )
        )

        # ====================================================
        # 5. RESULT VALIDATION
        # ====================================================

        result_validation_start = (
            time.perf_counter()
        )

        result_validation = (
            self.result_validator
            .validate(
                request=
                    validated_request,

                query_type=
                    query_type,

                execution_success=
                    execution_success,

                columns=
                    execution.get(
                        "columns",
                        [],
                    ),

                rows=
                    rows,

                row_count=
                    row_count,

                truncated=
                    truncated,
            )
        )

        result_validation_latency = (
            time.perf_counter()
            - result_validation_start
        )

        warnings.extend(
            result_validation.get(
                "warnings",
                [],
            )
        )

        errors.extend(
            result_validation.get(
                "errors",
                [],
            )
        )

        result_valid = bool(
            result_validation[
                "valid"
            ]
        )

        result_has_data = bool(
            result_validation[
                "has_data"
            ]
        )

        result_status = (
            result_validation[
                "status"
            ]
        )

        # ====================================================
        # 6. VISUALIZATION
        # ====================================================

        visualization_start = (
            time.perf_counter()
        )

        try:

            visualization = (
                self.visualizer
                .create(
                    user_query=
                        user_query,

                    intent=
                        intent,

                    query_type=
                        query_type,

                    metric_name=
                        validated_request.get(
                            "metric_name"
                        ),

                    metric_x=
                        validated_request.get(
                            "metric_x"
                        ),

                    metric_y=
                        validated_request.get(
                            "metric_y"
                        ),

                    rows=
                        rows,

                    result_valid=
                        result_valid,

                    result_has_data=
                        result_has_data,

                    chart_type=
                        chart_type,
                )
            )

        except Exception as error:

            message = (
                "Visualization failed: "
                f"{type(error).__name__}: "
                f"{error}"
            )

            errors.append(
                message
            )

            visualization = {
                "success":
                    False,

                "chart_created":
                    False,

                "chart_type":
                    None,

                "chart_path":
                    None,

                "chart_title":
                    None,

                "warnings":
                    [],

                "error":
                    message,
            }

        visualization_latency = (
            time.perf_counter()
            - visualization_start
        )

        warnings.extend(
            visualization.get(
                "warnings",
                [],
            )
        )

        visualization_error = (
            visualization.get(
                "error"
            )
        )

        if (
            visualization_error
            and
            not any(
                visualization_error
                in existing_error
                for existing_error
                in errors
            )
        ):

            errors.append(
                "Visualization failed: "
                f"{visualization_error}"
            )

        # ====================================================
        # 7. DEDICATED INSIGHT AGENT
        # ====================================================

        insight_start = (
            time.perf_counter()
        )

        # ----------------------------------------------------
        # InsightAgent only needs the ResultValidation warnings,
        # not every pipeline warning.
        # ----------------------------------------------------

        analytical_result_warnings = (
            result_validation.get(
                "warnings",
                [],
            )
        )

        # ----------------------------------------------------
        # If there is valid data, InsightAgent will make the
        # second LLM call.
        #
        # No-data responses are deterministic and therefore do
        # not require an Insight LLM call.
        # ----------------------------------------------------

        should_attempt_insight_llm = (
            result_valid
            and
            result_has_data
            and
            row_count > 0
            and
            bool(
                rows
            )
        )

        if should_attempt_insight_llm:

            llm_calls_attempted += 1

        try:

            raw_insight = (
                self.insight_agent
                .generate(
                    user_query=
                        user_query,

                    request=
                        validated_request,

                    query_type=
                        query_type,

                    rows=
                        rows,

                    row_count=
                        row_count,

                    result_valid=
                        result_valid,

                    result_has_data=
                        result_has_data,

                    result_warnings=
                        analytical_result_warnings,

                    truncated=
                        truncated,
                )
            )

        except Exception as error:

            message = (
                "Insight Agent failed: "
                f"{type(error).__name__}: "
                f"{error}"
            )

            errors.append(
                message
            )

            raw_insight = (
                self._failed_insight(
                    message=
                        message
                )
            )

        insight_latency = (
            time.perf_counter()
            - insight_start
        )

        insight_generation_success = bool(
            raw_insight.get(
                "success",
                False,
            )
        )

        insight_error = (
            raw_insight.get(
                "error"
            )
        )

        # ----------------------------------------------------
        # Preserve InsightAgent diagnostic errors.
        #
        # A deterministic fallback caused by a local model
        # timeout is diagnostic information, not necessarily an
        # end-to-end analytical failure.
        # ----------------------------------------------------

        if (
            not insight_generation_success
            and
            insight_error
        ):

            insight_error_message = (
                "Insight generation failed: "
                f"{insight_error}"
            )

            if (
                insight_error_message
                not in errors
            ):

                errors.append(
                    insight_error_message
                )

        # ====================================================
        # 8. INSIGHT GUARD
        # ====================================================

        insight_guard_start = (
            time.perf_counter()
        )

        try:

            guarded_result = (
                self.insight_guard
                .guard(
                    raw_insight
                )
            )

        except Exception as error:

            message = (
                "Insight Guard failed: "
                f"{type(error).__name__}: "
                f"{error}"
            )

            errors.append(
                message
            )

            guarded_result = (
                self._failed_guard_result(
                    message=
                        message,

                    raw_insight=
                        raw_insight,
                )
            )

        insight_guard_latency = (
            time.perf_counter()
            - insight_guard_start
        )

        # ====================================================
        # 9. FINAL GUARDED INSIGHT
        # ====================================================

        final_insight = (
            self._normalise_final_insight(
                guarded_result
            )
        )

        insight_guard_valid = bool(
            guarded_result.get(
                "valid",
                False,
            )
        )

        insight_used_fallback = bool(
            guarded_result.get(
                "used_fallback",
                False,
            )
        )

        guard_reasons = (
            guarded_result.get(
                "reasons",
                [],
            )
            or []
        )

        # ----------------------------------------------------
        # A guard replacement is a successful reliability
        # mechanism, not an analytical pipeline failure.
        # Record guard reasons as warnings.
        # ----------------------------------------------------

        if insight_used_fallback:

            for reason in guard_reasons:

                warning = (
                    "Insight Guard replaced LLM output: "
                    f"{reason}"
                )

                if warning not in warnings:

                    warnings.append(
                        warning
                    )

        final_summary = str(
            final_insight.get(
                "summary",
                "",
            )
            or ""
        ).strip()

        raw_key_points = (
            final_insight.get(
                "key_points",
                [],
            )
        )

        if not isinstance(
            raw_key_points,
            list,
        ):

            raw_key_points = []

        final_key_points = [
            str(
                point
            ).strip()
            for point in raw_key_points
            if str(
                point
            ).strip()
        ]

        final_caution = str(
            final_insight.get(
                "caution",
                "",
            )
            or ""
        ).strip()

        final_generated_by = (
            final_insight.get(
                "generated_by"
            )
        )

        final_facts = (
            final_insight.get(
                "facts",
                {},
            )
        )

        if not isinstance(
            final_facts,
            dict,
        ):

            final_facts = {}

        # ====================================================
        # 10. TOTAL LATENCY
        # ====================================================

        total_latency = (
            time.perf_counter()
            - total_start
        )

        # ====================================================
        # FINAL EXPERIMENT 3 STATE
        # ====================================================

        return {
            # ------------------------------------------------
            # General
            # ------------------------------------------------

            "user_query":
                user_query,

            "system":
                "optimized_hybrid",

            "experiment":
                "experiment_3",

            # ------------------------------------------------
            # Parsing
            # ------------------------------------------------

            "parser_source":
                parser_source,

            "parser_error":
                parser_error,

            "intent":
                intent,

            "chart_type":
                chart_type,

            "mapping":
                mapping,

            # ------------------------------------------------
            # Validation
            # ------------------------------------------------

            "validation":
                validation,

            "validated_request":
                validated_request,

            # ------------------------------------------------
            # Query
            # ------------------------------------------------

            "sql_query":
                sql,

            "sql_params":
                params,

            "query_type":
                query_type,

            # ------------------------------------------------
            # SQL
            # ------------------------------------------------

            "execution_success":
                execution_success,

            "sql_columns":
                execution.get(
                    "columns",
                    [],
                ),

            "sql_results":
                rows,

            "sql_row_count":
                row_count,

            "sql_stored_row_count":
                stored_row_count,

            "sql_result_truncated":
                truncated,

            # ------------------------------------------------
            # Result validation
            # ------------------------------------------------

            "result_validation":
                result_validation,

            "result_valid":
                result_valid,

            "result_status":
                result_status,

            "result_has_data":
                result_has_data,

            # ------------------------------------------------
            # Visualization
            # ------------------------------------------------

            "visualization":
                visualization,

            "visualization_success":
                bool(
                    visualization.get(
                        "success",
                        False,
                    )
                ),

            "chart_created":
                bool(
                    visualization.get(
                        "chart_created",
                        False,
                    )
                ),

            "final_chart_type":
                visualization.get(
                    "chart_type"
                ),

            "chart_path":
                visualization.get(
                    "chart_path"
                ),

            "chart_title":
                visualization.get(
                    "chart_title"
                ),

            # ------------------------------------------------
            # Raw dedicated Insight Agent output
            # ------------------------------------------------

            "raw_insight":
                raw_insight,

            "insight_generation_success":
                insight_generation_success,

            # ------------------------------------------------
            # Insight Guard
            # ------------------------------------------------

            "insight_guard":
                guarded_result,

            "insight_guard_valid":
                insight_guard_valid,

            "insight_used_fallback":
                insight_used_fallback,

            "insight_guard_reasons":
                guard_reasons,

            # ------------------------------------------------
            # Final user-visible guarded insight
            # ------------------------------------------------

            "final_insight":
                final_insight,

            "insight_generated_by":
                final_generated_by,

            "insight_summary":
                final_summary,

            "insight_key_points":
                final_key_points,

            "insight_caution":
                final_caution,

            "insight_evidence":
                final_facts,

            "insight_error":
                insight_error,

            # ------------------------------------------------
            # Experiment diagnostics
            # ------------------------------------------------

            "llm_calls_attempted":
                llm_calls_attempted,

            # ------------------------------------------------
            # Component latency
            # ------------------------------------------------

            "parser_latency":
                parser_latency,

            "validation_latency":
                validation_latency,

            "query_builder_latency":
                query_builder_latency,

            "sql_latency":
                sql_latency,

            "result_validation_latency":
                result_validation_latency,

            "visualization_latency":
                visualization_latency,

            "insight_latency":
                insight_latency,

            "insight_guard_latency":
                insight_guard_latency,

            "total_latency":
                total_latency,

            # ------------------------------------------------
            # Diagnostics
            # ------------------------------------------------

            "warnings":
                warnings,

            "errors":
                errors,
        }