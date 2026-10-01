from __future__ import annotations

from pathlib import Path
from typing import Any, TypedDict

import pandas as pd
from langgraph.graph import END, START, StateGraph

from src.agents.data_mapping_agent import DataMappingAgent
from src.agents.data_mapping_guard import guard_mapping
from src.agents.insight_agent import InsightAgent
from src.agents.insight_guard import InsightGuard
from src.agents.intent_agent import IntentAgent
from src.agents.intent_guard import guard_intent
from src.agents.query_builder_agent import QueryBuilderAgent
from src.agents.request_validation_agent import RequestValidationAgent
from src.agents.result_validation_agent import ResultValidationAgent
from src.agents.sql_execution_agent import SQLExecutionAgent
from src.agents.visualization_agent import VisualizationAgent

from src.sql_database import SQLDatabase


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "room_level_metrics.csv"
)


# ============================================================
# LANGGRAPH SHARED STATE
# ============================================================

class SmartVizState(TypedDict, total=False):

    # --------------------------------------------------------
    # User input
    # --------------------------------------------------------

    user_query: str

    # --------------------------------------------------------
    # Intent Agent
    # --------------------------------------------------------

    raw_intent: str
    intent_confidence: float

    intent: str
    chart_type: str

    intent_source: str
    intent_reason: str

    # --------------------------------------------------------
    # Data Mapping
    # --------------------------------------------------------

    raw_mapping: dict[str, Any]
    mapping_source: str
    mapping: dict[str, Any]

    # --------------------------------------------------------
    # Request Validation
    # --------------------------------------------------------

    validation: dict[str, Any]
    validated_request: dict[str, Any]

    # --------------------------------------------------------
    # Query Builder
    # --------------------------------------------------------

    sql_query: str
    sql_params: list[Any]

    query_type: str
    query_ready: bool

    # --------------------------------------------------------
    # SQL Execution
    # --------------------------------------------------------

    execution_success: bool

    sql_columns: list[str]

    sql_results: list[
        dict[str, Any]
    ]

    sql_row_count: int
    sql_stored_row_count: int

    sql_result_truncated: bool

    # --------------------------------------------------------
    # Result Validation
    # --------------------------------------------------------

    result_validation: dict[str, Any]

    result_valid: bool
    result_status: str
    result_has_data: bool

    # --------------------------------------------------------
    # Visualization
    # --------------------------------------------------------

    visualization: dict[str, Any]

    visualization_success: bool
    chart_created: bool

    final_chart_type: str | None
    chart_path: str | None
    chart_title: str | None

    # --------------------------------------------------------
    # Insight Agent
    # --------------------------------------------------------

    raw_insight: dict[str, Any]
    insight_generation_success: bool

    # --------------------------------------------------------
    # Insight Guard
    # --------------------------------------------------------

    insight_guard: dict[str, Any]

    insight_guard_valid: bool
    insight_used_fallback: bool

    final_insight: dict[str, Any]

    insight_generated_by: str | None
    insight_summary: str
    insight_key_points: list[str]
    insight_caution: str

    # --------------------------------------------------------
    # General
    # --------------------------------------------------------

    errors: list[str]
    warnings: list[str]


# ============================================================
# SMARTVIZ MULTI-AGENT GRAPH
# ============================================================

class SmartVizMultiAgentGraph:
    """
    SmartViz LangGraph workflow.

    Architecture:

    START
      ↓
    Intent Agent
      ↓
    Intent Guard
      ↓
    Data Mapping Agent
      ↓
    Data Mapping Guard
      ↓
    Request Validation
      ↓
      ├── valid
      │      ↓
      │   Query Builder
      │      ↓
      │   SQL Execution
      │      ↓
      │   Result Validation
      │      ↓
      │   Visualization Agent
      │      ↓
      │   Insight Agent
      │      ↓
      │   Insight Guard
      │      ↓
      │     END
      │
      └── invalid
             ↓
         Invalid Request
             ↓
            END

    Important design principle:

    The Insight Agent receives only validated analytical
    results. Deterministic facts are calculated before the
    LLM writes the user-facing explanation.

    The Insight Guard then checks the LLM output and replaces
    unsafe, incomplete or unsupported explanations with a
    deterministic fallback.
    """

    # ========================================================
    # INITIALISATION
    # ========================================================

    def __init__(
        self,
        data_path: str | Path = DEFAULT_DATA_PATH,
        model: str = "deepseek-r1:1.5b",
    ) -> None:

        self.data_path = Path(
            data_path
        )

        self.model = model

        # ----------------------------------------------------
        # CHECK DATASET
        # ----------------------------------------------------

        if not self.data_path.exists():

            raise FileNotFoundError(
                f"Dataset not found: "
                f"{self.data_path}"
            )

        # ----------------------------------------------------
        # LOAD VALID ROOM NAMES
        # ----------------------------------------------------

        room_data = pd.read_csv(
            self.data_path,
            usecols=[
                "display_name"
            ],
        )

        self.available_rooms = (
            room_data[
                "display_name"
            ]
            .dropna()
            .astype(str)
            .drop_duplicates()
            .tolist()
        )

        # ----------------------------------------------------
        # INTENT AGENT
        # ----------------------------------------------------

        self.intent_agent = (
            IntentAgent(
                model=self.model
            )
        )

        # ----------------------------------------------------
        # DATA MAPPING AGENT
        # ----------------------------------------------------

        self.data_mapping_agent = (
            DataMappingAgent(
                available_rooms=
                    self.available_rooms,

                model=
                    self.model,
            )
        )

        # ----------------------------------------------------
        # REQUEST VALIDATION AGENT
        # ----------------------------------------------------

        self.request_validation_agent = (
            RequestValidationAgent(
                available_rooms=
                    self.available_rooms
            )
        )

        # ----------------------------------------------------
        # QUERY BUILDER AGENT
        # ----------------------------------------------------

        self.query_builder_agent = (
            QueryBuilderAgent(
                table_name=
                    "dbo.room_level_metrics"
            )
        )

        # ----------------------------------------------------
        # SQL DATABASE
        # ----------------------------------------------------

        self.database = SQLDatabase(
            server=
                r"localhost",

            database=
                "SmartVizDB",

            driver=
                "ODBC Driver 18 for SQL Server",
        )

        # ----------------------------------------------------
        # SQL EXECUTION AGENT
        # ----------------------------------------------------

        self.sql_execution_agent = (
            SQLExecutionAgent(
                database=
                    self.database,

                max_result_rows=
                    50_000,
            )
        )

        # ----------------------------------------------------
        # RESULT VALIDATION AGENT
        # ----------------------------------------------------

        self.result_validation_agent = (
            ResultValidationAgent()
        )

        # ----------------------------------------------------
        # VISUALIZATION AGENT
        # ----------------------------------------------------

        self.visualization_agent = (
            VisualizationAgent()
        )

        # ----------------------------------------------------
        # INSIGHT AGENT
        # ----------------------------------------------------

        self.insight_agent = (
            InsightAgent(
                model=
                    self.model
            )
        )

        # ----------------------------------------------------
        # INSIGHT GUARD
        # ----------------------------------------------------

        self.insight_guard = (
            InsightGuard()
        )

        # ----------------------------------------------------
        # BUILD LANGGRAPH LAST
        # ----------------------------------------------------

        self.graph = (
            self._build_graph()
        )

    # ========================================================
    # NODE 1 - INTENT AGENT
    # ========================================================

    def _intent_agent_node(
        self,
        state: SmartVizState,
    ) -> dict[str, Any]:

        user_query = (
            state[
                "user_query"
            ]
        )

        try:

            result = (
                self.intent_agent.analyse(
                    user_query
                )
            )

            return {
                "raw_intent":
                    result[
                        "intent"
                    ],

                "intent_confidence":
                    result[
                        "confidence"
                    ],
            }

        except TimeoutError:

            print(
                "\n⚠ Intent Agent timed out. "
                "Using deterministic Intent Guard fallback."
            )

            return {
                "raw_intent":
                    "unknown",

                "intent_confidence":
                    0.0,
            }

        except Exception as error:

            print(
                "\n⚠ Intent Agent failed: "
                f"{type(error).__name__}: "
                f"{error}"
            )

            print(
                "Using deterministic Intent Guard fallback."
            )

            return {
                "raw_intent":
                    "unknown",

                "intent_confidence":
                    0.0,
            }

    # ========================================================
    # NODE 2 - INTENT GUARD
    # ========================================================

    def _intent_guard_node(
        self,
        state: SmartVizState,
    ) -> dict[str, Any]:

        guarded = (
            guard_intent(
                user_query=
                    state[
                        "user_query"
                    ],

                llm_intent=
                    state[
                        "raw_intent"
                    ],
            )
        )

        final_intent = (
            guarded[
                "intent"
            ]
        )

        # ----------------------------------------------------
        # DETERMINISTIC CHART SELECTION
        # ----------------------------------------------------

        chart_type = (
            self.visualization_agent
            .select_chart_type(
                user_query=
                    state[
                        "user_query"
                    ],

                intent=
                    final_intent,
            )
        )

        return {
            "intent":
                final_intent,

            "chart_type":
                chart_type,

            "intent_source":
                guarded[
                    "source"
                ],

            "intent_reason":
                guarded[
                    "reason"
                ],
        }

    # ========================================================
    # NODE 3 - DATA MAPPING AGENT
    # ========================================================

    def _data_mapping_agent_node(
        self,
        state: SmartVizState,
    ) -> dict[str, Any]:

        user_query = (
            state[
                "user_query"
            ]
        )

        intent = (
            state[
                "intent"
            ]
        )

        try:

            raw_mapping = (
                self.data_mapping_agent
                .analyse(
                    user_query=
                        user_query,

                    intent=
                        intent,
                )
            )

            mapping_source = (
                "llm"
            )

        except TimeoutError:

            print(
                "\n⚠ Data Mapping Agent timed out. "
                "Using deterministic mapping fallback."
            )

            raw_mapping = {
                "metric_name":
                    None,

                "metric_x":
                    None,

                "metric_y":
                    None,

                "room":
                    None,

                "room_names":
                    [],

                "aggregation":
                    "mean",

                "frequency":
                    "hourly",

                "start_date":
                    None,

                "end_date":
                    None,

                "top_n":
                    10,

                "ascending":
                    False,

                "threshold":
                    None,

                "time_scope":
                    "all",

                "continuous":
                    False,
            }

            mapping_source = (
                "timeout_fallback"
            )

        except Exception as error:

            print(
                "\n⚠ Data Mapping Agent failed: "
                f"{type(error).__name__}: "
                f"{error}"
            )

            print(
                "Using deterministic mapping fallback."
            )

            raw_mapping = {
                "metric_name":
                    None,

                "metric_x":
                    None,

                "metric_y":
                    None,

                "room":
                    None,

                "room_names":
                    [],

                "aggregation":
                    "mean",

                "frequency":
                    "hourly",

                "start_date":
                    None,

                "end_date":
                    None,

                "top_n":
                    10,

                "ascending":
                    False,

                "threshold":
                    None,

                "time_scope":
                    "all",

                "continuous":
                    False,
            }

            mapping_source = (
                "error_fallback"
            )

        return {
            "raw_mapping":
                raw_mapping,

            "mapping_source":
                mapping_source,
        }

    # ========================================================
    # NODE 4 - DATA MAPPING GUARD
    # ========================================================

    def _data_mapping_guard_node(
        self,
        state: SmartVizState,
    ) -> dict[str, Any]:

        final_mapping = (
            guard_mapping(
                user_query=
                    state[
                        "user_query"
                    ],

                intent=
                    state[
                        "intent"
                    ],

                llm_mapping=
                    state[
                        "raw_mapping"
                    ],

                available_rooms=
                    self.available_rooms,
            )
        )

        return {
            "mapping":
                final_mapping
        }

    # ========================================================
    # NODE 5 - REQUEST VALIDATION
    # ========================================================

    def _request_validation_node(
        self,
        state: SmartVizState,
    ) -> dict[str, Any]:

        validation = (
            self.request_validation_agent
            .validate(
                user_query=
                    state[
                        "user_query"
                    ],

                intent=
                    state[
                        "intent"
                    ],

                chart_type=
                    state[
                        "chart_type"
                    ],

                mapping=
                    state[
                        "mapping"
                    ],
            )
        )

        return {
            "validation":
                validation,

            "validated_request":
                validation[
                    "request"
                ],

            "errors":
                validation[
                    "errors"
                ],

            "warnings":
                validation[
                    "warnings"
                ],
        }

    # ========================================================
    # ROUTE AFTER REQUEST VALIDATION
    # ========================================================

    def _route_after_validation(
        self,
        state: SmartVizState,
    ) -> str:

        if (
            state[
                "validation"
            ][
                "valid"
            ]
        ):

            return "valid"

        return "invalid"

    # ========================================================
    # NODE 6A - QUERY BUILDER
    # ========================================================

    def _query_builder_node(
        self,
        state: SmartVizState,
    ) -> dict[str, Any]:

        request = (
            state[
                "validated_request"
            ]
        )

        query_result = (
            self.query_builder_agent
            .build(
                request
            )
        )

        return {
            "sql_query":
                query_result[
                    "sql"
                ],

            "sql_params":
                query_result[
                    "params"
                ],

            "query_type":
                query_result[
                    "query_type"
                ],

            "query_ready":
                True,
        }

    # ========================================================
    # NODE 6B - INVALID REQUEST
    # ========================================================

    def _invalid_request_node(
        self,
        state: SmartVizState,
    ) -> dict[str, Any]:

        return {
            "query_ready":
                False,

            "execution_success":
                False,

            "result_valid":
                False,

            "result_status":
                "invalid_request",

            "result_has_data":
                False,

            "visualization_success":
                False,

            "chart_created":
                False,

            "insight_generation_success":
                False,

            "insight_guard_valid":
                False,

            "insight_used_fallback":
                False,

            "insight_generated_by":
                None,

            "insight_summary":
                (
                    "No insight was generated because "
                    "the request did not pass validation."
                ),

            "insight_key_points":
                [],

            "insight_caution":
                "",

            "final_insight": {
                "success":
                    False,

                "generated_by":
                    "none",

                "summary":
                    (
                        "No insight was generated because "
                        "the request did not pass validation."
                    ),

                "key_points":
                    [],

                "caution":
                    "",
            },
        }

    # ========================================================
    # NODE 7 - SQL EXECUTION
    # ========================================================

    def _sql_execution_node(
        self,
        state: SmartVizState,
    ) -> dict[str, Any]:

        print(
            "\n[SQL EXECUTION]"
        )

        # ----------------------------------------------------
        # SAFETY
        # ----------------------------------------------------

        if not state.get(
            "query_ready",
            False,
        ):

            message = (
                "SQL execution skipped because "
                "query_ready is False."
            )

            errors = list(
                state.get(
                    "errors",
                    [],
                )
            )

            errors.append(
                message
            )

            return {
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

                "errors":
                    errors,
            }

        sql = (
            state[
                "sql_query"
            ]
        )

        params = (
            state.get(
                "sql_params",
                [],
            )
        )

        query_type = (
            state[
                "query_type"
            ]
        )

        # ----------------------------------------------------
        # EXECUTE SQL
        # ----------------------------------------------------

        try:

            result = (
                self.sql_execution_agent
                .execute(
                    sql=
                        sql,

                    params=
                        params,

                    query_type=
                        query_type,
                )
            )

            print(
                "✅ SQL execution successful"
            )

            print(
                f"Query type: "
                f"{result['query_type']}"
            )

            print(
                f"Rows returned: "
                f"{result['row_count']:,}"
            )

            print(
                f"Rows stored: "
                f"{result['stored_row_count']:,}"
            )

            if result[
                "truncated"
            ]:

                print(
                    "⚠ SQL result truncated "
                    "before storing in graph state."
                )

            return {
                "execution_success":
                    True,

                "sql_columns":
                    result[
                        "columns"
                    ],

                "sql_results":
                    result[
                        "rows"
                    ],

                "sql_row_count":
                    result[
                        "row_count"
                    ],

                "sql_stored_row_count":
                    result[
                        "stored_row_count"
                    ],

                "sql_result_truncated":
                    result[
                        "truncated"
                    ],
            }

        except Exception as error:

            message = (
                "SQL execution failed: "
                f"{type(error).__name__}: "
                f"{error}"
            )

            print(
                f"❌ {message}"
            )

            errors = list(
                state.get(
                    "errors",
                    [],
                )
            )

            errors.append(
                message
            )

            return {
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

                "errors":
                    errors,
            }

    # ========================================================
    # NODE 8 - RESULT VALIDATION
    # ========================================================

    def _result_validation_node(
        self,
        state: SmartVizState,
    ) -> dict[str, Any]:

        print(
            "\n[RESULT VALIDATION]"
        )

        result = (
            self.result_validation_agent
            .validate(
                request=
                    state[
                        "validated_request"
                    ],

                query_type=
                    state[
                        "query_type"
                    ],

                execution_success=
                    state.get(
                        "execution_success",
                        False,
                    ),

                columns=
                    state.get(
                        "sql_columns",
                        [],
                    ),

                rows=
                    state.get(
                        "sql_results",
                        [],
                    ),

                row_count=
                    state.get(
                        "sql_row_count",
                        0,
                    ),

                truncated=
                    state.get(
                        "sql_result_truncated",
                        False,
                    ),
            )
        )

        print(
            f"Status: "
            f"{result['status']}"
        )

        print(
            f"Valid result: "
            f"{result['valid']}"
        )

        print(
            f"Has data: "
            f"{result['has_data']}"
        )

        # ----------------------------------------------------
        # MERGE ERRORS + WARNINGS
        # ----------------------------------------------------

        errors = list(
            state.get(
                "errors",
                [],
            )
        )

        warnings = list(
            state.get(
                "warnings",
                [],
            )
        )

        errors.extend(
            result[
                "errors"
            ]
        )

        warnings.extend(
            result[
                "warnings"
            ]
        )

        return {
            "result_validation":
                result,

            "result_valid":
                result[
                    "valid"
                ],

            "result_status":
                result[
                    "status"
                ],

            "result_has_data":
                result[
                    "has_data"
                ],

            "errors":
                errors,

            "warnings":
                warnings,
        }

    # ========================================================
    # NODE 9 - VISUALIZATION AGENT
    # ========================================================

    def _visualization_node(
        self,
        state: SmartVizState,
    ) -> dict[str, Any]:

        print(
            "\n[VISUALIZATION]"
        )

        mapping = (
            state[
                "mapping"
            ]
        )

        result = (
            self.visualization_agent
            .create(
                user_query=
                    state[
                        "user_query"
                    ],

                intent=
                    state[
                        "intent"
                    ],

                query_type=
                    state[
                        "query_type"
                    ],

                metric_name=
                    mapping.get(
                        "metric_name"
                    ),

                metric_x=
                    mapping.get(
                        "metric_x"
                    ),

                metric_y=
                    mapping.get(
                        "metric_y"
                    ),

                rows=
                    state.get(
                        "sql_results",
                        [],
                    ),

                result_valid=
                    state.get(
                        "result_valid",
                        False,
                    ),

                result_has_data=
                    state.get(
                        "result_has_data",
                        False,
                    ),

                chart_type=
                    state.get(
                        "chart_type"
                    ),
            )
        )

        print(
            f"Visualization success: "
            f"{result['success']}"
        )

        print(
            f"Chart created: "
            f"{result['chart_created']}"
        )

        if result[
            "chart_created"
        ]:

            print(
                f"Chart type: "
                f"{result['chart_type']}"
            )

            print(
                f"Chart title: "
                f"{result.get('chart_title')}"
            )

            print(
                f"Chart path: "
                f"{result['chart_path']}"
            )

        # ----------------------------------------------------
        # MERGE WARNINGS
        # ----------------------------------------------------

        warnings = list(
            state.get(
                "warnings",
                [],
            )
        )

        warnings.extend(
            result.get(
                "warnings",
                [],
            )
        )

        # ----------------------------------------------------
        # MERGE ERRORS
        # ----------------------------------------------------

        errors = list(
            state.get(
                "errors",
                [],
            )
        )

        visualization_error = (
            result.get(
                "error"
            )
        )

        if visualization_error:

            errors.append(
                "Visualization failed: "
                f"{visualization_error}"
            )

        return {
            "visualization":
                result,

            "visualization_success":
                result[
                    "success"
                ],

            "chart_created":
                result[
                    "chart_created"
                ],

            "final_chart_type":
                result.get(
                    "chart_type"
                ),

            "chart_path":
                result.get(
                    "chart_path"
                ),

            "chart_title":
                result.get(
                    "chart_title"
                ),

            "warnings":
                warnings,

            "errors":
                errors,
        }

    # ========================================================
    # NODE 10 - INSIGHT AGENT
    # ========================================================

    def _insight_agent_node(
        self,
        state: SmartVizState,
    ) -> dict[str, Any]:

        print(
            "\n[INSIGHT AGENT]"
        )

        # ----------------------------------------------------
        # RESULT VALIDATION WARNINGS ONLY
        #
        # These are the warnings directly relevant to the
        # analytical result, such as result truncation.
        # ----------------------------------------------------

        result_validation = (
            state.get(
                "result_validation",
                {},
            )
        )

        result_warnings = (
            result_validation.get(
                "warnings",
                [],
            )
            if isinstance(
                result_validation,
                dict,
            )
            else []
        )

        # ----------------------------------------------------
        # GENERATE INSIGHT
        # ----------------------------------------------------

        try:

            insight = (
                self.insight_agent
                .generate(
                    user_query=
                        state[
                            "user_query"
                        ],

                    request=
                        state[
                            "validated_request"
                        ],

                    query_type=
                        state[
                            "query_type"
                        ],

                    rows=
                        state.get(
                            "sql_results",
                            [],
                        ),

                    row_count=
                        state.get(
                            "sql_row_count",
                            0,
                        ),

                    result_valid=
                        state.get(
                            "result_valid",
                            False,
                        ),

                    result_has_data=
                        state.get(
                            "result_has_data",
                            False,
                        ),

                    result_warnings=
                        result_warnings,

                    truncated=
                        state.get(
                            "sql_result_truncated",
                            False,
                        ),
                )
            )

        except Exception as error:

            message = (
                "Insight Agent failed: "
                f"{type(error).__name__}: "
                f"{error}"
            )

            print(
                f"❌ {message}"
            )

            errors = list(
                state.get(
                    "errors",
                    [],
                )
            )

            errors.append(
                message
            )

            insight = {
                "success":
                    False,

                "generated_by":
                    "none",

                "summary":
                    "",

                "key_points":
                    [],

                "caution":
                    "",

                "facts":
                    {},

                "error":
                    message,
            }

            return {
                "raw_insight":
                    insight,

                "insight_generation_success":
                    False,

                "errors":
                    errors,
            }

        # ----------------------------------------------------
        # PRINT RESULT
        # ----------------------------------------------------

        print(
            f"Insight generation success: "
            f"{insight.get('success')}"
        )

        print(
            f"Generated by: "
            f"{insight.get('generated_by')}"
        )

        if insight.get(
            "summary"
        ):

            print(
                "Raw insight summary:"
            )

            print(
                insight[
                    "summary"
                ]
            )

        # ----------------------------------------------------
        # MERGE ERROR IF PRESENT
        # ----------------------------------------------------

        errors = list(
            state.get(
                "errors",
                [],
            )
        )

        insight_error = (
            insight.get(
                "error"
            )
        )

        # Do not treat deterministic fallback from an Ollama
        # timeout as a fatal graph failure. Preserve the error
        # only as diagnostic information.
        if (
            not insight.get(
                "success",
                False,
            )
            and
            insight_error
        ):

            errors.append(
                "Insight generation failed: "
                f"{insight_error}"
            )

        return {
            "raw_insight":
                insight,

            "insight_generation_success":
                bool(
                    insight.get(
                        "success",
                        False,
                    )
                ),

            "errors":
                errors,
        }

    # ========================================================
    # NODE 11 - INSIGHT GUARD
    # ========================================================

    def _insight_guard_node(
        self,
        state: SmartVizState,
    ) -> dict[str, Any]:

        print(
            "\n[INSIGHT GUARD]"
        )

        raw_insight = (
            state.get(
                "raw_insight",
                {},
            )
        )

        # ----------------------------------------------------
        # RUN DETERMINISTIC GUARD
        # ----------------------------------------------------

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

            print(
                f"❌ {message}"
            )

            errors = list(
                state.get(
                    "errors",
                    [],
                )
            )

            errors.append(
                message
            )

            final_insight = {
                "success":
                    False,

                "generated_by":
                    "none",

                "summary":
                    (
                        "The analytical result was produced, "
                        "but the final insight could not be "
                        "validated."
                    ),

                "key_points":
                    [],

                "caution":
                    "",

                "facts":
                    raw_insight.get(
                        "facts",
                        {},
                    )
                    if isinstance(
                        raw_insight,
                        dict,
                    )
                    else {},

                "error":
                    message,
            }

            return {
                "insight_guard": {
                    "valid":
                        False,

                    "used_fallback":
                        False,

                    "reasons": [
                        message
                    ],

                    "insight":
                        final_insight,
                },

                "insight_guard_valid":
                    False,

                "insight_used_fallback":
                    False,

                "final_insight":
                    final_insight,

                "insight_generated_by":
                    "none",

                "insight_summary":
                    final_insight[
                        "summary"
                    ],

                "insight_key_points":
                    [],

                "insight_caution":
                    "",

                "errors":
                    errors,
            }

        # ----------------------------------------------------
        # GET FINAL GUARDED INSIGHT
        # ----------------------------------------------------

        final_insight = (
            guarded_result.get(
                "insight"
            )
        )

        if not isinstance(
            final_insight,
            dict,
        ):

            final_insight = {
                "success":
                    False,

                "generated_by":
                    "none",

                "summary":
                    (
                        "No validated insight "
                        "was available."
                    ),

                "key_points":
                    [],

                "caution":
                    "",

                "facts":
                    {},

                "error":
                    None,
            }

        # ----------------------------------------------------
        # PRINT GUARD RESULT
        # ----------------------------------------------------

        print(
            f"Guard valid: "
            f"{guarded_result.get('valid')}"
        )

        print(
            f"Used deterministic fallback: "
            f"{guarded_result.get('used_fallback')}"
        )

        reasons = (
            guarded_result.get(
                "reasons",
                [],
            )
        )

        if reasons:

            print(
                "Guard reasons:"
            )

            for reason in reasons:

                print(
                    f"- {reason}"
                )

        print(
            f"Final insight generated by: "
            f"{final_insight.get('generated_by')}"
        )

        summary = str(
            final_insight.get(
                "summary",
                "",
            )
        ).strip()

        key_points = (
            final_insight.get(
                "key_points",
                [],
            )
        )

        if not isinstance(
            key_points,
            list,
        ):

            key_points = []

        clean_key_points = [
            str(
                point
            ).strip()
            for point in key_points
            if str(
                point
            ).strip()
        ]

        caution = str(
            final_insight.get(
                "caution",
                "",
            )
        ).strip()

        # ----------------------------------------------------
        # PRINT FINAL INSIGHT
        # ----------------------------------------------------

        if summary:

            print(
                "Final insight summary:"
            )

            print(
                summary
            )

        if clean_key_points:

            print(
                "Final key points:"
            )

            for point in clean_key_points:

                print(
                    f"- {point}"
                )

        if caution:

            print(
                "Caution:"
            )

            print(
                caution
            )

        # ----------------------------------------------------
        # RETAIN GUARD REASONS AS WARNINGS
        #
        # A replacement is not a pipeline failure. It is an
        # expected reliability mechanism, so record it as a
        # warning rather than as an error.
        # ----------------------------------------------------

        warnings = list(
            state.get(
                "warnings",
                [],
            )
        )

        if guarded_result.get(
            "used_fallback",
            False,
        ):

            for reason in reasons:

                warning = (
                    "Insight Guard replaced LLM output: "
                    f"{reason}"
                )

                if warning not in warnings:

                    warnings.append(
                        warning
                    )

        return {
            "insight_guard":
                guarded_result,

            "insight_guard_valid":
                bool(
                    guarded_result.get(
                        "valid",
                        False,
                    )
                ),

            "insight_used_fallback":
                bool(
                    guarded_result.get(
                        "used_fallback",
                        False,
                    )
                ),

            "final_insight":
                final_insight,

            "insight_generated_by":
                final_insight.get(
                    "generated_by"
                ),

            "insight_summary":
                summary,

            "insight_key_points":
                clean_key_points,

            "insight_caution":
                caution,

            "warnings":
                warnings,
        }

    # ========================================================
    # BUILD LANGGRAPH
    # ========================================================

    def _build_graph(
        self,
    ):

        builder = (
            StateGraph(
                SmartVizState
            )
        )

        # ====================================================
        # REGISTER NODES
        # ====================================================

        builder.add_node(
            "intent_agent",
            self._intent_agent_node,
        )

        builder.add_node(
            "intent_guard",
            self._intent_guard_node,
        )

        builder.add_node(
            "data_mapping_agent",
            self._data_mapping_agent_node,
        )

        builder.add_node(
            "data_mapping_guard",
            self._data_mapping_guard_node,
        )

        builder.add_node(
            "request_validation",
            self._request_validation_node,
        )

        builder.add_node(
            "query_builder",
            self._query_builder_node,
        )

        builder.add_node(
            "sql_execution",
            self._sql_execution_node,
        )

        builder.add_node(
            "result_validation",
            self._result_validation_node,
        )

        builder.add_node(
            "visualization",
            self._visualization_node,
        )

        builder.add_node(
            "insight_agent",
            self._insight_agent_node,
        )

        builder.add_node(
            "insight_guard",
            self._insight_guard_node,
        )

        builder.add_node(
            "invalid_request",
            self._invalid_request_node,
        )

        # ====================================================
        # MAIN EDGES
        # ====================================================

        builder.add_edge(
            START,
            "intent_agent",
        )

        builder.add_edge(
            "intent_agent",
            "intent_guard",
        )

        builder.add_edge(
            "intent_guard",
            "data_mapping_agent",
        )

        builder.add_edge(
            "data_mapping_agent",
            "data_mapping_guard",
        )

        builder.add_edge(
            "data_mapping_guard",
            "request_validation",
        )

        # ====================================================
        # REQUEST VALIDATION ROUTING
        # ====================================================

        builder.add_conditional_edges(
            "request_validation",

            self._route_after_validation,

            {
                "valid":
                    "query_builder",

                "invalid":
                    "invalid_request",
            },
        )

        # ====================================================
        # SQL + VISUALIZATION + INSIGHT PIPELINE
        # ====================================================

        builder.add_edge(
            "query_builder",
            "sql_execution",
        )

        builder.add_edge(
            "sql_execution",
            "result_validation",
        )

        builder.add_edge(
            "result_validation",
            "visualization",
        )

        builder.add_edge(
            "visualization",
            "insight_agent",
        )

        builder.add_edge(
            "insight_agent",
            "insight_guard",
        )

        builder.add_edge(
            "insight_guard",
            END,
        )

        # ====================================================
        # INVALID REQUEST
        # ====================================================

        builder.add_edge(
            "invalid_request",
            END,
        )

        # ====================================================
        # COMPILE
        # ====================================================

        return (
            builder.compile()
        )

    # ========================================================
    # PUBLIC RUN
    # ========================================================

    def run(
        self,
        user_query: str,
    ) -> SmartVizState:

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

        initial_state: SmartVizState = {

            "user_query":
                user_query,

            # ------------------------------------------------
            # General
            # ------------------------------------------------

            "errors":
                [],

            "warnings":
                [],

            # ------------------------------------------------
            # Query
            # ------------------------------------------------

            "query_ready":
                False,

            # ------------------------------------------------
            # SQL
            # ------------------------------------------------

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

            # ------------------------------------------------
            # Result validation
            # ------------------------------------------------

            "result_valid":
                False,

            "result_has_data":
                False,

            # ------------------------------------------------
            # Visualization
            # ------------------------------------------------

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

            # ------------------------------------------------
            # Insight
            # ------------------------------------------------

            "insight_generation_success":
                False,

            "insight_guard_valid":
                False,

            "insight_used_fallback":
                False,

            "insight_generated_by":
                None,

            "insight_summary":
                "",

            "insight_key_points":
                [],

            "insight_caution":
                "",
        }

        result = (
            self.graph.invoke(
                initial_state
            )
        )

        return result