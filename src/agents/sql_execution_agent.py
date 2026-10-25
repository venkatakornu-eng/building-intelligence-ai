from __future__ import annotations

from typing import Any

import pandas as pd

from src.sql_database import SQLDatabase


class SQLExecutionAgent:
    """
    Deterministic SQL execution component.

    Responsibilities:
    - execute parameterised SQL produced by QueryBuilderAgent;
    - return SQL results in a LangGraph-friendly format;
    - prevent excessively large result sets from being stored
      directly in graph state.

    This component does NOT generate SQL.
    """

    def __init__(
        self,
        database: SQLDatabase,
        max_result_rows: int = 10_000,
    ) -> None:

        self.database = database
        self.max_result_rows = max_result_rows

    # ========================================================
    # SAFE VALUE CONVERSION
    # ========================================================

    def _safe_value(
        self,
        value: Any,
    ) -> Any:
        """
        Convert pandas / NumPy values into standard Python
        values suitable for LangGraph state.
        """

        if pd.isna(value):
            return None

        if isinstance(
            value,
            pd.Timestamp,
        ):
            return value.isoformat(
                sep=" "
            )

        # Convert NumPy scalar to normal Python scalar
        if hasattr(
            value,
            "item",
        ):

            try:
                return value.item()

            except Exception:
                pass

        return value

    # ========================================================
    # DATAFRAME → RECORDS
    # ========================================================

    def _to_records(
        self,
        dataframe: pd.DataFrame,
    ) -> list[dict[str, Any]]:
        """
        Convert a pandas DataFrame into serialisable
        dictionaries for LangGraph state.
        """

        records: list[
            dict[str, Any]
        ] = []

        raw_records = dataframe.to_dict(
            orient="records"
        )

        for raw_record in raw_records:

            record = {
                key: self._safe_value(
                    value
                )
                for key, value
                in raw_record.items()
            }

            records.append(
                record
            )

        return records

    # ========================================================
    # SQL EXECUTION
    # ========================================================

    def execute(
        self,
        sql: str,
        params: list[Any],
        query_type: str,
    ) -> dict[str, Any]:
        """
        Execute a validated parameterised SQL query.
        """

        if not sql:
            raise ValueError(
                "SQL query cannot be empty."
            )

        if not query_type:
            raise ValueError(
                "query_type is required."
            )

        # ----------------------------------------------------
        # Execute through SQLDatabase
        # ----------------------------------------------------

        dataframe = (
            self.database.execute_query(
                sql,
                params,
            )
        )

        total_rows = len(
            dataframe
        )

        # ----------------------------------------------------
        # Limit the amount stored in LangGraph state
        # ----------------------------------------------------

        truncated = (
            total_rows
            > self.max_result_rows
        )

        if truncated:

            stored_dataframe = (
                dataframe.head(
                    self.max_result_rows
                )
            )

        else:

            stored_dataframe = dataframe

        records = self._to_records(
            stored_dataframe
        )

        # ----------------------------------------------------
        # Return structured execution result
        # ----------------------------------------------------

        return {
            "success": True,

            "query_type":
                query_type,

            "columns":
                list(
                    dataframe.columns
                ),

            "rows":
                records,

            "row_count":
                total_rows,

            "stored_row_count":
                len(records),

            "truncated":
                truncated,

            "error":
                None,
        }