from __future__ import annotations

import os
from typing import Any

import pandas as pd
import pyodbc


class SQLDatabase:
    """
    SQL Server connection and execution layer
    for the SmartViz multi-agent system.
    """

    def __init__(
        self,
        server: str,
        database: str,
        driver: str | None = None,
        trusted_connection: bool = True,
    ) -> None:

        self.server = server
        self.database = database
        self.trusted_connection = trusted_connection

        # ------------------------------------------------
        # Automatically select SQL Server ODBC driver
        # ------------------------------------------------

        if driver is None:
            driver = self._find_driver()

        self.driver = driver

    def _find_driver(self) -> str:
        """
        Select the newest available SQL Server ODBC driver.
        """

        available = pyodbc.drivers()

        preferred = [
            "ODBC Driver 18 for SQL Server",
            "ODBC Driver 17 for SQL Server",
        ]

        for driver in preferred:

            if driver in available:
                return driver

        raise RuntimeError(
            "No supported SQL Server ODBC driver was found. "
            f"Installed drivers: {available}"
        )

    def _connection_string(self) -> str:
        """
        Build the SQL Server connection string.
        """

        if self.trusted_connection:

            return (
                f"DRIVER={{{self.driver}}};"
                f"SERVER={self.server};"
                f"DATABASE={self.database};"
                "Trusted_Connection=yes;"
                "TrustServerCertificate=yes;"
            )

        raise NotImplementedError(
            "Only Windows Authentication is currently configured."
        )

    def connect(self) -> pyodbc.Connection:
        """
        Open a SQL Server connection.
        """

        return pyodbc.connect(
            self._connection_string(),
            timeout=10,
        )

    def test_connection(self) -> bool:
        """
        Verify that SQL Server can be reached.
        """

        with self.connect() as connection:

            cursor = connection.cursor()

            cursor.execute(
                "SELECT 1"
            )

            result = cursor.fetchone()

            return (
                result is not None
                and result[0] == 1
            )

    def list_tables(self) -> pd.DataFrame:
        """
        Return all user tables in the current database.
        """

        sql = """
SELECT
    TABLE_SCHEMA,
    TABLE_NAME
FROM INFORMATION_SCHEMA.TABLES
WHERE TABLE_TYPE = 'BASE TABLE'
ORDER BY
    TABLE_SCHEMA,
    TABLE_NAME;
"""

        with self.connect() as connection:

            return pd.read_sql_query(
                sql,
                connection,
            )

    def list_columns(
        self,
        schema_name: str,
        table_name: str,
    ) -> pd.DataFrame:
        """
        Return column information for a table.
        """

        sql = """
SELECT
    COLUMN_NAME,
    DATA_TYPE,
    IS_NULLABLE
FROM INFORMATION_SCHEMA.COLUMNS
WHERE
    TABLE_SCHEMA = ?
    AND TABLE_NAME = ?
ORDER BY
    ORDINAL_POSITION;
"""

        with self.connect() as connection:

            return pd.read_sql_query(
                sql,
                connection,
                params=[
                    schema_name,
                    table_name,
                ],
            )

    def execute_query(
        self,
        sql: str,
        params: list[Any] | None = None,
    ) -> pd.DataFrame:
        """
        Execute a parameterised SELECT query
        and return the result as a DataFrame.
        """

        if params is None:
            params = []

        cleaned = sql.strip().casefold()

        # ------------------------------------------------
        # Basic final safety check
        # ------------------------------------------------

        allowed_start = (
            cleaned.startswith("select")
            or cleaned.startswith("with")
        )

        if not allowed_start:
            raise ValueError(
                "Only SELECT/CTE queries are permitted."
            )

        forbidden = [
            " drop ",
            " delete ",
            " update ",
            " insert ",
            " alter ",
            " truncate ",
            " create ",
            " merge ",
            " execute ",
            " exec ",
        ]

        padded = f" {cleaned} "

        if any(
            keyword in padded
            for keyword in forbidden
        ):
            raise ValueError(
                "Potentially unsafe SQL command detected."
            )

        with self.connect() as connection:

            result = pd.read_sql_query(
                sql,
                connection,
                params=params,
            )

        return result