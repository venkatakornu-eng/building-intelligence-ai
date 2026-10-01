from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pandas as pd

from src.nl_parser import NaturalLanguageParser


EVALUATED_FIELDS = [
    "intent",
    "metric_name",
    "room",
    "room_names",
    "aggregation",
    "frequency",
    "top_n",
    "ascending",
    "threshold",
    "chart_type",
]


def load_evaluation_prompts(
    path: str | Path,
) -> list[dict[str, Any]]:
    """Load evaluation prompts and expected requests."""

    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"Evaluation file was not found: {file_path}"
        )

    with file_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        prompts = json.load(file)

    if not isinstance(prompts, list):
        raise ValueError(
            "Evaluation prompts must be stored as a JSON list."
        )

    return prompts


def normalise_value(
    field: str,
    value: Any,
) -> Any:
    """Normalise values before comparison."""

    if field == "room_names":
        if not isinstance(value, list):
            return []

        return [
            str(room).strip().casefold()
            for room in value
        ]

    if isinstance(value, str):
        return value.strip().casefold()

    if field == "threshold" and value is not None:
        return float(value)

    return value

def evaluate_parser(
        parser: NaturalLanguageParser,
        prompts: list[dict[str, Any]],
        parse_method: str = "parse",
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        """
        Evaluate one parser method.

        parse_method:
            parse_baseline -> LLM baseline
            parse          -> hybrid parser
        """

        if not hasattr(parser, parse_method):
            raise AttributeError(
                f"Parser does not contain method: {parse_method}"
            )

        parse_function = getattr(
            parser,
            parse_method,
        )

        rows = []
        total_latency = 0.0

        for item in prompts:
            prompt_id = item["id"]
            query = item["query"]
            expected = item["expected"]

            start_time = time.perf_counter()

            try:
                predicted = parse_function(query)
                error_message = None

            except Exception as error:
                predicted = {}
                error_message = str(error)

            latency = time.perf_counter() - start_time
            total_latency += latency

            field_results = {}

            for field in EVALUATED_FIELDS:
                expected_value = normalise_value(
                    field,
                    expected.get(field),
                )

                predicted_value = normalise_value(
                    field,
                    predicted.get(field),
                )

                field_results[field] = (
                    expected_value == predicted_value
                )

            exact_match = all(field_results.values())

            row = {
                "prompt_id": prompt_id,
                "query": query,
                "exact_match": exact_match,
                "latency_seconds": round(latency, 3),
                "error": error_message,
            }

            for field, is_correct in field_results.items():
                row[f"{field}_correct"] = is_correct

            rows.append(row)

        results = pd.DataFrame(rows)

        field_accuracy = {}

        for field in EVALUATED_FIELDS:
            column = f"{field}_correct"

            field_accuracy[field] = round(
                results[column].mean() * 100,
                2,
            )

        summary = {
            "parser_method": parse_method,
            "total_prompts": len(results),
            "exact_matches": int(
                results["exact_match"].sum()
            ),
            "exact_match_accuracy": round(
                results["exact_match"].mean() * 100,
                2,
            ),
            "average_latency_seconds": round(
                total_latency / len(results),
                3,
            ),
            "field_accuracy": field_accuracy,
        }

        return results, summary