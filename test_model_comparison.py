import json
from pathlib import Path

import pandas as pd

from src.data_loader import load_room_data
from src.evaluation import (
    evaluate_parser,
    load_evaluation_prompts,
)
from src.nl_parser import NaturalLanguageParser


PROJECT_ROOT = Path(__file__).resolve().parent

OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "outputs"
    / "evaluation"
    / "comparison"
)

OUTPUT_DIRECTORY.mkdir(
    parents=True,
    exist_ok=True,
)


# -----------------------------------------------------
# Load data and parser
# -----------------------------------------------------

data = load_room_data(
    PROJECT_ROOT
    / "data"
    / "processed"
    / "room_level_metrics.csv"
)

parser = NaturalLanguageParser(
    data=data,
    model="deepseek-r1:1.5b",
)

prompts = load_evaluation_prompts(
    PROJECT_ROOT
    / "evaluation"
    / "evaluation_prompts.json"
)


# -----------------------------------------------------
# Warm up the local model
# -----------------------------------------------------

print("Warming up the local model...")

try:
    parser.parse_baseline(
        "Show the CO2 trend for Seminar Room 3."
    )

except Exception as error:
    print("Warm-up warning:", error)


# -----------------------------------------------------
# Evaluate baseline
# -----------------------------------------------------

print("\nEvaluating single-LLM baseline...")

baseline_results, baseline_summary = evaluate_parser(
    parser=parser,
    prompts=prompts,
    parse_method="parse_baseline",
)


# -----------------------------------------------------
# Evaluate hybrid parser
# -----------------------------------------------------

print("Evaluating hybrid parser...")

hybrid_results, hybrid_summary = evaluate_parser(
    parser=parser,
    prompts=prompts,
    parse_method="parse",
)


# -----------------------------------------------------
# Save detailed results
# -----------------------------------------------------

baseline_results.to_csv(
    OUTPUT_DIRECTORY / "baseline_results.csv",
    index=False,
)

hybrid_results.to_csv(
    OUTPUT_DIRECTORY / "hybrid_results.csv",
    index=False,
)

with (
    OUTPUT_DIRECTORY / "baseline_summary.json"
).open("w", encoding="utf-8") as file:
    json.dump(
        baseline_summary,
        file,
        indent=4,
    )

with (
    OUTPUT_DIRECTORY / "hybrid_summary.json"
).open("w", encoding="utf-8") as file:
    json.dump(
        hybrid_summary,
        file,
        indent=4,
    )


# -----------------------------------------------------
# Build summary comparison
# -----------------------------------------------------

accuracy_improvement = (
    hybrid_summary["exact_match_accuracy"]
    - baseline_summary["exact_match_accuracy"]
)

latency_difference = (
    hybrid_summary["average_latency_seconds"]
    - baseline_summary["average_latency_seconds"]
)

comparison_rows = [
    {
        "system": "Single-LLM baseline",
        "exact_matches": baseline_summary[
            "exact_matches"
        ],
        "total_prompts": baseline_summary[
            "total_prompts"
        ],
        "exact_match_accuracy": baseline_summary[
            "exact_match_accuracy"
        ],
        "average_latency_seconds": baseline_summary[
            "average_latency_seconds"
        ],
    },
    {
        "system": "Hybrid parser",
        "exact_matches": hybrid_summary[
            "exact_matches"
        ],
        "total_prompts": hybrid_summary[
            "total_prompts"
        ],
        "exact_match_accuracy": hybrid_summary[
            "exact_match_accuracy"
        ],
        "average_latency_seconds": hybrid_summary[
            "average_latency_seconds"
        ],
    },
]

comparison = pd.DataFrame(
    comparison_rows
)

comparison.to_csv(
    OUTPUT_DIRECTORY / "model_comparison.csv",
    index=False,
)


# -----------------------------------------------------
# Field-level comparison
# -----------------------------------------------------

field_rows = []

for field in baseline_summary["field_accuracy"]:
    field_rows.append(
        {
            "field": field,
            "baseline_accuracy": baseline_summary[
                "field_accuracy"
            ][field],
            "hybrid_accuracy": hybrid_summary[
                "field_accuracy"
            ][field],
            "improvement_percentage_points": round(
                hybrid_summary["field_accuracy"][field]
                - baseline_summary["field_accuracy"][field],
                2,
            ),
        }
    )

field_comparison = pd.DataFrame(
    field_rows
)

field_comparison.to_csv(
    OUTPUT_DIRECTORY / "field_comparison.csv",
    index=False,
)


# -----------------------------------------------------
# Display summary
# -----------------------------------------------------

print("\nMODEL COMPARISON")
print("=" * 70)
print(comparison.to_string(index=False))

print("\nAccuracy improvement:")
print(
    f"{accuracy_improvement:.2f} percentage points"
)

print("\nLatency difference:")
print(
    f"{latency_difference:.3f} seconds"
)

print("\nFIELD-LEVEL COMPARISON")
print("=" * 70)
print(field_comparison.to_string(index=False))


print("\nBASELINE FAILED PROMPTS")

baseline_failed = baseline_results[
    baseline_results["exact_match"] == False
]

if baseline_failed.empty:
    print("None")
else:
    print(
        baseline_failed[
            [
                "prompt_id",
                "query",
                "error",
            ]
        ].to_string(index=False)
    )


print("\nHYBRID FAILED PROMPTS")

hybrid_failed = hybrid_results[
    hybrid_results["exact_match"] == False
]

if hybrid_failed.empty:
    print("None")
else:
    print(
        hybrid_failed[
            [
                "prompt_id",
                "query",
                "error",
            ]
        ].to_string(index=False)
    )


print(
    "\nResults saved:",
    OUTPUT_DIRECTORY,
)