import json
from pathlib import Path

from src.data_loader import load_room_data
from src.evaluation import (
    evaluate_parser,
    load_evaluation_prompts,
)
from src.nl_parser import NaturalLanguageParser


PROJECT_ROOT = Path(__file__).resolve().parent

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

results, summary = evaluate_parser(
    parser=parser,
    prompts=prompts,
)

output_directory = PROJECT_ROOT / "outputs" / "evaluation"
output_directory.mkdir(
    parents=True,
    exist_ok=True,
)

results_path = output_directory / "hybrid_parser_results.csv"
summary_path = output_directory / "hybrid_parser_summary.json"

results.to_csv(
    results_path,
    index=False,
)

with summary_path.open(
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        summary,
        file,
        indent=4,
    )

print("\nEVALUATION SUMMARY")
print("=" * 60)
print("Total prompts:", summary["total_prompts"])
print("Exact matches:", summary["exact_matches"])
print(
    "Exact-match accuracy:",
    f'{summary["exact_match_accuracy"]}%'
)
print(
    "Average latency:",
    f'{summary["average_latency_seconds"]} seconds'
)

print("\nFIELD ACCURACY")

for field, accuracy in summary["field_accuracy"].items():
    print(f"{field}: {accuracy}%")

print("\nFAILED PROMPTS")

failed = results[
    results["exact_match"] == False
]

if failed.empty:
    print("None")
else:
    print(
        failed[
            [
                "prompt_id",
                "query",
                "error",
            ]
        ].to_string(index=False)
    )

print("\nResults saved:", results_path)
print("Summary saved:", summary_path)