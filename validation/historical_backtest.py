import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    mean_absolute_error
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from historical_simulation import run_historical_simulation


TEST_DATES = [
    "2023-01-15",
    "2023-07-15",
    "2024-01-15",
    "2024-07-15",
    "2025-01-15",
    "2025-07-15",
]


def main():
    all_results = []
    date_summaries = []

    print("=" * 70)
    print("MULTI-DATE HISTORICAL RAINFALL BACKTEST")
    print("=" * 70)

    for date in TEST_DATES:
        print(f"\nRunning simulation for {date}...")

        try:
            results = run_historical_simulation(date)

            if results is None or results.empty:
                print(f"WARNING: No results returned for {date}")
                continue

            results = results.copy()
            results["simulation_date"] = date

            actual = (results["actual_rainfall_mm"] >= 1).astype(int)
            predicted = (
                results["prediction"] == "Rain Expected"
            ).astype(int)

            accuracy = accuracy_score(actual, predicted)
            amount_mae = mean_absolute_error(
                results["actual_rainfall_mm"],
                results["estimated_rainfall_mm"]
            )

            date_summaries.append({
                "date": date,
                "locations": len(results),
                "accuracy": accuracy,
                "amount_mae_mm": amount_mae,
            })

            all_results.append(results)

            print(f"  Locations: {len(results)}")
            print(f"  Occurrence accuracy: {accuracy:.4f}")
            print(f"  Amount MAE: {amount_mae:.4f} mm")

        except Exception as error:
            print(f"ERROR for {date}: {error}")

    if not all_results:
        print("\nNo successful simulations. Backtest stopped.")
        return

    combined = pd.concat(all_results, ignore_index=True)

    actual = (combined["actual_rainfall_mm"] >= 1).astype(int)
    predicted = (
        combined["prediction"] == "Rain Expected"
    ).astype(int)

    print("\n" + "=" * 70)
    print("OVERALL BACKTEST RESULTS")
    print("=" * 70)

    print(f"Successful dates: {len(date_summaries)}")
    print(f"Total location predictions: {len(combined)}")
    print(f"Accuracy:  {accuracy_score(actual, predicted):.4f}")
    print(f"Precision: {precision_score(actual, predicted, zero_division=0):.4f}")
    print(f"Recall:    {recall_score(actual, predicted, zero_division=0):.4f}")
    print(f"F1-score:  {f1_score(actual, predicted, zero_division=0):.4f}")

    amount_mae = mean_absolute_error(
        combined["actual_rainfall_mm"],
        combined["estimated_rainfall_mm"]
    )
    print(f"Amount MAE: {amount_mae:.4f} mm")

    tn, fp, fn, tp = confusion_matrix(
        actual, predicted, labels=[0, 1]
    ).ravel()

    print("\nCONFUSION MATRIX COUNTS")
    print(f"True Negatives:  {tn}")
    print(f"False Positives: {fp}")
    print(f"False Negatives: {fn}")
    print(f"True Positives:  {tp}")

    print("\n" + "=" * 70)
    print("PER-DATE SUMMARY")
    print("=" * 70)

    summary = pd.DataFrame(date_summaries)
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    output_file = "reports/historical_backtest_results.csv"
    combined.to_csv(output_file, index=False)

    summary_file = "reports/historical_backtest_summary.csv"
    summary.to_csv(summary_file, index=False)

    print(f"\nDetailed results saved to: {output_file}")
    print(f"Date summary saved to: {summary_file}")
    print("\nBACKTEST COMPLETED")


if __name__ == "__main__":
    main()

