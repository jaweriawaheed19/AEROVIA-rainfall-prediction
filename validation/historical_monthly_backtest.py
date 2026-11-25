import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    mean_absolute_error
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from historical_simulation import run_historical_simulation


TEST_DATES = [
    f"2025-{month:02d}-15"
    for month in range(1, 13)
]


def main():
    all_results = []
    summaries = []

    print("=" * 75)
    print("12-MONTH HISTORICAL RAINFALL BACKTEST")
    print("=" * 75)

    for date in TEST_DATES:
        print(f"\nRunning: {date}")

        try:
            results = run_historical_simulation(date)

            if results is None or results.empty:
                print("No results. Skipping.")
                continue

            results = results.copy()
            results["simulation_date"] = date

            actual = (results["actual_rainfall_mm"] >= 1).astype(int)
            predicted = (
                results["prediction"] == "Rain Expected"
            ).astype(int)

            summaries.append({
                "date": date,
                "locations": len(results),
                "rainy_locations": int(actual.sum()),
                "dry_locations": int((actual == 0).sum()),
                "predicted_rainy": int(predicted.sum()),
                "accuracy": accuracy_score(actual, predicted),
                "balanced_accuracy": balanced_accuracy_score(actual, predicted),
                "precision": precision_score(actual, predicted, zero_division=0),
                "recall": recall_score(actual, predicted, zero_division=0),
                "f1": f1_score(actual, predicted, zero_division=0),
                "amount_mae_mm": mean_absolute_error(
                    results["actual_rainfall_mm"],
                    results["estimated_rainfall_mm"]
                )
            })

            all_results.append(results)

            print(f"  Locations: {len(results)}")
            print(f"  Actual rainy locations: {int(actual.sum())}")
            print(f"  Accuracy: {accuracy_score(actual, predicted):.4f}")

        except Exception as error:
            print(f"ERROR: {error}")

    if not all_results:
        print("\nNo successful simulations.")
        return

    combined = pd.concat(all_results, ignore_index=True)
    summary = pd.DataFrame(summaries)

    actual = (combined["actual_rainfall_mm"] >= 1).astype(int)
    predicted = (
        combined["prediction"] == "Rain Expected"
    ).astype(int)

    print("\n" + "=" * 75)
    print("OVERALL 12-MONTH RESULTS")
    print("=" * 75)

    print(f"Successful dates: {len(summary)}")
    print(f"Total predictions: {len(combined)}")
    print(f"Actual rainy cases: {int(actual.sum())}")
    print(f"Actual dry cases: {int((actual == 0).sum())}")

    print(f"\nAccuracy:          {accuracy_score(actual, predicted):.4f}")
    print(f"Balanced accuracy: {balanced_accuracy_score(actual, predicted):.4f}")
    print(f"Precision:         {precision_score(actual, predicted, zero_division=0):.4f}")
    print(f"Recall:            {recall_score(actual, predicted, zero_division=0):.4f}")
    print(f"F1-score:          {f1_score(actual, predicted, zero_division=0):.4f}")

    print(f"\nAmount MAE:        {mean_absolute_error(combined['actual_rainfall_mm'], combined['estimated_rainfall_mm']):.4f} mm")

    print("\n" + "=" * 75)
    print("MONTH-BY-MONTH RESULTS")
    print("=" * 75)

    print(summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    ))

    print("\n" + "=" * 75)
    print("BASELINE COMPARISON")
    print("=" * 75)

    always_dry_accuracy = (actual == 0).mean()

    print(f"Always-dry accuracy: {always_dry_accuracy:.4f}")
    print(f"Model accuracy:      {accuracy_score(actual, predicted):.4f}")

    combined.to_csv(
        "reports/historical_12_month_backtest_results.csv",
        index=False
    )

    summary.to_csv(
        "reports/historical_12_month_backtest_summary.csv",
        index=False
    )

    print("\nDetailed results saved.")
    print("12-month backtest completed.")


if __name__ == "__main__":
    main()

