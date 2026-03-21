from __future__ import annotations

import argparse
from pathlib import Path

from model_registry_helpers import (
    model_registry_metrics_to_long_csv,
    model_registry_to_csv,
    model_registry_to_normalized_csvs,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export model_registry.json into analysis-ready CSV formats."
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=Path("models") / "model_registry.json",
        help="Path to model_registry.json",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results"),
        help="Output directory for generated CSV files",
    )
    parser.add_argument(
        "--test-metrics-only",
        action="store_true",
        help="If set, metrics long CSV includes only metrics with names starting with 'test_'",
    )

    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    flat_csv = args.out_dir / "model_registry_flat.csv"
    metrics_csv = args.out_dir / "model_registry_metrics_long.csv"

    flat_rows = model_registry_to_csv(
        registry_path=args.registry,
        csv_output_path=flat_csv,
        sort_by=["created_at"],
        ascending=True,
    )

    metric_rows = model_registry_metrics_to_long_csv(
        registry_path=args.registry,
        csv_output_path=metrics_csv,
        include_non_test_metrics=not args.test_metrics_only,
    )

    normalized_counts = model_registry_to_normalized_csvs(
        registry_path=args.registry,
        output_dir=args.out_dir,
    )

    print(f"Wrote {len(flat_rows)} rows to {flat_csv}")
    print(f"Wrote {len(metric_rows)} rows to {metrics_csv}")
    for file_name, row_count in normalized_counts.items():
        print(f"Wrote {row_count} rows to {args.out_dir / file_name}")


if __name__ == "__main__":
    main()
