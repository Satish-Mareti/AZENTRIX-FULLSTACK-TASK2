from __future__ import annotations

import argparse
from pathlib import Path

from churn_pipeline import RANDOM_STATE, train_end_to_end


def build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Train and evaluate the Telco Customer Churn pipeline.")
	parser.add_argument(
		"--data-path",
		type=Path,
		default=Path("data") / "Telco-Customer-Churn.csv",
		help="Path to the Telco churn CSV file.",
	)
	parser.add_argument(
		"--model-path",
		type=Path,
		default=Path("artifacts") / "telco_churn_model.joblib",
		help="Where to save the trained model bundle.",
	)
	parser.add_argument(
		"--eda-path",
		type=Path,
		default=Path("artifacts") / "eda_summary.md",
		help="Where to save the EDA summary.",
	)
	parser.add_argument(
		"--comparison-path",
		type=Path,
		default=Path("artifacts") / "model_comparison.csv",
		help="Where to save the cross-validation comparison table.",
	)
	parser.add_argument("--random-state", type=int, default=RANDOM_STATE, help="Random seed for reproducibility.")
	parser.add_argument(
		"--sampler",
		type=str,
		choices=["smote", "oversample", "none", "smotenc"],
		default="smote",
		help="Resampling strategy to use inside the pipeline (default: smote)",
	)
	return parser


def main() -> int:
	parser = build_parser()
	args = parser.parse_args()

	if not args.data_path.exists():
		raise FileNotFoundError(
			f"Could not find dataset at {args.data_path}. Download the Telco churn CSV and pass --data-path."
		)

	artifact = train_end_to_end(
		csv_path=args.data_path,
		model_path=args.model_path,
		eda_path=args.eda_path,
		comparison_path=args.comparison_path,
		random_state=args.random_state,
		sampler=args.sampler,
	)

	print("Model comparison:")
	print(artifact["comparison"].to_string(index=False))
	print()
	print(f"Best model: {artifact['best_model_name']}")
	print(f"Best parameters: {artifact['best_params']}")
	print("Test metrics:")
	for metric_name, value in artifact["test_metrics"].items():
		if metric_name == "classification_report":
			continue
		print(f"  {metric_name}: {value:.4f}")
	print()
	print("Classification report:")
	print(artifact["test_metrics"]["classification_report"])
	print()
	print(f"Saved model bundle to {args.model_path}")
	print(f"Saved EDA summary to {args.eda_path}")
	print(f"Saved model comparison to {args.comparison_path}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
