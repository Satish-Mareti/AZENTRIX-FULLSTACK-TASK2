from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from churn_pipeline import load_model_bundle, parse_input_json, predict_churn


FIELD_SPECS = [
	("gender", "Gender", "Male", "text"),
	("SeniorCitizen", "Senior citizen (0 or 1)", 0, "int"),
	("Partner", "Partner (Yes/No)", "No", "yes_no"),
	("Dependents", "Dependents (Yes/No)", "No", "yes_no"),
	("tenure", "Tenure in months", 1, "int"),
	("PhoneService", "Phone service (Yes/No)", "Yes", "yes_no"),
	("MultipleLines", "Multiple lines (Yes/No/No phone service)", "No", "text"),
	("InternetService", "Internet service (DSL/Fiber optic/No)", "Fiber optic", "text"),
	("OnlineSecurity", "Online security (Yes/No/No internet service)", "No", "text"),
	("OnlineBackup", "Online backup (Yes/No/No internet service)", "No", "text"),
	("DeviceProtection", "Device protection (Yes/No/No internet service)", "No", "text"),
	("TechSupport", "Tech support (Yes/No/No internet service)", "No", "text"),
	("StreamingTV", "Streaming TV (Yes/No/No internet service)", "No", "text"),
	("StreamingMovies", "Streaming movies (Yes/No/No internet service)", "No", "text"),
	("Contract", "Contract (Month-to-month/One year/Two year)", "Month-to-month", "text"),
	("PaperlessBilling", "Paperless billing (Yes/No)", "Yes", "yes_no"),
	("PaymentMethod", "Payment method", "Electronic check", "text"),
	("MonthlyCharges", "Monthly charges", 70.0, "float"),
	("TotalCharges", "Total charges", 1000.0, "float"),
]


def build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Predict Telco churn from customer details.")
	parser.add_argument(
		"--model-path",
		type=Path,
		default=Path("artifacts") / "telco_churn_model.joblib",
		help="Path to the saved model bundle.",
	)
	parser.add_argument(
		"--input-json",
		type=str,
		default=None,
		help="JSON string with raw feature values. If omitted, the CLI prompts interactively.",
	)
	parser.add_argument(
		"--input-file",
		type=Path,
		default=None,
		help="Path to a JSON file with raw feature values.",
	)
	return parser


def normalize_value(raw_value: str, value_type: str) -> Any:
	text = raw_value.strip()
	if value_type == "int":
		return int(float(text))
	if value_type == "float":
		return float(text)
	if value_type == "yes_no":
		lowered = text.lower()
		if lowered in {"y", "yes", "true", "1"}:
			return "Yes"
		if lowered in {"n", "no", "false", "0"}:
			return "No"
		raise ValueError("Enter Yes or No.")
	return text


def prompt_for_record() -> dict[str, Any]:
	record: dict[str, Any] = {}
	print("Enter customer details. Press Enter to accept the suggested default.")
	for key, label, default, value_type in FIELD_SPECS:
		prompt = f"{label} [{default}]: "
		raw_value = input(prompt).strip()
		chosen = default if raw_value == "" else normalize_value(raw_value, value_type)
		record[key] = chosen
	return record


def load_record_from_file(path: Path) -> dict[str, Any]:
	return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
	parser = build_parser()
	args = parser.parse_args()

	if not args.model_path.exists():
		raise FileNotFoundError(
			f"Could not find a trained model at {args.model_path}. Run train.py first."
		)

	bundle = load_model_bundle(args.model_path)

	if args.input_file is not None:
		record = load_record_from_file(args.input_file)
	elif args.input_json is not None:
		record = parse_input_json(args.input_json)
	else:
		record = prompt_for_record()

	prediction = predict_churn(bundle, record)
	print(f"Prediction: {prediction['label']}")
	print(f"Churn probability: {prediction['probability']:.2%}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
