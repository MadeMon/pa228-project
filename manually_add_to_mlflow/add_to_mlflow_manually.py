import mlflow


import json
import os

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("segmentation_experiment")

# Path to the JSON file (adjust if needed)
LOG_PATH = os.path.join(os.path.dirname(__file__), "training_log.json")

with open(LOG_PATH, "r") as f:
	epochs = json.load(f)

with mlflow.start_run(run_name="manual_log_from_json"):
	for epoch in epochs:
		metrics = {
			"train_loss": epoch["train_loss"],
			"val_loss": epoch["val_loss"],
		}
		# Log per-class mIoU as class_miou_{idx}
		for class_idx, miou in epoch.get("class_miou", {}).items():
			metrics[f"miou_{class_idx}"] = miou
		mlflow.log_metrics(metrics, step=epoch["epoch"])
	print(f"Logged {len(epochs)} epochs to MLflow.")

