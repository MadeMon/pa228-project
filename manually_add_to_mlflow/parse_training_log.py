# STUDENT's UČO: 514143
"""
Script to parse training log and output per-epoch metrics as JSON.
"""
import re
import json

# Change this to your log file path or use sys.argv
LOG_FILE = "training_log.txt"
OUTPUT_FILE = "training_log.json"

# Regex patterns
EPOCH_PATTERN = re.compile(r"Epoch (\d+), train loss: ([\d.]+), val loss: ([\d.]+)")
CLASS_PATTERN = re.compile(r"\s*Class (\d+): ([\d.]+)")

results = []

with open(LOG_FILE, "r") as f:
    lines = f.readlines()

current_epoch = None
current_metrics = None
class_metrics = {}

for line in lines:
    epoch_match = EPOCH_PATTERN.match(line)
    class_match = CLASS_PATTERN.match(line)
    if epoch_match:
        # Save previous epoch if exists
        if current_epoch is not None:
            current_metrics["class_miou"] = class_metrics
            results.append(current_metrics)
        # Start new epoch
        current_epoch = int(epoch_match.group(1))
        train_loss = float(epoch_match.group(2))
        val_loss = float(epoch_match.group(3))
        current_metrics = {
            "epoch": current_epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
        }
        class_metrics = {}
    elif class_match and current_epoch is not None:
        class_idx = int(class_match.group(1))
        miou = float(class_match.group(2))
        class_metrics[class_idx] = miou

# Save last epoch
if current_epoch is not None:
    current_metrics["class_miou"] = class_metrics
    results.append(current_metrics)

with open(OUTPUT_FILE, "w") as f:
    json.dump(results, f, indent=2)

print(f"Parsed {len(results)} epochs. Output written to {OUTPUT_FILE}.")
