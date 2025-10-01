import os
import sys
import subprocess
from datetime import datetime

WORK_DIR = r"assets"
ATTRIBUTES_FILE = r"assets\attributes.json"
CONFIG = os.path.join(WORK_DIR, "..", "configs", "sample.yaml")

if len(sys.argv) < 2:
    print("Please provide the difficulty range parameter (e.g., 1:10)")
    sys.exit(1)

DIFFICULTY_RANGE = sys.argv[1]
MIN_DIFF, MAX_DIFF = DIFFICULTY_RANGE.split(":")
print(f"min:{MIN_DIFF}")
print(f"max:{MAX_DIFF}")

NUM_SAMPLES = 50  # 500
SCRIPT = f"..\scene_graph_builder\scene_graph_builder.py"
MAX_RETRY = 50
DUP_PROB = 0.1

LOG_DIR = os.path.join(WORK_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

OUTPUT_DIR = os.path.join(WORK_DIR, "scene_graphs", "c5v10_10", f"difficulty_{MIN_DIFF}_{MAX_DIFF}")
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_FILE = os.path.join(OUTPUT_DIR, f"scene_graph_sample{NUM_SAMPLES}_diff_{MIN_DIFF}_{MAX_DIFF}.json")
LOG_FILE = os.path.join(LOG_DIR, f"difficulty_{MIN_DIFF}_{MAX_DIFF}.log")

with open(LOG_FILE, "a") as log_file:
    log_file.write(f"[{datetime.now()}] Generating difficulty (Range: {MIN_DIFF}-{MAX_DIFF})...\n")

python_command = [
    "python", SCRIPT,
    "--attributes", ATTRIBUTES_FILE,
    "--output", OUTPUT_FILE,
    "--num_samples", str(NUM_SAMPLES),
    "--min_diff", MIN_DIFF,
    "--max_diff", MAX_DIFF,
    "--max_retry", str(MAX_RETRY),
    "--duplicate_prob", str(DUP_PROB),
    "--config", CONFIG
]

with open(LOG_FILE, "a") as log_file:
    subprocess.run(python_command, stdout=log_file, stderr=subprocess.STDOUT)
