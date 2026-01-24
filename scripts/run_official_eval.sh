#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/run_official_eval.sh <pred_json> [<gt_json>] [<out_txt>]
#
# Example:
#   bash scripts/run_official_eval.sh results/vslnet_19_7340_val_official.json \
#     "$HOME/ego4d_data/v2/annotations/nlq_val.json" results/official_eval_val.txt

PRED_JSON="${1:-}"
GT_JSON="${2:-$HOME/ego4d_data/v2/annotations/nlq_val.json}"
OUT_TXT="${3:-results/official_eval_val.txt}"

if [[ -z "$PRED_JSON" ]]; then
  echo "ERROR: missing <pred_json>"
  echo "Usage: bash scripts/run_official_eval.sh <pred_json> [<gt_json>] [<out_txt>]"
  exit 2
fi

if [[ ! -f "$PRED_JSON" ]]; then
  echo "ERROR: pred_json not found: $PRED_JSON"
  exit 2
fi

if [[ ! -f "$GT_JSON" ]]; then
  echo "ERROR: gt_json not found: $GT_JSON"
  exit 2
fi

mkdir -p "$(dirname "$OUT_TXT")"

echo "Reading predictions: $PRED_JSON"
echo "Reading gt: $GT_JSON"

# Patch predictions into the exact format the official evaluator expects.
# Required by evaluator:
#   predictions["challenge"] == "ego4d_nlq_challenge"
TMP_JSON="$(mktemp --suffix=.json)"
python - "$PRED_JSON" "$TMP_JSON" <<'PY'
import json, sys
inp, outp = sys.argv[1], sys.argv[2]
d = json.load(open(inp, "r"))
if not isinstance(d, dict):
    raise SystemExit("Predictions JSON must be a dict at top-level.")
# Add/overwrite required challenge tag
d["challenge"] = "ego4d_nlq_challenge"
# Keep version if present; if missing, add a default
d.setdefault("version", "1.0")
# Ensure results exists
if "results" not in d:
    raise SystemExit("Predictions JSON missing required key: 'results'")
json.dump(d, open(outp, "w"))
print("Wrote patched predictions:", outp)
PY

# Run official evaluator
python "$HOME/projects/episodic-memory/NLQ/VSLNet/utils/evaluate_ego4d_nlq.py" \
  --ground_truth_json "$GT_JSON" \
  --model_prediction_json "$TMP_JSON" \
  --thresholds 0.3 0.5 \
  --topK 1 3 5 | tee "$OUT_TXT"

rm -f "$TMP_JSON"
echo "Saved evaluator output to: $OUT_TXT"

