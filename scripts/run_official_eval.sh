
#!/usr/bin/env bash

set -e



GT="/home/moho1597/ego4d_data/v2/annotations/nlq_val.json"

PRED="/home/moho1597/ego4d_data/official_nlq_val_pred_middle20.json"



python /home/moho1597/projects/episodic-memory/NLQ/VSLNet/utils/evaluate_ego4d_nlq.py \

  --ground_truth_json "$GT" \

  --model_prediction_json "$PRED" \

  --thresholds 0.3 0.5 \

  --topK 1 5


