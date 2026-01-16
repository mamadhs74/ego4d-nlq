
#!/usr/bin/env bash

set -u



GT="/home/moho1597/ego4d_data/v2/annotations/nlq_val.json"

PRED_DIR="checkpoints_gpu_20e/vslnet_nlq_egovlp_new_128_bert/model"



for f in "$PRED_DIR"/vslnet_*_preds.json; do

  echo "== $f"



  # Run evaluator, capture full output, then print last 4 lines

  out="$(python utils/evaluate_ego4d_nlq.py --ground_truth_json "$GT" --model_prediction_json "$f" --thresholds 0.3 0.5 --topK 1 3 5)"

  printf "%s\n" "$out" | tail -n 4



  echo

done

