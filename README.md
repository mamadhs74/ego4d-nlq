
Ego4D NLQ (Episodic Memory) – baseline + official evaluator



Data:

- Uses Ego4D v2 NLQ annotations at: ~/ego4d_data/v2/annotations/nlq_val.json



Scripts:

- scripts/make_official_pred_json.py: writes Ego4D NLQ challenge-format predictions (version 1.0).

- scripts/run_official_eval.sh: runs official evaluator from EGO4D/episodic-memory repo.



Results:

- results/official_eval_val_middle20.txt: official evaluator output table.



