
## Local paths expected on cluster

- Ego4D v2 annotations: ~/ego4d_data/v2/annotations

- EgoVLP NLQ features:  ~/ego4d_assets/egovlp/nlq_features

- VSLNet code (episodic-memory): ~/projects/episodic-memory/NLQ/VSLNet



## Key scripts in this repo

- scripts/prepare_ego4d_dataset_egovlp.py

- scripts/make_official_pred_json.py

- scripts/run_official_eval.sh



## Notes

- NLQ test is unannotated; only prediction JSON can be produced, no official score.

