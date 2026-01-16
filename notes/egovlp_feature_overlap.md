
EgoVLP NLQ features vs Ego4D v2 NLQ annotations

- EgoVLP released NLQ features are clip_uid.pt files.

- Ego4D v2 NLQ has clip_uids not present in the feature pack.

- We filter train/val to overlap subset and log kept/dropped counts.

Command:

python utils/prepare_ego4d_dataset_egovlp.py --input_train_split ... --input_val_split ... --output_save_path ... --video_feature_read_path ... --clip_feature_save_path ... --symlink

