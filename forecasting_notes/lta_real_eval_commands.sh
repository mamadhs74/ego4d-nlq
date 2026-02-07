
#!/usr/bin/env bash

set -euo pipefail

export OMP_NUM_THREADS=1

export MKL_NUM_THREADS=1

export NUMEXPR_NUM_THREADS=1

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128



cd /scratch/moho1597/projects/forecasting



python -m scripts.run_lta \

  --job_name lta_eval_val_ctx2 \

  --working_directory /scratch/moho1597/ego4d_runs/output_lta_eval_val_ctx2 \

  --cfg configs/Ego4dLTA/MULTISLOWFAST_8x8_R101.yaml \

  NUM_GPUS 1 \

  TRAIN.ENABLE False \

  TEST.ENABLE True \

  TEST.BATCH_SIZE 1 \

  DATA_LOADER.NUM_WORKERS 0 \

  DATA_LOADER.PIN_MEMORY False \

  DATA_LOADER.ENABLE_MULTI_THREAD_DECODE False \

  DATA.PATH_TO_DATA_DIR data/long_term_anticipation \

  DATA.PATH_PREFIX data/long_term_anticipation/clips_hq \

  FORECASTING.NUM_INPUT_CLIPS 2 \

  FORECASTING.AGGREGATOR TransformerAggregator \

  FORECASTING.DECODER MultiHeadDecoder \

  CHECKPOINT_FILE_PATH /scratch/moho1597/ego4d_models/v2/lta_models/lta_slowfast_trf.ckpt

