#!/bin/bash


NUM_BATCH_STEPS=${1}
PY="python3"
GAMES=$(python3 -c "import pgx; print(' '.join(pgx.available_envs()))")

BATCH_SIZE=8192

# all gpus
for game in $GAMES; do
  $PY -O run_pgx.py $game $BATCH_SIZE $NUM_BATCH_STEPS
done

BATCH_SIZE=1024

# 1 gpu
for game in $GAMES; do
  CUDA_VISIBLE_DEVICES=0 $PY -O run_pgx.py $game $BATCH_SIZE $NUM_BATCH_STEPS
done
