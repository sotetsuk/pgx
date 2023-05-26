#!/bin/bash


NUM_BATCH_STEPS=${1}
PY="python3"
BATCH_SIZES="8 16 32 64 128 256 512 1024 2048 4096 8192 16384 32768"
GAMES="tic_tac_toe connect_four chess go"

# all gpus
for game in $GAMES; do
for bs in $BATCH_SIZES; do
  $PY O run_pgx.py $game $bs NUM_BATCH_STEPS
done
done

# 1 gpu
for game in $GAMES; do
for bs in $BATCH_SIZES; do
  CUDA_VISIBLE_DEVICES=0 $PY -O run_pgx.py $game $bs ${NUM_BATCH_STEPS}
done
done

BATCH_SIZES="2 4 8 16 32 64 128 256"

for game in $GAMES; do
for bs in $BATCH_SIZES; do
  $PY -O run_open_spiel_subproc.py $game $bs $NUM_BATCH_STEPS
done
done

for game in $GAMES; do
for bs in $BATCH_SIZES; do
  $PY -O run_open_spiel_forloop.py $game $bs $NUM_BATCH_STEPS
done
done

for game in $GAMES; do
for bs in $BATCH_SIZES; do
  $PY -O run_petting_zoo.py $game subproc $bs $NUM_BATCH_STEPS 2>/dev/null
done
done

for game in $GAMES; do
for bs in $BATCH_SIZES; do
  $PY -O run_petting_zoo.py $game for-loop $bs $NUM_BATCH_STEPS 2>/dev/null
done
done


