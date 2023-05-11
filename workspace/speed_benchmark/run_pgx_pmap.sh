#!/bin/bash


NUM_BATCH_STEPS=${1}
BATCH_SIZES="8 16 32 64 128 256 512 1024 2048 4096 8192 16384 32768"

 
for game in "tic_tac_toe" "connect_four" "backgammon" "go" "chess"; do
for bs in $BATCH_SIZES; do
  python3 -O run_pgx_pmap.py $game $bs ${NUM_BATCH_STEPS}
done
done
