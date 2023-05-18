#!/bin/bash


NUM_BATCH_STEPS=${1}
BATCH_SIZES="2 4 8 16 32 64 128 256"

# for game in "tic_tac_toe" "backgammon" "connect_four" "chess" "go"; do
# for bs in $BATCH_SIZES; do
#   python3 -O run_open_spiel_forloop.py $game $bs ${NUM_BATCH_STEPS}
# done
# done
# 
# for game in "tic_tac_toe" "backgammon"  "connect_four" "chess" "go"; do
# for bs in $BATCH_SIZES; do
#   python3 -O run_open_spiel_subproc.py $game $bs ${NUM_BATCH_STEPS}
# done
# done
# 
# for game in  "connect_four" "tic_tac_toe" "go"; do
# for bs in $BATCH_SIZES; do
#   python3 -O run_petting_zoo.py $game for-loop $bs ${NUM_BATCH_STEPS} 2>/dev/null
# done
# done
# 
# for game in  "connect_four" "tic_tac_toe" "go"; do
# for bs in $BATCH_SIZES; do
#   python3 -O run_petting_zoo.py $game subproc $bs ${NUM_BATCH_STEPS} 2>/dev/null
# done
# done

game=chess
# BATCH_SIZES="2 4 8 16"
BATCH_SIZES="128 256"
for bs in $BATCH_SIZES; do
  # python3 -O run_petting_zoo.py $game for-loop $bs ${NUM_BATCH_STEPS} 2>/dev/null
  python3 -O run_petting_zoo.py $game subproc $bs ${NUM_BATCH_STEPS} 2>/dev/null
done

