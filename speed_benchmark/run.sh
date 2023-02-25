#!/bin/bash


N=10240
BATCH_SIZES="2 4 8 16 32 64 128 256 512 1024"

 
CUDA_VISIBLE_DEVICES=0 python3 -O run_pgx.py $N

for game in "tic_tac_toe" "backgammon" "go"; do
for bs in $BATCH_SIZES; do
  python3 -O run_open_spiel_forloop.py $game $bs $N
done
done

for game in "tic_tac_toe" "backgammon" "go"; do
for bs in $BATCH_SIZES; do
  python3 -O run_open_spiel_subproc.py $game $bs $N 
done
done

for game in "tic_tac_toe" "go"; do
for bs in $BATCH_SIZES; do
  python3 -O run_petting_zoo.py $game for-loop $bs $N 2>/dev/null
done
done

for game in "tic_tac_toe" "go"; do
for bs in $BATCH_SIZES; do
  python3 -O run_petting_zoo.py $game subproc $bs $N 2>/dev/null
done
done

for bs in $BATCH_SIZES; do
  python3 -O run_cshogi_forloop.py $game $bs $N
done

for bs in $BATCH_SIZES; do
  python3 -O run_cshogi_subproc.py $game $bs $N
done
