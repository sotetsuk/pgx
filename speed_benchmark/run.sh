#!/bin/bash


N=102400

 
CUDA_VISIBLE_DEVICES=0 python3 -O run_pgx.py $N > results.json

for game in "tic_tac_toe" "backgammon" "go"; do
for bs in 2 4 8 16 32 64 128 256 512 1024; do
  python3 -O run_open_spiel_forloop.py $game $bs $N >> results.json
done
done

for game in "tic_tac_toe" "backgammon" "go"; do
for bs in 2 4 8 16 32 64 128 256 512 1024; do
  python3 -O run_open_spiel_subproc.py $game $bs $N >> results.json
done
done

for game in "tic_tac_toe" "go"; do
for bs in 2 4 8 16 32 64 128 256 512 1024; do
  python3 -O run_petting_zoo.py $game for-loop $bs $N >> results.json 2>/dev/null
done
done

for game in "tic_tac_toe" "go"; do
for bs in 2 4 8 16 32 64 128 256 512 1024; do
  python3 -O run_petting_zoo.py $game subproc $bs $N >> results.json 2>/dev/null
done
done
