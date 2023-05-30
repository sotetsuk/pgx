#! /bin/bash

for game in "minatar-breakout" "minatar-freeway" "minatar-space_invaders" "minatar-asterix" "minatar-seaquest" "play2048" "backgammon" "leduc_holdem" "kuhn_poker"; do
    python3 ppo.py ENV_NAME=$game MAKE_ANCHOR=True&
done