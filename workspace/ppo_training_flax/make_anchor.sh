for game in "backgammon" "leduc_holdem" "kuhn_poker"; do
    python ppo.py ENV_NAME=$game MAKE_ANCHOR=True
done