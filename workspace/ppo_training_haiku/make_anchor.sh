for game in "backgammon" "sparrow_mahjong" "leduc_holdem" "kuhn_poker"; do
    python ppo.py ENV_NAME=$game MAKE_ANCHOR=True
done