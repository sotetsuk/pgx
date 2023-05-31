for game in "2048" "backgammon" "sparrow_mahjong" "leduc_holdem" "kuhn_poker"; do
    python vis.py --env_name=$game
done