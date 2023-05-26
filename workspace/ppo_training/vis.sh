for game in "minatar-breakout" "minatar-freeway" "minatar-space_invaders" "minatar-asterix" "minatar-seaquest" "2048" "backgammon" "leduc_holdem" "kuhn_poker"; do
    python vis.py --env_name=$game
done