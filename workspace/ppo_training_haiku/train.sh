for game in "minatar-breakout" "minatar-freeway" "minatar-space_invaders" "minatar-asterix" "minatar-seaquest" "play2048" "backgammon"; do
    python ppo.py ENV_NAME=$game MAKE_ANCHOR=False
done