for game in "minatar-breakout" "minatar-freeway" "minatar-space_invaders" "minatar-asterix" "minatar-seaquest" "play2048" "backgammon"; do
for seed in $(seq 1 5); do
    python ppo_single.py ENV_NAME=$game SEED=$seed
done
done