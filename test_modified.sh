#! /bin/bash

git fetch -a
for game in animal_shogi backgammon chess contractbridgebidding go shogi suzume_jong tic_tac_toe; do
    if [[ -n $(git diff --name-only origin/main | grep py | grep ${game}) ]]; then
        python3 -m pytest --doctest-modules -v pgx/${game}.py tests/test_${game}.py || exit 1
    fi
done
