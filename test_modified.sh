#! /bin/bash

git fetch -a
for game in backgammon connect_four go hex shogi sparrow_mahjong tic_tac_toe; do
    if [[ -n $(git diff --name-only origin/main | grep py | grep ${game}) ]]; then
        python3 -m pytest --doctest-modules -v pgx/${game}.py tests/test_${game}.py || exit 1
    fi
done

# games under constructions
for game in animal_shogi chess bridge_bidding ; do
    if [[ -n $(git diff --name-only origin/main | grep py | grep ${game}) ]]; then
        python3 -m pytest --doctest-modules -v pgx/_${game}.py tests/test_${game}.py || exit 1
    fi
done
