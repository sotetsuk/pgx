# Kuhn poker

=== "dark" 

    <p align="center">
    <img src="https://raw.githubusercontent.com/sotetsuk/pgx/main/docs/assets/kuhn_poker_dark.gif" width="30%">
    </p>

=== "light" 

    <p align="center">
    <img src="https://raw.githubusercontent.com/sotetsuk/pgx/main/docs/assets/kuhn_poker_light.gif" width="30%">
    </p>
    
## Description

Kuhn poker is a simplified poker with three cards: J, Q, and K.

## Rules

Each player is dealt one card and the remaining card is unused.
There are four actions: *check*, *call*, *bet*, and *fold* and five possible scenarios.

1. `bet   (1st) - call  (2nd)` : *Showdown* and the winner takes `+2`
2. `bet   (1st) - fold  (2nd)` : 1st player takes `+1`
3. `check (1st) - check (2nd)` : *showdown* and the winner takes `+1`
4. `check (1st) - bet   (2nd) - call (1st)` : *Showdown* and the winner takes `+2`
5. `check (1st) - bet   (2nd) - fold (1st)` : 2nd takes `+1`

