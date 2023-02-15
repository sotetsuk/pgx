echo "| function name | # jaxpr lines | compile time |"
echo "| :--- | ---: | ---: |"
for funcname in init step _winning_step _no_winning_step observe _change_turn _legal_action_mask _is_action_legal _legal_action_mask_for_valid_single_dice _calc_win_score _is_all_on_home_board _rear_distance _update_by_action _move
do
    python3 benchmark.py $funcname
done