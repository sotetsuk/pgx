echo "| function name | # jaxpr lines | compile time |"
echo "| :--- | ---: | ---: |"
for funcname in init step _legal_actions _update_legal_move_actions _effected_positions _is_check _is_try _move _init_legal_actions _board_status _dlaction_to_action _action_to_dlaction _filter_leave_check_actions _filter_my_piece_move_actions _filter_occupied_drop_actions
do
    python3 benchmark.py $funcname
done
