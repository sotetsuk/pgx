echo "| function name | compile time |"
echo "| :--- | ---: |"
for funcname in _move _update_legal_move_actions _effected_positions _is_check _is_try init _legal_actions step _init_legal_actions _board_status _dlaction_to_action
do
    python3 benchmark.py $funcname
done